# standard library packages
# ===============
from __future__ import division, print_function
from math import sqrt, floor
import warnings
import time
import copy
# ------------------------------------------------------------------------------

# packages installed with `conda install anaconda`
# ================================================
import numpy as np
import scipy
import scipy.sparse as sparse
from numexpr import evaluate as ne_eval
import numexpr

from scipy.stats import norm as gaussian
# ------------------------------------------------------------------------------

# local packages
# ==============
from matrixops import (
    euclidean_proj_simplex, normalize, stack_matrices, proj_mat_to_simplex
)
from optimization import (
    first_last_stopping_condition, universal_stopping_condition, qf_min
)
from initialization import initialize_nmf

# ------------------------------------------------------------------------------

# logging
# =======
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------


debug = 0
eps_div_by_zero = np.spacing(10)  # added to denominators to avoid /0
OBJ = None  # will be instantiated from inside nmf()

class TrueObjComputer(object):
    def __init__(self, X, W, T, reg_w_l2, reg_t_l2, reg_w_l1, reg_t_l1, Wm, wr):
        self.X = X
        self.W = W
        self.T = T
        self.reg_w_l2 = reg_w_l2
        self.reg_t_l2 = reg_t_l2
        self.reg_t_l1 = reg_t_l1
        self.reg_w_l1 = reg_w_l1
        self.Wm = Wm
        self.wr = wr

    def true_objective(self):
        W2 = np.sum(self.W**2)
        T2 = np.sum(self.T**2)
        T1 = np.sum(np.abs(self.T))
        W1 = np.sum(np.abs(self.W))

        R = (self.X - np.dot(self.W, self.T))**2
        if self.Wm is not None:
            R = self.Wm * R
        if self.wr is not None:
            R = self.wr * R

        base_obj = 0.5 * np.sum(R)
        obj = base_obj + 0.5 * self.reg_w_l2 * W2 + 0.5 *self.reg_t_l2 * T2 + \
              0.5 * self.reg_t_l1 * T1 + 0.5 * self.reg_w_l1 * W1
        return obj


# @jit
def nmf(X, k, w_row=None, W_mat=None, fix_W=False, fix_T=False,
        random_state=None, init='nndsvd', T_in=[], W_in=[], max_iter=200,
        max_time=600, eps_stop=1e-4, compute_obj_each_iter=False,
        project_W_each_iter=False, w_row_sum=None, project_T_each_iter=True,
        t_row_sum=1.0, early_stop=None, reset_topic_method='random',
        fix_reset_seed=False, reg_w_l2=0, reg_t_l2=0, reg_w_l1=0, reg_t_l1=0,
        diagnostics=[], store_gradients=False,
        ind_rows_to_store=None, eps_gauss_t=None, delta_gauss_t=None):
    """
    Compute the non-negative matrix factorization.

    Factorize the n*d document*feature matrix `X` into the product `WT`
    where `W` is a non-negative n*k doc*topic matrix and `T` is a
    non-negative k*d topic*feature matrix. We use the term 'topic's to refer
    to latent basis vectors. Documents are expressed as non-negative linear
    combinations of topics.

    :math:`\min_{W\geq 0, T\geq 0} ||X-WT||_F^2 + wr1||W||_1 + wr2||W||_2^2 +
    tr1||T||_1 + tr2||T||_2^2`

    Parameters
    ----------
    X : array_like
        The n*d document-feature ndarray to be factorized.
    k : int
        Number of latent factors, i.e. the rank of the factorization.
    w_row : array_like or None, optional
        A n*1 array of importance weights for each document, row of `X`.
        None by default, meaning equal weight for each row.
    W_mat : array_like or None, optional
        A n*d array of importance weights for each entry of `X`. None by
        default. Meaning equal entry for each entry.
    fix_W
    fix_T
    random_state
    init
    T_in
    W_in
    debug
    max_iter
    max_time
    eps_stop
    compute_obj_each_iter : bool, optional
        Should the full NMF objective be computed for each iteration? False
        by default; carries a 2x performance penalty due to current
        implementation. It is set to True if `logger.level` <= logging.DEBUG.
        It is also useful as a more general stopping condition than the
        projected gradient norm-based stopping condition.
    project_W_each_iter
    w_row_sum
    project_T_each_iter
    t_row_sum
    early_stop
    reset_topic_method
    fix_reset_seed
    reg_w_l2
    reg_t_l2
    reg_w_l1
    reg_t_l1
    negative_denom_correction
    saddle_point_handling
    damping_w
    damping_t
    diagnostics
    store_gradients
    ind_rows_to_store
    n_words_beam
    eps_gauss_t
    delta_gauss_t

    Returns
    -------
    W : array_like
        n * k doc-topic weight matrix
    T : array_like
        k * d topic-feature matrix

    """
    global debug, eps_div_by_zero, OBJ

    """ Factorize non-negative [n*d] X as  non-negative [n*k] W x [k*d] T

    :param X: ndarray of shape (n, d)
    :param k: inner dim (and rank) of W*T
    :param w_row: ndarray of shape (n, ) w_row[i] is weight assigned to reconst
                  error of row X[i,:]. None is equivalent to w_row = 1 forall i
    :param W_mat: ndarray of shape (n,d) W_mat[i,j] is the weight of X[i,j] for
            the objective function
    :param fix_W: W isn't updated if True
    :param fix_T: T isn't updated if True
    :param random_state: integer used to seed the initialization
    :param init: the initialization method.
            'nndsvd' uses randomized partial SVD
            'coherence_pmi' uses beam search on the word co-occurrence matrix

    :param debug: integer setting debug level, higher = more debug is shown.
                    1= iteration count & timer
                    2= plot objective + 1
    :param max_iter: maximum number of iterations. 1 it = 1 full update of W,T
    :param max_time: maximum time (in sec)
    :param eps_stop: (may be deprecated) stop if
                                (obj[-1]-obj[-2])/(obj[0]-obj[1]) <= eps_stop
    :param compute_obj_each_iter [True] required for updating the obj_history
            output as well as for using eps_stop, but may slow down everything
            by 2x
    :param project_W_each_iter: project each row of W onto l_1 simplex each it
                        if False only project once at the end. Projecting
                        each iter is an extra O(nklogk) work, but improves obj
                        by around 1%
    :param w_row_sum a scalar or vector that each row of W should sum to
        should be equal to w_row if w_row is used

    :param early_stop a function that takes X, W, T and outputs a validation
    score
        if this score is higher than the previous iteration, we stop and return
        [None] by default
    :param reset_topic_method ['max_resid_document'] or 'random'; how should
        we reset topics that become empty
    :param fix_reset_seed [False] should the same seed be used to reset empty
        topics and their weights? True is good for comparing to distributed
        computation.

    :param reg_w_l2 : regularization penalty applied to \ell_2 norm of
        columns of W matrix. Positive => more dense solutions; Negative => more
        sparse. Reason: (1,0) has a high ell_2 norm than (0.5, 0.5), so if we
        are subtracting the norm from the objective, we will favor sparser.
    :param reg_t_l2 : regularization penality applied to \ell_2 norm of rows
        of T matrix. Positive => more dense solutions; Negative => more sparse.
    :param reg_w_l1 : regularization penality applied to \ell_1 norm of
        columns of W. Negative => more dense; Positive => more sparse.
    :param reg_t_l1 : regularization penalty applied to \ell_1 norm of rows
        of T. Negative => more dense; Positive => more sparse.
    :param negative_denom_correction : boolean, True by default. When
        negative l_2 regularization is used, the 2nd derivatives of W and T
        (Hessians) may become negative (negative definite), so we need to
        check the boundaries of our constraint set for optimal solutions
        rather than simply set the first derivative to 0.
    :param damping_w: weight of previous iteration's w added to current w
    :param damping_t: weight of previous iteration's t added to current t
    :param diagnosticics: a function that should be measured at each iter
        outputs will be in 'diagnostic' in same order. Each function must
        take X, W, T
    :param store_gradients : boolean [False] store the numerators and
        denominators for each topic update for each iteration (used for
        detailed analysis or debuging)
    :param ind_rows_to_store : list or None (default) when storing 
    intermediate results
        they can be stored for all rows (if None) or a given list of rows. The
        use is to see whether a particular subset of rows has a particular
        effect on the intermediate results
    :param eps_gauss_t : numeric epsilon for Gaussian mechanism for
        calculation of T in each iteration
    :param delta_gauss_t : numeric delta for Gaussian mechanism for
        calculation of T in each iteration
    """
    logger.log(logging.DEBUG-1, 'Locals: {}'.format(locals()))
    rtv = {}

    if type(diagnostics) is not list:
        diagnostics = [diagnostics]

    if len(diagnostics) > 0:
        rtv['diagnostics'] = {}
        for func in diagnostics:
            rtv['diagnostics'][func.func_name] = []

    if store_gradients:
        rtv['numer_W'] = {}
        rtv['denom_W'] = {}

    if random_state is None:
        random_state = int(time.time()) % 4294967296

    t_global_start = time.time()
    max_time = max_time - 10  # subtract 10sec for projecting W

    n, d = X.shape

    X_orig = None
    if w_row is not None:
        X_orig = X.copy()
        X = np.sqrt(w_row) * X  # needs to be sqrt since we do (wX-WT)^2

    if not w_row_sum is None and not np.isscalar(w_row_sum):
        w_row_sum = w_row_sum.reshape((w_row_sum.size, 1))
        if not w_row is None:
            w_row_sum = np.sqrt(w_row_sum)  # rows of X will be multiplied by
            #  sqrt of w_row, so the rows of W should also sum to sqrt of w_row

    if n <= k:
        init = 'random'

    start_time = time.clock()

    W, T = _initialize_and_validate(**locals())

    iter_cputime = []  # time per iteration
    proj_gradient_norm = []  # the norm of projected gradients is used as a
    # stopping condition. Thesis Ho section 3.5

    if W_mat is not None:
        logger.info('W_mat is k times slower than w_row or no weighting.')

        def rshp(x):
            """ make it (n,1) instead of (n,)"""
            return x.reshape(x.size, 1)

    numexpr.set_num_threads(numexpr.detect_number_of_cores())

    if early_stop:
        last_score = np.inf
        W_prev = copy.deepcopy(W)
        T_prev = copy.deepcopy(T)

    obj_history = []
    if logger.level <= logging.DEBUG:
        compute_obj_each_iter = True
    if compute_obj_each_iter:
        OBJ = TrueObjComputer(X, W, T, reg_w_l1=reg_w_l1, reg_t_l2=reg_t_l2,
                              reg_w_l2=reg_w_l2, reg_t_l1=reg_t_l1, Wm=W_mat,
                              wr=w_row)

    if len(diagnostics) > 0:
        for func in diagnostics:
            rtv['diagnostics'][func.func_name].append(func(X, W, T))

    for iter_no in range(max_iter):
        logger.info('\n\n{s:^80}\n{e:-<80}'.format(
            s='Iteration %d'%iter_no, e=''))

        if early_stop:
            if type(early_stop)==type(lambda x:x):
                this_score = early_stop(X, W, T)
            else:
                if compute_obj_each_iter:
                    if len(obj_history)==0:
                        this_score = np.inf
                    else:
                        this_score = obj_history[-1]
            logger.info('Iter %d stopping score %.3f' % (iter_no, this_score))
            if this_score > last_score:  # STOP EARLY
                logger.info('Stopping early at iter %d' % iter_no)
                W = W_prev
                T = T_prev
                obj_history = obj_history[:-1]
                iter_cputime = iter_cputime[:-1]
                if len(diagnostics)>0:
                    for func in diagnostics:
                        rtv['diagnostics'][func.func_name] = rtv[
                            'diagnostics'][func.func_name][:-1]
                break
            # else this_score <= last_score
            last_score = this_score
            W_prev = copy.deepcopy(W)
            T_prev = copy.deepcopy(T)

        if debug >= 1:
            it_start_time = time.time()

        grad_norm_this_iter = 0

        if store_gradients:
            rtv['numer_W'][iter_no] = []
            rtv['denom_W'][iter_no] = []

        for t in range(k):
            logger.debug('\n{: ^80}'.format('-= Topic %d =-' % t))
            if not fix_T:

                with _MeasureDelta('update T'):
                    wR, nw, wR_store, nw_store = _compute_update_T(**locals())

                    if eps_gauss_t and delta_gauss_t:
                        # both parameters are not None received.
                        # The intent is to use the Gaussian differentially
                        # private mechanism pg 261 of Dwork Roth Differential
                        # Privacy
                        c2 = 2 * np.log(1.25 / delta_gauss_t)
                        df2 = 1.0  # an upper bound on the l2 sensitivity here
                        sigma2 = c2 * df2**2 * (1 / eps_gauss_t)**2
                        N = gaussian(0, np.sqrt(sigma2))  # scipy's norm takes mean,
                        #  std
                        wR += N.rvs(wR.size).reshape(wR.shape)
                        nw += N.rvs(nw.size).reshape(nw.shape)

                    numer = (wR - reg_t_l1)
                    denom = nw + reg_t_l2

                    # with _MeasureDelta('Thesis Ho'):
                    # T[t, :] = np.maximum(numer, 0) / (denom + eps_div_by_zero)
                    # I = np.argwhere(denom==0).ravel()
                    # T[t, I] = 0
                    # nt1 = np.sum(T[t,:])
                    # T[t, :] = T[t, :]/nt1
                    # W[:,t] = W[:,t]*nt1

                    # T[t, :] = Tprev
                    # with _MeasureDelta('qf_min'):
                    #
                    #nt1 = np.sum(T[t, :])
                    T[t, :], nt1 = qf_min(-numer, denom, s=t_row_sum)
                    W[:, t] = W[:, t]*nt1

                if store_gradients:
                    rtv['numer_W'][iter_no].append(wR_store)
                    rtv['denom_W'][iter_no].append(nw_store)

                _project_and_check_reset_t(**locals())

                # Ho says to compute gradient after normalization (Section 3.5)
                grad = T[t, :] * denom - numer  # x*d - n = df/dx
                grad = grad**2  # for purposes of computing Frob norm
                if iter_no == 0:  # dont use proj grad on first iter (Lin 2007)
                    grad_norm_this_iter += _projected_gradient(grad, T[t, :],
                                                               lb=-np.inf,
                                                               ub=np.inf)
                else:
                    grad_norm_this_iter += _projected_gradient(grad, T[t, :])

            if not fix_W:
                with _MeasureDelta('update W'):
                    Rt, nt = _compute_update_W(**locals())

                    numer = Rt - reg_w_l1
                    denom = nt + reg_w_l2

                    W[:, t], nw1 = qf_min(-numer, denom, s=None)

                _check_reset_W(**locals())

                # this assertion is useful while qf_min is still under
                # development, once qf_min is stable it can be removed
                assert np.all(W[:, t] >= 0), 'W contains negative entries'
                assert np.sum(W[:, t]) > 0, 'W[:, t] sums to 0'

                grad = W[:, t] * denom - numer
                grad = grad**2  # for purposes of computing Frob norm
                if iter_no == 0:  # dont use proj grad on first iter (Lin 2007)
                    grad_norm_this_iter += _projected_gradient(grad, W[:, t],
                                                               lb=-np.inf,
                                                               ub=np.inf)
                else:
                    grad_norm_this_iter += _projected_gradient(grad, W[:, t])
            # END for t in range(k)
        if project_W_each_iter and not fix_W and not w_row_sum is None:
            logger.info('\nAfter iter {iter_no} projecting each W row'.format(
                        iter_no=iter_no))
            W = proj_mat_to_simplex(W, w_row_sum)


        logger.info('\n{s: ^80}\n'.format(s='Summary at end of '
                                            'iteration %d'%iter_no))
        if compute_obj_each_iter:
            obj_history.append(OBJ.true_objective())
            logger.info('\tObj: {0:3.3e}'.format(obj_history[-1]))

        iter_cputime.append(time.clock())

        proj_gradient_norm.append(np.sqrt(grad_norm_this_iter))

        # run diagnostics after timing
        if len(diagnostics) > 0:
            for func in diagnostics:
                dval = func(X, W, T)
                rtv['diagnostics'][func.func_name].append(dval)
                logger.info('\t{1}: {2}'.format(iter_no, func.func_name,
                                                    dval))

        if debug >= 1:
            t_now = time.time()
            logger.info('Iter done; ||grad_i||/||grad_1||=%.3e; Took %.3fsec' % (
                np.sqrt(grad_norm_this_iter) / proj_gradient_norm[0],
                t_now - it_start_time))


        if time.time() - t_global_start >= max_time:
            logger.info('STOPPING because max_time after iter %d' % iter_no)
            break

        if first_last_stopping_condition(proj_gradient_norm,
                                         eps_stop=eps_stop) and not (
                    W_mat is not None):
            # dont use proj grad stop cond if recsys
            logger.info('STOPPING because proj grad norm after iter %d' %
                        iter_no)
            break

        if compute_obj_each_iter and universal_stopping_condition(obj_history,
                                                                  eps_stop=eps_stop):
            if debug >= 1:
                logger.info('STOPPING because obj_history after iter %d' %
                        iter_no)
            break
        #logger.info('{:=^80}'.format(''))

    iter_cputime = [x - start_time for x in iter_cputime]

    # project after completing iterations
    if not project_W_each_iter and not w_row_sum is None and not fix_W:
        if np.isscalar(w_row_sum):
            logger.info('Post completion W row projection to {}'.format(
                        w_row_sum))
            for i in range(n):
                W[i, :] = euclidean_proj_simplex(W[i, :], s=w_row_sum)
        else:  # w_row_sum is a vector with a individual sum for each row
            logger.info('Post completion W row-wise projection')
            for i in range(n):
                W[i, :] = euclidean_proj_simplex(W[i, :], s=w_row_sum[i])

    if w_row is not None:
        sub = nmf(X_orig, k, T_in=T, fix_T=True, max_iter=10,
                  w_row_sum=w_row_sum, project_W_each_iter=True,
                  compute_obj_each_iter=compute_obj_each_iter)
        for oh in sub['obj_history']:
            obj_history.append(oh)
        for itc in sub['iter_cputime']:
            iter_cputime.append(itc)
        W = sub['W']

    if store_gradients:
        for itno in rtv['numer_W']:
            rtv['numer_W'][itno] = stack_matrices(rtv['numer_W'][itno],
                                                  lambda row: row.reshape(
                                                          (1, row.size)))
        for itno in rtv['denom_W']:
            rtv['denom_W'][itno] = stack_matrices(rtv['denom_W'][itno],
                                                  lambda row: row.reshape(
                                                          (1, row.size)))

    rtv['W'] = W
    rtv['T'] = T
    if compute_obj_each_iter:
        rtv['obj_history'] = obj_history
        rtv['obj_calculator'] = OBJ
    rtv['proj_gradient_norm'] = proj_gradient_norm
    rtv['iter_cputime'] = iter_cputime
    rtv['random_state'] = random_state

    return rtv

def _log_delta_obj(f):
    """Measure the change in objective in a decorated function"""
    def wrapper(*args, **kwargs):
        global OBJ
        if logger.level <= logging.DEBUG:
            obj_before = OBJ.true_objective()
        rtv = f(*args, **kwargs)
        if logger.level <= logging.DEBUG:
            obj_after = OBJ.true_objective()
            logger.debug('\t\t\t{0}() delta = {1:.2f}'.format(
                f.func_name, obj_after-obj_before))
        return rtv
    return wrapper


class _MeasureDelta(object):
    """Measure the change in NMF objective around a block of code

    Parameters
    ----------
    name : string, optional
    A useful name for this block of code, None by default.

    Examples
    --------
    >>> with _MeasureDelta('min'):
    >>>     W[0,:]=0
    >>>     W[0,0]=1
    """
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        global OBJ
        if logger.level <= logging.DEBUG:
            self.obj = OBJ.true_objective()
    def __exit__(self, type, value, traceback):
        global OBJ
        if logger.level <= logging.DEBUG:
            obj_after = OBJ.true_objective()
            name_s = '{}: '.format(self.name) if self.name else ''
            logger.debug('\t\t\t{0}delta = {1:.2f}'.format(
                name_s, obj_after - self.obj))

def _projected_gradient(grad, vec, lb=0, ub=1):
    """
    Compute the projected gradient defined as (where X=vec):

    :math:`[\grad_X^P]_ij = [\grad_X]_ij if X_ij>0 else min(0, [\grad_X]_ij)`

    :param grad: vector of shape (d,) containing the gradient of vec
    :param vec: vector of shape (d,) containing the elements of vec
    :param [zero]: floating point that represents zero (1e-10 by default)
    """
    global eps_div_by_zero
    lb = lb + eps_div_by_zero
    ub = ub - eps_div_by_zero

    rtv = 0
    rtv += np.sum(grad[np.logical_and(vec > lb, vec < ub)])
    rtv += np.sum(scipy.minimum(grad[vec <= lb], 0))
    rtv += np.sum(scipy.maximum(grad[vec >= ub], 0))
    return rtv



def _compute_update_T(X,W,T,t, store_gradients, ind_rows_to_store,
                      rshp=None, W_mat=None, **kwargs):
    """
    Compute update to one row of T

    Returns
    -------
    gradients

    """
    logger.debug('\t\tT:'.format(t))
    wR_store = None
    nw_store = None

    if W_mat is None:
        w = W[:, t]
        wX = w.T.dot(X)
        wW = w.T.dot(W)
        wW[t] = 0
        wR = wX - wW.dot(T)
        nw = (W[:, t]**2).sum()  # ||W[:, t]||^2, this is a scalar
        if store_gradients and not (ind_rows_to_store is not None):
            wR_store = wR
            nw_store = nw
        elif store_gradients and ind_rows_to_store is not None:
            ws = W[ind_rows_to_store, :][:, t]
            wXs = ws.T.dot(X[ind_rows_to_store, :])
            wWs = ws.T.dot(W[ind_rows_to_store, :])
            wWs[t] = 0
            wR_store = wXs - wWs.dot(T)
            nw_store = (ws**2).sum()
    else:
        Rt = X - np.dot(W, T) + np.dot(rshp(W[:, t]), rshp(T[t, :]).T)
        Rt = ne_eval('Rt * W_mat')
        wR = np.dot(W[:, t].T, Rt)
        nw = np.dot(W_mat.T, rshp(W[:, t]**2)).reshape(wR.shape)
        # this is a vector
        #  but python broadcasting implements Lemma 6.5 correct
        if store_gradients and ind_rows_to_store is None:
            wR_store = wR
            nw_store = nw
        elif store_gradients and ind_rows_to_store is not None:
            wR_store = np.dot(W[ind_rows_to_store, :][:, t].T,
                              Rt[ind_rows_to_store, :])
            nw_store = np.dot(rshp(W[ind_rows_to_store, :][:, t]**2).T,
                              W_mat[ind_rows_to_store, :]).ravel()

    return wR, nw, wR_store, nw_store


def _compute_update_W(X,W,T,W_mat, t, rshp=None, **kwargs):
    """
    Compute update to one column of W.

    Returns
    -------
    gradients

    """
    logger.debug('\t\tW:'.format(t=t))
    if W_mat is None:
        Xt = X.dot(T[t, :].T)
        Tt = T.dot(T[t, :].T)
        Tt[t] = 0
        Rt = Xt - W.dot(Tt)
        nt = (T[t, :]**2).sum()  # ||T[t, :]||^2, a scalar
    else:
        Rt = X - np.dot(W, T) + np.dot(rshp(W[:, t]), rshp(T[t, :]).T)
        Rt = ne_eval('W_mat * Rt')
        Rt = np.dot(Rt, T[t, :].T)
        nt = np.dot(W_mat, rshp(T[t, :]**2)).reshape(Rt.shape)
    return Rt, nt


@_log_delta_obj
def _project_and_check_reset_t(X,W,T, t, d, project_T_each_iter, t_row_sum,
                               reset_topic_method, fix_reset_seed, **kwargs):
    """
    Project a row of t to simplex and check reset it if it's 0.
    """
    nt1 = np.sum(T[t, :])
    if nt1 > 1e-10 or reset_topic_method==None:
        if project_T_each_iter and np.abs(np.sum(T[t,:])-t_row_sum)>1e-15:
            T[t, :] = euclidean_proj_simplex(T[t, :], s=t_row_sum)
    else:  # pick the largest positive residual
        logger.debug('\t\tReseting T{t} method={m} fixed_seed={'
                  's}'.format(t=t, m=reset_topic_method, s=fix_reset_seed))
        if reset_topic_method == 'max_resid_document':
            Rt = scipy.maximum(X - W.dot(T), 0)
            Rts = (Rt**2).sum(1)
            mi = scipy.argmax(Rts)
            T[t, :] = Rt[mi, :]
        elif reset_topic_method == 'random':
            if fix_reset_seed:
                np.random.seed(t)
            T[t, :] = np.random.rand(1, d)
            T[t, :] /= T[t, :].sum()


@_log_delta_obj
def _check_reset_W(X,W,T, n, d, t, iter_no, reset_topic_method,
                   fix_reset_seed, **kwargs):
    """
    Reset column of w if it becomes 0.
    """
    nw1 = np.sum(W[:, t])
    if nw1 > 1e-10 or reset_topic_method==None:
        pass
    else:  # pick the largest positive residual
        logger.debug('\t\tReseting W{t} method={m} fixed_seed={'
                  's}'.format(t=t, m=reset_topic_method, s=fix_reset_seed))
        if reset_topic_method == 'max_resid_document':
            Rt = scipy.maximum(X - W.dot(T), 0)
            Rts = (Rt**2).sum(1)
            mi = scipy.argmax(Rts)
            T[t, :] = Rt[mi, :]
            W[mi, t] = 1.0
        elif reset_topic_method == 'random':
            if fix_reset_seed:
                np.random.seed(t)
            T[t, :] = np.random.rand(1, d)
            T[t, :] /= T[t, :].sum()
            W[:, t] = np.random.rand(n)


def _initialize_and_validate(W_in, T_in, W_mat, X, k, init, random_state,
                             project_T_each_iter, project_W_each_iter,
                             w_row_sum, t_row_sum, fix_W, fix_T, n, d,
                             **kwargs):
    """Initializes `W`, `T`, or sets them to `W_in`, `T_in`, respectively.
    Ensures that `W`, `T` satisfy non-negativity and row-sum constraints.

    Parameters
    ----------
    **kwargs
        All parameters are passed directly from `nmf` using locals()

    Returns
    -------
    W : array_like
        n*k array of doc-topic weights
    T : array_like
        k*d array of topic-word probabilities
    """
    global debug

    if np.prod(np.shape(W_in)) == 0 or np.prod(np.shape(T_in)) == 0:
        if not W_mat is None:
            W, T = initialize_nmf(W_mat * X, k, init, random_state=random_state,
                                  row_normalize=False)
        else:
            W, T = initialize_nmf(X, k, init, random_state=random_state,
                                  row_normalize=False)
        if project_T_each_iter:
            T = normalize(T) * t_row_sum
            # if project_W_each_iter and not w_row_sum is None:
            #    W = normalize(W) * w_row_sum
    if np.prod(np.shape(W_in)) > 0:
        if not np.shape(W_in) == (n, k):
            raise ValueError('W_in has wrong dimensions, must be n*k')
        W = W_in

    if np.prod(np.shape(T_in)) > 0:
        if not np.shape(T_in) == (k, d):
            raise ValueError('T_in has wrong dimensions, must be k*d')
        T = T_in

    if scipy.sparse.issparse(T):
        T = T.toarray()
    if scipy.sparse.issparse(W):
        W = W.toarray()

    W = np.maximum(W, 0)
    T = np.maximum(T, 0)

    if project_W_each_iter and not fix_W and not w_row_sum is None:
        if debug >= 1:
            print('Projecting W rows after initialization')
        W = proj_mat_to_simplex(W, w_row_sum)

    if project_T_each_iter and not fix_T and not t_row_sum is None:
        if debug >= 1:
            print('Projecting W rows after initialization')
        T = proj_mat_to_simplex(T, t_row_sum)

    return W, T