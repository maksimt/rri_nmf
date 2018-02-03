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
    euclidean_proj_simplex, normalize, stack_matrices,
    proj_mat_to_simplex
)
from optimization import (
    first_last_stopping_condition, universal_stopping_condition, qf_min
)
from initialization import initialize_nmf
# ------------------------------------------------------------------------------

debug = 0


class TrueObjComputer(object):
    def __init__(self, X, reg_w_l2, reg_t_l2, reg_w_l1, reg_t_l1, Wm, wr):
        self.trXTX = np.sum(X**2)
        self.reg_w_l2 = reg_w_l2
        self.reg_t_l2 = reg_t_l2
        self.reg_t_l1 = reg_t_l1
        self.reg_w_l1 = reg_w_l1
        self.Wm = Wm
        self.wr = wr

    def true_objective(self, X, W, T):
        W2 = np.sum(W**2)
        T2 = np.sum(T**2)
        T1 = np.sum(np.abs(T))
        W1 = np.sum(np.abs(W))

        R = (X - np.dot(W, T))**2
        if self.Wm is not None:
            R = self.Wm * R
        if self.wr is not None:
            R = self.wr * R

        base_obj = 0.5 * np.sum(R)
        obj = base_obj + self.reg_w_l2 * W2 + self.reg_t_l2 * T2 + \
              self.reg_t_l1 * T1 + self.reg_w_l1 * W1
        return obj


# @jit
def nmf(X, k, w_row=None, W_mat=None, fix_W=False, fix_T=False,
        random_state=None, init='nndsvd', T_in=[], W_in=[],
        max_iter=200, max_time=600, eps_stop=1e-4, compute_obj_each_iter=False,
        project_W_each_iter=False, w_row_sum=None,
        project_T_each_iter=True, t_row_sum=1.0,
        early_stop=None, reset_topic_method='random', fix_reset_seed=False,
        reg_w_l2=0, reg_t_l2=0, reg_w_l1=0, reg_t_l1=0,
        negative_denom_correction=True,
        damping_w=0, damping_t=0, diagnostics=[], store_intermediate=False,
        I_store=None, eps_gauss_t=None, delta_gauss_t=None):
    """

    Parameters
    ----------
    X
    k
    w_row
    W_mat
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
    compute_obj_each_iter
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
    store_intermediate
    I_store
    n_words_beam
    eps_gauss_t
    delta_gauss_t

    Returns
    -------

    """
    global debug

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
    :param saddle_point_handling : string 'exception' by default. When
        negative l_2 regularization is used, how should saddle points be handled
        ? The default is to throw an exception. Until I see a saddle point I
        won't think about how to handle them. :P

    :param damping_w: weight of previous iteration's w added to current w
    :param damping_t: weight of previous iteration's t added to current t
    :param diagnosticics: a function that should be measured at each iter
        outputs will be in 'diagnostic' in same order. Each function must
        take X, W, T
    :param store_intermediate : boolean [False] store the numerators and
        denominators for each topic update for each iteration (used for
        detailed analysis or debuging)
    :param I_store : list or None (default) when storing intermediate results
        they can be stored for all rows (if None) or a given list of rows. The
        use is to see whether a particular subset of rows has a particular
        effect on the intermediate results
    :param eps_gauss_t : numeric epsilon for Gaussian mechanism for
        calculation of T in each iteration
    :param delta_gauss_t : numeric delta for Gaussian mechanism for
        calculation of T in each iteration
    """

    rtv = {}

    if type(diagnostics) is not list:
        diagnostics = [diagnostics]

    if len(diagnostics) > 0:
        rtv['diagnostics'] = {}
        for func in diagnostics:
            rtv['diagnostics'][func.func_name] = []

    if store_intermediate:
        rtv['numer_W'] = {}
        rtv['denom_W'] = {}

    if random_state is None:
        random_state = int(time.time()) % 4294967296

    t_global_start = time.time()
    max_time = max_time - 10  # subtract 10sec for projecting W

    n, d = X.shape

    _eps_div_by_zero = np.spacing(10)  # added to denominators to avoid /0


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

    def project_and_check_reset_t():
        nt1 = np.sum(T[t, :])

        if nt1 > 1e-10:
            if project_T_each_iter:
                T[t, :] = euclidean_proj_simplex(T[t, :], s=t_row_sum)
        else:  # pick the largest positive residual
            if debug >= 2:
                print('\t\t\tReseting T{t} method={m} fixed_seed={'
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

            if debug >= 5 and T[t, :].size <= 50:
                print('\t\t\t\tReset to {}'.format(T[t, :]))


    W, T = _initialize_and_validate(**locals())

    start_time = time.clock()
    iter_cputime = []  # time per iteration
    proj_gradient_norm = []  # the norm of projected gradients is used as a
    # stopping condition. Thesis Ho section 3.5

    if W_mat is not None:
        if debug >= 1:
            print('WARNING: W_mat is currently implemented inefficiently')

        def rshp(x):
            """ make it (n,1) instead of (n,)"""
            return x.reshape(x.size, 1)

    numexpr.set_num_threads(numexpr.detect_number_of_cores())

    if early_stop:
        last_score = np.inf
        W_prev = copy.deepcopy(W)
        T_prev = copy.deepcopy(T)

    if project_W_each_iter and not fix_W and not w_row_sum is None:
        if debug >= 1:
            print('Projecting W rows after initialization')
        if np.isscalar(w_row_sum):
            for i in range(n):
                W[i, :] = euclidean_proj_simplex(W[i, :], s=w_row_sum)
        else:  # w_row_sum is a vector with a individual sum for each row
            for i in range(n):
                W[i, :] = euclidean_proj_simplex(W[i, :], s=w_row_sum[i])

    if project_T_each_iter and not fix_T and not t_row_sum is None:
        T = proj_mat_to_simplex(T, t_row_sum)

    obj_history = []
    if compute_obj_each_iter:
        OBJ = TrueObjComputer(X, reg_w_l1=reg_w_l1, reg_t_l2=reg_t_l2,
                              reg_w_l2=reg_w_l2, reg_t_l1=reg_t_l1,
                              Wm=W_mat, wr=w_row)

    if len(diagnostics) > 0:
        for func in diagnostics:
            rtv['diagnostics'][func.func_name].append(func(X, W, T))

    for iter_no in range(max_iter):
        if debug >= 1:
            print('\nStarting iteration {'
                  'iter_no}\n----------------------'.format(iter_no=iter_no))

        if early_stop:
            this_score = early_stop(X, W, T)
            if debug >= 1:
                print('Iter %d stopping score %.3f' % (iter_no, this_score))
            if this_score > last_score:  # STOP EARLY
                if debug >= 1:
                    print('Stopping early at iter %d' % iter_no)
                W = W_prev
                T = T_prev
                break
            # else this_score <= last_score
            last_score = this_score
            W_prev = copy.deepcopy(W)
            T_prev = copy.deepcopy(T)

        if debug >= 1:
            it_start_time = time.time()

        grad_norm_this_iter = 0

        if store_intermediate:
            rtv['numer_W'][iter_no] = []
            rtv['denom_W'][iter_no] = []

        for t in range(k):
            if debug >= 2:
                print('\tTopic %d' % t)
            if not fix_T:
                if debug >= 2:
                    print('\t\tT{}'.format(t))
                if not (W_mat is not None):
                    w = W[:, t]
                    wX = w.T.dot(X)
                    wW = w.T.dot(W)
                    wW[t] = 0
                    wR = wX - wW.dot(T)
                    nw = (W[:, t]**2).sum()  # ||W[:, t]||^2, this is a scalar
                    if store_intermediate and not (I_store is not None):
                        wR_store = wR
                        nw_store = nw
                    elif store_intermediate and I_store is not None:
                        ws = W[I_store, :][:, t]
                        wXs = ws.T.dot(X[I_store, :])
                        wWs = ws.T.dot(W[I_store, :])
                        wWs[t] = 0
                        wR_store = wXs - wWs.dot(T)
                        nw_store = (ws**2).sum()
                else:
                    Rt = X - np.dot(W, T) + np.dot(rshp(W[:, t]),
                                                   rshp(T[t, :]).T)
                    Rt = ne_eval('Rt * W_mat')
                    wR = np.dot(W[:, t].T, Rt)
                    nw = np.dot(rshp(W[:, t]**2).T, W_mat).ravel()
                    # this is a vector
                    #  but python broadcasting implements Lemma 6.5 correct
                    if store_intermediate and not (I_store is not None):
                        wR_store = wR
                        nw_store = nw
                    elif store_intermediate and I_store is not None:
                        wR_store = np.dot(W[I_store, :][:, t].T, Rt[I_store, :])
                        nw_store = np.dot(rshp(W[I_store, :][:, t]**2).T,
                                          W_mat[I_store, :]).ravel()

                if eps_gauss_t and delta_gauss_t:
                    # we received non-Nones for both and the intent is to use
                    #  the Gaussian differentially private mechanism
                    #  pg 261 of Dwork Roth Differential Privacy
                    c2 = 2 * np.log(1.25 / delta_gauss_t)
                    df2 = 1.0  # an upper bound on the l2 sensitivity here
                    sigma2 = c2 * df2**2 * (1 / eps_gauss_t)**2
                    N = gaussian(0, np.sqrt(sigma2))  # scipy's norm takes mean,
                    #  std
                    wR += N.rvs(wR.size).reshape(wR.shape)
                    nw += N.rvs(nw.size).reshape(nw.shape)

                numer = (wR - reg_t_l1)
                denom = nw + reg_t_l2
                if debug >= 3:
                    print('\t\t\tdenom (||W_t||+rt2) == {'
                          '0:.2e}+{1:.2e} = {2:.2e}'.format(nw, reg_t_l2,
                                                            scipy.maximum(
                                                                nw + reg_t_l2,
                                                                0)))

                if store_intermediate:
                    rtv['numer_W'][iter_no].append(wR_store)
                    rtv['denom_W'][iter_no].append(nw_store)

                if abs(damping_t) > 0:
                    numer = numer + damping_t * T[t, :]

                # we need to divide everyithng because denom may make things
                # negative
                # if iter_no == 1:
                #     print(t)
                #
                # import pdb; pdb.set_trace()
                if negative_denom_correction:
                    project_and_check_reset_t()

                if np.all(denom > 0) or not negative_denom_correction:
                    if debug >= 4:
                        print('\t\t\tdenom > 0 using soln of grad==0')
                    T[t, :] = np.maximum(numer, 0) / (denom + np.spacing(1))
                elif np.any(denom <= 0):  # correct to include ==0 here because
                    # then
                    # the objective is a line, so minimum will be at boundary
                    if debug >= 2:
                        # print('\t\t\tlin={}'.format(-numer))
                        print('\t\t\tdenom <= 0 using qf_min')
                    if debug >= 3:
                        obj_before = OBJ.true_objective(X, W, T)
                    T[t, :] = qf_min(-numer, denom, s=t_row_sum)  #
                    # .reshape((1,d))
                    if debug >= 3:
                        obj_after = OBJ.true_objective(X, W, T)
                        print('\t\t\t\tChange in obj = {0:.2e}'.format(
                                obj_after - obj_before))
                #
                project_and_check_reset_t()

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
                if debug >= 2:
                    print('\t\tW{t}'.format(t=t))
                if not (W_mat is not None):
                    Xt = X.dot(T[t, :].T)
                    Tt = T.dot(T[t, :].T)
                    Tt[t] = 0
                    Rt = Xt - W.dot(Tt)
                    nt = (T[t, :]**2).sum()  # ||T[t, :]||^2, a scalar
                else:
                    Rt = X - np.dot(W, T) + np.dot(rshp(W[:, t]),
                                                   rshp(T[t, :]).T)
                    Rt = ne_eval('W_mat * Rt')
                    Rt = np.dot(Rt, T[t, :].T)
                    nt = np.dot(W_mat, rshp(T[t, :]**2)).ravel()

                numer = Rt - reg_w_l1
                if abs(damping_w) > 0:
                    numer = numer + damping_w * W[:, t]

                denom = nt + reg_w_l2
                if debug >= 3:
                    print('\t\t\tdenom W (||T_t||+rw2) == {0:.2e}+{'
                          '1:.2e} = {2:.2e}'.format(nt, reg_w_l2,
                                                    nt + reg_w_l2))

                nw1 = np.sum(W[:, t])
                if True or nw1 > 1e-10:
                    pass
                else:  # pick the largest positive residual
                    if debug >= 2:
                        print('\t\t\tReseting W{t} method={m} fixed_seed={'
                              's}'.format(t=t, m=reset_topic_method,
                                          s=fix_reset_seed))
                    if reset_topic_method == 'max_resid_document':
                        Rt = scipy.maximum(X - W.dot(T), 0)
                        Rts = (Rt**2).sum(1)
                        mi = scipy.argmax(Rts)
                        T[t, :] = Rt[mi, :]
                    elif reset_topic_method == 'random':
                        if fix_reset_seed:
                            np.random.seed(t)
                        # T[t, :] = np.random.rand(1, d)
                        # T[t, :] /= T[t, :].sum()
                        W[:, t] = np.random.rand(n)
                    elif reset_topic_method == 'iterative_T_only':
                        _, y = iterative_minimizer(X - np.dot(W, T))
                        T[t, :] = y.T
                        T[t, :] /= T[t, :].sum()
                    elif reset_topic_method == 'iterative_refinement':
                        import pdb;
                        pdb.set_trace()

                        x, y = iterative_minimizer(X - np.dot(W, T))
                        T[t, :] = y.T
                        T[t, :] /= T[t, :].sum()
                        W[:, t] = x.ravel()
                    if debug >= 5 and W[:, t].size <= 50:
                        print('\t\t\t\tW reset to', W[:, t])

                if np.all(denom > 0) or not negative_denom_correction:
                    if debug >= 4:
                        print('\t\t\tdenom > 0 using soln of grad==0')
                    W[:, t] = np.maximum(numer, 0) / (denom + np.spacing(1))
                elif np.any(denom <= 0):
                    if debug >= 2:
                        # print('\t\t\tlin={}'.format(-numer))
                        print('\t\t\tdenom <= 0 using qf_min')
                    if debug >= 3:
                        obj_before = OBJ.true_objective(X, W, T)
                    W[:, t] = qf_min(-numer, denom, s=None, w_correction=True)
                    # .reshape((
                    # n,1))
                    if debug >= 3:
                        obj_after = OBJ.true_objective(X, W, T)
                        print('\t\t\t\tChange in obj = {0:.2e}'.format(
                                obj_after - obj_before))

                assert np.all(W[:, t] >= 0), 'W contains negative entries'

                # import pdb; pdb.set_trace()
                grad = W[:, t] * denom - numer
                grad = grad**2  # for purposes of computing Frob norm
                if iter_no == 0:  # dont use proj grad on first iter (Lin 2007)
                    grad_norm_this_iter += _projected_gradient(grad, W[:, t],
                                                               lb=-np.inf,
                                                               ub=np.inf)
                else:
                    grad_norm_this_iter += _projected_gradient(grad, W[:, t])

        if project_W_each_iter and not fix_W and not w_row_sum is None:
            if debug >= 1:
                print('\nAfter iter {iter_no} projecting each W row'.format(
                        iter_no=iter_no))
            if debug >= 5 and W.size <= 50:
                print('Before projection:\n', W)
            c = reg_w_l2 + np.sum(T**2)
            if not negative_denom_correction or reg_w_l2 >= 0:
                if debug >= 2:
                    # print('\t\t\tlin={}'.format(-numer))
                    print('\t\t\tdenom > 0 using proj_to_simplex to proj W')
                W = proj_mat_to_simplex(W, w_row_sum)
            else:
                if debug >= 2:
                    # print('\t\t\tlin={}'.format(-numer))
                    print('\t\t\tdenom <= 0 using qf_min to proj W')
                if debug >= 3:
                    obj_before = OBJ.true_objective(X, W, T)
                for i in range(X.shape[0]):
                    W[i, :] = qf_min(-2.0 * np.dot(T, X[i, :].T), c,
                                     s=w_row_sum).reshape((1, k))
                if debug >= 3:
                    obj_after = OBJ.true_objective(X, W, T)
                    print('\t\t\t\tChange in obj = {0:.2e}'.format(
                            obj_after - obj_before))
            if debug >= 5 and W.size <= 50:
                print('After projection:\n', W)

        if compute_obj_each_iter:
            obj_history.append(OBJ.true_objective(X, W, T))
            if debug >= 1:
                print('Obj: {0:3.3e}'.format(obj_history[-1]))
        iter_cputime.append(time.clock())

        proj_gradient_norm.append(np.sqrt(grad_norm_this_iter))

        # run diagnostics after timing
        if len(diagnostics) > 0:
            for func in diagnostics:
                dval = func(X, W, T)
                rtv['diagnostics'][func.func_name].append(dval)
                if debug >= 1:
                    print('Iter {0} {1}={2}'.format(iter_no, func.func_name,
                                                    dval))

        if debug >= 1:
            t_now = time.time()
            print('Iter done; ||grad_i||/||grad_1||=%.3e; Took %.3fsec' % (
                np.sqrt(grad_norm_this_iter) / proj_gradient_norm[0],
                t_now - it_start_time))
            if debug >= 5 and W.size <= 50 and T.size <= 50:
                print('W:\n', W)
                print('T:\n', T)

        if time.time() - t_global_start >= max_time:
            if debug >= 1:
                print('\n\nSTOPPING because max_time after iter %d' % iter_no)
            break

        if first_last_stopping_condition(proj_gradient_norm,
                                         eps_stop=eps_stop) and not (
                    W_mat is not None):
            # dont use proj grad stop cond if recsys
            if debug >= 1:
                print(
                        '\n\nSTOPPING because proj grad norm after iter %d' %
                        iter_no)
            break

        if compute_obj_each_iter and universal_stopping_condition(obj_history,
                                                                  eps_stop=eps_stop):
            if debug >= 1:
                print(
                        '\n\nSTOPPING because obj_history after iter %d' %
                        iter_no)
            break

    iter_cputime = [x - start_time for x in iter_cputime]

    # project after completing iterations
    if not project_W_each_iter and not w_row_sum is None and not fix_W:
        if np.isscalar(w_row_sum):
            if debug >= 1:
                print('Post completion W row projection to {}'.format(
                    w_row_sum))
            for i in range(n):
                W[i, :] = euclidean_proj_simplex(W[i, :], s=w_row_sum)
        else:  # w_row_sum is a vector with a individual sum for each row
            if debug >= 1:
                print('Post completion W row-wise projection')
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

    if store_intermediate:
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


def _projected_gradient(grad, vec, lb=0, ub=1, zero=1e-10):
    """
    Compute the projected gradient defined as (where X=vec):
    [\grad_X^P]_ij = [\grad_X]_ij if X_ij>0 else min(0, [\grad_X]_ij)

    :param grad: vector of shape (d,) containing the gradient of vec
    :param vec: vector of shape (d,) containing the elements of vec
    :param [zero]: floating point that represents zero (1e-10 by default)
    """
    lb = lb + zero
    ub = ub - zero
    rtv = 0
    rtv += np.sum(grad[np.logical_and(vec > lb, vec < ub)])
    rtv += np.sum(scipy.minimum(grad[vec <= lb], 0))
    rtv += np.sum(scipy.maximum(grad[vec >= ub], 0))
    return rtv

def _initialize_and_validate(W_in, T_in, W_mat, X, k, init, random_state,
                             project_T_each_iter, t_row_sum, n, d, **kwargs):
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
    return W, T