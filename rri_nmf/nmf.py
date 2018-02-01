from __future__ import division, print_function

from math import sqrt, floor
import warnings
import numbers

import numpy as np
import scipy
import scipy.sparse as sp
import time
# import matplotlib.pyplot as plt
from numexpr import evaluate as ne_eval
import numexpr
import copy

from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.extmath import fast_dot
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.stats import norm as gaussian

import unittest

# maksim packages:
from matrixops.compute import *
from matrixops import transform
from optimization.utils import first_last_stopping_condition, \
    universal_stopping_condition
from merge_topicmodels.utils import stack_matrices


# from numba import jitmerge


# experimental topic reset:
def iterative_minimizer(M, n_iter=10, random_state=0):
    raise NotImplementedError('Current Implementation returns nans')
    n, d = M.shape
    np.random.seed(random_state)
    x = np.random.rand(n, 1)
    yt = np.random.rand(1, d)
    for _ in range(n_iter):
        x = np.maximum(np.dot(M, yt.T), 0) / np.sum(yt ** 2)
        yt = np.maximum(np.dot(x.T, M), 0) / np.sum(x ** 2)
    return x, yt.T


def qf_min(w, c, s=1.0, w_correction=False):
    """min_{0<=x<=1} <w,x> + 0.5*c<x,x> s.t. sum(x)=s
    Parameters
    ----------
    w: ndarray of linear coefficients
    c: scalar coefficient for quadratic term
    s: float or None sum constraint for x
    w_correction: specific to finding columns of W in RRI_NMF; should be
        false by default, otherwise try to set them proportional to objective
        gain so we have a decent solution after row-wise normalization
    """
    if np.all(c > 0):
        x = -w / (np.spacing(1) + c)
        return transform.euclidean_proj_simplex(x, s)
    if np.any(c <= 0):
        x = np.zeros_like(w)
        if s:
            I = np.argsort(w + c)
            _s = int(floor(s))
            r = s - _s
            x[I[0:_s]] = 1
            if r > 0:
                x[I[_s]] = r
        else:
            if not w_correction:
                x[np.argwhere(w + c < 0)] = 1
            else:
                d = w + c
                I = np.argwhere(d < 0)
                x[I] = d[I] ** 2  # we place the actual values here so when we
                # project onto the simplex, the more important values will be
                #  saved
        return x.reshape(w.shape)


class TrueObjComputer(object):
    def __init__(self, X, reg_w_l2, reg_t_l2, reg_w_l1, reg_t_l1):
        self.trXTX = np.sum(X ** 2)
        self.reg_w_l2 = reg_w_l2
        self.reg_t_l2 = reg_t_l2
        self.reg_t_l1 = reg_t_l1
        self.reg_w_l1 = reg_w_l1

    def true_objective(self, X, W, T):
        W2 = np.sum(W ** 2)
        T2 = np.sum(T ** 2)
        T1 = np.sum(np.abs(T))
        W1 = np.sum(np.abs(W))

        # base_obj = 0.5*self.trXTX \
        #            + np.trace(np.dot( np.dot(X,W), T)) \
        #            + 0.5*W2*T2
        base_obj = 0.5 * np.sum((X - np.dot(W, T)) ** 2)
        obj = base_obj + \
              self.reg_w_l2 * W2 + \
              self.reg_t_l2 * T2 + \
              self.reg_t_l1 * T1 + \
              self.reg_w_l1 * W1
        return obj


# @jit
def nmf(X, k, w_row=None, W_mat=None, fix_W=False, fix_T=False,
        random_state=None,
        init='nndsvd', T_in=[], W_in=[],
        debug=0, max_iter=200, max_time=600, eps_stop=1e-4,
        compute_obj_each_iter=False,
        project_W_each_iter=False,
        w_row_sum=None,
        project_T_each_iter=True,
        t_row_sum=1.0,
        early_stop=None,
        reset_topic_method='random', fix_reset_seed=False,
        reg_w_l2=0, reg_t_l2=0,
        reg_w_l1=0, reg_t_l1=0,
        negative_denom_correction=True, saddle_point_handling='exception',
        damping_w=0, damping_t=0,
        diagnostics=[], store_intermediate=False, I_store=None,
        n_words_beam=20,
        eps_gauss_t=None,
        delta_gauss_t=None):
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
    # TODO: early stopping based on splitting part of X into a validation set,
    #        and not using that part to update T but use it for W
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

    sparse_implemented = False
    # scipy's sparse makes me cry
    if not sparse_implemented and sp.issparse(X):
        X = X.toarray()

    X_orig = None
    if w_row is not None:
        X_orig = X.copy()
        X = np.sqrt(w_row) * X  # needs to be sqrt since we do (wX-WT)^2

    if not w_row_sum is None and not np.isscalar(w_row_sum):
        w_row_sum = w_row_sum.reshape((w_row_sum.size, 1))
        if not w_row is None:
            w_row_sum = np.sqrt(w_row_sum)  # rows of X will be multiplied by
            #  sqrt of w_row, so the rows of W should also sum to sqrt of w_row

            # TODO: change row_normalize to be project_T_each_iter?
    if n <= k:
        init = 'random'

    # import pdb; pdb.set_trace()

    def project_and_check_reset_t():
        nt1 = np.sum(T[t, :])

        if nt1 > 1e-10:
            if project_T_each_iter:
                T[t, :] = transform.euclidean_proj_simplex(T[t, :],
                                                           s=t_row_sum)
        else:  # pick the largest positive residual
            if debug >= 2:
                print('\t\t\tReseting T{t} method={m} fixed_seed={'
                      's}'.format(t=t, m=reset_topic_method,
                                  s=fix_reset_seed))
            if reset_topic_method == 'max_resid_document':
                Rt = scipy.maximum(X - W.dot(T), 0)
                Rts = (Rt ** 2).sum(1)
                mi = scipy.argmax(Rts)
                T[t, :] = Rt[mi, :]
            elif reset_topic_method == 'random':
                if fix_reset_seed:
                    np.random.seed(t)
                T[t, :] = np.random.rand(1, d)
                T[t, :] /= T[t, :].sum()
            elif reset_topic_method == 'iterative_T_only':
                _, y = iterative_minimizer(X - np.dot(W, T))
                T[t, :] = y.T
                T[t, :] /= T[t, :].sum()
            elif reset_topic_method == 'iterative_refinement':
                x, y = iterative_minimizer(X - np.dot(W, T))
                T[t, :] = y.T
                T[t, :] /= T[t, :].sum()
                W[:, t] = x.ravel()
            if debug >= 5 and T[t, :].size <= 50:
                print('\t\t\t\tReset to {}'.format(T[t, :]))

    if np.prod(np.shape(W_in)) == 0 or np.prod(np.shape(T_in)) == 0:
        if not W_mat is None:
            W, T = _initialize_nmf(W_mat * X, k, init,
                                   random_state=random_state,
                                   row_normalize=False,
                                   n_words_beam=n_words_beam)
        else:
            W, T = _initialize_nmf(X, k, init, random_state=random_state,
                                   row_normalize=False,
                                   n_words_beam=n_words_beam)
        if project_T_each_iter:
            T = transform.normalize(T) * t_row_sum
            # if project_W_each_iter and not w_row_sum is None:
            #    W = transform.normalize(W) * w_row_sum
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

        if not sparse_implemented and sp.issparse(W_mat):
            W_mat = W_mat.toarray()

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
                W[i, :] = transform.euclidean_proj_simplex(W[i, :],
                                                           s=w_row_sum)
        else:  # w_row_sum is a vector with a individual sum for each row
            for i in range(n):
                W[i, :] = transform.euclidean_proj_simplex(W[i, :],
                                                           s=w_row_sum[i])

    if project_T_each_iter and not fix_T and not t_row_sum is None:
        T = transform.proj_mat_to_simplex(T, t_row_sum)

    if compute_obj_each_iter:
        OBJ = TrueObjComputer(X, reg_w_l1=reg_w_l1, reg_t_l2=reg_t_l2,
                              reg_w_l2=reg_w_l2, reg_t_l1=reg_t_l1)
        obj_history = [OBJ.true_objective(X, W, T)]  # obj after i iters
    else:
        obj_history = []

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
                    nw = (W[:, t] ** 2).sum()  # ||W[:, t]||^2, this is a scalar
                    if store_intermediate and not (I_store is not None):
                        wR_store = wR
                        nw_store = nw
                    elif store_intermediate and I_store is not None:
                        ws = W[I_store, :][:, t]
                        wXs = ws.T.dot(X[I_store, :])
                        wWs = ws.T.dot(W[I_store, :])
                        wWs[t] = 0
                        wR_store = wXs - wWs.dot(T)
                        nw_store = (ws ** 2).sum()
                else:
                    Rt = X \
                         - np.dot(W, T) \
                         + np.dot(rshp(W[:, t]), rshp(T[t, :]).T)
                    Rt = ne_eval('Rt * W_mat')
                    wR = np.dot(W[:, t].T, Rt)
                    nw = np.dot(rshp(W[:, t] ** 2).T, W_mat).ravel()
                    # this is a vector
                    #  but python broadcasting implements Lemma 6.5 correct
                    if store_intermediate and not (I_store is not None):
                        wR_store = wR
                        nw_store = nw
                    elif store_intermediate and I_store is not None:
                        wR_store = np.dot(
                            W[I_store, :][:, t].T, Rt[I_store, :])
                        nw_store = np.dot(rshp(W[I_store, :][:, t] ** 2).T,
                                          W_mat[I_store, :]).ravel()

                if eps_gauss_t and delta_gauss_t:
                    # we received non-Nones for both and the intent is to use
                    #  the Gaussian differentially private mechanism
                    #  pg 261 of Dwork Roth Differential Privacy
                    c2 = 2*np.log(1.25/delta_gauss_t)
                    df2 = 1.0  # an upper bound on the l2 sensitivity here
                    sigma2 = c2 * df2**2 * (1/eps_gauss_t)**2
                    N = gaussian(0, np.sqrt(sigma2))  # scipy's norm takes mean,
                    #  std
                    wR += N.rvs(wR.size).reshape(wR.shape)
                    nw += N.rvs(nw.size).reshape(nw.shape)

                numer = (wR - reg_t_l1)
                denom = nw + reg_t_l2
                if debug >= 3:
                    print('\t\t\tdenom (||W_t||+rt2) == {'
                          '0:.2e}+{1:.2e} = {2:.2e}'.format(
                        nw,
                        reg_t_l2,
                        scipy.maximum(nw + reg_t_l2, 0))
                    )

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
                grad = grad ** 2  # for purposes of computing Frob norm
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
                    nt = (T[t, :] ** 2).sum()  # ||T[t, :]||^2, a scalar
                else:
                    Rt = X \
                         - np.dot(W, T) \
                         + np.dot(rshp(W[:, t]), rshp(T[t, :]).T)
                    Rt = ne_eval('W_mat * Rt')
                    Rt = np.dot(Rt, T[t, :].T)
                    nt = np.dot(W_mat, rshp(T[t, :] ** 2)).ravel()

                numer = Rt - reg_w_l1
                if abs(damping_w) > 0:
                    numer = numer + damping_w * W[:, t]

                denom = nt + reg_w_l2
                if debug >= 3:
                    print('\t\t\tdenom W (||T_t||+rw2) == {0:.2e}+{'
                          '1:.2e} = {2:.2e}'.format(
                        nt,
                        reg_w_l2,
                        nt + reg_w_l2)
                    )

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
                        Rts = (Rt ** 2).sum(1)
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
                grad = grad ** 2  # for purposes of computing Frob norm
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
            c = reg_w_l2 + np.sum(T ** 2)
            if not negative_denom_correction or reg_w_l2 >= 0:
                if debug >= 2:
                    # print('\t\t\tlin={}'.format(-numer))
                    print('\t\t\tdenom > 0 using proj_to_simplex to proj W')
                W = transform.proj_mat_to_simplex(W, w_row_sum)
            else:
                if debug >= 2:
                    # print('\t\t\tlin={}'.format(-numer))
                    print('\t\t\tdenom <= 0 using qf_min to proj W')
                if debug >= 3:
                    obj_before = OBJ.true_objective(X, W, T)
                for i in range(X.shape[0]):
                    W[i, :] = qf_min(-2.0*np.dot(T, X[i, :].T), c,
                                     s=w_row_sum).reshape((1, k))
                if debug >= 3:
                    obj_after = OBJ.true_objective(X, W, T)
                    print('\t\t\t\tChange in obj = {0:.2e}'.format(
                        obj_after - obj_before))
            if debug >= 5 and W.size <= 50:
                print('After projection:\n', W)

        if compute_obj_each_iter:
            if not (W_mat is not None):
                obj_history.append(OBJ.true_objective(X, W, T))
            else:
                warnings.warn(warnings.WarningMessage(
                    'nmf: compute objective not supported if W_mat is input'))
                Xh = np.dot(W, T)
                # obj_history.append(np.sum((X[Ix, Jx] - Xh[Ix, Jx])**2))
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
                    print('Iter {0} {1}={2}'.format(iter_no,
                                                    func.func_name,
                                                    dval
                                                    )
                          )

        if debug >= 1:
            t_now = time.time()
            print('Iter done; ||grad_i||/||grad_1||=%.3e; Took %.3fsec'
                  % (np.sqrt(grad_norm_this_iter) / proj_gradient_norm[0],
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
                print('\n\nSTOPPING because proj grad norm after iter %d' %
                      iter_no)
            break

        if compute_obj_each_iter and \
                universal_stopping_condition(obj_history, eps_stop=eps_stop):
            if debug >= 1:
                print('\n\nSTOPPING because obj_history after iter %d' %
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
                W[i, :] = transform.euclidean_proj_simplex(W[i, :],
                                                           s=w_row_sum)
        else:  # w_row_sum is a vector with a individual sum for each row
            if debug >= 1:
                print('Post completion W row-wise projection')
            for i in range(n):
                W[i, :] = transform.euclidean_proj_simplex(W[i, :],
                                                           s=w_row_sum[i])

    if w_row is not None:
        sub = nmf(X_orig, k, T_in=T, fix_T=True, max_iter=10,
                  w_row_sum=w_row_sum,
                  project_W_each_iter=True,
                  compute_obj_each_iter=compute_obj_each_iter)
        for oh in sub['obj_history']:
            obj_history.append(oh)
        for itc in sub['iter_cputime']:
            iter_cputime.append(itc)
        W = sub['W']

    if store_intermediate:
        for itno in rtv['numer_W']:
            rtv['numer_W'][itno] = stack_matrices(
                rtv['numer_W'][itno], transform=lambda row:
                row.reshape((1, row.size))
            )
        for itno in rtv['denom_W']:
            rtv['denom_W'][itno] = stack_matrices(
                rtv['denom_W'][itno], transform=lambda row:
                row.reshape((1, row.size))
            )

    rtv['W'] = W
    rtv['T'] = T
    if compute_obj_each_iter:
        rtv['obj_history'] = obj_history
        rtv['obj_calculator'] = OBJ
    rtv['proj_gradient_norm'] = proj_gradient_norm
    rtv['iter_cputime'] = [0] + iter_cputime
    rtv['random_state'] = random_state

    return rtv


def _safe_compute_error(X, W, H, use_old_impl=False):
    """Frobenius norm between X and WH, safe for sparse array

    copied directly from sklearn.decomposition.nmf
    Orignal Authors:
         Vlad Niculae
         Lars Buitinck
         Mathieu Blondel <mathieu@mblondel.org>
         Tom Dupre la Tour
         Chih-Jen Lin, National Taiwan University
    """

    if not sp.issparse(X):
        error = norm(X - np.dot(W, H))
    else:
        norm_X = np.dot(X.data, X.data)
        norm_WH = trace_dot(np.dot(np.dot(W.T, W), H), H)
        cross_prod = trace_dot((X * H.T), W)
        error = sqrt(norm_X + norm_WH - 2. * cross_prod)
    return error


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


def _initialize_nmf(X, n_components, init=None, eps=1e-6,
                    random_state=None, row_normalize=False,
                    n_words_beam=20):
    """Algorithms for NMF initialization.
    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH

    copied directly from sklearn.decomposition.nmf
    Orignal Authors:
         Vlad Niculae
         Lars Buitinck
         Mathieu Blondel <mathieu@mblondel.org>
         Tom Dupre la Tour
         Chih-Jen Lin, National Taiwan University


    Parameters
    ----------

    m: number of words to use to beam search for in coherence_pmi init
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    n_components : integer
        The number of components desired in the approximation.
    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: 'nndsvdar' if n_components < n_features, otherwise 'random'.
        Valid options:
        - 'coherence_pmi': beam search to maximize pointwise mutual information
            of m words in each topic
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)
        - 'custom': use custom matrices W and H
    eps : float
        Truncate all values less then this in output to zero.
    random_state : int seed, RandomState instance, or None (default)
        Random number generator seed control, used in 'nndsvdar' and
        'random' modes.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH
    H : array-like, shape (n_components, n_features)
        Initial guesses for solving X ~= WH
    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    #  check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if init is None:
        if n_components < n_features:
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        #        avg = np.sqrt(X.mean() / n_components)
        #        rng = check_random_state(random_state)
        #        H = avg * rng.randn(n_components, n_features)
        #        W = avg * rng.randn(n_samples, n_components)
        #        # we do not write np.abs(H, out=H) to stay compatible with
        #        # numpy 1.5 and earlier where the 'out' keyword is not
        #        # supported as a kwarg on ufuncs
        #        np.abs(H, H)
        #        np.abs(W, W)
        #        return W, H
        rng = check_random_state(random_state)
        T = rng.rand(n_components, n_features)
        W = rng.rand(n_samples, n_components)
        # W = transform.normalize(W)
        if row_normalize:
            T = transform.normalize(T)
        return W, T

    if init == 'coherence_pmi':
        X = transform.normalize(transform.tfidf(X))
        C = np.dot(X.T, X)
        k = n_components

        [n, d] = X.shape
        P_i = np.log(C.sum(1) + np.spacing(1))
        P_ij = np.log(C + np.spacing(1))

        xs = X.sum(0)
        T = []
        for t in range(k):
            j = np.argmax(xs)
            xs[j] = 0  # dont use this test again
            tpc = [j]
            for i in range(1, n_words_beam):
                best_score = -1 * np.inf
                best_word = None
                for jj in range(d):
                    if xs[jj] > 0:  # this word is still available
                        score_jj = 0
                        for c in tpc:
                            score_jj += P_ij[jj, c] - P_i[jj] - P_i[c]
                        if score_jj > best_score:
                            best_score = score_jj
                            best_word = jj
                tpc.append(best_word)
                xs[best_word] = 0  # dont use this word again
            T.append(tpc)
        J = T
        xs = X.sum(0)
        T = np.zeros((k, d))
        for t in range(k):
            tpc = J[t]
            for j in tpc:
                # wt of word in a topic proportional to its global importance
                T[t, j] = xs[j]

        T = transform.normalize(T)
        W = transform.normalize(scipy.maximum(np.dot(X, T.T), 0))
        return W, T

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    if row_normalize:
        # W = transform.normalize(W)
        H = transform.normalize(H)

    return W, H


# import scikitlearn as sklearn
import sklearn
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array, \
    check_non_negative
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
import inspect


class NMF_RS_Estimator(sklearn.base.BaseEstimator):
    def __init__(self, n, d, k, wr1=0, wr2=0, tr1=0, tr2=0, random_state=0,
                 W=np.array([]), T=np.array([]), max_iter=30, nmf_kwargs={},
                 use_early_stopping=True):

        self.n = n
        self.d = d
        self.k = k

        self.max_iter = max_iter
        self.wr1 = wr1
        self.wr2 = wr2
        self.tr1 = tr1
        self.tr2 = tr2
        self.random_state = random_state

        self.min_rating = None
        self.max_rating = None

        self.Xpred = np.array([])

        self.use_early_stopping = use_early_stopping

        self.W = W
        self.T = T

        self.nmf_kwargs = nmf_kwargs

    # def __getitem__(self, key): return self.__getattribute__(key)

    def sparsify(self):
        if not sp.issparse(self.W):
            self.W = sp.csr_matrix(self.W)
        else:
            self.W = self.W.tocsr()

        if not sp.issparse(self.T):
            self.T = sp.csr_matrix(self.T)
        else:
            self.T = self.T.tocsr()

    def densify(self):
        if sp.issparse(self.W):
            self.W = self.W.toarray()
        if sp.issparse(self.T):
            self.T = self.T.toarray()

    def fit(self, X, y=None):
        """
            X - n*2 indexes, i.e. (i, j) pairs
            y - n*1 values of X[i,j]
        """
        # X, y = check_X_y(X, y)

        max_iter = self.max_iter

        self.min_rating = np.min(y)
        self.max_rating = np.max(y)

        if self.use_early_stopping:

            UItr, UIval, Rtr, Rval = train_test_split(
                X,
                y,
                test_size=0.05,
                random_state=0,
                stratify=None
            )

            # ntr, dtr = len(np.unique(UItr[:,0])), len(np.unique(UItr[:,1]))
            Xtr = sp.coo_matrix((Rtr, (UItr[:, 0], UItr[:, 1])),
                                shape=(self.n, self.d)).toarray()

            # nv, dv = len(np.unique(UIval[:,0])), len(np.unique(UIval[:,1]))
            Xv = sp.coo_matrix((Rval, (UIval[:, 0], UIval[:, 1])),
                               shape=(self.n, self.d)).toarray()

            def RMSE_val(X, W, T):
                # X is ignore
                # return 99
                I, J = Xv.nonzero()
                Xpred = np.dot(W, T)
                Xpred = np.clip(Xpred, self.min_rating, self.max_rating)
                return np.sqrt(np.mean((Xpred[I, J] - Xv[I, J]) ** 2))

            early_stop = RMSE_val

        else:
            early_stop = None
            Xtr = sp.coo_matrix((y, (X[:, 0], X[:, 1])),
                                shape=(self.n, self.d)).toarray()

        W_mat_tr = np.zeros(Xtr.shape)
        [Itr, Jtr] = Xtr.nonzero()
        W_mat_tr[Itr, Jtr] = 1

        # we can continue fitting an existing model
        if self.W.size > 0:
            W_in = self.W
        else:
            W_in = []
        if self.T.size > 0:
            T_in = self.T
        else:
            T_in = []

        # import pdb; pdb.set_trace()

        soln = nmf(Xtr, self.k, max_iter=self.max_iter, max_time=7200,
                   project_W_each_iter=False, project_T_each_iter=True,
                   W_mat=W_mat_tr,
                   W_in=W_in, T_in=T_in,
                   early_stop=early_stop,
                   reg_w_l1=self.wr1, reg_w_l2=self.wr2,
                   reg_t_l1=self.tr1, reg_t_l2=self.tr2,
                   random_state=self.random_state,
                   negative_denom_correction=True,
                   **self.nmf_kwargs
                   )

        self.W = soln.pop('W')
        self.T = soln.pop('T')
        self.nmf_outputs = soln

        return self

    def fit_from_Xtr(self, Xtr):
        """ construct X,y from Xtr and send it to fit """
        # import pdb; pdb.set_trace()
        if sp.issparse(Xtr):
            Xtr = Xtr.tocsr()
        else:
            Xtr = sp.csr_matrix(Xtr)
        NZ = Xtr.nonzero()
        # X = np.array([(NZ[0][i], NZ[1][i]) for i in range(len(NZ[0]))])
        X = np.hstack((NZ[0].reshape((NZ[0].size, 1)),
                       NZ[1].reshape((NZ[1].size, 1))
                       )
                      )
        y = Xtr.data
        return self.fit(X, y)

    def transform(self, Xnew):
        """express Xnew in terms of topics self.T"""
        W_mat_tr = np.zeros(Xnew.shape)
        [Itr, Jtr] = Xnew.nonzero()
        W_mat_tr[Itr, Jtr] = 1

        soln = nmf(Xnew, self.k, max_iter=4, max_time=7200,
                   project_W_each_iter=False,
                   project_T_each_iter=True,
                   W_mat=W_mat_tr,
                   T_in=self.T, fix_T=True,
                   reg_w_l1=self.wr1, reg_w_l2=self.wr2,
                   reg_t_l1=self.tr1, reg_t_l2=self.tr2,
                   random_state=self.random_state,
                   negative_denom_correction=True,
                   **self.nmf_kwargs)
        return soln['W']

    def make_Xpred(self):
        if self.Xpred.size == 0:
            self.Xpred = np.dot(self.W, self.T)
            self.Xpred = np.clip(self.Xpred,
                                 a_min=self.min_rating,
                                 a_max=self.max_rating
                                 )

    def predict(self, X):
        self.make_Xpred()
        check_is_fitted(self, ['W', 'T'])

        X = check_array(X)

        return self.Xpred[X[:, 0], X[:, 1]]

    def score(self, X, y=np.array([])):
        """Return RMSE of predictions"""
        self.make_Xpred()
        if sp.issparse(X):
            X = X.toarray()
        if y.size > 0:
            yh = self.predict(X)
            return np.sqrt(np.mean((y - yh) ** 2))
        else:  # X is a n*d matrix
            I, J = X.nonzero()
            return np.sqrt(np.mean((X[I, J] - self.Xpred[I, J]) ** 2))


class NMF_TM_Estimator(sklearn.base.BaseEstimator,
                       sklearn.base.TransformerMixin):
    def __init__(self, n, d, k, wr1=0, wr2=0, tr1=0, tr2=0, random_state=0,
                 handle_tfidf=False, handle_normalization=False, max_iter=300,
                 W=np.array([]), T=np.array([]), nmf_kwargs={}):
        """

        Parameters
        ----------
        n : int
            number of documents
        d : int
            size of dictionary
        k : int
            number of topics
        wr1 : float [0]
            regularization for l_1 norm of W
        wr2 : float [0]
            regularization for l_2 norm of W
        tr1 : float [0]
            regularization for l_1 norm of T
        tr2 : float [0]
            regularization of l_2 norm of T
        handle_tfidf : boolean [False]
            Apply tfidf before fit / transform
        handle_normalization: boolean [False]
            Normalize rows to sum to 1 before fit/transform
        W = n*k np.array [np.array([])]
            initial W matrix for NMF, initialized automatically if empty
        T = k*d np.array [(np.array([]))]
            initial T matrix for NMF, initialized automatically if empty
        nmf_kwargs = dictionary [{}]
            additional keywoard arguments to pass to the nmf method. See
            @nmf for options.
        """

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    # def __getitem__(self, key): return self.__getattribute__(key)

    def sparsify(self):
        if not sp.issparse(self.W):
            self.W = sp.csr_matrix(self.W)
        else:
            self.W = self.W.tocsr()

        if not sp.issparse(self.T):
            self.T = sp.csr_matrix(self.T)
        else:
            self.T = self.T.tocsr()

    def densify(self):
        if sp.issparse(self.W):
            self.W = self.W.toarray()
        if sp.issparse(self.T):
            self.T = self.T.toarray()

    def fit_transform(self, X, y=None):
        """
            X - n*d
        """
        assert np.all(X >= 0), 'X must be non-negative'

        # we can continue fitting an existing model
        if self.W.size > 0:
            W_in = self.W
        else:
            W_in = []
        if self.T.size > 0:
            T_in = self.T
        else:
            T_in = []

        if self.handle_tfidf:
            X, idf = transform.tfidf(X, return_idf=True)
            self.idf = idf
        if self.handle_normalization:
            X = transform.normalize(X)

        soln = nmf(X, self.k, max_iter=self.max_iter, max_time=7200,
                   project_W_each_iter=True,
                   w_row_sum=1,
                   project_T_each_iter=True,
                   W_in=W_in, T_in=T_in,
                   reg_w_l1=self.wr1, reg_w_l2=self.wr2,
                   reg_t_l1=self.tr1, reg_t_l2=self.tr2,
                   negative_denom_correction=True,
                   random_state=self.random_state,
                   **self.nmf_kwargs)

        self.W = soln.pop('W')
        self.T = soln.pop('T')
        self.nmf_outputs = soln

        return self.W

    def one_iter(self, X):
        if self.W.size > 0:
            W_in = self.W
        else:
            W_in = []
        if self.T.size > 0:
            T_in = self.T
        else:
            T_in = []

        if self.handle_tfidf:
            X, idf = transform.tfidf(X, return_idf=True)
            self.idf = idf
        if self.handle_normalization:
            X = transform.normalize(X)

        soln = nmf(X, self.k, max_iter=1, max_time=240,
                   project_W_each_iter=True,
                   w_row_sum=1,
                   project_T_each_iter=True,
                   W_in=W_in, T_in=T_in,
                   reg_w_l1=self.wr1, reg_w_l2=self.wr2,
                   reg_t_l1=self.tr1, reg_t_l2=self.tr2,
                   random_state=self.random_state,
                   negative_denom_correction=True,
                   **self.nmf_kwargs)

        self.W = soln.pop('W')
        self.T = soln.pop('T')
        self.nmf_outputs = soln

        return self

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, Xnew):
        """express Xnew in terms of topics self.T"""
        if self.handle_tfidf:
            Xnew = Xnew * self.idf
        if self.handle_normalization:
            Xnew = transform.normalize(Xnew)

        soln = nmf(Xnew, self.k, max_iter=4, max_time=7200,
                   project_W_each_iter=True,
                   w_row_sum=1,
                   T_in=self.T, fix_T=True,
                   reg_w_l1=self.wr1, reg_w_l2=self.wr2,
                   reg_t_l1=self.tr1, reg_t_l2=self.tr2,
                   negative_denom_correction=True,
                   random_state=self.random_state)
        return soln['W']

    def constrained_transform(self, X):
        return self.transform(X)

    def score(self, X, y=None):
        """Return R^2 of new X """

        SST = ((X - np.mean(X, axis=0)) ** 2).sum()
        W = self.transform(X)
        SSE = ((X - np.dot(W, self.T)) ** 2).sum()
        return 1 - SSE / SST


# @profile


def tm_prof(dn='20NG', k=10, mi=5):
    ds = datasets.load_dataset(dn)
    X = ds['X'].toarray()
    return nmf(X, k, max_iter=5, random_state=0)


# @profile


def rs_prof(dn='ML-1M', k=10, mi=5):
    ds = datasets.load_recsys_dataset(dn)
    X = sp.coo_matrix((ds['R'], (ds['UI'][:, 0], ds['UI'][:, 1])),
                      shape=(ds['n'], ds['d'])).toarray()
    W_mat_tr = np.zeros(X.shape)
    [Itr, Jtr] = X.nonzero()
    W_mat_tr[Itr, Jtr] = 1

    return nmf(X, k, max_iter=mi,
               W_mat=W_mat_tr,
               random_state=0)


if __name__ == '__main__':
    from matlabinterface import datasets

    reference_alg = True

    sol_rs = rs_prof()
    sol_tm = tm_prof()

    if reference_alg:
        with open('/home/maks/experiments/NMF_Bench_sol_tm.pickle', 'w') as f:
            datasets.dump(sol_tm, f)
        with open('/home/maks/experiments/NMF_Bench_sol_rs.pickle', 'w') as f:
            datasets.dump(sol_rs, f)
    else:
        with open('/home/maks/experiments/NMF_Bench_sol_tm.pickle', 'r') as f:
            ref_tm = datasets.load(f)
        with open('/home/maks/experiments/NMF_Bench_sol_rs.pickle', 'r') as f:
            ref_rs = datasets.load(f)
        for (k, v) in {'RS': (sol_rs, ref_rs), 'TM': (sol_tm, ref_tm)}:
            for kk in v[0]:
                if not np.allclose(v[0][kk], v[1][kk]):
                    print(kk, ' not all close on ', k)
                    # unittest.main()
