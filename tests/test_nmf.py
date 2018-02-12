import scipy as sp
import numpy as np
from rri_nmf.initialization import initialize_nmf
from rri_nmf.matrixops import proj_mat_to_simplex
from rri_nmf.nmf import nmf, eps_div_by_zero, _compute_update_T
from rri_nmf.sklearn_interface import NMF_RS_Estimator, NMF_TM_Estimator

import pytest

constraint_violation_tolerance = 1e-13


def test_init(small_X_W_T):
    X, Wt, Tt = small_X_W_T

    W, T = initialize_nmf(X, 2, init='nndsvd', random_state=0)

    assert np.allclose(Wt, W)
    assert np.allclose(Tt, T)


@pytest.mark.parametrize('nmf_params', [{'k': 25}, {'k': 15, 'reg_t_l2': 0.1},
                                        {'k': 15, 'reg_t_l2': -0.1},
                                        {'k': 15, 'reg_w_l2': 0.1},
                                        # {'k': 15, 'reg_w_l2': -0.1}
])
def test_convergence_tm_setting(nmf_params, text_train):
    """Test that topic modeling NMF converges with a variety of params"""
    X = text_train
    tm_settings = {
        'max_iter'           : 15, 'w_row_sum': 1.0, 'random_state': 0,
        'eps_stop'           : 1e-4, 'project_T_each_iter': True,
        'project_W_each_iter': True, 'compute_obj_each_iter': True,
        't_row_sum'          : 1.0, 'early_stop': False
    }
    nmf_params.update(tm_settings)
    soln = nmf(X, **nmf_params)
    oh = soln['obj_history']
    # print nmf_params, oh[-1], np.sum(np.abs(soln['W'].sum(1)-1)), oh
    assert np.all(np.diff(oh) <= 0)
    assert _constraint_violation_WT(soln['W'],
                                    soln['T']) <= constraint_violation_tolerance


def _constraint_violation_WT(W, T, quiet=True):
    assert np.all(W >= 0 - constraint_violation_tolerance), 'W>=0 doesnt hold'
    assert np.all(T >= 0 - constraint_violation_tolerance), 'T>=0 doesnt hold'
    cvW = np.sum(np.abs(W.sum(1) - 1))
    if not quiet:
        print 'constraint violation W', cvW
    cvT = np.sum(np.abs(T.sum(1) - 1))
    if not quiet:
        print 'constraint violation T', cvT
    return cvW + cvT


@pytest.mark.parametrize('nmf_params', [{}, {'reg_w_l1': 0.1, 'reg_t_l1': 0.1},
                                        {'reg_w_l1': 0.1}, {'reg_t_l1': 0.1}, ])
def test_convergence_rs_setting(nmf_params, recsys_train):
    """Test that topic modeling NMF converges with a variety of params"""
    X = recsys_train
    Wm = np.zeros(X.shape)
    [Itr, Jtr] = X.nonzero()
    Wm[Itr, Jtr] = 1.0

    rs_settings = {
        'max_iter'             : 15, 'random_state': 0, 'W_mat': Wm,
        'compute_obj_each_iter': True, 'reset_topic_method': None,
        'early_stop'           : False, 'k': 7, 'project_T_each_iter': False,
        't_row_sum'            : 1.0, 'project_W_each_iter': False,
        'w_row_sum'            : None
    }
    nmf_params.update(rs_settings)

    soln = nmf(X, **nmf_params)
    oh = soln['obj_history']
    # print nmf_params, oh[-1], oh
    assert np.all(np.diff(oh) <= 0)


def test_convergence_RS_Estimator(recsys_train):
    """Test that the sklearn interface for RS converges"""
    X = recsys_train
    n, d = X.shape
    E = NMF_RS_Estimator(n, d, 5, random_state=0, max_iter=20)
    E = E.fit_from_Xtr(X)
    score = E.score(X)
    assert score < 1.0

def test_convergence_TM_Estimator(text_train):
    X = text_train
    n, d = X.shape
    M = NMF_TM_Estimator(n, d, 5, random_state=0, max_iter=10)
    M = M.fit(X)
    print M.nmf_outputs['obj_history']
    assert np.linalg.norm(X-np.dot(M.W, M.T), 'fro') < np.linalg.norm(X,'fro')
    M2 = NMF_TM_Estimator(n,d,5, random_state=0, max_iter=2,
                          do_final_project_W=False)
    M2 = M2.fit(X)
    print M2.nmf_outputs['obj_history']
    M2.max_iter = 10
    for _ in range(7):
        M2 = M2.one_iter(X)
        print M2.nmf_outputs['obj_history']
    M2 = M2.one_iter(X)
    M2.W = proj_mat_to_simplex(M2.W)
    print M2.nmf_outputs['obj_history']

    assert np.allclose(M2.T, M.T)
    assert np.allclose(M2.W, M.W)