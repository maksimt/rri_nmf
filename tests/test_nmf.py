import scipy as sp
import numpy as np
from rri_nmf.initialization import initialize_nmf
from rri_nmf.nmf import nmf
from rri_nmf.sklearn_interface import NMF_RS_Estimator, NMF_TM_Estimator

import pytest


def test_init(small_X_W_T):
    X, Wt, Tt = small_X_W_T

    W, T = initialize_nmf(X, 2, init='nndsvd', random_state=0)

    assert np.allclose(Wt, W)
    assert np.allclose(Tt, T)


@pytest.mark.parametrize('nmf_params',
                         [{'reg_w_l2': 0.0, 'project_W_each_iter': True},
                             {'reg_w_l2': 0.0},
                             {'reg_w_l2': 0.1},
                          {'reg_w_l2': -0.1, 'project_W_each_iter':True},
                             {'reg_t_l2': 0.1},
                          {'reg_t_l2': -0.1}, ])
def test_convergence_tm_setting(nmf_params, text_train):
    """Test that topic modeling NMF converges with a variety of params"""
    X = text_train
    tm_settings = {
        'k'                    : 5, 'max_iter': 10,
        'project_T_each_iter'  : True, 'random_state': 0,
        'compute_obj_each_iter': True, 'w_row_sum': 1.0,
        'reset_topic_method':'random', 'early_stop': True
    }
    nmf_params.update(tm_settings)
    soln = nmf(X, **nmf_params)
    oh = soln['obj_history']
    #print nmf_params, oh[-1], np.sum(np.abs(soln['W'].sum(1)-1)), oh
    assert np.all(np.diff(oh) <= 0)


@pytest.mark.parametrize('nmf_params', [
    {'k': 5},
    {'reg_w_l2': 0.1},
  {'reg_w_l2': -0.1},
     {'reg_t_l2': 0.1},
  {'reg_t_l2': -0.1}
])
def test_convergence_rs_setting(nmf_params, recsys_train):
    """Test that topic modeling NMF converges with a variety of params"""
    X = recsys_train
    Wm = np.zeros(X.shape)
    [Itr, Jtr] = X.nonzero()
    Wm[Itr, Jtr] = 1.0

    rs_settings = {
        'k'                  : 5, 'max_iter': 30, 'project_W_each_iter': False,
        'w_row_sum'          : None, 'project_T_each_iter': True,
        'random_state'       : 0, 'W_mat': Wm, 'compute_obj_each_iter': True,
        't_row_sum'          : 1.0,
        'reset_topic_method': 'random', 'early_stop':True
    }
    nmf_params.update(rs_settings)

    soln = nmf(X, **nmf_params)
    oh = soln['obj_history']
    #print nmf_params, oh[-1], oh
    assert np.all(np.diff(oh) <= 0)

def test_convergence_RS_Estimator(recsys_train):
    """Test that the sklearn interface for RS converges"""
    X = recsys_train
    n,d=X.shape
    E = NMF_RS_Estimator(n,d,5,random_state=0, max_iter=20)
    E = E.fit_from_Xtr(X)
    score = E.score(X)
    assert score < 1.0