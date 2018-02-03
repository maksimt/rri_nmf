import scipy as sp
import numpy as np
from rri_nmf.initialization import initialize_nmf
from rri_nmf.nmf import nmf

import pytest


def test_init(small_X_W_T):
    X, Wt, Tt = small_X_W_T

    W, T = initialize_nmf(X, 2, init='nndsvd', random_state=0)

    assert np.allclose(Wt, W)
    assert np.allclose(Tt, T)

@pytest.mark.parametrize('nmf_params',
                     [
                         {'k':5}
                     ])
def test_convergence_tm_setting(nmf_params, text_train):
    """Test that topic modeling NMF converges with a variety of params"""
    X = text_train
    tm_settings = {'max_iter':10, 'project_W_each_iter':True, 'w_row_sum':1.0,
                   'project_T_each_iter':True, 'random_state':0,
                   'compute_obj_each_iter':True}
    nmf_params.update(tm_settings)
    soln = nmf(X, **nmf_params)
    oh = soln['obj_history']
    assert np.all(np.diff(oh) <= 0)


@pytest.mark.parametrize('nmf_params',
                     [
                         {'k':5}
                     ])
def test_convergence_rs_setting(nmf_params, recsys_train):
    """Test that topic modeling NMF converges with a variety of params"""
    X = recsys_train
    Wm = np.zeros(X.shape)
    [Itr, Jtr] = X.nonzero()
    Wm[Itr, Jtr] = 1.0

    rs_settings = {'max_iter':10, 'project_W_each_iter':False, 'w_row_sum':None,
                   'project_T_each_iter':True, 'random_state':0, 'W_mat':Wm,
                   'compute_obj_each_iter':True, 't_row_sum':1.0,
                   'negative_denom_correction':False}
    nmf_params.update(rs_settings)

    soln = nmf(X, **nmf_params)
    oh = soln['obj_history']
    assert np.all(np.diff(oh) <= 0)