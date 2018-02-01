import scipy as sp
import numpy as np
from rri_nmf_old.nmf import qf_min, _initialize_nmf

import pytest


def test_init():
    X = np.matrix('[1,0;    \
                    0.5,0.5;\
                    0.25,0.75]'
                       )
    W, T = _initialize_nmf(X, 2, init='nndsvd', random_state=0)

    Wt = np.fromstring(
        '\xb9X\x18pb\xbd\xe8?\x00\x00\x00\x00\x00\x00\x00\x00\x114#('
        'e\x8c\xe3?%\x86\x8c"D\x08\xcd?\xbd\xa1('
        '\x84\xe6\xf3\xe0?\xbc\xad\x84\xb3f\xec\xe4?'
        ).reshape(3, 2)
    Tt = np.fromstring(
        '\x04\x89=\x03\x95\xf6\xee?v)\xdfe\xf9\xf7\xe1?\x00\x00\x00\x00'
        '\x00\x00\x00\x00l\x8d.\xd8\x84%\xe6?'
        ).reshape(2, 2)
    assert np.allclose(Wt, W)
    assert np.allclose(Tt, T)




@pytest.mark.parametrize("s", [1,1.0,3,None,3.3])
def test_qf_min(s):
    """make sure that qf_min is within 1e-9 or better of scipy optimize"""
    np.random.seed(0)
    ff = lambda x, w, c: np.inner(x, w) + c * np.inner(x, x)
    for i in range(23):
        w = np.random.randn(7)
        x_sp = _min_opt(w, -1, s)
        x_sp = x_sp['x']
        x_me = qf_min(w, -1, s)
        assert ff(x_sp, w, -1) + 1e-9 >= ff(x_me, w,-1) , 'qf worse for w={} ' \
                                                        'ff(' \
                                                        'x_sp)={} ff(x_me)={' \
                                                        '}'.format(
            w, ff(x_sp, w, -1), ff(x_me, w, -1))

def _min_opt(w, c, s=1):
    """Unfortunately this isn't actually optimal, and QP in SLSQP doesnt
    stand for quadratic programming -- need to find an actual QP solver"""
    f = lambda x: np.inner(w, x) + c * np.sum(x ** 2)

    if s:
        constr = {'type': 'eq', 'fun': lambda x: np.sum(x) - s}
    else:
        constr = tuple()

    return sp.optimize.minimize(f,
                                x0=np.zeros(w.size),
                                method='SLSQP',
                                bounds=[(0, 1)] * w.size,
                                constraints=constr,
                                tol=1e-10)