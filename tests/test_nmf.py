import scipy as sp
import numpy as np
from rri_nmf.initialization import initialize_nmf

import pytest


def test_init():
    X = np.matrix('[1,0;    \
                    0.5,0.5;\
                    0.25,0.75]')
    W, T = initialize_nmf(X, 2, init='nndsvd', random_state=0)

    Wt = np.fromstring(
        '\xb9X\x18pb\xbd\xe8?\x00\x00\x00\x00\x00\x00\x00\x00\x114#('
        'e\x8c\xe3?%\x86\x8c"D\x08\xcd?\xbd\xa1('
        '\x84\xe6\xf3\xe0?\xbc\xad\x84\xb3f\xec\xe4?').reshape(3, 2)
    Tt = np.fromstring(
        '\x04\x89=\x03\x95\xf6\xee?v)\xdfe\xf9\xf7\xe1?\x00\x00\x00\x00'
        '\x00\x00\x00\x00l\x8d.\xd8\x84%\xe6?').reshape(2, 2)
    assert np.allclose(Wt, W)
    assert np.allclose(Tt, T)
