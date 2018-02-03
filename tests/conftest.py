import pytest
import numpy as np
import scipy as sp
import scipy.sparse
from rri_nmf.matrixops import normalize, tfidf

@pytest.fixture(scope='session')
def small_X_W_T():
    X = np.matrix('[1,0;    \
                        0.5,0.5;\
                        0.25,0.75]')
    Wt = np.fromstring(
            '\xb9X\x18pb\xbd\xe8?\x00\x00\x00\x00\x00\x00\x00\x00\x114#('
            'e\x8c\xe3?%\x86\x8c"D\x08\xcd?\xbd\xa1('
            '\x84\xe6\xf3\xe0?\xbc\xad\x84\xb3f\xec\xe4?').reshape(3, 2)
    Tt = np.fromstring(
            '\x04\x89=\x03\x95\xf6\xee?v)\xdfe\xf9\xf7\xe1?\x00\x00\x00\x00'
            '\x00\x00\x00\x00l\x8d.\xd8\x84%\xe6?').reshape(2, 2)
    return X, Wt, Tt

@pytest.fixture(scope='session')
def text_train():
    X = sp.sparse.load_npz('data/text_data_train.npz')
    return _tm_xform(X.toarray())

@pytest.fixture(scope='session')
def text_test():
    X = sp.sparse.load_npz('data/text_data_test.npz')
    return _tm_xform(X.toarray())

def _tm_xform(X):
    return normalize(tfidf(X))

@pytest.fixture(scope='session')
def recsys_train():
    X = sp.sparse.load_npz('data/recsys_data_train.npz')
    return X.toarray()

@pytest.fixture(scope='session')
def recsys_test():
    X = sp.sparse.load_npz('data/recsys_data_test.npz')
    return X.toarray()