import pytest
import scipy as sp
import scipy.sparse

@pytest.fixture(scope='session')
def text_train():
    X = sp.sparse.load_npz('data/text_data_train.npz')
    return X.toarray()

@pytest.fixture(scope='session')
def text_test():
    X = sp.sparse.load_npz('data/text_data_test.npz')
    return X.toarray()

@pytest.fixture(scope='session')
def recsys_train():
    X = sp.sparse.load_npz('data/recsys_data_train.npz')
    return X.toarray()

@pytest.fixture(scope='session')
def recsys_test():
    X = sp.sparse.load_npz('data/recsys_data_test.npz')
    return X.toarray()