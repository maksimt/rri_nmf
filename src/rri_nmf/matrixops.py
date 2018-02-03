import numpy as np
import scipy as sp


def euclidean_proj_simplex(v_in, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

    Author
    ------
    Adrien Gaidon - INRIA - 2011
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s

    if sp.sparse.issparse(v_in):
        v = v_in.toarray()
    else:
        v = v_in

    n = np.prod(v_in.shape)
    v = v.reshape((n,))
    # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v

    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)

    return sp.sparse.csr_matrix(w.reshape(v_in.shape)) if sp.sparse.issparse(
            v_in) \
        else w.reshape(v_in.shape)


def proj_mat_to_simplex(W, s=1.0, axis=1):
    """project mat onto simplex, minimizing ell_2 recons err s.t. vectors along
    axis have ell_1 norm s

    Parameters
    ----------
    W: ndarray (n,d)
    s: float or ndarray (n,) or (d,), global or elementwise of simplex.
        default: 1.0
    axis: int, vectors along which to project. E.g. 1-> rows will be projected.

    Returns
    -------
    W: ndarray (n,d)

    """
    if axis == 1:
        if np.isscalar(s):
            for i in range(W.shape[0]):
                W[i, :] = euclidean_proj_simplex(W[i, :], s)
        else:
            assert s.size == W.shape[0], 'proj_mat_to_simplex: expected s to ' \
                                         'have size {n} but s has size {' \
                                         's}'.format(n=W.shape[0], s=s.size)
            for i in range(W.shape[0]):
                W[i, :] = euclidean_proj_simplex(W[i, :], s[i])
        return W
    elif axis == 1:
        return proj_mat_to_simplex(W.T, s, axis=1).T


def normalize_l2(X, dim=1):
    """Normalize X's dims to have l2 norm = 1

    :param X: nparray of shape (n,d)
    :type X: numpy array

    :param dim: dimension along which to normalize 1 (default): rows
                0: columns
    :type dim: int
    """

    if dim == 1:
        xs = (1 / np.sqrt(np.sum(X**2, dim) + 1e-10))
        return X * xs.reshape(xs.size, 1)
    elif dim == 0:
        # raise NotImplementedError("dim 0 not yet implemented")
        return normalize_l2(X.T, 1).T
    else:
        raise ValueError("dim must be 0 or 1")


def normalize(X, dim=1, zero_sum_fix=True):
    """Normalize X's dim to sum to 1.

    :param X: scipy matrix of shape (n,d)
    :type X: scipy matrix

    :param dim: dimension along which to normalize. 1 (default) corresponds to
    rows, 0 corresponds to columns
    :type dim: int

    :param zero_sum_fix: for rows/columns that sum to 0, replace all entries
        with 1/d or 1/n
    :type zero_sum_fix: bool
    """

    if dim == 1:
        xs = sp.sum(X, 1) + np.spacing(1)
        m = xs.size
        d = 1.0 / xs.reshape((m, 1))
        Xn = d * X
        if zero_sum_fix:
            I = np.nonzero(xs < 1e-10)[0]
            for i in I:
                Xn[i, :] = np.ones((1, Xn.shape[1])) * (1.0 / Xn.shape[1])

        return Xn

    elif dim == 0:
        xs = sp.sum(X, 0) + np.spacing(1)
        d = 1.0 / xs

        Xn = X * d
        if zero_sum_fix:
            I = np.nonzero(xs < 1e-10)[0]
            for i in I:
                Xn[:, i] = np.ones((Xn.shape[0], 1)) * (1.0 / Xn.shape[0])

        return Xn
    else:
        raise Exception('Unknown dim=' + dim)


def tfidf(X, return_idf=False):
    """X is n docs * d features. Transform it to TFIDF."""
    n, d = X.shape
    df = (X > 0).sum(0)
    idf = np.log(n / (df + np.spacing(1)))
    if not sp.sparse.issparse(X):
        rtvx = X * idf
    else:
        idf = sp.sparse.coo_matrix(idf)
        rtvx = X.multiply(idf)
    if return_idf:
        return rtvx, idf
    else:
        return rtvx


def labels_to_mat(y):
    """Transforms a (n,) sample label vector to a (n,k) array where each row
    is a probability distribution over labels"""
    if y.size == y.shape[0]:
        k = len(np.unique(y))
        W = np.zeros((y.size, k))
        y = y.astype(np.int)
        W[range(y.size), y] = 1
        return W
    else:
        if abs(y.sum() - y.shape[0]) < 1e-5:  # already normalized
            return y
        k = len(np.unique(y))
        if y.shape[1] == k:
            return normalize(y)  # Y is already correct shape, just normalize
        else:
            raise Exception(
                    'labels_to_mat: number of columns of y = {0} ' + 'doesnt '
                                                                     'match '
                                                                     'number '
                                                                     'of '
                                                                     'unique '
                                                                     'elements = {'
                                                                     '1}'.format(
                            y.shape[1], k))


def harden_distributions(W):
    """If each row of W represents a distribution over columns, set the
        column with highest probability to 1"""
    Wh = np.zeros_like(W)
    I = np.argmax(W, 1)
    Wh[range(W.shape[0]), I] = 1
    return Wh


def stack_matrices(L, dict_key=None, transform=None, dim='tall'):
    """
    Stack a list of matrices, either horizontally or vertically


    :param L: the list of (matrices) or (dicts of matrices) to stack.
    :param dict_key: if L is a list of dicts, then the key of which element of
                        each dict should be stacked. If L is a list of matrices
                        then None (default).
    :param transform: f:ndarray->ndarray to transform each matrix
                        before stacking (default is None)
    :param dim should the result be tall (default) or fat?
                'tall'=> vstack, 'fat'=> hstack
    :return: a single ndarray of the transformed matrices
    """

    assert isinstance(L[0], np.ndarray) or (isinstance(L[0],
                                                       dict) and dict_key), \
        'if L is a list no dick_key is needed, if L is a dict, dict_key' + \
        'must be the key of the matrices to stack.'

    R = None

    if dim == 'tall':
        stack_op = np.vstack
    elif dim == 'fat':
        stack_op = np.hstack
    else:
        assert dim in ['tall', 'fat'], 'dim must be "tall" or "fat".'

    for E in L:
        if dict_key:
            try:
                M = E[dict_key]
            except TypeError:
                M = E.__getattribute__(dict_key)
        else:
            M = E

        if not isinstance(M, np.ndarray):
            M = np.asarray(M)

        if transform:
            M = transform(M)

        if R is not None:
            R = stack_op((R, M))
        else:
            R = M

    return R