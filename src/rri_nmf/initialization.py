from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, squared_norm
from math import sqrt
import numpy as np

from matrixops import normalize, tfidf


def initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None,
                   row_normalize=False, n_words_beam=20):
    """Algorithms for NMF initialization.
    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH


    Authors for nndsvd; directly from sklearn.decomposition.nmf:
    ----------------
    Vlad Niculae
    Lars Buitinck
    Mathieu Blondel <mathieu@mblondel.org>
    Tom Dupre la Tour
    Chih-Jen Lin, National Taiwan University


    Parameters
    ----------

    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    n_components : integer
        The number of components desired in the approximation.
    init :  None | 'random' | 'smart_random' | 'nndsvd' | 'nndsvda' |
    'nndsvdar' | 'coherence_pmi'
        Method used to initialize the procedure.
        Default: 'nndsvdar' if n_components < n_features, otherwise 'random'.
        Valid options:
        - 'coherence_pmi': beam search to maximize pointwise mutual information
            of m words in each topic
        - 'random': completely
        - 'smart_random': non-negative random matrices, scaled with:
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

    n_words_beam: int
        number of words to use to beam search for in coherence_pmi init
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
        rng = check_random_state(random_state)
        T = rng.rand(n_components, n_features)
        W = rng.rand(n_samples, n_components)
        # W = normalize(W)
        if row_normalize:
            T = normalize(T)
        return W, T

    # Slightly smarter random initialization
    if init == 'smart_random':
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features)
        W = avg * rng.randn(n_samples, n_components)
        # we do not write np.abs(H, out=H) to stay compatible with
        # numpy 1.5 and earlier where the 'out' keyword is not
        # supported as a kwarg on ufuncs
        np.abs(H, H)
        np.abs(W, W)
        if row_normalize:
            H = normalize(H)
        return W, H

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
        x_p_nrm, y_p_nrm = _norm(x_p), _norm(y_p)
        x_n_nrm, y_n_nrm = _norm(x_n), _norm(y_n)

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
                'Invalid init parameter: got %r instead of one of %r' % (
                    init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    if row_normalize:
        # W = normalize(W)
        H = normalize(H)

    return W, H


def init_coherence_beam_search(X, n_components, n_words_beam=20):
    """Initialize the topics using beam search to maximize coherence defined
    by pointwise mutual information"""
    X = normalize(tfidf(X))
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

    T = normalize(T)
    W = normalize(np.maximum(np.dot(X, T.T), 0))
    return W, T


def _norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    """
    return sqrt(squared_norm(x))