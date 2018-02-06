import sklearn
from sklearn.utils.validation import (
    check_X_y, check_is_fitted, check_array, check_non_negative
)
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
import inspect
import numpy as np
import scipy.sparse as sp
from nmf import nmf
from matrixops import tfidf, normalize


class NMF_RS_Estimator(sklearn.base.BaseEstimator):
    def __init__(self, n, d, k, wr1=0, wr2=0, tr1=0, tr2=0, random_state=0,
                 W=np.array([]), T=np.array([]), max_iter=30, nmf_kwargs={},
                 use_validation_early_stopping=True):

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

        self.use_validation_early_stopping = use_validation_early_stopping

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
        X, y = check_X_y(X, y)

        max_iter = self.max_iter

        self.min_rating = np.min(y)
        self.max_rating = np.max(y)

        if self.use_validation_early_stopping:

            UItr, UIval, Rtr, Rval = train_test_split(X, y, test_size=0.05,
                                                      random_state=0,
                                                      stratify=None)

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
                return np.sqrt(np.mean((Xpred[I, J] - Xv[I, J])**2))

            early_stop = RMSE_val

        else:
            early_stop = True
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
                   W_mat=W_mat_tr, W_in=W_in, T_in=T_in,
                   reg_w_l1=self.wr1, reg_w_l2=self.wr2, reg_t_l1=self.tr1,
                   reg_t_l2=self.tr2, random_state=self.random_state,
                   compute_obj_each_iter=True, early_stop=False,
                   **self.nmf_kwargs)
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
        X = np.hstack((
        NZ[0].reshape((NZ[0].size, 1)), NZ[1].reshape((NZ[1].size, 1))))
        y = Xtr.data
        return self.fit(X, y)

    def transform(self, Xnew):
        """express Xnew in terms of topics self.T"""
        W_mat_tr = np.zeros(Xnew.shape)
        [Itr, Jtr] = Xnew.nonzero()
        W_mat_tr[Itr, Jtr] = 1

        soln = nmf(Xnew, self.k, max_iter=4, max_time=7200,
                   project_W_each_iter=False, project_T_each_iter=True,
                   W_mat=W_mat_tr, T_in=self.T, fix_T=True, reg_w_l1=self.wr1,
                   reg_w_l2=self.wr2, reg_t_l1=self.tr1, reg_t_l2=self.tr2,
                   early_stop=True,
                   random_state=self.random_state, **self.nmf_kwargs)
        return soln['W']

    def make_Xpred(self):
        if self.Xpred.size == 0:
            self.Xpred = np.dot(self.W, self.T)
            self.Xpred = np.clip(self.Xpred, a_min=self.min_rating,
                                 a_max=self.max_rating)

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
            return np.sqrt(np.mean((y - yh)**2))
        else:  # X is a n*d matrix
            I, J = X.nonzero()
            return np.sqrt(np.mean((X[I, J] - self.Xpred[I, J])**2))


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
            X, idf = tfidf(X, return_idf=True)
            self.idf = idf
        if self.handle_normalization:
            X = normalize(X)

        soln = nmf(X, self.k, max_iter=self.max_iter, max_time=7200,
                   project_W_each_iter=True, w_row_sum=1,
                   project_T_each_iter=True, W_in=W_in, T_in=T_in,
                   reg_w_l1=self.wr1, reg_w_l2=self.wr2, reg_t_l1=self.tr1,
                   reg_t_l2=self.tr2, negative_denom_correction=True,
                   random_state=self.random_state, **self.nmf_kwargs)

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
            X, idf = tfidf(X, return_idf=True)
            self.idf = idf
        if self.handle_normalization:
            X = normalize(X)

        soln = nmf(X, self.k, max_iter=1, max_time=240,
                   project_W_each_iter=True, w_row_sum=1,
                   project_T_each_iter=True, W_in=W_in, T_in=T_in,
                   reg_w_l1=self.wr1, reg_w_l2=self.wr2, reg_t_l1=self.tr1,
                   reg_t_l2=self.tr2, random_state=self.random_state,
                   negative_denom_correction=True, **self.nmf_kwargs)

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
            Xnew = normalize(Xnew)

        soln = nmf(Xnew, self.k, max_iter=4, max_time=7200,
                   project_W_each_iter=True, w_row_sum=1, T_in=self.T,
                   fix_T=True, reg_w_l1=self.wr1, reg_w_l2=self.wr2,
                   reg_t_l1=self.tr1, reg_t_l2=self.tr2,
                   negative_denom_correction=True,
                   random_state=self.random_state)
        return soln['W']

    def constrained_transform(self, X):
        return self.transform(X)

    def score(self, X, y=None):
        """Return R^2 of new X """

        SST = ((X - np.mean(X, axis=0))**2).sum()
        W = self.transform(X)
        SSE = ((X - np.dot(W, self.T))**2).sum()
        return 1 - SSE / SST