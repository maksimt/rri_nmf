# rri_nmf
Non-negative Matrix Factorization using Rank-One Residue Iterations implemented in Python

[Ngoc Diep Ho's Thesis](https://perso.uclouvain.be/paul.vandooren/ThesisHo.pdf)

[CJ Lin's Paper](https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf)

[`sklearn` NMF](http://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf)


# TODOs
## RRI NMF Base algorithm

### NMF for Topic Modeling
1. Add a score method to the estimator that uses a bunch of scores.

### NMF for Recommender Systems
1. Initialize using elementwise weighted (masked) SVD. E.g. [BIRSVD](https://github.com/xr0038/birsvd/blob/master/svd_imputation_with_mask.py)
2. Cythonized implementation of the elementwise division in the gradient step loop. (Algorithm 10, WRRI in Ho's Thesis)