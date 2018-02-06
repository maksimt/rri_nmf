import numpy as np
from matrixops import euclidean_proj_simplex

eps_div_by_zero = np.spacing(10)


def qf_min(w, c, s=1.0, ub=1.0):
    """
    Minimize a simple quadratic vector function.

    :math:`min_{0<=x<=1,\, \sum x=s}\, f(x)\equiv w^Tx + 0.5x^Tdiag(c)x`

    Parameters
    ----------
    w : array_like
        The linear cost coefficients
    c : float
        Scalar coefficient for quadratic term
    s : float or None, optional
        Sum constraint for x, default is 1.0
    ub : float or None, optional
        Upper bound for each entry of x, 1.0 by default. If None, may raise a
        ValueError in case of unbounded objective

    Returns
    -------
    x : array_like
        The vector x, of same shape as w, that minimizes :math:`f(x)`.
    nx : float
        The 1-norm of vector x before it was scaled.

    Raises
    ------
    ValueError
        If the input w, c, s, ub has a solution with no lower bound.
    """
    if np.isscalar(c):
        if c > 0:
            x = np.maximum(-w, 0) / (c + eps_div_by_zero)
            nx = x.sum()
            # it's correct to project to simplex here since c>0 makes
            # this p.d. and hence convex
            if s is not None:
                x = euclidean_proj_simplex(x, s)
        elif c <= 0:
            x = np.zeros_like(w)
            if s is None:
                I = np.argwhere(w + c < 0)
                if ub:
                    x[I] = ub
                else:
                    _unbounded_objective(**locals())
            elif s == 1.0:
                i = np.argmin(w)
                x[i] = 1.0
            else:
                raise NotImplementedError('s={} is not yet '
                                          'implemented'.format(s))
            nx = 1.0
    elif np.shape(w) == np.shape(c):
        I = np.argwhere(c > 0).ravel()
        x = np.zeros_like(w)
        x[I] = np.maximum(-w[I], 0) / (c[I] + eps_div_by_zero)
        nx = x.sum()
        if s is not None:
            x = s * x / x.sum()  # dont project to the simplex here,
            # since that's wrong with a multi-dimensional c
        J = np.argwhere(c < 0).ravel()
        if J.size > 0:
            if ub:
                J = np.argwhere(w * ub + c * ub**2 < 0).ravel()
                x[J] = ub
            else:
                _unbounded_objective(**locals())

                # Ho Thesis Alg10 Line18 (pg 119) says to leave everything
                # else 0
                # TODO: this is incorrect in the presence of regularization
                # parameters

    return x, nx


def _unbounded_objective(w, c, s, ub):
    raise ValueError('Minimum objective is unbounded. w={w}, c={c}, s={s}, '
                     'ub={ub}'.format(locals()))


def universal_stopping_condition(obj_history, eps_stop=1e-4):
    """ Check if last change in objective is <= eps_stop * first change"""
    if len(obj_history) < 2:
        return False  # dont stop

    d1 = abs(obj_history[0] - obj_history[1])
    de = abs(obj_history[-1] - obj_history[-2])
    return de <= eps_stop * d1


def first_last_stopping_condition(obj_history, eps_stop=1e-4):
    if len(obj_history) < 2:
        return False
    return obj_history[-1] <= obj_history[0] * eps_stop
