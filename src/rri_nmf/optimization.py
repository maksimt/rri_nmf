import numpy as np
from matrixops import euclidean_proj_simplex


def qf_min(w, c, s=1.0, w_correction=False):
    """min_{0<=x<=1} <w,x> + 0.5*c<x,x> s.t. sum(x)=s

    Parameters
    ----------
    w: ndarray of linear coefficients
    c: scalar coefficient for quadratic term
    s: float or None sum constraint for x

    """
    if np.isscalar(c):
        if c > 0:
            x = -w / c
            # it's correct to project to simplex here since c>0 makes
            # this p.d. and hence convex
            return euclidean_proj_simplex(x, s)
        if c <= 0:
            i = np.argmin(w)
            x = np.zeros_like(w)
            x[i] = s

    return x.reshape(w.shape)

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