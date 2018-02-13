import numpy as np
from matrixops import euclidean_proj_simplex
from scipy.optimize import minimize, linprog

eps_div_by_zero = np.spacing(10)
constraint_violation_tolerance = 1e-13  # constraint violation tolerance

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def qf_min(w, c, s=1.0, ub=1.0, x0=None):
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
    d = w.size
    if s:
        if ub:
            ub = min(ub, s)
            assert d*ub >= s, 'Impossible to satisfy sum and upper bound ' \
                              'constraints.'
        else:
            ub = s  # since x>=0

    if np.isscalar(c):
        logger.log(logging.DEBUG-2, 'scalar c')
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
        if np.any(c<0) and (s is None and ub is None):
            _unbounded_objective(**locals())
        I = np.argwhere(c > 0).ravel()
        x = np.zeros_like(w)
        x[I] = np.maximum(-w[I], 0) / (c[I] + eps_div_by_zero)

        if ub is not None:
            x = np.minimum(x, ub)  # dont ignore ub
        nx = x.sum()
        if s is not None:
            x = s * x / x.sum()  # dont project to the simplex here,
            # since that's wrong with a multi-dimensional c
    return x, nx

def _cons_violation(x, s, ub):
    if s is None and ub is None:
        return 0
    if ub is None:
        return np.abs(np.sum(x)-s)
    I0 = np.argwhere(x<0).ravel()
    Iub = np.argwhere(x>ub).ravel()
    cv = 0
    for i in I0:
        cv += np.abs(x[i])
    for i in Iub:
        cv += np.abs(x[i]-ub)
    cv += np.abs(np.sum(x)-s)
    return cv

def _unbounded_objective(w, c, s, ub, **kwargs):
    raise ValueError('Minimum objective is unbounded. w={w}, c={c}, s={s}, '
                     'ub={ub}'.format(w=w, c=c, s=s, ub=ub))


def kkt_qf_min(w, d, s=1.0, ub=1.0):
    d = d * 2  # optimization is derived for w'x + 0.5x'diag(d)x#

    U = np.arange(w.size)
    J = np.array([np.argmin(w + d * ub**2)])
    order_added = [J[0]]
    I = np.setdiff1d(U, J)
    #  print 'J={}'.format(J)
    x, l = _solve_x_lambda(J, w[J], d[J], s, ub)
    # print 'x={} lambda={}'.format(x, l)
    I_not_df_in_I = np.argwhere(w[I] < -l).ravel()
    I_not_df = I[I_not_df_in_I]
    while I_not_df.size > 0:
        xo = np.zeros_like(w)
        xo[J] = x
        # _check_kkt_conditions(xo, l, w, d, s, ub)

        #  print 'Adding {} (w={}) to J'.format(I_not_df, w[I_not_df])
        for i in I_not_df:
            order_added.append(i)
        J = np.concatenate((J, I_not_df))
        I = np.setdiff1d(U, J)
        # print 'J={}'.format(J)
        try:
            x, l = _solve_x_lambda(J, w[J], d[J], s, ub)
        except TypeError:  # there is no feasible solution with
            # current all x in current J > 0
            j = order_added.pop(0)  # remove least recently added j
            J = J[J != j]
            # print '\n{} removed from J\n updating x,l'.format(j)
            I = np.setdiff1d(U, J)
            x, l = _solve_x_lambda(J, w[J], d[J], s, ub)
            # print 'x={} lambda={}'.format(x, l)
        I_not_df_in_I = np.argwhere(w[I] < -l).ravel()
        I_not_df = I[I_not_df_in_I]
    xo = np.zeros_like(w)
    xo[J] = x

    assert _check_kkt_conditions(xo, l, w, d, s, ub)

    return xo


def _solve_x_lambda(J, w, d, s, ub):
    n = J.size
    A_eq = np.zeros((n + 1, n + 1))
    A_eq[range(n), range(n)] = d
    A_eq[range(n), -1] = 1
    A_eq[-1, range(n)] = 1
    b_eq = np.zeros((n + 1, 1))
    b_eq[range(n), :] = -w[:, np.newaxis]
    b_eq[-1] = s
    # print '{:=^40}'.format('')
    # print '\n{:-^30}'.format('A_eq, b_eq')
    # print '{}\n{}'.format(A_eq, b_eq)
    x_uncons = np.linalg.solve(A_eq, b_eq)
    # print 'x={}'.format(x_uncons)

    A_ub = np.zeros((2 * n, n + 1))
    A_ub[range(n), range(n)] = -1
    A_ub[np.arange(n) + n, range(n)] = 1
    b_ub = np.zeros((2 * n, 1))
    b_ub[n:, :] = ub
    # print '\n{:-^30}'.format('A_ub, b_ub')
    # print '{}\n{}'.format(A_ub, b_ub)

    violation_eq = np.linalg.norm((np.dot(A_eq, x_uncons) - b_eq))
    violation_ub = np.min(b_ub - np.dot(A_ub, x_uncons))
    # print '||A_eq*x-b_eq||_2={} min(b_ub-A_ub*x)={}'.format(violation_eq,
    # violation_ub)

    if violation_eq > constraint_violation_tolerance or violation_ub < -constraint_violation_tolerance:
        # c = np.concatenate((w+d,[0]))
        c = np.zeros((n + 1,))
        rtv = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                                     b_eq=b_eq, method='interior-point')
        # print '\nrtv=',rtv
        if not rtv['success']:
            # print 'raising TypeError'
            raise TypeError('not feasible')
        x = rtv['x']
    else:
        x = x_uncons.ravel()
    x, _lambda = x[:-1], x[-1]

    return x, _lambda


def _check_kkt_conditions(x, l, w, d, s, ub):
    #print '\n{:*^30}'.format('')
    #print 'Primal Feasibility:'
    rtv = True
    # print '0<=x<=ub: {}'.format(
    #     np.all(0 - constraint_violation_tolerance <= x) and np.all(
    #         x <= ub + constraint_violation_tolerance))
    # print 'sum(x)==s: {}\n'.format(
    #     np.abs(s - sum(x)) < constraint_violation_tolerance)

    rtv *= np.all(0 - constraint_violation_tolerance <= x) and np.all(
            x <= ub + constraint_violation_tolerance)
    rtv *= np.abs(s - sum(x)) < constraint_violation_tolerance

    #print 'Complementnary Slackness:'
    u = w + d * x + l
    I = np.argwhere(np.abs(x * u - 0) > constraint_violation_tolerance).ravel()
    if len(I) > 0:
        return False


    #print 'Stationarity:'
    if np.all(np.abs(-w - d * x - (-u + l)) <= constraint_violation_tolerance):
        pass
    else:
        return False

    # dual feasibility
    if np.all(u >= 0 - constraint_violation_tolerance):
        pass
    else:
        return False
    return rtv

def optimize_scipy(w, c, s, ub, x0=None):
    B = [(0.0, ub)] * c.size

    def f(x):
        return np.sum(x * w) + np.sum(c * x**2)

    constraints = tuple()  # [{'type':'ineq', 'fun':lambda x: x[i]-ub} for i in
    # range(d)]
    if s:
        constraints = [{
            'type': 'ineq', 'fun': lambda x: sum(x) - s,
            'jac' : (lambda x: np.ones(x.shape))
        }, {
            'type': 'ineq', 'fun': lambda x: s - sum(x),
            'jac' : (lambda x: np.ones(x.shape))
        }]

    if not x0:
        x0 = np.zeros_like(w)
        I = np.argwhere(c > 0).ravel()
        x0[I] = np.maximum(-w[I], 0) / (c[I] + eps_div_by_zero)
        if s is not None:
            if x0.sum() > 0 + eps_div_by_zero:
                x0 = s * x0 / x0.sum()
            else:
                # x0 = np.ones_like(w)*s/d
                i = np.argmin(w + c)
                x0[i] = ub

    x0 = minimize(f, x0, bounds=B, jac=lambda x: w + c * x,
                  hess=lambda x: np.diag(c), method='SLSQP',
                  constraints=constraints, options={'maxiter': 10})
    cv0 = _cons_violation(x0['x'], s, ub)
    x1 = minimize(f, x0['x'], bounds=B, jac=lambda x: w + c * x,
                  hess=lambda x: np.diag(c), method='COBYLA',
                  constraints=constraints, options={'maxiter': 10})
    cv1 = _cons_violation(x1['x'], s, ub)
    if cv1 <= constraint_violation_tolerance and cv0 <= constraint_violation_tolerance:
        if x0['fun'] < x1['fun']:
            x = x0['x']
        else:
            x = x1['x']
    elif cv1 <= constraint_violation_tolerance:
        x = x1['x']
    elif cv0 <= constraint_violation_tolerance:
        x = x0['x']
    else:
        raise ValueError('Both solvers violated constraints by more than '
                         '{}. SLSQP={} COBYLA={}'.format(constraint_violation_tolerance, cv0, cv1))
    nx = np.sum(x)


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
