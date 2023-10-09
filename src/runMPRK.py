#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:47:59 2023

@author: robert
"""
import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt
import matplotlib.pyplot as plt

#### EQUATIONS
# dc1/dt = c2 - a c1
# dc2/dt = a c1 - c2

# c_11 = c_n1 / (1 - dt * (p1 / cn1 - d1 / cn1))
# c_nn1 = c_n1 / (1 - dt/2 * (p1 / c_11 - d1 / c_11))

def solve_mprk(fun, t_span, y0, dt, args):
    """
    Solve an initial value problem for a system of ODE.

    The method used to solve is a 2nd order modified patankar-runge-kutta.


    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here `t` is a scalar, and there are two options for the ndarray `y`:
        It can either have shape (n,); then `fun` must return a tuple of 2
        array_likes with shapes (n,). Alternatively, it can have shape (n, k);
        then `fun` must return an array_like with shape (n, k), i.e., each
        column corresponds to a single column in `y`. The choice between the
        two options is determined by `vectorized` argument (see below). The
        vectorized implementation allows a faster approximation of the Jacobian
        by finite differences (required for stiff solvers).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    dt : float, optional
        Time step.
    args : tuple, optional
        Additional arguments to pass to the user-defined functions.  If given,
        the additional arguments are passed to all user-defined functions.
        So if, for example, `fun` has the signature ``fun(t, y, a, b, c)``,
        then `jac` (if given) and any event functions must have the same
        signature, and `args` must be a tuple of length 3. The default is None.

    Returns
    -------
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.

    Examples
    --------
    >>> def fun(t, y, a):
    ...     "Production and destruction term for a simple linear model."
    ...     c1, c2 = y
    ...
    ...     p = [[0, c2],
    ...          [a*c1, 0]]
    ...
    ...     d = [[a*c1, 0],
    ...          [0, c2]]
    ...
    ...     return p, d

    >>> # Solve the problem:
    >>> t, y = solve_mprk(fun, t_span=[0, 2], y0=[0.9, 0.1], args=[5], dt=0.25)
    >>> print(y)
    [[0.9        0.34985219 0.21016238 0.17649204 0.16884278 0.16714614
      0.16677219 0.16668988]
     [0.1        0.65014781 0.78983762 0.82350796 0.83115722 0.83285386
      0.83322781 0.83331012]]
    """
    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters.  Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        def fun(t, y, fun=fun): return fun(t, y, *args)

    len_y0 = len(y0)
    t = np.arange(*t_span, step=dt)
    y = np.zeros([len_y0, len((t))])
    y[:, 0] = y0
    eye = np.identity(len_y0, dtype=bool)
    a = np.zeros_like(eye, dtype=float)
    r = np.zeros_like(a[:, 0], dtype=float)
    for ci, ti in enumerate(t[1:]):
        # Get the production and destruction term:
        p0, d0 = fun(ti, y[:, ci])
        p0 = np.asarray(p0)
        d0 = np.asarray(d0)

        # Calculate diagonal:
        a[eye] = dt * d0.sum(1) / y[:, ci] + 1

        # Calculate non-diagonal:
        c_rep = np.broadcast_to(y[:, ci], (len_y0, len_y0))
        a[~eye] = -dt * p0[~eye] / c_rep[~eye]

        # Something:
        r[:] = y[:, ci] + dt*p0[eye]

        # Solve system of equation:
        c0 = np.linalg.solve(a, r)

        # Run the algorithm a second time:
        # Get the production and destruction term:
        p, d = fun(ci, c0)
        p = np.asarray(p)
        d = np.asarray(d)

        # Calculate the mean value of the terms:
        p = 0.5 * (p0 + p)
        d = 0.5 * (d0 + d)

        # Calculate diagonal:
        a[eye] = dt * d.sum(1) / c0 + 1

        # Calculate non-diagonal:
        c_rep = np.broadcast_to(c0, (len_y0, len_y0))
        a[~eye] = -dt * p[~eye] / c_rep[~eye]

        # Something:
        r[:] = y[:, ci] + dt*p[eye]

        # Solve system of equation:
        y[:, ci+1] = np.linalg.solve(a, r)

    return t, y


def fun(t, y, a):
    "Production and destruction term for a simple linear model."
    c1, c2, c3 = y
    #p = [[0, c2],
    #     [a*c1, 0]]
    #d = [[a*c1, 0],
    #     [0, c2]]
    p = [[0, c2, c3/2],
         [a*c1, 0, c3/2],
         [a/2 * c1, a/2 * c2, 0]]
    d = [[a*c1, 0, a/2 * c1],
         [0, c2, a/2 * c2],
         [c3/2, c3/2, 0]]
    return p, d

# Solve the problem:
t, y = solve_mprk(fun, t_span=[0, 10], y0=[0.9, 0.1, 0.5], args=[5], dt=0.25)
print(y)

fig=plt.figure()
plt.plot(np.arange(*[0,10],step = 0.25), y[0], color = 'blue')
plt.plot(np.arange(*[0,10],step = 0.25), y[1], color = 'red')
plt.plot(np.arange(*[0,10],step = 0.25), y[2], color = 'green')
plt.show()