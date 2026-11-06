# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Refits with one quantity held at a value: the points of a profile likelihood.

The fitted model is compiled once for all the points: the held value is an input of the
compiled functions, not a new problem (a new problem per point compiled the model again, about
two thirds of each point's time on theophylline). What is minimized is what the fit minimized
(-2LL, plus the written priors); what is reported is -2LL."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .optimize import BAD, BAD_VALUE, DEFAULT_MAXITER, TOL, to_x
from .priors import written_penalty


def held_fitter(result, j: int):
    """``fit(h, start, warm) -> (u, m2ll, status, warm)``: the quantities other than ``j``
    refitted from ``start`` (all of them, estimation scale) with quantity ``j`` held at ``h``
    (estimation scale). ``warm`` carries the modes of ``laplace`` from one point to the next."""
    if result.method == "laplace":
        return _laplace(result, j)
    if result.method == "optimize":
        return _optimize(result, j)
    raise NotImplementedError(f"profile: not available after method {result.method}.")


def _full(v, h, j):
    """All quantities from the others ``v`` and the held one ``h`` (estimation scale)."""
    return jnp.concatenate([v[:j], jnp.reshape(h, (1,)), v[j:]])


def _optimize(result, j):
    import optimistix as optx

    obj, sc, problem = result.objective, result.scales, result.problem
    maxiter = int(problem._options.get("maxiter") or DEFAULT_MAXITER)
    penalty = written_penalty(problem, result.names, np.asarray(result.x0, float), sc)
    if not obj.error_estimated and penalty is None:  # as the fit: least squares

        def fn(v, h):
            r = obj._residuals(to_x(_full(v, h, j), sc))
            return jnp.where(jnp.all(jnp.isfinite(r)), r, BAD)

        solver = optx.LevenbergMarquardt(rtol=TOL, atol=TOL)

        def solve(v0, h):
            return optx.least_squares(fn, solver, v0, args=h, max_steps=maxiter, throw=False)
    else:

        def fn(v, h):
            x = to_x(_full(v, h, j), sc)
            f = obj._minus2ll(x) + (0.0 if penalty is None else penalty(x))
            return jnp.where(jnp.isfinite(f), f, BAD_VALUE)

        solver = optx.BFGS(rtol=TOL, atol=TOL)

        def solve(v0, h):
            return optx.minimise(fn, solver, v0, args=h, max_steps=maxiter, throw=False)

    @jax.jit
    def run(v0, h):  # arrays only out of jit (see fit_optimize on memory maps)
        sol = solve(v0, h)
        return sol.value, sol.result == optx.RESULTS.successful

    def fit(h, start, warm=None):
        v, ok = run(jnp.asarray(np.delete(start, j)), jnp.asarray(h, float))
        u = np.asarray(_full(v, h, j))
        m2ll = float(obj(to_x(jnp.asarray(u), sc)))
        status = "converged" if bool(ok) else "max iterations"
        return u, (m2ll if np.isfinite(m2ll) else np.nan), status, warm

    return fit


def _laplace(result, j):
    from scipy.optimize import minimize

    from .laplace import GTOL

    lap, sc, problem = result.objective, result.scales, result.problem
    maxiter = int(problem._options.get("maxiter") or DEFAULT_MAXITER)
    penalty = written_penalty(problem, result.names, np.asarray(result.x0, float), sc)
    pen = (lambda u: 0.0) if penalty is None else jax.jit(lambda u: penalty(to_x(u, sc)))
    pen_grad = (lambda u: 0.0) if penalty is None else jax.jit(jax.grad(pen))
    grad_ll = jax.jit(lambda u, Z: lap._gradient(u, Z, lambda w: to_x(w, sc)))

    def fit(h, start, warm=None):
        Z = {"Z": np.asarray(result.modes if warm is None else warm)}

        def fun(v):
            u = jnp.asarray(np.insert(v, j, h))
            f, Zh = lap._jit_value(to_x(u, sc), jnp.asarray(Z["Z"]))
            f = float(f) + float(pen(u))
            if not np.isfinite(f):
                return BAD_VALUE, np.zeros(v.size)
            g = np.delete(np.asarray(grad_ll(u, Zh)) + np.asarray(pen_grad(u)), j)
            if not np.all(np.isfinite(g)):
                return BAD_VALUE, np.zeros(v.size)
            Z["Z"] = np.asarray(Zh)
            return f, g

        res = minimize(fun, np.delete(start, j), jac=True, method="BFGS",
                       options={"gtol": GTOL, "maxiter": maxiter})  # fmt: skip
        u = np.insert(res.x, j, h)
        m2ll = float(lap(to_x(jnp.asarray(u), sc)))
        done = bool(res.success) or float(np.max(np.abs(res.jac))) < 10 * GTOL
        status = "converged" if done else ("max iterations" if res.nit >= maxiter else "ended")
        return u, (m2ll if np.isfinite(m2ll) else np.nan), status, Z["Z"]

    return fit
