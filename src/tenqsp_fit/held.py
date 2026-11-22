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

from .optimize import BAD, BAD_VALUE, DEFAULT_MAXITER, minimize_bounded, to_x
from .priors import written_penalty


def held_fitter(result, j: int):
    """``fit(h, start, warm) -> (u, m2ll, status, warm)``: the quantities other than ``j``
    refitted from ``start`` (all of them, estimation scale) with quantity ``j`` held at ``h``
    (estimation scale). ``m2ll`` is what the fit minimized: -2LL, plus the priors' terms when
    priors are written. ``warm`` carries the modes of ``laplace`` from one point to the next."""
    if result.method == "laplace":
        return _laplace(result, j)
    if result.method == "optimize":
        return _optimize(result, j)
    raise NotImplementedError(f"profile: not available after method {result.method}.")


def _full(v, h, j):
    """All quantities from the others ``v`` and the held one ``h`` (estimation scale)."""
    return jnp.concatenate([v[:j], jnp.reshape(h, (1,)), v[j:]])


def _optimize(result, j):
    obj, sc, problem = result.objective, result.scales, result.problem
    maxiter = int(problem._options.get("maxiter") or DEFAULT_MAXITER)
    penalty = written_penalty(problem, result.names, np.asarray(result.x0, float), sc)
    others = sc[:j] + sc[j + 1 :]
    least_squares = obj.sum_of_squares and penalty is None  # as the fit
    if least_squares:

        def res(v, h):
            return obj._residuals(to_x(_full(v, h, j), sc))

        both = jax.jit(lambda v, h: (res(v, h), jax.jacfwd(res)(v, h)))
    else:

        def total(v, h):
            x = to_x(_full(v, h, j), sc)
            return obj._minus2ll(x) + (0.0 if penalty is None else penalty(x))

        both = jax.jit(jax.value_and_grad(total))

    def fit(h, start, warm=None):
        hh = jnp.asarray(h, float)

        def fun(v):
            a, g = both(jnp.asarray(v), hh)
            a, g = np.asarray(a, float), np.asarray(g, float)
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(g)):  # no derivatives
                if least_squares:
                    return np.full_like(a, BAD), np.zeros_like(g)
                return BAD_VALUE, np.zeros_like(g)
            return (a, g) if least_squares else (float(a), g)

        r = minimize_bounded(fun, np.delete(start, j), others, maxiter,
                             least_squares=least_squares)  # fmt: skip
        u = np.asarray(_full(jnp.asarray(r["u"]), hh, j))
        x = to_x(jnp.asarray(u), sc)
        m2ll = float(obj(x)) + (0.0 if penalty is None else float(penalty(x)))
        status = "converged" if r["converged"] else "max iterations"
        return u, (m2ll if np.isfinite(m2ll) else np.nan), status, warm

    return fit


def _laplace(result, j):
    from .laplace import _Warm

    lap, sc, problem = result.objective, result.scales, result.problem
    maxiter = int(problem._options.get("maxiter") or DEFAULT_MAXITER)
    penalty = written_penalty(problem, result.names, np.asarray(result.x0, float), sc)
    pen = (lambda u: 0.0) if penalty is None else jax.jit(lambda u: penalty(to_x(u, sc)))
    pen_grad = (lambda u: 0.0) if penalty is None else jax.jit(jax.grad(pen))
    others = sc[:j] + sc[j + 1 :]

    def fit(h, start, warm=None):
        Z = _Warm(result.modes if warm is None else warm)

        def fun(v):
            u = np.insert(v, j, h)
            f, Zh, g, dz = lap.evaluate(u, Z.start(u))
            f = f + float(pen(jnp.asarray(u)))
            g = np.delete(g + np.asarray(pen_grad(jnp.asarray(u))), j)
            if not np.isfinite(f) or not np.all(np.isfinite(g)):
                return BAD_VALUE, np.zeros(v.size)
            Z.seen(u, Zh, dz)
            return f, g

        r = minimize_bounded(fun, np.delete(start, j), others, maxiter,
                             accepted=lambda v, f: Z.accept(np.insert(v, j, h)))  # fmt: skip
        u = np.insert(r["u"], j, h)
        m2ll = float(lap(to_x(jnp.asarray(u), sc))) + float(pen(jnp.asarray(u)))
        status = (
            "converged"
            if r["converged"]
            else ("max iterations" if r["message"] == "max iterations" else "ended")
        )
        return u, (m2ll if np.isfinite(m2ll) else np.nan), status, Z.Z

    return fit
