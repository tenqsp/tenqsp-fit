# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``method laplace``: parameters that vary between units, by the Laplace approximation of the
likelihood (FOCE-I).

**Normal scores.** A parameter that varies between units is a transform of a standard-normal
score per unit, ``ka_i = T(z_i)``, with ``T`` its distribution's quantile function (lognormal:
``median * exp(sdlog * z)``; exponential: ``-mean * log(1 - Phi(z))``; ...). The scores of
correlated parameters are correlated (the matrix ``R`` of ``correlate`` lines); for two
lognormals this is NONMEM's OMEGA block. Whatever the distribution, the scores are normal, so
the method below is the same for all of them.

**-2LL.** For unit i with measurements y_i, ``-2 log p(y_i)`` is approximated by Laplace's
method at the score that maximizes the unit's joint density (its conditional mode, NONMEM's
EBE):

    -2LL_i = L_i(z^) + z^' R^-1 z^ + log|R| + log|H_i|,
    H_i = J_i' W_i J_i + R^-1,

with ``L_i`` the unit's -2 log-likelihood (as ``tf.objective``, including n log(2 pi) and the
M3 terms), ``J_i`` the derivative of its predictions by z at z^, and ``W_i`` the Fisher
information of each prediction (measured: ``1/sd^2 + 2 (d log sd/df)^2``, so an error model
that depends on the prediction is FOCE with interaction; censored: the Gauss-Newton term of
``sqrt(-2 log Phi)``). With an additive error this is NONMEM's FOCE-I.

**The inner fit.** The conditional modes of all units at once, from the last modes found
(warm start): Newton steps with the Hessian from differences of the exact gradient by the
scores (quadratic convergence), or a Fisher-scoring step (H, always downhill) when that
Hessian is not positive definite or its step is much longer; with step halving.

**The outer fit.** BFGS (SciPy) on the estimation scales of ``method optimize`` (log, logit
within bounds; a correlation within (-1, 1)), as NONMEM minimizes the FOCE objective itself.
The gradient is exact up to central differences, by the implicit function theorem:
``dO/du = dF/du + dF/dz dz^/du`` with ``dz^/du = -(d2g/dz2)^-1 d2g/dz du``, every term a
difference of first derivatives at the fixed modes (no inner fit; 2p + 2q evaluations in one
batched call). TenQSP has no second derivatives of the solution yet; the first-order
"FOCE gradient" that holds the solver's sensitivities fixed stops short of the minimum
(theophylline: at -2LL 354.04, the Lindstrom-Bates point of nlme, against 353.98).

**Standard errors** from the Hessian of -2LL by central differences of that gradient
(NONMEM's covariance step, R matrix): covariance = 2 H^-1 on the estimation scale, carried to
the reported scale; the 95 % intervals are the estimation-scale intervals carried over (so a
positive quantity's interval stays positive). **Shrinkage**: 1 - SD(z^) per parameter, SD
with n - 1 (NONMEM's ETA shrinkage, SD-based).
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .likelihood import Objective
from .live import default_progress
from .optimize import (
    BAD_VALUE,
    DEFAULT_MAXITER,
    NOT_ASKED,
    _Progress,
    _Run,
    covariance,
    draw_starts,
    finish,
    locked_show,
    no_covariance,
    scales,
    to_u,
    to_x,
    user_bounds,
    watch,
)
from .problem import seconds

__all__ = ["Laplace"]

GTOL = 1e-3  # BFGS: the largest derivative of -2LL on the estimation scale at the end
FD_GRAD = 1e-4  # central differences for the gradient, on the estimation scale
INNER_TOL = 1e-8  # the conditional modes: the largest step of a score
INNER_MAXITER = 50
HALVINGS = 12
FD_HESS = 1e-4  # differences of the gradient for the inner Newton step
NOISE = 1e-9  # a step that raises 2g by less than this is round-off, accepted


class Laplace:
    """-2 log-likelihood of a problem whose parameters vary between units, by the Laplace
    approximation (FOCE-I): ``obj = tf.objective(problem)``, ``obj(x)`` with ``x`` in
    ``obj.names`` order (``obj.x0``: the starts). ``obj.modes(x)``: the units' conditional
    modes (normal scores, one column per varying parameter); ``obj.individual(x)``: each
    unit's parameter values. Its derivatives need second derivatives of the solution, which
    TenQSP does not give: ``jax.grad`` of it is not the gradient (the fit uses the implicit
    function theorem, see the module notes)."""

    def __init__(self, problem):
        cond = Objective(problem, _population=True)
        if cond.written:
            raise NotImplementedError(
                f"{problem.name}: a written likelihood with parameters that vary between units "
                "(method laplace) is not available yet; use an error model."
            )
        self.problem = problem
        self._cond = cond
        self.names = cond.names
        self.x0 = cond.x0
        self.bounds = cond.bounds
        self.kinds = cond.kinds
        self.units = cond.units
        self.n_observations = cond.n_observations
        self.n_censored = cond.n_censored
        self.floor = cond.floor
        self.varying = cond.varying
        self.n, self.q = len(cond.units), len(cond.varying)
        self.wrt = cond._wrt
        self.n_wrt = len(cond._wrt) if cond._wrt else 1
        position = {name: k for k, (name, _) in enumerate(cond.varying)}
        index = {n: i for i, n in enumerate(self.names)}
        self._pairs = [
            (position[a], position[b], index[f"corr({a},{b})"])
            for c in problem._correlations
            for i, a in enumerate(c.names)
            for b in c.names[i + 1 :]
        ]
        value = jax.jit(cond._tolerant(self._value))
        modes = jax.jit(cond._tolerant(self._modes))
        # the tolerances are an input (None: the problem's): see Objective._tolerant
        at = lambda tol: cond._tol0 if tol is None else tol  # noqa: E731
        self._jit_value = lambda x, Z0, tol=None: value(x, Z0, tol=at(tol))
        self._jit_modes = lambda x, Z0, tol=None: modes(x, Z0, tol=at(tol))

    # ------------------------------------------------------------------ pieces
    def _R(self, x):
        R = jnp.eye(self.q)
        for a, b, i in self._pairs:
            R = R.at[a, b].set(x[i]).at[b, a].set(x[i])
        return R

    def _preds(self, P):
        cond = self._cond
        out = cond._predictions(None, cond._run(P))
        return tuple(out[lk["lhs"]] for lk in cond._links)

    def _local(self, x, preds):
        """Per link: -2 log-density per measurement, its derivative by the prediction, and
        the Fisher information; and L (units,)."""
        links = self._cond._links
        parts = [self._cond._link_local(lk, x, f) for lk, f in zip(links, preds, strict=True)]
        L = sum(jnp.sum(t, axis=1) for t, _, _ in parts)
        return parts, L

    def _dP_dz(self, x, Z):
        """(q, units, model parameters): d P_i / d z_ik (each unit's parameters depend on its
        own scores only)."""
        E = jnp.eye(self.q)[:, None, :] * jnp.ones((1, self.n, 1))
        mp = self._cond._model_params
        return jax.vmap(lambda e: jax.jvp(lambda Z_: mp(x, Z_), (Z,), (e,))[1])(E)

    def _Jp(self, x, Z):
        """Predictions and their derivative by each unit's model parameters (the solver's
        sensitivities): per link (model parameters, units, measurements)."""
        P = self._cond._model_params(x, Z)
        T = jnp.eye(self.n_wrt)[:, None, :] * jnp.ones((1, self.n, 1))
        preds, lin = jax.linearize(self._preds, P)
        return preds, jax.vmap(lin)(T)

    def _H(self, parts, Jp, dPdZ, Rinv):
        """J per link (units, measurements, q); H (units, q, q); and the gradient of L by z."""
        H = jnp.broadcast_to(Rinv, (self.n, self.q, self.q))
        grad = jnp.zeros((self.n, self.q))
        for (_, dterm, info), jp in zip(parts, Jp, strict=True):
            J = jnp.einsum("wnm,knw->nmk", jp, dPdZ)
            H = H + jnp.einsum("nmk,nm,nml->nkl", J, info, J)
            grad = grad + jnp.einsum("nmk,nm->nk", J, dterm)
        return H, grad

    # ------------------------------------------------------------------ the inner fit
    def _modes(self, x, Z0):
        """The conditional modes at x, by Fisher scoring with step halving, from Z0."""
        x = jax.lax.stop_gradient(x)
        Rinv = jnp.linalg.inv(self._R(x))
        mp = self._cond._model_params

        def two_g(Z):
            _, L = self._local(x, self._preds(mp(x, Z)))
            return L + jnp.sum((Z @ Rinv) * Z, axis=1)

        g0 = two_g(Z0)
        Z0 = jnp.where(jnp.isfinite(g0)[:, None], Z0, 0.0)

        def gradient(Z):
            preds, Jp = self._Jp(x, Z)
            parts, L = self._local(x, preds)
            H, gradL = self._H(parts, Jp, self._dP_dz(x, Z), Rinv)
            return 0.5 * gradL + Z @ Rinv, H, L + jnp.sum((Z @ Rinv) * Z, axis=1)

        def newton(Z):
            """The step: Newton with the Hessian from differences of the exact gradient when
            it is positive definite and the step is not much longer than Fisher scoring's;
            Fisher scoring (always downhill) otherwise."""
            grad, H, g = gradient(Z)
            fisher = -jnp.linalg.solve(H, grad[..., None])[..., 0]
            cols = [(gradient(Z.at[:, k].add(FD_HESS))[0] - grad) / FD_HESS for k in range(self.q)]
            Hn = jnp.stack(cols, axis=2)
            Hn = 0.5 * (Hn + jnp.swapaxes(Hn, 1, 2))
            pd = jnp.all(jnp.isfinite(jnp.linalg.cholesky(Hn)), axis=(1, 2))
            step = -jnp.linalg.solve(jnp.where(pd[:, None, None], Hn, H), grad[..., None])[..., 0]
            size = jnp.linalg.norm(step, axis=1)
            use = (
                pd
                & jnp.all(jnp.isfinite(step), axis=1)
                & (size <= 2.0 * jnp.linalg.norm(fisher, axis=1) + 0.1)
            )
            return jnp.where(use[:, None], step, fisher), g

        def body(carry):
            Z, it, _ = carry
            delta, g = newton(Z)
            delta = jnp.where(jnp.isfinite(delta), delta, 0.0)

            def halve(c):
                alpha, accepted, Zn, k = c
                Zt = Z + alpha[:, None] * delta
                gt = two_g(Zt)
                ok = ~accepted & jnp.isfinite(gt) & (gt <= g + NOISE)
                Zn = jnp.where(ok[:, None], Zt, Zn)
                accepted = accepted | ok
                return jnp.where(accepted, alpha, 0.5 * alpha), accepted, Zn, k + 1

            alpha, accepted, Zn, _ = jax.lax.while_loop(
                lambda c: (c[3] < HALVINGS) & ~jnp.all(c[1]),
                halve,
                (jnp.ones(self.n), jnp.zeros(self.n, bool), Z, 0),
            )
            moved = jnp.max(jnp.abs(jnp.where(accepted, alpha, 0.0)[:, None] * delta))
            return Zn, it + 1, moved < INNER_TOL

        Z, _, _ = jax.lax.while_loop(
            lambda c: (c[1] < INNER_MAXITER) & ~c[2], body, (Z0, 0, jnp.array(False))
        )
        return Z

    # ------------------------------------------------------------------ the outer function
    def _FG(self, x, Z):
        """Per unit, at x and the scores Z: its term of -2LL, ``L + z'R^-1 z + log|H| +
        log|R|``, and the gradient of ``g = (L + z'R^-1 z)/2`` by its scores."""
        R = self._R(x)
        C = jnp.linalg.cholesky(R)  # NaN when R is not a correlation matrix
        logdetR = 2.0 * jnp.sum(jnp.log(jnp.diag(C)))
        Rinv = jnp.linalg.inv(R)
        preds, Jp = self._Jp(x, Z)
        parts, L = self._local(x, preds)
        H, gradL = self._H(parts, Jp, self._dP_dz(x, Z), Rinv)
        C_H = jnp.linalg.cholesky(H)
        logdetH = 2.0 * jnp.sum(jnp.log(jnp.diagonal(C_H, axis1=1, axis2=2)), axis=1)
        F = L + jnp.sum((Z @ Rinv) * Z, axis=1) + logdetH + logdetR
        return F, 0.5 * gradL + Z @ Rinv

    def _value(self, x, Z0):
        """-2LL at x, and the modes (from Z0)."""
        Z = self._modes(x, Z0)
        return jnp.sum(self._FG(x, Z)[0]), Z

    def _gradient(self, u, Z, to_x):
        """The gradient of -2LL by u (x = to_x(u)) at the modes Z, by the implicit function
        theorem: dO/du = dF/du + dF/dz dz^/du, with dz^/du = -(d2g/dz2)^-1 d2g/dzdu. Each
        derivative is a central difference of first derivatives at fixed scores (no inner
        fit), all in one batched call: 2 p + 2 q evaluations."""
        p, h = u.shape[0], FD_GRAD
        Ex = jnp.eye(p) * h
        Fx, Gx = jax.vmap(lambda w: self._FG(to_x(w), Z))(jnp.concatenate([u + Ex, u - Ex]))
        Ez = jnp.eye(self.q)[:, None, :] * jnp.ones((1, self.n, 1)) * h
        x = to_x(u)
        Fz, Gz = jax.vmap(lambda Zk: self._FG(x, Zk))(jnp.concatenate([Z + Ez, Z - Ez]))
        q = self.q
        dF_du = (jnp.sum(Fx[:p], axis=1) - jnp.sum(Fx[p:], axis=1)) / (2 * h)  # (p,)
        dG_du = (Gx[:p] - Gx[p:]) / (2 * h)  # (p, units, q)
        dF_dz = (Fz[:q] - Fz[q:]) / (2 * h)  # (q, units)
        Hz = jnp.transpose((Gz[:q] - Gz[q:]) / (2 * h), (1, 2, 0))  # (units, q, q)
        Hz = 0.5 * (Hz + jnp.swapaxes(Hz, 1, 2))
        dz_du = -jnp.linalg.solve(Hz, jnp.transpose(dG_du, (1, 2, 0)))  # (units, q, p)
        return dF_du + jnp.einsum("kn,nkp->p", dF_dz, dz_du)

    # ------------------------------------------------------------------ public
    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape != (len(self.names),):
            raise ValueError(f"x has {len(self.names)} entries ({list(self.names)}).")
        return self._jit_value(x, jnp.zeros((self.n, self.q)))[0]

    def modes(self, x, Z0=None) -> np.ndarray:
        """The units' conditional modes at x: (units, varying parameters), normal scores."""
        x = jnp.asarray(x, dtype=jnp.float64)
        Z0 = jnp.zeros((self.n, self.q)) if Z0 is None else jnp.asarray(Z0)
        return np.asarray(self._jit_modes(x, Z0))

    def individual(self, x, Z=None) -> pd.DataFrame:
        """Each unit's normal scores and values: the varying quantities and the model
        parameters the fit sets, at x and the modes Z."""
        x = jnp.asarray(x, dtype=jnp.float64)
        Z = self.modes(x) if Z is None else np.asarray(Z)
        env = self._cond._unit_values(x, jnp.asarray(Z))
        table = {"unit": list(self.units)}
        table.update({f"z({name})": Z[:, k] for k, (name, _) in enumerate(self.varying)})
        for n in [name for name, _ in self.varying] + [w for w in self.wrt]:
            if n not in table:
                table[n] = np.asarray(env[n])
        return pd.DataFrame(table)

    def status(self, x) -> dict:
        """``{unit: "ok" or why its run failed}``, at the typical unit (scores 0)."""
        return self._cond.status(x)

    def unpack(self, x) -> dict[str, float]:
        return self._cond.unpack(x)

    def __repr__(self) -> str:
        cens = f", {self.n_censored} censored" if self.n_censored else ""
        return (
            f"<tenqsp_fit Laplace objective (-2 log-likelihood, FOCE-I) of {list(self.names)}: "
            f"{self.n} units, {', '.join(n for n, _ in self.varying)} varying, "
            f"{self.n_observations} measurements{cens}>"
        )


# ----------------------------------------------------------------------------- the fit
def fit_laplace(problem, obj: Laplace, *, progress=None, with_se=True):
    opts = problem._options
    names, x0 = obj.names, np.asarray(obj.x0, float)
    sc = scales(names, x0, obj.bounds, obj.kinds)
    bounds_written = user_bounds(problem, names)
    n_starts = int(opts.get("starts", 1))
    rng = np.random.default_rng(int(opts.get("seed", 0)))
    maxiter = int(opts.get("maxiter") or DEFAULT_MAXITER)
    progress = progress or opts.get("progress") or default_progress()
    t_start = time.perf_counter()
    p = len(names)
    Z0 = np.zeros((obj.n, obj.q))
    cores = os.cpu_count() or 1
    pool = ThreadPoolExecutor(max_workers=max(1, cores))

    from .priors import written_penalty

    penalty = written_penalty(problem, names, x0, sc)
    pen = (lambda u: 0.0) if penalty is None else jax.jit(lambda u: penalty(to_x(u, sc)))
    pen_grad = (lambda u: 0.0) if penalty is None else jax.jit(jax.grad(pen))

    def value(u, Z):  # -2LL, plus -2 log of the written priors
        v, Zh = obj._jit_value(to_x(jnp.asarray(u), sc), jnp.asarray(Z))
        return float(v) + float(pen(jnp.asarray(u))), np.asarray(Zh)

    grad_ll = jax.jit(lambda u, Z: obj._gradient(u, Z, lambda w: to_x(w, sc)))

    def gradient(u, Z):
        return grad_ll(u, Z) + pen_grad(u)

    def runs(u):
        return jnp.asarray(np.isfinite(value(np.asarray(u), Z0)[0]))

    starts_pool = ThreadPoolExecutor(max_workers=max(1, min(n_starts, cores)))
    U_start, redrawn = draw_starts(problem, runs, x0, sc, rng, n_starts, starts_pool)
    n = len(U_start)

    m2ll = np.full(n, np.nan)
    X = np.array([np.asarray(to_x(jnp.asarray(u), sc)) for u in U_start])
    steps = np.zeros(n, int)
    done = np.zeros(n, bool)
    failed = np.zeros(n, int)
    messages = [""] * n
    trace: list[tuple] = []
    stop = threading.Event()
    lock = threading.Lock()  # a step's -2LL and values are shown together

    def run(j):
        warm = {"Z": Z0}

        def fun(u):
            if stop.is_set():
                raise _Stopped
            v, Zh = value(u, warm["Z"])
            if not np.isfinite(v):
                failed[j] += 1
                return BAD_VALUE, np.zeros(p)
            warm["Z"] = Zh
            if "first" not in warm:  # scipy starts at the start: the trace's step 0
                warm["first"] = True
                trace.append((j + 1, 0, v, *np.asarray(to_x(jnp.asarray(u), sc))))
            g = np.asarray(gradient(jnp.asarray(u), jnp.asarray(Zh)))
            if not np.all(np.isfinite(g)):
                failed[j] += 1
                return BAD_VALUE, np.zeros(p)
            return v, g

        def callback(intermediate_result):  # scipy passes the step by this name
            res = intermediate_result
            fj, xj = float(res.fun), np.asarray(to_x(jnp.asarray(res.x), sc))
            with lock:
                steps[j] += 1
                m2ll[j], X[j] = fj, xj
            trace.append((j + 1, int(steps[j]), fj, *xj))

        try:
            res = minimize(fun, U_start[j], jac=True, method="BFGS", callback=callback,
                           options={"gtol": GTOL, "maxiter": maxiter})  # fmt: skip
        except _Stopped:
            return
        xj = np.asarray(to_x(jnp.asarray(res.x), sc))
        with lock:
            X[j], m2ll[j] = xj, float(res.fun)
        done[j] = bool(res.success) or float(np.max(np.abs(res.jac))) < 10 * GTOL
        messages[j] = "" if done[j] else str(res.message)
        if not done[j] and res.nit < maxiter:  # stopped short (e.g. precision loss)
            done[j] = True
            messages[j] = f"ended: {res.message}"

    shown = _Progress(progress, names, n, "laplace", problem.name, trace)
    futures = [starts_pool.submit(run, j) for j in range(n)]
    show = locked_show(shown, lock, steps, m2ll, X, done, t_start)
    status, stopped_by = watch(futures, stop, seconds(opts.get("timeout")), t_start, show)
    starts_pool.shutdown()
    show(last=True, status=status)

    final = np.array([float(v) for v in pool.map(obj, list(X))])
    rank = final + np.array([float(pen(jnp.asarray(to_u(x, sc)))) for x in X])
    run_ = _Run(X=X, steps=steps, done=done, failed=failed, trace=trace, status=status,
                stopped_by=stopped_by)  # fmt: skip
    best = int(np.argmin(np.where(np.isfinite(rank), rank, np.inf)))
    Z = obj.modes(X[best])
    pool.shutdown()
    if status == "stopped" or not with_se:
        cov = no_covariance(p, None if with_se else NOT_ASKED)
    else:
        cov = covariance(
            None,
            to_u(X[best], sc),
            sc,
            gradient=lambda u: gradient(
                jnp.asarray(u), jnp.asarray(obj.modes(to_x(jnp.asarray(u), sc), Z))
            ),
        )
    shrink = {name: float(1.0 - np.std(Z[:, k], ddof=1)) if obj.n > 1 else float("nan")
              for k, (name, _) in enumerate(obj.varying)}  # fmt: skip
    return finish(problem, obj, "laplace", "BFGS, FOCE-I", names, x0, sc, U_start, run_, final,
                  redrawn, bounds_written, t_start, rank=rank, modes=Z, shrinkage=shrink,
                  objective=obj, notes=[m for m in messages if m],
                  with_priors=None if penalty is None else float(rank[best]),
                  **cov)  # fmt: skip


class _Stopped(Exception):
    pass
