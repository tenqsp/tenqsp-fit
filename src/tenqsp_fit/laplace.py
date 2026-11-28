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

**The inner fit.** The conditional modes of all units at once, by Fisher scoring (the step
``-H^-1 grad g``, H as above: always downhill), with step halving per unit: each iteration is
one run of the units, which gives the trial point's value, its gradient by the scores and H
together. It starts from the modes the implicit function theorem predicts from the last point
(``z^ + dz^/du du``, below), so near the optimum a few iterations suffice.

**The outer fit.** L-BFGS-B (SciPy) on the estimation scales of ``method optimize``, bounds
kept as bounds (ADR 0006), as NONMEM minimizes the FOCE objective itself.
The gradient follows the implicit function theorem:
``dO/du = dF/du + dF/dz dz^/du`` with ``dz^/du = -(d2g/dz2)^-1 d2g/dz du``. It needs the
predictions' first and second derivatives by each unit's model parameters ``P_i``: one run of
the units carries both (forward mode over the solver's forward sensitivities), so each unit's
predictions are known as a quadratic in ``P_i``; ``F``, ``g`` and the derivatives above are
automatic derivatives of that quadratic (no further run), and the gradient is exact. (Second
derivatives by differences of the sensitivities, each from a run with its own steps, are too
noisy at the usual tolerances: errors of 1e-3 in the gradient, BFGS's own tolerance.) When
the solver gives no second derivatives (through events, or on TenQSP's sparse tier), or the
model has many parameters set by the fit (``w + 1 > 2p + 2q``, w model parameters, p
estimates, q varying), each term is instead a central difference of the objective or of its
derivative by the scores, at fixed modes (2p + 2q evaluations). The first-order "FOCE
gradient" that holds the solver's sensitivities fixed stops short of the minimum
(theophylline: at -2LL 354.04, the Lindstrom-Bates point of nlme, against 353.98).

**One compiled function** gives -2LL, the modes and (when asked) the gradient and ``dz^/du``,
with the solver's tolerances as an input: the fit, the standard errors, ``obj(x)``,
``obj.modes`` and the fit checks all call it.

**Standard errors** from the Hessian of -2LL by central differences of that gradient, each
from the modes predicted at its point (NONMEM's covariance step, R matrix): covariance =
2 H^-1 on the estimation scale, carried to the reported scale; the 95 % intervals are the
estimation-scale intervals carried over (so a positive quantity's interval stays positive).
**Shrinkage**: 1 - SD(z^) per parameter, SD with n - 1 (NONMEM's ETA shrinkage, SD-based).
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

from .likelihood import Objective
from .live import default_progress
from .optimize import (
    BAD_VALUE,
    DEFAULT_MAXITER,
    FAILED_KEPT,
    NOT_ASKED,
    _Progress,
    _Run,
    covariance,
    draw_starts,
    finish,
    fit_scales,
    locked_show,
    minimize_bounded,
    near_bounds,
    no_covariance,
    to_u,
    to_x,
    user_bounds,
    watch,
)
from .problem import seconds

__all__ = ["Laplace"]

FD_GRAD = 1e-4  # central differences for the gradient, on the estimation scale
INNER_TOL = 1e-8  # the conditional modes: the largest step of a score
INNER_MAXITER = 100  # Fisher-scoring iterations, one run of the units each
HALVINGS = 12
NOISE = 1e-9  # a step that raises 2g by less than this is round-off, accepted
# A unit's conditional density can have more than one local mode (a lag time with a sampling
# time on either side): where no starting modes are given, and at the start and the end of a
# fit, each unit's inner fit runs from zero, from the modes in hand and from SEARCH_STARTS
# draws of N(0, SEARCH_SD^2) scores (fixed: the value at x is a function of x), and keeps the
# lowest 2g. A fit whose end the search lowers by more than POLISH_GAIN goes on from there.
SEARCH_STARTS = 10
SEARCH_SD = 2.0
SEARCH_MAXITER = 10  # enough to tell the modes apart; the inner fit then ends from the best
POLISH_GAIN = 1e-2
POLISH_ROUNDS = 3


class Laplace:
    """-2 log-likelihood of a problem whose parameters vary between units, by the Laplace
    approximation (FOCE-I): ``obj = tf.objective(problem)``, ``obj(x)`` with ``x`` in
    ``obj.names`` order (``obj.x0``: the starts). ``obj.modes(x)``: the units' conditional
    modes (normal scores, one column per varying parameter); ``obj.individual(x)``: each
    unit's parameter values. The fit's gradient comes from the implicit function theorem (see
    the module notes), not from ``jax.grad`` of this function."""

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
        self.scales = fit_scales(self.names, np.asarray(self.x0, float), self.bounds, self.kinds)
        p = len(self.names)
        # the gradient through the units' quadratic predictions (one run with w + w^2
        # derivatives), unless differences in the estimates take fewer runs; the solver may
        # also refuse second derivatives (through events, or on its sparse tier): see _core
        self._quadratic = bool(cond._wrt) and self.n_wrt + 1 <= 2 * p + 2 * self.q
        self._compiled = core = jax.jit(cond._tolerant(self._core))  # (tests count compiles)
        # the tolerances are an input (None: the problem's): see Objective._tolerant
        self._core_jit = lambda u, Z0, want, tol: core(
            u, Z0, want, tol=cond._tol0 if tol is None else tol
        )
        search = jax.jit(cond._tolerant(self._search))
        self._search_jit = lambda u, Zs, tol: search(u, Zs, tol=cond._tol0 if tol is None else tol)
        draws = np.random.default_rng(0).standard_normal((SEARCH_STARTS, self.n, self.q))
        self._draws = SEARCH_SD * draws

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
    def _modes(self, x, Z0, maxiter=INNER_MAXITER):
        """The conditional modes at x from Z0, by Fisher scoring with step halving per unit.
        Returns the modes, ``2g`` and H at them, and the number of iterations."""
        x = jax.lax.stop_gradient(x)
        Rinv = jnp.linalg.inv(self._R(x))

        def at(Z):  # one run: 2g, its gradient by the scores (halved: of g), H
            preds, Jp = self._Jp(x, Z)
            parts, L = self._local(x, preds)
            H, gradL = self._H(parts, Jp, self._dP_dz(x, Z), Rinv)
            return L + jnp.sum((Z @ Rinv) * Z, axis=1), 0.5 * gradL + Z @ Rinv, H

        def step(G, H):
            """Fisher scoring's step; where it is not a number (H singular or not finite),
            a step down the gradient of length at most 1 (a zero step would end the unit's
            fit where it stands)."""
            d = -jnp.linalg.solve(H, G[..., None])[..., 0]
            size = jnp.maximum(1.0, jnp.linalg.norm(G, axis=1, keepdims=True))
            down = jnp.where(jnp.isfinite(G), -G / size, 0.0)
            return jnp.where(jnp.all(jnp.isfinite(d), axis=1, keepdims=True), d, down)

        first = at(Z0)
        bad = ~jnp.isfinite(first[0])  # a unit whose run fails from Z0 starts from 0
        Z0 = jnp.where(bad[:, None], 0.0, Z0)
        g0, G0, H0 = jax.lax.cond(jnp.any(bad), lambda: at(Z0), lambda: first)

        def body(c):
            Z, g, G, H, alpha, delta, done, it = c
            Zt = Z + jnp.where(done, 0.0, alpha)[:, None] * delta
            gt, Gt, Ht = at(Zt)
            ok = ~done & jnp.isfinite(gt) & (gt <= g + NOISE)
            Z = jnp.where(ok[:, None], Zt, Z)
            g = jnp.where(ok, gt, g)
            G = jnp.where(ok[:, None], Gt, G)
            H = jnp.where(ok[:, None, None], Ht, H)
            moved = jnp.max(jnp.abs(alpha[:, None] * delta), axis=1)
            done = done | (ok & (moved < INNER_TOL)) | (~ok & (alpha < 0.5**HALVINGS))
            delta = jnp.where(ok[:, None], step(G, H), delta)
            alpha = jnp.where(ok, 1.0, 0.5 * alpha)
            return Z, g, G, H, alpha, delta, done, it + 1

        d0 = step(G0, H0)
        c0 = (Z0, g0, G0, H0, jnp.ones(self.n), d0, ~jnp.isfinite(g0), 0)
        Z, g, _, H, _, _, _, it = jax.lax.while_loop(
            lambda c: (c[7] < maxiter) & ~jnp.all(c[6]), body, c0
        )
        return Z, g, H, it

    def _search(self, u, Zs):
        """Per unit, where the inner fit (at most SEARCH_MAXITER iterations) gets with the
        lowest 2g over the starts Zs (starts, units, scores), and that 2g; the starts run
        together."""
        x = to_x(u, self.scales)
        Zk, gk, _, _ = jax.vmap(lambda Z0: self._modes(x, Z0, SEARCH_MAXITER))(Zs)
        gk = jnp.where(jnp.isfinite(gk), gk, jnp.inf)
        k = jnp.argmin(gk, axis=0)
        return jnp.take_along_axis(Zk, k[None, :, None], axis=0)[0], jnp.min(gk, axis=0)

    def search(self, u, Z=None, *, tol=None) -> np.ndarray:
        """The conditional modes at u (the estimation scale) by a search: each unit's inner
        fit from zero, from Z (None: zero) and from fixed random scores; per unit the lowest
        2g. Where a unit's conditional density has more than one local mode, one start can
        find a poor one."""
        Z = np.zeros((self.n, self.q)) if Z is None else np.asarray(Z, float)
        Zs = np.concatenate([np.zeros((1, self.n, self.q)), Z[None], self._draws])
        u = jnp.asarray(u, dtype=jnp.float64)
        return np.asarray(self._search_jit(u, jnp.asarray(Zs), tol)[0])

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

    def _core(self, u, Z0, want):
        """-2LL at u (estimation scale) with the modes from Z0; when ``want``, also its
        gradient by u and the modes' derivative by u (else zeros): (-2LL, Z, gradient, dZ/du,
        inner iterations)."""
        x = to_x(u, self.scales)
        Z, g, H, it = self._modes(x, Z0)
        logdetR = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(self._R(x)))))
        C_H = jnp.linalg.cholesky(H)
        logdetH = 2.0 * jnp.sum(jnp.log(jnp.diagonal(C_H, axis1=1, axis2=2)), axis=1)
        value = jnp.sum(g + logdetH) + self.n * logdetR
        p = u.shape[0]
        none = (jnp.zeros(p), jnp.zeros((self.n, self.q, p)))
        if self._quadratic:
            try:
                grad, dz = jax.lax.cond(want, lambda: self._gradient_quadratic(u, Z), lambda: none)
                return value, Z, grad, dz, it
            except (ValueError, NotImplementedError):  # no second derivatives from the solver
                self._quadratic = False
        grad, dz = jax.lax.cond(want, lambda: self._gradient(u, Z), lambda: none)
        return value, Z, grad, dz, it

    def _quadratic_of(self, x, Z):
        """Each unit's predictions as a quadratic in its model parameters around their values
        at (x, Z): one run carrying the sensitivities and their derivatives (forward mode over
        forward mode), exact. Returns ``f(P)`` (a tuple per link, (units, measurements))."""
        w, n = self.n_wrt, self.n
        P0 = self._cond._model_params(x, Z)  # (units, w)
        T = jnp.eye(w)[:, None, :] * jnp.ones((1, n, 1))  # each unit's own parameter a

        def sens(P):
            preds, lin = jax.linearize(self._preds, P)
            return preds, jax.vmap(lin)(T)

        (preds, sens0), (_, K) = jax.vmap(lambda t: jax.jvp(sens, (P0,), (t,)))(T)
        f0 = [f[0] for f in preds]  # (units, measurements)
        J0 = [s_[0] for s_ in sens0]  # (a, units, measurements)
        K = [0.5 * (k + jnp.swapaxes(k, 0, 1)) for k in K]  # (a, b, units, measurements)

        def f(P):
            d = P - P0
            return tuple(
                a + jnp.einsum("anm,na->nm", b, d) + 0.5 * jnp.einsum("abnm,na,nb->nm", c, d, d)
                for a, b, c in zip(f0, J0, K, strict=True)
            )

        return f

    def _gradient_quadratic(self, u, Z):
        """The gradient by the implicit function theorem (module notes) at the modes Z, every
        derivative automatic through the units' quadratic predictions: one run."""
        sc, mp = self.scales, self._cond._model_params
        fq = self._quadratic_of(to_x(u, sc), Z)
        Ez = jnp.eye(self.q)[:, None, :] * jnp.ones((1, self.n, 1))

        def FG(u_, Z_):
            x_ = to_x(u_, sc)
            R = self._R(x_)
            logdetR = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(R))))
            Rinv = jnp.linalg.inv(R)
            preds, lin = jax.linearize(lambda Zz: fq(mp(x_, Zz)), Z_)
            Jz = jax.vmap(lin)(Ez)  # per link (q, units, measurements)
            parts, L = self._local(x_, preds)
            H = jnp.broadcast_to(Rinv, (self.n, self.q, self.q))
            gradL = jnp.zeros((self.n, self.q))
            for (_, dterm, info), J in zip(parts, Jz, strict=True):
                H = H + jnp.einsum("knm,nm,lnm->nkl", J, info, J)
                gradL = gradL + jnp.einsum("knm,nm->nk", J, dterm)
            C_H = jnp.linalg.cholesky(H)
            logdetH = 2.0 * jnp.sum(jnp.log(jnp.diagonal(C_H, axis1=1, axis2=2)), axis=1)
            F = L + jnp.sum((Z_ @ Rinv) * Z_, axis=1) + logdetH + logdetR
            return F, 0.5 * gradL + Z_ @ Rinv

        dF_du = jax.grad(lambda v: jnp.sum(FG(v, Z)[0]))(u)
        dF_dz = jax.grad(lambda Zz: jnp.sum(FG(u, Zz)[0]))(Z)  # (units, q)
        dG_du = jax.jacfwd(lambda v: FG(v, Z)[1])(u)  # (units, q, p)
        Hz = jax.vmap(lambda e: jax.jvp(lambda Zz: FG(u, Zz)[1], (Z,), (e,))[1])(Ez)
        Hz = jnp.transpose(Hz, (1, 2, 0))  # (units, q, q)
        Hz = 0.5 * (Hz + jnp.swapaxes(Hz, 1, 2))
        dz_du = -jnp.linalg.solve(Hz, dG_du)
        return dF_du + jnp.einsum("nk,nkp->p", dF_dz, dz_du), dz_du

    def _gradient(self, u, Z):
        """The gradient by the implicit function theorem at the modes Z, each derivative a
        central difference of first derivatives at fixed scores (no inner fit), all in one
        batched call: 2 p + 2 q evaluations. For models with many parameters set by the fit
        (module notes)."""
        sc = self.scales
        p, h = u.shape[0], FD_GRAD
        Ex = jnp.eye(p) * h
        Fx, Gx = jax.vmap(lambda w: self._FG(to_x(w, sc), Z))(jnp.concatenate([u + Ex, u - Ex]))
        Ez = jnp.eye(self.q)[:, None, :] * jnp.ones((1, self.n, 1)) * h
        x = to_x(u, sc)
        Fz, Gz = jax.vmap(lambda Zk: self._FG(x, Zk))(jnp.concatenate([Z + Ez, Z - Ez]))
        q = self.q
        dF_du = (jnp.sum(Fx[:p], axis=1) - jnp.sum(Fx[p:], axis=1)) / (2 * h)  # (p,)
        dG_du = (Gx[:p] - Gx[p:]) / (2 * h)  # (p, units, q)
        dF_dz = (Fz[:q] - Fz[q:]) / (2 * h)  # (q, units)
        Hz = jnp.transpose((Gz[:q] - Gz[q:]) / (2 * h), (1, 2, 0))  # (units, q, q)
        Hz = 0.5 * (Hz + jnp.swapaxes(Hz, 1, 2))
        dz_du = -jnp.linalg.solve(Hz, jnp.transpose(dG_du, (1, 2, 0)))  # (units, q, p)
        return dF_du + jnp.einsum("kn,nkp->p", dF_dz, dz_du), dz_du

    def evaluate(self, u, Z0=None, *, gradient: bool = True, tol=None):
        """-2LL at ``u`` (the estimation scale, ``self.scales``) with the modes from ``Z0``
        (None: from the modes ``search`` finds), and with ``gradient`` its gradient by u and
        the modes' derivative by u: ``(value, Z, grad, dZ/du)`` as NumPy (grad and dZ/du None
        without ``gradient``). ``tol``: the solver's tolerances (``Objective.tighter``; None:
        the problem's)."""
        u = jnp.asarray(u, dtype=jnp.float64)
        Z0 = self.search(u, tol=tol) if Z0 is None else Z0
        Z0 = jnp.asarray(Z0, dtype=jnp.float64)
        v, Z, g, dz, _ = self._core_jit(u, Z0, jnp.asarray(bool(gradient)), tol)
        if not gradient:
            return float(v), np.asarray(Z), None, None
        return float(v), np.asarray(Z), np.asarray(g), np.asarray(dz)

    def _u(self, x) -> np.ndarray:
        try:
            return to_u(np.asarray(x, float), self.scales)
        except ValueError:  # outside a quantity's range (a log-scale value <= 0): NaN
            return np.full(len(self.names), np.nan)

    # ------------------------------------------------------------------ public
    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape != (len(self.names),):
            raise ValueError(f"x has {len(self.names)} entries ({list(self.names)}).")
        return self.evaluate(self._u(x), gradient=False)[0]

    def modes(self, x, Z0=None) -> np.ndarray:
        """The units' conditional modes at x: (units, varying parameters), normal scores."""
        return self.evaluate(self._u(x), Z0, gradient=False)[1]

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
WARM_MOVE = 1.0  # the predicted modes move at most this far (in any score) from the base


class _Warm:
    """Where each inner fit starts: the modes at the last point the optimizer accepted, moved
    as the implicit function theorem predicts (``Z + dZ/du du``), each unit's move capped at
    ``WARM_MOVE``. The trial points of a line search are remembered, but one becomes the base
    only when the optimizer accepts it: a bad trial (far out, with modes far out) must not be
    where the next fits start."""

    KEPT = 64

    def __init__(self, Z0):
        self.u, self.Z, self.dz = None, np.asarray(Z0, float), None
        self.tried: dict[bytes, tuple] = {}

    def start(self, u) -> np.ndarray:
        if self.u is None:
            return self.Z
        dZ = np.einsum("nkp,p->nk", self.dz, np.asarray(u, float) - self.u)
        big = np.max(np.abs(dZ), axis=1, keepdims=True)
        dZ = np.where(big > WARM_MOVE, dZ * WARM_MOVE / np.where(big > 0, big, 1.0), dZ)
        return self.Z + np.where(np.isfinite(dZ), dZ, 0.0)

    def seen(self, u, Z, dz) -> None:
        u = np.asarray(u, float)
        if len(self.tried) >= self.KEPT:
            self.tried.pop(next(iter(self.tried)))
        self.tried[u.tobytes()] = (u, Z, dz)
        if self.u is None:  # the start: the first base
            self.u, self.Z, self.dz = u, Z, dz

    def accept(self, u) -> None:
        got = self.tried.get(np.asarray(u, float).tobytes())
        if got is not None:
            self.u, self.Z, self.dz = got


def fit_laplace(problem, obj: Laplace, *, progress=None, with_se=True):
    opts = problem._options
    names, x0 = obj.names, np.asarray(obj.x0, float)
    sc = obj.scales
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
        v, Zh, _, _ = obj.evaluate(u, Z, gradient=False)
        return v + float(pen(jnp.asarray(u))), Zh

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
    evaluations = np.zeros(n, int)
    failed_at: list[np.ndarray] = []
    messages = [""] * n
    ends: list = [None] * n  # a start's modes at its end, found by the search there
    trace: list[tuple] = []
    stop = threading.Event()
    lock = threading.Lock()  # a step's -2LL and values are shown together

    def run(j):
        warm = _Warm(obj.search(U_start[j]))
        first = []

        def fun(u):
            if stop.is_set():
                raise _Stopped
            v, Zh, g, dz = obj.evaluate(u, warm.start(u))
            evaluations[j] += 1
            v += float(pen(jnp.asarray(u)))
            g = g + np.asarray(pen_grad(jnp.asarray(u)))
            if not np.isfinite(v) or not np.all(np.isfinite(g)):
                failed[j] += 1
                if len(failed_at) < FAILED_KEPT:
                    failed_at.append(np.asarray(to_x(jnp.asarray(u), sc)))
                return BAD_VALUE, np.zeros(p)
            warm.seen(u, Zh, dz)
            if not first:  # scipy starts at the start: the trace's step 0
                first.append(True)
                trace.append((j + 1, 0, v, *np.asarray(to_x(jnp.asarray(u), sc))))
            return v, g

        def accepted(u, f):
            warm.accept(u)
            xj = np.asarray(to_x(jnp.asarray(u), sc))
            with lock:
                steps[j] += 1
                m2ll[j], X[j] = f, xj
            trace.append((j + 1, int(steps[j]), f, *xj))

        try:
            r = minimize_bounded(fun, U_start[j], sc, maxiter, accepted=accepted, stop=stop)
            for _ in range(POLISH_ROUNDS):  # a unit in a poor local mode: go on from better
                u_end = np.asarray(r["u"], float)
                v, Zs = value(u_end, obj.search(u_end, warm.Z))
                if not v < r["f"] - POLISH_GAIN:
                    ends[j] = Zs
                    break
                warm = _Warm(Zs)
                r = minimize_bounded(fun, u_end, sc, maxiter, accepted=accepted, stop=stop)
        except _Stopped:
            return
        xj = np.asarray(to_x(jnp.asarray(r["u"]), sc))
        with lock:
            X[j], m2ll[j] = xj, r["f"]
        done[j] = r["converged"]
        messages[j] = r["message"] if r["message"] not in ("", "stopped") else ""

    shown = _Progress(progress, names, n, "laplace", problem.name, trace)
    futures = [starts_pool.submit(run, j) for j in range(n)]
    show = locked_show(shown, lock, steps, m2ll, X, done, t_start)
    status, stopped_by = watch(futures, stop, seconds(opts.get("timeout")), t_start, show)
    starts_pool.shutdown()
    show(last=True, status=status)

    def last(j):  # -2LL at a start's end: from the modes searched there, else by a search
        u = to_u(X[j], sc)
        return float(obj.evaluate(u, ends[j], gradient=False)[0])

    final = np.array(list(pool.map(last, range(n))))
    rank = final + np.array([float(pen(jnp.asarray(to_u(x, sc)))) for x in X])
    run_ = _Run(X=X, steps=steps, done=done, failed=failed, failed_at=failed_at,
                evaluations=evaluations, trace=trace, status=status,
                stopped_by=stopped_by)  # fmt: skip
    best = int(np.argmin(np.where(np.isfinite(rank), rank, np.inf)))
    u_best = to_u(X[best], sc)
    _, Z, _, dz = obj.evaluate(u_best, ends[best])
    pool.shutdown()
    if status == "stopped" or not with_se:
        cov = no_covariance(p, None if with_se else NOT_ASKED)
    else:

        def gradient(u):  # from the modes predicted at u
            Zs = Z + np.einsum("nkp,p->nk", dz, np.asarray(u) - u_best)
            return obj.evaluate(u, Zs)[2] + np.asarray(pen_grad(jnp.asarray(u)))

        cov = covariance(
            None,
            u_best,
            sc,
            gradient=gradient,
            held=[names.index(n) for n in near_bounds(names, X[best], bounds_written)],
            names=names,
        )
    shrink = {name: float(1.0 - np.std(Z[:, k], ddof=1)) if obj.n > 1 else float("nan")
              for k, (name, _) in enumerate(obj.varying)}  # fmt: skip
    return finish(problem, obj, "laplace", "L-BFGS-B, FOCE-I", names, x0, sc, U_start, run_, final,
                  redrawn, bounds_written, t_start, rank=rank, modes=Z, shrinkage=shrink,
                  objective=obj, notes=[m for m in messages if m],
                  with_priors=None if penalty is None else float(rank[best]),
                  **cov)  # fmt: skip


class _Stopped(Exception):
    pass
