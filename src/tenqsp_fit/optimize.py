# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``method optimize``: one set of parameters for all the data, by multi-start optimization
(Optimistix) with TenQSP's exact derivatives.

- **The optimizer.** With the error model fixed, -2LL is a sum of squares up to a constant of
  the data: ``(DV - f)/sd`` for each measured value, ``sqrt(-2 log Phi((limit - f)/sd))`` for
  each censored one; Levenberg-Marquardt minimizes it, with exact Jacobians. With an error
  model's quantities estimated, -2LL is minimized by BFGS with its exact gradient
  (Levenberg-Marquardt converges slowly there: the sd's terms do not go to zero at the
  minimum; theophylline, pooled: 48-61 steps against 18 with sd fixed). Results report -2LL.
- **Scales.** Each quantity is estimated on a scale where any value is allowed: within its
  bounds ``(lo, hi)`` by ``lo + (hi - lo)*sigmoid(u)`` (one bound: ``lo + exp(u)``), else on the
  log scale when its start is above 0, else as it is. The summary lists the scales.
- **Starts.** Start 1 is the values written; the others are drawn around them on those
  scales (log scale: a factor within about 5 either way, 95 %; bounded: 1 on the logit scale;
  as is: half the value, at least 0.5) with the ``seed``. A start whose run fails or gives a
  non-finite residual is drawn again (PRD F4.10). Each start runs in its own thread, so the
  starts run on separate cores (B3, 8 starts: 7x faster than one ``jax.vmap`` call, which on
  the CPU runs them one after the other).
- **Failed runs** during the fit are rejected steps: every residual becomes a large constant,
  so the step is refused and the step size shrinks. They are counted.
- **Progress.** A block of text rewritten in place in a terminal, a line every 10 s
  elsewhere; ``progress none`` for quiet. The record is kept (``result.trace``), and Ctrl-C
  stops the fit and returns what it has, marked "stopped".
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd

__all__ = ["Result"]

BAD = 1e3  # each residual of a failed run (a rejected step), least squares
BAD_VALUE = 1e12  # -2LL of a failed run, BFGS
SPREAD_LOG = math.log(5) / 2  # starts on the log scale: 95 % within a factor 5
MAX_REDRAWS = 20
DEFAULT_MAXITER = 200
TOL = 1e-8  # Levenberg-Marquardt's rtol and atol (changes of the parameters and of -2LL)


# ----------------------------------------------------------------------------- scales
def scales(names, x0, bounds) -> list[tuple]:
    """The scale of each quantity: ("logit", lo, hi), ("above", lo), ("below", hi), ("log",)
    or ("as is",)."""
    out = []
    for x, (lo, hi) in zip(x0, bounds, strict=True):
        if np.isfinite(lo) and np.isfinite(hi):
            out.append(("logit", lo, hi))
        elif np.isfinite(lo):
            out.append(("above", lo))
        elif np.isfinite(hi):
            out.append(("below", hi))
        elif x > 0:
            out.append(("log",))
        else:
            out.append(("as is",))
    return out


def to_x(u, sc):
    parts = []
    for i, s in enumerate(sc):
        v = u[i]
        if s[0] == "logit":
            v = s[1] + (s[2] - s[1]) * jax.nn.sigmoid(v)
        elif s[0] == "above":
            v = s[1] + jnp.exp(v)
        elif s[0] == "below":
            v = s[1] - jnp.exp(v)
        elif s[0] == "log":
            v = jnp.exp(v)
        parts.append(v)
    return jnp.stack(parts)


def to_u(x, sc) -> np.ndarray:
    out = []
    for v, s in zip(x, sc, strict=True):
        if s[0] == "logit":
            p = (v - s[1]) / (s[2] - s[1])
            out.append(math.log(p / (1 - p)))
        elif s[0] == "above":
            out.append(math.log(v - s[1]))
        elif s[0] == "below":
            out.append(math.log(s[1] - v))
        elif s[0] == "log":
            out.append(math.log(v))
        else:
            out.append(float(v))
    return np.array(out)


def scale_text(names, sc) -> str:
    groups: dict[str, list[str]] = {}
    for n, s in zip(names, sc, strict=True):
        key = {"logit": "within bounds", "above": "above a bound", "below": "below a bound"}.get(
            s[0], s[0] + (" scale" if s[0] == "log" else "")
        )
        groups.setdefault(key, []).append(n)
    return "; ".join(f"{k}: {' '.join(v)}" for k, v in groups.items())


def _spread(sc, x0) -> np.ndarray:
    return np.array(
        [
            SPREAD_LOG
            if s[0] in ("log", "above", "below")
            else 1.0
            if s[0] == "logit"
            else max(0.5 * abs(v), 0.5)
            for s, v in zip(sc, x0, strict=True)
        ]
    )


# ----------------------------------------------------------------------------- bounds
NEAR_BOUND = 0.01  # an estimate within 1 % of its interval's width from a bound


def near_bounds(names, x, bounds) -> dict[str, tuple[str, float]]:
    """Estimates that ended near a bound written for them: ``{name: ("lower" or "upper",
    bound)}``."""
    out = {}
    for n, v, (lo, hi) in zip(names, x, bounds, strict=True):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if v - lo <= NEAR_BOUND * (hi - lo):
            out[n] = ("lower", float(lo))
        elif hi - v <= NEAR_BOUND * (hi - lo):
            out[n] = ("upper", float(hi))
    return out


def near_bound_text(result) -> list[str]:
    return [
        f"{n} ended at {result.estimates[n]:.6g}, near its {side} bound {b:.6g}: the bound "
        f"may be too tight, or the data do not determine {n}. Widen the bound, or fix {n}."
        for n, (side, b) in result.near_bounds.items()
    ]


class NearBoundWarning(UserWarning):
    """An estimate ended near a bound written for it (as NONMEM's 'near its boundary')."""


def warn_near_bounds(result) -> None:
    for text in near_bound_text(result):
        warnings.warn(f"{result.problem.name}: {text}", NearBoundWarning, stacklevel=3)


# ----------------------------------------------------------------------------- standard errors
FD_STEP = 1e-3  # central differences for the standard errors, on the estimation scale


def covariance(value, u, sc, *, gradient=None) -> dict:
    """Standard errors from the Hessian of -2LL on the estimation scale, by central
    differences of the values (or of a ``gradient``, when there is one; NONMEM's covariance
    step, R matrix): covariance = 2 H^-1, carried to the reported scale. Returns
    ``se``, ``ci`` (95 %, carried from the estimation scale), ``cov_u`` and
    ``covariance_status``."""
    u = np.asarray(u, float)
    p, h = len(u), FD_STEP
    E = np.eye(p) * h
    workers = ThreadPoolExecutor(max_workers=max(1, min(4 * p, os.cpu_count() or 1)))
    if gradient is not None:
        G = list(workers.map(lambda d: np.asarray(gradient(u + d)), [*E, *(-E)]))
        H = np.array([(G[i] - G[p + i]) / (2 * h) for i in range(p)])
    else:
        points = [u]
        for i in range(p):
            points += [u + E[i], u - E[i]]
            for j in range(i + 1, p):
                points += [u + E[i] + E[j], u + E[i] - E[j], u - E[i] + E[j], u - E[i] - E[j]]
        v = list(workers.map(lambda w: float(value(w)), points))
        f0, H, k = v[0], np.zeros((p, p)), 1
        for i in range(p):
            H[i, i] = (v[k] - 2 * f0 + v[k + 1]) / h**2
            k += 2
            for j in range(i + 1, p):
                H[i, j] = (v[k] - v[k + 1] - v[k + 2] + v[k + 3]) / (4 * h * h)
                k += 4
        H = np.triu(H) + np.triu(H, 1).T
    workers.shutdown()
    H = 0.5 * (H + H.T)
    status = "ok"
    try:
        np.linalg.cholesky(H)
        cov_u = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_u = np.full((p, p), np.nan)
        status = (
            "failed: the Hessian of -2LL is not positive definite at the estimates (some "
            "quantities are not determined by the data, or the fit has not converged)"
        )
    se_u = np.sqrt(np.diag(cov_u))
    du = np.asarray(jax.jacfwd(lambda w: to_x(w, sc))(jnp.asarray(u)))
    D = np.diag(du)
    se = np.abs(D) * se_u
    lo = np.asarray(to_x(jnp.asarray(u - 1.96 * se_u), sc))
    hi = np.asarray(to_x(jnp.asarray(u + 1.96 * se_u), sc))
    cov_x = (D[:, None] * cov_u) * D[None, :]
    return dict(
        se=se,
        ci=np.stack([np.minimum(lo, hi), np.maximum(lo, hi)], axis=1),
        cov_u=cov_u,
        cov=cov_x,
        covariance_status=status,
        hessian_u=H,
    )


# ----------------------------------------------------------------------------- result
class Result:
    """A fit: ``result.estimates`` ({name: value}), ``result.m2ll`` (-2 log-likelihood at
    them), ``result.status`` ("converged", "max iterations", "stopped" or "failed"),
    ``result.table()`` (estimates, standard errors, RSE %, 95 % intervals), ``result.se``,
    ``result.cov`` (covariance of the estimates), ``result.starts`` (each start's values,
    its end, -2LL, steps, status), ``result.trace`` (every step of every start),
    ``result.plot_waterfall()``. With parameters that vary between units (method laplace),
    also ``result.individual()`` (each unit's scores and values) and ``result.shrinkage``.
    ``print`` shows the table."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def estimates(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.x), strict=True))

    def plot_waterfall(self, ax=None):
        """The final -2LL of every start, sorted, above the best (the shaded band: within 0.1,
        one optimum): a flat step at the bottom shows how many starts reached the best fit; a
        staircase warns of local optima."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError(
                "plots need matplotlib: pip install matplotlib (or tenqsp-fit[plot])."
            ) from None
        ok = self.starts[np.isfinite(self.starts["-2LL"])]
        v = np.sort(ok["-2LL"].to_numpy())
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3.2))
        d = v - v[0]  # above the best; differences below 0.1 are one optimum
        ax.plot(np.arange(1, len(v) + 1), d, "o-", color="#1f77b4")
        ax.axhspan(-0.05, 0.1, color="#2ca02c", alpha=0.12, lw=0)
        ax.set_ylim(-0.05 * max(d[-1], 1.0), 1.05 * max(d[-1], 1.0))
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_xlabel("start, sorted by its final -2LL")
        ax.set_ylabel(f"-2LL above the best ({v[0]:.4g})")
        lost = len(self.starts) - len(v)
        failed = f" ({lost} failed)" if lost else ""
        ax.set_title(
            f"{self.problem.name}: {self.reached} of {len(self.starts)} starts at the best "
            f"fit{failed}"
        )
        ax.figure.tight_layout()
        return ax

    def table(self) -> pd.DataFrame:
        """Estimates with standard errors, RSE % (100 SE / |estimate|) and 95 % intervals
        (from the estimation scale, so a positive quantity's interval stays positive); for a
        distribution's spread, the other usual forms alongside (a lognormal's sdlog, CV and
        variance, NONMEM's OMEGA)."""
        se = getattr(self, "se", np.full(len(self.names), np.nan))
        ci = getattr(self, "ci", np.full((len(self.names), 2), np.nan))
        with np.errstate(divide="ignore", invalid="ignore"):
            rse = 100.0 * se / np.abs(self.x)
        return pd.DataFrame(
            {
                "quantity": list(self.names),
                "estimate": self.x,
                "SE": se,
                "RSE %": rse,
                "CI 2.5 %": ci[:, 0],
                "CI 97.5 %": ci[:, 1],
                "also": [self._also(n) for n in self.names],
            }
        )

    def _also(self, name: str) -> str:
        from .distributions import Fixed, spreads

        varying = getattr(getattr(self, "objective", None), "varying", ())
        for vname, dist in varying:
            prefix = f"{vname}."
            if not name.startswith(prefix):
                continue
            key = name[len(prefix) :]
            args = {}
            for k, v in dist.args:
                if isinstance(v, Fixed):
                    args[k] = v.value
                elif isinstance(v, float):
                    args[k] = self.estimates[f"{vname}.{k}"]
            try:
                other = spreads(dist.name, args)
            except (KeyError, TypeError):
                return ""
            if key not in other:
                return ""
            return ", ".join(
                f"cv {100 * v:.3g} %" if k == "cv" else f"{k} {v:.4g}"
                for k, v in other.items() if k != key
            )  # fmt: skip
        return ""

    def individual(self) -> pd.DataFrame:
        """Each unit's normal scores (conditional modes, NONMEM's EBEs) and its values of
        the varying quantities and the model parameters the fit sets."""
        if self.method != "laplace":
            raise ValueError(
                "individual values are for parameters that vary between units (method "
                "laplace); here one set of values fits every unit."
            )
        return self.objective.individual(self.x, self.modes)

    @property
    def reached(self) -> int:
        """How many starts ended within 0.1 of the best -2LL."""
        v = self.starts["-2LL"].to_numpy()
        return int(np.sum(np.isfinite(v) & (v <= np.nanmin(v) + 0.1)))

    def __repr__(self) -> str:
        units = ""
        if self.method == "laplace":
            units = f", {self.objective.n} units"
        lines = [
            f"{self.problem.name}: method {self.method} ({self.optimizer}), {self.status}; "
            f"-2LL {self.m2ll:.4f} ({self.n_observations} measurements{units})",
            f"starts: {len(self.starts)}; {self.reached} at the best fit (within 0.1); "
            f"redrawn {self.redrawn}; failed runs during the fit {self.failed_runs}; "
            f"{self.seconds:.0f} s",
            "",
            f"{'quantity':<22}{'estimate':>12}{'SE':>11}{'RSE %':>7}   {'95 % CI':<22}  also",
        ]
        for row in self.table().itertuples(index=False):
            ci = f"({row[4]:.4g}, {row[5]:.4g})" if np.isfinite(row[4]) else ""
            rse = f"{row[3]:.3g}" if np.isfinite(row[3]) else ""
            se = f"{row[2]:.4g}" if np.isfinite(row[2]) else ""
            lines.append(f"{row[0]:<22}{row[1]:>12.5g}{se:>11}{rse:>7}   {ci:<22}  {row[6]}")
        status = getattr(self, "covariance_status", "ok")
        if status != "ok":
            lines += ["", f"standard errors {status}."]
        if self.method == "laplace":
            sh = ", ".join(f"{k} {100 * v:.1f} %" for k, v in self.shrinkage.items())
            lines += ["", f"shrinkage (1 - SD of the units' scores): {sh}"]
        lines += [f"note: {t}" for t in getattr(self, "notes", [])]
        lines += ["", *(f"warning: {t}" for t in near_bound_text(self))] if self.near_bounds else []
        return "\n".join(lines)


# ----------------------------------------------------------------------------- the fit
def fit_optimize(problem, obj, *, progress=None) -> Result:
    opts = problem._options
    names, x0 = obj.names, np.asarray(obj.x0, float)
    sc = scales(names, x0, obj.bounds)
    n_starts = int(opts.get("starts", 1))
    rng = np.random.default_rng(int(opts.get("seed", 0)))
    maxiter = int(opts.get("maxiter") or DEFAULT_MAXITER)
    progress = progress or opts.get("progress") or "text"
    t_start = time.perf_counter()

    least_squares = not obj.error_estimated

    def residuals(u, args=None):
        r = obj._residuals(to_x(u, sc))
        ok = jnp.all(jnp.isfinite(r))
        return jnp.where(ok, r, BAD), ok

    def value(u, args=None):
        v = obj._minus2ll(to_x(u, sc))
        ok = jnp.isfinite(v)
        return jnp.where(ok, v, BAD_VALUE), ok

    # ---- one start's steps, compiled once and shared by every start
    if least_squares:
        solver, fn = optx.LevenbergMarquardt(rtol=TOL, atol=TOL), residuals
    else:
        solver, fn = optx.BFGS(rtol=TOL, atol=TOL), value
    tags = frozenset()
    f_struct, aux_struct = jax.eval_shape(lambda u: fn(u), jnp.asarray(to_u(x0, sc)))
    runs = eqx.filter_jit(lambda u: fn(u)[1])
    init = eqx.filter_jit(lambda u: solver.init(fn, u, None, {}, f_struct, aux_struct, tags))
    term = eqx.filter_jit(lambda u, st: solver.terminate(fn, u, None, {}, st, tags)[0])

    @eqx.filter_jit
    def step(u, state):
        u2, state2, ok = solver.step(fn, u, None, {}, state, tags)
        done = solver.terminate(fn, u2, None, {}, state2, tags)[0]
        f = 2.0 * state2.f_info.as_min() if least_squares else state2.f_info.f
        return u2, state2, done, ok, f

    workers = ThreadPoolExecutor(max_workers=max(1, min(n_starts, os.cpu_count() or 1)))
    U_start, redrawn = draw_starts(problem, runs, x0, sc, rng, n_starts, workers)

    # sum of squares -> -2LL: a constant of the data (0 for BFGS, which minimizes -2LL)
    const = 0.0
    if least_squares:
        const = float(obj(to_x(jnp.asarray(U_start[0]), sc))) - float(
            jnp.sum(fn(jnp.asarray(U_start[0]))[0] ** 2)
        )

    run = run_starts(names, sc, U_start, init, step, term, maxiter, progress, workers,
                     "optimize", const)  # fmt: skip
    final = np.array([float(v) for v in ThreadPoolExecutor(len(U_start)).map(obj, list(run.X))])
    best = int(np.argmin(np.where(np.isfinite(final), final, np.inf)))
    gradient = jax.jit(jax.grad(lambda u: obj._minus2ll(to_x(u, sc))))
    cov = covariance(None, to_u(run.X[best], sc), sc,
                     gradient=lambda u: gradient(jnp.asarray(u)))  # fmt: skip
    return finish(problem, obj, "optimize", "Levenberg-Marquardt" if least_squares else "BFGS",
                  names, x0, sc, U_start, run, final, redrawn, obj.bounds, t_start,
                  **cov)  # fmt: skip


def draw_starts(problem, runs, x0, sc, rng, n_starts, workers):
    """Start 1 is the values written; the others are drawn around them; a start that does not
    run is drawn again. Returns the starts (on the estimation scale) and how many were redrawn."""
    u0 = to_u(x0, sc)
    spread = _spread(sc, x0)
    U = [u0] + [u0 + spread * rng.normal(size=len(u0)) for _ in range(n_starts - 1)]
    redrawn = 0
    ok = [bool(v) for v in workers.map(lambda u: runs(jnp.asarray(u)), U)]
    for _ in range(MAX_REDRAWS):
        if all(ok):
            break
        for i in [i for i, v in enumerate(ok) if not v]:
            U[i] = u0 + spread * rng.normal(size=len(u0))
            redrawn += 1
        ok = [bool(v) for v in workers.map(lambda u: runs(jnp.asarray(u)), U)]
    if not any(ok):
        workers.shutdown()
        raise RuntimeError(
            f"{problem.name}: the model does not run at any start (also after {redrawn} "
            "draws); check the starting values (obj.status(x) says why a run fails)."
        )
    if not all(ok):
        print(f"{ok.count(False)} start(s) dropped: no run after {MAX_REDRAWS} draws.")
    return np.stack([u for u, v in zip(U, ok, strict=True) if v]), redrawn


class _Run:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run_starts(names, sc, U_start, init, step, term, maxiter, progress, workers, label,
               const=0.0) -> _Run:  # fmt: skip
    """Every start in its own thread (JAX runs them on separate cores), with the live text
    and Ctrl-C keeping the work. ``step(u, state) -> (u, state, done, ok, f)``."""
    t_start = time.perf_counter()
    n = len(U_start)
    m2ll = np.full(n, np.nan)
    X = np.array([np.asarray(to_x(jnp.asarray(u), sc)) for u in U_start])
    steps = np.zeros(n, int)
    done = np.zeros(n, bool)
    trace: list[tuple] = []
    failed = np.zeros(n, int)
    stop = threading.Event()

    def run(j):
        u = jnp.asarray(U_start[j])
        state = init(u)
        if bool(term(u, state)):
            done[j] = True
            return
        for it in range(1, maxiter + 1):
            if stop.is_set():
                return
            u, state, d, okj, f = step(u, state)
            steps[j] = it
            failed[j] += int(not bool(okj))
            m2ll[j] = float(f) + const
            X[j] = np.asarray(to_x(u, sc))
            trace.append((j + 1, it, m2ll[j], *X[j]))
            if bool(d):
                done[j] = True
                return

    shown = _Progress(progress, names, n, label)
    status = None
    futures = [workers.submit(run, j) for j in range(n)]
    try:
        while not all(f.done() for f in futures):
            shown.update(steps, m2ll, X, done, time.perf_counter() - t_start)
            time.sleep(0.25)
        for f in futures:
            f.result()  # raises what a start raised
    except KeyboardInterrupt:
        stop.set()
        status = "stopped"
        for f in futures:
            f.exception()  # wait for each start to finish its step
    workers.shutdown()
    shown.update(steps, m2ll, X, done, time.perf_counter() - t_start, last=True)
    return _Run(X=X, steps=steps, done=done, failed=failed, trace=trace, status=status)


def finish(problem, obj, method, optimizer, names, x0, sc, U_start, run, final, redrawn,
           user_bounds, t_start, **extra) -> Result:  # fmt: skip
    """The result from the runs: the best start, the table of starts, the trace."""
    X, steps, done, status, n = run.X, run.steps, run.done, run.status, len(U_start)
    start_status = []
    for j in range(n):
        if done[j]:
            start_status.append("converged" if np.isfinite(final[j]) else "failed")
        else:
            start_status.append("stopped" if status == "stopped" else "max iterations")
    best = int(np.argmin(np.where(np.isfinite(final), final, np.inf)))
    if status is None:
        status = start_status[best]
    starts_table = pd.DataFrame(
        {
            "start": np.arange(1, n + 1),
            "-2LL": final,
            "steps": steps,
            "status": start_status,
            **{f"{k} (start)": v for k, v in zip(names, _x_of(U_start, sc).T, strict=True)},
            **{k: X[:, i] for i, k in enumerate(names)},
        }
    )
    trace_table = pd.DataFrame(sorted(run.trace), columns=["start", "iteration", "-2LL", *names])
    result = Result(
        problem=problem,
        method=method,
        optimizer=optimizer,
        names=names,
        x=X[best],
        x0=x0,
        m2ll=float(final[best]),
        status=status,
        starts=starts_table,
        trace=trace_table,
        scales=sc,
        redrawn=redrawn,
        failed_runs=int(run.failed.sum()),
        n_observations=obj.n_observations,
        seconds=time.perf_counter() - t_start,
        near_bounds=near_bounds(names, X[best], user_bounds),
        **extra,
    )
    warn_near_bounds(result)
    return result


def _x_of(U, sc):
    return np.asarray(jax.vmap(lambda u: to_x(u, sc))(jnp.asarray(U)))


class _Progress:
    """The live text: a block rewritten in place in a terminal; elsewhere a line every 10 s
    (and at the end)."""

    def __init__(self, mode, names, n, label="optimize"):
        self.mode, self.names, self.n, self.label = mode, names, n, label
        self.tty = sys.stdout.isatty()
        self.lines = 0
        self.printed = -1e9

    def update(self, steps, m2ll, X, done, seconds, last=False):
        if self.mode == "none" or not np.isfinite(m2ll).any():
            return
        best = int(np.nanargmin(np.where(np.isfinite(m2ll), m2ll, np.inf)))
        clock = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
        head = (
            f"{self.label}  {self.n} start{'s' * (self.n != 1)}: {int((~done).sum())} "
            f"running, {int(done.sum())} "
            f"done   steps {steps.min()}-{steps.max()}   best -2LL {m2ll[best]:.4f}   {clock}"
        )
        vals = "  ".join(f"{k} {v:.4g}" for k, v in zip(self.names, X[best], strict=True))
        if self.tty:
            block = [head, f"          best: {vals}"]
            if self.lines:
                sys.stdout.write(f"\033[{self.lines}F")
            sys.stdout.write("".join(f"\033[2K{line}\n" for line in block))
            sys.stdout.flush()
            self.lines = len(block)
        elif last or seconds - self.printed >= 10:
            print(head + "  |  " + vals, flush=True)
            self.printed = seconds
