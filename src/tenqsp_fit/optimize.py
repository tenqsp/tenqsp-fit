# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``method optimize``: one set of parameters for all the data, by multi-start optimization
(Optimistix) with TenQSP's exact derivatives.

- **The optimizer.** With additive errors of known sd, -2LL is a sum of squares up to a
  constant of the data: ``(DV - f)/sd`` for each measured value,
  ``sqrt(-2 log Phi((limit - f)/sd))`` for each censored one; Levenberg-Marquardt minimizes
  it, with exact Jacobians. Otherwise (an error model's quantities estimated, or an sd that
  changes with the prediction: proportional, combined, power, whose ``2 log sd`` terms change
  with the parameters) -2LL is minimized by BFGS with its exact gradient
  (Levenberg-Marquardt converges slowly there: the sd's terms do not go to zero at the
  minimum; theophylline, pooled: 48-61 steps against 18 with sd fixed). Results report -2LL.
- **Scales.** Each quantity is estimated on a scale where any value is allowed: within its
  bounds ``(lo, hi)`` by ``lo + (hi - lo)*sigmoid(u)`` (one bound: ``lo + exp(u)``), else on the
  log scale when its start is above 0, else as it is. The summary lists the scales.
- **Starts.** Start 1 is the values written; the others are drawn around them on those
  scales (log scale: a factor within about 5 either way, 95 %; bounded: 1 on the logit scale;
  as is: half the value, at least 0.5) with the ``seed``. A start whose run fails or gives a
  non-finite residual is drawn again. Each start runs in its own thread, so the starts run
  on separate cores (an infection-age model, 8 starts: 7x faster than one ``jax.vmap`` call,
  which on the CPU runs them one after the other).
- **Failed runs** during the fit are rejected steps: every residual becomes a large constant,
  so the step is refused and the step size shrinks. They are counted.
- **Progress.** ``progress text``: a block rewritten in place in a terminal, a line every 10 s
  elsewhere. ``progress plot`` (the default in a notebook): the live plot (``live.py``).
  ``progress none`` for quiet. The record is kept (``result.trace``,
  ``result.plot_trace()``), and Ctrl-C (or a notebook's stop) stops the fit and returns what
  it has, marked "stopped", without standard errors. ``options timeout=2h`` does the same when
  the time is up (off by default: a large model may need hours); a step already being taken
  is finished first.
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

from .live import default_progress, draw_trace
from .problem import seconds

__all__ = ["Result"]

# a failed run (a rejected step): far above any cost a run can have, so that a step to it is
# never accepted (with 1e3 per residual, a poor start with residuals over 1000 sd had a larger
# cost than a failed run, and the fit stepped to where the model does not run and ended there)
BAD = 1e100  # each residual, least squares
BAD_VALUE = 1e100  # -2LL, BFGS
FAILED_KEPT = 50  # failed runs whose cause is asked of TenQSP after the fit
SPREAD_LOG = math.log(5) / 2  # starts on the log scale: 95 % within a factor 5
MAX_REDRAWS = 20
DEFAULT_MAXITER = 200
TOL = 1e-8  # Levenberg-Marquardt's rtol and atol (changes of the parameters and of -2LL)
SINGULAR = 1e-12  # Hessian eigenvalue ratio below which it is singular (no standard errors)


# ----------------------------------------------------------------------------- scales
def scales(names, x0, bounds, kinds=None) -> list[tuple]:
    """The scale of each quantity: ("logit", lo, hi), ("above", lo), ("below", hi), ("log",)
    or ("as is",). ``kinds`` (one per quantity, or None): the scale a distribution's family
    sets ("log" or "as is"); without one, a start above 0 gives the log scale."""
    out = []
    kinds = kinds or [None] * len(names)
    for x, (lo, hi), kind in zip(x0, bounds, kinds, strict=True):
        if np.isfinite(lo) and np.isfinite(hi):
            out.append(("logit", lo, hi))
        elif np.isfinite(lo):
            out.append(("above", lo))
        elif np.isfinite(hi):
            out.append(("below", hi))
        elif kind is not None:
            out.append((kind,))
        elif x > 0:
            out.append(("log",))
        else:
            out.append(("as is",))
    return out


def user_bounds(problem, names):
    """The bounds written with ``in (lo, hi)`` (the near-bound warning's), not the natural
    range of a quantity (a logitnormal's (0, 1), a correlation's (-1, 1))."""
    from .problem import _Estimate

    written = {d.name: d.bounds for d in problem._params
               if isinstance(d, _Estimate) and d.bounds and d.written is None}  # fmt: skip
    return [written.get(n, (-np.inf, np.inf)) for n in names]


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
    bound)}``. Within 1 % of the interval's width; for a bound on one side, within 1 % of the
    larger of the bound and the estimate (in size)."""
    out = {}
    for n, v, (lo, hi) in zip(names, x, bounds, strict=True):
        if np.isfinite(lo) and np.isfinite(hi):
            width = hi - lo
        else:
            width = max(abs(b) for b in (lo, hi, v) if np.isfinite(b))
        if np.isfinite(lo) and v - lo <= NEAR_BOUND * width:
            out[n] = ("lower", float(lo))
        elif np.isfinite(hi) and hi - v <= NEAR_BOUND * width:
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


STOPPED_SE = "not computed: the fit was stopped before it converged"


NOT_ASKED = "not computed (not asked for)"


def no_covariance(p: int, status: str | None = None) -> dict:
    """No standard errors (a stopped fit: the Hessian away from the minimum means nothing)."""
    nan = np.full((p, p), np.nan)
    return dict(se=np.full(p, np.nan), ci=np.full((p, 2), np.nan), cov_u=nan, cov=nan,
                covariance_status=status or STOPPED_SE, hessian_u=nan)  # fmt: skip


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
        eig = np.linalg.eigvalsh(H)
        if eig[0] <= SINGULAR * eig[-1]:  # positive only by round-off: an exact ridge
            raise np.linalg.LinAlgError
        cov_u = 2.0 * np.linalg.inv(H)
        if not np.all(np.isfinite(np.diag(cov_u)) & (np.diag(cov_u) > 0)):
            raise np.linalg.LinAlgError
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
def _corr_frame(cov: np.ndarray, names) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        c = cov / np.outer(sd, sd)
    return pd.DataFrame(c, index=list(names), columns=list(names))


class Result:
    """A fit: ``result.estimates`` ({name: value}), ``result.m2ll`` (-2 log-likelihood at them),
    ``result.status`` ("converged", "max iterations", "stopped" or "failed"), ``result.table()``
    (estimates, standard errors, RSE %, 95 % intervals), ``result.se``, ``result.cov``
    (covariance of the estimates), ``result.corr`` (their correlations), ``result.starts`` (each
    start's values, its end, -2LL, steps, status), ``result.trace`` (every step of every start),
    ``result.plot_trace()``, ``result.plot_waterfall()``; ``result.tables()`` (NONMEM's sdtab
    and patab), ``result.plot_gof()``, ``result.plot_individual()``, ``result.vpc()``,
    ``result.profile(name)``, ``result.save(folder)``. With parameters that vary between units
    (method laplace), also ``result.individual()`` (each unit's scores and values) and
    ``result.shrinkage``. ``print`` shows the table."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def estimates(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.x), strict=True))

    @property
    def corr(self) -> pd.DataFrame:
        """Correlations of the estimates, one row and one column per quantity (from
        ``result.cov``; NaN where the standard errors failed)."""
        return _corr_frame(
            np.asarray(getattr(self, "cov", np.full((len(self.names),) * 2, np.nan))), self.names
        )

    def plot_trace(self):
        """The path of the fit, as the live plot showed it: -2LL against the step and one
        panel per quantity, one line per start (the best dark). Returns the figure."""
        return self._draw_trace()

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
        from matplotlib.ticker import MaxNLocator

        ok = self.starts[np.isfinite(self.starts["-2LL"])]
        v = np.sort(ok["-2LL"].to_numpy())
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3.2))
        d = v - v[0]  # above the best; differences below 0.1 are one optimum
        ax.plot(np.arange(1, len(v) + 1), d, "o-", color="#1f77b4")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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
                "laplace); here one set of values fits every unit (result.tables() has each "
                "unit's values and covariates)."
            )
        return self.objective.individual(self.x, self.modes)

    def tables(self, folder=None) -> dict:
        """NONMEM-style tables: ``{"sdtab": per measurement (DV, PRED, IPRED, IWRES,
        censoring), "patab": per unit (scores, values, covariates)}``; with a ``folder``, also
        written there as CSV."""
        from . import results

        if folder is not None:
            return results.write_tables(self, folder)
        return {"sdtab": results.sdtab(self), "patab": results.patab(self)}

    def plot_gof(self):
        """Goodness of fit: DV against PRED and IPRED, IWRES against time and IPRED, one row
        per link. Returns the figure."""
        from .results import plot_gof

        return plot_gof(self)

    def plot_individual(self, units=None, *, ncols: int = 4):
        """Each unit's measurements with its own curve (and the typical unit's, dashed);
        ``units``: a list of unit ids (default: all). Returns the figure."""
        from .results import plot_individual

        return plot_individual(self, units, ncols=ncols)

    def vpc(self, *, n: int = 200, seed: int = 0, bins: int | None = None,
            percentiles=(5, 50, 95), interval: float = 0.95, log: bool = False):  # fmt: skip
        """Visual predictive check: the data simulated ``n`` times from the fit with the
        same design (new units each time); the data's percentiles over time against the
        simulated ones' ``interval``. Returns a ``VPC`` (``.table``, ``.figure``)."""
        from .results import vpc

        return vpc(self, n=n, seed=seed, bins=bins, percentiles=percentiles,
                   interval=interval, log=log)  # fmt: skip

    def profile(self, name: str, *, points: int = 8, step: float | None = None,
                level: float = 0.95, scan: bool = True, scan_range=None,
                scan_points: int = 8):  # fmt: skip
        """Profile likelihood of one estimated quantity: held at values on each side of its
        estimate, the others refitted; then (``scan``, the default) scanned further out, so
        that a second optimum shows as a second piece of the set. Returns a ``Profile``
        (``.table``, ``.pieces``, ``.interval`` over all pieces, ``.cut``, ``.rule``,
        ``.better``, ``.plot()``); ``scan=False``: the walk only (faster). The cut-off is the
        small-sample one when the residual SD is estimated and the data are individual or
        pooled (see ``Profile``); otherwise the chi-square one."""
        from .results import profile

        prof = profile(self, name, points=points, step=step, level=level, scan=scan,
                       scan_range=scan_range, scan_points=scan_points)  # fmt: skip
        self.__dict__.setdefault("_profiles", {})[name] = prof  # the passport shows its shape
        d = getattr(self, "_doctor", None)
        if d is not None:  # the doctor already ran: add what this profile found
            from .doctor import _ORDER, profile_findings

            d.findings = [
                f
                for f in d.findings
                if not f.message.startswith(
                    (f"the {prof.level:.0%} profile set of {name} ", f"the profile of {name} ")
                )
            ]
            d.findings = sorted([*d.findings, *profile_findings({name: prof})],
                                key=lambda f: _ORDER[f.severity])  # fmt: skip
        return prof

    def save(self, folder):
        """Write everything to ``folder``: the problem text, the data, estimates, starts,
        trace, sdtab and patab, the printed result, plots, and ``run.json`` (versions,
        hashes, options) to reproduce it. Returns the folder."""
        from .results import save

        return save(self, folder)

    def doctor(self):
        """What the fit can and cannot tell (``tenqsp_fit.doctor``): ridges, weakly
        determined quantities, optima, bounds, shrinkage, the tolerance and gradient checks,
        jumps in the model. Run by ``tf.fit``; returns a ``Doctor`` (``print`` lists the
        findings, ``.table()``)."""
        from .doctor import examine

        if getattr(self, "_doctor", None) is None:
            self._doctor = examine(self)
        return self._doctor

    def passport(self):
        """One document on the fit (``tenqsp_fit.passport``): the run and how to repeat it,
        each quantity and how well the data determine it, ridges, the doctor's findings, the
        numerical checks, named predictions, TenQSP's own reports. ``print`` shows it;
        ``.markdown()``, ``.to_dict()``, ``.save(path)``."""
        from .passport import passport

        return passport(self)

    def predict(self, *, regimen=None, times, outputs=None, covariates=None,
                name: str | None = None, n: int = 200, between_units: bool = False,
                seed: int = 0):  # fmt: skip
        """The model's outputs at ``times`` under ``regimen`` (a TenQSP regimen; None: the
        model's own doses), for the typical unit, with an interval from ``n`` draws: after
        ``nuts`` a posterior interval; after ``optimize`` and ``laplace`` approximate, from the
        standard errors. ``outputs``: observables (default: those the fit measures);
        ``covariates``: the covariate values to predict for, one number each (needed when the
        fit uses covariates: ``covariates={"WT": 70}``); ``between_units=True`` adds a new
        unit's 90 % range; ``name`` keeps the prediction in the passport; ``seed`` sets the
        random draws (the same seed, the same prediction). Returns a ``Prediction``
        (``.table``, ``.plot()``)."""
        from .predict import predict

        return predict(self, regimen=regimen, times=times, outputs=outputs,
                       covariates=covariates, name=name, n=n, between_units=between_units,
                       seed=seed)  # fmt: skip

    def plot_diagnostics(self):
        """One figure: goodness of fit, the visual predictive check (200 simulations), and the
        path of the fit (each start's, or each chain's). Returns the figure."""
        from .results import plot_diagnostics

        return plot_diagnostics(self)

    def to_arviz(self):
        """For ``method nuts``: the posterior as ArviZ data."""
        raise ValueError(f"to_arviz is for a posterior (method nuts); this fit is "
                         f"{self.method}, with estimates and standard errors: "
                         "result.table().")  # fmt: skip

    def _draw_trace(self, fig=None):
        return draw_trace(self.trace, self.names, fig=fig,
                          title=f"{self.problem.name}: {self.method}, {self.status}")  # fmt: skip

    def _doctor_lines(self) -> list[str]:
        d = getattr(self, "_doctor", None)
        return [] if d is None else ["", repr(d)]

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
            # each column keeps a space before it, however long a number (same widths)
            lines.append(f"{row[0]:<21} {row[1]:>11.5g} {se:>10} {rse:>6}   {ci:<22}  {row[6]}")
        if getattr(self, "with_priors", None) is not None:
            lines += ["", f"with the written priors: the estimates are the posterior mode; "
                      f"-2LL plus the priors' terms {self.with_priors:.4f}"]  # fmt: skip
        status = getattr(self, "covariance_status", "ok")
        if status != "ok":
            lines += ["", f"standard errors {status}."]
        if self.method == "laplace":
            sh = ", ".join(f"{k} {100 * v:.1f} %" for k, v in self.shrinkage.items())
            lines += ["", f"shrinkage (1 - SD of the units' scores): {sh}"]
        lines += [f"note: {t}" for t in getattr(self, "notes", [])]
        if getattr(self, "_doctor", None) is not None:  # the doctor reports the bounds
            return "\n".join(lines + self._doctor_lines())
        lines += ["", *(f"warning: {t}" for t in near_bound_text(self))] if self.near_bounds else []
        return "\n".join(lines)


# ----------------------------------------------------------------------------- the fit
def fit_optimize(problem, obj, *, progress=None, with_se=True) -> Result:
    opts = problem._options
    names, x0 = obj.names, np.asarray(obj.x0, float)
    sc = scales(names, x0, obj.bounds, obj.kinds)
    n_starts = int(opts.get("starts", 1))
    rng = np.random.default_rng(int(opts.get("seed", 0)))
    maxiter = int(opts.get("maxiter") or DEFAULT_MAXITER)
    progress = progress or opts.get("progress") or default_progress()
    t_start = time.perf_counter()

    from .priors import written_penalty

    penalty = written_penalty(problem, names, x0, sc)
    least_squares = obj.sum_of_squares and penalty is None

    def total(x):  # what is minimized: -2LL, plus -2 log of the written priors
        v = obj._minus2ll(x)
        return v if penalty is None else v + penalty(x)

    def residuals(u, args=None):
        r = obj._residuals(to_x(u, sc))
        ok = jnp.all(jnp.isfinite(r))
        return jnp.where(ok, r, BAD), ok

    def value(u, args=None):
        v = total(to_x(u, sc))
        ok = jnp.isfinite(v)
        return jnp.where(ok, v, BAD_VALUE), ok

    # ---- one start's steps, compiled once and shared by every start
    if least_squares:
        solver, fn = optx.LevenbergMarquardt(rtol=TOL, atol=TOL), residuals
    else:
        solver, fn = optx.BFGS(rtol=TOL, atol=TOL), value
    tags = frozenset()
    f_struct, aux_struct = jax.eval_shape(lambda u: fn(u), jnp.asarray(to_u(x0, sc)))
    # jax.jit, not eqx.filter_jit: filter_jit keeps every compiled function of every fit
    # (about 760 memory maps each; Linux allows 65530 per process, so a long session or a
    # profile ran out). The solver's state holds a traced function besides arrays, so its
    # fixed part is kept here and only its arrays pass through jit.
    u_shape = jnp.asarray(to_u(x0, sc))
    shapes = eqx.filter_eval_shape(
        lambda u: solver.init(fn, u, None, {}, f_struct, aux_struct, tags), u_shape
    )
    _, static = eqx.partition(shapes, lambda v: isinstance(v, jax.ShapeDtypeStruct))

    def arrays(state):
        return eqx.partition(state, eqx.is_array)[0]

    runs = jax.jit(lambda u: fn(u)[1])
    init = jax.jit(lambda u: arrays(solver.init(fn, u, None, {}, f_struct, aux_struct, tags)))
    term = jax.jit(
        lambda u, st: solver.terminate(fn, u, None, {}, eqx.combine(st, static), tags)[0]
    )

    @jax.jit
    def step(u, st):
        # the step's aux is the last accepted point's, so whether the point tried ran is
        # read here: a rejected step is checked by ``runs`` at the point it tried
        state = eqx.combine(st, static)
        tried = state.y_eval
        u2, state2, _ = solver.step(fn, u, None, {}, state, tags)
        done = solver.terminate(fn, u2, None, {}, state2, tags)[0]
        f = 2.0 * state2.f_info.as_min() if least_squares else state2.f_info.f
        rejected = state2.num_accepted_steps == state.num_accepted_steps
        return u2, arrays(state2), done, rejected, tried, f

    workers = ThreadPoolExecutor(max_workers=max(1, min(n_starts, os.cpu_count() or 1)))
    U_start, redrawn = draw_starts(problem, runs, x0, sc, rng, n_starts, workers)

    # sum of squares -> -2LL: a constant of the data (0 for BFGS, which minimizes -2LL)
    const = 0.0
    if least_squares:
        const = float(obj(to_x(jnp.asarray(U_start[0]), sc))) - float(
            jnp.sum(fn(jnp.asarray(U_start[0]))[0] ** 2)
        )

    total_jit = jax.jit(total)
    run = run_starts(names, sc, U_start, init, step, term, maxiter, progress, workers,
                     "optimize", const, problem.name, total_jit, runs=runs,
                     timeout=seconds(opts.get("timeout")), t_start=t_start)  # fmt: skip
    final = np.array([float(v) for v in ThreadPoolExecutor(len(U_start)).map(obj, list(run.X))])
    rank = final if penalty is None else np.array([float(total_jit(jnp.asarray(x)))
                                                   for x in run.X])  # fmt: skip
    best = int(np.argmin(np.where(np.isfinite(rank), rank, np.inf)))
    if run.status == "stopped" or not with_se:
        cov = no_covariance(len(names), None if with_se else NOT_ASKED)
    else:
        gradient = jax.jit(jax.grad(lambda u: total(to_x(u, sc))))
        cov = covariance(None, to_u(run.X[best], sc), sc,
                         gradient=lambda u: gradient(jnp.asarray(u)))  # fmt: skip
    return finish(problem, obj, "optimize", "Levenberg-Marquardt" if least_squares else "BFGS",
                  names, x0, sc, U_start, run, final, redrawn, user_bounds(problem, names),
                  t_start, objective=obj, rank=rank,
                  with_priors=None if penalty is None else float(rank[best]),
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
               const=0.0, problem_name="", value=None, runs=None, timeout=None,
               t_start=None) -> _Run:  # fmt: skip
    """Every start in its own thread (JAX runs them on separate cores), with the live text
    and Ctrl-C keeping the work. ``step(u, state) -> (u, state, done, rejected, tried, f)``;
    ``runs(u)``: whether the model runs at ``u``, asked of each rejected step's point;
    ``value(x)``, -2LL at a start, gives the trace its step 0. ``timeout``: seconds from
    ``t_start`` (the fit's start) after which the starts stop as for Ctrl-C."""
    t_start = time.perf_counter() if t_start is None else t_start
    n = len(U_start)
    m2ll = np.full(n, np.nan)
    X = np.array([np.asarray(to_x(jnp.asarray(u), sc)) for u in U_start])
    steps = np.zeros(n, int)
    done = np.zeros(n, bool)
    trace: list[tuple] = []
    failed = np.zeros(n, int)
    failed_at: list[np.ndarray] = []  # the points of failed runs (the first FAILED_KEPT)
    stop = threading.Event()
    lock = threading.Lock()  # a step's -2LL and values are shown together

    def run(j):
        u = jnp.asarray(U_start[j])
        if value is not None:
            trace.append((j + 1, 0, float(value(X[j])), *X[j]))
        state = init(u)
        if bool(term(u, state)):
            done[j] = True
            return
        for it in range(1, maxiter + 1):
            if stop.is_set():
                return
            u, state, d, rejected, tried, f = step(u, state)
            if runs is not None and bool(rejected) and not bool(runs(tried)):
                failed[j] += 1
                if len(failed_at) < FAILED_KEPT:
                    failed_at.append(np.asarray(to_x(tried, sc)))
            fj, xj = float(f) + const, np.asarray(to_x(u, sc))
            with lock:
                steps[j], m2ll[j], X[j] = it, fj, xj
            trace.append((j + 1, it, fj, *xj))
            if bool(d):
                done[j] = True
                return

    shown = _Progress(progress, names, n, label, problem_name, trace)
    futures = [workers.submit(run, j) for j in range(n)]
    show = locked_show(shown, lock, steps, m2ll, X, done, t_start)
    status, stopped_by = watch(futures, stop, timeout, t_start, show)
    workers.shutdown()
    show(last=True, status=status)
    return _Run(X=X, steps=steps, done=done, failed=failed, failed_at=failed_at,
                evaluations=steps.copy(), trace=trace, status=status,
                stopped_by=stopped_by)  # fmt: skip


def locked_show(shown, lock, steps, m2ll, X, done, t_start):
    """``shown.update`` with a copy of the starts' state taken under ``lock``, so a line
    never shows one step's -2LL with another step's values."""

    def show(last=False, status=None):
        with lock:
            state = steps.copy(), m2ll.copy(), X.copy(), done.copy()
        shown.update(*state, time.perf_counter() - t_start, last=last, status=status)

    return show


def watch(futures, stop, timeout, t_start, show) -> tuple[str | None, str | None]:
    """Wait for the starts, showing progress; on Ctrl-C, or once ``timeout`` seconds have
    passed since ``t_start``, set ``stop`` and wait for each start to finish its step.
    Returns the status (None, or "stopped") and what stopped the fit."""
    stopped_by = None
    try:
        while not all(f.done() for f in futures):
            show()
            if timeout is not None and time.perf_counter() - t_start > timeout:
                stopped_by = f"the time limit (options timeout, {_duration(timeout)})"
                break
            time.sleep(0.25)
        if stopped_by is None:
            for f in futures:
                f.result()  # raises what a start raised
    except KeyboardInterrupt:
        stopped_by = "Ctrl-C"
    if stopped_by is None:
        return None, None
    stop.set()
    for f in futures:
        f.exception()  # wait for each start to finish its step
    return "stopped", stopped_by


def _duration(s: float) -> str:
    return f"{s / 3600:g} h" if s >= 3600 else f"{s / 60:g} min" if s >= 60 else f"{s:g} s"


def finish(problem, obj, method, optimizer, names, x0, sc, U_start, run, final, redrawn,
           bounds_written, t_start, rank=None, **extra) -> Result:  # fmt: skip
    """The result from the runs: the best start (lowest ``rank``, by default -2LL ``final``;
    with written priors, -2LL plus their terms), the table of starts, the trace."""
    rank = final if rank is None else rank
    X, steps, done, status, n = run.X, run.steps, run.done, run.status, len(U_start)
    start_status = []
    for j in range(n):
        if done[j]:
            start_status.append("converged" if np.isfinite(final[j]) else "failed")
        else:
            start_status.append("stopped" if status == "stopped" else "max iterations")
    best = int(np.argmin(np.where(np.isfinite(rank), rank, np.inf)))
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
        evaluations=int(np.sum(run.evaluations)),
        failure_causes=failure_causes(obj, run.failed_at),
        n_observations=obj.n_observations,
        n_censored=obj.n_censored,
        seconds=time.perf_counter() - t_start,
        stopped_by=getattr(run, "stopped_by", None),
        near_bounds=near_bounds(names, X[best], bounds_written),
        **extra,
    )
    warn_near_bounds(result)
    return result


def failure_causes(obj, points) -> dict[str, int]:
    """Why the kept runs failed, asked of TenQSP at each one's point: ``{cause: count}``. A
    point where every unit's run ends but a value is not finite counts as "values not
    finite"."""
    out: dict[str, int] = {}
    for x in points:
        try:
            st = obj.status(x)
        except Exception:  # noqa: BLE001 - the cause is reported, not needed
            st = {}
        bad = [v for v in st.values() if v != "ok"]
        cause = bad[0] if bad else "values not finite"
        out[cause] = out.get(cause, 0) + 1
    return out


def _x_of(U, sc):
    return np.asarray(jax.vmap(lambda u: to_x(u, sc))(jnp.asarray(U)))


class _Progress:
    """The live text: a block rewritten in place in a terminal; elsewhere a line every 10 s
    (and at the end). With ``progress plot``, the live plot instead (and the text's last
    line)."""

    def __init__(self, mode, names, n, label="optimize", problem_name="", trace=None):
        self.mode, self.names, self.n, self.label = mode, names, n, label
        self.tty = sys.stdout.isatty()
        self.lines = 0
        self.printed = -1e9
        self.trace = trace if trace is not None else []
        self.plot = None
        if mode == "plot":
            from .live import LivePlot

            self.plot = LivePlot(names, label, problem_name or label)

    def update(self, steps, m2ll, X, done, seconds, last=False, status=None):
        if self.mode == "none":
            return
        if self.plot is not None:
            n_done = int(done.sum())
            text = status or (f"{n_done} of {self.n} start{'s' * (self.n != 1)} done"
                              if self.n > 1 else ("done" if last else "running"))  # fmt: skip
            self.plot.update(list(self.trace), ["start", "iteration", "-2LL", *self.names],
                             text, seconds, last=last)  # fmt: skip
            if not last:
                return
        if not np.isfinite(m2ll).any():
            return
        best = int(np.nanargmin(np.where(np.isfinite(m2ll), m2ll, np.inf)))
        clock = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
        head = (
            f"{self.label}  {self.n} start{'s' * (self.n != 1)}: {int((~done).sum())} "
            f"running, {int(done.sum())} "
            f"done   steps {steps.min()}-{steps.max()}   best -2LL {m2ll[best]:.4f}   {clock}"
        )
        vals = "  ".join(f"{k} {v:.4g}" for k, v in zip(self.names, X[best], strict=True))
        if self.plot is not None:  # the plot showed the run; the text gives its end
            print(head + "  |  " + vals, flush=True)
        elif self.tty:
            block = [head, f"          best: {vals}"]
            if self.lines:
                sys.stdout.write(f"\033[{self.lines}F")
            sys.stdout.write("".join(f"\033[2K{line}\n" for line in block))
            sys.stdout.flush()
            self.lines = len(block)
        elif last or seconds - self.printed >= 10:
            print(head + "  |  " + vals, flush=True)
            self.printed = seconds
