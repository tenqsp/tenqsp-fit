# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""What a fit gives after the table (the ``Result`` methods call these):

- **Tables** as NONMEM's: per measurement (sdtab: DV, PRED, IPRED, IWRES, censoring) and per
  unit (patab: scores, values, covariates).
- **Goodness of fit**: DV against PRED and IPRED, IWRES against time and IPRED; each unit's
  measurements with its curves.
- **VPC** (visual predictive check): the data's percentiles over time against those of
  data simulated from the fit with the same design (doses, times, limits), new units each
  time (marginal).
- **Profile likelihood**: one quantity held at values around its estimate, the others
  refitted; the 95 % interval is where -2LL is within 3.84 of its minimum.
- **save**: the problem, every table, the trace, plots and what reproduces the run.

Words: PRED is the prediction of the typical unit (normal scores 0); IPRED the unit's own
(its conditional modes, NONMEM's EBEs); IWRES = (DV - IPRED)/sd, sd from the error model at
IPRED. Without parameters that vary between units, PRED = IPRED. A censored measurement has
no DV; it is shown at its limit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

CHI2_95 = 3.841458820694124  # chi-square, 1 degree of freedom, 95 %


# ----------------------------------------------------------------------------- helpers
def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError(
            "plots need matplotlib: pip install matplotlib (or tenqsp-fit[plot])."
        ) from None
    return plt


def _cond(result):
    """The likelihood given the units' values (for laplace, the Laplace objective's)."""
    obj = result.objective
    return getattr(obj, "_cond", obj)


def _modes(result):
    return getattr(result, "modes", None)


def _unit_column(result) -> str:
    return result.problem._unit or "unit"


def _xs(result):
    return jnp.asarray(result.x, dtype=jnp.float64)


# ----------------------------------------------------------------------------- tables
def sdtab(result) -> pd.DataFrame:
    """One row per measurement: the unit, TIME, the link (``observe`` line), DV, PRED,
    IPRED, IWRES, CENS (1 if censored) and LIMIT (its limit, for a censored link)."""
    cond, x = _cond(result), _xs(result)
    Z = _modes(result)
    ipred = cond._predictions(x, cond._fs(cond._model_params(x, None if Z is None else Z)))
    pred = cond._predictions(x, cond._fs(cond._model_params(x, None)))
    rows = []
    for lk in cond._links:
        lhs = lk["lhs"]
        for i, u in enumerate(cond._units):
            o = u.observations.get(lhs)
            if o is None:
                continue
            k = len(o.times)
            f = jnp.asarray(ipred[lhs])[i, :k]
            sd = np.asarray(cond._sd(lk, x, f)) if lk["error"] else np.full(k, np.nan)
            cens = np.zeros(k, bool) if o.censored is None else np.asarray(o.censored, bool)
            limit = np.full(k, np.nan) if o.below is None else np.asarray(o.below, float)
            dv = np.where(cens, np.nan, np.asarray(o.values, float))
            f = np.asarray(f)
            rows.append(
                pd.DataFrame(
                    {
                        "unit": u.id,
                        "TIME": np.asarray(o.times, float),
                        "link": lhs,
                        "DV": dv,
                        "PRED": np.asarray(pred[lhs])[i, :k],
                        "IPRED": f,
                        "IWRES": (dv - f) / sd,
                        "CENS": cens.astype(int),
                        "LIMIT": limit,
                    }
                )
            )
    out = pd.concat(rows, ignore_index=True)
    return out.rename(columns={"unit": _unit_column(result)})


def patab(result) -> pd.DataFrame:
    """One row per unit: its normal scores (``z(CL)``, with parameters that vary), the
    varying quantities, the model parameters the fit sets, and the covariates the fit uses."""
    cond, x = _cond(result), _xs(result)
    Z = _modes(result)
    env = cond._unit_values(x, None if Z is None else jnp.asarray(Z))
    table = {"unit": list(cond.units)}
    varying = [name for name, _ in getattr(cond, "varying", ())]
    if Z is not None:
        table.update({f"z({name})": np.asarray(Z)[:, k] for k, name in enumerate(varying)})
    for n in [*varying, *cond._wrt]:
        if n not in table:
            table[n] = np.asarray(env[n])
    for c in result.problem._covariates():
        table[c] = [u.covariates[c] for u in cond._units]
    return pd.DataFrame(table).rename(columns={"unit": _unit_column(result)})


def write_tables(result, folder) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = {"sdtab": sdtab(result), "patab": patab(result)}
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=False)
    return tables


# ----------------------------------------------------------------------------- goodness of fit
def plot_gof(result, fig=None):
    """One row per link: DV against PRED (with parameters that vary) and IPRED, with the
    line of identity; IWRES against TIME and against IPRED, with 0 and +-2. Censored values
    are crosses at their limit (DV panels only). ``fig``: a figure (or subfigure) to draw in."""
    plt = _pyplot()
    tab = sdtab(result)
    links = list(dict.fromkeys(tab.link))
    population = _modes(result) is not None
    ncols = 4 if population else 3
    if fig is None:
        fig = plt.figure(figsize=(3.3 * ncols, 3.1 * len(links)))
    axes = fig.subplots(len(links), ncols, squeeze=False)
    for row, lhs in zip(axes, links, strict=True):
        d = tab[tab.link == lhs]
        meas, cens = d[d.CENS == 0], d[d.CENS == 1]
        panels = (["PRED"] if population else []) + ["IPRED"]
        for ax, col in zip(row, panels, strict=False):
            ax.plot(meas[col], meas.DV, "o", ms=3.5, alpha=0.7, color="#1f77b4")
            if len(cens):
                ax.plot(cens[col], cens.LIMIT, "x", ms=4, color="#d62728", label="censored")
            v = np.concatenate([meas[col], meas.DV, cens.LIMIT.dropna()])
            v = v[np.isfinite(v)]
            if len(v):
                lo, hi = float(v.min()), float(v.max())
                ax.plot([lo, hi], [lo, hi], "-", color="0.4", lw=1)
            ax.set_xlabel(col)
            ax.set_ylabel(f"DV ({lhs})")
        for ax, col in zip(row[len(panels) :], ["TIME", "IPRED"], strict=True):
            ax.plot(meas[col], meas.IWRES, "o", ms=3.5, alpha=0.7, color="#1f77b4")
            for h, style in ((0, "-"), (-2, "--"), (2, "--")):
                ax.axhline(h, ls=style, color="0.4", lw=1)
            ax.set_xlabel(col)
            ax.set_ylabel("IWRES")
        row[0].set_title(lhs, loc="left", fontsize=10)
    fig.suptitle(f"{result.problem.name}: goodness of fit", fontsize=11)
    _tight(fig)
    return fig


def _tight(fig):
    """Tight layout for a figure; a subfigure takes its parent's layout."""
    if hasattr(fig, "tight_layout"):
        fig.tight_layout()


def plot_individual(result, units=None, *, ncols: int = 4, points: int = 200):
    """Each unit's measurements with its own curve (IPRED, solid) and, with parameters that
    vary, the typical unit's (PRED, dashed); one panel per unit and link."""
    plt = _pyplot()
    cond, x = _cond(result), _xs(result)
    Z = _modes(result)
    ids = list(cond.units)
    pick = ids if units is None else list(units)
    unknown = [u for u in pick if u not in ids]
    if unknown:
        raise ValueError(f"plot_individual: no unit {unknown[0]!r}; units: {ids[:8]}...")
    t_end = max(float(np.max(o.times)) for u in cond._units for o in u.observations.values())
    grid = np.unique(np.concatenate([np.linspace(0.0, t_end, points),
                                     *[o.times for u in cond._units
                                       for o in u.observations.values()]]))  # fmt: skip
    ipred = cond.curves(x, grid, Z)
    pred = cond.curves(x, grid, None) if Z is not None else None
    links = [lk["lhs"] for lk in cond._links]
    panels = [(u, lhs) for u in pick for lhs in links]
    ncols = min(ncols, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    for ax, (uid, lhs) in zip(axes, panels, strict=False):
        i = ids.index(uid)
        u = cond._units[i]
        o = u.observations.get(lhs)
        ax.plot(grid, ipred[lhs][i], "-", color="#1f3b73", lw=1.4)
        if pred is not None:
            ax.plot(grid, pred[lhs][i], "--", color="0.5", lw=1)
        if o is not None:
            cens = np.zeros(len(o.times), bool) if o.censored is None else o.censored
            ax.plot(o.times[~cens], o.values[~cens], "o", ms=3.5, color="#1f77b4")
            if cens.any():
                ax.plot(o.times[cens], o.below[cens], "x", ms=4, color="#d62728")
        title = f"{uid}" if len(links) == 1 else f"{uid}: {lhs}"
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    dashed = ", dashed: typical unit)" if pred is not None else ")"
    fig.suptitle(f"{result.problem.name}: each unit (solid: own fit{dashed}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_diagnostics(result):
    """Goodness of fit, the VPC and the path of the fit, one above the other in one figure."""
    plt = _pyplot()
    cond = _cond(result)
    n_links = len(cond._links)
    from .live import _figsize

    heights = [3.1 * n_links, 3.8, _figsize(result.names)[1]]
    fig = plt.figure(figsize=(13.2, sum(heights) + 0.6), layout="constrained")
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=heights)
    plot_gof(result, fig=top)
    vpc(result, fig=middle)
    result._draw_trace(fig=bottom)
    return fig


# ----------------------------------------------------------------------------- VPC
@dataclasses.dataclass
class VPC:
    """A visual predictive check: ``table`` (per link and time bin: the data's percentiles
    and the simulated ones, median and interval over the simulations), ``figure``."""

    table: pd.DataFrame
    figure: object
    n: int
    percentiles: tuple
    draws: pd.DataFrame | None = None  # the prior predictive check: the quantities drawn
    failed: int = 0  # simulated studies whose model did not run, left out
    ran: np.ndarray | None = None

    def __repr__(self) -> str:
        what = "prior predictive check" if self.draws is not None else "VPC"
        more = ", .draws" if self.draws is not None else ""
        left = f" ({self.failed} did not run, left out)" if self.failed else ""
        return (f"<{what}: {self.n} simulations{left}, percentiles {list(self.percentiles)}, "
                f"{len(self.table)} bins; .table, .figure{more}>")  # fmt: skip


def _bins(t: np.ndarray, n_bins: int | None):
    """Time bins: the distinct times when there are few (a designed study), else bins with
    equal numbers of measurements."""
    distinct = np.unique(t)
    if n_bins is None and len(distinct) <= 12:
        return np.searchsorted(distinct, t), len(distinct)
    n_bins = n_bins or 8
    edges = np.unique(np.quantile(t, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
    return which, len(edges) - 1


def vpc(result, *, n: int = 200, seed: int = 0, bins: int | None = None,
        percentiles=(5, 50, 95), interval: float = 0.95, log: bool = False,
        fig=None) -> VPC:  # fmt: skip
    """Simulate the data ``n`` times from the fit (new units each time: scores drawn from
    their distribution and correlation; residual errors from the error model; the same
    doses, times and censoring limits), and compare the percentiles over time. After NUTS,
    each simulated study takes the quantities of a different posterior draw."""
    rng = np.random.default_rng(seed)
    return simulate_vpc(result.problem.name, result.objective, _vpc_xs(result, n, rng), rng,
                        bins=bins, percentiles=percentiles, interval=interval, log=log,
                        title="visual predictive check", fig=fig)  # fmt: skip


def _vpc_xs(result, n, rng):
    """The quantities of each simulated study: the estimates, or after NUTS a draw each."""
    draws = getattr(result, "draws", None)
    if draws is not None and len(draws):
        k = draws[list(result.names)].to_numpy()
        return k[rng.integers(0, len(k), n)]
    return np.tile(np.asarray(result.x, float), (n, 1))


def simulate_vpc(name, objective, xs, rng, *, bins=None, percentiles=(5, 50, 95),
                 interval=0.95, log=False, title="visual predictive check",
                 fig=None) -> VPC:  # fmt: skip
    """The VPC's simulations and plot, one study per row of ``xs`` (the quantities)."""
    cond = getattr(objective, "_cond", objective)
    if any(lk["error"] is None for lk in cond._links):
        raise NotImplementedError("vpc: a link with a written likelihood cannot be simulated.")
    n = len(xs)
    n_units = len(cond.units)
    q = getattr(objective, "q", 0)
    E = rng.standard_normal((n, n_units, q))
    X = jnp.asarray(xs, dtype=jnp.float64)

    def one(x, e):
        Z = e @ jnp.linalg.cholesky(objective._R(x)).T if q else None
        return cond._predictions(x, cond._fs(cond._model_params(x, Z)))

    batched = jax.jit(jax.vmap(one))
    preds = []
    for k in range(0, n, 50):  # in chunks: memory
        preds.append(jax.tree.map(np.asarray, batched(X[k : k + 50], jnp.asarray(E[k : k + 50]))))
    sims = {lhs: np.concatenate([p[lhs] for p in preds]) for lhs in preds[0]}
    # a study whose model did not run (values out of the model's range: wide priors, a
    # unit far in a distribution's tail) is left out, and counted
    ran = np.ones(n, bool)
    for lk in cond._links:
        at = np.asarray(lk["obs"]) | np.asarray(lk["cens"])
        ran &= np.all(np.isfinite(sims[lk["lhs"]][:, at]), axis=1)
    failed = int((~ran).sum())
    if failed == n:
        raise RuntimeError(f"{name}: none of the {n} simulated studies ran (the model fails at "
                           "every set of values drawn).")  # fmt: skip
    if failed:
        sims = {k: v[ran] for k, v in sims.items()}
        X = X[jnp.asarray(ran)]
        title = f"{title} ({failed} of {n} studies did not run, left out)"

    rows = []
    ptiles = tuple(float(p) for p in percentiles)
    a = (1 - interval) / 2
    for lk in cond._links:
        lhs = lk["lhs"]
        f = sims[lhs]  # (n, units, measurements)
        sd = np.asarray(jax.vmap(lambda x, g, lk=lk: cond._sd(lk, x, g))(X, jnp.asarray(f)))
        y = f + sd * rng.standard_normal(f.shape)
        valid = np.asarray(lk["obs"]) | np.asarray(lk["cens"])
        limit = np.asarray(lk["limit"])
        censored_link = bool(np.asarray(lk["cens"]).any())
        obs_v = np.where(np.asarray(lk["cens"]), limit, np.asarray(lk["dv"]))
        times = np.zeros_like(obs_v)
        for i, u in enumerate(cond._units):
            o = u.observations.get(lhs)
            if o is not None:
                times[i, : len(o.times)] = o.times
                if censored_link and o.below is not None:
                    lim = np.asarray(o.below, float)
                    y[:, i, : len(lim)] = np.maximum(y[:, i, : len(lim)], lim)
        t, v = times[valid], obs_v[valid]
        ys = y[:, valid]  # (n, measurements)
        which, nb = _bins(t, bins)
        for b in range(nb):
            sel = which == b
            if not sel.any():
                continue
            row = {"link": lhs, "bin": b + 1, "time": float(np.median(t[sel])),
                   "n": int(sel.sum())}  # fmt: skip
            sim_p = np.percentile(ys[:, sel], ptiles, axis=1)  # (percentiles, n)
            for j, p in enumerate(ptiles):
                tag = f"p{p:g}"
                row[f"obs {tag}"] = float(np.percentile(v[sel], p))
                row[f"sim {tag}"] = float(np.median(sim_p[j]))
                row[f"sim {tag} lo"] = float(np.quantile(sim_p[j], a))
                row[f"sim {tag} hi"] = float(np.quantile(sim_p[j], 1 - a))
            rows.append(row)
    table = pd.DataFrame(rows)
    fig = _plot_vpc(name, table, cond, ptiles, interval, log, title, fig)
    return VPC(table=table, figure=fig, n=n, percentiles=ptiles, failed=failed, ran=ran)


def prior_predictive(problem, *, n: int = 200, seed: int = 0, bins: int | None = None,
                     percentiles=(5, 50, 95), interval: float = 0.95,
                     log: bool = False) -> VPC:  # fmt: skip
    """The prior predictive check (before fitting: what data do the priors expect?):
    ``n`` studies simulated with the data's doses, times and limits, each with its
    quantities drawn from the priors (written, or by default as under ``method nuts``), new
    units from their distributions, and residual errors. The data's percentiles over the
    simulated ones' bands, as a VPC. Returns a ``VPC`` with ``.draws``, the quantities
    drawn (and whether the model ran with them; a study that did not run is left out of the
    bands and counted in ``.failed``)."""
    from .errors import ProblemError
    from .laplace import Laplace
    from .likelihood import Objective
    from .optimize import scales
    from .priors import lkj_draw, priors

    problem.check()
    objective = Laplace(problem) if problem._varying() else Objective(problem)
    names = list(objective.names)
    x0 = np.asarray(objective.x0, float)
    sc = scales(names, x0, objective.bounds, objective.kinds)
    found, problems = priors(problem, names, x0, sc, defaults=True)
    if problems:
        raise ProblemError(problems)
    rng = np.random.default_rng(seed)
    xs = np.tile(x0, (n, 1))
    index = {k: i for i, k in enumerate(names)}
    for p in found:
        xs[:, index[p.name]] = p.draw(rng, n)
    for c in problem._correlations:
        R = lkj_draw(rng, len(c.names), n)
        for i, a in enumerate(c.names):
            for j in range(i + 1, len(c.names)):
                xs[:, index[f"corr({a},{c.names[j]})"]] = R[:, i, j]
    vp = simulate_vpc(problem.name, objective, xs, rng, bins=bins, percentiles=percentiles,
                      interval=interval, log=log,
                      title=f"prior predictive check, {n} studies from the priors")  # fmt: skip
    vp.draws = pd.DataFrame(xs, columns=names)
    vp.draws["ran"] = vp.ran  # False: the model did not run with these values
    return vp


def _plot_vpc(name, table, cond, ptiles, interval, log, title, fig=None):
    plt = _pyplot()
    links = list(dict.fromkeys(table.link))
    if fig is None:
        fig = plt.figure(figsize=(5.2 * len(links), 3.8))
    axes = fig.subplots(1, len(links), squeeze=False)
    mid = ptiles[len(ptiles) // 2]
    for ax, lhs in zip(axes[0], links, strict=True):
        d = table[table.link == lhs]
        for u in cond._units:
            o = u.observations.get(lhs)
            if o is None:
                continue
            cens = np.zeros(len(o.times), bool) if o.censored is None else o.censored
            ax.plot(o.times[~cens], o.values[~cens], "o", ms=2.5, color="0.6", alpha=0.6)
        for p in ptiles:
            tag = f"p{p:g}"
            color = "#d62728" if p == mid else "#1f77b4"
            ax.fill_between(d.time, d[f"sim {tag} lo"], d[f"sim {tag} hi"], color=color,
                            alpha=0.18, lw=0)  # fmt: skip
            ax.plot(d.time, d[f"obs {tag}"], "-" if p == mid else "--", color=color, lw=1.4)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("TIME")
        ax.set_ylabel(lhs)
        ax.set_title(f"{lhs}: data percentiles {', '.join(f'{p:g}' for p in ptiles)} (lines), "
                     f"simulated {interval:.0%} intervals (bands)", fontsize=8)  # fmt: skip
    fig.suptitle(f"{name}: {title}", fontsize=11)
    _tight(fig)
    return fig


# ----------------------------------------------------------------------------- profile
@dataclasses.dataclass
class Profile:
    """The profile likelihood of one quantity: ``table`` (each value held, -2LL with the
    others refitted, and their estimates), ``pieces`` (each stretch of values where -2LL is
    within ``cut`` of its minimum, in order; NaN at an end the profile did not reach),
    ``interval`` (from the lowest to the highest value of the pieces), ``plot()``.

    The profile walks out from the estimate on each side until -2LL is past the cut-off,
    then (``scan``) goes on past each end, refitting the rest at each point from the
    neighbouring point and from the ends of the fit's other starts: where the likelihood
    has a second optimum beyond a hump higher than the cut-off, the set of values within
    the cut-off has two pieces, and the walk alone sees only the one around the estimate.
    ``better``: a point of the profile with -2LL lower than the fit's (by more than 0.01:
    the fit missed a better optimum), or None; the cut-off is then from that point.

    ``cut`` and ``rule``: the chi-square cut-off (1 degree of freedom; 3.84 at 95 %) holds
    for large data. When the residual SD is estimated from few measurements it gives
    intervals that are too narrow (90 % intervals covered the truth in 79-86 % of fits of
    one subject with 11 measurements). So for individual or pooled data (method optimize)
    with an error-model quantity estimated and no likelihood written by hand, the cut-off
    is the small-sample one, n log(1 + F/(n - p)) with F the ``level`` quantile of the F
    distribution with 1 and n - p degrees of freedom, n the measurements (not censored)
    and p the estimated quantities other than the error model's (Bates and Watts 1988,
    profile t; as R's ``confint`` for ``nls``). It approaches the chi-square one as n grows.
    """

    name: str
    estimate: float
    m2ll: float
    table: pd.DataFrame
    interval: tuple[float, float]
    level: float
    cut: float = float("nan")
    rule: str = ""
    pieces: list = dataclasses.field(default_factory=list)
    scanned: tuple = (float("nan"), float("nan"))
    better: dict | None = None

    def plot(self, ax=None):
        plt = _pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(4.6, 3.3))
        d = self.table.sort_values("value")
        low = self._minimum()
        walk = d[d.status != "scan"] if "status" in d else d
        ax.plot(d.value, d["-2LL"] - low, "-", color="#1f3b73", lw=1)
        ax.plot(walk.value, walk["-2LL"] - low, "o", color="#1f3b73", ms=4, label="walk")
        scan = d[d.status == "scan"] if "status" in d else d.iloc[:0]
        if len(scan):
            ax.plot(scan.value, scan["-2LL"] - low, "o", mfc="white", color="#1f3b73", ms=4,
                    label="scan")  # fmt: skip
            ax.legend(fontsize=7, frameon=False)
        ax.axhline(self.cut, ls="--", color="#d62728", lw=1)
        for lo_, hi_ in self.pieces:
            a = lo_ if np.isfinite(lo_) else d.value.min()
            b = hi_ if np.isfinite(hi_) else d.value.max()
            ax.axvspan(a, b, color="#d62728", alpha=0.08, lw=0)
        ax.axvline(self.estimate, color="0.5", lw=1)
        if self.estimate > 0 and (d.value > 0).all() and d.value.max() / d.value.min() > 20:
            ax.set_xscale("log")
        ax.set_xlabel(self.name)
        ax.set_ylabel("-2LL above the minimum")
        ax.set_title(f"{self.name}: {self.level:.0%} {self._set_text()}", fontsize=9)
        ax.figure.tight_layout()
        return ax

    def _minimum(self) -> float:
        return self.m2ll if self.better is None else float(self.better["-2LL"])

    def _set_text(self) -> str:
        if len(self.pieces) > 1:
            parts = " and ".join(f"({lo:.4g}, {hi:.4g})" for lo, hi in self.pieces)
            return f"set in {len(self.pieces)} pieces {parts}"
        lo, hi = self.interval
        return f"interval ({lo:.4g}, {hi:.4g})"

    def __repr__(self) -> str:
        better = ""
        if self.better is not None:
            better = (f"; a better fit than the estimate at {self.name} = "
                      f"{self.better['value']:.4g} (-2LL lower by "
                      f"{self.m2ll - self.better['-2LL']:.3g})")  # fmt: skip
        return (f"<Profile of {self.name}: estimate {self.estimate:.4g}, {self.level:.0%} "
                f"{self._set_text()}, {len(self.table)} points; cut-off {self.cut:.3g} "
                f"({self.rule}){better}>")  # fmt: skip


def _cut(level: float, result=None) -> tuple[float, str]:
    """The profile's cut-off on -2LL above the minimum, and the rule (see ``Profile``)."""
    from scipy.stats import chi2
    from scipy.stats import f as f_dist

    large = (float(chi2.ppf(level, 1)), "chi-square, 1 df")
    if result is None or getattr(result, "method", None) != "optimize":
        return large
    problem = result.problem
    if problem._likelihoods:
        return large
    errors = {f"{e.lhs}.{q}" for e in problem._errors for q in e.model.quantities}
    if not any(n in errors for n in result.names):
        return large
    n = int(result.n_observations) - int(getattr(result, "n_censored", 0))
    p = sum(1 for name in result.names if name not in errors)
    df = n - p
    if df < 1:
        return large
    F = float(f_dist.ppf(level, 1, df))
    return float(n * np.log1p(F / df)), f"small sample: n {n}, p {p}, F(1, {df})"


def _copy(problem):
    import copy

    p = copy.copy(problem)
    for attr in ("_params", "_correlations", "_errors", "_defines", "_options"):
        setattr(p, attr, copy.copy(getattr(problem, attr)))
    return p


def profile(result, name: str, *, points: int = 8, step: float | None = None,
            level: float = 0.95, scan: bool = True, scan_range=None,
            scan_points: int = 8) -> Profile:  # fmt: skip
    """Hold ``name`` at values on each side of its estimate (on its estimation scale, in
    steps of ``step``: by default half its standard error there, else 0.2), refit the rest
    from the neighbouring point, and stop a side once -2LL is past the cut-off (or after
    ``points``). Then (``scan``) go on past each end of the walk to ``scan_range`` (by
    default 1/100 and 100 times the estimate, within its bounds; for a quantity of any sign,
    20 steps beyond the walk) in ``scan_points`` points a side, each refitted from the
    neighbouring point and from the ends of the fit's other starts (the best kept), and
    each end of a piece found there found more closely by two halvings. The model is
    compiled once for every point (``tenqsp_fit.held``), the two sides run at the same time,
    and a line per point is printed (not with ``options progress=none``)."""
    from concurrent.futures import ThreadPoolExecutor

    from .held import held_fitter
    from .optimize import to_u, to_x

    names = list(result.names)
    if name not in names:
        from .errors import nearest

        raise ValueError(f"profile: '{name}' is not estimated.{nearest(name, names)}")
    if not isinstance(scan, bool):
        raise TypeError("profile: scan= is True or False.")
    j, sc = names.index(name), result.scales
    u_best = np.asarray(to_u(np.asarray(result.x, float), sc), float)
    u_hat = float(u_best[j])
    if step is None:
        se_u = np.sqrt(np.diag(getattr(result, "cov_u", np.full((len(names),) * 2, np.nan))))[j]
        step = 0.5 * float(se_u) if np.isfinite(se_u) and se_u > 0 else 0.2
    cut, rule = _cut(level, result)
    quiet = result.problem._options.get("progress") == "none"
    t0 = time.perf_counter()
    fit = held_fitter(result, j)
    fit(u_hat, u_best)  # compiles, at the estimate (a step or two), before the sides share it

    def row(u, m2ll, status):
        x = np.asarray(to_x(jnp.asarray(u), sc))
        return {"value": float(x[j]), "-2LL": m2ll, "status": status,
                **{n: float(x[i]) for i, n in enumerate(names) if i != j}}  # fmt: skip

    def say(r, sign, what):
        if not quiet:
            print(f"   profile {name} = {r['value']:.4g}: -2LL {r['-2LL'] - result.m2ll:+.3g} "
                  f"from the best ({'below' if sign < 0 else 'above'} the estimate, {what}; "
                  f"{time.perf_counter() - t0:.0f} s)", flush=True)  # fmt: skip

    def side(sign):
        out, start, warm, u_end = [], u_best, None, u_hat
        for k in range(1, points + 1):
            u_end = u_hat + sign * k * step
            u, m2ll, status, warm = fit(u_end, start, warm)
            if np.isfinite(m2ll):
                start = u
            r = row(u, m2ll, status)
            out.append(r)
            say(r, sign, f"point {k}")
            if not np.isfinite(m2ll) or m2ll - result.m2ll > cut:
                break
        if scan:
            out += scanned(sign, u_end, start, out[-1]["-2LL"] if out else np.nan)
        return out

    others = _other_ends(result, sc, u_best)
    lo_u, hi_u = _scan_range(result, j, sc, u_hat, step, scan_range)

    def best_of(h, starts):
        best = None
        for st in starts:
            u, m2ll, status, _ = fit(h, st)
            if np.isfinite(m2ll) and (best is None or m2ll < best[1]):
                best = (u, m2ll, status)
        return best

    def scanned(sign, u_from, start, f_from):
        far = hi_u if sign > 0 else lo_u
        if not np.isfinite(far) or (far - u_from) * sign <= 0:
            return []
        out = []

        def at(h, starts, what="refining"):
            got = best_of(h, starts)
            if got is None:
                return None, np.nan
            out.append(row(got[0], got[1], "scan"))
            say(out[-1], sign, what)
            return got[0], got[1]

        grid = u_from + (far - u_from) * np.arange(1, scan_points + 1) / scan_points
        pts = [(u_from, start, f_from)]  # the walk's end
        for k, h in enumerate(grid, start=1):
            u, m2ll = at(h, [pts[-1][1], *others], f"scan {k} of {scan_points}")
            pts.append((h, start if u is None else u, m2ll))
        f = np.array([p_[2] for p_ in pts])
        inside = lambda m: m - result.m2ll <= cut  # noqa: E731
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            # a dip between grid points: the bottom by golden section, then its piece's ends
            dip = (i + 1 < len(pts) and np.isfinite(f[i]) and not (f[i - 1] <= f[i])
                   and not (f[i + 1] <= f[i]))  # fmt: skip
            if dip and not inside(f[i]):
                best = _bottom(at, pts[i - 1][0], pts[i + 1][0], (b[0], b[1], f[i]), inside,
                               others)  # fmt: skip
                if inside(best[2]):
                    for edge in (pts[i - 1], pts[i + 1]):
                        _halve(at, best[0], best[1], edge[0], inside, others)
            elif np.isfinite(f[i]) and np.isfinite(f[i - 1]) and inside(f[i]) != inside(f[i - 1]):
                ins, outs = (b, a) if inside(f[i]) else (a, b)
                _halve(at, ins[0], ins[1], outs[0], inside, others)
        return out

    with ThreadPoolExecutor(2) as pool:
        lo_side, hi_side = pool.map(side, (-1, 1))
    centre = {"value": float(result.x[j]), "-2LL": result.m2ll, "status": "estimate",
              **result.estimates}  # fmt: skip
    table = pd.DataFrame([*lo_side, centre, *hi_side]).sort_values("value", kind="stable")
    table = table[["value", "-2LL", "status", *[n for n in names if n != name]]]
    table = table.reset_index(drop=True)
    low = float(np.nanmin(table["-2LL"]))
    better = None
    if low < result.m2ll - 0.01:
        better = table.loc[int(np.nanargmin(table["-2LL"].to_numpy()))].to_dict()
    floor = min(low, result.m2ll)
    pieces = _pieces(table, floor, cut)
    interval = (pieces[0][0], pieces[-1][1]) if pieces else (float("nan"), float("nan"))

    def x_at(h):
        u = u_best.copy()
        u[j] = h
        return float(np.asarray(to_x(jnp.asarray(u), sc))[j])

    p = Profile(name=name, estimate=float(result.x[j]), m2ll=result.m2ll, table=table,
                interval=interval, level=level, cut=cut, rule=rule, pieces=pieces,
                scanned=(x_at(lo_u), x_at(hi_u)) if scan else (float("nan"),) * 2,
                better=better)  # fmt: skip
    p.seconds = time.perf_counter() - t0
    return p


def _bottom(at, lo, hi, best, inside, others, times: int = 8):
    """The bottom of a dip between ``lo`` and ``hi`` (estimation scale) by golden section,
    refitting from the best point so far; stops early once inside the cut-off. Returns the
    best (value, solution, -2LL)."""
    g = (3 - np.sqrt(5)) / 2
    val = lambda m: m if np.isfinite(m) else np.inf  # noqa: E731
    x1, x2 = lo + g * (hi - lo), hi - g * (hi - lo)
    u1, f1 = at(x1, [best[1], *others])
    u2, f2 = at(x2, [best[1], *others])
    for _ in range(times):
        for x, u, m in ((x1, u1, f1), (x2, u2, f2)):
            if val(m) < val(best[2]):
                best = (x, u, m)
        if inside(best[2]):
            break
        if val(f1) < val(f2):  # the bottom is in (lo, x2)
            hi, x2, u2, f2 = x2, x1, u1, f1
            x1 = lo + g * (hi - lo)
            u1, f1 = at(x1, [best[1], *others])
        else:  # in (x1, hi)
            lo, x1, u1, f1 = x1, x2, u2, f2
            x2 = hi - g * (hi - lo)
            u2, f2 = at(x2, [best[1], *others])
    for x, u, m in ((x1, u1, f1), (x2, u2, f2)):
        if val(m) < val(best[2]):
            best = (x, u, m)
    return best


def _halve(at, h_in, u_in, h_out, inside, others, times: int = 4) -> None:
    """Halve the stretch between a point inside the cut-off and one outside, ``times``
    times, refitting from the inside point: an end of a piece, more closely."""
    for _ in range(times):
        mid = 0.5 * (h_in + h_out)
        u, m = at(mid, [u_in, *others])
        if u is None:
            return
        if inside(m):
            h_in, u_in = mid, u
        else:
            h_out = mid


def _other_ends(result, sc, u_best, most: int = 3) -> list:
    """The ends of the fit's other starts (estimation scale), the lowest -2LL first, one per
    optimum: where the scan also starts, so that it can fall into another optimum."""
    from .optimize import to_u

    st = getattr(result, "starts", None)
    if st is None or not len(st):
        return []
    st = st[np.isfinite(st["-2LL"])].sort_values("-2LL")
    out, seen = [], [u_best]
    for _, r in st.iterrows():
        try:
            u = np.asarray(to_u(np.array([float(r[n]) for n in result.names]), sc), float)
        except (ValueError, KeyError):
            continue
        if not np.all(np.isfinite(u)) or any(np.max(np.abs(u - v)) < 1e-3 for v in seen):
            continue
        seen.append(u)
        out.append(u)
        if len(out) == most:
            break
    return out


def _scan_range(result, j, sc, u_hat, step, scan_range) -> tuple[float, float]:
    """Where the scan ends on each side (estimation scale)."""
    from .optimize import to_u

    if scan_range is not None:
        lo, hi = (float(v) for v in scan_range)
        if not lo < hi:
            raise ValueError("profile: scan_range is (low, high) with low < high.")
        x = np.asarray(result.x, float)

        def u_of(v):
            y = x.copy()
            y[j] = v
            try:
                return float(to_u(y, sc)[j])
            except ValueError:
                return float("nan")

        return u_of(lo), u_of(hi)
    kind = sc[j][0]
    if kind in ("log", "above", "below"):
        return u_hat - np.log(100.0), u_hat + np.log(100.0)
    if kind == "logit":  # up to near the bounds
        return -10.0, 10.0
    return u_hat - 20 * step, u_hat + 20 * step


def _pieces(table, m2ll, cut) -> list[tuple[float, float]]:
    """Each stretch of values where -2LL - ``m2ll`` is within ``cut``, its ends found by
    linear interpolation between neighbouring points; NaN at an end that is within the
    cut-off at the last point."""
    d = table.sort_values("value")
    d = d[np.isfinite(d["-2LL"])]
    v, f = d.value.to_numpy(), d["-2LL"].to_numpy() - m2ll
    inside = f <= cut
    out, start = [], None
    for i in range(len(v)):
        if inside[i] and start is None:
            if i == 0:
                start = float("nan")
            else:
                w = (cut - f[i - 1]) / (f[i] - f[i - 1])
                start = float(v[i - 1] + w * (v[i] - v[i - 1]))
        if not inside[i] and start is not None:
            w = (cut - f[i - 1]) / (f[i] - f[i - 1])
            out.append((start, float(v[i - 1] + w * (v[i] - v[i - 1]))))
            start = None
    if start is not None:
        out.append((start, float("nan")))
    return out


# ----------------------------------------------------------------------------- save
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_record(result, text: str | None = None, data_bytes: bytes | None = None) -> dict:
    """What reproduces a run (PRD F4.7, F11.1): method, status, -2LL, estimates, options, the
    SHA-256 of the problem text (model and fit) and of the data as read, versions,
    platform."""
    import tenqsp as tq

    from . import __version__

    p = result.problem
    if text is None:
        text = p.to_text(model=True)
    if data_bytes is None:
        data_bytes = p._data.table.to_csv(index=False).encode() if p._data is not None else b""
    versions = {"tenqsp_fit": __version__, "tenqsp": tq.__version__, "jax": jax.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "python": platform.python_version()}  # fmt: skip
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        pass
    return {
        "problem": p.name,
        "method": result.method,
        "optimizer": result.optimizer,
        "status": result.status,
        "m2ll": result.m2ll,
        "estimates": result.estimates,
        "covariance_status": getattr(result, "covariance_status", None),
        "seconds": round(float(result.seconds), 2),
        "options": {k: v for k, v in p._options.items()},
        "method options": dict(p._method.options) if p._method is not None else {},
        "hashes": {"problem.tq": _sha256(text.encode()), "data.csv": _sha256(data_bytes)},
        "versions": versions,
        "platform": f"{platform.system()} {platform.machine()}",
        "argv": sys.argv[:1],
    }


def save(result, folder) -> Path:
    """Everything a run gives, in one folder: ``problem.tq`` (model and fit text),
    ``data.csv`` (the data as read), ``estimates.csv`` (the table), ``starts.csv``,
    ``trace.csv``, ``sdtab.csv``, ``patab.csv``, ``result.txt`` (as printed), plots (trace,
    goodness of fit, each unit), ``passport.md`` and ``passport.json``, and ``run.json``
    (versions, hashes of the model, fit and data texts, options, status, -2LL, time)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = result.problem
    text = p.to_text(model=True)
    (folder / "problem.tq").write_text(text, encoding="utf-8")
    data_bytes = b""
    if p._data is not None:
        data_bytes = p._data.table.to_csv(index=False).encode()
        (folder / "data.csv").write_bytes(data_bytes)
    result.table().to_csv(folder / "estimates.csv", index=False)
    result.starts.to_csv(folder / "starts.csv", index=False)
    result.trace.to_csv(folder / "trace.csv", index=False)
    if getattr(result, "draws", None) is not None:  # NUTS: every draw, and each chain's checks
        result.draws.to_csv(folder / "draws.csv", index=False)
        result.convergence().to_csv(folder / "convergence.csv", index=False)
    write_tables(result, folder)
    (folder / "result.txt").write_text(repr(result) + "\n", encoding="utf-8")
    plt = _pyplot()
    figures = {"trace.png": result.plot_trace, "gof.png": result.plot_gof,
               "individual.png": result.plot_individual}  # fmt: skip
    for fname, draw in figures.items():
        fig = draw()
        fig.savefig(folder / fname, dpi=110)
        plt.close(fig)
    if len(result.starts) > 1:
        ax = result.plot_waterfall()
        ax.figure.savefig(folder / "waterfall.png", dpi=110)
        plt.close(ax.figure)
    pp = result.passport()
    pp.save(folder / "passport.md")
    pp.save(folder / "passport.json")
    run = run_record(result, text, data_bytes)
    run["saved"] = datetime.now(UTC).isoformat(timespec="seconds")
    (folder / "run.json").write_text(json.dumps(run, indent=2, default=float) + "\n",
                                     encoding="utf-8")  # fmt: skip
    return folder
