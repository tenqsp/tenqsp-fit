# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Predictions from a fit for a new regimen: ``result.predict(regimen=, times=)``.

Each prediction is the model run once per set of values drawn from what the fit knows:

- after ``nuts``, the posterior draws, so the interval is a **posterior interval**;
- after ``optimize`` and ``laplace``, values drawn from the estimates' normal approximation (on
  the estimation scale: log, logit or as is, with the standard errors and correlations), so the
  interval is **approximate (from the standard errors)**; confidence bounds come later;
- without standard errors, the estimates only, and no interval.

The curves are the typical unit's (normal scores 0); with ``between_units=True``, each draw also
gets a new unit (scores drawn from their distribution), which gives a new unit's 90 % range.
They are the model's values, without the residual error: what the model says the unit does,
not what one more measurement would read.

A fit with covariates predicts for the covariate values given (``covariates={"WT": 70}``): the
typical unit, and a new unit, of that weight. The draws are random, from ``seed``: the same seed
gives the same prediction."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

APPROXIMATE = "approximate (from the standard errors)"
POSTERIOR = "posterior"
ESTIMATES = "the estimates only (no standard errors, so no interval)"
CHUNK = 50  # draws per batched run (memory)


@dataclasses.dataclass
class Prediction:
    """A prediction: ``table`` (time, output, median, 2.5 % and 97.5 %; with
    ``between_units``, also ``new unit 5 %`` and ``new unit 95 %``), ``kind`` (posterior,
    approximate, or the estimates only), ``plot()``, ``summary()``."""

    name: str
    kind: str
    table: pd.DataFrame
    n: int
    failed: int
    between_units: bool
    regimen: str
    covariates: dict = dataclasses.field(default_factory=dict)

    def summary(self) -> str:
        """One line: what was predicted, and how the interval was made."""
        outs = ", ".join(dict.fromkeys(self.table.output))
        t = self.table.time
        failed = f"; {self.failed} of {self.n} runs failed, left out" if self.failed else ""
        units = "; with a new unit's 90 % range" if self.between_units else ""
        covs = "".join(f"; {k} {v:.4g}" for k, v in self.covariates.items())
        return (f"{outs} from {t.min():.4g} to {t.max():.4g} ({self.regimen}{covs}); "
                f"{self.kind}, {self.n} draws{units}{failed}")  # fmt: skip

    def plot(self, *, log: bool = False):
        """One panel per output: the median, its 95 % band, and a new unit's 90 % range."""
        from .results import _pyplot

        plt = _pyplot()
        outs = list(dict.fromkeys(self.table.output))
        fig, axes = plt.subplots(1, len(outs), figsize=(4.6 * len(outs), 3.4), squeeze=False)
        for ax, out in zip(axes[0], outs, strict=True):
            d = self.table[self.table.output == out]
            if self.between_units:
                ax.fill_between(d.time, d["new unit 5 %"], d["new unit 95 %"], color="#1f3b73",
                                alpha=0.12, lw=0, label="a new unit, 90 %")  # fmt: skip
            if self.kind != ESTIMATES:
                ax.fill_between(d.time, d["2.5 %"], d["97.5 %"], color="#1f3b73", alpha=0.3,
                                lw=0, label=f"typical unit, 95 %, {self.kind}")  # fmt: skip
            ax.plot(d.time, d["median"], color="#1f3b73", lw=1.6, label="typical unit, median")
            if log:
                ax.set_yscale("log")
            ax.set_xlabel("time")
            ax.set_ylabel(out)
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(self.name, fontsize=10)
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"<Prediction {self.name}: {self.summary()}>"


def predict(result, *, regimen=None, times, outputs=None, covariates=None, name=None,
            n: int = 200, between_units: bool = False, seed: int = 0) -> Prediction:  # fmt: skip
    import tenqsp as tq

    from .errors import nearest
    from .optimize import to_x

    problem = result.problem
    obj = result.objective
    cond = getattr(obj, "_cond", obj)
    model = problem.model
    times = np.asarray(times, float)
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("predict: times are one increasing list of times.")
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"predict: n is a positive whole number, got {n!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"predict: seed is a whole number, got {seed!r}.")
    if regimen is not None and not isinstance(regimen, tq.Regimen):
        raise TypeError("predict: regimen is a TenQSP regimen (tq.bolus(...), "
                        f"tq.infusion(...)), got {type(regimen).__name__}.")  # fmt: skip
    known = list(problem._m.observables)
    outputs = list(problem._outputs()) if outputs is None else list(outputs)
    for o in outputs:
        if o not in known:
            raise ValueError(f"predict: '{o}' is not an observable of the model.{nearest(o, known)}"
                             f" Observables: {', '.join(known)}.")  # fmt: skip
    covariates = _covariates(problem, covariates)
    q = getattr(obj, "q", 0)
    if between_units and not q:
        raise ValueError("predict: between_units=True needs parameters that vary between units "
                         "(method laplace, or nuts on such a problem).")  # fmt: skip

    # ---- the draws of the quantities
    rng = np.random.default_rng(seed)
    draws = getattr(result, "draws", None)
    if draws is not None and len(draws):
        k = draws[list(result.names)].to_numpy()
        X = k[rng.choice(len(k), size=n, replace=len(k) < n)]
        kind = POSTERIOR
    else:
        cov_u = getattr(result, "cov_u", None)
        if cov_u is None or not np.all(np.isfinite(cov_u)):
            X, kind, n = np.asarray(result.x, float)[None, :], ESTIMATES, 1
        else:
            from .optimize import to_u

            sc = result.scales
            u0 = np.asarray(to_u(np.asarray(result.x, float), sc), float)
            U = rng.multivariate_normal(u0, np.asarray(cov_u), size=n, method="eigh")
            U[0] = u0  # the first draw is the estimates themselves
            X = np.asarray(jax.vmap(lambda u: to_x(u, sc))(jnp.asarray(U)))
            kind = APPROXIMATE
    E = rng.standard_normal((len(X), 1, q)) if between_units else None

    # ---- one build of the model for these times and this regimen
    from .entries import to_tenqsp

    m = problem._m
    fixed = to_tenqsp({k: v for k, v in cond._fixed.items() if k in m.params}, m.tables)
    scenario = tq.Scenario(times=tuple(map(float, times)), params=fixed,
                           **({} if regimen is None else {"regimen": regimen}))  # fmt: skip
    opts = problem._options
    sim_opts = {"rtol": float(opts.get("rtol", 1e-6))}
    if opts.get("atol") is not None:
        sim_opts["atol"] = float(opts["atol"])
    fs = model.simulator(outputs=outputs, **sim_opts).function([scenario], wrt=cond._wrt_arg)

    n_units = len(cond.units)
    covs = {k: jnp.full(n_units, v) for k, v in covariates.items()} or None

    def typical(x):
        return fs(cond._model_params(x, None, covs)[:1])

    def new_unit(x, e):
        Z = jnp.broadcast_to(e @ jnp.linalg.cholesky(obj._R(x)).T, (n_units, q))
        return fs(cond._model_params(x, Z, covs)[:1])

    runs = {"typical": jax.jit(jax.vmap(typical))}
    if between_units:
        runs["new"] = jax.jit(jax.vmap(new_unit))
    curves = {}
    for key, fn in runs.items():
        parts = []
        for a in range(0, len(X), CHUNK):
            xs = jnp.asarray(X[a : a + CHUNK])
            out = fn(xs) if key == "typical" else fn(xs, jnp.asarray(E[a : a + CHUNK]))
            parts.append({o: np.asarray(out[o])[:, 0, :] for o in outputs})
        curves[key] = {o: np.concatenate([p[o] for p in parts]) for o in outputs}

    # ---- a run that failed (values far out in the tails) is left out, and counted
    ran = np.ones(len(X), bool)
    for c in curves.values():
        for v in c.values():
            ran &= np.all(np.isfinite(v), axis=1)
    failed = int((~ran).sum())
    if failed == len(X):
        raise RuntimeError(f"predict: the model failed for all {len(X)} draws.")

    rows = []
    for o in outputs:
        f = curves["typical"][o][ran]
        cols = {"median": np.median(f, axis=0), "2.5 %": np.percentile(f, 2.5, axis=0),
                "97.5 %": np.percentile(f, 97.5, axis=0)}  # fmt: skip
        if between_units:
            g = curves["new"][o][ran]
            cols["new unit 5 %"] = np.percentile(g, 5, axis=0)
            cols["new unit 95 %"] = np.percentile(g, 95, axis=0)
        rows.append(pd.DataFrame({"time": times, "output": o, **cols}))
    table = pd.concat(rows, ignore_index=True)
    label = name or "prediction"
    pr = Prediction(name=label, kind=kind, table=table, n=len(X), failed=failed,
                    between_units=between_units, regimen=_regimen_text(regimen),
                    covariates=covariates)  # fmt: skip
    if name is not None:
        result.__dict__.setdefault("_predictions", {})[name] = pr
    return pr


def _covariates(problem, given) -> dict:
    """The covariate values to predict for: one number for each covariate the fit uses."""
    from .errors import nearest

    needed = list(problem._covariates())
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise TypeError(f"predict: covariates is a dict of values, e.g. {{'WT': 70}}, got "
                        f"{type(given).__name__}.")  # fmt: skip
    if not needed and given:
        raise ValueError("predict: the fit uses no covariates, so covariates= has nothing to set.")
    for k in given:
        if k not in needed:
            raise ValueError(f"predict: '{k}' is not a covariate of the fit.{nearest(k, needed)} "
                             f"Its covariates: {', '.join(needed)}.")  # fmt: skip
    missing = [k for k in needed if k not in given]
    if missing:
        example = ", ".join(f"'{k}': ..." for k in needed)
        raise ValueError(f"predict: the fit uses the covariate{'s' if len(needed) > 1 else ''} "
                         f"{', '.join(needed)}; give the value{'s' if len(needed) > 1 else ''} "
                         f"to predict for: covariates={{{example}}} "
                         f"(missing: {', '.join(missing)}).")  # fmt: skip
    out = {}
    for k in needed:
        v = given[k]
        if isinstance(v, bool) or not isinstance(v, int | float | np.number) or not np.isfinite(v):
            raise ValueError(f"predict: covariate {k} is one finite number, got {v!r}.")
        out[k] = float(v)
    return out


def _regimen_text(regimen) -> str:
    if regimen is None:
        return "the model's own doses"
    lines = [ln.strip() for ln in repr(regimen).splitlines()]
    text = "; ".join(ln for ln in lines if ln and not ln.startswith("Regimen"))
    return text if len(text) <= 120 else text[:117] + "..."
