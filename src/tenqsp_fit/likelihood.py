# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The likelihood of a problem: ``tf.objective(problem)``, the JAX function of the estimated
quantities whose value is -2 log-likelihood of all the data.

Every unit is one TenQSP scenario, and all units run as one function (``sim.function`` over
the list of scenarios: one build, one ``jax.vmap`` call). For each link, the prediction is the
link's left side (``conc``, ``log10(bact)``) at the times of its measurements, and each
measurement adds

- ``((DV - f)/sd)^2 + 2 log(sd) + log(2 pi)``, with ``sd`` from the error model (``f`` the
  prediction on the link's scale), or
- for a censored link and ``DV`` at or below the limit (M3, Beal 2001):
  ``-2 log Phi((limit - f)/sd)``, the probability that the value is below the limit.

**Predictions near zero** (PRD F2.7). A population driven extinct, or solver round-off below
``atol`` (which can be slightly negative), would make ``log10(bact)`` NaN or minus infinity.
Inside a transformed link the observables are taken through the smooth floor
``(v + sqrt(v^2 + 4 eps^2))/2`` (``v`` for ``v >> eps``, ``eps`` at 0, above 0 and rising
with ``v`` everywhere, so the residual stays finite and its gradient points back), with
``eps`` the solver's absolute tolerance: below it a prediction is round-off. The prediction
``f`` in a proportional, combined or power error model is taken as ``sqrt(f^2 + eps^2)``, so
``sd`` stays above 0.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import tenqsp as tq
from jax.scipy.stats import norm

from .distributions import Fixed
from .errors import ProblemError

__all__ = ["Objective", "objective"]

ATOL = 1e-6  # TenQSP's default absolute tolerance
_LOG_2PI = math.log(2 * math.pi)


def _floor(v, eps):
    return 0.5 * (v + jnp.sqrt(v * v + 4.0 * eps * eps))


class Objective:
    """-2 log-likelihood of a problem as a JAX function of its estimated quantities:
    ``obj = tf.objective(problem)``, then ``obj(x)`` with ``x`` in ``obj.names`` order
    (``obj.x0``: the starts). ``jax.grad``, ``jax.jit`` and ``jax.jacfwd`` work on it. A run
    that fails gives NaN; ``obj.status(x)`` gives one status per unit. ``obj.bounds``: the
    bounds of each quantity (``(-inf, inf)`` when none is given); ``obj.floor``: the level
    below which predictions are floored (see the module notes)."""

    def __init__(self, problem):
        from .problem import _Estimate, _Fix, _key

        problem.check()
        varying = problem._varying()
        if varying:
            raise NotImplementedError(
                f"tf.objective: {', '.join(d.name for d in varying)} vary between units; the "
                "objective of such a problem (method laplace) is not available yet."
            )
        m = problem._m
        self.problem = problem
        self.names = tuple(problem.estimated())
        index = {n: i for i, n in enumerate(self.names)}
        units = problem.units()
        self.units = tuple(u.id for u in units)
        opts = problem._options
        self.floor = float(opts.get("atol") or ATOL)

        # ---- model parameters: estimated, fixed, or set by a formula of the fit
        params = [d for d in problem._params if isinstance(d, _Fix) or d.dist is None]
        fixed = {d.name: d.value for d in params if isinstance(d, _Fix)}
        starts = {d.name: d.start for d in params if isinstance(d, _Estimate)}
        self.x0 = np.array([self._start(n, starts, problem) for n in self.names])
        self.bounds = tuple(
            next(
                (d.bounds for d in params if isinstance(d, _Estimate) and d.name == n and d.bounds),
                (-np.inf, np.inf),
            )
            for n in self.names
        )
        defined = {d.name: d.formula for d in problem._defines}
        wrt = [n for n in m.params if (n in starts) or (n in defined)]
        missing = [
            n for n, p in m.params.items() if p.value is None and n not in wrt and n not in fixed
        ]
        if missing:
            raise ProblemError(
                [
                    f"the model gives no value to {', '.join(missing)}; estimate or fix "
                    f"{'it' if len(missing) == 1 else 'them'} (estimate {missing[0]} = <start>)."
                ]
            )
        model_fixed = {n: v for n, v in fixed.items() if n in m.params}
        scenarios = [
            u.scenario.replace(params=model_fixed) if model_fixed else u.scenario for u in units
        ]
        covs = {
            c: jnp.asarray([float(u.covariates[c]) for u in units]) for c in problem._covariates()
        }
        base = {n: p.value for n, p in m.params.items() if isinstance(p.value, int | float)}
        self._wrt = tuple(wrt)

        # values of a model parameter for every unit, from x: (units,)
        fit_values = {**{n: ("x", index[n]) for n in starts if n in index}, **fixed}
        formulas = []  # (target, jax function, names) in declaration order
        for d in problem._defines:
            f = tq.formula(d.formula)
            formulas.append((d.name, f.jax(list(f.names)), f.names))
        n_units = len(units)

        def model_params(x):
            env = {}
            for n, v in base.items():
                env[n] = jnp.full(n_units, float(v))
            for n, v in fit_values.items():
                env[n] = jnp.full(n_units, x[v[1]] if isinstance(v, tuple) else float(v))
            env.update(covs)
            for target, fn, names in formulas:
                env[target] = jnp.broadcast_to(fn(*(env[k] for k in names)), (n_units,))
            return jnp.stack([env[n] for n in wrt], axis=1)

        self._model_params = model_params

        # ---- the batched simulation
        outputs = problem._outputs()
        sim_opts = {"rtol": float(opts.get("rtol", 1e-6))}
        if opts.get("atol") is not None:
            sim_opts["atol"] = float(opts["atol"])
        sim = problem.model.simulator(outputs=list(outputs), **sim_opts)
        self._fs = sim.function(scenarios, wrt=list(wrt) or [next(iter(m.params))])
        if not wrt:  # nothing reaches the model: still one function, at its own values
            self._model_params = lambda x: jnp.asarray(self._fs.x0)

        # ---- links: predictions at the measurements, data, censoring, error model
        errors = {_key(e.lhs): e.model for e in problem._errors}
        links = []
        for lhs, _, _ in problem._plan().links:
            f = tq.formula(lhs)
            names = list(f.names)
            transformed = "".join(lhs.split()) not in outputs
            n_max = max((len(u.observations[lhs].times) for u in units if lhs in u.observations),
                        default=0)  # fmt: skip
            idx = np.zeros((n_units, n_max), np.int64)
            dv = np.zeros((n_units, n_max))
            valid = np.zeros((n_units, n_max), bool)
            limit = np.full((n_units, n_max), np.nan)
            for i, u in enumerate(units):
                o = u.observations.get(lhs)
                if o is None:
                    continue
                k = len(o.times)
                idx[i, :k] = np.searchsorted(np.asarray(u.scenario.times), o.times)
                dv[i, :k] = o.values
                valid[i, :k] = True
                if o.below is not None:
                    limit[i, :k] = o.below
            censored = valid & np.isfinite(limit) & (dv <= np.where(np.isfinite(limit), limit, 0))
            model = errors[_key(lhs)]
            sd_args = [
                ("const", float(a.value)) if isinstance(a, Fixed) else ("x", index[f"{lhs}.{q}"])
                for q, a in zip(model.quantities, model.args, strict=True)
            ]
            links.append(
                dict(
                    lhs=lhs,
                    fn=f.jax(names),
                    names=names,
                    transformed=transformed,
                    idx=jnp.asarray(idx),
                    dv=jnp.asarray(dv),
                    obs=jnp.asarray(valid & ~censored),
                    cens=jnp.asarray(censored),
                    limit=jnp.asarray(np.where(censored, limit, 0.0)),
                    error=model.name,
                    sd_args=sd_args,
                    n=int(valid.sum()),
                    n_censored=int(censored.sum()),
                )
            )
        self._links = links
        self.n_observations = sum(lk["n"] for lk in links)
        self.n_censored = sum(lk["n_censored"] for lk in links)
        self._jit = jax.jit(self._minus2ll)

    @staticmethod
    def _start(name, starts, problem):
        if name in starts:
            return starts[name]
        for e in problem._errors:  # an error model's quantity: lhs.additive, ...
            for q, a in zip(e.model.quantities, e.model.args, strict=True):
                if f"{e.lhs}.{q}" == name:
                    return float(a)
        raise AssertionError(name)  # pragma: no cover - estimated() lists only these

    # ------------------------------------------------------------------ the function
    def predictions(self, x) -> dict:
        """Each link's prediction at its measurements: ``{lhs: (units, measurements)}``, on
        the link's scale (0 where a unit has fewer measurements)."""
        out = self._fs(self._model_params(x))
        result = {}
        for lk in self._links:
            vals = [out[n] for n in lk["names"]]
            if lk["transformed"]:
                vals = [_floor(v, self.floor) for v in vals]
            pred = lk["fn"](*vals)  # (units, times)
            safe = jnp.where(jnp.asarray(self._fs.mask), pred, 0.0)
            result[lk["lhs"]] = jnp.take_along_axis(safe, lk["idx"], axis=1)
        return result

    def _sd(self, lk, x, f):
        a = [v if kind == "const" else x[v] for kind, v in lk["sd_args"]]
        g = jnp.sqrt(f * f + self.floor**2)
        if lk["error"] == "additive":
            return jnp.broadcast_to(a[0], f.shape)
        if lk["error"] == "proportional":
            return a[0] * g
        if lk["error"] == "combined":
            return jnp.sqrt(a[0] ** 2 + (a[1] * g) ** 2)
        return a[0] * g ** a[1]  # power

    def terms(self, x) -> dict:
        """Each measurement's contribution to -2 log-likelihood: ``{lhs: (units,
        measurements)}`` (0 for padding)."""
        x = jnp.asarray(x, dtype=jnp.float64)
        preds = self.predictions(x)
        out = {}
        for lk in self._links:
            f = preds[lk["lhs"]]
            sd = self._sd(lk, x, f)
            z = (lk["dv"] - f) / sd
            normal = z * z + 2.0 * jnp.log(sd) + _LOG_2PI
            m3 = -2.0 * norm.logcdf((lk["limit"] - f) / sd)
            out[lk["lhs"]] = jnp.where(lk["obs"], normal, 0.0) + jnp.where(lk["cens"], m3, 0.0)
        return out

    def _minus2ll(self, x):
        return sum(jnp.sum(t) for t in self.terms(x).values())

    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape != (len(self.names),):
            raise ValueError(f"x has {len(self.names)} entries ({list(self.names)}).")
        return self._jit(x)

    def status(self, x) -> dict:
        """``{unit: "ok" or why its run failed}``."""
        st = self._fs.status(self._model_params(jnp.asarray(x, dtype=jnp.float64)))
        return dict(zip(self.units, st, strict=True))

    def unpack(self, x) -> dict[str, float]:
        """``x`` by name."""
        return dict(zip(self.names, (float(v) for v in np.asarray(x)), strict=True))

    def __repr__(self) -> str:
        cens = f", {self.n_censored} censored" if self.n_censored else ""
        return (
            f"<tenqsp_fit objective (-2 log-likelihood) of {list(self.names)}: "
            f"{len(self.units)} unit{'s' * (len(self.units) != 1)}, "
            f"{self.n_observations} measurements{cens}>"
        )


def objective(problem) -> Objective:
    """-2 log-likelihood of ``problem`` as a JAX function of its estimated quantities (see
    ``Objective``): for an optimizer or sampler of your own."""
    return Objective(problem)
