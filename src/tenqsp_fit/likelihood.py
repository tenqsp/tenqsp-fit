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
- for a censored measurement (M3, Beal 2001; which are censored: ``Problem.censor``):
  ``-2 log Phi((limit - f)/sd)``, the probability that the value is below the limit.

So -2LL = sum over measured values of ``((DV - f)/sd)^2 + 2 log(sd)``, plus
``n log(2 pi)`` with ``n`` the number of measured (not censored) values, plus the M3 terms.
``n log(2 pi)`` is a constant of the data: it does not depend on the parameters, so it changes
no estimate, standard error or likelihood-ratio test on the same data. Monolix reports -2LL
with it; NONMEM's objective function value (OFV) leaves it out: OFV = -2LL - n log(2 pi).

**Predictions near zero.** A population driven extinct, or solver round-off below
``atol`` (which can be slightly negative), would make ``log10(bact)`` NaN or minus infinity.
In a link whose formula needs its observables above 0 (a log, a root, logit, a power that
is not a whole number of 0 or more, a division by them; ``needs_floor``) they are taken
through the smooth floor
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

from . import entries as E
from .distributions import Fixed, quantile
from .errors import ProblemError

__all__ = ["Objective", "objective"]

ATOL = 1e-6  # TenQSP's default absolute tolerance
# a limit on each solve's steps between two doses or events (TenQSP's max_steps): this many
# times the most a unit takes at the starts, but not below STEP_FLOOR; TenQSP's own 100 000
# above that
STEP_FACTOR, STEP_FLOOR, STEP_MAX = 50, 5000, 100_000
_LOG_2PI = math.log(2 * math.pi)


def _floor(v, eps):
    return 0.5 * (v + jnp.sqrt(v * v + 4.0 * eps * eps))


_NEEDS_POSITIVE = {"log", "log10", "log2", "sqrt", "logit"}


def needs_floor(lhs: str) -> bool:
    """Whether a link's formula needs its observables above 0: it takes a log, a root or a
    logit of them, a power that is not a whole number of 0 or more, or divides by them. Other
    formulas (``2*E``, ``conc*1e9``, ``E - E0``) read the observables as they are, negative
    values included."""
    import ast

    tree = ast.parse(lhs.replace("^", "**"), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _NEEDS_POSITIVE:
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            e = node.right
            whole = isinstance(e, ast.Constant) and float(e.value).is_integer() and e.value >= 0
            if not whole:
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if any(isinstance(n, ast.Name) for n in ast.walk(node.right)):
                return True
    return False


class Objective:
    """-2 log-likelihood of a problem as a JAX function of its estimated quantities:
    ``obj = tf.objective(problem)``, then ``obj(x)`` with ``x`` in ``obj.names`` order
    (``obj.x0``: the starts). ``jax.grad``, ``jax.jit`` and ``jax.jacfwd`` work on it. A run
    that fails gives NaN; ``obj.status(x)`` gives one status per unit. ``obj.bounds``: the
    bounds of each quantity (``(-inf, inf)`` when none is given); ``obj.floor``: the level
    below which predictions are floored (see the module notes); ``obj.max_steps``: the limit
    on each solve's steps between two doses or events, 50 times the most a unit takes at
    the starts (at least 5000, at most TenQSP's 100 000): a run that needs more fails."""

    def __init__(self, problem, *, _population: bool = False):
        from .problem import _Estimate, _Fix, _key

        problem.check()
        varying = problem._varying()
        if varying and not _population:
            raise TypeError(  # pragma: no cover (tf.objective gives the Laplace objective)
                "this problem has parameters that vary between units; use tf.objective."
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
        _, self.x0, bounds = problem._starts()
        self.bounds = tuple(bounds)
        self.kinds = problem._kinds(self.names)
        defined = {d.name: d.formula for d in problem._defines}
        vary = {d.name for d in varying}
        wrt = [n for n in m.params if (n in starts) or (n in defined) or (n in vary)]
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
        model_fixed = E.to_tenqsp({n: v for n, v in fixed.items() if n in m.params}, m.tables)
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
            f = E.formula(d.formula)
            formulas.append((d.name, f.jax(list(f.names)), f.names))
        n_units = len(units)
        # parameters that vary between units: a unit's value is the distribution's quantile
        # at its normal score z (one column of Z per parameter)
        self.varying = tuple((d.name, d.dist) for d in varying)
        spec = []
        for d in varying:
            args = []
            for k, v in d.dist.args:
                if isinstance(v, Fixed):
                    args.append((k, "const", float(v.value)))
                elif isinstance(v, str):
                    f = E.formula(v)
                    args.append((k, "formula", (f.jax(list(f.names)), f.names)))
                else:
                    args.append((k, "x", index[f"{d.name}.{k}"]))
            spec.append((d.name, d.dist.name, args))

        def values(x, Z=None, covariates=None):
            """Every quantity per unit: model parameters, one-value and varying quantities,
            covariates (the data's, or ``covariates``: {name: (units,)}), formulas."""
            env = {}
            for n, v in base.items():
                env[n] = jnp.full(n_units, float(v))
            for n, v in fit_values.items():
                env[n] = jnp.full(n_units, x[v[1]] if isinstance(v, tuple) else float(v))
            env.update(covs if covariates is None else covariates)
            for k, (name, dist, args) in enumerate(spec):
                a = {}
                for key, kind, v in args:
                    if kind == "const":
                        a[key] = v
                    elif kind == "x":
                        a[key] = x[v]
                    else:
                        fn, names = v
                        a[key] = fn(*(env[n] for n in names))
                z = jnp.zeros(n_units) if Z is None else Z[:, k]
                env[name] = jnp.broadcast_to(quantile(dist, a, z), (n_units,))
            for target, fn, names in formulas:
                env[target] = jnp.broadcast_to(fn(*(env[k] for k in names)), (n_units,))
            return env

        def model_params(x, Z=None, covariates=None):
            env = values(x, Z, covariates)
            return jnp.stack([env[n] for n in wrt], axis=1)

        self._unit_values = values

        self._model_params = model_params

        # ---- the batched simulation
        outputs = problem._outputs()
        sim_opts = {"rtol": float(opts.get("rtol", 1e-6))}
        if opts.get("atol") is not None:
            sim_opts["atol"] = float(opts["atol"])
        sim = problem.model.simulator(outputs=list(outputs), **sim_opts)
        self._sim, self._scenarios, self._units = sim, scenarios, units
        # the tolerances of every compiled call, as an input (so the doctor reruns a compiled
        # function at tighter ones without a new compile): the problem's, and ``_tol`` those
        # of the call being traced (``_tolerant``)
        self._tol0 = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in sim_opts.items()}
        self._tol = None
        self._wrt_arg = list(wrt) or [next(iter(m.params))]
        self._fs = sim.function(scenarios, wrt=self._wrt_arg)
        if not wrt:  # nothing reaches the model: still one function, at its own values
            self._model_params = lambda x, Z=None, covariates=None: jnp.asarray(self._fs.x0)
        # a trial point far out of range (an optimizer's k = 1e133) can need millions of
        # steps: limit each solve to STEP_FACTOR times the work at the starts, so such a run
        # fails (NaN, a rejected point) in seconds instead of running for hours
        most = max(self._fs.steps(self._model_params(jnp.asarray(self.x0, dtype=jnp.float64))))
        self.max_steps = min(STEP_MAX, max(STEP_FLOOR, STEP_FACTOR * most))
        if self.max_steps < STEP_MAX:
            sim = problem.model.simulator(
                outputs=list(outputs), max_steps=self.max_steps, **sim_opts
            )
            self._sim, self._fs = sim, sim.function(scenarios, wrt=self._wrt_arg)

        # ---- links: predictions at the measurements, data, censoring, error model
        errors = {_key(e.lhs): e.model for e in problem._errors}
        written = {_key(w.lhs): w.density for w in problem._likelihoods}
        self.written = bool(written)
        self._fixed = {k: float(v) for k, v in fixed.items()}
        self._base = {k: float(v) for k, v in base.items()}
        self._index = index
        links = []
        for lhs, _, _ in problem._plan().links:
            f = tq.formula(lhs)
            names = list(f.names)
            transformed = "".join(lhs.split()) not in outputs and needs_floor(lhs)
            n_max = max((len(u.observations[lhs].times) for u in units if lhs in u.observations),
                        default=0)  # fmt: skip
            idx = np.zeros((n_units, n_max), np.int64)
            dv = np.zeros((n_units, n_max))
            valid = np.zeros((n_units, n_max), bool)
            limit = np.full((n_units, n_max), np.nan)
            cens = np.zeros((n_units, n_max), bool)
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
                    cens[i, :k] = o.censored
            censored = valid & cens
            dv = np.where(censored, 0.0, dv)  # a censored value is not read (NaN for text)
            density = written.get(_key(lhs))
            model = errors.get(_key(lhs))
            sd_args = (
                []
                if model is None
                else [
                    ("const", float(a.value))
                    if isinstance(a, Fixed)
                    else ("const", float(fixed[a]))
                    if isinstance(a, str) and a in fixed
                    else ("x", index[a if isinstance(a, str) else f"{lhs}.{q}"])
                    for q, a in zip(model.quantities, model.args, strict=True)
                ]
            )
            dens = None
            if isinstance(density, str):
                df_ = E.formula(density)
                dens = ("formula", df_.jax(list(df_.names)), df_.names)
            elif density is not None:
                dens = ("function", density, ())
            links.append(
                dict(
                    lhs=lhs,
                    fn=f.jax(names),
                    names=names,
                    transformed=transformed and dens is None,
                    density=dens,
                    idx=jnp.asarray(idx),
                    dv=jnp.asarray(dv),
                    obs=jnp.asarray(valid & ~censored),
                    cens=jnp.asarray(censored),
                    limit=jnp.asarray(np.where(censored, limit, 0.0)),
                    error=None if model is None else model.name,
                    sd_args=sd_args,
                    n=int(valid.sum()),
                    n_censored=int(censored.sum()),
                )
            )
        self._links = links
        self.n_observations = sum(lk["n"] for lk in links)
        self.n_censored = sum(lk["n_censored"] for lk in links)
        jitted = jax.jit(self._tolerant(self._minus2ll))
        self._jit = lambda x, tol=None: jitted(x, tol=self._tol0 if tol is None else tol)

    # ------------------------------------------------------------------ the function
    def _run(self, P):
        """The batched simulation at the units' model parameters ``P``, at the tolerances of
        the call being traced (the problem's outside ``_tolerant``)."""
        return self._fs(P, **(self._tol or {}))

    def _tolerant(self, fn):
        """``fn`` with one more keyword input, ``tol`` ({"rtol": ..., "atol": ...} as
        arrays): every simulation inside runs at those tolerances. For ``jax.jit``: a call
        at other tolerances reuses the compiled function."""

        def at(*args, tol):
            before, self._tol = self._tol, tol
            try:
                return fn(*args)
            finally:
                self._tol = before

        return at

    def tighter(self, factor: float) -> dict:
        """The problem's tolerances divided by ``factor`` (atol only when the problem sets
        it), in the form ``tol`` takes."""
        return {k: v / factor for k, v in self._tol0.items()}

    def predictions(self, x) -> dict:
        """Each link's prediction at its measurements: ``{lhs: (units, measurements)}``, on
        the link's scale (0 where a unit has fewer measurements)."""
        return self._predictions(x, self._run(self._model_params(x)))

    def _predictions(self, x, out) -> dict:
        result = {}
        for lk in self._links:
            vals = [out[n] for n in lk["names"]]
            if lk["transformed"]:
                vals = [_floor(v, self.floor) for v in vals]
            pred = lk["fn"](*vals)  # (units, times)
            safe = jnp.where(jnp.asarray(self._fs.mask), pred, 0.0)
            result[lk["lhs"]] = jnp.take_along_axis(safe, lk["idx"], axis=1)
        return result

    def curves(self, x, times, Z=None) -> dict:
        """Each link's prediction at ``times`` (one grid, the same for every unit), for
        every unit: ``{lhs: (units, times)}``, on the link's scale; ``Z``: the units' normal
        scores (None: the typical unit, scores 0). One more build of the model (the times
        are new)."""
        times = tuple(float(t) for t in np.asarray(times))
        fs = self._sim.function([s.replace(times=times) for s in self._scenarios],
                                wrt=self._wrt_arg)  # fmt: skip
        x = jnp.asarray(x, dtype=jnp.float64)
        P = self._model_params(x, None if Z is None else jnp.asarray(Z))
        out = fs(P)
        result = {}
        for lk in self._links:
            vals = [out[n] for n in lk["names"]]
            if lk["transformed"]:
                vals = [_floor(v, self.floor) for v in vals]
            result[lk["lhs"]] = np.asarray(lk["fn"](*vals))
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

    def _values(self, x) -> dict:
        """Every estimated and fixed quantity and model parameter by name (for a written
        likelihood)."""
        return {**self._base, **self._fixed, **{n: x[i] for n, i in self._index.items()}}

    def terms(self, x) -> dict:
        """Each measurement's contribution to -2 log-likelihood: ``{lhs: (units,
        measurements)}`` (0 for padding)."""
        x = jnp.asarray(x, dtype=jnp.float64)
        return self._terms(x, self._run(self._model_params(x)))

    def _link_terms(self, lk, x, f):
        """-2 log-density of each measurement of a link given its prediction ``f`` (an error
        model; 0 for padding)."""
        sd = self._sd(lk, x, f)
        z = (lk["dv"] - f) / sd
        normal = z * z + 2.0 * jnp.log(sd) + _LOG_2PI
        m3 = -2.0 * norm.logcdf((lk["limit"] - f) / sd)
        return jnp.where(lk["obs"], normal, 0.0) + jnp.where(lk["cens"], m3, 0.0)

    def _link_local(self, lk, x, f):
        """For the Laplace approximation, per measurement: -2 log-density, its derivative by
        the prediction, and the Fisher information of the prediction in -log-density
        (measured: 1/sd^2 + 2 (d log sd/df)^2; censored: the Gauss-Newton term of
        sqrt(-2 log Phi))."""
        ones = jnp.ones_like(f)
        term, dterm = jax.jvp(lambda g: self._link_terms(lk, x, g), (f,), (ones,))
        _, dlogsd = jax.jvp(lambda g: jnp.log(self._sd(lk, x, g)), (f,), (ones,))
        sd = self._sd(lk, x, f)
        r = jnp.sqrt(jnp.maximum(term, 1e-300))
        info = jnp.where(lk["obs"], 1.0 / sd**2 + 2.0 * dlogsd**2, 0.0) + jnp.where(
            lk["cens"], (dterm / (2.0 * r)) ** 2, 0.0
        )
        return term, dterm, info

    def _terms(self, x, sim) -> dict:
        preds = self._predictions(x, sim)
        out = {}
        mask = jnp.asarray(self._fs.mask)
        for lk in self._links:
            f = preds[lk["lhs"]]
            if lk["density"] is not None:
                kind, fn, names = lk["density"]
                values = self._values(x)
                if kind == "formula":
                    at = {
                        n: jnp.take_along_axis(jnp.where(mask, sim[n], 0.0), lk["idx"], axis=1)
                        for n in lk["names"]
                    }
                    args = [
                        lk["dv"] if n == "DV" else at[n] if n in at else values[n] for n in names
                    ]
                    logd = fn(*args)
                else:
                    logd = fn(lk["dv"], f, values)
                out[lk["lhs"]] = jnp.where(lk["obs"], -2.0 * logd, 0.0)
                continue
            out[lk["lhs"]] = self._link_terms(lk, x, f)
        return out

    @property
    def sum_of_squares(self) -> bool:
        """Whether -2LL is a sum of squares up to a constant of the data: every link has an
        additive error of known sd. An estimated sd, an sd that changes with the prediction
        (proportional, combined, power: ``2 log sd`` changes with the parameters) or a
        written likelihood make it not one."""
        return all(
            lk["density"] is None
            and lk["error"] == "additive"
            and all(kind == "const" for kind, _ in lk["sd_args"])
            for lk in self._links
        )

    def _residuals(self, x):
        """-2LL as a sum of squares, up to a constant of the data, when ``sum_of_squares``:
        ``(DV - f)/sd`` per measured value, ``sqrt(-2 log Phi(...))`` per censored one.
        Flat, padding 0."""
        x = jnp.asarray(x, dtype=jnp.float64)
        preds = self.predictions(x)
        parts = []
        for lk in self._links:
            f = preds[lk["lhs"]]
            sd = self._sd(lk, x, f)
            z = jnp.where(lk["obs"], (lk["dv"] - f) / sd, 0.0)
            m3 = -2.0 * norm.logcdf((lk["limit"] - f) / sd)
            m3 = jnp.where(lk["cens"], jnp.sqrt(jnp.maximum(m3, 1e-300)), 0.0)
            parts += [z.ravel(), m3.ravel()]
        return jnp.concatenate(parts)

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


def objective(problem):
    """-2 log-likelihood of ``problem`` as a JAX function of its estimated quantities (see
    ``Objective``): for an optimizer or sampler of your own. When parameters vary between
    units, the Laplace approximation of it (see ``tenqsp_fit.laplace.Laplace``)."""
    if problem._varying():
        from .laplace import Laplace

        return Laplace(problem)
    return Objective(problem)
