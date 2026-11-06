# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``method nuts``: draws from the posterior by the No-U-Turn Sampler (NumPyro's).

**What is sampled.** The estimated quantities on their estimation scales (log, logit, as is;
``u``) and, when parameters vary between units, each unit's standard-normal scores ``E``
(non-centred: the scores of ``laplace`` are ``Z = E L'``, ``L L' = R`` the correlation matrix
of the ``correlate`` lines). The potential (minus the log posterior) is

    U = -2LL/2 + |E|^2/2 - log prior(x) - log |dx/du|,

with -2LL that of ``tf.objective`` given the units' values (the data, error models, M3), the
priors written or by default (``priors``), and the Jacobian of the estimation scale.

**How a run goes.** (1) The fast fit (``laplace`` when parameters vary, else ``optimize``), with
the written priors. (2) The curvature of U there (central differences of its exact gradient)
gives NUTS the shape of the posterior: its inverse is the mass matrix, held through the run;
the chains start one curvature-SD away from the fast fit in random directions. Without this,
NUTS learns the shape during warmup, in windows too short for a dense matrix: on theophylline
(CL and ke move together, 7 times narrower together than apart) it took 15 times more steps
per draw. (3) Warmup tunes the step size (dual averaging, target acceptance 0.8). (4) The
draws. All chains run as one batched (``jax.vmap``) computation, in chunks of ``CHUNK`` draws,
between which the progress is shown and a stop (Ctrl-C) keeps the draws so far.

**Convergence**, per quantity and chain: rank-normalized split R-hat, bulk and tail
effective sample sizes (ArviZ), divergences after warmup, energy BFMI per chain, and how often
a trajectory reached the maximum tree depth.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .live import default_progress
from .optimize import Result, scales, to_u, to_x
from .priors import lkj_log_density, priors
from .problem import seconds

__all__ = ["PosteriorResult", "fit_nuts"]

CHUNK = 10  # draws per chain between two progress updates
TARGET_ACCEPT = 0.8
MAX_TREE_DEPTH = 10
FD_HESS = 1e-4  # central differences of the gradient, for the curvature at the fast fit
RHAT_MAX, ESS_MIN, BFMI_MIN = 1.01, 400, 0.3


def _log_jacobian(u, sc):
    """log |dx/du| of the estimation scales."""
    parts = []
    for i, s in enumerate(sc):
        if s[0] in ("log", "above", "below"):
            parts.append(u[i])
        elif s[0] == "logit":
            parts.append(math.log(s[2] - s[1]) + jax.nn.log_sigmoid(u[i])
                         + jax.nn.log_sigmoid(-u[i]))  # fmt: skip
    return sum(parts) if parts else 0.0


class Posterior:
    """The potential of a problem (minus its log posterior) over the flat vector
    ``[u, E.ravel()]``, and the maps back to the quantities and the units' scores."""

    def __init__(self, problem, lap, cond):
        self.problem, self.lap, self.cond = problem, lap, cond
        self.names = cond.names
        self.x0 = np.asarray(cond.x0, float)
        self.sc = scales(self.names, self.x0, cond.bounds, cond.kinds)
        self.p = len(self.names)
        self.n = len(cond.units)
        self.q = len(cond.varying)
        self.priors, problems = priors(problem, self.names, self.x0, self.sc, defaults=True)
        if problems:  # pragma: no cover (problem.check() reports them first)
            raise ValueError("; ".join(problems))
        index = {n: i for i, n in enumerate(self.names)}
        self._at = [(index[p.name], p) for p in self.priors]
        position = {name: k for k, (name, _) in enumerate(cond.varying)}
        self._blocks = [[position[a] for a in c.names] for c in problem._correlations]
        self.d = self.p + self.n * self.q

    def split(self, flat):
        return flat[: self.p], flat[self.p :].reshape(self.n, self.q)

    def scores(self, x, E):
        """The units' scores Z (laplace's) from the standard-normal E."""
        if not self.q:
            return None
        L = jnp.linalg.cholesky(self.lap._R(x))
        return E @ L.T

    def log_prior(self, x):
        lp = sum(p.log_density(x[i]) for i, p in self._at) if self._at else 0.0
        if self._blocks:
            R = self.lap._R(x)
            for b in self._blocks:
                lp = lp + lkj_log_density(R[jnp.ix_(jnp.asarray(b), jnp.asarray(b))])
        return lp

    def m2ll(self, x, Z):
        c = self.cond
        return sum(jnp.sum(t) for t in c._terms(x, c._run(c._model_params(x, Z))).values())

    def potential(self, flat):
        u, E = self.split(flat)
        x = to_x(u, self.sc)
        Z = self.scores(x, E)
        return (0.5 * self.m2ll(x, Z) + 0.5 * jnp.sum(E * E) - self.log_prior(x)
                - _log_jacobian(u, self.sc))  # fmt: skip

    def flat(self, x, Z):
        """The flat vector at the quantities x and the units' scores Z."""
        u = np.asarray(to_u(np.asarray(x), self.sc))
        if not self.q:
            return u
        L = np.linalg.cholesky(np.asarray(self.lap._R(jnp.asarray(x))))
        E = np.linalg.solve(L, np.asarray(Z).T).T
        return np.concatenate([u, E.ravel()])


def _curvature(grad, flat0):
    """The Hessian of the potential by central differences of its gradient (batched)."""
    d = flat0.size
    h = FD_HESS
    pts = np.concatenate([flat0 + h * np.eye(d), flat0 - h * np.eye(d)])
    G = []
    for k in range(0, len(pts), 64):
        G.append(np.asarray(jax.vmap(grad)(jnp.asarray(pts[k : k + 64]))))
    G = np.concatenate(G)
    H = (G[:d] - G[d:]) / (2 * h)
    return 0.5 * (H + H.T)


def _mass(H) -> tuple[np.ndarray, str | None]:
    """The inverse mass matrix from the curvature: its inverse, with directions of no or
    negative curvature given the largest spread found (and a note)."""
    w, V = np.linalg.eigh(H)
    good = w > 0
    note = None
    if not good.all() or not np.isfinite(w).all():
        note = (
            f"the curvature at the fast fit is not positive in {int((~good).sum())} "
            "direction(s): there the sampler starts with the widest spread found"
        )
    floor = np.max(w[good]) * 1e-10 if good.any() else 1.0
    widest = np.min(w[good]) if good.any() else 1.0
    w = np.where(good & np.isfinite(w), np.maximum(w, floor), widest)
    return (V / w) @ V.T, note


def fit_nuts(problem, *, progress=None) -> PosteriorResult:
    from numpyro.infer.hmc import hmc

    from .laplace import Laplace, fit_laplace
    from .likelihood import Objective
    from .optimize import fit_optimize
    from .problem import METHOD_DEFAULTS

    opts = problem._options
    mo = {**METHOD_DEFAULTS["nuts"], **problem._method.options}
    chains, warmup, samples = int(mo["chains"]), int(mo["warmup"]), int(mo["samples"])
    seed = int(opts.get("seed", 0))
    progress = progress or opts.get("progress") or default_progress()
    t_start = time.perf_counter()
    say = (lambda text: None) if progress == "none" else (lambda text: print(text, flush=True))

    # ---- 1. the fast fit
    if problem._varying():
        lap = Laplace(problem)
        say("nuts: the starting point, by laplace (FOCE-I) with the written priors")
        start = fit_laplace(problem, lap, progress="none", with_se=False)
        cond, Z0 = lap._cond, np.asarray(start.modes)
    else:
        lap = None
        cond = Objective(problem)
        say("nuts: the starting point, by optimize with the written priors")
        start = fit_optimize(problem, cond, progress="none", with_se=False)
        Z0 = None
    post = Posterior(problem, lap, cond)
    flat0 = post.flat(start.x, Z0)
    say(f"nuts: started at -2LL {start.m2ll:.4f} ({start.status}, {start.seconds:.0f} s)")

    # ---- 2. the shape of the posterior there
    U = jax.jit(post.potential)
    grad = jax.jit(jax.grad(post.potential))
    H = _curvature(grad, flat0)
    Minv, note = _mass(H)
    notes = [note] if note else []
    rng = np.random.default_rng(seed)
    Lm = np.linalg.cholesky(Minv + 1e-12 * np.eye(post.d))
    inits = flat0[None] + (Lm @ rng.normal(size=(post.d, chains))).T
    for c in range(chains):  # a start where the model does not run: drawn again, closer
        for k in range(20):
            if np.isfinite(float(U(jnp.asarray(inits[c])))):
                break
            inits[c] = flat0 + 0.5 ** (k + 1) * (Lm @ rng.normal(size=post.d))

    # ---- 3-4. warmup and draws, all chains batched
    init_kernel, sample_kernel = hmc(potential_fn=post.potential, algo="NUTS")
    keys = jax.random.split(jax.random.PRNGKey(seed), chains)
    states = jax.vmap(
        lambda key, z: init_kernel(
            z, num_warmup=warmup, rng_key=key, dense_mass=True,
            inverse_mass_matrix=jnp.asarray(Minv), adapt_mass_matrix=False, step_size=0.5,
            target_accept_prob=TARGET_ACCEPT, max_tree_depth=MAX_TREE_DEPTH,
        )
    )(keys, jnp.asarray(inits))  # fmt: skip

    @jax.jit
    def chunk(states):
        def body(s, _):
            s = jax.vmap(sample_kernel)(s)
            return s, (s.z, s.diverging, s.num_steps, s.potential_energy, s.energy,
                       s.adapt_state.step_size)  # fmt: skip

        return jax.lax.scan(body, states, None, length=CHUNK)

    total = warmup + samples
    rec = {k: [] for k in ("z", "div", "steps", "pe", "energy", "step")}
    shown = _NutsProgress(progress, post.names, chains, warmup, total, problem.name)
    done, status, stopped_by = 0, None, None
    timeout = seconds(opts.get("timeout"))
    try:
        while done < total:
            states, out = chunk(states)
            for k, v in zip(rec, out, strict=True):
                rec[k].append(np.asarray(v))
            done += CHUNK
            shown.update(rec, post, time.perf_counter() - t_start)
            if done < total and timeout is not None and time.perf_counter() - t_start > timeout:
                from .optimize import _duration

                status = "stopped"
                stopped_by = f"the time limit (options timeout, {_duration(timeout)})"
                break
    except KeyboardInterrupt:
        status, stopped_by = "stopped", "Ctrl-C"
    # (draws, chains, ...) -> (chains, draws, ...); the last chunk may overshoot
    arr = {k: np.swapaxes(np.concatenate(v), 0, 1)[:, :total] for k, v in rec.items() if v}
    shown.update(rec, post, time.perf_counter() - t_start, last=True, status=status)
    result = _result(problem, post, start, arr, chains, warmup, samples, status, notes,
                     time.perf_counter() - t_start, Minv)  # fmt: skip
    result.stopped_by = stopped_by
    return result


class _NutsProgress:
    """A line per few seconds (or the live plot: each chain's path, the warmup's end dotted),
    with the divergences after warmup."""

    def __init__(self, mode, names, chains, warmup, total, problem_name):
        self.mode, self.names, self.chains = mode, names, chains
        self.warmup, self.total = warmup, total
        self.printed = -1e9
        self.plot = None
        if mode == "plot":
            from .live import LivePlot

            self.plot = LivePlot(names, "nuts", problem_name, value="-2 log posterior",
                                 line="chain", xlabel="draw (warmup, then draws)",
                                 warmup=warmup)  # fmt: skip

    def update(self, rec, post, seconds, last=False, status=None):
        if self.mode == "none" or not rec["z"]:
            return
        z = np.concatenate(rec["z"])  # (draws so far, chains, d)
        done = min(len(z), self.total)
        div = np.concatenate(rec["div"])[:done]
        n_div = int(div[self.warmup :].sum())
        steps = np.concatenate(rec["steps"])[max(0, done - CHUNK) : done].mean()
        phase = "warmup" if done <= self.warmup else "draws"
        clock = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
        at, of = ((min(done, self.warmup), self.warmup) if phase == "warmup"
                  else (done - self.warmup, self.total - self.warmup))  # fmt: skip
        head = (f"nuts  {self.chains} chains  {phase} {at}/{of}   divergences {n_div}   "
                f"steps/draw {steps:.0f}   {clock}")  # fmt: skip
        if self.plot is not None:
            pe = np.concatenate(rec["pe"])[:done]
            X = np.asarray(jax.vmap(jax.vmap(lambda f: to_x(f[: post.p], post.sc)))(
                jnp.asarray(z[:done])))  # fmt: skip
            rows = [(c + 1, i + 1, 2.0 * float(pe[i, c]), *X[i, c])
                    for c in range(self.chains) for i in range(done)]  # fmt: skip
            text = status or (f"{phase}, {n_div} divergences" if not last else "done")
            self.plot.update(rows, ["chain", "iteration", "-2 log posterior", *self.names],
                             text, seconds, last=last)  # fmt: skip
            if not last:
                return
        if last or seconds - self.printed >= 10:
            print(head + (f"   {status}" if status else ""), flush=True)
            self.printed = seconds


def _diagnostics(X, arr, warmup):
    from arviz_stats import bfmi, ess, rhat

    post = X[:, warmup:]  # (chains, draws, p)
    rows = []
    for i in range(X.shape[2]):
        a = post[:, :, i]
        ok = a.shape[1] >= 4
        rows.append(dict(
            rhat=float(rhat(a)) if ok and a.shape[0] > 1 else math.nan,
            ess_bulk=float(ess(a)) if ok else math.nan,
            ess_tail=float(ess(a, method="tail", prob=(0.05, 0.95))) if ok else math.nan,
        ))  # fmt: skip
    energy = arr["energy"][:, warmup:]
    per_chain = pd.DataFrame({
        "chain": np.arange(1, X.shape[0] + 1),
        "draws": [energy.shape[1]] * X.shape[0],
        "divergences": arr["div"][:, warmup:].sum(axis=1).astype(int),
        "BFMI": (np.asarray(bfmi(energy)) if energy.shape[1] > 2
                 else np.full(X.shape[0], math.nan)),
        "step size": arr["step"][:, -1],
        "max tree depth reached": (arr["steps"][:, warmup:] >= 2**MAX_TREE_DEPTH - 1).sum(axis=1),
        "mean steps per draw": arr["steps"][:, warmup:].mean(axis=1),
    })  # fmt: skip
    return rows, per_chain


def _result(problem, post, start, arr, chains, warmup, samples, status, notes, seconds, Minv):
    names = post.names
    z = arr["z"]  # (chains, all draws, d)
    n_all = z.shape[1]
    warm = min(warmup, n_all)
    X = np.asarray(jax.vmap(jax.vmap(lambda f: to_x(f[: post.p], post.sc)))(jnp.asarray(z)))
    kept = X[:, warm:]
    flat_kept = kept.reshape(-1, post.p)
    stats, per_chain = _diagnostics(X, arr, warm)
    med = np.median(flat_kept, axis=0) if len(flat_kept) else np.full(post.p, np.nan)
    modes, Zs = None, None
    if post.q and len(flat_kept):
        zk = z[:, warm:].reshape(-1, post.d)

        def scores(f):
            return post.scores(to_x(f[: post.p], post.sc), post.split(f)[1])

        Zs = np.asarray(jax.vmap(scores)(jnp.asarray(zk)))
        modes = np.median(Zs, axis=0)
        Zs = Zs.reshape(chains, kept.shape[1], post.n, post.q)
    sample_stats = {  # per chain and draw, after warmup (ArviZ's names)
        "diverging": arr["div"][:, warm:].astype(bool),
        "energy": arr["energy"][:, warm:],
        "lp": -arr["pe"][:, warm:],
        "n_steps": arr["steps"][:, warm:],
        "step_size": arr["step"][:, warm:],
    }
    draws = pd.DataFrame(
        {
            "chain": np.repeat(np.arange(1, chains + 1), kept.shape[1]),
            "draw": np.tile(np.arange(1, kept.shape[1] + 1), chains),
            **{n: kept[:, :, i].ravel() for i, n in enumerate(names)},
            "divergent": arr["div"][:, warm:].ravel().astype(bool),
        }
    )
    trace = pd.DataFrame(
        [(c + 1, i + 1, 2.0 * float(arr["pe"][c, i]), *X[c, i])
         for c in range(chains) for i in range(n_all)],
        columns=["chain", "iteration", "-2 log posterior", *names],
    )  # fmt: skip
    if status is None:
        status = "sampled"
    m2ll = float("nan")
    if np.all(np.isfinite(med)):  # -2LL as the fast fit's (laplace: over the units' values)
        m2ll = float((post.lap or post.cond)(med))
    starts = pd.DataFrame({"start": [1], "-2LL": [start.m2ll], "steps": [0],
                           "status": [start.status]})  # fmt: skip
    return PosteriorResult(
        problem=problem, method="nuts", optimizer=f"NUTS (NumPyro), {chains} chains",
        names=names, x=med, x0=post.x0, m2ll=m2ll, status=status, starts=starts,
        trace=trace, scales=post.sc, redrawn=0, failed_runs=0,
        n_observations=post.cond.n_observations, seconds=seconds, near_bounds={},
        objective=post.lap if post.lap is not None else post.cond, modes=modes,
        draws=draws, stats=stats, per_chain=per_chain, chains=chains, warmup=warmup,
        samples=samples, kept=kept.shape[1], priors=post.priors, start=start, notes=notes,
        posterior=post, inverse_mass=Minv, flat_draws=np.asarray(z[:, warm:]),
        kept_draws=kept, unit_scores=Zs, sample_stats=sample_stats,
        shrinkage={},
    )  # fmt: skip


class PosteriorResult(Result):
    """A NUTS fit: ``result.table()`` (posterior median, mean, SD, 95 % interval, R-hat,
    effective sample sizes), ``result.draws`` (every draw after warmup), ``result.convergence()``
    (per chain), ``result.plot_trace()`` (each chain's path); ``individual()``, ``plot_gof()``,
    ``plot_individual()`` and ``tables()`` at the posterior medians; ``vpc()`` with each
    simulated study from a different draw. ``result.x`` and ``result.estimates`` are the
    posterior medians."""

    def table(self) -> pd.DataFrame:
        k = self.draws[list(self.names)].to_numpy()
        lo, hi = (np.percentile(k, q, axis=0) if len(k) else np.full(len(self.names), np.nan)
                  for q in (2.5, 97.5))  # fmt: skip
        return pd.DataFrame(
            {
                "quantity": list(self.names),
                "median": self.x,
                "mean": k.mean(axis=0) if len(k) else np.nan,
                "SD": k.std(axis=0, ddof=1) if len(k) > 1 else np.nan,
                "CI 2.5 %": lo,
                "CI 97.5 %": hi,
                "R-hat": [s["rhat"] for s in self.stats],
                "ESS bulk": [s["ess_bulk"] for s in self.stats],
                "ESS tail": [s["ess_tail"] for s in self.stats],
                "prior": [self._prior_text(n) for n in self.names],
            }
        )

    def _prior_text(self, name: str) -> str:
        for p in self.priors:
            if p.name == name:
                return str(p)
        return "LKJ(2) (default)" if name.startswith("corr(") else ""

    def convergence(self) -> pd.DataFrame:
        """Per chain: draws after warmup, divergences, energy BFMI, the step size, how often
        a trajectory reached the maximum tree depth (10), and the mean steps per draw."""
        return self.per_chain.copy()

    def warnings(self) -> list[str]:
        """The convergence checks that fail."""
        out = []
        t = self.table()
        bad = t[t["R-hat"] > RHAT_MAX]
        if len(bad):
            out.append(f"R-hat above {RHAT_MAX} for {', '.join(bad.quantity)}: the chains do "
                       "not agree yet; run longer (more warmup and samples).")  # fmt: skip
        low = t[np.minimum(t["ESS bulk"], t["ESS tail"]) < ESS_MIN]
        if len(low):
            out.append(f"effective sample size below {ESS_MIN} for {', '.join(low.quantity)}: "
                       "the intervals are rough; run longer.")  # fmt: skip
        d = int(self.per_chain.divergences.sum())
        if d:
            out.append(f"{d} divergent transition(s) after warmup: the sampler missed part of "
                       "the posterior; the results may be biased (reparameterize, or give "
                       "priors to quantities the data do not determine).")  # fmt: skip
        b = self.per_chain.BFMI
        if (b < BFMI_MIN).any():
            out.append(f"energy BFMI below {BFMI_MIN} in chain(s) "
                       f"{', '.join(str(c) for c in self.per_chain.chain[b < BFMI_MIN])}: "
                       "the sampler explores the posterior's tails poorly.")  # fmt: skip
        hits = int(self.per_chain["max tree depth reached"].sum())
        if hits:
            out.append(f"{hits} draw(s) reached the maximum tree depth ({MAX_TREE_DEPTH}): "
                       "slow mixing.")  # fmt: skip
        return out

    def individual(self) -> pd.DataFrame:
        """Each unit's values at the posterior medians (of the quantities and of its
        scores)."""
        if self.modes is None:
            raise ValueError("individual values are for parameters that vary between units; "
                             "none does here.")  # fmt: skip
        return self.objective.individual(self.x, self.modes)

    def plot_trace(self):
        """Each chain's path per quantity (and -2 log posterior), warmup and draws, the
        warmup's end dotted. Returns the figure."""
        return self._draw_trace()

    def plot_waterfall(self, ax=None):
        raise ValueError("the waterfall is for multi-start fits; NUTS has chains: "
                         "result.plot_trace().")  # fmt: skip

    def to_arviz(self):
        """The posterior as ArviZ data (a DataTree): ``posterior`` (every quantity, and with
        parameters that vary between units each unit's normal score ``z(name)``),
        ``sample_stats`` (diverging, energy, lp, n_steps, step_size) and ``observed_data`` (the
        measured values per link; censored ones left out), by chain and draw after warmup."""
        import arviz as az

        posterior = {n: self.kept_draws[:, :, i] for i, n in enumerate(self.names)}
        coords, dims = {}, {}
        if self.unit_scores is not None:
            coords["unit"] = list(self.objective.units)
            for k, (name, _) in enumerate(self.objective.varying):
                posterior[f"z({name})"] = self.unit_scores[:, :, :, k]
                dims[f"z({name})"] = ["unit"]
        observed = {}
        cond = getattr(self.objective, "_cond", self.objective)
        for lk in cond._links:
            observed[lk["lhs"]] = np.asarray(lk["dv"])[np.asarray(lk["obs"])]
        return az.from_dict({"posterior": posterior, "sample_stats": self.sample_stats,
                             "observed_data": observed}, coords=coords, dims=dims,
                            name=self.problem.name)  # fmt: skip

    def _draw_trace(self, fig=None):
        from .live import draw_trace

        return draw_trace(self.trace, self.names, value="-2 log posterior", line="chain",
                          xlabel="draw (warmup, then draws)", warmup=self.warmup, fig=fig,
                          title=f"{self.problem.name}: nuts, {self.status}")  # fmt: skip

    def profile(self, name, **kw):
        raise ValueError("profile is for likelihood fits (optimize, laplace); NUTS gives each "
                         "quantity's posterior: result.table() has its 95 % interval, "
                         "result.draws every draw.")  # fmt: skip

    def __repr__(self) -> str:
        d = int(self.per_chain.divergences.sum())
        t = self.table()
        rh, es = np.nanmax(t["R-hat"]) if len(t) else np.nan, np.nanmin(
            np.minimum(t["ESS bulk"], t["ESS tail"])) if len(t) else np.nan  # fmt: skip
        units = f", {self.objective.n} units" if self.modes is not None else ""
        lines = [
            f"{self.problem.name}: method nuts, {self.status}; {self.chains} chains x "
            f"{self.kept} draws (after {self.warmup} warmup), {d} divergences, R-hat <= "
            f"{rh:.3f}, ESS >= {es:.0f}; {self.seconds:.0f} s",
            f"started from {self.start.method}: -2LL {self.start.m2ll:.4f} "
            f"({self.n_observations} measurements{units}); -2LL at the posterior medians "
            f"{self.m2ll:.4f}",
            "",
            f"{'quantity':<22}{'median':>11}{'SD':>10}   {'95 % interval':<22}{'R-hat':>7}"
            f"{'ESS':>7}  prior",
        ]
        for r in t.itertuples(index=False):
            ci = f"({r[4]:.4g}, {r[5]:.4g})"
            ess_ = min(r[7], r[8])
            lines.append(f"{r[0]:<21} {r[1]:>10.5g} {r[3]:>9.3g}   {ci:<21} {r[6]:>6.3f}"
                         f" {ess_:>6.0f}  {r[9]}")  # fmt: skip
        lines += [f"note: {n}" for n in self.notes]
        if getattr(self, "_doctor", None) is not None:  # the doctor reports the warnings
            return "\n".join(lines + self._doctor_lines())
        warns = self.warnings()
        if warns:
            lines += ["", *(f"warning: {w}" for w in warns)]
        return "\n".join(lines)
