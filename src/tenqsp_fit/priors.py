# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Priors: written (``prior CL.median ~ lognormal(median=0.04, sdlog=0.5)``) or by default;
their log-densities as JAX functions, and draws from them (the prior predictive
check).

**The defaults**, one rule by the scale a quantity is estimated on, centred on its start:

- positive (log scale): ``lognormal(median=start, sdlog=2)``, 95 % within a factor 50 of the
  start (above a lower bound ``lo`` only: the same for ``x - lo``; below ``hi``: for ``hi - x``);
- any sign: ``normal(mean=start, sd=2*|start|)``; a start of 0 gives no scale, so a prior must
  be written;
- bounded, ``in (lo, hi)`` (and a logitnormal's median, in (0, 1)): ``uniform(lower=lo,
  upper=hi)``;
- the correlations of each ``correlate`` block: LKJ(2) on the block's matrix, density
  proportional to det(R)^(2 - 1), which mildly favours small correlations.

Written priors are used by every method (``optimize`` and ``laplace`` then give the posterior
mode); the defaults by ``method nuts`` only. A written prior must give density to every value
the quantity can take (a lognormal prior is for a positive quantity); a correlation's prior is
LKJ(2) in this release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, logit
from jax.scipy.stats import norm

from .distributions import Distribution, Fixed, _num

__all__ = ["DEFAULT_SDLOG", "LKJ_ETA", "Prior", "log_density", "priors"]

DEFAULT_SDLOG = 2.0  # default prior of a positive quantity: 95 % within a factor exp(2*1.96) = 50
DEFAULT_SD_FACTOR = 2.0  # default prior of a quantity of any sign: SD twice its start
LKJ_ETA = 2.0

# where a family puts its density: the values a quantity must stay within
SUPPORT = {
    "normal": (-math.inf, math.inf),
    "lognormal": (0.0, math.inf),
    "exponential": (0.0, math.inf),
    "weibull": (0.0, math.inf),
    "gamma": (0.0, math.inf),
    "logitnormal": (0.0, 1.0),
    "beta": (0.0, 1.0),
}


def numbers(dist: Distribution) -> dict[str, float]:
    """A prior's arguments as numbers (``fix(v)`` counts as ``v``)."""
    return {k: float(v.value if isinstance(v, Fixed) else v) for k, v in dist.args}


def support(dist: Distribution) -> tuple[float, float]:
    if dist.name == "uniform":
        a = numbers(dist)
        return a["lower"], a["upper"]
    return SUPPORT[dist.name]


def log_density(dist: Distribution, v):
    """log of ``dist``'s density at ``v`` (a JAX scalar), with every normalizing constant."""
    a = numbers(dist)
    name = dist.name
    if name == "normal":
        mean = a["mean"]
        sd = a["sd"] if "sd" in a else math.sqrt(a["var"]) if "var" in a else a["cv"] * abs(mean)
        return norm.logpdf(v, mean, sd)
    if name == "lognormal":
        sdlog = a["sdlog"] if "sdlog" in a else math.sqrt(math.log1p(a["cv"] ** 2))
        if "median" in a:
            meanlog = math.log(a["median"])
        elif "meanlog" in a:
            meanlog = a["meanlog"]
        else:
            meanlog = math.log(a["mean"]) - 0.5 * sdlog**2
        lv = jnp.log(v)
        return norm.logpdf(lv, meanlog, sdlog) - lv
    if name == "exponential":
        rate = a["rate"] if "rate" in a else 1.0 / a["mean"]
        return math.log(rate) - rate * v
    if name == "logitnormal":
        return norm.logpdf(logit(v), logit(a["median"]), a["sdlogit"]) - jnp.log(v * (1 - v))
    if name == "weibull":
        k, lam = a["shape"], a["scale"]
        return math.log(k / lam) + (k - 1) * jnp.log(v / lam) - (v / lam) ** k
    if name == "gamma":
        if "shape" in a:
            shape = a["shape"]
            rate = a["rate"] if "rate" in a else 1.0 / a["scale"]
        else:
            mean = a["mean"]
            var = a["var"] if "var" in a else (a["sd"] if "sd" in a else a["cv"] * mean) ** 2
            shape, rate = mean**2 / var, mean / var
        return shape * math.log(rate) - gammaln(shape) + (shape - 1) * jnp.log(v) - rate * v
    if name == "beta":
        if "alpha" in a:
            al, be = a["alpha"], a["beta"]
        else:
            m = a["mean"]
            c = (
                a["concentration"]
                if "concentration" in a
                else (m * (1 - m) / (a["var"] if "var" in a else a["sd"] ** 2) - 1)
            )
            al, be = m * c, (1 - m) * c
        const = gammaln(al + be) - gammaln(al) - gammaln(be)
        return const + (al - 1) * jnp.log(v) + (be - 1) * jnp.log1p(-v)
    if name == "uniform":
        lo, hi = a["lower"], a["upper"]
        inside = (v >= lo) & (v <= hi)
        return jnp.where(inside, -math.log(hi - lo), -jnp.inf)
    raise ValueError(f"no prior density for {name}.")  # pragma: no cover


def draw(dist: Distribution, rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` draws from ``dist`` (NumPy)."""
    a = numbers(dist)
    name = dist.name
    if name == "uniform":
        return rng.uniform(a["lower"], a["upper"], n)
    if name == "gamma":
        if "shape" in a:
            shape = a["shape"]
            scale = a["scale"] if "scale" in a else 1.0 / a["rate"]
        else:
            mean = a["mean"]
            var = a["var"] if "var" in a else (a["sd"] if "sd" in a else a["cv"] * mean) ** 2
            shape, scale = mean**2 / var, var / mean
        return rng.gamma(shape, scale, n)
    if name == "beta":
        if "alpha" in a:
            al, be = a["alpha"], a["beta"]
        else:
            m = a["mean"]
            c = (
                a["concentration"]
                if "concentration" in a
                else (m * (1 - m) / (a["var"] if "var" in a else a["sd"] ** 2) - 1)
            )
            al, be = m * c, (1 - m) * c
        return rng.beta(al, be, n)
    from .distributions import quantile  # normal, lognormal, exponential, logitnormal, weibull

    return np.asarray(quantile(name, a, jnp.asarray(rng.normal(size=n))))


@dataclass(frozen=True)
class Prior:
    """The prior of one estimated quantity: ``dist`` on ``x`` (``shift`` None), on ``x - lo``
    (``("above", lo)``) or on ``hi - x`` (``("below", hi)``); ``written`` or by default."""

    name: str
    dist: Distribution
    written: bool
    shift: tuple[str, float] | None = None

    def __str__(self) -> str:
        on = ""
        if self.shift is not None:
            kind, b = self.shift
            on = f" of {self.name} - {_num(b)}"
            if kind == "below":
                on = f" of {_num(b)} - {self.name}"
        return f"{self.dist}{on}{'' if self.written else ' (default)'}"

    def log_density(self, x):
        v = x
        if self.shift is not None:
            kind, b = self.shift
            v = x - b if kind == "above" else b - x
        return log_density(self.dist, v)

    def draw(self, rng, n):
        d = draw(self.dist, rng, n)
        if self.shift is not None:
            kind, b = self.shift
            d = b + d if kind == "above" else b - d
        return d


def _range(scale) -> tuple[float, float]:
    """The values a quantity can take on its estimation scale."""
    kind = scale[0]
    if kind == "logit":
        return scale[1], scale[2]
    if kind == "above":
        return scale[1], math.inf
    if kind == "below":
        return -math.inf, scale[1]
    if kind == "log":
        return 0.0, math.inf
    return -math.inf, math.inf


def default(name: str, start: float, scale) -> Prior | str:
    """The default prior of a quantity, or why there is none (a start of 0 on any sign)."""
    kind = scale[0]
    if kind == "logit":
        lo, hi = scale[1], scale[2]
        return Prior(name, Distribution("uniform", (("lower", lo), ("upper", hi))), False)
    if kind in ("above", "below"):
        b = scale[1]
        median = start - b if kind == "above" else b - start
        dist = Distribution("lognormal", (("median", median), ("sdlog", DEFAULT_SDLOG)))
        return Prior(name, dist, False, (kind, b))
    if kind == "log":
        dist = Distribution("lognormal", (("median", start), ("sdlog", DEFAULT_SDLOG)))
        return Prior(name, dist, False)
    if start == 0:
        return (
            f"{name} starts at 0 and takes any sign, so there is no scale for a default "
            f"prior; write one, e.g. prior {name} ~ normal(mean=0, sd=1) with the SD on the "
            "scale you expect"
        )
    dist = Distribution("normal", (("mean", start), ("sd", DEFAULT_SD_FACTOR * abs(start))))
    return Prior(name, dist, False)


def written_problem(name: str, dist: Distribution, scale) -> str | None:
    """Why a written prior does not suit its quantity, or None."""
    lo, hi = _range(scale)
    if name.startswith("corr("):
        return (
            f"the prior of a correlation is LKJ(2) on its correlate block in this release; "
            f"remove the prior line of {name}"
        )
    s_lo, s_hi = support(dist)
    if lo < s_lo or hi > s_hi:
        where = (f"in ({_num(lo)}, {_num(hi)})" if math.isfinite(lo) and math.isfinite(hi)
                 else f"above {_num(lo)}" if math.isfinite(lo)
                 else f"below {_num(hi)}" if math.isfinite(hi) else "of any sign")  # fmt: skip
        if dist.name == "uniform":
            return (
                f"{name} can take values {where}, outside uniform({_num(s_lo)}, {_num(s_hi)}); "
                f"give the estimate these bounds instead: estimate {name} = <start> in "
                f"({_num(s_lo)}, {_num(s_hi)}) (a uniform prior over them is its default)"
            )
        return (
            f"{name} can take values {where}, but {dist.name} has density only in "
            f"({_num(s_lo)}, {_num(s_hi)}); use a prior over all its values (normal for any "
            "sign) or bound the estimate"
        )
    return None


def priors(problem, names, x0, scales, *, defaults: bool) -> tuple[list[Prior], list[str]]:
    """The prior of every estimated quantity except correlations: written, or (``defaults``)
    by the rule; and the problems (a written prior that does not suit, a quantity with no
    default)."""
    written = {p.name: p.dist for p in problem._priors}
    out, problems = [], []
    for name, start, sc in zip(names, x0, scales, strict=True):
        if name.startswith("corr("):
            if name in written:
                problems.append(f"prior {name}: {written_problem(name, written[name], sc)}.")
            continue
        if name in written:
            why = written_problem(name, written[name], sc)
            if why:
                problems.append(f"prior {name}: {why}.")
            else:
                out.append(Prior(name, written[name], True))
        elif defaults:
            p = default(name, float(start), sc)
            if isinstance(p, str):
                problems.append(f"prior {name}: {p}.")
            else:
                out.append(p)
    return out, problems


def lkj_log_density(R, eta: float = LKJ_ETA):
    """LKJ(eta) on a correlation matrix, up to a constant: (eta - 1) log det R (NaN when R is
    not a correlation matrix)."""
    C = jnp.linalg.cholesky(R)
    return (eta - 1.0) * 2.0 * jnp.sum(jnp.log(jnp.diag(C)))


def lkj_draw(rng: np.random.Generator, dim: int, n: int, eta: float = LKJ_ETA) -> np.ndarray:
    """``n`` correlation matrices from LKJ(eta) (the onion method, Lewandowski et al. 2009)."""
    out = np.empty((n, dim, dim))
    for k in range(n):
        beta = eta + (dim - 2) / 2.0
        r12 = 2 * rng.beta(beta, beta) - 1
        R = np.array([[1.0, r12], [r12, 1.0]]) if dim >= 2 else np.ones((1, 1))
        for m in range(2, dim):
            beta -= 0.5
            y = rng.beta(m / 2.0, beta)
            u = rng.normal(size=m)
            u /= np.linalg.norm(u)
            w = np.sqrt(y) * u
            A = np.linalg.cholesky(R)
            z = A @ w
            R = np.block([[R, z[:, None]], [z[None, :], np.ones((1, 1))]])
        out[k] = R
    return out


def written_penalty(problem, names, x0, scales):
    """-2 log of the written priors as a JAX function of x (the term ``optimize`` and
    ``laplace`` add to -2LL, so their estimates are the posterior mode), or None when no prior
    is written."""
    ps = priors(problem, names, x0, scales, defaults=False)[0]
    if not ps:
        return None
    index = {n: i for i, n in enumerate(names)}
    at = [(index[p.name], p) for p in ps]

    def penalty(x):
        return -2.0 * sum(p.log_density(x[i]) for i, p in at)

    return penalty
