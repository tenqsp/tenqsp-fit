# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Distributions (of a parameter between units, or as a prior) and residual error models.

The names and keyword combinations are TenQSP's (``normal(mean, sd)``, ``lognormal(median,
sdlog)``, ...), so a distribution reads the same in a model and in a fit. An argument is a
number (the start of an estimated quantity), ``fix(number)`` (held), or, for a parameter that
varies between units, a formula of estimated quantities and data columns (a covariate model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import tenqsp as tq

from .errors import nearest

_Formula = type(tq.formula("x"))  # TenQSP's formula class

__all__ = [
    "Distribution",
    "ErrorModel",
    "Fixed",
    "additive",
    "combined",
    "exponential",
    "fix",
    "gamma",
    "beta",
    "logitnormal",
    "lognormal",
    "normal",
    "power",
    "proportional",
    "uniform",
    "weibull",
]


@dataclass(frozen=True)
class Fixed:
    """A value held fixed, written ``fix(value)``."""

    value: float

    def __str__(self) -> str:
        return f"fix({_num(self.value)})"


def fix(value: float) -> Fixed:
    """Hold one argument of a distribution or an error model at ``value``:
    ``tf.lognormal(median=1.5, sdlog=tf.fix(0.5))``."""
    return Fixed(_as_float(value, "fix"))


def _num(x: float) -> str:
    """A number as it would be written: 0.5, 1e-6, 30."""
    text = repr(float(x))
    if "e" in text:
        mant, exp = text.split("e")
        return f"{mant.removesuffix('.0')}e{int(exp)}"
    return text.removesuffix(".0")


def _as_float(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, int | float):
        raise TypeError(f"{what}: expected a number, got {x!r}.")
    if not math.isfinite(x):
        raise ValueError(f"{what}: the value must be finite, got {x!r}.")
    return float(x)


# ----------------------------------------------------------------------------- distributions
# Accepted keyword combinations (as TenQSP's), in the order they are written.
_COMBOS: dict[str, list[tuple[str, ...]]] = {
    "normal": [("mean", "sd"), ("mean", "var"), ("mean", "cv")],
    "lognormal": [
        ("median", "sdlog"),
        ("median", "cv"),
        ("meanlog", "sdlog"),
        ("meanlog", "cv"),
        ("mean", "cv"),
        ("mean", "sdlog"),
    ],
    "exponential": [("mean",), ("rate",)],
    "logitnormal": [("median", "sdlogit")],
    "weibull": [("shape", "scale")],
    "gamma": [
        ("shape", "rate"),
        ("shape", "scale"),
        ("mean", "sd"),
        ("mean", "var"),
        ("mean", "cv"),
    ],
    "beta": [("alpha", "beta"), ("mean", "concentration"), ("mean", "var"), ("mean", "sd")],
    "uniform": [("lower", "upper")],
}
# Usable for a parameter that varies between units in this release (closed-form quantile
# functions); the others are priors only for now.
BETWEEN_UNITS = ("normal", "lognormal", "exponential", "logitnormal", "weibull")
_LATER = {
    "gamma": "gamma between units arrives in tenqsp-fit 0.2; lognormal is close for most "
    "data, and gamma(mean, cv=1) is the exponential",
    "beta": "beta between units arrives in tenqsp-fit 0.2; logitnormal(median, sdlogit) also "
    "stays in (0, 1)",
    "uniform": "uniform is a prior, not a distribution between units",
}
# a spread of 0 means "does not vary" (estimate x ~ normal(mean=0.75, sd=0))
SPREADS = ("sd", "var", "cv", "sdlog", "sdlogit")
_POSITIVE = {
    "sd", "var", "cv", "sdlog", "sdlogit", "rate", "shape", "scale", "alpha", "beta",
    "concentration",
}  # fmt: skip


@dataclass(frozen=True)
class Distribution:
    """A distribution with its arguments as written, e.g. ``lognormal(median=1.5, sdlog=0.5)``.
    Each argument is a number, a ``Fixed`` value, or a formula (text)."""

    name: str
    args: tuple[tuple[str, float | Fixed | str], ...] = field(default=())

    def __str__(self) -> str:
        return f"{self.name}(" + ", ".join(f"{k}={_arg_text(v)}" for k, v in self.args) + ")"

    def __repr__(self) -> str:
        return f"tf.{self}"

    @property
    def keywords(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.args)


def _arg_text(v) -> str:
    return _num(v) if isinstance(v, float) else str(v)


def _make(name: str, kwargs: dict, *, formulas: bool = True) -> Distribution:
    combos = _COMBOS[name]
    known = sorted({k for c in combos for k in c})
    for k in kwargs:
        if k not in known:
            raise TypeError(
                f"{name}: unknown argument '{k}'.{nearest(k, known)} Accepted: {_accepted(name)}."
            )
    combo = next((c for c in combos if set(c) == set(kwargs)), None)
    if combo is None:
        raise TypeError(
            f"{name}: the arguments {sorted(kwargs) or '(none)'} are not a valid combination. "
            f"Accepted: {_accepted(name)}."
        )
    args = []
    for k in combo:  # written order: location first, then spread
        v = kwargs[k]
        if isinstance(v, Fixed):
            num = v.value
        elif isinstance(v, str | _Formula):
            if not formulas:
                raise TypeError(f"{name}: '{k}' must be a number here, got the formula {v!s}.")
            v = _formula_text(v, f"{name}: '{k}'")
            args.append((k, v))
            continue
        else:
            v = num = _as_float(v, f"{name}: '{k}'")
        _check(name, k, num)
        args.append((k, v))
    _check_together(name, {k: (v.value if isinstance(v, Fixed) else v) for k, v in args})
    return Distribution(name, tuple(args))


def _check_together(name: str, a: dict) -> None:
    """Arguments that are each valid but not together (formulas are not checked)."""
    num = {k: v for k, v in a.items() if isinstance(v, int | float)}
    if name == "normal" and "cv" in num and num.get("mean") == 0 and num["cv"] != 0:
        raise ValueError("normal: a cv of a mean of 0 is an sd of 0; give normal(mean=0, sd=...).")
    if name == "beta" and "mean" in num and ("sd" in num or "var" in num):
        m = num["mean"]
        var = num["sd"] ** 2 if "sd" in num else num["var"]
        if not var < m * (1 - m):
            raise ValueError(
                f"beta: with mean {_num(m)} the variance must be below mean*(1 - mean) = "
                f"{_num(m * (1 - m))} (an sd below {_num(math.sqrt(m * (1 - m)))}); got "
                f"{_num(var)}."
            )


def _formula_text(v, what: str) -> str:
    from .entries import formula

    text = str(v).strip()
    try:
        formula(text)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{what}: {e}") from None
    return text


def _check(name: str, key: str, v: float) -> None:
    if name in BETWEEN_UNITS and key in SPREADS and v == 0:
        return  # does not vary
    if key in _POSITIVE and not v > 0:
        raise ValueError(f"{name}: '{key}' must be > 0 (got {_num(v)}).")
    if name in ("lognormal", "exponential", "gamma") and key in ("mean", "median") and not v > 0:
        raise ValueError(f"{name}: '{key}' must be > 0 (got {_num(v)}).")
    if name in ("logitnormal", "beta") and key in ("median", "mean") and not 0 < v < 1:
        raise ValueError(f"{name}: '{key}' must be between 0 and 1 (got {_num(v)}).")


def _accepted(name: str) -> str:
    return "; ".join(f"{name}(" + ", ".join(f"{k}=..." for k in c) + ")" for c in _COMBOS[name])


def make_distribution(name: str, kwargs: dict, *, between_units: bool) -> Distribution:
    """A distribution by name (as the text reader finds it), checked for its use."""
    if name not in _COMBOS:
        raise ValueError(f"unknown distribution '{name}'.{nearest(name, _COMBOS)}")
    if between_units and name not in BETWEEN_UNITS:
        raise ValueError(_LATER[name] + ".")
    return _make(name, kwargs, formulas=between_units)


def normal(**kwargs) -> Distribution:
    """``normal(mean=, sd=)``, or ``(mean, var)``, ``(mean, cv)``."""
    return _make("normal", kwargs)


def lognormal(**kwargs) -> Distribution:
    """``lognormal(median=, sdlog=)``, or ``(median, cv)``, ``(meanlog, sdlog)``,
    ``(meanlog, cv)``, ``(mean, cv)``, ``(mean, sdlog)``. ``median`` is the typical value
    (NONMEM's THETA, Monolix's _pop); ``mean`` is the arithmetic mean, higher by
    exp(sdlog^2/2)."""
    return _make("lognormal", kwargs)


def exponential(**kwargs) -> Distribution:
    """``exponential(mean=)`` or ``(rate=)``: one number, so the spread is set by the mean
    (CV 100 %)."""
    return _make("exponential", kwargs)


def logitnormal(**kwargs) -> Distribution:
    """``logitnormal(median=, sdlogit=)``: a value in (0, 1), normal on the logit scale."""
    return _make("logitnormal", kwargs)


def weibull(**kwargs) -> Distribution:
    """``weibull(shape=, scale=)``."""
    return _make("weibull", kwargs)


def gamma(**kwargs) -> Distribution:
    """``gamma(shape=, rate=)``, or ``(shape, scale)``, ``(mean, sd)``, ``(mean, var)``,
    ``(mean, cv)``. A prior for now; between units in 0.2."""
    return _make("gamma", kwargs, formulas=False)


def beta(**kwargs) -> Distribution:
    """``beta(alpha=, beta=)``, or ``(mean, concentration)``, ``(mean, var)``, ``(mean, sd)``.
    A prior for now; between units in 0.2."""
    return _make("beta", kwargs, formulas=False)


def uniform(**kwargs) -> Distribution:
    """``uniform(lower=, upper=)``, a prior."""
    d = _make("uniform", kwargs, formulas=False)
    lo, hi = (v.value if isinstance(v, Fixed) else v for _, v in d.args)
    if not lo < hi:
        raise ValueError(f"uniform: 'lower' must be below 'upper' (got {_num(lo)}, {_num(hi)}).")
    return d


# ----------------------------------------------------------------------------- residual error
_ERRORS: dict[str, tuple[str, ...]] = {
    # model: names of its quantities (after the observable: conc.additive, ...)
    "additive": ("additive",),
    "proportional": ("proportional",),
    "combined": ("additive", "proportional"),
    "power": ("proportional", "exponent"),
}
ERROR_FORMS = {
    "additive": "sd = a",
    "proportional": "sd = b*f",
    "combined": "sd = sqrt(a^2 + (b*f)^2)",
    "power": "sd = b*f^k",
}


@dataclass(frozen=True)
class ErrorModel:
    """A residual error model with its starts (or fixed values), e.g. ``additive(0.7)``."""

    name: str
    args: tuple[float | Fixed, ...]

    def __str__(self) -> str:
        return f"{self.name}(" + ", ".join(_arg_text(v) for v in self.args) + ")"

    def __repr__(self) -> str:
        return f"tf.{self}"

    @property
    def quantities(self) -> tuple[str, ...]:
        return _ERRORS[self.name]


def make_error(name: str, args: list) -> ErrorModel:
    if name not in _ERRORS:
        raise ValueError(
            f"unknown error model '{name}'.{nearest(name, _ERRORS)} Error models: "
            "additive(a), proportional(b), combined(a, b), power(b, k)."
        )
    need = _ERRORS[name]
    if len(args) != len(need):
        sig = {"additive": "a", "proportional": "b", "combined": "a, b", "power": "b, k"}[name]
        raise TypeError(f"{name}({sig}) takes {len(need)} number(s), got {len(args)}.")
    out = []
    for q, v in zip(need, args, strict=True):
        num = v.value if isinstance(v, Fixed) else _as_float(v, name)
        if q != "exponent" and not num > 0:
            raise ValueError(f"{name}: the {q} part must be > 0 (got {_num(num)}).")
        out.append(v if isinstance(v, Fixed) else num)
    return ErrorModel(name, tuple(out))


def additive(a) -> ErrorModel:
    """Additive residual error: the SD is ``a`` whatever the prediction."""
    return make_error("additive", [a])


def proportional(b) -> ErrorModel:
    """Proportional residual error: the SD is ``b`` times the prediction."""
    return make_error("proportional", [b])


def combined(a, b) -> ErrorModel:
    """Combined residual error: SD = sqrt(a^2 + (b*prediction)^2) (variances add, as
    NONMEM's ``Y = F + F*EPS(1) + EPS(2)``)."""
    return make_error("combined", [a, b])


def power(b, k) -> ErrorModel:
    """Power residual error: SD = b * prediction^k (k = 1 is proportional)."""
    return make_error("power", [b, k])


# ----------------------------------------------------------------------------- normal scores
def quantile(name: str, args: dict, z):
    """A unit's value from its normal score ``z`` (standard normal between units): the
    distribution's quantile function at Phi(z), in closed form. ``args`` maps each keyword to a
    value (a JAX array, one per unit or one for all)."""
    import jax.numpy as jnp
    from jax.scipy.special import logit
    from jax.scipy.stats import norm

    a = args
    if name == "normal":
        mean = a["mean"]
        sd = a["sd"] if "sd" in a else jnp.sqrt(a["var"]) if "var" in a else a["cv"] * jnp.abs(mean)
        return mean + sd * z
    if name == "lognormal":
        if "sdlog" in a:
            sdlog = a["sdlog"]
        else:
            sdlog = jnp.sqrt(jnp.log1p(a["cv"] ** 2))
        if "median" in a:
            log_median = jnp.log(a["median"])
        elif "meanlog" in a:
            log_median = a["meanlog"]
        else:  # the arithmetic mean
            log_median = jnp.log(a["mean"]) - 0.5 * sdlog**2
        return jnp.exp(log_median + sdlog * z)
    if name == "exponential":
        mean = a["mean"] if "mean" in a else 1.0 / a["rate"]
        return -mean * norm.logcdf(-z)  # -mean log(1 - Phi(z))
    if name == "logitnormal":
        return 1.0 / (1.0 + jnp.exp(-(logit(a["median"]) + a["sdlogit"] * z)))
    if name == "weibull":
        return a["scale"] * (-norm.logcdf(-z)) ** (1.0 / a["shape"])
    raise ValueError(f"{name} between units is not available.")  # pragma: no cover


def spreads(name: str, args: dict) -> dict:
    """The other usual ways of stating a distribution's spread, from its arguments (numbers):
    for a lognormal, sdlog, CV and the variance on the log scale (NONMEM's OMEGA)."""
    a = args
    if name == "lognormal":
        sdlog = a["sdlog"] if "sdlog" in a else math.sqrt(math.log1p(a["cv"] ** 2))
        return {"sdlog": sdlog, "cv": math.sqrt(math.expm1(sdlog**2)), "var": sdlog**2}
    if name == "normal":
        sd = a["sd"] if "sd" in a else math.sqrt(a["var"]) if "var" in a else None
        return {"sd": sd, "var": sd**2} if sd is not None else {}
    if name == "logitnormal":
        return {"sdlogit": a["sdlogit"], "var": a["sdlogit"] ** 2}
    return {}
