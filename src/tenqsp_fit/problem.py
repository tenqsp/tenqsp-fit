# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The fitting problem: a TenQSP model, data, and what is estimated, written as declarations.

Every statement of the fit text is one method of ``Problem`` with the same words, so a problem
written in Python prints as text (``to_text``) and reads back to the same problem.
"""

from __future__ import annotations

import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tenqsp as tq

from . import data as D
from . import units as U
from .distributions import (
    _LATER,
    BETWEEN_UNITS,
    ERROR_FORMS,
    SPREADS,
    Distribution,
    ErrorModel,
    Fixed,
    _as_float,
    _formula_text,
    _num,
)
from .errors import ProblemError, nearest

__all__ = ["Problem"]

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
METHODS = {
    "optimize": ("one set of parameters (least squares or maximum likelihood)", ()),
    "laplace": ("population, FOCE-I", ()),
    "nuts": ("Bayesian, NUTS sampling", ("chains", "warmup", "samples")),
}
# a method's options with their defaults (printed in the summary)
METHOD_DEFAULTS = {"nuts": {"chains": 4, "warmup": 1000, "samples": 1000}}
# options with their defaults (printed in the summary; rule: no silent defaults)
OPTIONS = {"rtol": 1e-6, "atol": None, "starts": 1, "seed": 0, "maxiter": None, "progress": None}


class _Text(str):
    """Text that shows itself as it prints (so ``problem.summary()`` reads well at a prompt)."""

    def __repr__(self) -> str:
        return str(self)


# ----------------------------------------------------------------------------- declarations
# ``origin`` is "line N: <text>" when read from a fit text, so later problems name the line.
@dataclass
class _Decl:
    origin: str | None = field(default=None, compare=False, kw_only=True)


@dataclass
class _Data(_Decl):
    source: str | None  # the file name as written; None for a table given in Python
    columns: dict[str, str]
    table: pd.DataFrame = field(compare=False, repr=False)
    sha256: str = field(default="", compare=False, repr=False)
    path: str | None = field(default=None, compare=False, repr=False)


@dataclass
class _Doses(_Decl):
    into: str | None = None
    cmt: dict | None = None
    column: str | None = None
    at: float | None = None


@dataclass
class _Observe(_Decl):
    lhs: str
    column: str
    where: str | None


@dataclass
class _Ignore(_Decl):
    column: str
    where: str


@dataclass
class _Estimate(_Decl):
    name: str
    start: float | None = None
    bounds: tuple[float, float] | None = None
    dist: Distribution | None = None
    written: Distribution | None = None  # a distribution with spread 0: does not vary
    kind: str | None = None  # its scale, set by the family: "log" or "as is"


def _spread_zero(dist: Distribution) -> bool:
    """A distribution whose spread is 0 (or fix(0)): the quantity does not vary."""
    for k, v in dist.args:
        if k in SPREADS:
            num = v.value if isinstance(v, Fixed) else v
            return isinstance(num, float) and num == 0
    return False


def _not_varying(name: str, dist: Distribution) -> _Estimate:
    """``estimate x ~ family(location, spread=0)``: one value for all units, estimated on the
    family's scale (normal: any sign; lognormal: log scale; logitnormal: within (0, 1))."""
    args = dict(dist.args)
    key, loc = dist.args[0]
    if isinstance(loc, str):
        raise ValueError(
            f"estimate {name}: {name} does not vary (spread 0) and its {key} is a formula; "
            f"write it as a formula: {name} = {loc}"
        )
    if isinstance(loc, Fixed):
        raise ValueError(
            f"estimate {name}: {name} does not vary (spread 0) and its {key} is fixed; write "
            f"fix {name} = {_num(loc.value)}"
        )
    if dist.name == "normal":
        return _Estimate(name, start=loc, written=dist, kind="as is")
    if dist.name == "lognormal":
        start = math.exp(loc) if key == "meanlog" else loc  # spread 0: mean = median
        return _Estimate(name, start=start, written=dist, kind="log")
    if dist.name == "logitnormal":
        return _Estimate(name, start=loc, bounds=(0.0, 1.0), written=dist)
    raise AssertionError(args)  # pragma: no cover


@dataclass
class _Fix(_Decl):
    name: str
    value: float


@dataclass
class _Correlate(_Decl):
    names: tuple[str, ...]
    start: float | None


@dataclass
class _Define(_Decl):
    name: str
    formula: str


@dataclass
class _Error(_Decl):
    lhs: str
    model: ErrorModel


@dataclass
class _Likelihood(_Decl):
    lhs: str
    density: object  # a formula (text), or a Python function (dv, prediction, values)


@dataclass
class _Censor(_Decl):
    lhs: str
    below: float | str
    where: str | None = None


@dataclass
class _Prior(_Decl):
    name: str
    dist: Distribution


@dataclass
class _Method(_Decl):
    name: str
    options: dict


def _key(lhs: str) -> str:
    """A link's left side without spaces, to match ``error log10(bact)`` to its ``observe``."""
    return "".join(lhs.split())


class _ModelInfo:
    """What the fit needs to know about the model, from its IR."""

    def __init__(self, m: tq.Model):
        ir = m.to_ir()
        self.name = ir.name
        self.params = {p.name: p for p in ir.params}
        self.definitions = {d.name: d.expr for d in ir.definitions}
        self.compartments = [c.name for c in ir.compartments]
        self.fields = [f.name for f in ir.fields]
        self.observables = [o.name for o in ir.observables]
        self.observable_exprs = {o.name: o.expr for o in ir.observables}
        self.n_states = sum(ir.state_count().values())
        self.hash = ir.content_hash()

    def reads(self, observable: str) -> set[str]:
        """Compartments and fields an observable reads at the same time point, through the
        model's named formulas."""
        out, seen, todo = set(), set(), [observable]
        states = set(self.compartments) | set(self.fields)
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in states:
                out.add(n)
            elif n in self.observable_exprs:
                todo.extend(tq.formula(self.observable_exprs[n]).names)
            elif n in self.definitions:
                todo.extend(tq.formula(self.definitions[n]).names)
        return out

    def param_names(self, text: str) -> set[str]:
        """Model parameters a formula depends on, through the model's named formulas."""
        out, seen, todo = set(), set(), list(tq.formula(text).names)
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in self.params:
                out.add(n)
            elif n in self.definitions:
                todo.extend(tq.formula(self.definitions[n]).names)
        return out


# ----------------------------------------------------------------------------- the problem
class Problem:
    """A fitting problem: ``tf.Problem(model, name="theoph_pop")``, then one call per
    statement (``data``, ``unit``, ``doses``, ``observe``, ``estimate``, ``fix``,
    ``correlate``, ``define``, ``error``, ``censor``, ``prior``, ``method``, ``options``).
    ``summary()`` shows the whole problem and ``to_text()`` writes it as fit text."""

    def __init__(self, model: tq.Model, *, name: str | None = None):
        if not isinstance(model, tq.Model):
            raise TypeError(
                f"Problem: give a TenQSP model (tq.load('model.tq')), not {type(model).__name__}."
            )
        if name is not None and not _NAME.fullmatch(name):
            raise ValueError(f"Problem: the name must be one word, got '{name}'.")
        self.model = model
        self.name = name or model.name
        self._m = _ModelInfo(model)
        self._data: _Data | None = None
        self._unit: str | None = None
        self._unit_origin: str | None = None
        self._doses: _Doses | None = None
        self._links: list[_Observe | _Ignore] = []
        self._params: list[_Estimate | _Fix] = []
        self._correlations: list[_Correlate] = []
        self._defines: list[_Define] = []
        self._errors: list[_Error] = []
        self._likelihoods: list[_Likelihood] = []
        self._censors: list[_Censor] = []
        self._priors: list[_Prior] = []
        self._method: _Method | None = None
        self._options: dict = {}
        self._options_origin: str | None = None

    # ------------------------------------------------------------------ statements
    def data(self, source, **columns: str) -> None:
        """The data table: a CSV file (NONMEM format by default) or a pandas DataFrame. The
        keywords map the data's own column names to the standard ones, as NONMEM's ``$INPUT``:
        ``p.data("theoph.csv", ID="Subject", TIME="Time", DV="conc")``."""
        self._set_data(source, columns)

    def _set_data(self, source, columns, *, base: Path | None = None, origin=None) -> None:
        if self._data is not None:
            raise ValueError("the data are given twice; give one table.")
        table, sha, path = D.read_table(source, base)
        table = D.apply_map(table, columns)
        written = None if isinstance(source, pd.DataFrame) else str(source)
        self._data = _Data(written, dict(columns), table, sha, path, origin=origin)

    def unit(self, column: str) -> None:
        """The column naming the experimental unit (a subject, animal, flask or well): rows
        with the same value belong together, and parameters written with a distribution vary
        between units. Without it, all rows are one unit."""
        self._unit_origin = None
        if self._unit is not None:
            raise ValueError(f"the unit is named twice ({self._unit} and {column}).")
        if not isinstance(column, str) or not _NAME.fullmatch(column):
            raise ValueError(f"unit: give one column name, e.g. unit ID; got {column!r}.")
        self._unit = column

    def doses(
        self,
        *,
        into: str | None = None,
        cmt: dict | None = None,
        column: str | None = None,
        at: float | None = None,
    ) -> None:
        """Where the data's doses go: ``into="depot"`` (every dose row, AMT), ``cmt={1:
        "depot", 2: "centr"}`` (by the CMT column), or ``column="Dose", at=0, into="depot"``
        (the dose as a column of the data, given once per unit at time ``at``)."""
        if self._doses is not None:
            raise ValueError("doses are declared twice; give one doses statement.")
        if cmt is not None:
            if into is not None or column is not None or at is not None:
                raise TypeError("doses: give cmt={...} alone, or into= (with column= and at=).")
            if not isinstance(cmt, dict) or not cmt:
                raise TypeError("doses: cmt maps CMT numbers to places, e.g. {1: 'depot'}.")
            for k, place in cmt.items():
                _as_float(k, "doses: a CMT number")
                self._dose_place(place)
            self._doses = _Doses(
                cmt={int(k) if float(k).is_integer() else k: v for k, v in cmt.items()}
            )
            return
        if into is None:
            raise TypeError("doses: say where the doses go: into='depot', or cmt={1: 'depot'}.")
        self._dose_place(into)
        if (column is None) != (at is None):
            raise TypeError("doses: a dose column needs its time: column='Dose', at=0.")
        if column is not None:
            if not _NAME.fullmatch(str(column)):
                raise ValueError(f"doses: '{column}' is not a column name.")
            at = _as_float(at, "doses: at")
        self._doses = _Doses(into=into, column=column, at=at)

    def _dose_place(self, place) -> None:
        if place in self._m.compartments:
            return
        if place in self._m.fields:
            raise ValueError(
                f"doses into the field '{place}' (with a shape over its axis) arrive in a later "
                "release; doses go into compartments for now."
            )
        raise ValueError(
            f"doses: the model has no compartment '{place}'.{nearest(place, self._m.compartments)}"
            f" Compartments: {', '.join(self._m.compartments) or '(none)'}."
        )

    def observe(self, observable: str, column: str = "DV", *, where: str | None = None) -> None:
        """Link an observable to the measurements in a data column: ``p.observe("conc",
        "DV")``. ``where`` picks the rows (``"DVID == 1"``). The observable may be a formula of
        observables when the data are on another scale: ``p.observe("log10(bact)", "DV")``."""
        lhs = self._lhs(observable, "observe")
        column = self._column_name(column, "observe")
        where = D.check_condition(where) if where is not None else None
        for link in self._links:
            if isinstance(link, _Observe) and _key(link.lhs) == _key(lhs):
                raise ValueError(
                    f"{lhs} is linked twice; one observe line per observable (use 'where' to "
                    "pick its rows)."
                )
        self._links.append(_Observe(lhs, column, where))

    def ignore(self, column: str = "DV", *, where: str) -> None:
        """Leave some measurements out on purpose: ``p.ignore("DV", where="DVID == 2")``."""
        column = self._column_name(column, "ignore")
        self._links.append(_Ignore(column, D.check_condition(where)))

    def _lhs(self, text, what: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"{what}: give the observable as text, e.g. 'conc'.")
        text = text.strip()
        try:
            names = tq.formula(text).names
        except (ValueError, TypeError) as e:
            raise ValueError(f"{what} {text}: {e}") from None
        if not names:
            raise ValueError(f"{what} {text}: name an observable of the model.")
        obs = self._m.observables
        for n in names:
            if n not in obs:
                listed = ", ".join(obs) or "(none; add an observable line to the model)"
                raise ValueError(
                    f"{what}: '{n}' is not an observable of the model.{nearest(n, obs)}"
                    f" Observables: {listed}."
                )
        return text

    @staticmethod
    def _column_name(column, what: str) -> str:
        if not isinstance(column, str) or not _NAME.fullmatch(column):
            raise ValueError(f"{what}: give a column name, e.g. DV; got {column!r}.")
        return column

    def estimate(self, name: str, value, *, bounds: tuple[float, float] | None = None) -> None:
        """Estimate a parameter. ``value`` is a number (one value for all units, estimated from
        this start) or a distribution (the parameter varies between units:
        ``tf.lognormal(median=1.5, sdlog=0.5)``; each number is the start of an estimated
        quantity, ``ka.median`` and ``ka.sdlog``). ``bounds=(0, 1)`` keeps a value inside."""
        self._new_name(name, "estimate")
        if isinstance(value, Distribution):
            if bounds is not None:
                raise TypeError(
                    f"estimate {name}: a distribution sets its own range; bounds are for one value."
                )
            if value.name not in BETWEEN_UNITS:
                raise ValueError(f"estimate {name}: {_LATER[value.name]}.")
            if _spread_zero(value):
                self._params.append(_not_varying(name, value))
            else:
                self._params.append(_Estimate(name, dist=value))
            return
        if isinstance(value, Fixed):
            raise TypeError(f"estimate {name}: a fixed value is written fix {name} = value.")
        start = _as_float(value, f"estimate {name}")
        if bounds is not None:
            try:
                lo, hi = (_as_float(b, f"estimate {name}: bounds") for b in bounds)
            except (TypeError, ValueError):
                raise TypeError(f"estimate {name}: bounds are two numbers, e.g. (0, 1).") from None
            if not lo < start < hi:
                raise ValueError(
                    f"estimate {name}: the start {_num(start)} must be inside the bounds "
                    f"({_num(lo)}, {_num(hi)})."
                )
            bounds = (lo, hi)
        self._params.append(_Estimate(name, start=start, bounds=bounds))

    def fix(self, name: str, value: float) -> None:
        """Hold a parameter at a value (NONMEM's FIX): ``p.fix("V", 30)``."""
        self._new_name(name, "fix")
        self._params.append(_Fix(name, _as_float(value, f"fix {name}")))

    def _new_name(self, name, what: str) -> None:
        if not isinstance(name, str) or not _NAME.fullmatch(name) or keyword.iskeyword(name):
            raise ValueError(f"{what}: '{name}' is not a name.")
        m = self._m
        if name in m.definitions:
            ps = sorted(m.param_names(m.definitions[name]))
            instead = f"; {what} {' or '.join(ps)} instead" if ps else ""
            raise ValueError(
                f"{what} {name}: {name} is not a parameter but a formula of the model "
                f"({name} = {m.definitions[name]}){instead}."
            )
        if name in m.compartments or name in m.fields or name in m.observables:
            raise ValueError(f"{what} {name}: {name} is a state or an observable of the model.")
        p = m.params.get(name)
        if p is not None and p.over is not None:
            raise ValueError(
                f"{what} {name}: {name} has one value per category of '{p.over}'; estimating "
                "such a parameter arrives in a later release."
            )
        for d in self._params:
            if d.name == name:
                was = "estimated" if isinstance(d, _Estimate) else "fixed"
                raise ValueError(f"{what} {name}: {name} is already {was}.")
        for d in self._defines:
            if d.name == name:
                raise ValueError(f"{what} {name}: {name} is already set by {name} = {d.formula}.")

    def correlate(self, *names: str, start: float | None = None) -> None:
        """Estimate the correlation between parameters that vary between units:
        ``p.correlate("ka", "CL")``; three or more names give every pair (NONMEM's BLOCK).
        ``start`` (two names only) is the starting correlation; otherwise 0."""
        if len(names) < 2:
            raise TypeError("correlate: name two or more parameters, e.g. correlate('ka', 'CL').")
        if len(set(names)) != len(names):
            raise ValueError(f"correlate: a name is given twice in {', '.join(names)}.")
        for n in names:
            if not isinstance(n, str) or not _NAME.fullmatch(n):
                raise ValueError(f"correlate: '{n}' is not a name.")
        if start is not None:
            if len(names) != 2:
                raise TypeError("correlate: a start is for one pair; give two names with it.")
            start = _as_float(start, "correlate: start")
            if not -1 < start < 1:
                raise ValueError(
                    f"correlate: a correlation is between -1 and 1, got {_num(start)}."
                )
        self._correlations.append(_Correlate(tuple(names), start))

    def define(self, name: str, formula: str) -> None:
        """Set a model parameter by a formula for this fit only, leaving the model unchanged
        (as NONMEM's $PK): ``p.define("ka", "mu")``. Not needed when the model already has the
        formula."""
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"'{name}' is not a name.")
        m = self._m
        if name in m.definitions:
            raise ValueError(
                f"{name} = {formula}: the model already sets {name} = {m.definitions[name]}; "
                "estimate what it depends on, or change the model."
            )
        if name not in m.params:
            raise ValueError(
                f"{name} = {formula}: the model has no parameter '{name}'."
                f"{nearest(name, m.params)} A formula after 'fit' sets a model parameter; a "
                "new quantity is declared with estimate or fix."
            )
        if any(d.name == name for d in self._params) or any(d.name == name for d in self._defines):
            raise ValueError(f"{name} = {formula}: {name} is already estimated, fixed or set.")
        self._defines.append(_Define(name, _formula_text(formula, f"{name} = ")))

    def error(self, observable: str, model: ErrorModel) -> None:
        """The residual error of a linked observable: ``p.error("conc", tf.additive(0.7))``;
        also ``proportional(b)``, ``combined(a, b)``, ``power(b, k)``. On a transformed link
        (``log10(bact)``) it applies on that scale."""
        lhs = self._lhs(observable, "error")
        if not isinstance(model, ErrorModel):
            raise TypeError(
                f"error {lhs}: give an error model, e.g. tf.additive(0.7), not {model!r}."
            )
        if any(_key(e.lhs) == _key(lhs) for e in self._errors):
            raise ValueError(f"error {lhs}: its error model is given twice.")
        self._errors.append(_Error(lhs, model))

    def likelihood(self, observable: str, density) -> None:
        """A link's log-density written out, in place of an error model (as NONMEM's
        $ERROR with -2LL): ``p.likelihood("conc", "-0.5*((DV - conc)/(a + b*conc))^2 -
        log(a + b*conc)")``. In the formula the observables are the model's predictions, DV
        the measurement (on the link's scale), and the other names estimated or fixed
        quantities (``estimate a = 0.1, b = 0.1``). Or a Python function of JAX arrays,
        ``density(dv, prediction, values)`` with ``prediction`` the link's left side and
        ``values`` the estimated and fixed quantities by name; a function is not written
        by ``to_text``."""
        lhs = self._lhs(observable, "likelihood")
        if isinstance(density, str):
            density = _formula_text(density, f"likelihood {lhs}: ")
        elif not callable(density):
            raise TypeError(
                f"likelihood {lhs}: give a formula (text) or a Python function, not {density!r}."
            )
        if any(_key(e.lhs) == _key(lhs) for e in self._likelihoods):
            raise ValueError(f"likelihood {lhs}: its likelihood is given twice.")
        self._likelihoods.append(_Likelihood(lhs, density))

    def censor(self, observable: str, *, below, where: str | None = None) -> None:
        """Measurements below a limit of quantification, by the M3 likelihood. ``below`` is
        the limit (a number, or a column: ``below="LLOQ"``), on the link's scale. Which
        measurements are censored: those meeting ``where`` when it is given (a flag column,
        ``where="CENS == 1"``, or a text in DV, ``where='DV == "BLQ"'``; their DV is not
        read); otherwise those whose value is strictly below the limit."""
        lhs = self._lhs(observable, "censor")
        if isinstance(below, str):
            below = self._column_name(below, f"censor {lhs}")
        else:
            below = _as_float(below, f"censor {lhs}: below")
        where = D.check_condition(where) if where is not None else None
        if any(_key(c.lhs) == _key(lhs) for c in self._censors):
            raise ValueError(f"censor {lhs}: its limit is given twice.")
        self._censors.append(_Censor(lhs, below, where))

    def prior(self, name: str, dist: Distribution) -> None:
        """A prior on an estimated quantity: ``p.prior("ke", tf.lognormal(median=0.08,
        sdlog=0.5))``; quantities of a distribution are named ``ka.median``, ``ka.sdlog``."""
        if not isinstance(dist, Distribution):
            raise TypeError(f"prior {name}: give a distribution, e.g. tf.normal(mean=0, sd=1).")
        for _, v in dist.args:
            if isinstance(v, str):
                raise ValueError(f"prior {name}: the arguments of a prior are numbers.")
        for k, v in dist.args:
            if k in SPREADS and (v.value if isinstance(v, Fixed) else v) == 0:
                raise ValueError(f"prior {name}: '{k}' must be > 0 (a prior needs a spread).")
        if any(p.name == name for p in self._priors):
            raise ValueError(f"prior {name}: given twice.")
        self._priors.append(_Prior(name, dist))

    def method(self, name: str, **options) -> None:
        """How to fit: ``"optimize"`` (one set of parameters), ``"laplace"`` (parameters that
        vary between units; FOCE-I), ``"nuts"`` (Bayesian; options ``chains``, ``samples``,
        ``warmup``)."""
        if name not in METHODS:
            raise ValueError(
                f"unknown method '{name}'.{nearest(name, METHODS)} Methods: {', '.join(METHODS)}."
            )
        allowed = METHODS[name][1]
        for k, v in options.items():
            if k not in allowed:
                more = f" Its options: {', '.join(allowed)}." if allowed else ""
                raise TypeError(f"method {name}: unknown option '{k}'.{nearest(k, allowed)}{more}")
            if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
                raise ValueError(f"method {name}: {k} must be a positive whole number, got {v!r}.")
        if self._method is not None:
            raise ValueError(f"the method is given twice ({self._method.name} and {name}).")
        self._method = _Method(name, dict(options))

    def options(self, **options) -> None:
        """Solver and run options: ``rtol``, ``atol``, ``starts`` (multi-start), ``seed``,
        ``maxiter``, ``progress`` ("text", "plot" or "none")."""
        for k, v in options.items():
            if k not in OPTIONS:
                raise TypeError(
                    f"options: unknown option '{k}'.{nearest(k, OPTIONS)} Options: "
                    f"{', '.join(OPTIONS)}."
                )
            if k in ("rtol", "atol"):
                if not _as_float(v, f"options: {k}") > 0:
                    raise ValueError(f"options: {k} must be > 0.")
            elif k == "progress":
                if v not in ("text", "plot", "none"):
                    raise ValueError(f"options: progress is 'text', 'plot' or 'none', got {v!r}.")
            elif not (isinstance(v, int) and not isinstance(v, bool) and v >= (k == "starts")):
                raise ValueError(f"options: {k} must be a whole number, got {v!r}.")
            if k in self._options:
                raise ValueError(f"options: {k} is given twice.")
        self._options.update(options)

    # ------------------------------------------------------------------ derived lists
    def _varying(self) -> list[_Estimate]:
        return [d for d in self._params if isinstance(d, _Estimate) and d.dist is not None]

    def _fit_names(self) -> set[str]:
        """Names declared by estimate or fix (model parameters and new quantities)."""
        return {d.name for d in self._params}

    def estimated(self) -> list[str]:
        """The names of the estimated quantities, in the order declared."""
        out = []
        for d in self._params:
            if isinstance(d, _Estimate) and d.dist is None:
                out.append(d.name)
            elif isinstance(d, _Estimate):
                out += [f"{d.name}.{k}" for k, v in d.dist.args if isinstance(v, float)]
        for c in self._correlations:
            out += [f"corr({a},{b})" for i, a in enumerate(c.names) for b in c.names[i + 1 :]]
        for e in self._errors:
            out += [
                f"{e.lhs}.{q}"
                for q, v in zip(e.model.quantities, e.model.args, strict=True)
                if not isinstance(v, Fixed)
            ]
        return out

    # ------------------------------------------------------------------ checks
    def check(self) -> None:
        """Check that the problem is complete and consistent; raise one error listing every
        problem, each with what to write. ``summary()`` and fitting check first."""
        problems = self._problems()
        if problems:
            raise ProblemError(problems)

    def _problems(self) -> list[str]:
        out: list[str] = []

        def add(decl, message: str) -> None:
            origin = decl if isinstance(decl, str) or decl is None else decl.origin
            out.append(f"{origin}\n{message}" if origin else message)

        m = self._m
        fit_names = self._fit_names()
        df = self._data.table if self._data is not None else None
        columns = set(map(str, df.columns)) if df is not None else set()
        varying = {d.name: d for d in self._varying()}
        defined = {d.name for d in self._defines}

        # a new quantity (not a model parameter) must be used by a formula of the fit
        used: set[str] = set()
        for d in self._defines:
            used |= set(tq.formula(d.formula).names)
        for d in varying.values():
            for _, v in d.dist.args:
                if isinstance(v, str):
                    used |= set(tq.formula(v).names)
        for lk in self._likelihoods:
            if isinstance(lk.density, str):
                used |= set(tq.formula(lk.density).names)
            else:
                used |= {d.name for d in self._params}  # a function may use any of them
        for d in self._params:
            if d.name not in m.params and d.name not in used:
                add(
                    d,
                    f"'{d.name}' is not a parameter of the model, and no formula of this fit "
                    f"uses it.{nearest(d.name, m.params)} Parameters: "
                    f"{', '.join(m.params) or '(none)'}.",
                )
        # names in formulas: estimated or fixed quantities, model parameters, data columns
        known = fit_names | set(m.params) | columns
        for d in self._defines:
            for n in sorted(tq.formula(d.formula).names):
                if n not in known:
                    add(d, self._unknown_in_formula(n, f"{d.name} = {d.formula}", columns))
        for d in varying.values():
            for k, v in d.dist.args:
                if not isinstance(v, str):
                    continue
                for n in sorted(tq.formula(v).names):
                    if n in varying:
                        add(
                            d,
                            f"{d.name}: {k} = {v} uses '{n}', which varies between units; "
                            "a distribution's arguments use one-value quantities and columns.",
                        )
                    elif n not in (fit_names | columns):
                        add(d, self._unknown_in_formula(n, f"{d.name}: {k} = {v}", columns))
        # a model parameter is set once: by estimate, fix or a formula of the fit
        for d in self._params:
            if d.name in defined:
                add(d, f"{d.name} is both declared here and set by a formula of this fit.")

        # varying parameters need a unit; the unit is a column
        if varying and self._unit is None:
            col = "ID" if "ID" in columns else "<column>"
            first = next(iter(varying.values()))
            add(first, f"{first.name} varies between units, but no unit is named; add: unit {col}")
        if self._unit is not None and df is not None and self._unit not in columns:
            add(
                self._unit_origin,
                f"unit {self._unit}: the data have no column "
                f"'{self._unit}'.{nearest(self._unit, columns)}",
            )

        # correlations: between varying parameters, each pair once
        pairs: dict[frozenset, _Correlate] = {}
        for c in self._correlations:
            for n in c.names:
                if n not in varying:
                    add(
                        c,
                        f"correlate: {n} does not vary between units; correlate names "
                        "parameters written with a distribution (estimate ka ~ ...).",
                    )
            for i, a in enumerate(c.names):
                for b in c.names[i + 1 :]:
                    if frozenset((a, b)) in pairs:
                        add(c, f"correlate: the pair {a}, {b} is given twice.")
                    pairs[frozenset((a, b))] = c

        # links, errors, censoring
        links = [x for x in self._links if isinstance(x, _Observe)]
        linked = {_key(x.lhs): x for x in links}
        for e in self._errors:
            if _key(e.lhs) not in linked:
                add(
                    e,
                    f"error {e.lhs}: {e.lhs} is not linked to the data; add observe {e.lhs} = DV.",
                )
        for c in self._censors:
            if _key(c.lhs) not in linked:
                add(c, f"censor {c.lhs}: {c.lhs} is not linked to the data.")
            elif isinstance(c.below, str) and df is not None and c.below not in columns:
                add(
                    c,
                    f"censor {c.lhs}: the data have no column '{c.below}'."
                    f"{nearest(c.below, columns)}",
                )
            elif df is not None and c.where is not None:
                try:
                    D.evaluate_condition(c.where, df)
                except ValueError as e:
                    add(c, f"censor {c.lhs}: {e}")
            elif df is not None:  # no condition: values equal to the limit are suspect
                x = linked[_key(c.lhs)]
                if x.column in columns:
                    try:
                        rows = U.link_rows(df, x.column, x.where)
                    except ValueError:
                        rows = np.zeros(len(df), bool)
                    dv = pd.to_numeric(df[x.column], errors="coerce").to_numpy(float)
                    lim = (
                        pd.to_numeric(df[c.below], errors="coerce").to_numpy(float)
                        if isinstance(c.below, str)
                        else np.full(len(df), c.below)
                    )
                    equal = np.flatnonzero(rows & (dv == lim))
                    if len(equal) >= 2:
                        add(
                            c,
                            f"censor {c.lhs}: {len(equal)} measurements equal the limit exactly "
                            f"({U.rows_text(equal)}). If they are values below the limit written "
                            "as the limit, say how to tell them: censor "
                            f"{c.lhs} where CENS == 1 below {_arg(c.below)} (a flag column). "
                            "If they are measured values, write the rule out: censor "
                            f"{c.lhs} where {x.column} < {_arg(c.below)} below {_arg(c.below)}.",
                        )
        written = {_key(e.lhs) for e in self._likelihoods}
        for lk in self._likelihoods:
            if _key(lk.lhs) not in linked:
                add(lk, f"likelihood {lk.lhs}: {lk.lhs} is not linked to the data.")
            if any(_key(e.lhs) == _key(lk.lhs) for e in self._errors):
                add(lk, f"likelihood {lk.lhs}: give an error model or a likelihood, not both.")
            if any(_key(c.lhs) == _key(lk.lhs) for c in self._censors):
                add(
                    lk,
                    f"likelihood {lk.lhs}: a written likelihood with censoring is not "
                    "supported; write the censored part into the formula instead.",
                )
            if isinstance(lk.density, str):
                obs = set(tq.formula(lk.lhs).names)
                known = {"DV"} | obs | fit_names | set(m.params)
                for n in sorted(tq.formula(lk.density).names):
                    if n not in known:
                        add(lk, self._unknown_in_formula(n, f"likelihood {lk.lhs}", set()))
        with_error = {_key(e.lhs) for e in self._errors} | written
        for x in links:
            if _key(x.lhs) not in with_error:
                add(
                    x,
                    f"{x.lhs} has no error model; add one, e.g.: error {x.lhs} ~ "
                    "additive(<start of the SD>)  (or proportional, combined, power).",
                )
        if df is not None:
            before = len(out)
            U.row_problems(
                df, self._unit if self._unit in columns else None, add, self._data,
                self._censor_marked(df),
            )  # fmt: skip
            unit = self._unit if self._unit in columns else None
            if self._doses is not None and self._doses.column is not None:
                U.constant_problems(df, unit, self._doses.column, "the dose", add, self._doses)
            for c in self._covariates():
                U.constant_problems(df, unit, c, "a covariate", add, self._data)
            out += self._link_problems(df, add)
            out += self._dose_problems(df, add)
            if len(out) == before:  # the rows are readable: build the units
                self._build(add)

        # priors name estimated quantities
        est = set(self.estimated())
        for p in self._priors:
            if p.name not in est:
                add(
                    p,
                    f"prior {p.name}: not an estimated quantity.{nearest(p.name, est)} "
                    f"Estimated: {', '.join(self.estimated()) or '(none)'}.",
                )
        # written priors suit their quantities; under NUTS every quantity has a prior
        if est and all(p.name in est for p in self._priors):
            from .optimize import scales
            from .priors import priors

            names, x0, bounds = self._starts()
            sc = scales(names, x0, bounds, self._kinds(names))
            nuts = self._method is not None and self._method.name == "nuts"
            origin = {p.name: p for p in self._priors}
            for text in priors(self, names, x0, sc, defaults=nuts)[1]:
                name = text.split(":")[0].removeprefix("prior ")
                add(origin.get(name), text)

        # the method suits the problem
        if self._method is None:
            suggest = "laplace" if varying else "optimize"
            out.append(f"no method; add: method {suggest}")
        elif self._method.name == "optimize" and varying:
            add(
                self._method,
                "method optimize fits one set of parameters, but "
                f"{', '.join(varying)} vary between units; use method laplace (or nuts).",
            )
        elif self._method.name == "laplace" and not varying:
            add(
                self._method,
                "method laplace is for parameters that vary between units, and "
                "none does; use method optimize.",
            )
        if not self.estimated():
            out.append("nothing is estimated; add an estimate line.")
        return out

    def _starts(self) -> tuple[list[str], np.ndarray, list[tuple[float, float]]]:
        """The estimated quantities, their starts and bounds ((-inf, inf) when none is given;
        a logitnormal's median is in (0, 1), a correlation in (-1, 1))."""
        names = self.estimated()
        start, bounds = {}, {}
        for d in self._params:
            if isinstance(d, _Estimate) and d.dist is None:
                start[d.name] = d.start
                bounds[d.name] = d.bounds or (-np.inf, np.inf)
            elif isinstance(d, _Estimate):
                for k, v in d.dist.args:
                    if isinstance(v, float):
                        start[f"{d.name}.{k}"] = v
                        if d.dist.name == "logitnormal" and k == "median":
                            bounds[f"{d.name}.{k}"] = (0.0, 1.0)
        for c in self._correlations:
            for i, a in enumerate(c.names):
                for b in c.names[i + 1 :]:
                    start[f"corr({a},{b})"] = c.start or 0.0
                    bounds[f"corr({a},{b})"] = (-1.0, 1.0)
        for e in self._errors:
            for q, a in zip(e.model.quantities, e.model.args, strict=True):
                if not isinstance(a, Fixed):
                    start[f"{e.lhs}.{q}"] = float(a)
        x0 = np.array([start[n] for n in names], dtype=float)
        return names, x0, [bounds.get(n, (-np.inf, np.inf)) for n in names]

    def _kinds(self, names) -> list[str | None]:
        """The scale each family sets ("log", "as is"), or None (by the start's sign): a
        normal's mean and a lognormal's meanlog take any sign; a distribution with spread 0
        sets its quantity's scale."""
        kinds = {}
        for d in self._params:
            if isinstance(d, _Estimate) and d.kind is not None:
                kinds[d.name] = d.kind
            elif isinstance(d, _Estimate) and d.dist is not None:
                for k, _ in d.dist.args:
                    if (d.dist.name, k) in (("normal", "mean"), ("lognormal", "meanlog")):
                        kinds[f"{d.name}.{k}"] = "as is"
        return [kinds.get(n) for n in names]

    def _one_value_lines(self) -> list[str]:
        """Each quantity that does not vary, in the long form (its family, spread 0): the
        scale it is estimated on is never hidden."""
        out = []
        for d in self._params:
            if not isinstance(d, _Estimate) or d.dist is not None:
                continue
            if d.written is not None:
                out.append(f"{d.name} ~ {d.written}")
            elif d.bounds is not None:
                lo, hi = d.bounds
                out.append(f"{d.name} = {_num(d.start)} in ({_num(lo)}, {_num(hi)})")
            elif d.start > 0:
                out.append(f"{d.name} ~ lognormal(median={_num(d.start)}, sdlog=0)")
            else:
                out.append(f"{d.name} ~ normal(mean={_num(d.start)}, sd=0)")
        return out

    def _covariates(self) -> list[str]:
        """Data columns the fit's formulas use (read once per unit)."""
        if self._data is None:
            return []
        columns = set(map(str, self._data.table.columns))
        taken = self._fit_names() | set(self._m.params)
        texts = [d.formula for d in self._defines] + [
            v for d in self._varying() for _, v in d.dist.args if isinstance(v, str)
        ]
        names = {n for t in texts for n in tq.formula(t).names}
        return sorted(n for n in names if n in columns and n not in taken)

    def _censor_marked(self, df) -> np.ndarray:
        """Rows a censor line's condition marks (their DV is not read)."""
        out = np.zeros(len(df), bool)
        for c in self._censors:
            if c.where is not None:
                try:
                    out |= D.evaluate_condition(c.where, df)
                except ValueError:
                    pass  # reported by the censor checks
        return out

    def _censor_of(self, lhs):
        return next((c for c in self._censors if _key(c.lhs) == _key(lhs)), None)

    def _rows_of(self, x, df) -> np.ndarray:
        """The rows a link reads (with its censor line's marked rows)."""
        c = self._censor_of(x.lhs) if isinstance(x, _Observe) else None
        return U.link_rows(df, x.column, x.where, c.where if c else None)

    def _plan(self) -> U.Plan:
        return U.Plan(
            unit=self._unit,
            doses=self._doses,
            links=[(x.lhs, x.column, x.where) for x in self._links if isinstance(x, _Observe)],
            ignored=[(x.column, x.where) for x in self._links if isinstance(x, _Ignore)],
            censors={_key(c.lhs): (c.below, c.where) for c in self._censors},
            covariates=self._covariates(),
            reads={o: self._m.reads(o) for o in self._m.observables},
        )

    def _build(self, add) -> list[U.Unit]:
        origins = {"doses": self._doses, "data": self._data}
        return U.build(self._data.table, self._plan(), add, lambda k: origins[k])

    def _outputs(self) -> tuple[str, ...]:
        """The observables the links read: the outputs of every build."""
        names = {n for x in self._links if isinstance(x, _Observe) for n in tq.formula(x.lhs).names}
        return tuple(o for o in self._m.observables if o in names)

    # ------------------------------------------------------------------ the data as read
    def units(self) -> tuple[U.Unit, ...]:
        """The data as the fit reads them: one ``tf.Unit`` per unit, in the order they first
        appear, each with a TenQSP scenario (its doses and the times of its measurements),
        its observations by link and its covariates. Checks the problem first."""
        self.check()
        problems: list[str] = []
        units = self._build(lambda decl, msg: problems.append(msg))
        if problems:  # check() builds the units too, so this does not happen
            raise ProblemError(problems)
        return tuple(units)

    def table(self) -> pd.DataFrame:
        """The data as the fit reads them, as one NONMEM-style table: a row per dose (EVID 1,
        AMT, RATE for an infusion, CMT the compartment's name) and per linked measurement
        (EVID 0, DV, DVID the link), sorted by unit and time, with the covariates and, for a
        censored link, the limit (LLOQ). Doses set by ADDL are written out one per row.
        ``table().to_csv("data.csv", index=False, na_rep=".")`` writes it for NONMEM-style
        tools (compartments by name)."""
        rows = []
        units = self.units()
        for u in units:
            for e in U.events(u.scenario):
                inf = e["kind"] == "infusion"
                rows.append(
                    {
                        "ID": u.id,
                        "TIME": e["start"] if inf else e["time"],
                        "EVID": 1,
                        "AMT": e["rate"] * e["duration"] if inf else e["amount"],
                        "RATE": e["rate"] if inf else 0.0,
                        "CMT": e["target"],
                        "DVID": None,
                        "DV": np.nan,
                        "LLOQ": np.nan,
                        "_order": 0,
                        **u.covariates,
                    }
                )
            for lhs, o in u.observations.items():
                for i, t in enumerate(o.times):
                    rows.append(
                        {
                            "ID": u.id,
                            "TIME": t,
                            "EVID": 0,
                            "AMT": np.nan,
                            "RATE": np.nan,
                            "CMT": None,
                            "DVID": lhs,
                            "DV": o.values[i],
                            "LLOQ": np.nan if o.below is None else o.below[i],
                            "_order": 1,
                            **u.covariates,
                        }
                    )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        rank = {u.id: i for i, u in enumerate(units)}
        df["_unit"] = df["ID"].map(rank)
        df = df.sort_values(["_unit", "TIME", "_order"], kind="stable")
        df = df.drop(columns=["_unit", "_order"]).reset_index(drop=True)
        if not self._censors:
            df = df.drop(columns="LLOQ")
        if "RATE" in df and not (df["RATE"].fillna(0) > 0).any():
            df = df.drop(columns="RATE")
        return df

    def provenance(self) -> dict:
        """Where the data came from and what the fit reads of them, as recorded with every
        fit: the file and its SHA-256, the column map, the rows read, measured, dosed and
        left out (with the conditions), the units, and the model and package versions."""
        from . import __version__

        self.check()
        d, df = self._data, self._data.table
        obs, dose = D.observation_rows(df), D.dose_rows(df)
        units = self.units()
        linked = np.zeros(len(df), bool)
        for x in self._links:
            if isinstance(x, _Observe):
                linked |= self._rows_of(x, df)
        left_out = []
        for x in self._links:
            if isinstance(x, _Ignore):
                r = U.link_rows(df, x.column, x.where)
                left_out.append(
                    {"rule": f"ignore {x.column} where {x.where}", "rows": int(r.sum())}
                )
        all_units = df[self._unit].nunique() if self._unit else 1
        return {
            "data": {
                "file": d.path,
                "as written": d.source,
                "sha256": d.sha256,
                "column map": dict(d.columns),
                "rows": len(df),
                "columns": list(map(str, df.columns)),
            },
            "rows": {
                "measurements": int(obs.sum()),
                "linked": int(linked.sum()),
                "left out": left_out,
                "doses": int(dose.sum())
                if not (self._doses and self._doses.column)
                else sum(bool(U.events(u.scenario)) for u in units),
                "neither": int((~obs & ~dose).sum()),
            },
            "units": {
                "column": self._unit,
                "in the data": int(all_units),
                "with measurements": len(units),
            },
            "model": {"name": self._m.name, "hash": self._m.hash},
            "versions": {"tenqsp": tq.__version__, "tenqsp_fit": __version__},
        }

    def _unknown_in_formula(self, n: str, where: str, columns: set[str]) -> str:
        if n in self._m.definitions:
            return f"{where}: '{n}' is a formula of the model, not a quantity of the fit."
        cands = self._fit_names() | set(self._m.params) | columns
        what = "" if self._data is not None else " (no data are given, so no column either)"
        return (
            f"{where}: '{n}' is not an estimated or fixed quantity, a model parameter or a data "
            f"column{what}.{nearest(n, cands)} Declare it: estimate {n} = <start>."
        )

    def _link_problems(self, df: pd.DataFrame, add) -> list[str]:
        """Every measurement is linked once or left out on purpose; a missing link proposes
        the lines to write (adds to the problems through ``add``)."""
        out: list[str] = []
        obs = D.observation_rows(df)
        covered = np.zeros(len(df), int)
        by_row: list[tuple[object, np.ndarray]] = []
        for x in self._links:
            if x.column not in df.columns:
                hint = nearest(x.column, map(str, df.columns))
                add(x, f"the data have no column '{x.column}'.{hint}")
                continue
            rows = obs if x.column == "DV" else (df[x.column].notna().to_numpy() & ~D.dose_rows(df))
            if x.where is not None:
                try:
                    rows = rows & D.evaluate_condition(x.where, df)
                except ValueError as e:
                    add(x, str(e))
                    continue
            if not rows.any():
                add(x, f"no measurement matches{' where ' + x.where if x.where else ''}.")
            if x.column == "DV":
                covered += rows
            by_row.append((x, rows))
        twice = covered > 1
        if twice.any():
            first = [x for x, r in by_row if (r & twice).any()]
            add(
                first[1] if len(first) > 1 else first[0],
                f"{int(twice.sum())} measurement rows are linked twice; make the 'where' "
                "conditions of the observe lines exclusive.",
            )
        missing = obs & (covered == 0) & ("DV" in df.columns)
        wide = [x for x in self._links if x.column != "DV"]
        if missing.any() and not (wide and not any(x.column == "DV" for x in self._links)):
            out.append(self._propose(df, missing))
        elif not self._links:
            out.append(self._propose(df, missing))
        return out

    def _propose(self, df: pd.DataFrame, missing: np.ndarray) -> str:
        """The proposing error: what is not linked, and the lines that would link it."""
        linked = {
            n for x in self._links if isinstance(x, _Observe) for n in tq.formula(x.lhs).names
        }
        free = [o for o in self._m.observables if o not in linked] or list(self._m.observables)
        unit = self._unit if self._unit in df.columns else None

        def count(rows) -> str:
            s = f"{int(rows.sum())} rows"
            return s + (f", {df.loc[rows, unit].nunique()} units" if unit else "")

        head = (
            "The data have measurements (DV) that are not linked to the model."
            if self._links
            else "The data are not linked to the model."
        )
        lines = [head]
        proposals, leave = [], []
        split = next((c for c in ("DVID", "CMT") if c in df.columns), None)
        if missing.any() and split is not None:
            values = list(pd.unique(df.loc[missing, split]))
            parts = [
                f"{D.value_text(v)} -> {count(missing & (df[split] == v).to_numpy())}"
                for v in values
            ]
            lines.append(f"  DV rows by {split}:  " + ";  ".join(parts))
            for i, v in enumerate(values):
                obs_name = free[i] if i < len(free) else f"<one of: {', '.join(free)}>"
                proposals.append(f"observe {obs_name} = DV where {split} == {D.value_text(v)}")
            leave.append(f"ignore DV where {split} == {D.value_text(values[-1])}")
        elif missing.any():
            lines.append(f"  DV rows: {count(missing)}")
            target = free[0] if len(free) == 1 else f"<one of: {', '.join(free)}>"
            proposals.append(f"observe {target} = DV")
        for o in free:  # a wide table: a column named as an observable
            if o in df.columns and not any(x.column == o for x in self._links):
                proposals.append(f"observe {o} = {o}")
        lines.append(f"  observables of the model:  {', '.join(self._m.observables) or '(none)'}")
        if proposals:
            lines.append("Write one line per kind of measurement, for example:")
            lines += [f"  {p}" for p in proposals]
        if leave:
            lines.append(f"(or leave some out on purpose: {leave[0]})")
        return "\n".join(lines)

    def _dose_problems(self, df: pd.DataFrame, add) -> list[str]:
        out: list[str] = []
        rows = D.dose_rows(df)
        d = self._doses
        if d is None:
            if rows.any():
                lines = [
                    f"The data have {int(rows.sum())} dose rows (AMT), but no doses line "
                    "says where they go."
                ]
                if "CMT" in df.columns:
                    cmts = sorted(pd.unique(df.loc[rows, "CMT"]))
                    places = self._m.compartments
                    pairs = ", ".join(
                        f"{D.value_text(c)} into "
                        f"{places[i] if i < len(places) else '<compartment>'}"
                        for i, c in enumerate(cmts)
                    )
                    lines.append(f"Write, for example:\n  doses CMT {pairs}")
                lines.append(
                    "or, for every dose into one place:\n  doses into "
                    f"{self._m.compartments[0] if self._m.compartments else '<compartment>'}"
                )
                lines.append(f"Compartments: {', '.join(self._m.compartments) or '(none)'}.")
                out.append("\n".join(lines))
            return out
        if d.column is not None:
            if d.column not in df.columns:
                add(
                    d,
                    f"doses: the data have no column '{d.column}'."
                    f"{nearest(d.column, map(str, df.columns))}",
                )
            if rows.any():
                add(
                    d,
                    f"doses: the data also have {int(rows.sum())} dose rows (AMT); give the "
                    "doses either as rows or as a column, not both.",
                )
            return out
        if not rows.any():
            add(
                d,
                "doses: the data have no dose rows (AMT, or EVID 1); for a dose given as a "
                "column write: doses <column> at t=0 into <compartment>.",
            )
        elif d.cmt is not None:
            if "CMT" not in df.columns:
                add(d, "doses: the data have no CMT column; write doses into <compartment>.")
            else:
                unmapped = sorted(set(pd.unique(df.loc[rows, "CMT"])) - set(d.cmt))
                if unmapped:
                    add(
                        d,
                        f"doses: dose rows with CMT {', '.join(map(D.value_text, unmapped))} "
                        "have no place; add them to the doses line.",
                    )
        return out

    # ------------------------------------------------------------------ summary and text
    def summary(self) -> str:
        """The whole problem on a few lines: the data, every link with its row count, the
        doses, the units, what is estimated, fixed or kept from the model, and the method
        with its options (defaults marked). Checks the problem first."""
        self.check()
        m, df = self._m, self._data.table if self._data is not None else None
        unit = self._unit
        n_units = df[unit].nunique() if (df is not None and unit) else 1
        out = []
        head = f"{self.name}: model {m.name} ({m.n_states} states)"
        if df is not None:
            obs, doses = D.observation_rows(df), D.dose_rows(df)
            src = f'"{self._data.source}"' if self._data.source else "a table given in Python"
            n_doses = n_units if (self._doses and self._doses.column) else int(doses.sum())
            head += (
                f", data {src}: {n_units} unit{'s' * (n_units != 1)}, {int(obs.sum())} "
                f"observations, {n_doses} doses"
            )
        else:
            head += ", no data yet"
        out.append(head)

        def row(label: str, text: str) -> None:
            out.append(f"{label:<11}{text}")

        if df is not None:
            built = self._build(lambda decl, msg: None)
            empty = n_units - len(built)
            row(
                "units",
                (f"{unit}: {n_units} units" if unit else "none named: all rows are one unit")
                + (f", {empty} without linked measurements (not fitted)" if empty else ""),
            )
            for x in self._links:
                rows = self._rows_of(x, df)
                n = f"{int(rows.sum())} rows" + (
                    f", {df.loc[rows, unit].nunique()} units" if unit else ""
                )
                where = f"where {x.where}" if x.where else "(all rows)"
                if isinstance(x, _Observe):
                    cens = self._censor_of(x.lhs)
                    extra = ""
                    if cens:
                        k = int(
                            (rows & U.censored_rows(df, x.column, cens.below, cens.where)).sum()
                        )
                        how = f"where {cens.where}" if cens.where else "below the limit"
                        extra = f"   censored (M3, limit {_arg(cens.below)}): {k} rows, {how}"
                    row(
                        "links" if x is self._links[0] else "",
                        f"{x.lhs} <- {x.column} {where}   {n}{extra}",
                    )
                else:
                    row(
                        "links" if x is self._links[0] else "",
                        f"left out: {x.column} {where}   {n}",
                    )
            d = self._doses
            if d is None:
                row("doses", "none")
            elif d.column is not None:
                row(
                    "doses",
                    f"column {d.column} at t={_num(d.at)} -> {d.into}   "
                    f"{n_units} doses (one per unit)",
                )
            else:
                ndose = int(D.dose_rows(df).sum())
                to = (
                    f"AMT -> {d.into}"
                    if d.into
                    else "AMT by CMT: " + ", ".join(f"{k} -> {v}" for k, v in d.cmt.items())
                )
                row("doses", f"{to}   {ndose} rows")
        for i, e in enumerate(self._errors):
            row(
                "errors" if i == 0 else "",
                f"{e.lhs} ~ {e.model}   ({ERROR_FORMS[e.model.name]}"
                + (", f = prediction)" if "f" in ERROR_FORMS[e.model.name] else ")"),
            )
        for i, lk in enumerate(self._likelihoods):
            what = lk.density if isinstance(lk.density, str) else "a Python function"
            row(
                "errors" if i == 0 and not self._errors else "",
                f"{lk.lhs}: log-density {what} (written)",
            )
        for i, d in enumerate(self._varying()):
            row("varying" if i == 0 else "", f"{d.name} ~ {d.dist}")
        for i, c in enumerate(self._correlations):
            row(
                "correlated" if i == 0 else "",
                ", ".join(c.names)
                + (f" (start {_num(c.start)})" if c.start is not None else " (start 0)"),
            )
        est = self.estimated()
        row("estimated", f"{len(est)}: " + " ".join(est))
        if self._method.name in ("optimize", "laplace"):
            from .optimize import scale_text, scales

            names, x0, bounds = self._starts()
            row("scales", scale_text(names, scales(names, x0, bounds, self._kinds(names))))
            for i, line in enumerate(self._one_value_lines()):
                row("one value" if i == 0 else "", line)
        fixed = [f"{d.name}={_num(d.value)}" for d in self._params if isinstance(d, _Fix)]
        fixed += [
            f"{d.name}.{k}={_num(v.value)}"
            for d in self._varying()
            for k, v in d.dist.args
            if isinstance(v, Fixed)
        ]
        fixed += [
            f"{e.lhs}.{q}={_num(v.value)}"
            for e in self._errors
            for q, v in zip(e.model.quantities, e.model.args, strict=True)
            if isinstance(v, Fixed)
        ]
        row("fixed", " ".join(fixed) or "none")
        taken = self._fit_names() | {d.name for d in self._defines}
        kept = [
            f"{p.name}={_param_value(p.value)}"
            for p in m.params.values()
            if p.name not in taken and not self._feeds_fit_formula(p.name)
        ]
        row("from model", " ".join(kept) or "none")
        links = [f"{n} = {e} (model)" for n, e in m.definitions.items() if m.param_names(e) & taken]
        links += [f"{d.name} = {d.formula} (this fit)" for d in self._defines]
        row("formulas", "; ".join(links) or "none")
        meth = self._method
        if meth.name == "nuts":  # every quantity has a prior, the defaults marked
            from .optimize import scales
            from .priors import priors

            names, x0, bounds = self._starts()
            sc = scales(names, x0, bounds, self._kinds(names))
            shown = [f"{p.name} ~ {p}" for p in priors(self, names, x0, sc, defaults=True)[0]]
            shown += [f"corr({', '.join(c.names)}) ~ LKJ(2) (default)" for c in self._correlations]
        else:
            shown = [f"{p.name} ~ {p.dist}" for p in self._priors]
            if shown:
                shown[-1] += "  (used: the estimates are the posterior mode)"
        for i, text in enumerate(shown):
            row("prior" if i == 0 else "", text)
        meth = self._method
        mo = " ".join(
            f"{k}={meth.options[k]}" if k in meth.options else f"{k}={v} (default)"
            for k, v in {**METHOD_DEFAULTS.get(meth.name, {}), **meth.options}.items()
        )
        row("method", f"{meth.name} ({METHODS[meth.name][0]})" + (f", {mo}" if mo else ""))
        opts = []
        for k, default in OPTIONS.items():
            if k in self._options:
                opts.append(f"{k} {_opt(self._options[k])}")
            elif default is not None:
                opts.append(f"{k} {_opt(default)} (default)")
        row("options", ", ".join(opts))
        floored = [x.lhs for x in self._links if isinstance(x, _Observe)
                   and _key(x.lhs) not in set(self._m.observables)]  # fmt: skip
        floored += [
            f"f in {e.lhs} ~ {e.model.name}" for e in self._errors if e.model.name != "additive"
        ]
        if floored:
            atol = self._options.get("atol") or 1e-6
            row(
                "near zero",
                f"kept above 0 by a smooth floor at {_num(float(atol))} (the solver's atol): "
                + ", ".join(floored),
            )
        return _Text("\n".join(out))

    def _feeds_fit_formula(self, name: str) -> bool:
        texts = [d.formula for d in self._defines] + [
            v for d in self._varying() for _, v in d.dist.args if isinstance(v, str)
        ]
        return any(name in tq.formula(t).names for t in texts)

    def to_text(self, *, model: bool = False) -> str:
        """The problem as fit text (the part after the model's ``fit`` line). With
        ``model=True``, the model text first, so the result is one file for ``tf.load``."""
        lines = [f"fit {self.name}"]
        if self._data is not None:
            if self._data.source is None:
                lines.append(
                    f"# data: a table given in Python ({len(self._data.table)} rows), "
                    "not written here"
                )
            else:
                cols = ", ".join(f"{k}={v}" for k, v in self._data.columns.items())
                lines.append(f'data "{self._data.source}"' + (f" with {cols}" if cols else ""))
        if self._unit is not None:
            lines.append(f"unit {self._unit}")
        if (d := self._doses) is not None:
            if d.cmt is not None:
                lines.append(
                    "doses CMT " + ", ".join(f"{_arg(k)} into {v}" for k, v in d.cmt.items())
                )
            elif d.column is not None:
                lines.append(f"doses {d.column} at t={_num(d.at)} into {d.into}")
            else:
                lines.append(f"doses into {d.into}")
        for x in self._links:
            if isinstance(x, _Observe):
                lines.append(
                    f"observe {x.lhs} = {x.column}" + (f" where {x.where}" if x.where else "")
                )
            else:
                lines.append(f"ignore {x.column} where {x.where}")
        if self._links or self._data is not None:
            lines.append("")
        for d in self._defines:  # formulas first (NONMEM's $PK), then what they use
            lines.append(f"{d.name} = {d.formula}")
        for d in self._params:
            if isinstance(d, _Fix):
                lines.append(f"fix {d.name} = {_num(d.value)}")
            elif d.dist is not None or d.written is not None:
                lines.append(f"estimate {d.name} ~ {d.dist or d.written}")
            else:
                b = f" in ({_num(d.bounds[0])}, {_num(d.bounds[1])})" if d.bounds else ""
                lines.append(f"estimate {d.name} = {_num(d.start)}{b}")
        for c in self._correlations:
            lines.append(
                "correlate "
                + ", ".join(c.names)
                + (f" = {_num(c.start)}" if c.start is not None else "")
            )
        for e in self._errors:
            lines.append(f"error {e.lhs} ~ {e.model}")
        for lk in self._likelihoods:
            if isinstance(lk.density, str):
                lines.append(f"likelihood {lk.lhs}: {lk.density}")
            else:
                lines.append(f"# likelihood {lk.lhs}: a Python function (not written as text)")
        for c in self._censors:
            w = f" where {c.where}" if c.where else ""
            lines.append(f"censor {c.lhs}{w} below {_arg(c.below)}")
        for p in self._priors:
            lines.append(f"prior {p.name} ~ {p.dist}")
        if self._method is not None or self._options:
            lines.append("")
        if self._method is not None:
            mo = "".join(f" {k}={v}" for k, v in self._method.options.items())
            lines.append(f"method {self._method.name}{mo}")
        if self._options:
            lines.append("options " + " ".join(f"{k}={_opt(v)}" for k, v in self._options.items()))
        text = "\n".join(lines).rstrip("\n") + "\n"
        return (self.model.to_text().rstrip("\n") + "\n\n" + text) if model else text

    @classmethod
    def from_text(cls, text: str, *, model=None, name: str | None = None) -> Problem:
        """Read a fit text: a TenQSP model followed by a ``fit`` part, or only the fit part
        with ``model=`` (a model or a ``.tq`` file). Data files are found relative to the
        working directory (``tf.load`` reads them relative to the file)."""
        from .text import read

        return read(text, model=model, name=name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Problem):
            return NotImplemented
        return self._state() == other._state()

    __hash__ = None  # a problem is changed by its methods

    def _state(self):
        return (
            self.name,
            self._m.hash,
            self._data,
            self._unit,
            self._doses,
            self._links,
            self._params,
            self._correlations,
            self._defines,
            self._errors,
            self._likelihoods,
            self._censors,
            self._priors,
            self._method,
            self._options,
        )

    def __repr__(self) -> str:
        return f"<tenqsp_fit.Problem '{self.name}' (model {self._m.name})>"


def _arg(v) -> str:
    return _num(v) if isinstance(v, int | float) else str(v)


def _opt(v) -> str:
    return _num(v) if isinstance(v, float) else str(v)


def _param_value(v) -> str:
    return _num(v) if isinstance(v, int | float) else str(v)
