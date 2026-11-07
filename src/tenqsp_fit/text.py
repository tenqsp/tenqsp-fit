# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Reading fit text: the part after a model's ``fit`` line.

Lexical rules are TenQSP's: one statement per line, ``#`` comments, and a line continues the
statement above when a bracket is open, when it is indented, when it starts with an operator,
or when the line above ends with a comma. Formulas are read by TenQSP (``tq.formula``). Each
statement is one call of ``Problem``'s Python API, so text and Python mean the same.
"""

from __future__ import annotations

import ast
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import tenqsp as tq

from .distributions import Fixed, make_distribution, make_error
from .errors import FitTextError, nearest
from .problem import Problem

__all__ = ["load", "read"]

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_FIT_LINE = re.compile(rf"fit(\s+({_NAME}))?")
_AT_START = re.compile(rf"at\s+{_NAME}\s*=")
_OPENERS, _CLOSERS = "([{", ")]}"
KEYWORDS = (
    "fit data unit doses observe ignore estimate fix correlate error likelihood censor prior "
    "method options"
).split()


# ----------------------------------------------------------------------------- statements
@dataclass
class _Statement:
    lines: list[tuple[int, str]]

    @property
    def lineno(self) -> int:
        return self.lines[0][0]

    @property
    def text(self) -> str:
        return " ".join(t.strip() for _, t in self.lines)

    @property
    def origin(self) -> str:
        return f"line {self.lineno}: {self.lines[0][1].strip()}"


def _strip_comment(line: str) -> str:
    quote = None
    for i, c in enumerate(line):
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#":
            return line[:i]
    return line


def _depth_change(text: str) -> int:
    d, quote = 0, None
    for c in text:
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in _OPENERS:
            d += 1
        elif c in _CLOSERS:
            d -= 1
    return d


def _statements(text: str) -> list[_Statement]:
    out: list[_Statement] = []
    depth = 0
    for no, raw in enumerate(textwrap.dedent(text.expandtabs(4)).splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        cont = (
            bool(out)
            and not (depth == 0 and _AT_START.match(stripped))
            and (
                depth > 0
                or line[0] in " \t"
                or stripped[0] in "+-*/"
                or out[-1].lines[-1][1].rstrip().endswith(",")
                or (
                    re.match(r"(elif\b|else\s*:)", stripped) is not None
                    and out[-1].lines[0][1].strip().startswith("if ")
                )
            )
        )
        if cont:
            out[-1].lines.append((no, line))
        else:
            out.append(_Statement([(no, line)]))
            depth = 0
        depth = max(depth + _depth_change(line), 0)
    return out


def _split_top(text: str, sep: str = ",") -> list[str]:
    parts, d, quote, start = [], 0, None, 0
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in _OPENERS:
            d += 1
        elif c in _CLOSERS:
            d -= 1
        elif c == sep and d == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def _top(text: str, pattern: str) -> re.Match | None:
    """The first match of ``pattern`` outside brackets and quotes."""
    d, quote = 0, None
    rx = re.compile(pattern)
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c in _OPENERS:
            d += 1
        elif c in _CLOSERS:
            d -= 1
        elif d == 0:
            m = rx.match(text, i)
            if m:
                return m
    return None


_ASSIGN = r"(?<![=<>!])=(?!=)"
_WHERE = r"(?<![A-Za-z0-9_])where(?![A-Za-z0-9_])"


def _split_at(text: str, pattern: str) -> tuple[str, str] | None:
    m = _top(text, pattern)
    return None if m is None else (text[: m.start()].strip(), text[m.end() :].strip())


class _Problem(Exception):
    pass


# ----------------------------------------------------------------------------- values
def _number(text: str, what: str) -> float:
    """A number, possibly as arithmetic of numbers (``4/100``, ``-1e-9``, ``log10(100)``)."""
    try:
        f = tq.formula(text)
    except (ValueError, TypeError):
        raise _Problem(f"{what}: '{text}' is not a number.") from None
    if f.names:
        raise _Problem(f"{what}: '{text}' is not a number.")
    return float(f.jax([])())


def _bound(text: str) -> float:
    """A bound: a number, or ``inf``/``-inf`` for an open side."""
    t = text.strip().replace(" ", "")
    if t in ("inf", "+inf", "-inf"):
        return -math.inf if t == "-inf" else math.inf
    return _number(text, "bounds")


def _call(text: str, what: str) -> tuple[str, list[ast.expr], list[ast.keyword], str]:
    """A call such as ``lognormal(median=1.5, sdlog=0.5)``: name, arguments, source."""
    src = text.strip()
    try:
        node = ast.parse(src, mode="eval").body
    except SyntaxError:
        node = None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise _Problem(f"{what}: write it as a call, e.g. {_EXAMPLE[what]}.")
    return node.func.id, node.args, node.keywords, src


_EXAMPLE = {
    "distribution": "lognormal(median=1.5, sdlog=0.5)",
    "error model": "additive(0.7)",
    "prior": "lognormal(median=0.04, sdlog=0.5)",
}


def _value(node: ast.expr, src: str, what: str):
    """An argument: a number, fix(number), or a formula (as text)."""
    seg = ast.get_source_segment(src, node).strip()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fix":
        if len(node.args) != 1 or node.keywords:
            raise _Problem(f"{what}: write fix(value), e.g. fix(0.5).")
        return Fixed(_number(ast.get_source_segment(src, node.args[0]), what))
    try:
        f = tq.formula(seg)
    except (ValueError, TypeError) as e:
        raise _Problem(f"{what}: {e}") from None
    return float(f.jax([])()) if not f.names else seg


def _distribution(text: str, what: str, *, between_units: bool):
    name, args, kws, src = _call(text, what)
    if args:
        raise _Problem(
            f"{what}: name each argument, e.g. {name}(median=1.5, sdlog=0.5)"
            if name == "lognormal"
            else f"{what}: name each argument, e.g. {name}(mean=..., sd=...)."
        )
    kwargs = {}
    for k in kws:
        if k.arg is None:
            raise _Problem(f"{what}: name each argument.")
        kwargs[k.arg] = _value(k.value, src, f"{name}: '{k.arg}'")
    if not between_units and any(isinstance(v, str) for v in kwargs.values()):
        raise _Problem(f"{what}: the arguments of a prior are numbers.")
    return make_distribution(name, kwargs, between_units=between_units)


# ----------------------------------------------------------------------------- statements
class _Reader:
    def __init__(self, problem: Problem, base: Path | None):
        self.p = problem
        self.base = base
        self.problems: list[str] = []

    def report(self, st: _Statement, message: str) -> None:
        self.problems.append(f"{st.origin}\n{message}")

    def statement(self, st: _Statement) -> None:
        text = st.text
        word = text.split()[0]
        rest = text[len(word) :].strip()
        n_before = self._count()
        try:
            if word in KEYWORDS:
                getattr(self, "st_" + word)(rest, st)
            elif re.match(rf"{_NAME}\s*{_ASSIGN}", text):
                name, formula = _split_at(text, _ASSIGN)
                self.p.define(name, formula)
            else:
                self.unknown(st, word)
                return
        except _Problem as e:
            self.report(st, str(e))
            return
        except (ValueError, TypeError) as e:
            self.report(st, str(e))
            return
        self._tag(n_before, st.origin)

    # the problem's lists, to tag new declarations with their line
    def _lists(self):
        p = self.p
        return [p._links, p._params, p._correlations, p._defines, p._errors, p._censors, p._priors]

    def _count(self):
        return [len(x) for x in self._lists()]

    def _tag(self, before, origin: str) -> None:
        for lst, n in zip(self._lists(), before, strict=True):
            for d in lst[n:]:
                d.origin = origin

    def unknown(self, st: _Statement, word: str) -> None:
        if word in (
            "model",
            "param",
            "observable",
            "compartment",
            "axis",
            "field",
            "flux",
        ) or word.startswith("d_t"):
            self.report(st, "a model statement after the 'fit' line; the model goes above it.")
        else:
            self.report(
                st,
                f"cannot read this statement.{nearest(word, KEYWORDS)} A fit statement starts "
                f"with a keyword ({', '.join(KEYWORDS[1:])}) or sets a model parameter by a "
                "formula (ka = mu).",
            )

    # ------------------------------------------------------------------ one per keyword
    def st_fit(self, rest: str, st: _Statement) -> None:
        raise _Problem("a second 'fit' line; one fit per text.")

    def st_data(self, rest: str, st: _Statement) -> None:
        m = re.fullmatch(r"(\"[^\"]*\"|'[^']*')\s*(?:with\s+(.*))?", rest, re.S)
        if not m:
            raise _Problem(
                'write the file name in quotes: data "theoph.csv" (with ID=Subject, '
                "... to map columns)."
            )
        columns = {}
        if m.group(2):
            for item in _split_top(m.group(2)):
                kv = re.fullmatch(rf"({_NAME})\s*=\s*({_NAME}|\"[^\"]*\"|'[^']*')", item)
                if not kv:
                    raise _Problem(f"cannot read the column map '{item}'; write ID=Subject.")
                columns[kv.group(1)] = kv.group(2).strip("\"'")
        self.p._set_data(m.group(1)[1:-1], columns, base=self.base, origin=st.origin)

    def st_unit(self, rest: str, st: _Statement) -> None:
        self.p.unit(rest)
        self.p._unit_origin = st.origin

    def st_doses(self, rest: str, st: _Statement) -> None:
        if m := re.fullmatch(rf"into\s+({_NAME})", rest):
            self.p.doses(into=m.group(1))
        elif m := re.fullmatch(r"CMT\s+(.*)", rest, re.S):
            cmt = {}
            for item in _split_top(m.group(1)):
                kv = re.fullmatch(rf"(\S+)\s+into\s+({_NAME})", item)
                if not kv:
                    raise _Problem(
                        f"cannot read '{item}'; write doses CMT 1 into depot, 2 into centr."
                    )
                k = _number(kv.group(1), "doses: a CMT number")
                if k in cmt:
                    raise _Problem(f"CMT {kv.group(1)} is given twice.")
                cmt[k] = kv.group(2)
            self.p.doses(cmt={int(k) if k.is_integer() else k: v for k, v in cmt.items()})
        elif m := re.fullmatch(rf"({_NAME})\s+at\s+t\s*=\s*(\S+)\s+into\s+({_NAME})", rest):
            self.p.doses(column=m.group(1), at=_number(m.group(2), "doses: at"), into=m.group(3))
        else:
            raise _Problem(
                "write doses into depot, doses CMT 1 into depot, 2 into centr, or "
                "doses Dose at t=0 into depot."
            )
        self.p._doses.origin = st.origin

    def st_observe(self, rest: str, st: _Statement) -> None:
        body, where = _split_at(rest, _WHERE) or (rest, None)
        parts = _split_at(body, _ASSIGN)
        if parts is None:
            raise _Problem("write observe conc = DV (the observable, '=', the data column).")
        lhs, column = parts
        self.p.observe(lhs, column, where=where)

    def st_ignore(self, rest: str, st: _Statement) -> None:
        parts = _split_at(rest, _WHERE)
        if parts is None:
            raise _Problem("write ignore DV where DVID == 2.")
        self.p.ignore(parts[0], where=parts[1])

    def st_estimate(self, rest: str, st: _Statement) -> None:
        for item in _split_top(rest):
            if (parts := _split_at(item, "~")) is not None:
                name, dist = parts
                self.p.estimate(name, _distribution(dist, "distribution", between_units=True))
                continue
            parts = _split_at(item, _ASSIGN)
            if parts is None:
                raise _Problem(
                    "write estimate ke = 0.08 (one value, from this start) or estimate ka ~ "
                    "lognormal(median=1.5, sdlog=0.5) (varies between units)."
                )
            name, value = parts
            bounds = None
            if (b := _split_at(value, r"(?<![A-Za-z0-9_])in(?![A-Za-z0-9_])")) is not None:
                value, interval = b
                m = re.fullmatch(r"\((.*),(.*)\)", interval)
                if not m:
                    raise _Problem(
                        f"write the bounds as an interval: estimate {name} = {value} in (0, 1), "
                        "or in (0, inf) for a bound on one side."
                    )
                bounds = tuple(_bound(g) for g in m.groups())
            if value.startswith("fix("):
                raise _Problem(f"a fixed value is written fix {name} = value.")
            self.p.estimate(name, _number(value, f"estimate {name}"), bounds=bounds)

    def st_fix(self, rest: str, st: _Statement) -> None:
        for item in _split_top(rest):
            parts = _split_at(item, _ASSIGN)
            if parts is None:
                raise _Problem("write fix V = 30.")
            self.p.fix(parts[0], _number(parts[1], f"fix {parts[0]}"))

    def st_correlate(self, rest: str, st: _Statement) -> None:
        names, start = _split_at(rest, _ASSIGN) or (rest, None)
        names = _split_top(names)
        self.p.correlate(*names, start=None if start is None else _number(start, "correlate"))

    def st_error(self, rest: str, st: _Statement) -> None:
        parts = _split_at(rest, "~")
        if parts is None:
            raise _Problem("write error conc ~ additive(0.7).")
        lhs, model = parts
        name, args, kws, src = _call(model, "error model")
        if kws:
            raise _Problem(f"error model: give the numbers in order, e.g. {name}(0.1, 0.2).")
        values = [_value(a, src, f"{name}") for a in args]
        if any(isinstance(v, str) for v in values):
            raise _Problem(f"{name}: the arguments are numbers (or fix(number)).")
        self.p.error(lhs, make_error(name, values))

    def st_likelihood(self, rest: str, st: _Statement) -> None:
        parts = _split_at(rest, ":")
        if parts is None:
            raise _Problem(
                "write likelihood conc: <log-density>, e.g. likelihood conc: "
                "-0.5*((DV - conc)/s)^2 - log(s)."
            )
        lhs, density = parts
        self.p.likelihood(lhs, density)

    def st_censor(self, rest: str, st: _Statement) -> None:
        parts = _split_at(rest, r"(?<![A-Za-z0-9_])below(?![A-Za-z0-9_])")
        if parts is None:
            raise _Problem(
                "write censor conc below LLOQ (a column) or below 0.1 (a number); with a flag: "
                "censor conc where CENS == 1 below LLOQ."
            )
        lhs, below = parts
        below = below if re.fullmatch(_NAME, below) else _number(below, "censor: below")
        where = None
        split = _split_at(lhs, r"(?<![A-Za-z0-9_])where(?![A-Za-z0-9_])")
        if split is not None:
            lhs, where = split
        self.p.censor(lhs, below=below, where=where)

    def st_prior(self, rest: str, st: _Statement) -> None:
        parts = _split_at(rest, "~")
        if parts is None:
            raise _Problem("write prior CL ~ lognormal(median=0.04, sdlog=0.5).")
        name, dist = parts
        self.p.prior(name, _distribution(dist, "prior", between_units=False))

    def st_method(self, rest: str, st: _Statement) -> None:
        if not rest:
            raise _Problem("write method optimize, method laplace or method nuts.")
        name, *opts = rest.split(None, 1)
        self.p.method(name, **_options(opts[0] if opts else ""))
        self.p._method.origin = st.origin

    def st_options(self, rest: str, st: _Statement) -> None:
        self.p.options(**_options(rest))


def _options(text: str) -> dict:
    out = {}
    for item in re.split(r"[\s,]+", text.strip()):
        if not item:
            continue
        kv = re.fullmatch(rf"({_NAME})=(\S+)", item)
        if not kv:
            raise _Problem(f"cannot read '{item}'; write options as key=value, e.g. starts=8.")
        k, v = kv.groups()
        if k in out:
            raise _Problem(f"{k} is given twice.")
        if re.fullmatch(r"[+-]?\d+", v):
            out[k] = int(v)
        elif re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", v):
            out[k] = float(v)
        else:
            out[k] = v.strip("\"'")
    return out


# ----------------------------------------------------------------------------- entry points
def read(
    text: str,
    *,
    model=None,
    name: str | None = None,
    base: Path | None = None,
) -> Problem:
    """Read a model text with a fit part (or a fit part alone, with ``model=``)."""
    statements = _statements(text)
    fit_at = next(
        (
            i
            for i, st in enumerate(statements)
            if len(st.lines) == 1 and _FIT_LINE.fullmatch(st.lines[0][1].strip())
        ),
        None,
    )
    if model is None:
        if fit_at is None:
            raise FitTextError(
                ["no 'fit' line: the fitting part starts with a line fit <name>, below the model."]
            )
        model = tq.Model.from_text(text, name=name)  # TenQSP reads up to the fit line
        body = statements[fit_at + 1 :]
    else:
        if isinstance(model, str | Path):
            model = tq.load(model)
        if fit_at is not None and fit_at > 0:
            raise FitTextError(
                [
                    f"{statements[0].origin}\nwith model= the text is the fit part "
                    "alone; the model goes in its own file."
                ]
            )
        body = statements if fit_at is None else statements[1:]
    fit_name = None
    if fit_at is not None:
        fit_name = _FIT_LINE.fullmatch(statements[fit_at].lines[0][1].strip()).group(2)
    p = Problem(model, name=fit_name or name or model.name)
    reader = _Reader(p, base)
    for st in body:
        reader.statement(st)
    if reader.problems:
        raise FitTextError(reader.problems)
    return p


def load(path: str | Path, *, model=None) -> Problem:
    """Read a fitting problem from a ``.tq`` file: a TenQSP model followed by its ``fit``
    part, or the fit part alone with ``model=`` (a model or its ``.tq`` file). The data file
    is found relative to the fit file."""
    p = Path(path)
    if p.suffix != ".tq":
        raise ValueError(f"tf.load reads .tq files, not '{p.name}'.")
    if isinstance(model, str | Path):
        mp = Path(model)
        model = tq.load(mp if mp.is_absolute() or mp.exists() else p.parent / mp)
    # the file's name names the problem when its fit line has none ("theoph-pop v2" ->
    # theoph_pop_v2)
    name = re.sub(r"\W+", "_", p.stem).strip("_") or "fit"
    if name[0].isdigit():
        name = "fit_" + name
    return read(p.read_text(encoding="utf-8"), model=model, name=name, base=p.parent)
