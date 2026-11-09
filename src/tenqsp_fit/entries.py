# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Entries of a parameter with one value per category.

A model parameter ``param k[strain] = {S: 1e-8, R: 1e-10}`` has one entry per category,
named as TenQSP names them: ``k[S]``, ``k[R]``; over two axes ``V[liver, tissue]``. In a fit
each entry is one parameter: estimated, fixed or set by a formula on its own.
``k[strain]`` (the axis in brackets) names every entry of the table at once.

TenQSP's formula language reads one value per name, so a fit formula that uses an entry
(``k[R] = k[S]*f``) is read with each entry replaced by a plain name (``formula``).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import tenqsp as tq

__all__ = ["Entry", "Table", "entry_name", "formula", "merged", "split", "to_tenqsp"]

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
# a name with labels in brackets: k[S], V[liver, tissue], k[strain]
_NAME_RE = re.compile(_NAME)
ENTRY = re.compile(rf"({_NAME})\s*\[\s*({_NAME}(?:\s*,\s*{_NAME})*)\s*\]")


def entry_name(table: str, labels: tuple[str, ...]) -> str:
    """``k[S]``, ``V[liver, tissue]``."""
    return f"{table}[{', '.join(labels)}]"


def split(name: str) -> tuple[str, tuple[str, ...]] | None:
    """``("k", ("S",))`` from ``k[S]`` (spaces allowed); None for a plain name."""
    m = ENTRY.fullmatch(name.strip())
    if m is None:
        return None
    return m.group(1), tuple(s.strip() for s in m.group(2).split(","))


@dataclass(frozen=True)
class Entry:
    """One entry of a per-category parameter, as the fit sees it: a parameter of its own."""

    name: str  # k[S]
    table: str  # k
    labels: tuple[str, ...]  # ("S",)
    value: float | None
    over: None = None  # one value, as a scalar parameter


@dataclass(frozen=True)
class Table:
    """A per-category parameter: its axes, and the label combinations in TenQSP's order."""

    name: str
    axes: tuple[str, ...]
    labels: tuple[tuple[str, ...], ...]  # one tuple per entry, first axis slowest

    @property
    def entries(self) -> list[str]:
        return [entry_name(self.name, lab) for lab in self.labels]

    def categories(self, i: int) -> list[str]:
        """The categories of axis ``i``, in order."""
        return list(dict.fromkeys(lab[i] for lab in self.labels))

    def bracket(self) -> str:
        """``k[strain]``."""
        return entry_name(self.name, self.axes)


def tables(ir) -> dict[str, Table]:
    """The model's per-category parameters."""
    out = {}
    for p in ir.params:
        if p.over is None:
            continue
        axes = (p.over,) if isinstance(p.over, str) else tuple(p.over)
        labels = tuple(itertools.product(*(tuple(ir.axis(a).labels) for a in axes)))
        out[p.name] = Table(p.name, axes, labels)
    return out


def entries(ir, tabs: dict[str, Table]) -> dict[str, Entry]:
    """Every entry of every per-category parameter, with the model's value (None when the
    model gives none)."""
    out = {}
    for p in ir.params:
        if p.over is None:
            continue
        t = tabs[p.name]
        values = (None,) * len(t.labels) if p.value is None else tuple(p.value)
        for lab, v in zip(t.labels, values, strict=True):
            n = entry_name(p.name, lab)
            out[n] = Entry(n, p.name, lab, None if v is None else float(v))
    return out


def to_tenqsp(params: dict, tabs: dict[str, Table]) -> dict:
    """Parameter values for TenQSP: entries (``{"k[S]": 1e-8}``) gathered by table
    (``{"k": {"S": 1e-8}}``, nested for several axes); other names as they are."""
    out: dict = {}
    for name, v in params.items():
        s = split(name) if "[" in name else None
        if s is None or s[0] not in tabs:
            out[name] = v
            continue
        table, labels = s
        d = out.setdefault(table, {})
        for lab in labels[:-1]:
            d = d.setdefault(lab, {})
        d[labels[-1]] = v
    return out


def merged(base, params: dict, tabs: dict[str, Table]) -> dict:
    """TenQSP parameter values ``base`` (as a scenario holds them) with ``params`` (entries
    or plain names) put in; the other categories of a table keep their values."""

    def put(d: dict, new: dict) -> dict:
        out = dict(d)
        for k, v in new.items():
            out[k] = put(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    return put({k: v for k, v in dict(base).items()}, to_tenqsp(params, tabs))


class Formula:
    """A fit formula that may use entries (``k[S]*f``): ``names`` lists them as written
    (``k[S]``, ``f``), and ``jax(names)`` takes values in that order. A formula without
    entries is TenQSP's formula as it is."""

    def __init__(self, text: str):
        self.text = text
        self._token: dict[str, str] = {}  # plain name -> entry

        def sub(m: re.Match) -> str:
            labels = tuple(s.strip() for s in m.group(2).split(","))
            token = "__".join((m.group(1), *labels)) + "__entry"
            self._token[token] = entry_name(m.group(1), labels)
            return token

        # an entry read by label in a model (TenQSP's V['spleen'], the notation's
        # V[organ=spleen]) is the entry V[spleen]; a label that is not one word stands for
        # the whole table
        text_ = re.sub(
            r"'([^']*)'|\"([^\"]*)\"",
            lambda m: lab if _NAME_RE.fullmatch(lab := m.group(1) or m.group(2) or "") else "_",
            text,
        )
        self._f = tq.formula(ENTRY.sub(sub, text_))
        self._name = {v: k for k, v in self._token.items()}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._token.get(n, n) for n in self._f.names)

    def jax(self, names):
        return self._f.jax([self._name.get(n, n) for n in names])


def formula(text: str) -> Formula:
    """Read a fit formula; entries (``k[S]``) are names of their own."""
    return Formula(text)
