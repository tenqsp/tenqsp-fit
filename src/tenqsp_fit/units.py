# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The data as the fit reads them: one unit per value of the unit column, each with a TenQSP
scenario (its doses and output times), its measurements per link and its covariates; checks
of the rows (with row numbers and what to change)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tenqsp as tq

from . import data as D
from .errors import nearest

__all__ = ["Observations", "Unit"]

# columns TenQSP's dose-table reader takes
_DOSE_COLUMNS = ("TIME", "AMT", "RATE", "DUR", "ADDL", "II", "EVID", "MDV", "SS")


@dataclass(frozen=True, eq=False)
class Observations:
    """The measurements of one link in one unit: ``times``, ``values``, the data ``rows``
    (counted from 1 after the header) and, for a censored link, the limit of each
    measurement (``below``) and which measurements are censored (``censored``; both None when
    the link is not censored). A censored measurement's value is not read (NaN for text)."""

    link: str
    times: np.ndarray
    values: np.ndarray
    rows: np.ndarray
    below: np.ndarray | None = None
    censored: np.ndarray | None = None

    def __eq__(self, other) -> bool:  # the same data; row numbers aside
        if not isinstance(other, Observations):
            return NotImplemented
        same_below = (self.below is None) == (other.below is None) and (
            self.below is None
            or (
                np.array_equal(self.below, other.below, equal_nan=True)
                and np.array_equal(self.censored, other.censored)
            )
        )
        return (
            self.link == other.link
            and np.array_equal(self.times, other.times)
            and np.array_equal(
                np.where(self.censored, np.nan, self.values)
                if self.censored is not None
                else self.values,
                np.where(other.censored, np.nan, other.values)
                if other.censored is not None
                else other.values,
                equal_nan=True,
            )
            and same_below
        )

    __hash__ = None

    def __repr__(self) -> str:
        return f"<{len(self.times)} observations of {self.link}>"


@dataclass(frozen=True, eq=False)
class Unit:
    """One experimental unit (a subject, an animal, a flask, a well) as the fit reads it:
    its ``id``, a TenQSP ``scenario`` (its doses as a regimen and the times of its
    measurements; no parameter values: those come from the fit), its ``observations`` by
    link, and its ``covariates`` (the data columns the fit's formulas use, one value per
    unit). Two units are equal when they hold the same data, whatever the rows they came
    from."""

    id: object
    scenario: tq.Scenario
    observations: dict
    covariates: dict

    def __eq__(self, other) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return (
            self.id == other.id
            and self.scenario == other.scenario
            and self.covariates == other.covariates
            and self.observations == other.observations
        )

    __hash__ = None

    def __repr__(self) -> str:
        n = len(events(self.scenario))
        obs = ", ".join(f"{k} {len(v.times)}" for k, v in self.observations.items())
        return (
            f"<unit {self.id}: {n} dose{'s' * (n != 1)}, {len(self.scenario.times)} times; {obs}>"
        )


# ----------------------------------------------------------------------------- helpers
def events(sc: tq.Scenario) -> list[dict]:
    """A scenario's doses as TenQSP lists them; none for the model's own doses."""
    return sc.regimen.to_list() if isinstance(sc.regimen, tq.Regimen) else []


def plain_id(v):
    """A unit's value as TenQSP gives it: a whole number as an int."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def rows_text(idx) -> str:
    """Data rows (0-based positions) as the user counts them: "row 3", "rows 3, 7 and 9",
    "rows 3, 7, 9, 12, 15 and 4 more"."""
    n = [int(i) + 1 for i in idx]
    if len(n) == 1:
        return f"row {n[0]}"
    if len(n) <= 5:
        return f"rows {', '.join(map(str, n[:-1]))} and {n[-1]}"
    return f"rows {', '.join(map(str, n[:5]))} and {len(n) - 5} more"


def _have(idx) -> str:
    return "has" if len(idx) == 1 else "have"


def measurement_rows(df: pd.DataFrame, column: str) -> np.ndarray:
    """The rows holding a measurement in ``column``: for DV, the observation records; for
    another column (a wide table, or ``LNDV``), rows with a value there that are observation
    records too (EVID 0 when there is an EVID column, else not MDV 1) and give no dose."""
    if column == "DV":
        return D.observation_rows(df)
    rows = df[column].notna().to_numpy() & ~D.dose_rows(df)
    if "EVID" in df.columns:
        rows &= (df["EVID"] == 0).to_numpy()
    elif "MDV" in df.columns:
        rows &= (df["MDV"].fillna(0) != 1).to_numpy()
    return rows


def ignored_rows(df: pd.DataFrame, column: str, ignored=()) -> np.ndarray:
    """The rows of ``column`` that ``ignore`` lines leave out (``ignored``: their
    ``(column, where)``)."""
    out = np.zeros(len(df), bool)
    for col, where in ignored:
        if col == column:
            out |= link_rows(df, col, where)
    return out


def link_rows(
    df: pd.DataFrame, column: str, where: str | None, censor_where: str | None = None,
    ignored=(),
) -> np.ndarray:  # fmt: skip
    """The rows a link (or an ``ignore`` line) reads: measurements in ``column``
    (``measurement_rows``), picked by ``where``. Rows its censor line's condition marks
    (``censor conc where CENS == 1 ...``) are measurements too, whatever their DV. Rows that
    ``ignore`` lines on the same column leave out (``ignored``) are not read."""
    rows = measurement_rows(df, column)
    if censor_where is not None:
        rows = rows | (D.evaluate_condition(censor_where, df) & ~D.dose_rows(df))
    if where is not None:
        rows = rows & D.evaluate_condition(where, df)
    if ignored:
        rows = rows & ~ignored_rows(df, column, ignored)
    return rows


def censored_rows(df: pd.DataFrame, column: str, below, where: str | None) -> np.ndarray:
    """The rows a censor line marks: those meeting its condition when it has one (a flag,
    ``CENS == 1``, or a text, ``DV == "BLQ"``); otherwise those whose value is strictly below
    the limit (a number or a column)."""
    if where is not None:
        return D.evaluate_condition(where, df)
    dv = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
    lim = (
        pd.to_numeric(df[below], errors="coerce").to_numpy(float)
        if isinstance(below, str)
        else np.full(len(df), float(below))
    )
    with np.errstate(invalid="ignore"):
        return dv < lim


def _num_column(df, col, add, origin, what, exempt=None):
    """A column as numbers; text that is not a number is a problem with its rows (except the
    ``exempt`` rows: those a censor line marks)."""
    values = pd.to_numeric(df[col], errors="coerce")
    bad = values.isna().to_numpy() & df[col].notna().to_numpy()
    if exempt is not None:
        bad &= ~exempt
    bad = np.flatnonzero(bad)
    if len(bad):
        hint = "."
        if col == "DV":
            words = ", ".join(f'"{w}"' for w in pd.unique(df[col].iloc[bad])[:3])
            hint = (
                f": {words}. If a text marks values below the limit of quantification, say "
                'so: censor <observable> where DV == "BLQ" below <limit>.'
            )
        add(
            origin,
            f"{col}: {rows_text(bad)} {'is' if len(bad) == 1 else 'are'} not numbers "
            f"({what}){hint}",
        )
    return values


# ----------------------------------------------------------------------------- row checks
def row_problems(df: pd.DataFrame, unit: str | None, add, origin, censored=None) -> None:
    """Pre-flight checks of the rows: each problem names its rows and what to change.
    ``censored``: rows a censor line's condition marks (their DV is not read)."""
    cols = set(df.columns)
    exempt = censored if censored is not None else np.zeros(len(df), bool)
    obs, dose = D.observation_rows(df), D.dose_rows(df)
    used = obs | dose
    if "TIME" not in cols:
        add(
            origin,
            f"the data have no TIME column.{nearest('TIME', cols)} Map it: data ... "
            "with TIME=<column>.",
        )
        return
    time = _num_column(df, "TIME", add, origin, "times")
    missing = np.flatnonzero(time.isna().to_numpy() & used)
    if len(missing):
        add(origin, f"TIME is missing on {rows_text(missing)}; give every record its time.")
    neg = np.flatnonzero((time < 0).to_numpy() & used)
    if len(neg):
        add(
            origin,
            f"TIME is below 0 on {rows_text(neg)}; the model starts at 0. Shift the times so the "
            "first record is at 0 or later.",
        )
    if unit is not None and unit in cols:
        no_unit = np.flatnonzero(df[unit].isna().to_numpy() & used)
        if len(no_unit):
            add(origin, f"{unit} (the unit) is missing on {rows_text(no_unit)}.")
    for c in ("AMT", "DV", "EVID", "MDV", "RATE", "SS", "ADDL", "II"):
        if c in cols:
            _num_column(
                df, c, add, origin, "a NONMEM column holds numbers", exempt if c == "DV" else None
            )
    if "EVID" in cols:
        evid = pd.to_numeric(df["EVID"], errors="coerce")
        no_evid = np.flatnonzero(df["EVID"].isna().to_numpy())
        if len(no_evid):  # neither a measurement nor a dose: it would be dropped unseen
            add(
                origin,
                f"EVID is missing on {rows_text(no_evid)}; give every row its EVID (0 a "
                "measurement, 1 a dose, 2 neither).",
            )
        bad = np.flatnonzero(~evid.isin([0, 1, 2, 3, 4]).to_numpy() & df["EVID"].notna())
        if len(bad):
            add(origin, f"EVID is 0, 1, 2, 3 or 4; {rows_text(bad)} {_have(bad)} other values.")
        resets = np.flatnonzero(evid.isin([3, 4]).to_numpy())
        if len(resets):
            add(
                origin,
                f"EVID 3 or 4 (a reset) on {rows_text(resets)}: resets are not supported yet. "
                "Split the unit at the reset into two units (a new ID for each period), or give "
                "the dosing history before it.",
            )
        amt = pd.to_numeric(df["AMT"], errors="coerce").fillna(0) if "AMT" in cols else None
        if amt is not None:
            r = np.flatnonzero(((evid == 0) & (amt != 0)).to_numpy())
            if len(r):
                add(
                    origin,
                    f"{rows_text(r)} {_have(r)} EVID 0 (a measurement) and a dose amount "
                    "(AMT); a dose has EVID 1. Give the dose and the measurement on separate rows.",
                )
            r = np.flatnonzero(((evid == 1) & (amt == 0)).to_numpy())
            if len(r):
                add(origin, f"{rows_text(r)} {_have(r)} EVID 1 (a dose) but no amount (AMT).")
        if "MDV" in cols:
            mdv = pd.to_numeric(df["MDV"], errors="coerce").fillna(0)
            r = np.flatnonzero(((evid == 0) & (mdv == 1)).to_numpy())
            if len(r):
                add(
                    origin,
                    f"{rows_text(r)} {_have(r)} EVID 0 (a measurement) and MDV 1 (no measurement); "
                    "set EVID 2 for a row that is not a measurement.",
                )
        if "DV" in cols:
            r = np.flatnonzero(((evid == 0) & df["DV"].isna()).to_numpy() & ~exempt)
            if len(r):
                add(
                    origin,
                    f"{rows_text(r)} {_have(r)} EVID 0 (a measurement) but no DV. Set EVID 2 for a "
                    "row that is not a measurement; a value below the limit of quantification "
                    "is written as the limit, with a censor line (censor <observable> below "
                    "LLOQ).",
                )
    if "SS" in cols:
        ss = pd.to_numeric(df["SS"], errors="coerce").fillna(0)
        r = np.flatnonzero((ss != 0).to_numpy() & dose)
        if len(r):
            add(
                origin,
                f"steady-state doses (SS) on {rows_text(r)} are not supported yet. Give the "
                "dosing history explicitly (ADDL and II before the first measurement), or start "
                "from a pre-run.",
            )
    if "RATE" in cols:
        rate = pd.to_numeric(df["RATE"], errors="coerce").fillna(0)
        r = np.flatnonzero((rate < 0).to_numpy() & dose)
        if len(r):
            add(
                origin,
                f"RATE -1 or -2 (a rate or duration set by the model) on {rows_text(r)} is not "
                "supported yet. Give the rate (RATE) or the duration (DUR) in the data.",
            )
    # times go forward within a unit, in the order of the file
    groups = (
        df.groupby(df[unit], sort=False).indices if unit in cols else {None: np.arange(len(df))}
    )
    back = []
    t = time.to_numpy()
    for idx in groups.values():
        idx = idx[used[idx] & ~np.isnan(t[idx])]
        if len(idx) > 1:
            back += list(idx[1:][np.diff(t[idx]) < 0])
    if back:
        add(
            origin,
            f"times go back on {rows_text(back)} (earlier than the row before, in the same "
            "unit). Sort the data by unit, then time.",
        )
    dup = np.flatnonzero(df.duplicated(keep="first").to_numpy() & used)
    if len(dup):
        warnings.warn(
            f"the data repeat earlier rows exactly at {rows_text(dup)}; they are read as "
            "separate records. Remove them if they are copies.",
            stacklevel=4,
        )


def constant_problems(df, unit, column, what, add, origin) -> None:
    """A column read once per unit (a dose column, a covariate) has one value per unit."""
    if column not in df.columns:
        return
    keys = df[unit] if unit in df.columns else pd.Series(1, index=df.index)
    missing = np.flatnonzero(df[column].isna().to_numpy())
    if len(missing):
        add(origin, f"{column} ({what}) is missing on {rows_text(missing)}.")
    changing = [
        plain_id(k) for k, g in df[column].groupby(keys, sort=False) if g.dropna().nunique() > 1
    ]
    if changing:
        shown = ", ".join(map(str, changing[:5])) + (" ..." if len(changing) > 5 else "")
        add(
            origin,
            f"{column} ({what}) changes within units {shown}; it is read once per unit. "
            "Values that change with time are not supported yet.",
        )


# ----------------------------------------------------------------------------- building
@dataclass
class Plan:
    """What ``build`` needs from the problem."""

    unit: str | None
    doses: object  # the _Doses declaration, or None
    links: list  # (lhs, column, where) of each observe line
    ignored: list  # (column, where) of each ignore line
    censors: dict  # lhs key -> (limit: a number or a column, condition or None)
    covariates: list  # data columns read once per unit
    reads: dict  # observable -> compartments it reads directly


def build(df: pd.DataFrame, plan: Plan, add, origin_of) -> list[Unit]:
    """The units, in the order they first appear in the data. Problems found while
    building (TenQSP refusing a dose record, a measurement at a dose time) go to ``add``."""
    unit = plan.unit
    keys = df[unit].map(plain_id) if unit else pd.Series(1, index=df.index)
    order = list(pd.unique(keys[keys.notna()]))
    time = pd.to_numeric(df["TIME"], errors="coerce").to_numpy(float)
    regimens = _regimens(df, keys, order, plan, add, origin_of)
    link_masks = []
    for lhs, col, where in plan.links:
        limit, cwhere = plan.censors.get("".join(lhs.split()), (None, None))
        cens = censored_rows(df, col, limit, cwhere) if limit is not None else None
        rows = link_rows(df, col, where, cwhere, plan.ignored)
        link_masks.append((lhs, col, rows, limit, cens))
    units = []
    for k in order:
        in_unit = (keys == k).to_numpy()
        observations = {}
        for lhs, col, mask, limit, cens in link_masks:
            rows = np.flatnonzero(mask & in_unit)
            if not len(rows):
                continue
            below = None
            if limit is not None:
                below = (
                    np.full(len(rows), float(limit))
                    if not isinstance(limit, str)
                    else pd.to_numeric(df[limit], errors="coerce").to_numpy(float)[rows]
                )
            sort = np.argsort(time[rows], kind="stable")
            rows = rows[sort]
            observations[lhs] = Observations(
                lhs,
                time[rows],
                pd.to_numeric(df[col], errors="coerce").to_numpy(float)[rows],
                rows + 1,
                None if below is None else below[sort],
                None if cens is None else cens[rows],
            )
        if not observations:
            continue  # no linked measurement: nothing to fit (reported in the summary)
        times = np.unique(np.concatenate([o.times for o in observations.values()]))
        cov = {c: _plain_value(df.loc[in_unit, c].dropna().iloc[0]) for c in plan.covariates}
        if plan.doses is None:  # no doses line: the model's own doses (if any)
            sc = tq.Scenario(name=str(k), times=times)
        else:
            sc = tq.Scenario(name=str(k), regimen=regimens.get(k, tq.Regimen()), times=times)
        units.append(Unit(k, sc, observations, cov))
        _dose_time_problems(df, units[-1], in_unit, time, plan, add, origin_of)
    return units


def _plain_value(v):
    v = plain_id(v)
    return float(v) if isinstance(v, int) else v


def _regimens(df, keys, order, plan, add, origin_of) -> dict:
    d = plan.doses
    if d is None:
        return {}
    if d.column is not None:
        out = {}
        for k in order:
            amount = df.loc[(keys == k).to_numpy(), d.column].dropna()
            if len(amount):
                out[k] = tq.bolus(d.into, float(amount.iloc[0]), times=[d.at])
        return out
    rows = D.dose_rows(df)
    if not rows.any():
        return {}
    table = df.loc[rows, [c for c in _DOSE_COLUMNS if c in df.columns]].copy()
    table.insert(0, "ID", keys[rows].to_numpy())
    if d.into is not None:
        table["CMT"] = d.into
        targets = None
    else:
        table["CMT"] = df.loc[rows, "CMT"].map(plain_id).to_numpy()
        targets = d.cmt
    try:
        return {plain_id(k): r for k, r in tq.regimens_from_table(table, targets=targets).items()}
    except ValueError as e:  # TenQSP refuses a record: say so, do not read it differently
        add(origin_of("doses"), f"TenQSP cannot read the dose records: {e}")
        return {}


def _dose_time_problems(df, u: Unit, in_unit, time, plan, add, origin_of) -> None:
    """A measurement at the time of a dose into a compartment it reads directly: TenQSP gives
    the value just after the dose, NONMEM the value just before when the measurement row
    comes first. Unambiguous only when the dose row comes first in the file. Only a bolus
    jumps the value; at the start of an infusion before and after are the same."""
    doses = [(e["time"], e["target"]) for e in events(u.scenario) if e["kind"] == "bolus"]
    if not doses:
        return
    dose_rows = np.flatnonzero(D.dose_rows(df) & in_unit) if plan.doses.column is None else []
    clash = []
    for lhs, o in u.observations.items():
        reads = set().union(*(plan.reads.get(n, set()) for n in tq.formula(lhs).names))
        for t_dose, target in doses:
            if target not in reads:
                continue
            for r, t in zip(o.rows - 1, o.times, strict=True):
                if t != t_dose:
                    continue
                before = [x for x in dose_rows if time[x] == t and x < r]
                if not before:
                    clash.append((r, t, target, lhs))
    if clash:
        r, t, target, lhs = clash[0]
        add(
            origin_of("data"),
            f"{rows_text(sorted({c[0] for c in clash}))}: a measurement of {lhs} at the time of "
            f"a dose into {target} (first: t={D.value_text(t)}, unit {u.id}), which {lhs} reads "
            "directly. TenQSP gives the value just after the dose; NONMEM reads a measurement "
            "recorded before the dose (or at a dose time set by ADDL) as the value just before "
            "it. Values just before a dose are not supported yet: if these are pre-dose "
            "samples, give them a time just before the dose; if they are after the dose, put "
            "the dose row first.",
        )
