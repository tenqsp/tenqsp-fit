# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The passport: one document per fit saying what the data determine and
what they do not, with every default, the checks and how the run can be repeated.
``result.passport()``; ``print`` shows it as text, ``.markdown()`` as Markdown,
``.to_dict()`` as plain data (JSON); ``result.save(folder)`` writes ``passport.md`` and
``passport.json``.

Sections, in order: (1) the run; (2) each quantity: estimate, uncertainty, how
well it is determined, its profile's shape when one was computed, its prior and, for NUTS,
how much the data narrowed it; (3) ridges and the condition number; (4) the
doctor's findings; (5) the numerical checks; (6) named predictions
(7) TenQSP's own reports: ``validate()``, ``diagnostics()`` at the estimate with how
well the ledger closed, and ``explain()`` with its numerical safeguards.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

__all__ = ["Passport", "passport"]

_NUMERICAL = ("tolerance check", "gradient check", "failed runs", "jumps in the model",
              "convergence", "tolerance and gradient checks")  # fmt: skip


def _data_lines(prov) -> list[str]:
    """The data file, its column map and the rows left out, from ``problem.provenance()``."""
    if not prov:
        return []
    d, rows = prov["data"], prov["rows"]
    out = [f"data file {d['as written']} SHA-256 {d['sha256']}"]
    if d["column map"]:
        out.append("columns: " + ", ".join(f"{k}={v}" for k, v in d["column map"].items()))
    out.append(f"rows: {d['rows']}; measurements {rows['measurements']}, linked "
               f"{rows['linked']}; doses {rows['doses']}")  # fmt: skip
    for r in rows["left out"]:
        out.append(f"left out: {r['rows']} row(s) by {r['rule']}")
    return out


class Passport:
    """A fit's passport: ``sections`` (title, content) in order, ``data`` (the same as plain
    values); ``markdown()``, ``to_dict()``, ``save(path)``; ``print`` shows it as text."""

    def __init__(self, title: str, sections: list[tuple[str, object]], data: dict):
        self.title, self.sections, self.data = title, sections, data

    def to_dict(self) -> dict:
        """The passport as plain values (numbers, text, lists, dicts), for JSON."""
        return self.data

    def markdown(self) -> str:
        """The passport as Markdown (tables as Markdown tables)."""
        out = [f"# {self.title}", ""]
        for title, content in self.sections:
            out += [f"## {title}", ""]
            if isinstance(content, pd.DataFrame):
                out += [_md_table(content), ""]
            else:
                out += [*(f"- {line}" for line in content), ""]
        return "\n".join(out)

    def save(self, path) -> None:
        """Write the passport: Markdown to ``path`` (``.md``) or plain data (``.json``)."""
        from pathlib import Path

        path = Path(path)
        if path.suffix == ".json":
            path.write_text(json.dumps(self.data, indent=2, default=_plain) + "\n",
                            encoding="utf-8")  # fmt: skip
        else:
            path.write_text(self.markdown() + "\n", encoding="utf-8")

    def __repr__(self) -> str:
        out = [self.title, "=" * len(self.title)]
        for title, content in self.sections:
            out += ["", title, "-" * len(title)]
            if isinstance(content, pd.DataFrame):
                out.append(content.to_string(index=False))
            else:
                out += list(content)
        return "\n".join(out)


def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [" | ".join(_cell(v) for v in r) for r in df.itertuples(index=False)]
    return "\n".join([" | ".join(cols), " | ".join("---" for _ in cols), *rows])


def _cell(v) -> str:
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.4g}"
    return str(v).replace("|", "/")


# ----------------------------------------------------------------------------- building
def passport(result) -> Passport:
    """Build the passport of ``result`` (the doctor's findings, run if needed)."""
    from .results import run_record

    doctor = result.doctor()
    run = run_record(result)
    sections: list[tuple[str, object]] = []
    data: dict = {"run": run}

    # 1. the run
    p = result.problem
    opts = ", ".join(f"{k}={v}" for k, v in {**run["method options"], **run["options"]}.items())
    lines = [
        f"problem {p.name}: method {result.method} ({result.optimizer}), {result.status}",
        f"-2LL {result.m2ll:.4f} ({result.n_observations} measurements"
        + (f", {result.objective.n} units" if getattr(result, "modes", None) is not None else "")
        + ")",
        f"options: {opts or 'none written'} (rtol {p._options.get('rtol', 1e-6):g} unless written)",
        f"problem text (model and fit) SHA-256 {run['hashes']['problem.tq']}",
        *_data_lines(run.get("provenance")),
        f"data as read SHA-256 {run['hashes']['data.csv']}",
        "versions: " + ", ".join(f"{k} {v}" for k, v in run["versions"].items()),
        f"platform {run['platform']}; {run['seconds']:.0f} s",
        "all units run as one build of the model (TenQSP pads the shorter ones with slots at "
        "t = +inf; this changes the results by about the solver's tolerance)",
    ]
    near_zero = p._near_zero()
    if near_zero:
        lines.append(f"near zero: {near_zero}")
    if getattr(result, "with_priors", None) is not None:
        lines.append("written priors used: the estimates are the posterior mode")
    sections.append(("The run", lines))

    # 2. each quantity
    q = _quantities(result, doctor)
    sections.append(("Each quantity", q))
    data["quantities"] = q.to_dict(orient="records")

    # 3. ridges and combinations
    ridge = [f for f in doctor.findings if f.check == "ridge"]
    det = [f for f in doctor.findings if f.check == "determination"]
    lines = [f.message for f in ridge] or ["no ridge: no direction weaker than 1e-3 of the "
                                           "strongest"]  # fmt: skip
    lines += [f.message for f in det]
    corr = result.corr.to_numpy()
    if result.method != "nuts" and np.all(np.isfinite(corr)):  # NONMEM's $COV PRINT=E
        eig = np.linalg.eigvalsh(corr)
        ratio = f" (largest over smallest {eig[-1] / eig[0]:.3g})" if eig[0] > 0 else ""
        lines.append("eigenvalues of the correlation matrix of the estimates: "
                     + ", ".join(f"{v:.3g}" for v in eig) + ratio)  # fmt: skip
        data["correlation eigenvalues"] = [float(v) for v in eig]
    if result.method == "nuts":
        lines = ["for NUTS, ridges show as quantities the data say little about (below and "
                 "in the findings); the priors bound them."]  # fmt: skip
    sections.append(("Ridges and determined combinations", lines))
    data["ridges"] = [f.evidence for f in ridge]

    # 4. findings
    main = [f for f in doctor.findings if f.check not in _NUMERICAL + ("determination",)]
    sections.append(("The doctor's findings", _finding_lines(main) or ["none"]))
    data["findings"] = [vars(f) for f in doctor.findings]

    # 5. numerical checks
    num = [f for f in doctor.findings if f.check in _NUMERICAL]
    lines = _finding_lines(num)
    if result.method != "nuts":
        from .doctor import failed_text

        lines.append(f"{failed_text(result)}; starts redrawn: "
                     f"{int(getattr(result, 'redrawn', 0))}")  # fmt: skip
        data["failed_runs"] = {"count": int(getattr(result, "failed_runs", 0)),
                               "evaluations": int(getattr(result, "evaluations", 0)),
                               "causes": dict(getattr(result, "failure_causes", {}))}  # fmt: skip
    if result.method == "laplace":
        lines.append("gradient check: not for laplace (its gradient is itself a difference)")
    sections.append(("Numerical checks", lines))

    # 6. named predictions
    preds = getattr(result, "_predictions", {}) or {}
    lines = [f"{name}: {pr.summary()}" for name, pr in preds.items()] or [
        "none (result.predict(..., name=) adds one)"]  # fmt: skip
    sections.append(("Named predictions", lines))
    data["predictions"] = {name: pr.summary() for name, pr in preds.items()}

    # 7. TenQSP's reports
    lines, tq_data = _tenqsp_reports(result)
    sections.append(("TenQSP's reports", lines))
    data["tenqsp"] = tq_data
    return Passport(f"Passport: {p.name}", sections, data)


def _finding_lines(findings) -> list[str]:
    out = []
    for f in findings:
        line = f"{f.severity}, {f.check}: {f.message}"
        if f.remedy:
            line += f" What to do: {f.remedy}"
        out.append(line)
    return out


def _quantities(result, doctor) -> pd.DataFrame:
    t = result.table()
    names = list(result.names)
    ridge = {n for f in doctor.findings if f.check == "ridge"
             for n in f.evidence.get("quantities", [])}  # fmt: skip
    weak = {n for f in doctor.findings if f.check == "weakly determined"
            for n in f.evidence.get("quantities", [])}  # fmt: skip
    nuts = result.method == "nuts"
    contraction = {}
    for f in doctor.findings:
        if f.check == "prior and data":
            contraction = f.evidence
    profiles = getattr(result, "_profiles", {}) or {}
    rows = []
    for i, n in enumerate(names):
        r = t.iloc[i]
        row = {
            "quantity": n,
            "estimate" if not nuts else "posterior median": float(r.iloc[1]),
            "SE" if not nuts else "posterior SD": float(r["SD"] if nuts else r["SE"]),
            "CI 2.5 %": float(r["CI 2.5 %"]),
            "CI 97.5 %": float(r["CI 97.5 %"]),
            "determined": "not (ridge)" if n in ridge else "weakly" if n in weak else "yes",
            "profile": _shape(profiles[n]) if n in profiles else "",
            "prior": _prior(result, n),
        }
        if nuts:
            c = contraction.get(n)
            row["posterior SD / prior SD"] = float(c) if c is not None else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


FLAT = 1.0  # a profile side that rose by less than this (-2LL) is flat


def _shape(prof) -> str:
    """A profile's shape: "closed" (past the cut-off on both sides); otherwise each side
    that did not reach the cut-off is "flat" (it rose by less than 1) or "not reached"
    (rising, but the points profiled stop short: profile further). A set in several pieces
    says so first; a profile point better than the fit is added."""
    d = prof.table.sort_values("value")
    rise = d["-2LL"].to_numpy() - prof.m2ll
    sides = []
    for name, v, end in (
        ("below", prof.interval[0], rise[0]),
        ("above", prof.interval[1], rise[-1]),
    ):
        if math.isfinite(v):
            continue
        if not math.isfinite(end) or end < FLAT:
            sides.append(f"flat {name}")
        else:
            sides.append(f"not reached {name} (profile further)")
    text = "closed" if not sides else ", ".join(sides)
    pieces = getattr(prof, "pieces", None) or []
    if len(pieces) > 1:
        text = f"{len(pieces)} pieces, {text}"
    if getattr(prof, "better", None) is not None:
        text += "; a better fit found"
    return text


def _prior(result, name: str) -> str:
    for p in getattr(result, "priors", None) or []:
        if p.name == name:
            return str(p)
    if result.method == "nuts" and name.startswith("corr("):
        return "LKJ(2) (default)"
    for p in result.problem._priors:
        if p.name == name:
            return f"{p.dist} (written)"
    return ""


def _tenqsp_reports(result) -> tuple[list[str], dict]:
    m = result.problem.model
    out, data = [], {}
    report = m.validate(raise_on_error=False)
    issues = [str(i) for i in report.issues]
    out += ["validate(): " + ("no issues" if not issues else f"{len(issues)} issue(s):")]
    out += [f"  {i}" for i in issues]
    data["validate"] = issues
    try:
        sol, unit = _solve_one_unit(result)
        diag = m.diagnostics(sol)
        lines = [str(i) for i in diag.issues]
        closure = max(sol.ledger.check().values(), default=0.0) if sol.ledger else float("nan")
        out.append(f"diagnostics() for unit {unit} at the estimate: "
                   + ("no issues" if not lines else f"{len(lines)} issue(s):"))  # fmt: skip
        out += [f"  {i}" for i in lines]
        out.append(f"ledger at the estimate (unit {unit}): the largest amount not accounted "
                   f"for is {closure:.2g} of the amounts involved")  # fmt: skip
        data["diagnostics"] = lines
        data["ledger closure"] = closure
    except Exception as e:  # noqa: BLE001 (the passport reports it, it does not fail)
        out.append(f"diagnostics() at the estimate: not available ({type(e).__name__}: {e})")
        data["diagnostics"] = f"not available: {e}"
    text = m.explain()
    out.append("explain():")
    out += [f"  {line}" for line in text.splitlines() if line.strip()]
    data["explain"] = text
    return out, data


def _solve_one_unit(result):
    """The first unit's run at the estimate (its own values, at the units' scores for a
    population fit), with TenQSP's scipy backend and the fit's tolerances."""
    import jax.numpy as jnp

    from .laplace import Laplace

    obj = result.objective
    cond = obj._cond if isinstance(obj, Laplace) else obj
    x = jnp.asarray(result.x)
    Z = getattr(result, "modes", None)
    P = np.asarray(cond._model_params(x, None if Z is None else jnp.asarray(Z)))
    sc = cond._scenarios[0]
    from . import entries as E

    params = E.merged(sc.params, dict(zip(cond._wrt_arg, map(float, P[0]), strict=True)),
                      result.problem._m.tables)  # fmt: skip
    times = np.asarray(sc.times, float)
    start = sc.start if sc.start is not None else 0.0
    opts = result.problem._options
    kw = {"rtol": float(opts.get("rtol") or 1e-6)}
    if opts.get("atol") is not None:
        kw["atol"] = float(opts["atol"])
    import tenqsp as tq

    if isinstance(sc.regimen, tq.Regimen):
        kw["regimen"] = sc.regimen
    sol = result.problem.model.solve((start, float(times.max())), params=params,
                                     times=times, **kw)  # fmt: skip
    return sol, cond.units[0]
