# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The doctor: what a fit can and cannot tell, run at the end of every ``tf.fit``.
``result.doctor()`` returns its findings; ``print(result)`` ends with them.

Each finding has a ``check``, a ``severity`` ("problem": the results are wrong or
meaningless as they stand; "warning": read them with care; "note": for the record), what was
found with its numbers, and what to do.

- **ridge**: the curvature of -2LL on the estimation scale (log for a positive
  quantity; the Hessian the standard errors come from) has a direction weaker than
  ``RIDGE`` (1e-3) of its strongest, as a ratio of singular values (square roots of the
  eigenvalues): along it the fit does not change. Named by the quantities whose loading is at
  least ``LOADING`` (0.3); for two of them, the product that is determined (``k x beta``).
- **weakly determined**: a 95 % interval spanning more than a factor 10 (a quantity on the
  log scale), covering more than half of its range (within bounds, e.g. a correlation), or
  an RSE above 50 % (others); partners with a correlation above 0.9 named.
- **several optima**: starts ending more than 0.1 apart in -2LL.
- **at a bound, a spread near 0**; **shrinkage** above 30 %.
- **tolerance check**: -2LL again at rtol / 100 (and atol / 100 when set): a problem
  if it changes by more than 0.1, or if the change of its gradient would move the estimates
  by more than 0.1 SE (one Newton step). NUTS: minus the log posterior at 20 draws.
- **gradient check**: the exact directional derivative against central differences
  at rtol / 100, along 3 random directions, at 3 points drawn from the estimates' normal
  approximation where -2LL is within the 99 % chi-square bound of the best (NUTS: at 3
  draws); a warning above 1 %. Not for ``laplace``, whose gradient is itself a difference.
- **jumps in the model**: extinction groups, lag times, and the jumps TenQSP's
  ``validate`` notes (``floor``, ``ceil``, ``rem``, conditions switching cell by cell).
- **failed runs**: counted with their causes; a warning above 1 % of the fit's runs, unless
  every failed run checked was far from the estimates.
- **scales**: a quantity held above 0 by the log scale (chosen by a positive start) whose
  written prior puts more than 5 % below 0, or which shares its prior, or its table
  (``k[S]``, ``k[R]``), with quantities that may be negative: a note.
- for NUTS, the convergence checks and how much the data narrowed each prior (posterior SD
  over prior SD, on the estimation scale).

The tolerance and gradient checks solve the model again (theophylline: about 8 s; an
infection-age model: about 100 s); ``options checks=quick`` leaves them out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

__all__ = ["Doctor", "Finding", "examine"]

RIDGE = 1e-3  # singular-value ratio below which a direction is a ridge
LOADING = 0.3  # loadings named in a ridge
WEAK_FACTOR = 10.0  # a log-scale quantity whose 95 % interval spans more than this factor
WEAK_RANGE = 0.5  # within bounds: a 95 % interval covering more than this part of the range
WEAK_RSE = 50.0  # others: RSE % above this
PARTNER = 0.9  # |correlation| that names a partner
OPTIMA_GAP = 0.1  # -2LL apart: different optima
SPREAD_ZERO = 0.01  # a spread (sd, sdlog, cv, sdlogit) below this; a var below its square
SHRINKAGE = 0.30
FAILED_SHARE = 0.01  # failed runs above this share of the fit's runs: a warning
TIGHTEN = 100.0  # the tolerance and gradient checks run at rtol / TIGHTEN
TOL_M2LL = 0.1  # tolerance check: largest change of -2LL
TOL_SHIFT = 0.1  # tolerance check: largest move of an estimate, in SEs
GRAD_DIRECTIONS = 3
GRAD_STEP = 1e-3  # central differences on the estimation scale
GRAD_REL = 0.01
NUTS_POINTS = 20  # draws for the NUTS tolerance check
CONTRACTION = 0.5  # NUTS: posterior SD above this fraction of the prior SD
_ORDER = {"problem": 0, "warning": 1, "note": 2}
_LOG = ("log", "above", "below")


@dataclass
class Finding:
    """One finding: ``check``, ``severity`` ("problem", "warning", "note"), ``message``,
    ``remedy`` (what to do; may be empty) and ``evidence`` (the numbers, by name)."""

    check: str
    severity: str
    message: str
    remedy: str = ""
    evidence: dict = field(default_factory=dict)

    def __str__(self) -> str:
        text = f"{self.severity:<8} {self.check}: {self.message}"
        if self.remedy:
            text += f"\n         what to do: {self.remedy}"
        return text


class Doctor:
    """The doctor's findings on a fit (most severe first): ``findings`` (a list of
    ``Finding``), ``table()``; ``print`` lists them."""

    def __init__(self, findings, *, quick: bool = False):
        self.findings = sorted(findings, key=lambda f: _ORDER[f.severity])
        self.quick = quick

    def table(self) -> pd.DataFrame:
        """One row per finding: check, severity, finding, what to do."""
        return pd.DataFrame(
            {
                "check": [f.check for f in self.findings],
                "severity": [f.severity for f in self.findings],
                "finding": [f.message for f in self.findings],
                "what to do": [f.remedy for f in self.findings],
            }
        )

    def count(self, severity: str) -> int:
        return sum(f.severity == severity for f in self.findings)

    def headline(self) -> str:
        parts = []
        for s in ("problem", "warning", "note"):
            n = self.count(s)
            if n:
                parts.append(f"{n} {s}{'s' * (n != 1)}")
        return "doctor: " + (", ".join(parts) if parts else "nothing to report")

    def __repr__(self) -> str:
        return "\n".join([self.headline(), *(str(f) for f in self.findings)])


# ----------------------------------------------------------------------------- the checks
def examine(result, *, quick: bool | None = None, say=None) -> Doctor:
    """Run every check on ``result``; ``quick`` (default: ``options checks=quick``) leaves
    out the tolerance and gradient checks. ``say``: a function for progress text."""
    if quick is None:
        quick = result.problem._options.get("checks", "full") == "quick"
    say = say or (lambda text: None)
    out: list[Finding] = []
    nuts = result.method == "nuts"
    if result.status == "stopped":
        by = getattr(result, "stopped_by", None)
        out.append(Finding("stopped", "note", "the fit was stopped before it ended"
                           + (f", by {by}" if by else "") + ": the findings are about where "
                           "it stopped."))  # fmt: skip
    if nuts:
        out += _convergence(result)
        out += _contraction(result)
        out += _weak_posterior(result)
    else:
        out += _converged(result)
        out += _determination(result)
        out += _optima(result)
        out += _bounds(result)
        out += _spreads(result)
        out += _shrinkage(result)
        out += _failed(result)
        out += profile_findings(getattr(result, "_profiles", {}) or {})
    out += _jumps(result)
    out += _scales(result)
    if quick:
        out.append(Finding("tolerance and gradient checks", "note", "left out "
                           "(options checks=quick)."))  # fmt: skip
    elif result.status != "stopped":
        try:
            say("checking the fit: the solver tolerance and the gradients ...")
            if result.problem._m.n_states == 0:  # a model given by formulas of t
                out.append(Finding("tolerance check", "note", "not needed: the model is given "
                                   "by formulas of t, so no solver runs."))  # fmt: skip
            else:
                out += _tolerance_nuts(result) if nuts else _tolerance(result)
            if result.method != "laplace":
                out += _gradients_nuts(result) if nuts else _gradients(result)
        except KeyboardInterrupt:
            out.append(Finding("tolerance and gradient checks", "note", "stopped (Ctrl-C) "
                               "before they ended."))  # fmt: skip
    return Doctor(out, quick=quick)


def _label(name: str, s: tuple) -> str:
    return f"log {name}" if s[0] in _LOG else f"logit {name}" if s[0] == "logit" else name


def _power(e: float) -> str:
    e = abs(e)
    return "" if abs(e - 1) < 0.05 else f"^{e:.2g}"


def _combination(v, names, sc) -> tuple[str, list[str]]:
    """How a ridge direction ``v`` (estimation scale) reads: the text and the quantities."""
    idx = [i for i in np.argsort(-np.abs(v)) if abs(v[i]) >= LOADING]
    who = [names[i] for i in idx]
    if len(idx) == 1:
        return f"{who[0]} is not determined (the fit does not change along it)", who
    if len(idx) == 2 and all(sc[i][0] in _LOG for i in idx):
        i, j = idx
        c = np.array([-v[j], v[i]])  # the combination orthogonal to v: determined
        c = c / np.max(np.abs(c))
        if c[0] < 0:
            c = -c
        top = [f"{names[k]}{_power(e)}" for k, e in zip(idx, c, strict=True) if e > 0]
        bottom = [f"{names[k]}{_power(e)}" for k, e in zip(idx, c, strict=True) if e < 0]
        prod = " x ".join(top) + (" / " + " x ".join(bottom) if bottom else "")
        return f"{who[0]} and {who[1]} are not determined apart, only {prod} is", who
    w = v[idx] / np.max(np.abs(v[idx]))
    move = " : ".join(f"{c:+.2f}" for c in w)
    labels = ", ".join(_label(names[i], sc[i]) for i in idx)
    return (f"{', '.join(who[:-1])} and {who[-1]} are not determined apart: the fit does not "
            f"change when {labels} move together as {move}"), who  # fmt: skip


def _determination(result) -> list[Finding]:
    """Ridges, and quantities weakly determined (from the curvature of the standard errors)."""
    H = np.asarray(getattr(result, "hessian_u", np.nan), float)
    out = []
    status = getattr(result, "covariance_status", "ok")
    if status.startswith("failed"):
        out.append(Finding("standard errors", "problem", status + ".",
                           "see the ridge below, if any; otherwise check that the fit "
                           "converged (result.plot_trace())."))  # fmt: skip
    elif status.startswith("conditional"):
        out.append(Finding("standard errors", "note", status + ".",
                           "if the bound is not part of the model, widen it, or fix the "
                           "quantity and say so."))  # fmt: skip
    if H.ndim != 2 or not np.all(np.isfinite(H)):
        return out
    names, sc = list(result.names), result.scales
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    top = float(np.max(w))
    if top <= 0:
        return out + [Finding(
            "not at a minimum", "problem",
            "-2LL does not curve upward in any direction at the estimates: the fit did not "
            "reach a minimum, and nothing can be said about what the data determine.",
            "better starts (values from an earlier fit, or more starts), or bounds.",
        )]  # fmt: skip
    in_ridge: set[str] = set()
    flat = []  # (ratio, direction, its quantities)
    for k in range(len(w)):
        if w[k] < -(RIDGE**2) * top:  # curves down, beyond round-off: not a ridge
            who = sorted(_combination(V[:, k], names, sc)[1], key=names.index)
            out.append(Finding(
                "not at a minimum", "problem",
                f"-2LL curves downward at the estimates along a direction of {_and_names(who)}: "
                "the fit stopped at a saddle or before the minimum, not at a minimum.",
                "refit from the estimates (or with more starts) and check the trace "
                "(result.plot_trace()).",
                {"quantities": who, "direction": dict(zip(names, map(float, V[:, k]),
                                                          strict=True))},
            ))  # fmt: skip
            continue
        ratio = math.sqrt(max(w[k], 0.0) / top)
        if ratio < RIDGE:
            flat.append((ratio, V[:, k], _combination(V[:, k], names, sc)[1]))
    for group in _groups(flat):
        who = sorted({n for _, _, q in group for n in q}, key=names.index)
        in_ridge.update(who)
        ratio = min(r for r, _, _ in group)
        if len(group) == 1:
            text = _combination(group[0][1], names, sc)[0]
        elif len(group) >= len(who):
            text = (
                f"none of {_and_names(who)} is determined, nor any combination of them "
                f"(the fit does not change in {len(group)} directions among them)"
            )
        else:
            text = (
                f"{_and_names(who)} are not determined apart: the fit does not change in "
                f"{len(group)} directions among them"
            )
        remedy = (
            f"measure what informs {who[0]}, write a prior on it, or fix it and say so."
            if len(who) == 1
            else "measure what tells them apart, write a prior on some of them, or estimate "
            "the determined combination (a formula after fit)."
            if len(group) < len(who)
            else "measure what informs them, write priors, or fix them and say so."
        )
        out.append(Finding(
            "ridge", "problem", f"{text} (curvature ratio {ratio:.1e}).", remedy,
            {"ratio": ratio, "quantities": who,
             "directions": [dict(zip(names, map(float, v), strict=True)) for _, v, _ in group]},
        ))  # fmt: skip
    C = np.asarray(getattr(result, "cov_u", np.nan), float)
    if status != "ok" or C.ndim != 2 or not np.all(np.isfinite(C)):
        return out
    se_u = np.sqrt(np.diag(C))
    R = C / np.outer(se_u, se_u)
    table = result.table()
    weak = []
    for i, n in enumerate(names):
        if n in in_ridge:
            continue
        row = table.iloc[i]
        what = _weakness(
            n, sc[i], float(row["CI 2.5 %"]), float(row["CI 97.5 %"]), se_u[i], float(row["RSE %"])
        )
        if what is None:
            continue
        partners = [f"{names[j]} (correlation {R[i, j]:.2f})" for j in range(len(names))
                    if j != i and abs(R[i, j]) > PARTNER]  # fmt: skip
        if partners:
            what += ", which moves with " + " and ".join(partners)
        weak.append((n, what))
    if weak:
        out.append(Finding(
            "weakly determined", "warning",
            _and([w for _, w in weak]) + ".",
            "a written prior, or fix the quantity and say so; or measurements that inform it.",
            {"quantities": [n for n, _ in weak]},
        ))  # fmt: skip
    cond = top / float(np.min(w)) if np.min(w) > 0 else math.inf
    out.append(Finding("determination", "note", f"condition number of the curvature "
                       f"{cond:.3g}" + ("" if out else "; no ridge, every quantity determined "
                                        "within a factor 10") + ".",
                       evidence={"condition number": cond}))  # fmt: skip
    return out


def _weakness(name, s, lo, hi, spread_u, rse) -> str | None:
    """How a weakly determined quantity reads (None when it is not): ``lo``, ``hi`` its 95 %
    interval, ``spread_u`` its SE (or posterior SD) on the estimation scale."""
    if s[0] in _LOG:
        factor = math.exp(2 * 1.96 * spread_u)
        if factor <= WEAK_FACTOR:
            return None
        return f"{name} (its 95 % interval spans a factor {factor:.3g})"
    if s[0] == "logit":
        part = (hi - lo) / (s[2] - s[1])
        if not part > WEAK_RANGE:
            return None
        return f"{name} (95 % interval {lo:.3g} to {hi:.3g}, {100 * part:.0f} % of its range)"
    if not rse > WEAK_RSE:
        return None
    return f"{name} (95 % interval {lo:.3g} to {hi:.3g})"


def _groups(flat):
    """Flat directions sharing a quantity, together (each group one finding)."""
    groups: list[list] = []
    for item in flat:
        joined = [g for g in groups if set(item[2]) & {n for _, _, q in g for n in q}]
        merged = [item] + [x for g in joined for x in g]
        groups = [g for g in groups if g not in joined] + [merged]
    return groups


def _and_names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else "; ".join(items[:-1]) + "; and " + items[-1]


def _converged(result) -> list[Finding]:
    if result.status in ("converged", "stopped"):
        return []
    return [Finding(
        "convergence", "problem",
        f"the fit ended with status '{result.status}': the estimates are where it stopped, "
        "not a minimum.",
        "more iterations (options maxiter=), better starts, or bounds; check "
        "result.plot_trace().",
    )]  # fmt: skip


def _optima(result) -> list[Finding]:
    v = result.starts["-2LL"].to_numpy(float)
    n = len(v)
    ok = np.sort(v[np.isfinite(v)])
    failed = n - len(ok)
    if n == 1:
        return [Finding("optima", "note", "one start: other optima were not looked for.",
                        "options starts=8 looks from 8 starts.")]  # fmt: skip
    if not len(ok):
        return []
    groups = [[ok[0]]]
    for a in ok[1:]:
        if a - groups[-1][0] > OPTIMA_GAP:  # within 0.1 of the group's best
            groups.append([a])
        else:
            groups[-1].append(a)
    lost = f"; {failed} failed" if failed else ""
    if len(groups) == 1:
        return [Finding("optima", "note", f"all {len(ok)} starts reached the same fit{lost}.")]
    parts = [f"-2LL {g[0]:.4f} ({len(g)} start{'s' * (len(g) != 1)})" for g in groups]
    from .results import CHI2_95

    what = (
        f"the next fit is within {CHI2_95:.2f} of the best, so the data hardly prefer one: the "
        "standard errors describe the best fit only, while a profile (result.profile(name)) "
        "gives an interval that covers the values of both fits. "
        if groups[1][0] - groups[0][0] < CHI2_95
        else ""
    )
    return [Finding(
        "several optima", "warning",
        f"the starts ended at {len(groups)} different fits: {', '.join(parts)}{lost}; the "
        "best is reported.",
        f"{what}if few starts reached the best, run more starts (options starts=); compare "
        "the fits in result.starts.",
        {"optima": [float(g[0]) for g in groups], "starts": [len(g) for g in groups]},
    )]  # fmt: skip


def _bounds(result) -> list[Finding]:
    from .optimize import near_bound_text

    return [Finding("at a bound", "warning", t) for t in near_bound_text(result)]


def _spreads(result) -> list[Finding]:
    from .distributions import SPREADS

    out = []
    for vname, _ in getattr(getattr(result, "objective", None), "varying", ()) or ():
        for key in SPREADS:
            name = f"{vname}.{key}"
            if name not in result.estimates:
                continue
            v = result.estimates[name]
            limit = SPREAD_ZERO**2 if key == "var" else SPREAD_ZERO
            if v < limit:
                out.append(Finding(
                    "spread near 0", "warning",
                    f"{name} is {v:.3g}: {vname} hardly varies between units; its interval "
                    "is unreliable this close to 0 (a boundary).",
                    f"consider a spread of 0 ({key}=0: {vname} the same in every unit).",
                ))  # fmt: skip
    return out


def _may_be_negative(s: tuple) -> bool:
    """Whether a quantity on the estimation scale ``s`` can take a value below 0."""
    return s[0] in ("as is", "below") or (s[0] in ("logit", "above") and s[1] < 0)


def _prior_below_zero(dist) -> float:
    """The weight a written prior puts below 0 (0 for a family on positive values)."""
    from scipy.stats import norm

    a = {k: (v.value if hasattr(v, "value") else v) for k, v in dist.args}
    if not all(isinstance(v, int | float) for v in a.values()):
        return 0.0
    if dist.name == "normal":
        sd = (
            a["sd"]
            if "sd" in a
            else math.sqrt(a["var"])
            if "var" in a
            else a["cv"] * abs(a["mean"])
        )
        return float(norm.cdf(-a["mean"] / sd)) if sd > 0 else float(a["mean"] < 0)
    if dist.name == "uniform":
        lo, hi = a["lower"], a["upper"]
        return min(1.0, max(0.0, -lo / (hi - lo))) if hi > lo else 0.0
    return 0.0


def _scales(result) -> list[Finding]:
    """A quantity held above 0 by its scale (log, from a positive start) although its prior
    allows negative values, or although quantities sharing its prior (or its table, k[S] and
    k[R]) may be negative: the sign of the start chose the scale, a common slip."""
    names, sc = list(result.names), getattr(result, "scales", None)
    if sc is None:
        return []
    neg = {n: _may_be_negative(s) for n, s in zip(names, sc, strict=True)}
    # on the log scale by its start alone: no bounds, no family that sets the scale
    by_start = {
        n
        for n, s, k in zip(names, sc, result.problem._kinds(names), strict=True)
        if s[0] == "log" and k is None
    }
    priors = {p.name: p.dist for p in result.problem._priors if p.name in neg}
    groups: dict[str, list[str]] = {}
    for n, d in priors.items():
        groups.setdefault(f"prior {d}", []).append(n)
    for n in names:
        if "[" in n and "." not in n:
            groups.setdefault(f"table {n.split('[')[0]}", []).append(n)
    out = []
    for key, members in groups.items():
        held = [n for n in members if n in by_start]
        free = [n for n in members if neg[n]]
        if not held:
            continue
        what = key.split(" ", 1)[1]
        weight = _prior_below_zero(priors[members[0]]) if key.startswith("prior") else 0.0
        if free:
            why = (
                f"while {_and_names(free)}, with the same prior {what}, "
                if key.startswith("prior")
                else f"while {_and_names(free)} of the same table {what}, "
            ) + f"{'is' if len(free) == 1 else 'are'} estimated as is"
        elif weight > 0.05:
            its = "its" if len(held) == 1 else "their"
            why = f"but {its} prior {what} puts {weight:.0%} of its weight below 0"
        else:
            continue
        out.append(Finding(
            "scales", "note",
            f"{_and_names(held)} {'is' if len(held) == 1 else 'are'} estimated on the log "
            f"scale (a positive start and no bounds), so {'it' if len(held) == 1 else 'they'} "
            f"cannot become negative, {why}.",
            f"if {'it' if len(held) == 1 else 'they'} may be negative, give bounds spanning "
            f"zero (estimate {held[0]} = <start> in (-1, 1)) or a start of 0.",
            {"held above 0": held, "may be negative": free},
        ))  # fmt: skip
    return out


def _shrinkage(result) -> list[Finding]:
    sh = getattr(result, "shrinkage", {}) or {}
    high = {k: v for k, v in sh.items() if v > SHRINKAGE}
    if not high:
        return []
    items = ", ".join(f"{k} {100 * v:.0f} %" for k, v in high.items())
    return [Finding(
        "shrinkage", "warning",
        f"shrinkage above {100 * SHRINKAGE:.0f} %: {items}. The units' own values of "
        f"{'these' if len(high) > 1 else 'it'} are pulled towards the typical value, so plots "
        "of them (and of their relation to covariates) are unreliable.",
        "judge the model by the VPC, not by the units' values.",
    )]  # fmt: skip


def failed_text(result) -> str:
    """The line "failed runs during the fit: n of m (p %); causes ...", as the passport has it."""
    n = int(getattr(result, "failed_runs", 0) or 0)
    m = int(getattr(result, "evaluations", 0) or 0)
    text = f"failed runs during the fit: {n}"
    if m:
        text += f" of {m} ({100 * n / m:.3g} %)"
    causes = getattr(result, "failure_causes", {}) or {}
    if causes:
        seen = sum(causes.values())
        of = f" (the first {seen})" if seen < n else ""
        text += f"; causes{of}: " + "; ".join(f"{k} ({v})" for k, v in causes.items())
    return text


def _failed(result) -> list[Finding]:
    n = int(getattr(result, "failed_runs", 0) or 0)
    if not n:
        return []
    m = int(getattr(result, "evaluations", 0) or 0)
    near = getattr(result, "failures_near", None)  # None: a result saved before it was kept
    seen = sum((getattr(result, "failure_causes", {}) or {}).values())
    if m and n > FAILED_SHARE * m and near != 0:
        where = ""
        if near is not None and seen:
            where = f" ({near} of the {seen} checked near the estimates)"
        return [Finding(
            "failed runs", "warning",
            f"{failed_text(result)}{where}: more than {100 * FAILED_SHARE:.0f} % of the fit's "
            "runs failed, so the fit spent much of its work outside the region where the "
            "model runs, and may have stopped at the edge of it.",
            "check the start values and bounds; obj.status(x) at the estimates says why a run "
            "fails; a tighter rtol or bounds that keep the parameters where the model runs "
            "may help.",
        )]  # fmt: skip
    far = ""
    if near == 0 and seen:
        far = (
            f"; {'all ' if seen > 1 else ''}{seen} checked {'were' if seen > 1 else 'was'} "
            "far from the estimates (a factor over 100 away in some quantity)"
        )
    return [Finding("failed runs", "note", f"{failed_text(result)}{far}; the fit went "
                    "around them (refused steps).")]  # fmt: skip


def _jumps(result) -> list[Finding]:
    model = result.problem.model
    ir = model.to_ir()
    out = []
    safe = (
        "gradient methods (all three) are safe only if it does not switch near the "
        "estimate: check a profile of the quantities involved (result.profile)."
    )
    for e in ir.extinctions:
        out.append(Finding(
            "jumps in the model", "warning",
            f"extinction group {e.name} (threshold {e.threshold:g}): where a population "
            "crosses it, the run changes abruptly and -2LL jumps.", safe,
        ))  # fmt: skip
    for lag in ir.lags:
        out.append(Finding(
            "jumps in the model", "warning",
            f"lag time {lag.value} on the doses into {lag.target}: -2LL jumps where a dose's "
            "time plus the lag passes a measurement time.", safe,
        ))  # fmt: skip
    report = model.validate(raise_on_error=False)
    for issue in report.issues:
        text = issue.message
        if "jumps" in text or "switches cell by cell" in text:
            out.append(Finding("jumps in the model", "warning", f"{issue.where}: {text}", safe))
    return out


# ----------------------------------------------------------------------------- tolerance
def _tighter(cond, problem):
    """The tolerances of the checks (the problem's / ``TIGHTEN``; atol only when set) and
    that rtol. The compiled functions take the tolerances as an input
    (``Objective._tolerant``), so the checks rerun them without a new compile."""
    return cond.tighter(TIGHTEN), float(problem._options.get("rtol") or 1e-6) / TIGHTEN


def _tolerance(result) -> list[Finding]:
    from .laplace import Laplace
    from .optimize import to_u, to_x

    obj = result.objective
    cond = getattr(obj, "_cond", obj)
    tight, rtol2 = _tighter(cond, result.problem)
    rtol = float(result.problem._options.get("rtol") or 1e-6)
    sc = result.scales
    x = jnp.asarray(result.x)
    u = jnp.asarray(to_u(np.asarray(result.x), sc))
    if isinstance(obj, Laplace):
        zeros = jnp.zeros((obj.n, obj.q))
        v1 = float(obj._jit_value(x, zeros)[0])
        v2, Z2 = obj._jit_value(x, zeros, tight)
        v2 = float(v2)
        grad = jax.jit(cond._tolerant(lambda w, Zk: obj._gradient(w, Zk, lambda v: to_x(v, sc))))
        dg = np.asarray(grad(u, Z2, tol=tight)) - np.asarray(
            grad(u, jnp.asarray(result.modes), tol=cond._tol0)
        )
    else:
        v1, v2 = float(obj._jit(x)), float(obj._jit(x, tight))
        grad = jax.jit(cond._tolerant(jax.grad(lambda w: obj._minus2ll(to_x(w, sc)))))
        dg = np.asarray(grad(u, tol=tight)) - np.asarray(grad(u, tol=cond._tol0))
    d = abs(v2 - v1)
    H = np.asarray(getattr(result, "hessian_u", np.nan), float)
    C = np.asarray(getattr(result, "cov_u", np.nan), float)
    shift = math.nan
    if H.ndim == 2 and np.all(np.isfinite(H)) and np.all(np.isfinite(C)):
        step = -np.linalg.lstsq(H, dg, rcond=None)[0]
        shift = float(np.max(np.abs(step) / np.sqrt(np.diag(C))))
    ev = {"rtol": rtol, "tighter rtol": rtol2, "change of -2LL": d, "shift in SE": shift}
    moved = (
        f"the estimates by {_se(shift)}"
        if np.isfinite(shift)
        else "the estimates: not computed (no standard errors)"
    )
    text = f"at rtol {rtol2:g} the -2LL changes by {d:.2g} and {moved}"
    if d > TOL_M2LL or (np.isfinite(shift) and shift > TOL_SHIFT):
        return [
            Finding(
                "tolerance check",
                "problem",
                text + f": rtol {rtol:g} is too loose.",
                f"fit again with a tighter rtol (options rtol={rtol / 100:g}).",
                ev,
            )
        ]
    return [Finding("tolerance check", "note", text + f"; rtol {rtol:g} is enough.",
                    evidence=ev)]  # fmt: skip


def _se(shift: float) -> str:
    return "less than 0.001 SE" if shift < 1e-3 else f"{shift:.2g} SE"


def _gradients(result) -> list[Finding]:
    from .optimize import to_u, to_x

    obj, sc = result.objective, result.scales
    tight, rtol2 = _tighter(obj, result.problem)
    u0 = np.asarray(to_u(np.asarray(result.x), sc))
    C = np.asarray(getattr(result, "cov_u", np.nan), float)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:  # no standard errors: a small spread on every scale
        L = 0.1 * np.eye(len(u0))
    rng = np.random.default_rng(int(result.problem._options.get("seed", 0)))

    f1 = lambda w: obj._minus2ll(to_x(w, sc))  # noqa: E731
    # the fit's own compiled -2LL, at either tolerance: no new compile
    f1_jit = lambda w: obj._jit(to_x(w, sc))  # noqa: E731
    f2 = lambda w: obj._jit(to_x(w, sc), tight)  # noqa: E731
    rel = []
    for u in _plausible(f1_jit, u0, L, float(f1_jit(jnp.asarray(u0))), rng):
        d = rng.normal(size=len(u0))
        d /= np.linalg.norm(d)
        _, exact = jax.jvp(f1, (u,), (jnp.asarray(d),))
        h = GRAD_STEP
        fd = (float(f2(u + h * d)) - float(f2(u - h * d))) / (2 * h)
        rel.append(abs(fd - float(exact)) / max(abs(fd), abs(float(exact)), 1e-2))
    return [_gradient_finding(rel, rtol2, result.problem)]


def _plausible(f, u0, L, best, rng):
    """``GRAD_DIRECTIONS`` points drawn from the estimates' normal approximation (with their
    correlations) where -2LL is within the 99 % chi-square bound of the best: the region the
    data allow, and the only one NUTS visits. Far outside it the model can be extreme and
    differences noisy (an infection-age model: -2LL 10^4 against -30 at the best). Draws
    outside are redrawn, then drawn closer."""
    from scipy.stats import chi2

    bound = float(chi2.ppf(0.99, len(u0)))
    out, scale = [], 1.0
    for k in range(10 * GRAD_DIRECTIONS):
        if k and k % GRAD_DIRECTIONS == 0 and len(out) < k // GRAD_DIRECTIONS:
            scale /= 2  # few draws inside: draw closer
        u = jnp.asarray(u0 + scale * (L @ rng.normal(size=len(u0))))
        if float(f(u)) - best <= bound:
            out.append(u)
            if len(out) == GRAD_DIRECTIONS:
                break
    return out or [jnp.asarray(u0)]


def _gradient_finding(rel, rtol2, problem) -> Finding:
    worst = float(np.max(rel))
    ev = {"relative differences": [float(r) for r in rel], "rtol of the differences": rtol2}
    text = (
        f"exact gradients agree with differences (at rtol {rtol2:g}) to {100 * worst:.2g} %"
        f" ({len(rel)} directions)"
    )
    if worst > GRAD_REL:
        rtol = float(problem._options.get("rtol") or 1e-6)
        return Finding("gradient check", "warning",
                       text + ": not consistent enough for NUTS, which needs exact gradients.",
                       f"a tighter rtol (options rtol={rtol / 10:g}).", ev)  # fmt: skip
    return Finding("gradient check", "note", text + ".", evidence=ev)


# ----------------------------------------------------------------------------- NUTS
def _some_draws(result, n):
    flat = result.flat_draws.reshape(-1, result.flat_draws.shape[-1])
    if not len(flat):
        return flat
    return flat[np.unique(np.linspace(0, len(flat) - 1, min(n, len(flat))).astype(int))]


def _tolerance_nuts(result) -> list[Finding]:
    post = result.posterior
    tight, rtol2 = _tighter(post.cond, result.problem)
    rtol = float(result.problem._options.get("rtol") or 1e-6)
    pts = _some_draws(result, NUTS_POINTS)
    if not len(pts):
        return []
    at = post.cond._tolerant(post.potential)
    U = jax.jit(jax.vmap(lambda f, tol: at(f, tol=tol), in_axes=(0, None)))
    U1 = np.asarray(U(jnp.asarray(pts), post.cond._tol0))
    U2 = np.asarray(U(jnp.asarray(pts), tight))
    d = float(np.max(np.abs(2.0 * (U2 - U1))))  # as -2 log posterior
    ev = {"rtol": rtol, "tighter rtol": rtol2, "largest change of -2 log posterior": d}
    text = f"at rtol {rtol2:g}, -2 log posterior changes by at most {d:.2g} over {len(pts)} draws"
    if d > TOL_M2LL:
        return [Finding("tolerance check", "problem", text + f": rtol {rtol:g} is too loose.",
                        f"sample again with a tighter rtol (options rtol={rtol / 100:g}).",
                        ev)]  # fmt: skip
    return [Finding("tolerance check", "note", text + f"; rtol {rtol:g} is enough.",
                    evidence=ev)]  # fmt: skip


def _gradients_nuts(result) -> list[Finding]:
    post = result.posterior
    tight, rtol2 = _tighter(post.cond, result.problem)
    pts = _some_draws(result, GRAD_DIRECTIONS)
    if not len(pts):
        return []
    U1 = post.potential
    at = jax.jit(post.cond._tolerant(post.potential))
    U2 = lambda f: at(f, tol=tight)  # noqa: E731
    rng = np.random.default_rng(int(result.problem._options.get("seed", 0)))
    rel = []
    for f in pts:
        d = rng.normal(size=f.size)
        d /= np.linalg.norm(d)
        _, exact = jax.jvp(U1, (jnp.asarray(f),), (jnp.asarray(d),))
        h = GRAD_STEP
        fd = (float(U2(jnp.asarray(f + h * d))) - float(U2(jnp.asarray(f - h * d)))) / (2 * h)
        rel.append(abs(fd - float(exact)) / max(abs(fd), abs(float(exact)), 1e-2))
    return [_gradient_finding(rel, rtol2, result.problem)]


def _convergence(result) -> list[Finding]:
    out = []
    for w in result.warnings():
        severity = "problem" if "divergent" in w or "R-hat" in w else "warning"
        out.append(Finding("convergence", severity, w))
    if not out:
        out.append(Finding("convergence", "note", "R-hat, effective sample sizes, divergences, "
                           "BFMI and tree depth: all within their limits."))  # fmt: skip
    return out


def _u_of(v: np.ndarray, s: tuple) -> np.ndarray:
    """Values on the estimation scale ``s`` (NumPy, many at once)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if s[0] == "logit":
            p = (v - s[1]) / (s[2] - s[1])
            return np.log(p / (1 - p))
        if s[0] == "above":
            return np.log(v - s[1])
        if s[0] == "below":
            return np.log(s[1] - v)
        if s[0] == "log":
            return np.log(v)
    return np.asarray(v, float)


def _contraction(result) -> list[Finding]:
    rng = np.random.default_rng(0)
    names, sc = list(result.names), result.scales
    wide = []
    ev = {}
    for p in result.priors:
        if p.name not in names:
            continue
        i = names.index(p.name)
        prior_u = _u_of(p.draw(rng, 4000), sc[i])
        prior_u = prior_u[np.isfinite(prior_u)]
        post_u = _u_of(result.draws[p.name].to_numpy(), sc[i])
        if len(prior_u) < 2 or len(post_u) < 2:
            continue
        ratio = float(np.std(post_u, ddof=1) / np.std(prior_u, ddof=1))
        ev[p.name] = ratio
        if ratio > CONTRACTION:
            wide.append(f"{p.name} (posterior SD {100 * ratio:.0f} % of the prior's; prior "
                        f"{p})")  # fmt: skip
    if not wide:
        return [Finding("prior and data", "note", "the data narrowed every prior to less than "
                        f"{100 * CONTRACTION:.0f} % of its width.", evidence=ev)]  # fmt: skip
    return [Finding(
        "prior and data", "warning",
        "the data say little about " + _and(wide) + ": the prior decides it.",
        "write a prior you can defend (from earlier studies or the literature), or fix the "
        "quantity and say so.", ev,
    )]  # fmt: skip


def _weak_posterior(result) -> list[Finding]:
    names, sc = list(result.names), result.scales
    if not len(result.draws):
        return []
    U = np.column_stack(
        [_u_of(result.draws[n].to_numpy(), s) for n, s in zip(names, sc, strict=True)]
    )
    sd = U.std(axis=0, ddof=1)
    R = np.corrcoef(U, rowvar=False) if len(names) > 1 else np.ones((1, 1))
    t = result.table()
    weak = []
    for i, n in enumerate(names):
        row = t.iloc[i]
        rel = 100 * float(row["SD"]) / abs(float(row["median"]))
        what = _weakness(n, sc[i], float(row["CI 2.5 %"]), float(row["CI 97.5 %"]), sd[i], rel)
        if what is None:
            continue
        partners = [f"{names[j]} (correlation {R[i, j]:.2f})" for j in range(len(names))
                    if j != i and abs(R[i, j]) > PARTNER]  # fmt: skip
        if partners:
            what += ", which moves with " + " and ".join(partners)
        weak.append(what)
    if not weak:
        return []
    return [Finding("weakly determined", "warning", _and(weak) + ".",
                    "a written prior, or fix the quantity and say so; or measurements that "
                    "inform it.")]  # fmt: skip


def profile_findings(profiles) -> list[Finding]:
    """Findings from the profiles computed so far: a set in more than one piece (optima the
    data cannot tell apart), and a profile point better than the fit (an optimum it
    missed)."""
    out = []
    for name, prof in profiles.items():
        pieces = getattr(prof, "pieces", None) or []
        if len(pieces) > 1:
            parts = " and ".join(f"({lo:.4g}, {hi:.4g})" for lo, hi in pieces)
            out.append(Finding(
                "profile in pieces", "warning",
                f"the {prof.level:.0%} profile set of {name} has {len(pieces)} pieces, {parts}: "
                "the likelihood has optima the data cannot tell apart.",
                "report the whole set (profile.interval spans the pieces), not one optimum's "
                "standard error; to separate the optima, more informative data (a design "
                "that tells them apart) or prior knowledge.",
                {"pieces": pieces}))  # fmt: skip
        better = getattr(prof, "better", None)
        if better is not None:
            out.append(Finding(
                "better fit in the profile", "problem",
                f"the profile of {name} found a fit better than the estimate by "
                f"{prof.m2ll - better['-2LL']:.3g} in {getattr(prof, 'objective', '-2LL')}, "
                f"at {name} = {better['value']:.4g}: "
                "the fit missed an optimum.",
                "refit with more starts, or from the profile's best point "
                "(profile.better has its values).",
                {"value": better["value"], "-2LL": better["-2LL"]}))  # fmt: skip
    return out
