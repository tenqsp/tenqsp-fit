# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""B1 against reference fits: theophylline (R's Theoph, 12 subjects), one compartment with
first-order absorption, fitted by R's nlme (population; ka and CL lognormal between subjects)
and nls (each subject alone). The references are stored CSV (``validation/b1``, made by
``reference.R``). Acceptance (PRD Section 11, B1): each estimate within max(10 %, one reference
SE) of the reference, compared on the log scale; each SE within a factor 1.5 of the
reference's.

nlme's population fit is by Lindstrom-Bates steps; tenqsp-fit's by the Laplace approximation
(FOCE-I), so the two land close but not on the same point: tenqsp-fit's -2LL must be at or
below its own objective at nlme's estimates. For each subject, nls is least squares with the
SE from s^2 = RSS / (n - p); tenqsp-fit is maximum likelihood (RSS / n), so its SEs are
smaller by about sqrt((n - p) / n), 0.85 for 11 samples and 3 parameters."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tenqsp_fit as tf

B1 = Path(__file__).parents[1] / "validation" / "b1"
MODEL = """model theoph
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""
FIT = """fit {name}
data "{file}" with ID=Subject, TIME=Time, DV=conc
unit ID
doses Dose at t=0 into depot
observe conc = DV
{estimates}
error conc ~ additive(1)
method {method}
options rtol=1e-8 atol=1e-10 progress=none {options}
"""
POP = ("estimate ka ~ lognormal(median=1.5, sdlog=0.5)\n"
       "estimate CL ~ lognormal(median=0.04, sdlog=0.5)\nestimate ke = 0.08")  # fmt: skip


def accept(ours, ours_se_log, ref, ref_se_log):
    """(estimate within max(10 %, one reference SE) on the log scale, SE within x1.5)."""
    close = abs(np.log(ours) - np.log(ref)) <= max(np.log(1.1), ref_se_log)
    return close, 1 / 1.5 <= ours_se_log / ref_se_log <= 1.5


def compare(result, ref, names):
    """One row per quantity: ours, the reference, and whether each criterion holds. ``ref``:
    quantity, estimate (log for 'log ...' rows), se_log; ``names``: quantity -> ours."""
    e, se = result.estimates, result.table().set_index("quantity").SE
    rows = []
    for _, r in ref.iterrows():
        name = names[r.quantity]
        value = float(np.exp(r.estimate)) if r.quantity.startswith("log ") else float(r.estimate)
        ours, ours_se = e[name], se[name] / e[name]
        close, se_ok = accept(ours, ours_se, value, r.se_log)
        rows.append(dict(quantity=name, ours=ours, reference=value,
                         log_difference=np.log(ours / value), se_log=ours_se,
                         reference_se_log=r.se_log, estimate_ok=close, se_ok=se_ok))  # fmt: skip
    return pd.DataFrame(rows)


@pytest.mark.timeout(1200)
def test_b1_population_against_nlme():
    file = B1 / "theoph.csv"
    p = tf.Problem.from_text(MODEL + FIT.format(name="theoph_pop", file=file, estimates=POP,
                                                method="laplace", options=""))  # fmt: skip
    r = tf.fit(p)
    assert r.status == "converged" and r.covariance_status == "ok"
    ref = pd.read_csv(B1 / "reference_population_nlme.csv")
    names = {
        "log ka.median": "ka.median",
        "log CL.median": "CL.median",
        "log ke": "ke",
        "ka.sdlog": "ka.sdlog",
        "CL.sdlog": "CL.sdlog",
        "conc.additive": "conc.additive",
    }
    table = compare(r, ref, names)
    print("\n" + table.to_string(index=False))
    assert table.estimate_ok.all() and table.se_ok.all(), table
    # the same Laplace objective at nlme's estimates is no lower than at ours
    est = dict(zip(table.quantity, table.reference, strict=True))
    at_nlme = float(r.objective(np.array([est[n] for n in r.names])))
    assert r.m2ll <= at_nlme + 1e-6, (r.m2ll, at_nlme)


@pytest.mark.timeout(1200)
def test_b1_each_subject_against_nls(tmp_path):
    data = pd.read_csv(B1 / "theoph.csv")
    ref = pd.read_csv(B1 / "reference_individual_nls.csv")
    names = {"log ka": "ka", "log CL": "CL", "log ke": "ke"}
    tables = []
    for s, d in data.groupby("Subject"):
        file = tmp_path / f"subject{s}.csv"
        d.to_csv(file, index=False)
        p = tf.Problem.from_text(MODEL + FIT.format(
            name=f"theoph_{s}", file=file, estimates="estimate ka = 1.5, CL = 0.04, ke = 0.08",
            method="optimize", options="starts=4 checks=quick"))  # fmt: skip
        r = tf.fit(p)
        assert r.status == "converged", (s, r.status)
        t = compare(r, ref[ref.Subject == s], names)
        tables.append(t.assign(subject=s))
        # the residual SD: maximum likelihood (RSS/n) against nls (RSS/(n - 3))
        n = int(ref[ref.Subject == s].n.iloc[0])
        sd_nls = float(ref[ref.Subject == s].residual_sd.iloc[0])
        np.testing.assert_allclose(r.estimates["conc.additive"], sd_nls * np.sqrt((n - 3) / n),
                                   rtol=1e-3)  # fmt: skip
    table = pd.concat(tables, ignore_index=True)
    print("\n" + table.to_string(index=False))
    assert table.estimate_ok.all() and table.se_ok.all(), table[~(table.estimate_ok & table.se_ok)]
