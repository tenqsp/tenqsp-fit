# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Fits of models given by formulas of t (no states; NONMEM's $PRED). Oracles: ordinary least
squares for a straight line, the exact likelihood of a linear mixed model (Laplace is exact
there), and the same model written with an equation."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

import tenqsp_fit as tf

LINE = """model line
param E0 = 40, prog = 1
observable E = E0 + prog*t
"""


def line_table(seed=5):
    rng = np.random.default_rng(seed)
    t = np.tile([0.0, 4, 8, 12, 26, 52], 3)
    return pd.DataFrame(dict(ID=np.repeat([1, 2, 3], 6), TIME=t,
                             DV=50 + 2 * t + 3 * rng.normal(size=t.size)))  # fmt: skip


def line_problem(tmp_path, extra="", data=None):
    f = tmp_path / "line.csv"
    (line_table() if data is None else data).to_csv(f, index=False)
    return tf.Problem.from_text(
        LINE + f'fit line\ndata "{f}"\nunit ID\n{extra}observe E = DV\n'
        "estimate E0 = 40, prog = 1\nerror E ~ additive(1)\nmethod optimize\n"
        "options starts=2 seed=1 progress=none\n"
    )


def test_a_straight_line_is_ordinary_least_squares(tmp_path):
    df = line_table()
    X = np.column_stack([np.ones(len(df)), df.TIME])
    beta, rss = np.linalg.lstsq(X, df.DV, rcond=None)[:2]
    sigma = np.sqrt(rss[0] / len(df))  # maximum likelihood
    se = sigma * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    r = tf.fit(line_problem(tmp_path))
    np.testing.assert_allclose([r.estimates["E0"], r.estimates["prog"]], beta, rtol=1e-6)
    assert r.estimates["E.additive"] == pytest.approx(sigma, rel=1e-5)
    t = r.table().set_index("quantity")
    np.testing.assert_allclose(t.loc[["E0", "prog"], "SE"], se, rtol=1e-3)
    assert r.covariance_status == "ok"
    (f,) = [f for f in r.doctor().findings if f.check == "tolerance check"]
    assert f.severity == "note" and "given by formulas of t" in f.message
    assert "(given by formulas of t: no states, no solver)" in str(r.problem.summary())


LIN_FORMULA = """model lin
param A=10, B=1, k=0.3
observable y = A*exp(-k*t) + B
"""
LIN_ODE = """model lin
param A=10, B=1, k=0.3
d_t(e) = -k*e
e(0) = 1
observable y = A*e + B
"""


def lin_table(n=8, seed=3):
    rng = np.random.default_rng(seed)
    t = np.array([0.5, 1, 2, 4, 6])
    rows = []
    for i in range(1, n + 1):
        a, b = 10 + 2 * rng.normal(), 1 + 0.5 * rng.normal()
        y = a * np.exp(-0.3 * t) + b + 0.3 * rng.normal(size=t.size)
        rows += [dict(ID=i, TIME=float(tt), DV=float(v)) for tt, v in zip(t, y, strict=True)]
    return pd.DataFrame(rows)


def lin_problem(tmp_path, model):
    f = tmp_path / "lin.csv"
    lin_table().to_csv(f, index=False)
    return tf.Problem.from_text(
        model + f'fit lin\ndata "{f}"\nunit ID\nobserve y = DV\n'
        "estimate A ~ normal(mean=8, sd=1)\nestimate B ~ normal(mean=0.5, sd=1)\n"
        "estimate k = 0.2\nerror y ~ additive(0.5)\nmethod laplace\n"
        "options rtol=1e-10 atol=1e-12 progress=none\n"
    )


def exact_m2ll(mA, sA, mB, sB, k, sigma, df):
    total = 0.0
    for _, d in df.groupby("ID"):
        e = np.exp(-k * d.TIME.to_numpy())
        one = np.ones_like(e)
        cov = sA**2 * np.outer(e, e) + sB**2 * np.outer(one, one) + sigma**2 * np.eye(e.size)
        total += -2 * multivariate_normal(mA * e + mB, cov).logpdf(d.DV.to_numpy())
    return total


def test_laplace_is_exact_and_equals_the_model_with_an_equation(tmp_path):
    formula = tf.objective(lin_problem(tmp_path, LIN_FORMULA))
    ode = tf.objective(lin_problem(tmp_path, LIN_ODE))
    assert formula.names == ode.names == ("A.mean", "A.sd", "B.mean", "B.sd", "k", "y.additive")
    for x in (formula.x0, np.array([10.5, 1.8, 1.2, 0.6, 0.31, 0.3])):
        want = exact_m2ll(*x, lin_table())
        assert float(formula(x)) == pytest.approx(want, rel=1e-10)
        assert float(ode(x)) == pytest.approx(want, rel=1e-7)


def test_a_dose_column_sets_a_parameter(tmp_path):
    """The dose as a parameter of the formula, set from a data column: as NONMEM's $PRED."""
    df = line_table().assign(DOSE=lambda d: d.ID * 10.0)
    df["DV"] = df.DOSE + 2 * df.TIME + np.random.default_rng(1).normal(size=len(df))
    f = tmp_path / "dose.csv"
    df.to_csv(f, index=False)
    p = tf.Problem.from_text(
        "model d\nparam D = 1, prog = 1\nobservable E = D + prog*t\n"
        f'fit d\ndata "{f}"\nunit ID\nD = DOSE\nobserve E = DV\nestimate prog = 1\n'
        "error E ~ additive(1)\nmethod optimize\noptions starts=1 progress=none\n"
    )
    r = tf.fit(p)
    assert r.estimates["prog"] == pytest.approx(2, abs=0.05)


def test_doses_are_refused_with_the_way_out(tmp_path):
    with pytest.raises(ValueError, match=r"has no states.*D = DOSE"):
        line_problem(tmp_path, extra="doses into E\n")
    df = line_table()
    df["AMT"] = np.where(df.TIME == 0, 100.0, 0.0)
    p = line_problem(tmp_path, data=df)
    with pytest.raises(ValueError, match=r"dose rows \(AMT\), but the model line has no states"):
        p.check()
