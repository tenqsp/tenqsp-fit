# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Predictions for a new regimen, the diagnostics figure, the profile's progress. Oracles: the
closed-form one-compartment model (a dose twice as large gives twice the curve; the interval's
width from the delta method) and the linear model (a new unit's spread known exactly)."""

import copy

import matplotlib
import numpy as np
import pytest
import tenqsp as tq
from test_laplace import lin_problem
from test_optimize import conc, problem

import tenqsp_fit as tf
from tenqsp_fit import predict as P

matplotlib.use("Agg")
T = np.linspace(0.25, 24, 25)


@pytest.fixture(scope="module")
def pooled(tmp_path_factory):
    return tf.fit(problem(tmp_path_factory.mktemp("pooled"), options="starts=1 checks=quick"))


@pytest.fixture(scope="module")
def population(tmp_path_factory):
    return tf.fit(lin_problem(tmp_path_factory.mktemp("pop"), correlate=False,
                              options="checks=quick"))  # fmt: skip


def test_a_prediction_after_optimize(pooled):
    e = pooled.estimates
    pr = pooled.predict(regimen=tq.bolus("depot", 5.0, times=[0]), times=T, n=4000)
    assert pr.kind == P.APPROXIMATE and pr.n == 4000 and pr.failed == 0
    t = pr.table
    exact = conc(T, 5.0, e["ka"], e["CL"], e["ke"])
    np.testing.assert_allclose(t["median"], exact, rtol=0.02)
    assert ((t["2.5 %"] < exact) & (exact < t["97.5 %"])).all()
    # the delta method: SD of the curve = sqrt(J cov_u J'), J the derivative of the closed form
    # by the estimation-scale quantities (log ka, log CL, log ke)
    u = np.log([e["ka"], e["CL"], e["ke"]])
    h = 1e-6
    J = np.stack([(conc(T, 5.0, *np.exp(u + h * d)) - conc(T, 5.0, *np.exp(u - h * d))) / (2 * h)
                  for d in np.eye(3)], axis=1)  # fmt: skip
    sd = np.sqrt(np.einsum("ti,ij,tj->t", J, pooled.cov_u, J))
    np.testing.assert_allclose((t["97.5 %"] - t["2.5 %"]) / (2 * 1.96), sd, rtol=0.1)
    # twice the dose, twice the curve (the same draws)
    pr2 = pooled.predict(regimen=tq.bolus("depot", 10.0, times=[0]), times=T, n=4000)
    for c in ("median", "2.5 %", "97.5 %"):
        np.testing.assert_allclose(pr2.table[c], 2 * t[c], rtol=1e-6)


def test_a_named_prediction_is_in_the_passport(pooled):
    pooled.predict(regimen=tq.bolus("depot", 5.0, start=0, every=12, n=3), times=T,
                   name="three doses", n=50)  # fmt: skip
    text = repr(pooled.passport())
    assert "three doses: conc from 0.25 to 24" in text
    assert "approximate (from the standard errors), 50 draws" in text
    fig = pooled._predictions["three doses"].plot()
    assert "three doses" in fig._suptitle.get_text()


def test_a_new_unit(population):
    """y = A exp(-k t) + B, A and B normal and independent: a new unit's spread around the
    typical curve adds sA^2 exp(-2kt) + sB^2 to the uncertainty of the typical curve."""
    e = population.estimates
    pr = population.predict(times=T, n=4000, between_units=True)
    t = pr.table
    np.testing.assert_allclose(t["median"], e["A.mean"] * np.exp(-e["k"] * T) + e["B.mean"],
                               rtol=0.02)  # fmt: skip
    typical = (t["97.5 %"] - t["2.5 %"]) / (2 * 1.96)
    between = e["A.sd"] ** 2 * np.exp(-2 * e["k"] * T) + e["B.sd"] ** 2
    new = (t["new unit 95 %"] - t["new unit 5 %"]) / (2 * 1.645)
    np.testing.assert_allclose(new, np.sqrt(typical**2 + between), rtol=0.1)
    assert "a new unit's 90 % range" in pr.summary()


def test_mistakes_are_named(pooled):
    with pytest.raises(ValueError, match="'cnc' is not an observable.*conc"):
        pooled.predict(times=T, outputs=["cnc"])
    with pytest.raises(ValueError, match="increasing"):
        pooled.predict(times=[2, 1])
    with pytest.raises(TypeError, match="TenQSP regimen"):
        pooled.predict(times=T, regimen={"depot": 5})
    with pytest.raises(ValueError, match="vary between units"):
        pooled.predict(times=T, between_units=True)
    with pytest.raises(TypeError):
        pooled.predict(times=T, seed=1)  # unknown arguments raise


def test_without_standard_errors_the_estimates_only(pooled):
    r = copy.copy(pooled)
    r.cov_u = np.full_like(pooled.cov_u, np.nan)
    pr = r.predict(regimen=tq.bolus("depot", 5.0, times=[0]), times=T)
    assert pr.kind == P.ESTIMATES and pr.n == 1
    e = pooled.estimates
    np.testing.assert_allclose(pr.table["median"], conc(T, 5.0, e["ka"], e["CL"], e["ke"]),
                               rtol=1e-6)  # fmt: skip
    np.testing.assert_allclose(pr.table["2.5 %"], pr.table["median"])


def test_the_diagnostics_figure(pooled, population):
    for r in (pooled, population):
        fig = r.plot_diagnostics()
        titles = [s._suptitle.get_text() for s in fig.subfigs]
        assert "goodness of fit" in titles[0] and "visual predictive check" in titles[1]
        assert r.method in titles[2]
    with pytest.raises(ValueError, match="method nuts"):
        pooled.to_arviz()


def test_the_profile_prints_a_line_per_point(tmp_path, capsys):
    r = tf.fit(problem(tmp_path, options="starts=1 checks=quick"))
    r.problem._options["progress"] = "text"
    pr = r.profile("ke", points=2)
    lines = [x for x in capsys.readouterr().out.splitlines() if x.strip().startswith("profile")]
    assert len(lines) == len(pr.table) - 1 == 4
    assert "below the estimate, point 1" in "".join(lines)
