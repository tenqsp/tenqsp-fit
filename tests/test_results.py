# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""What a fit gives after the table: sdtab/patab, goodness-of-fit plots, VPC, profile, save.
Oracles: the closed-form one-compartment model (pooled fit, ``test_optimize``) and the linear
model (``test_laplace``), whose predictions are known exactly."""

import json

import matplotlib
import numpy as np
import pytest
from scipy.optimize import least_squares
from test_laplace import TIMES as LIN_TIMES
from test_laplace import lin_problem
from test_optimize import conc, problem, table

import tenqsp_fit as tf

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def pooled(tmp_path_factory):
    return tf.fit(problem(tmp_path_factory.mktemp("pooled"), options="starts=1"))


@pytest.fixture(scope="module")
def population(tmp_path_factory):
    return tf.fit(lin_problem(tmp_path_factory.mktemp("pop"), correlate=False))


def test_tables_pooled(pooled):
    sd, pa = pooled.tables()["sdtab"], pooled.tables()["patab"]
    df = table()
    assert len(sd) == len(df) and list(sd.columns[:3]) == ["ID", "TIME", "link"]
    e = pooled.estimates
    f = conc(sd.TIME.to_numpy(), df.Dose.to_numpy(), e["ka"], e["CL"], e["ke"])
    np.testing.assert_allclose(sd.IPRED, f, rtol=1e-6)
    np.testing.assert_allclose(sd.PRED, sd.IPRED)  # nothing varies: PRED = IPRED
    np.testing.assert_allclose(sd.IWRES, (df.DV - f) / 0.3, rtol=1e-5)
    assert (sd.CENS == 0).all() and sd.LIMIT.isna().all()
    assert list(pa.ID) == [1, 2, 3]
    np.testing.assert_allclose(pa.ka, e["ka"])


def test_tables_population(population):
    """IPRED from each unit's own values (its modes), PRED from the typical unit."""
    sd, pa = population.tables()["sdtab"], population.tables()["patab"]
    ind = population.individual()
    assert list(pa.columns[:3]) == ["ID", "z(A)", "z(B)"]
    np.testing.assert_allclose(pa[["A", "B"]], ind[["A", "B"]])
    e = population.estimates
    k = e["k"]
    for _, u in pa.iterrows():
        d = sd[sd.ID == u.ID]
        t = d.TIME.to_numpy()
        np.testing.assert_allclose(d.IPRED, u.A * np.exp(-k * t) + u.B, rtol=1e-6)
        np.testing.assert_allclose(d.PRED, e["A.mean"] * np.exp(-k * t) + e["B.mean"],
                                   rtol=1e-6)  # fmt: skip
    np.testing.assert_allclose(sd.IWRES, (sd.DV - sd.IPRED) / e["y.additive"], rtol=1e-6)
    assert len(sd) == 8 * len(LIN_TIMES)


def test_censored_rows(tmp_path):
    r = tf.fit(problem(tmp_path, options="starts=1", extra="censor conc below 2\n"))
    sd = r.tables()["sdtab"]
    below = table().DV.to_numpy() <= 2
    assert below.any()
    np.testing.assert_array_equal(sd.CENS.to_numpy(), below.astype(int))
    assert sd.DV[below].isna().all() and (sd.LIMIT == 2).all()
    assert sd.IWRES[below].isna().all()
    fig = r.plot_gof()  # censored values drawn at their limit
    assert any(line.get_label() == "censored" for line in fig.axes[0].lines)


def test_vpc(pooled):
    v = pooled.vpc(n=100, seed=3)
    t = v.table
    assert list(t.time) == sorted(set(table().TIME))  # 8 designed times: one bin each
    assert (t.n == 3).all()
    for p in ("p5", "p50", "p95"):
        assert (t[f"sim {p} lo"] <= t[f"sim {p}"]).all() and (
            t[f"sim {p}"] <= t[f"sim {p} hi"]
        ).all()
    # the fitted model simulates data like these: the data's median within the band
    inside = (t["obs p50"] >= t["sim p50 lo"]) & (t["obs p50"] <= t["sim p50 hi"])
    assert inside.mean() >= 0.75
    again = pooled.vpc(n=100, seed=3).table
    np.testing.assert_allclose(again["sim p50"], t["sim p50"])  # the seed reproduces it
    assert len(v.figure.axes) == 1


def test_vpc_population_draws_new_units(population):
    v = population.vpc(n=200, seed=1)
    t = v.table
    # the spread between units is in the simulations: the 5-95 band is far wider than 2 SD
    # of the residual error alone
    width = (t["sim p95"] - t["sim p5"]).to_numpy()
    assert (width > 2 * 1.645 * population.estimates["y.additive"]).all()
    inside = (t["obs p50"] >= t["sim p50 lo"]) & (t["obs p50"] <= t["sim p50 hi"])
    assert inside.all()


def test_profile_matches_the_closed_form(pooled):
    """Each point: ke held, ka and CL refitted; -2LL above the minimum = (SSR - SSR_min)/sd^2
    of the closed-form least-squares fit at that ke."""
    pr = pooled.profile("ke", points=3)
    df = table()

    def ssr(ke):
        def r(lx):
            return conc(df.TIME.to_numpy(), df.Dose.to_numpy(), *np.exp(lx), ke) - df.DV.to_numpy()

        fit = least_squares(r, np.log([1.5, 0.04]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        return float(np.sum(fit.fun**2))

    base = ssr(pooled.estimates["ke"])
    for v, m in zip(pr.table.value, pr.table["-2LL"], strict=True):
        np.testing.assert_allclose(m - pooled.m2ll, (ssr(v) - base) / 0.09, atol=2e-4)
    assert (pr.table["-2LL"] >= pooled.m2ll - 1e-6).all()
    assert len(pr.table) == 7 and pr.table.status.iloc[3] == "estimate"


def test_profile_interval_and_plot(pooled):
    """Walked until -2LL passes 3.84 on both sides; the interval is near the one from the
    standard error (a nearly quadratic -2LL)."""
    pr = pooled.profile("CL")
    lo, hi = pr.interval
    ci = pooled.table().set_index("quantity").loc["CL", ["CI 2.5 %", "CI 97.5 %"]]
    assert np.isfinite(lo) and np.isfinite(hi)
    np.testing.assert_allclose([lo, hi], ci.to_numpy(dtype=float), rtol=0.02)
    ax = pr.plot()
    assert "interval" in ax.get_title()
    with pytest.raises(ValueError, match="'kee' is not estimated.*ke"):
        pooled.profile("kee")


def test_save(pooled, tmp_path):
    out = pooled.save(tmp_path / "run1")
    names = {p.name for p in out.iterdir()}
    assert {"problem.tq", "data.csv", "estimates.csv", "starts.csv", "trace.csv", "sdtab.csv",
            "patab.csv", "result.txt", "run.json", "trace.png", "gof.png",
            "individual.png"} <= names  # fmt: skip
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "converged" and run["m2ll"] == pytest.approx(pooled.m2ll)
    assert set(run["versions"]) >= {"tenqsp_fit", "tenqsp", "jax"}
    import hashlib

    text = (out / "problem.tq").read_text()
    assert run["hashes"]["problem.tq"] == hashlib.sha256(text.encode()).hexdigest()
    assert tf.Problem.from_text(text) == pooled.problem  # it reads back to the same problem


def test_plots_have_their_panels(pooled, population):
    assert len([a for a in pooled.plot_gof().axes if a.get_visible()]) == 3  # no PRED panel
    assert len([a for a in population.plot_gof().axes if a.get_visible()]) == 4
    fig = population.plot_individual(units=[1, 2, 3])
    assert len([a for a in fig.axes if a.get_visible()]) == 3
    assert len(fig.axes[0].lines) == 3  # own curve, typical curve, data
    with pytest.raises(ValueError, match="no unit 99"):
        population.plot_individual(units=[99])
