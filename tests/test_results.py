# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""What a fit gives after the table: sdtab/patab, goodness-of-fit plots, VPC, profile, save.
Oracles: the closed-form one-compartment model (pooled fit, ``test_optimize``) and the linear
model (``test_laplace``), whose predictions are known exactly."""

import json

import matplotlib
import numpy as np
import pandas as pd
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


def test_vpc_with_a_limit_only_on_censored_rows(tmp_path):
    """LLOQ filled only on the BLQ rows (as in many NONMEM files): the simulated values are
    floored only where a row has a limit, so the bands of the other bins are numbers."""
    df = pd.DataFrame(dict(TIME=[0.5, 1, 2, 4, 6, 8], DV=[0.78, 0.61, 0.37, 0.14, 0.05, 0.02],
                           LLOQ=[np.nan] * 4 + [0.1, 0.1], BLQ=[0, 0, 0, 0, 1, 1]))  # fmt: skip
    df.to_csv(tmp_path / "decay.csv", index=False)
    p = tf.Problem.from_text(
        "model decay\nparam k = 0.5\nd_t(A) = -k*A\nA(0) = 1\nobservable amount = A\n"
        f'fit decay\ndata "{tmp_path / "decay.csv"}"\nobserve amount = DV\n'
        "censor amount where BLQ == 1 below LLOQ\nestimate k = 0.3\n"
        "error amount ~ additive(0.05)\nmethod optimize\noptions progress=none checks=quick\n"
    )
    t = tf.fit(p).vpc(n=50, seed=1).table
    assert np.isfinite(t[["sim p5", "sim p50", "sim p95"]].to_numpy()).all()


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
    pr = pooled.profile("ke", points=3, scan=False)
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


def test_profile_with_priors_is_of_what_the_fit_minimized(tmp_path):
    """With a prior, the fit minimizes -2LL plus the prior's terms; the profile does too, so
    the estimate is its lowest point (no false 'better fit') and the interval is finite."""
    p = problem(tmp_path, extra="prior ke ~ lognormal(median=0.12, sdlog=0.1)\n",
                options="starts=1 seed=1 checks=quick")  # fmt: skip
    r = tf.fit(p)
    pr = r.profile("ke", scan=False)
    assert pr.better is None and pr.m2ll == pytest.approx(r.with_priors)
    assert pr.objective == "-2LL plus the priors' terms"
    assert pr.table["-2LL"].min() == pytest.approx(r.with_priors, abs=1e-6)
    lo, hi = pr.interval
    assert lo < r.estimates["ke"] < hi
    assert not [f for f in r.doctor().findings if f.check == "better fit in the profile"]


def test_ctrl_c_stops_a_profile_and_keeps_its_points(pooled, monkeypatch):
    import tenqsp_fit.held as H

    real = H.held_fitter

    def fitter(result, j):
        fit, calls = real(result, j), {"n": 0}

        def interrupted(*args):
            calls["n"] += 1
            if calls["n"] == 4:  # the compile at the estimate, then a point on each side
                raise KeyboardInterrupt
            return fit(*args)

        return interrupted

    monkeypatch.setattr(H, "held_fitter", fitter)
    pr = pooled.profile("ke")
    assert pr.stopped_by == "Ctrl-C" and "stopped by Ctrl-C" in repr(pr)
    assert 2 <= len(pr.table) < 8  # the estimate and the points computed before the stop
    assert "not reached" in repr(pr) and "nan" not in repr(pr)  # an end not reached, in words


def test_profile_scan_finds_the_flip_flop_optimum(tmp_path):
    """Oral one-compartment PK: swapping ka and ke gives the same curve (flip-flop), so the
    likelihood has a second optimum, exactly as good, at ka = the estimate of ke. The walk
    stops at the hump between them; the scan finds the second piece of the set, which
    mirrors ke's interval at the estimate."""
    r = tf.fit(problem(tmp_path, options="starts=3 seed=1"))
    e = r.estimates
    walk = r.profile("ka", scan=False, level=0.9)
    assert len(walk.pieces) == 1 and walk.pieces[0][0] > e["ke"]
    pr = r.profile("ka", level=0.9, scan_points=12)
    assert len(pr.pieces) == 2
    (lo2, hi2), (lo1, hi1) = pr.pieces
    np.testing.assert_allclose((lo1, hi1), walk.interval, rtol=1e-3)
    ke = r.profile("ke", scan=False, level=0.9)
    np.testing.assert_allclose((lo2, hi2), ke.interval, rtol=0.03)
    assert pr.interval == (lo2, hi1)
    assert pr.better is None  # the two optima are equally good
    assert "2 pieces" in repr(pr)
    d = r.doctor()
    assert any(f.check == "profile in pieces" and "ka" in f.message for f in d.findings)
    from tenqsp_fit.passport import _shape

    assert _shape(pr).startswith("2 pieces, closed")
    ax = pr.plot()
    assert "2 pieces" in ax.get_title()


def test_profile_scan_options(pooled):
    pr = pooled.profile("CL", level=0.9, scan_range=(0.02, 0.08), scan_points=3)
    assert len(pr.pieces) == 1
    assert pr.scanned == pytest.approx((0.02, 0.08))
    assert (pr.table.status == "scan").sum() >= 6
    with pytest.raises(ValueError, match="low < high"):
        pooled.profile("CL", scan_range=(0.08, 0.02))
    with pytest.raises(TypeError, match="scan="):
        pooled.profile("CL", scan="yes")


def test_profile_small_sample_cut_off(tmp_path):
    """With the residual SD estimated, the cut-off is n log(1 + F/(n - p)) and the interval
    ends where the closed form's n log(SSR(ke) / SSR_min) (the SD profiled out) reaches it;
    with the SD fixed, the chi-square cut-off (``test_profile_interval_and_plot``)."""
    from scipy.stats import f as f_dist

    r = tf.fit(problem(tmp_path, error="additive(0.3)", options="starts=1"))
    df = table()
    n, p = len(df), 3
    pr = r.profile("ke", level=0.9)
    cut = n * np.log1p(f_dist.ppf(0.9, 1, n - p) / (n - p))
    assert pr.cut == pytest.approx(cut) and pr.rule.startswith("small sample")
    assert cut > 2.706  # wider than chi-square

    def ssr(ke):
        def res(lx):
            return conc(df.TIME.to_numpy(), df.Dose.to_numpy(), *np.exp(lx), ke) - df.DV.to_numpy()

        fit = least_squares(res, np.log([1.5, 0.04]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        return float(np.sum(fit.fun**2))

    base = ssr(r.estimates["ke"])
    for end in pr.interval:  # linear between profile points: to within a few %
        assert np.isfinite(end)
        assert n * np.log(ssr(end) / base) == pytest.approx(cut, rel=0.05)
    fixed = tf.fit(problem(tmp_path, options="starts=1")).profile("ke", level=0.9)
    assert fixed.cut == pytest.approx(2.7055, abs=1e-4) and fixed.rule == "chi-square, 1 df"


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


def test_plot_data_before_fitting(tmp_path):
    """One panel per link, a line per unit, the dose times as ticks; nothing is fitted."""
    import matplotlib.pyplot as plt

    p = problem(tmp_path)
    fig = p.plot_data(log=True)
    (ax,) = [a for a in fig.axes if a.get_visible()]
    assert ax.get_title() == "conc (3 units)" and ax.get_yscale() == "log"
    assert len(ax.lines) == 3 + 1  # three units, the dose ticks
    np.testing.assert_allclose(ax.lines[0].get_ydata(), table().query("ID == 1").DV)
    plt.close(fig)


def test_the_run_record_has_the_versions_of_the_optimizers_and_nuts(pooled):
    from tenqsp_fit.results import run_record

    versions = run_record(pooled)["versions"]
    assert {"tenqsp_fit", "tenqsp", "jax", "scipy", "optimistix", "numpyro",
            "arviz_stats"} <= set(versions)  # fmt: skip
    assert versions["tenqsp_fit"] == tf.__version__


def test_vpc_at_other_solver_tolerances(pooled):
    """``vpc(rtol=, atol=)``: the simulations at other tolerances (the title says so), no
    rebuild; a looser tolerance moves the bands by far less than their width."""
    base = pooled.vpc(n=50, seed=2)
    loose = pooled.vpc(n=50, seed=2, rtol=1e-4, atol=1e-7)
    assert "solver rtol 0.0001, atol 1e-07" in loose.figure._suptitle.get_text()
    width = (base.table["sim p95"] - base.table["sim p5"]).to_numpy()
    shift = np.abs(loose.table["sim p50"] - base.table["sim p50"]).to_numpy()
    assert (shift < 0.05 * width).all()
    tight = pooled.vpc(n=50, seed=2, rtol=1e-9)
    np.testing.assert_allclose(tight.table["sim p50"], base.table["sim p50"], rtol=1e-4)
    for bad in (0, 1.5, "1e-6", True):
        with pytest.raises(ValueError, match="rtol is a number between 0 and 1"):
            pooled.vpc(n=5, rtol=bad)


def test_vpc_by_a_data_column(pooled):
    """``vpc(by="Dose")``: a row of panels per dose, each with its own bins; with nothing
    varying between units, each dose's simulated median is that dose's closed-form curve."""
    v = pooled.vpc(n=200, seed=4, by="Dose")
    t = v.table
    doses = sorted(set(table().Dose))
    assert list(t.columns[:3]) == ["link", "Dose", "bin"]
    assert sorted(set(t.Dose)) == doses and (t.n == 1).all()
    assert len(t) == len(doses) * len(set(table().TIME))
    e = pooled.estimates
    for d in doses:
        rows = t[t.Dose == d]
        f = conc(rows.time.to_numpy(), d, e["ka"], e["CL"], e["ke"])
        np.testing.assert_allclose(rows["sim p50"], f, atol=0.1)  # residual SD 0.3
    axes = v.figure.axes
    assert len(axes) == len(doses) and axes[0].get_title().startswith(f"Dose = {doses[0]}")
    # each panel shows its own dose's data only: one subject's 8 measurements
    for ax in axes:
        points = sum(len(x.get_xdata()) for x in ax.lines if x.get_marker() == "o")
        assert points == len(set(table().TIME))


@pytest.mark.parametrize(
    "by, message",
    [("dose", "names a column of the data"), ("DV", "takes 12 at most")],
)
def test_vpc_by_mistakes(pooled, by, message):
    with pytest.raises(ValueError, match=message):
        pooled.vpc(n=5, by=by)


def test_prior_predictive_by_a_data_column(tmp_path):
    p = problem(tmp_path, options="starts=1", extra="prior ka ~ lognormal(median=1.5, sdlog=0.3)\n")
    v = tf.prior_predictive(p, n=50, seed=1, by="Dose")
    assert sorted(set(v.table.Dose)) == sorted(set(table().Dose))
