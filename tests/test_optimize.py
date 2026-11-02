# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``method optimize``: multi-start Levenberg-Marquardt (error model fixed) or BFGS (error
model estimated). Oracles: oral one-compartment PK in closed form, fitted by scipy's
least squares; for an estimated additive sd, its closed form at the minimum (sd^2 = RSS/n)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares

import tenqsp_fit as tf
from tenqsp_fit import optimize as O

PK = """model pk
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""
TIMES = (0.25, 0.5, 1, 2, 4, 8, 12, 24)
DOSE = {1: 4.0, 2: 5.0, 3: 6.0}


def conc(t, dose, ka, CL, ke):
    return dose * ka * ke / (CL * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def table():
    rng = np.random.default_rng(2)
    rows = [
        dict(ID=i, TIME=t, DV=conc(t, DOSE[i], 1.5, 0.04, 0.08) + 0.3 * rng.normal(), Dose=DOSE[i])
        for i in DOSE
        for t in TIMES
    ]
    return pd.DataFrame(rows)


def problem(tmp_path, error="additive(fix(0.3))", estimate="ka = 1, CL = 0.05, ke = 0.1",
            options="starts=3 seed=1", extra=""):  # fmt: skip
    f = tmp_path / "pk.csv"
    table().to_csv(f, index=False)
    return tf.Problem.from_text(
        PK + f'fit pk\ndata "{f}"\nunit ID\ndoses Dose at t=0 into depot\nobserve conc = DV\n'
        f"estimate {estimate}\n{extra}error conc ~ {error}\nmethod optimize\n"
        f"options {options} rtol=1e-10 atol=1e-12 progress=none\n"
    )


@pytest.fixture(scope="module")
def oracle():
    """The least-squares fit of the closed form (scipy)."""
    df = table()

    def r(lx):
        return conc(df.TIME.to_numpy(), df.Dose.to_numpy(), *np.exp(lx)) - df.DV.to_numpy()

    fit = least_squares(r, np.log([1.0, 0.05, 0.1]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return np.exp(fit.x), float(np.sum(fit.fun**2)), len(df)


def test_least_squares_by_levenberg_marquardt(tmp_path, oracle):
    x, rss, n = oracle
    p = problem(tmp_path)
    assert "scales     log scale: ka CL ke" in p.summary()
    r = tf.fit(p)
    assert isinstance(r, tf.Result) and r.optimizer == "Levenberg-Marquardt"
    assert r.status == "converged" and r.reached == 3
    np.testing.assert_allclose([r.estimates[k] for k in ("ka", "CL", "ke")], x, rtol=1e-5)
    m2ll = rss / 0.09 + n * (np.log(2 * np.pi) + 2 * np.log(0.3))
    assert r.m2ll == pytest.approx(m2ll, rel=1e-8)
    assert list(r.starts.columns[:4]) == ["start", "-2LL", "steps", "status"]
    assert set(r.trace.columns) >= {"start", "iteration", "-2LL", "ka", "CL", "ke"}
    assert "3 at the best fit" in repr(r)


def test_estimated_sd_by_bfgs(tmp_path, oracle):
    """An additive sd does not move the least-squares point; at the minimum sd^2 = RSS/n."""
    x, rss, n = oracle
    r = tf.fit(problem(tmp_path, error="additive(0.5)"))
    assert r.optimizer == "BFGS" and r.status == "converged"
    np.testing.assert_allclose([r.estimates[k] for k in ("ka", "CL", "ke")], x, rtol=1e-4)
    assert r.estimates["conc.additive"] == pytest.approx(np.sqrt(rss / n), rel=1e-4)


def test_bounds_keep_values_inside(tmp_path):
    with pytest.warns(O.NearBoundWarning, match="ka ended at 1.3, near its upper bound 1.3"):
        r = tf.fit(problem(tmp_path, estimate="ka = 1.2 in (1, 1.3), CL = 0.05, ke = 0.1"))
    assert 1 < r.estimates["ka"] <= 1.3
    assert r.estimates["ka"] == pytest.approx(1.3, abs=1e-3)  # the optimum lies above
    assert r.scales[0] == ("logit", 1.0, 1.3)
    assert r.near_bounds == {"ka": ("upper", 1.3)}
    assert "warning: ka ended at 1.3, near its upper bound 1.3" in repr(r)


def test_failed_starts_are_drawn_again(tmp_path):
    """ka = sqrt(q): a start with q < 0 does not run (NaN); it is drawn again."""
    p = problem(tmp_path, estimate="q = 1.2, CL = 0.05, ke = 0.1", extra="ka = sqrt(q)\n",
                options="starts=6 seed=4")  # fmt: skip
    p._params[0].start = 0.0  # q on its own scale (as is), so draws can go below 0
    assert "as is: q" in p.summary()
    r = tf.fit(p)
    assert r.redrawn > 0
    assert len(r.starts) == 6 and np.isfinite(r.starts["-2LL"]).all()


def test_scales():
    inf = np.inf
    bounds = [(-inf, inf), (-inf, inf), (0, 1), (2, inf), (-inf, 1)]
    sc = O.scales(["a", "b", "c", "d", "e"], [1.0, -2.0, 0.5, 3.0, 0.0], bounds)
    assert [s[0] for s in sc] == ["log", "as is", "logit", "above", "below"]
    x = np.array([1.0, -2.0, 0.5, 3.0, 0.0])
    np.testing.assert_allclose(O.to_x(O.to_u(x, sc), sc), x, rtol=1e-12)


def test_ctrl_c_keeps_the_work(tmp_path, monkeypatch):
    calls = {"n": 0}

    def update(self, steps, *a, **k):
        calls["n"] += 1
        if steps.max() >= 2 and not k.get("last"):
            raise KeyboardInterrupt
        return None

    monkeypatch.setattr(O._Progress, "update", update)
    r = tf.fit(problem(tmp_path))
    assert r.status == "stopped"
    assert len(r.trace) > 0 and np.isfinite(r.m2ll)
    assert set(r.starts["status"]) <= {"stopped", "converged"}
    # standard errors at a point that is not the minimum would mislead: none are given
    assert np.isnan(r.se).all() and r.covariance_status == O.STOPPED_SE
    assert "standard errors not computed: the fit was stopped" in repr(r)


def test_live_plot_page(tmp_path, monkeypatch, capsys):
    import re
    import webbrowser

    from tenqsp_fit import live

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(live, "REFRESH", 0.0)
    assert live.default_progress() == "text"  # not in a notebook
    import matplotlib

    backend = matplotlib.get_backend()
    p = problem(tmp_path, options="starts=2 seed=1")
    p._options["progress"] = "plot"
    r = tf.fit(p)
    out = capsys.readouterr().out
    url = re.search(r"live plot: (\S+)", out).group(1)
    assert opened == [url]
    import matplotlib

    assert matplotlib.get_backend() == backend  # the session's backend is left alone
    page = Path(url.removeprefix("file://")).read_text()
    assert "data:image/png;base64," in page
    assert "http-equiv" not in page  # the last page no longer refreshes
    assert "optimize  2 starts:" in out  # the text's last line
    assert r.status == "converged"


def test_plot_trace(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    r = tf.fit(problem(tmp_path, options="starts=2 seed=1"))
    fig = r.plot_trace()
    shown = [ax for ax in fig.axes if ax.get_visible()]
    assert len(shown) == 1 + len(r.names)
    assert [ax.get_title() for ax in shown[1:]] == list(r.names)
    assert len(shown[0].lines) == 2  # one line per start
    # the trace starts at each start's values (step 0), and the -2LL axis shows all of it
    first = r.trace[r.trace.iteration == 0].set_index("start")
    assert list(first.index) == [1, 2]
    np.testing.assert_allclose(first.loc[1, list(r.names)], r.x0, rtol=1e-12)
    m2ll0 = float(tf.objective(r.problem)(np.asarray(r.x0)))
    np.testing.assert_allclose(first.loc[1, "-2LL"], m2ll0, rtol=1e-9)
    lo, hi = shown[0].get_ylim()
    assert lo <= r.trace["-2LL"].min() and hi >= r.trace["-2LL"].max()


def test_in_notebook_knows_positron(monkeypatch):
    """Positron's console runs its own kernel app; its shell is a ZMQInteractiveShell
    subclass, as Jupyter's is."""
    import sys
    import types

    from tenqsp_fit import live

    class ZMQInteractiveShell:
        pass

    class PositronShell(ZMQInteractiveShell):
        pass

    fake = types.SimpleNamespace(get_ipython=lambda: PositronShell())
    monkeypatch.setitem(sys.modules, "IPython", fake)
    assert live.in_notebook() and live.default_progress() == "plot"
    fake.get_ipython = lambda: None  # IPython imported, but no kernel (a plain script)
    assert not live.in_notebook()


def test_progress_text_and_none(tmp_path, capsys):
    tf.fit(problem(tmp_path, options="starts=2 seed=1"))  # progress=none
    assert capsys.readouterr().out == ""
    p = problem(tmp_path, options="starts=2 seed=1")
    p._options["progress"] = "text"
    tf.fit(p)
    out = capsys.readouterr().out
    assert "optimize  2 starts:" in out and "best -2LL" in out and "ka " in out


def test_waterfall(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    r = tf.fit(problem(tmp_path))
    ax = r.plot_waterfall()
    assert "3 of 3 starts at the best fit" in ax.get_title()


def test_methods_not_yet_and_unknown(tmp_path):
    p = problem(tmp_path)
    with pytest.raises(ValueError, match="method laplace is for parameters that vary"):
        tf.fit(p, method="laplace")
    with pytest.raises(NotImplementedError, match="method nuts is not available yet"):
        tf.fit(p, method="nuts")
    with pytest.raises(ValueError, match="unknown method 'optimise'. Did you mean 'optimize'"):
        tf.fit(p, method="optimise")


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="Linux /proc only")
def test_fits_do_not_keep_compiled_code(tmp_path):
    """Each fit's compiled functions go when the fit does. (equinox's filter_jit kept them:
    about 2500 memory maps per fit, and a profile ran out of Linux's 65530.)"""
    import gc

    def maps():
        with open("/proc/self/maps") as f:
            return sum(1 for _ in f)

    tf.fit(problem(tmp_path, options="starts=1"))
    gc.collect()
    before = maps()
    for _ in range(3):
        tf.fit(problem(tmp_path, options="starts=1"))
        gc.collect()
    assert maps() - before < 300
