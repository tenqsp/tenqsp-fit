# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``method optimize``: multi-start Levenberg-Marquardt (additive errors of known sd) or
BFGS (otherwise). Oracles: oral one-compartment PK in closed form, fitted by scipy's
least squares; for an estimated additive sd, its closed form at the minimum (sd^2 = RSS/n)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tenqsp as tq
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


def test_method_given_to_fit_stands_for_a_missing_method_line(tmp_path, oracle):
    """``tf.fit(p, method="optimize")`` fits a problem written without a method line."""
    p = problem(tmp_path)
    p._method = None
    with pytest.raises(tf.ProblemError, match="no method"):
        tf.fit(p)
    r = tf.fit(p, method="optimize")
    assert r.status == "converged" and p._method is None
    np.testing.assert_allclose([r.estimates[k] for k in ("ka", "CL", "ke")], oracle[0], rtol=1e-5)
    with pytest.raises(ValueError, match="unknown method 'optimise'. Did you mean 'optimize'"):
        tf.fit(p, method="optimise")


def test_estimated_sd_by_bfgs(tmp_path, oracle):
    """An additive sd does not move the least-squares point; at the minimum sd^2 = RSS/n."""
    x, rss, n = oracle
    r = tf.fit(problem(tmp_path, error="additive(0.5)"))
    assert r.optimizer == "BFGS" and r.status == "converged"
    np.testing.assert_allclose([r.estimates[k] for k in ("ka", "CL", "ke")], x, rtol=1e-4)
    assert r.estimates["conc.additive"] == pytest.approx(np.sqrt(rss / n), rel=1e-4)


def test_a_proportional_error_of_known_sd_is_fitted_by_maximum_likelihood(tmp_path):
    """sd = 0.3 f changes with the parameters, so -2LL is not a sum of squares (its 2 log sd
    terms move): BFGS on -2LL, at scipy's minimum of the closed form's -2LL."""
    from scipy.optimize import minimize

    df = table()
    t, dose, dv = df.TIME.to_numpy(), df.Dose.to_numpy(), df.DV.to_numpy()

    def m2ll(lx):
        f = conc(t, dose, *np.exp(lx))
        sd = 0.3 * np.abs(f)
        return float(np.sum(((dv - f) / sd) ** 2 + 2 * np.log(sd) + np.log(2 * np.pi)))

    ml = minimize(m2ll, np.log([1.5, 0.04, 0.08]), method="Nelder-Mead",
                  options=dict(xatol=1e-10, fatol=1e-10, maxiter=20000))  # fmt: skip
    r = tf.fit(problem(tmp_path, error="proportional(fix(0.3))"))
    assert r.optimizer == "BFGS" and r.status == "converged"
    np.testing.assert_allclose([r.estimates[k] for k in ("ka", "CL", "ke")], np.exp(ml.x),
                               rtol=1e-4)  # fmt: skip
    assert r.m2ll == pytest.approx(ml.fun, abs=1e-6)


def test_a_step_to_where_the_model_does_not_run_is_never_taken(tmp_path):
    """ka = sqrt(q) from a poor start with a small sd: the residuals are above 1000 sd, so a
    failed run must cost more than any run; the fit ends where the model runs and counts the
    failed runs it tried."""
    p = problem(tmp_path, error="additive(fix(0.001))", extra="ka = sqrt(q)\n",
                estimate="q = 0.005 in (-3, 5), CL = 0.2, ke = 0.1",
                options="starts=1 seed=2 checks=quick")  # fmt: skip
    r = tf.fit(p)
    assert np.isfinite(r.m2ll) and r.estimates["q"] > 0
    assert r.failed_runs > 0


def test_bounds_keep_values_inside(tmp_path):
    with pytest.warns(O.NearBoundWarning, match="ka ended at 1.3, near its upper bound 1.3"):
        r = tf.fit(problem(tmp_path, estimate="ka = 1.2 in (1, 1.3), CL = 0.05, ke = 0.1"))
    assert 1 < r.estimates["ka"] <= 1.3
    assert r.estimates["ka"] == pytest.approx(1.3, abs=1e-3)  # the optimum lies above
    assert r.scales[0] == ("logit", 1.0, 1.3)
    assert r.near_bounds == {"ka": ("upper", 1.3)}
    assert "at a bound: ka ended at 1.3, near its upper bound 1.3" in repr(r)  # the doctor


def test_one_sided_bounds(tmp_path, oracle):
    """A bound on one side: estimated as lo + exp(u) or hi - exp(u). Away from the bound the
    fit is scipy's; with the optimum beyond it, the estimate ends at the bound."""
    x, _, _ = oracle
    r = tf.fit(problem(tmp_path, estimate="ka = 1 in (0.5, inf), CL = 0.05 in (-inf, 1), ke = 0.1"))
    assert r.scales[:2] == [("above", 0.5), ("below", 1.0)]
    np.testing.assert_allclose([r.estimates[k] for k in ("ka", "CL", "ke")], x, rtol=1e-5)
    assert r.near_bounds == {}
    with pytest.warns(O.NearBoundWarning, match="ka ended at 1.3, near its upper bound 1.3"):
        r = tf.fit(problem(tmp_path, estimate="ka = 1.2 in (-inf, 1.3), CL = 0.05, ke = 0.1"))
    assert r.estimates["ka"] < 1.3
    assert r.estimates["ka"] == pytest.approx(1.3, abs=1e-3)  # the optimum lies above


def test_failed_starts_are_drawn_again(tmp_path):
    """ka = sqrt(q): a start with q < 0 does not run (NaN); it is drawn again."""
    p = problem(tmp_path, estimate="q = 1.2, CL = 0.05, ke = 0.1", extra="ka = sqrt(q)\n",
                options="starts=6 seed=4")  # fmt: skip
    p._params[0].start = 0.0  # q on its own scale (as is), so draws can go below 0
    assert "as is: q" in p.summary()
    r = tf.fit(p)
    assert r.redrawn > 0
    assert len(r.starts) == 6 and np.isfinite(r.starts["-2LL"]).all()


def test_failed_runs_are_counted_with_their_causes(tmp_path):
    """ka = sqrt(q) started at q = 0: the steps try q < 0, where the model does not run. Each
    is counted (the optimizer's own flag is the last accepted point's, which ran), with the
    cause TenQSP gives, and above 1 % of the runs the doctor warns."""
    p = problem(tmp_path, estimate="q = 1, CL = 0.05, ke = 0.1", extra="ka = sqrt(q)\n",
                options="starts=1")  # fmt: skip
    p._params[0].start = 0.0
    r = tf.fit(p)
    assert r.failed_runs > 0.01 * r.evaluations > 0
    assert sum(r.failure_causes.values()) == min(r.failed_runs, O.FAILED_KEPT)
    assert any("not finite" in c for c in r.failure_causes)
    (f,) = [f for f in r.doctor().findings if f.check == "failed runs"]
    assert f.severity == "warning" and f"{r.failed_runs} of {r.evaluations}" in f.message
    assert "failed runs during the fit" in str(r.passport())
    ok = tf.fit(problem(tmp_path, options="starts=1"))
    assert ok.failed_runs == 0 and ok.evaluations > 0 and ok.failure_causes == {}


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


def test_ctrl_c_during_the_standard_errors_keeps_the_fit():
    def gradient(u):
        raise KeyboardInterrupt

    cov = O.covariance(None, np.zeros(2), None, gradient=gradient)
    assert cov["covariance_status"] == O.STOPPED_COV and np.isnan(cov["se"]).all()


def test_a_time_limit_stops_the_fit_and_keeps_the_work(tmp_path):
    """options timeout: when the time is up the starts stop after their step, as for Ctrl-C,
    and the doctor says why."""
    r = tf.fit(problem(tmp_path, options="starts=2 seed=1 timeout=0.001"))
    assert r.status == "stopped" and np.isfinite(r.m2ll)
    assert r.stopped_by == "the time limit (options timeout, 0.001 s)"
    assert np.isnan(r.se).all() and r.covariance_status == O.STOPPED_SE
    note = [f for f in r.doctor().findings if f.check == "stopped"][0]
    assert "by the time limit" in note.message


@pytest.mark.parametrize(
    "value, seconds", [(90, 90.0), (1.5, 1.5), ("90s", 90.0), ("30min", 1800.0), ("2h", 7200.0)]
)
def test_a_time_limit_reads_seconds_or_a_time_with_its_unit(value, seconds):
    from tenqsp_fit.problem import seconds as read

    assert read(value) == seconds


@pytest.mark.parametrize("value", ["2 hours", "h", 0, -5, "0s", True])
def test_a_time_limit_that_cannot_be_read_is_refused(tmp_path, value):
    p = problem(tmp_path, options="starts=1")
    with pytest.raises(ValueError, match="timeout"):
        p.options(timeout=value)


def test_a_time_limit_from_text_is_shown_in_the_summary(tmp_path):
    assert "timeout 2h" in str(problem(tmp_path, options="starts=1 timeout=2h").summary())


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


def test_exact_ridge_has_no_standard_errors(tmp_path):
    """k1 and k2 enter only as k1*k2: the Hessian is singular, positive definite only by
    round-off; the standard errors fail instead of printing zeros."""
    times = [0.5, 1, 2, 3, 4, 6, 8]
    dv = [0.80, 0.61, 0.37, 0.21, 0.14, 0.050, 0.021]
    pd.DataFrame({"TIME": times, "DV": dv}).to_csv(tmp_path / "decay.csv", index=False)
    p = tf.Problem.from_text(f"""
model decay2
param k1 = 1, k2 = 0.5
d_t(A) = -k1*k2*A
A(0) = 1
observable amount = A
fit ridge
data "{tmp_path / "decay.csv"}"
observe amount = DV
estimate k1 = 0.8, k2 = 0.4
error amount ~ proportional(0.1)
method optimize
options checks=quick
""")
    r = tf.fit(p, progress="none")
    assert r.covariance_status.startswith("failed")
    assert np.all(np.isnan(r.table()["SE"].to_numpy(dtype=float)))


def test_progress_shows_one_step():
    """The progress is shown from a copy taken under the lock the starts write under, so a
    line never pairs one step's -2LL with another step's values."""
    import threading

    seen = []

    class Shown:
        def update(self, steps, m2ll, X, done, seconds, last=False, status=None):
            seen.append((steps, m2ll, X, done))

    steps, m2ll, done = np.array([3]), np.array([10.0]), np.array([False])
    X = np.array([[1.0, 2.0]])
    lock = threading.Lock()
    show = O.locked_show(Shown(), lock, steps, m2ll, X, done, 0.0)
    show()
    steps[0], m2ll[0], X[0] = 4, 9.0, [1.5, 2.5]  # the next step, after the copy
    assert seen[0][0][0] == 3 and seen[0][1][0] == 10.0 and list(seen[0][2][0]) == [1.0, 2.0]
    with lock:  # while a start writes, the display waits
        t = threading.Thread(target=show)
        t.start()
        t.join(0.2)
        assert t.is_alive()
    t.join()
    assert seen[1][1][0] == 9.0 and list(seen[1][2][0]) == [1.5, 2.5]


def test_trace_log_axis():
    """A quantity whose path spans more than a factor 10 is drawn on a log axis (a ridge
    walked in log k and log beta shows as two lines, not as a jump to zero); others not."""
    import matplotlib

    matplotlib.use("Agg")
    from tenqsp_fit.live import draw_trace

    k = [5e-8, 1e-8, 2e-9, 4e-10]
    g = [1.0, 0.95, 0.92, 0.91]
    m2ll = [9, 5, 4, 3.9]
    trace = pd.DataFrame({"start": 1, "iteration": range(4), "-2LL": m2ll, "k": k, "g": g})
    fig = draw_trace(trace, ["k", "g"])
    shown = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_yscale() for ax in shown] == ["linear", "log", "linear"]


def test_methods_unsuited_and_unknown(tmp_path):
    p = problem(tmp_path)
    with pytest.raises(ValueError, match="method laplace is for parameters that vary"):
        tf.fit(p, method="laplace")
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


def test_failed_runs_far_from_the_estimates_are_a_note():
    """A failed run more than one step (a factor 100) from the estimates is an overshoot the
    fit came back from: with all of them far, the doctor notes them instead of warning; a
    point not a number says nothing of where, and counts as near."""
    from types import SimpleNamespace

    from tenqsp_fit.doctor import _failed

    sc = [("log",), ("as is",)]
    x = np.array([1e-8, 2.0])
    far = [np.array([1e199, 2.0]), np.array([1e-8, 50.0]), np.array([np.inf, 2.0])]
    near = [np.array([3e-7, 2.0]), np.array([1e-9, -2.0]), np.array([np.nan, np.nan])]
    assert O.failures_near(far, x, sc) == 0
    assert O.failures_near(near, x, sc) == 3
    assert O.MAX_STEP == pytest.approx(np.log(100))

    def doctor(n_near):
        r = SimpleNamespace(failed_runs=30, evaluations=200, failures_near=n_near,
                            failure_causes={"too many steps": 30})  # fmt: skip
        (f,) = _failed(r)
        return f

    f = doctor(0)
    assert f.severity == "note"
    assert "all 30 checked were far from the estimates" in f.message
    f = doctor(4)
    assert f.severity == "warning" and "4 of the 30 checked near the estimates" in f.message


def test_steps_are_capped():
    """No quantity on a log or logit scale moves more than MAX_STEP in one step; one as is is
    not capped; the direction is kept."""
    import jax.numpy as jnp
    import optimistix as optx

    solver = O.capped(optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8), [("log",), ("as is",)])

    class Fixed(optx.AbstractDescent):
        def init(self, y, f):
            return None

        def query(self, y, f, state):
            return None

        def step(self, step_size, state):
            return jnp.array([46.0, 1000.0]), optx.RESULTS.successful

    d, _ = O._Capped(Fixed(), solver.descent.cap).step(1.0, None)
    np.testing.assert_allclose(d, [O.MAX_STEP, 1000.0 * O.MAX_STEP / 46.0])


def test_an_estimate_at_its_bound_has_no_se_and_the_others_hold_it(tmp_path):
    """CL (true 0.04) bounded in (0.045, 0.06) ends at 0.045: marked "lower", no SE or
    interval; ka's and ke's SEs are those of the fit with CL fixed at 0.045."""
    r = tf.fit(problem(tmp_path, estimate="ka = 1, CL = 0.05 in (0.045, 0.06), ke = 0.1",
                       options="starts=1 checks=quick"))  # fmt: skip
    t = r.table().set_index("quantity")
    assert t.loc["CL", "bound"] == "lower" and t.loc["ka", "bound"] == ""
    assert np.isnan(t.loc["CL", "SE"]) and np.isnan(t.loc["CL", "CI 2.5 %"])
    assert r.covariance_status == "ok"
    held = tf.fit(problem(tmp_path, estimate="ka = 1, ke = 0.1", extra="fix CL = 0.045\n",
                          options="starts=1 checks=quick"))  # fmt: skip
    h = held.table().set_index("quantity")
    for n in ("ka", "ke"):
        np.testing.assert_allclose(t.loc[n, "estimate"], h.loc[n, "estimate"], rtol=1e-3)
        np.testing.assert_allclose(t.loc[n, "SE"], h.loc[n, "SE"], rtol=0.02)
    assert "at its lower bound" in repr(r)


def _value(model, name):
    return next(p.value for p in model.to_ir().params if p.name == name)


def test_the_fitted_model_holds_the_estimates_at_full_precision(tmp_path):
    """result.model(): the model with the estimates in; written and read back, the values are
    the same to the last digit; result.save writes it as fitted_model.tq."""
    r = tf.fit(problem(tmp_path, options="starts=1 checks=quick"))
    fitted = r.model()
    again = tq.Model.from_text(fitted.to_text())
    for n in ("ka", "CL", "ke"):
        assert _value(again, n) == r.estimates[n]
    r.save(tmp_path / "run")
    assert _value(tq.load(tmp_path / "run" / "fitted_model.tq"), "CL") == r.estimates["CL"]


def test_the_fitted_model_keeps_the_entries_the_fit_does_not_set(tmp_path):
    two = """model two
axis s categorical [x, y]
param k[s] = {x: 0.1, y: 0.2}
field A(s)
d_t(A) = -k[s]*A
A(0) = 1
observable y = sum_s(A)/2
"""
    f = tmp_path / "d.csv"
    t = np.array([0.5, 1, 2, 4, 8])
    pd.DataFrame({"TIME": t, "DV": (np.exp(-0.3 * t) + np.exp(-0.2 * t)) / 2}).to_csv(
        f, index=False
    )
    fit = f'fit two\ndata "{f}"\nobserve y = DV\nestimate k[x] = 0.1\n'
    fit += "error y ~ additive(0.01)\nmethod optimize\noptions progress=none checks=quick\n"
    r = tf.fit(tf.Problem.from_text(two + fit))
    k = _value(tq.Model.from_text(r.model().to_text()), "k")
    assert k[1] == 0.2 and k[0] == r.estimates["k[x]"] and abs(k[0] - 0.3) < 1e-3
