# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The fitting problem: fit text, the Python API, the round trip, the summary and its checks."""

import numpy as np
import pandas as pd
import pytest
import tenqsp as tq

import tenqsp_fit as tf
from tenqsp_fit.distributions import make_error

PK = """model pk
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""

FIT = """
fit pk_pop
data "pk.csv"
unit ID
doses into depot
observe conc = DV

estimate ka ~ lognormal(median=1.5, sdlog=0.5)
estimate CL ~ lognormal(median=0.04, sdlog=0.3)
correlate ka, CL
estimate ke = 0.08
error conc ~ additive(0.7)

method laplace
"""


def pk_table(n=3):
    """A small NONMEM table: one dose (EVID 1) and four samples per unit, a WT column."""
    rows = []
    for i in range(1, n + 1):
        rows.append(dict(ID=i, TIME=0.0, AMT=4.0, DV=np.nan, EVID=1, WT=60 + 10 * i))
        for t in (0.5, 1, 2, 4):
            rows.append(dict(ID=i, TIME=t, AMT=np.nan, DV=5.0 / t, EVID=0, WT=60 + 10 * i))
    return pd.DataFrame(rows)


@pytest.fixture
def folder(tmp_path):
    pk_table().to_csv(tmp_path / "pk.csv", index=False, na_rep=".")
    (tmp_path / "pk_pop.tq").write_text(PK + FIT)
    (tmp_path / "pk.tq").write_text(PK)
    return tmp_path


def python_problem(folder):
    m = tq.load(folder / "pk.tq")
    p = tf.Problem(m, name="pk_pop")
    p.data(str(folder / "pk.csv"))
    p.unit("ID")
    p.doses(into="depot")
    p.observe("conc", "DV")
    p.estimate("ka", tf.lognormal(median=1.5, sdlog=0.5))
    p.estimate("CL", tf.lognormal(median=0.04, sdlog=0.3))
    p.correlate("ka", "CL")
    p.estimate("ke", 0.08)
    p.error("conc", tf.additive(0.7))
    p.method("laplace")
    return p


# ----------------------------------------------------------------------------- both ways
def test_text_reads_and_summarises(folder):
    p = tf.load(folder / "pk_pop.tq")
    assert p.name == "pk_pop"
    assert p.estimated() == [
        "ka.median", "ka.sdlog", "CL.median", "CL.sdlog", "ke", "corr(ka,CL)", "conc.additive",
    ]  # fmt: skip
    s = p.summary()
    assert "pk_pop: model pk (2 states)" in s
    assert "3 units, 12 observations, 3 doses" in s
    assert "conc <- DV (all rows)   12 rows, 3 units" in s
    assert "AMT -> depot   3 rows" in s
    assert "rtol 1e-6 (default)" in s
    assert repr(s) == str(s)  # shows itself at a prompt


def test_python_equals_text(folder):
    text = tf.load(folder / "pk_pop.tq")
    py = python_problem(folder)
    py._data.source = "pk.csv"  # the same file, as the text names it
    assert py == text


def test_round_trip(folder, monkeypatch):
    monkeypatch.chdir(folder)
    p = tf.load(folder / "pk_pop.tq")
    text = p.to_text()
    assert text.startswith("fit pk_pop\n")
    q = tf.Problem.from_text(text, model=p.model)
    assert q == p
    assert q.to_text() == text  # the written form is stable
    both = tf.Problem.from_text(p.to_text(model=True))
    assert both == p


def test_tenqsp_reads_the_model_part(folder):
    m = tq.load(folder / "pk_pop.tq")
    assert [o.name for o in m.to_ir().observables] == ["conc"]


def test_fit_part_in_its_own_file(folder):
    (folder / "fit_only.tq").write_text(FIT)
    p = tf.load(folder / "fit_only.tq", model="pk.tq")
    assert p.name == "pk_pop"
    assert p == tf.load(folder / "pk_pop.tq")


def test_all_statements_round_trip(folder, monkeypatch):
    monkeypatch.chdir(folder)
    text = (
        PK
        + """
fit full
data "pk.csv" with DV=DV
unit ID
doses CMT 1 into depot, 2 into centr
observe conc = DV where EVID == 0 and TIME > 0
ignore DV where TIME == 0

estimate ka ~ lognormal(median=1.5, sdlog=fix(0.5))
estimate CL_pop = 0.04
estimate CL ~ lognormal(median=CL_pop*(WT/70)^0.75, cv=0.3)
estimate ke = 0.08 in (0.001, 1)
correlate ka, CL = -0.3
error conc ~ combined(0.1, fix(0.2))
censor conc below 0.05
prior ke ~ gamma(mean=0.08, cv=0.5)

method nuts chains=4 samples=500 target_accept=0.95
options rtol=1e-8 starts=8 seed=2
"""
    )
    p = tf.Problem.from_text(text)
    q = tf.Problem.from_text(p.to_text(model=True))
    assert p == q
    assert q.to_text() == p.to_text()
    assert "estimate CL ~ lognormal(median=CL_pop*(WT/70)^0.75, cv=0.3)" in p.to_text()
    assert p.estimated() == [
        "ka.median", "CL_pop", "CL.cv", "ke", "corr(ka,CL)", "conc.additive",
    ]  # fmt: skip


def test_continuation_lines(folder, monkeypatch):
    monkeypatch.chdir(folder)
    text = PK + FIT.replace(
        "estimate ka ~ lognormal(median=1.5, sdlog=0.5)",
        "estimate ka ~ lognormal(median=1.5,\n    sdlog=0.5)   # a comment",
    )
    assert tf.Problem.from_text(text) == tf.load(folder / "pk_pop.tq")


# ----------------------------------------------------------------------------- links
TWO = """model phage
param g=1, k=1e-7
d_t(B) = g*B - k*B*P
d_t(P) = 50*k*B*P - 0.1*P
observable bact = B
observable phage = P
"""


def long_table():
    rows = []
    for t in (1.0, 2.0, 3.0):
        rows.append(dict(TIME=t, DV=6.0, DVID=1))
        rows.append(dict(TIME=t, DV=7.0, DVID=2))
    return pd.DataFrame(rows)


def two_problem(df):
    p = tf.Problem(tq.Model.from_text(TWO))
    p.data(df)
    p.estimate("g", 1.0)
    p.method("optimize")
    return p


def test_missing_link_proposes_the_lines():
    p = two_problem(long_table())
    with pytest.raises(tf.ProblemError) as e:
        p.check()
    msg = str(e.value)
    assert "DV rows by DVID:  1 -> 3 rows;  2 -> 3 rows" in msg
    assert "observe bact = DV where DVID == 1" in msg
    assert "observe phage = DV where DVID == 2" in msg
    assert "ignore DV where DVID == 2" in msg


def test_partly_linked_proposes_the_rest():
    p = two_problem(long_table())
    p.observe("log10(bact)", "DV", where="DVID == 1")
    p.error("log10(bact)", tf.additive(0.2))
    with pytest.raises(tf.ProblemError) as e:
        p.check()
    msg = str(e.value)
    assert "not linked" in msg and "observe phage = DV where DVID == 2" in msg
    p.ignore("DV", where="DVID == 2")
    p.check()
    assert "left out: DV where DVID == 2   3 rows" in p.summary()
    assert "bact.additive" not in p.estimated() and "log10(bact).additive" in p.estimated()


def test_rows_linked_twice():
    p = two_problem(long_table())
    p.observe("bact", "DV")
    p.observe("phage", "DV", where="DVID == 2")
    p.error("bact", tf.additive(0.2))
    p.error("phage", tf.additive(0.2))
    with pytest.raises(tf.ProblemError, match="linked twice"):
        p.check()


def test_wide_table_proposes_columns():
    df = pd.DataFrame(dict(TIME=[1.0, 2.0], bact=[6.0, 5.0], phage=[7.0, 8.0]))
    p = two_problem(df)
    with pytest.raises(tf.ProblemError) as e:
        p.check()
    assert "observe bact = bact" in str(e.value) and "observe phage = phage" in str(e.value)
    p.observe("bact", "bact")
    p.observe("phage", "phage")
    p.error("bact", tf.proportional(0.1))
    p.error("phage", tf.proportional(0.1))
    p.check()


def test_string_value_needs_quotes():
    df = long_table()
    df["DVID"] = df["DVID"].map({1: "bact", 2: "phage"})
    p = two_problem(df)
    p.observe("bact", "DV", where="DVID == bact")
    p.observe("phage", "DV", where='DVID == "phage"')
    p.error("bact", tf.additive(0.2))
    p.error("phage", tf.additive(0.2))
    with pytest.raises(tf.ProblemError, match='quote it: "bact"'):
        p.check()


def test_missing_doses_line_proposes(folder):
    df = pk_table()
    df["CMT"] = 1
    p = python_problem(folder)
    p._doses = None
    p._data = None
    p.data(df)
    with pytest.raises(tf.ProblemError) as e:
        p.check()
    assert "doses CMT 1 into depot" in str(e.value) and "doses into depot" in str(e.value)


def test_missing_error_model(folder):
    p = python_problem(folder)
    p._errors.clear()
    with pytest.raises(tf.ProblemError, match=r"error conc ~ additive"):
        p.check()


# ----------------------------------------------------------------------------- parameters
ONE_RATE = """model one_rate
param mu = 0.3, V = 2
ka = mu
ke = mu
d_t(depot) = -ka*depot
d_t(C) = ka*depot - ke*C
observable conc = C/V
"""


def test_reparameterised_model_estimates_mu(folder):
    p = tf.Problem(tq.Model.from_text(ONE_RATE))
    with pytest.raises(
        ValueError,
        match=r"ka is not a parameter but a formula of the model "
        r"\(ka = mu\); estimate mu instead",
    ):
        p.estimate("ka", tf.lognormal(median=0.3, sdlog=1))
    p.data(pk_table())
    p.unit("ID")
    p.doses(into="depot")
    p.observe("conc")
    p.estimate("mu", tf.lognormal(median=0.3, sdlog=1))
    p.error("conc", tf.proportional(0.1))
    p.method("laplace")
    s = p.summary()
    assert "ka = mu (model); ke = mu (model)" in s
    assert "from model V=2" in s


def test_formula_after_fit_ties_parameters(folder):
    p = python_problem(folder)
    p._params = [d for d in p._params if d.name != "ke"]
    with pytest.raises(ValueError, match="already estimated"):
        p.define("ka", "mu")
    p.estimate("mu", 0.1)
    p.define("ke", "mu")
    p.check()
    assert "ke = mu (this fit)" in p.summary()


def test_define_needs_a_model_parameter():
    p = tf.Problem(tq.Model.from_text(ONE_RATE))
    with pytest.raises(ValueError, match="already sets ka = mu"):
        p.define("ka", "2*mu")
    with pytest.raises(ValueError, match="no parameter 'kk'"):
        p.define("kk", "mu")


def test_covariate_in_a_distribution(folder):
    p = python_problem(folder)
    p._params = [d for d in p._params if d.name != "CL"]
    p._correlations.clear()
    p.estimate("CL_pop", 0.04)
    p.estimate("CL", tf.lognormal(median="CL_pop*(WT/70)^0.75", sdlog=0.3))
    p.check()
    assert "CL.sdlog" in p.estimated() and "CL.median" not in p.estimated()
    p2 = python_problem(folder)
    p2._params = [d for d in p2._params if d.name != "CL"]
    p2._correlations.clear()
    p2.estimate("CL", tf.lognormal(median="CL_pp*(WT/70)^0.75", sdlog=0.3))
    with pytest.raises(tf.ProblemError, match="'CL_pp' is not an estimated or fixed quantity"):
        p2.check()


def test_new_quantity_must_be_used(folder):
    p = python_problem(folder)
    p.estimate("kz", 1.0)
    with pytest.raises(tf.ProblemError, match="'kz' is not a parameter of the model"):
        p.check()


def test_varying_needs_a_unit(folder):
    p = python_problem(folder)
    p._unit = None
    with pytest.raises(tf.ProblemError, match="no unit is named; add: unit ID"):
        p.check()


def test_correlate_needs_varying(folder):
    p = python_problem(folder)
    p.correlate("ka", "ke")
    with pytest.raises(tf.ProblemError, match="ke does not vary between units"):
        p.check()


def test_estimate_checks():
    p = tf.Problem(tq.Model.from_text(PK))
    p.estimate("ke", 0.08, bounds=(0.001, 1))
    with pytest.raises(ValueError, match="already estimated"):
        p.fix("ke", 0.1)
    with pytest.raises(ValueError, match="inside the bounds"):
        p.estimate("ka", 2.0, bounds=(0, 1))
    with pytest.raises(ValueError, match="state or an observable"):
        p.estimate("depot", 1.0)
    with pytest.raises(ValueError, match="arrives in a later release"):
        p.estimate("CL", tf.gamma(mean=1, cv=0.5))
    with pytest.raises(ValueError, match="between -1 and 1"):
        p.correlate("ka", "CL", start=1.5)


def test_method_must_suit(folder):
    p = python_problem(folder)
    p._method = None
    with pytest.raises(tf.ProblemError, match="no method; add: method laplace"):
        p.check()
    p.method("optimize")
    with pytest.raises(tf.ProblemError, match="use method laplace"):
        p.check()
    with pytest.raises(TypeError, match="unknown option 'chain'. Did you mean 'chains'"):
        tf.Problem(tq.Model.from_text(PK)).method("nuts", chain=4)
    for bad in (1.0, 0, "0.9", True):
        with pytest.raises(ValueError, match="target_accept is a number between 0 and 1"):
            tf.Problem(tq.Model.from_text(PK)).method("nuts", target_accept=bad)
    with pytest.raises(TypeError, match="Did you mean 'starts'"):
        tf.Problem(tq.Model.from_text(PK)).options(start=8)


def test_written_likelihood_with_varying_parameters(folder):
    """A written likelihood with parameters that vary between units is refused by check()
    (laplace and nuts), not when the fit starts. pi is known; 'e' gets exp(1), not 'estimate e'."""
    p = python_problem(folder)
    p._errors.clear()
    p.likelihood("conc", "-0.5*((DV - conc)/0.7)^2 - log(0.7) - 0.5*log(2*pi)")
    with pytest.raises(tf.ProblemError) as e:
        p.check()
    msg = str(e.value)
    assert "written likelihood with parameters that vary between units (method laplace)" in msg
    assert "pi" not in msg
    p._likelihoods.clear()
    p.likelihood("conc", "-0.5*((DV - conc)/0.7)^2 - log(0.7*sqrt(2*pi*e))")
    with pytest.raises(tf.ProblemError) as e:
        p.check()
    assert "'e' is not a known name: formulas have pi but no constant 'e'. Write exp(1)" in str(
        e.value
    )
    assert "estimate e" not in str(e.value)


# ----------------------------------------------------------------------------- distributions
def test_distributions():
    d = tf.lognormal(sdlog=0.5, median=1.5)
    assert str(d) == "lognormal(median=1.5, sdlog=0.5)"  # location first, as written
    assert str(tf.exponential(mean=1e-6)) == "exponential(mean=1e-6)"
    assert str(tf.lognormal(median=1, sdlog=tf.fix(0.5))) == "lognormal(median=1, sdlog=fix(0.5))"
    with pytest.raises(TypeError, match="unknown argument 'sd'"):
        tf.lognormal(mean=1, sd=1)
    with pytest.raises(TypeError, match="not a valid combination"):
        tf.lognormal(median=1)
    with pytest.raises(ValueError, match="'sdlog' must be > 0"):
        tf.lognormal(median=1, sdlog=-1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        tf.logitnormal(median=1.2, sdlogit=0.4)
    with pytest.raises(ValueError, match="'lower' must be below 'upper'"):
        tf.uniform(lower=1, upper=0)
    with pytest.raises(ValueError, match="Cannot read formula 'CL_pop\\*'"):
        tf.lognormal(median="CL_pop*", sdlog=0.3)
    assert str(tf.power(0.1, tf.fix(0.5))) == "power(0.1, fix(0.5))"
    with pytest.raises(TypeError, match="combined\\(a, b\\) takes 2"):
        make_error("combined", [0.1])


# ----------------------------------------------------------------------------- text errors
def test_text_errors_name_their_lines(folder, monkeypatch):
    monkeypatch.chdir(folder)
    bad = (
        PK
        + """
fit bad
data "pk.csv"
unit ID
doses into dept
observe conc = DV
estimat ke = 0.08
estimate ka ~ lognormal(mean=1, sd=1)
error conc ~ additiv(0.7)
method laplace
"""
    )
    with pytest.raises(tf.FitTextError) as e:
        tf.Problem.from_text(bad)
    msg = str(e.value)
    assert "4 problems:" in msg
    assert "line 10: doses into dept" in msg and "Did you mean 'depot'" in msg
    assert "line 12: estimat ke = 0.08" in msg and "Did you mean 'estimate'" in msg
    assert "line 13:" in msg and "unknown argument 'sd'" in msg
    assert "line 14:" in msg and "Did you mean 'additive'" in msg


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("doses CMT 1 into depot, 1 into centr", "CMT 1 is given twice"),
        ("options starts=2 starts=5", "starts is given twice"),
        ("method nuts chains=2 chains=8", "chains is given twice"),
    ],
)
def test_a_key_given_twice_on_one_line_is_refused(line, message):
    text = PK + "fit twice\n" + line + "\n"
    with pytest.raises(tf.FitTextError, match=message):
        tf.Problem.from_text(text)


def test_check_errors_name_their_lines(folder, monkeypatch):
    monkeypatch.chdir(folder)
    text = PK + FIT.replace("correlate ka, CL", "correlate ka, ke")
    p = tf.Problem.from_text(text)
    with pytest.raises(tf.ProblemError, match=r"line 15: correlate ka, ke\ncorrelate: ke does not"):
        p.check()


def test_no_fit_line():
    with pytest.raises(tf.FitTextError, match="no 'fit' line"):
        tf.Problem.from_text(PK)


def test_one_sided_bounds_text_and_python():
    fit = "fit b\nestimate ke = 0.08 in (0.001, inf), ka = -2 in (-inf, 0)\nmethod optimize\n"
    p = tf.Problem.from_text(PK + fit)
    assert "estimate ke = 0.08 in (0.001, inf)" in p.to_text()
    assert "estimate ka = -2 in (-inf, 0)" in p.to_text()
    assert "above a bound: ke; below a bound: ka" in p.summary()
    q = tf.Problem(tq.Model.from_text(PK), name="b")
    q.estimate("ke", 0.08, bounds=(0.001, None))
    q.estimate("ka", -2, bounds=(-np.inf, 0))
    q.method("optimize")
    assert q == p
    with pytest.raises(ValueError, match="inside the bounds"):
        q.estimate("CL", 0.5, bounds=(1, None))
    with pytest.raises(ValueError, match="both open; leave out bounds"):
        q.estimate("CL", 0.5, bounds=(None, np.inf))
    with pytest.raises(ValueError, match="lower bound 1 must be below the upper bound 0"):
        q.estimate("CL", 0.5, bounds=(1, 0))
    with pytest.raises(TypeError, match="bounds are two numbers"):
        q.estimate("CL", 0.5, bounds=(np.inf, None))


def test_model_statement_after_fit():
    with pytest.raises(tf.FitTextError, match="the model goes above it"):
        tf.Problem.from_text(PK + "fit x\nparam z = 1\n")


def test_spread_zero_means_one_value(folder, monkeypatch):
    """A distribution with spread 0 does not vary: one estimated value, on the scale its
    family sets (normal: any sign, even from a start above 0; lognormal: log scale;
    logitnormal: within (0, 1)); formulas come first in the text."""
    monkeypatch.chdir(folder)
    text = PK.replace("param ka=1.6", "param F=0.5, ka=1.6") + (
        'fit cov\ndata "pk.csv"\nunit ID\ndoses into depot\nobserve conc = DV\n'
        "CL = CL_std*(WT/70)^beta\n"
        "estimate ka ~ lognormal(median=1.5, sdlog=0.5)\n"
        "estimate CL_std ~ lognormal(median=0.04, sdlog=0.3)\n"
        "estimate beta ~ normal(mean=0.75, sd=0)\n"
        "estimate ke ~ lognormal(median=0.08, sdlog=fix(0))\n"
        "estimate F ~ logitnormal(median=0.5, sdlogit=0)\n"
        "error conc ~ additive(0.7)\nmethod laplace\n"
    )
    p = tf.Problem.from_text(text)
    assert [d.name for d in p._varying()] == ["ka", "CL_std"]
    names = p.estimated()
    assert {"beta", "ke", "F"} <= set(names) and "beta.mean" not in names
    from tenqsp_fit.optimize import scales

    names, x0, bounds = p._starts()
    sc = dict(zip(names, scales(names, x0, bounds, p._kinds(names)), strict=True))
    assert sc["beta"] == ("as is",) and sc["ke"] == ("log",) and sc["F"][0] == "logit"
    s = p.summary()
    assert "one value  beta ~ normal(mean=0.75, sd=0)" in s
    assert "ke ~ lognormal(median=0.08, sdlog=fix(0))" in s
    out = p.to_text()
    assert out.index("CL = CL_std*(WT/70)^beta") < out.index("estimate ka")
    assert "estimate beta ~ normal(mean=0.75, sd=0)" in out
    q = tf.Problem.from_text(p.to_text(model=True))
    assert p == q and q.to_text() == out
    # the shorthand shows its long form
    p2 = tf.Problem.from_text(PK + FIT.replace("pk.csv", str(folder / "pk.csv")))
    assert "one value  ke ~ lognormal(median=0.08, sdlog=0)" in p2.summary()


def test_spread_zero_needs_a_number(folder):
    p = tf.Problem.from_text(PK + FIT.replace("pk.csv", str(folder / "pk.csv")))
    with pytest.raises(ValueError, match="does not vary .* write it as a formula: V = V0"):
        p.estimate("V", tf.lognormal(median="V0*(WT/70)", sdlog=0))
    with pytest.raises(ValueError, match="its median is fixed; write fix V = 2"):
        p.estimate("V", tf.lognormal(median=tf.fix(2), sdlog=0))
    with pytest.raises(ValueError, match="'shape' must be > 0"):
        tf.weibull(shape=0, scale=1)  # no spread to set to 0: it always varies
    with pytest.raises(ValueError, match="'sd' must be > 0"):
        p.prior("ke", tf.normal(mean=0.08, sd=0))  # a prior needs a spread


def test_impossible_priors_are_refused():
    with pytest.raises(ValueError, match="variance must be below mean"):
        tf.beta(mean=0.5, sd=0.6)
    with pytest.raises(ValueError, match="'mean' must be > 0"):
        tf.gamma(mean=-1, sd=1)
    with pytest.raises(ValueError, match="a cv of a mean of 0 is an sd of 0"):
        tf.normal(mean=0, cv=0.5)
    assert str(tf.beta(mean=0.5, sd=0.2)) == "beta(mean=0.5, sd=0.2)"


def test_doses_and_method_problems_name_their_lines(folder, monkeypatch):
    monkeypatch.chdir(folder)
    text = PK + FIT.replace("doses into depot", "doses Dose at t=0 into depot")
    with pytest.raises(tf.ProblemError, match=r"line 10: doses Dose at t=0 into depot"):
        tf.Problem.from_text(text).check()
    with pytest.raises(tf.ProblemError, match=r"line 19: method optimize\nmethod optimize fits"):
        tf.Problem.from_text(PK + FIT.replace("method laplace", "method optimize")).check()


def test_text_round_trips_column_maps_and_no_time_limit(tmp_path):
    df = pd.DataFrame({"Subject": [1, 1], "Time (h)": [0.0, 1.0], "conc mg/L": [np.nan, 1.0],
                       "AMT": [5, 0]})  # fmt: skip
    df.to_csv(tmp_path / "sp.csv", index=False)
    p = tf.Problem(tq.Model.from_text(PK), name="x")
    p.data(str(tmp_path / "sp.csv"), ID="Subject", TIME="Time (h)", DV="conc mg/L")
    p.options(timeout=None)  # no limit, as when it is left out
    assert 'TIME="Time (h)", DV="conc mg/L"' in p.to_text()
    q = tf.Problem.from_text(p.to_text(), model=tq.Model.from_text(PK))
    assert q._data.columns == p._data.columns and "timeout" not in q._options


def test_a_dose_column_time_below_0_is_refused():
    p = tf.Problem(tq.Model.from_text(PK))
    with pytest.raises(ValueError, match="at is 0 or later"):
        p.doses(column="Dose", at=-1, into="depot")


def test_load_names_a_problem_from_any_file_name(folder):
    (folder / "pk-pop v2.tq").write_text(PK + FIT.replace("fit pk_pop", "fit"))
    assert tf.load(folder / "pk-pop v2.tq").name == "pk_pop_v2"


# ----------------------------------------------------------------------------- a shared SD
TWO_LINKS = """
fit pk_pop
data "pk.csv"
unit ID
doses into depot
observe conc = DV where TIME < 2
observe conc2 = DV where TIME >= 2
estimate ka = 1.5, CL = 0.04, ke = 0.08
estimate s = 0.5
error conc ~ additive(s)
error conc2 ~ proportional(fix(0.1))
method optimize
"""


def test_a_shared_sd_reads_the_same_in_text_and_python(folder, monkeypatch):
    monkeypatch.chdir(folder)
    text = TWO_LINKS.replace("proportional(fix(0.1))", "additive(s)")
    m = tq.Model.from_text(PK + "observable conc2 = centr*ke/CL\n")
    p = tf.Problem.from_text(text, model=m)
    assert p.estimated() == ["ka", "CL", "ke", "s"]
    assert "error conc ~ additive(s)" in p.to_text()
    q = tf.Problem(m, name="pk_pop")
    q.data("pk.csv")
    q.unit("ID")
    q.doses(into="depot")
    q.observe("conc", "DV", where="TIME < 2")
    q.observe("conc2", "DV", where="TIME >= 2")
    q.estimate("ka", 1.5)
    q.estimate("CL", 0.04)
    q.estimate("ke", 0.08)
    q.estimate("s", 0.5)
    q.error("conc", tf.additive("s"))
    q.error("conc2", tf.additive("s"))
    q.method("optimize")
    assert q == p
    p.check()


@pytest.mark.parametrize(
    "change, message",
    [
        (("estimate s = 0.5\n", ""), r"'s' is not declared; add estimate s = <start>"),
        (
            ("estimate s = 0.5", "estimate s ~ lognormal(median=0.5, sdlog=0.3)"),
            r"'s' varies between units; the additive part",
        ),
        (("additive(s)", "additive(ke)"), r"'ke' is a parameter of the model"),
        (("estimate s = 0.5", "estimate s = 0.5 in (-1, 1)"), r"must be > 0; give 's' a positive"),
        (("estimate s = 0.5", "fix s = -0.5"), r"must be > 0; give 's' a positive value"),
    ],
)
def test_a_named_sd_that_cannot_be_used(folder, monkeypatch, change, message):
    monkeypatch.chdir(folder)
    m = tq.Model.from_text(PK + "observable conc2 = centr*ke/CL\n")
    p = tf.Problem.from_text(TWO_LINKS.replace(*change), model=m)
    with pytest.raises(tf.ProblemError, match=message):
        p.check()


def test_an_error_model_argument_is_not_a_formula(folder, monkeypatch):
    monkeypatch.chdir(folder)
    m = tq.Model.from_text(PK + "observable conc2 = centr*ke/CL\n")
    with pytest.raises(tf.FitTextError, match=r"name of an estimated or fixed quantity"):
        tf.Problem.from_text(TWO_LINKS.replace("additive(s)", "additive(2*s)"), model=m)
