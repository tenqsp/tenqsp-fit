# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``tf.objective``: -2 log-likelihood of a problem. Oracles: oral one-compartment PK in
closed form, with the error models, M3 and the floor near zero written out by hand."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import tenqsp_fit as tf

jax.config.update("jax_enable_x64", True)

PK = """model pk
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""
TIMES = (0.5, 1, 2, 4, 8, 24)
DOSE = {1: 4.0, 2: 5.0, 3: 6.0}
WT = {1: 50.0, 2: 70.0, 3: 90.0}
TIGHT = "options rtol=1e-10 atol=1e-12\n"


def conc(t, dose, ka, CL, ke):
    t = np.asarray(t, float)
    return dose * ka * ke / (CL * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def table():
    rng = np.random.default_rng(1)
    rows = []
    for i in (1, 2, 3):
        for t in TIMES:
            y = conc(t, DOSE[i], 1.5, 0.04, 0.08) * (1 + 0.1 * rng.normal()) + 0.2 * rng.normal()
            rows.append(dict(ID=i, TIME=t, DV=y, Dose=DOSE[i], WT=WT[i]))
    return pd.DataFrame(rows)


def make(f, fit, model=PK, method="optimize"):
    """A problem on the data file ``f`` with the fit statements ``fit``."""
    return tf.Problem.from_text(
        model
        + f'fit pk\ndata "{f}"\nunit ID\ndoses Dose at t=0 into depot\n'
        + fit
        + f"\nmethod {method}\n"
        + TIGHT
    )


@pytest.fixture
def data_file(tmp_path):
    f = tmp_path / "pk.csv"
    table().to_csv(f, index=False)
    return f


def hand_normal(dv, f, sd):
    return np.sum(((dv - f) / sd) ** 2 + 2 * np.log(sd) + math.log(2 * math.pi))


def predictions(x, cl=None):
    df = table()
    ka, CL, ke = x
    cl = np.array([cl[i] for i in df.ID]) if cl is not None else CL
    return df.DV.to_numpy(), conc(df.TIME, df.Dose, ka, cl, ke)


@pytest.mark.parametrize(
    "error, sd",
    [
        ("additive(0.3)", lambda f: 0.3 + 0 * f),
        ("proportional(0.2)", lambda f: 0.2 * np.sqrt(f**2 + 1e-24)),
        ("combined(0.3, 0.2)", lambda f: np.sqrt(0.09 + (0.2 * f) ** 2)),
        ("power(0.2, 0.8)", lambda f: 0.2 * np.sqrt(f**2 + 1e-24) ** 0.8),
    ],
)
def test_error_models_against_the_closed_form(data_file, error, sd):
    obj = tf.objective(
        make(
            data_file,
            f"observe conc = DV\nestimate ka = 1.4, CL = 0.05, ke = 0.07\nerror conc ~ {error}",
        )
    )
    assert obj.names[:3] == ("ka", "CL", "ke") and obj.n_observations == 18
    dv, f = predictions(obj.x0[:3])
    np.testing.assert_allclose(float(obj(obj.x0)), hand_normal(dv, f, sd(f)), rtol=1e-8)


def test_gradient_against_finite_differences(data_file):
    obj = tf.objective(
        make(
            data_file,
            "observe conc = DV\nestimate ka = 1.4, CL = 0.05, ke = 0.07\nerror conc ~ "
            "combined(0.3, 0.2)",
        )  # fmt: skip
    )
    x = jnp.asarray(obj.x0)
    g = jax.grad(obj)(x)
    for i in range(len(x)):
        h = 1e-6 * abs(float(x[i]))
        fd = (float(obj(x.at[i].add(h))) - float(obj(x.at[i].add(-h)))) / (2 * h)
        np.testing.assert_allclose(float(g[i]), fd, rtol=1e-5)


def test_fixed_error_quantities_and_bounds(data_file):
    obj = tf.objective(
        make(
            data_file,
            "observe conc = DV\nestimate ka = 1.4 in (0.1, 10), CL = 0.05, ke = 0.07\n"
            "error conc ~ combined(fix(0.3), 0.2)",
        )
    )
    assert obj.names == ("ka", "CL", "ke", "conc.proportional")
    assert obj.bounds[0] == (0.1, 10) and obj.bounds[1] == (-np.inf, np.inf)
    dv, f = predictions(obj.x0[:3])
    np.testing.assert_allclose(
        float(obj(obj.x0)), hand_normal(dv, f, np.sqrt(0.09 + (0.2 * f) ** 2)), rtol=1e-8
    )
    assert obj.unpack(obj.x0)["conc.proportional"] == 0.2
    assert obj.status(obj.x0) == {1: "ok", 2: "ok", 3: "ok"}
    assert "3 units, 18 measurements" in repr(obj)


def test_covariates_by_a_formula_of_the_fit(data_file):
    obj = tf.objective(
        make(
            data_file,
            "observe conc = DV\nestimate ka = 1.4, CL_pop = 0.05, ke = 0.07\n"
            "CL = CL_pop*(WT/70)^0.75\nerror conc ~ additive(0.3)",
        )
    )
    ka, cl_pop, ke, _ = obj.x0
    cl = {i: cl_pop * (WT[i] / 70) ** 0.75 for i in WT}
    dv, f = predictions((ka, None, ke), cl=cl)
    np.testing.assert_allclose(float(obj(obj.x0)), hand_normal(dv, f, 0.3), rtol=1e-8)


def m3_problem(tmp_path, df, censor, column_map=""):
    f_ = tmp_path / "pk.csv"
    df.to_csv(f_, index=False)
    return tf.Problem.from_text(
        PK + f'fit pk\ndata "{f_}"{column_map}\nunit ID\ndoses Dose at t=0 into depot\n'
        f"observe conc = DV\n{censor}\nestimate ka = 1.4, CL = 0.05, ke = 0.07\n"
        "error conc ~ additive(0.3)\nmethod optimize\n" + TIGHT
    )


def m3_want(obj, c):
    dv, f = predictions(obj.x0[:3])
    return hand_normal(dv[~c], f[~c], 0.3) - 2 * np.sum(norm.logcdf((2.0 - f[c]) / 0.3))


@pytest.mark.parametrize(
    "how, censor",
    [
        ("flag", "censor conc where CENS == 1 below LLOQ"),
        ("text", 'censor conc where DV == "BLQ" below 2'),
        ("below", "censor conc below 2"),
    ],
)
def test_m3_flag_text_or_below_the_limit(tmp_path, how, censor):
    df = table().assign(LLOQ=2.0, CENS=0)
    low = (df.DV < 2.0).to_numpy()
    assert low.sum() >= 2
    df["CENS"] = low.astype(int)
    if how == "flag":
        df["DV"] = np.where(low, 2.0, df.DV)  # BLQ written as the limit, with a flag
    elif how == "text":
        df["DV"] = np.where(low, "BLQ", df.DV.astype(str))
    p = m3_problem(tmp_path, df, censor)
    obj = tf.objective(p)
    assert obj.n_censored == low.sum()
    np.testing.assert_allclose(float(obj(obj.x0)), m3_want(obj, low), rtol=1e-8)
    s = p.summary()
    assert f"censored (M3, limit {'LLOQ' if how == 'flag' else '2'}): {low.sum()} rows" in s
    assert tf.Problem.from_text(p.to_text(model=True)) == p


def test_flagged_rows_may_have_no_dv(tmp_path):
    """A BLQ row with DV missing ("."): read through the flag, with or without EVID."""
    df = table().assign(CENS=0)
    low = (df.DV < 2.0).to_numpy()
    df["CENS"] = low.astype(int)
    df["DV"] = np.where(low, np.nan, df.DV)
    obj = tf.objective(m3_problem(tmp_path, df, "censor conc where CENS == 1 below 2"))
    assert obj.n_censored == low.sum()
    np.testing.assert_allclose(float(obj(obj.x0)), m3_want(obj, low), rtol=1e-8)


def test_values_equal_to_the_limit_need_a_rule(tmp_path):
    df = table()
    low = (df.DV < 2.0).to_numpy()
    df["DV"] = np.where(low, 2.0, df.DV)  # BLQ written as the limit, no flag
    with pytest.raises(tf.ProblemError, match="measurements equal the limit exactly"):
        m3_problem(tmp_path, df, "censor conc below 2").check()
    p = m3_problem(tmp_path, df, "censor conc where DV < 2 below 2")  # measured values
    assert tf.objective(p).n_censored == 0


def test_text_in_dv_without_a_censor_rule(tmp_path):
    df = table()
    df["DV"] = df.DV.astype(str)
    df.loc[3, "DV"] = "BLQ"
    with pytest.raises(tf.ProblemError, match=r'DV: row 4 is not numbers .*: "BLQ"\. If a text'):
        m3_problem(tmp_path, df, "censor conc below 2").check()
    with pytest.raises(tf.ProblemError, match="compare it with a quoted text"):
        m3_problem(tmp_path, df, "censor conc where DV < 2 below 2").check()


def test_log_scale_link_and_the_floor_near_zero(tmp_path):
    """A unit whose prediction is 0 (no dose) on a log10 link: finite, and the gradient finite."""
    df = table()
    df["logDV"] = np.log10(np.abs(df.pop("DV")) + 1e-3)
    df.loc[df.ID == 3, "Dose"] = 0.0  # unit 3 predicts exactly 0
    f_ = tmp_path / "pk.csv"
    df.to_csv(f_, index=False)
    p = tf.Problem.from_text(
        PK + f'fit pk\ndata "{f_}" with DV=logDV\nunit ID\ndoses Dose at t=0 into depot\n'
        "observe log10(conc) = DV\nestimate ka = 1.4, CL = 0.05, ke = 0.07\n"
        "error log10(conc) ~ additive(0.2)\nmethod optimize\n" + TIGHT
    )
    obj = tf.objective(p)
    assert obj.floor == 1e-12  # the solver's atol
    x = jnp.asarray(obj.x0)
    assert np.isfinite(float(obj(x))) and np.all(np.isfinite(jax.grad(obj)(x)))
    dv, f = predictions(obj.x0[:3])
    dose = table().Dose.to_numpy() * (table().ID.to_numpy() != 3)
    f = conc(table().TIME, dose, *obj.x0[:3])
    eps = 1e-12
    floored = 0.5 * (f + np.sqrt(f**2 + 4 * eps**2))
    want = hand_normal(df.logDV.to_numpy(), np.log10(floored), 0.2)
    np.testing.assert_allclose(float(obj(x)), want, rtol=1e-8)


def test_the_models_own_doses_without_a_doses_line(tmp_path):
    model = PK.replace("observable", "dose depot at t=0: 4\nobservable")
    df = table()
    df = df[df.ID == 1].drop(columns="Dose")
    f_ = tmp_path / "pk.csv"
    df.to_csv(f_, index=False)
    p = tf.Problem.from_text(
        model + f'fit pk\ndata "{f_}"\nobserve conc = DV\nestimate ka = 1.4, CL = 0.05, '
        "ke = 0.07\nerror conc ~ additive(0.3)\nmethod optimize\n" + TIGHT
    )
    obj = tf.objective(p)
    dv = df.DV.to_numpy()
    f = conc(df.TIME, 4.0, *obj.x0[:3])
    np.testing.assert_allclose(float(obj(obj.x0)), hand_normal(dv, f, 0.3), rtol=1e-8)


def test_varying_parameters_wait_for_laplace(data_file):
    p = make(
        data_file,
        "observe conc = DV\nestimate ka ~ lognormal(median=1.4, sdlog=0.3)\n"
        "estimate CL = 0.05, ke = 0.07\nerror conc ~ additive(0.3)",
        method="laplace",
    )
    with pytest.raises(NotImplementedError, match="ka vary between units"):
        tf.objective(p)


def test_a_parameter_without_a_value(data_file):
    model = PK.replace("param ka=1.6, CL=0.04, ke=0.086", "param ka, CL=0.04, ke=0.086")
    fit = "observe conc = DV\nestimate CL = 0.05, ke = 0.07\nerror conc ~ additive(0.3)"
    p = make(data_file, fit, model=model)
    with pytest.raises(tf.ProblemError, match="the model gives no value to ka"):
        tf.objective(p)


# ----------------------------------------------------------------------------- written
SD = "sqrt(a^2 + (b*conc)^2)"
COMBINED = f"-0.5*((DV - conc)/{SD})^2 - log({SD}) - 0.9189385332046727"  # 0.5 log(2 pi)


def test_a_written_likelihood_equals_the_error_model(data_file):
    written = make(
        data_file,
        f"observe conc = DV\nestimate ka = 1.4, CL = 0.05, ke = 0.07, a = 0.3, b = 0.2\n"
        f"likelihood conc: {COMBINED}",
    )
    model = make(
        data_file,
        "observe conc = DV\nestimate ka = 1.4, CL = 0.05, ke = 0.07\n"
        "error conc ~ combined(0.3, 0.2)",
    )
    ow, om = tf.objective(written), tf.objective(model)
    assert ow.names == ("ka", "CL", "ke", "a", "b")
    np.testing.assert_allclose(float(ow(ow.x0)), float(om(om.x0)), rtol=1e-12)
    g = jax.grad(ow)(jnp.asarray(ow.x0))
    np.testing.assert_allclose(g, jax.grad(om)(jnp.asarray(om.x0)), rtol=1e-8)
    assert f"conc: log-density {COMBINED} (written)" in written.summary()
    assert tf.Problem.from_text(written.to_text(model=True)) == written


def test_a_likelihood_as_a_python_function(data_file):
    p = make(
        data_file,
        "observe conc = DV\nestimate ka = 1.4, CL = 0.05, ke = 0.07, a = 0.3\n"
        "error conc ~ additive(0.3)",
    )
    p._errors.clear()

    def density(dv, f, v):
        return -0.5 * ((dv - f) / v["a"]) ** 2 - jnp.log(v["a"]) - 0.5 * jnp.log(2 * jnp.pi)

    p.likelihood("conc", density)
    obj = tf.objective(p)
    dv, f = predictions(obj.x0[:3])
    np.testing.assert_allclose(float(obj(obj.x0)), hand_normal(dv, f, 0.3), rtol=1e-8)
    assert "# likelihood conc: a Python function" in p.to_text()


def test_written_likelihood_checks(data_file):
    with pytest.raises(tf.ProblemError, match="give an error model or a likelihood, not both"):
        fit = "observe conc = DV\nestimate ka = 1.4, CL = 0.05, ke = 0.07, s = 1\n"
        make(
            data_file, fit + "error conc ~ additive(0.3)\nlikelihood conc: -0.5*(DV - conc)^2/s"
        ).check()
    with pytest.raises(tf.ProblemError, match="likelihood conc: 'sd' is not an estimated"):
        fit = "observe conc = DV\nestimate ka = 1.4, CL = 0.05, ke = 0.07\n"
        make(data_file, fit + "likelihood conc: -0.5*((DV - conc)/sd)^2").check()
