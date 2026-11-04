# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``method laplace``. Oracle: a model linear in its varying parameters (y = A e(t) + B, with
A and B normal between units) with an additive error, where the Laplace approximation is
exact: each unit's measurements are multivariate normal, and -2LL is their log-density."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm, weibull_min

import tenqsp_fit as tf
from tenqsp_fit.distributions import quantile
from tenqsp_fit.optimize import scales, to_u, to_x

LIN = """model lin
param A=10, B=1, k=0.3
d_t(e) = -k*e
e(0) = 1
observable y = A*e + B
"""
TIMES = (0.5, 1, 2, 4, 6)
TRUE = dict(mA=10.0, sA=2.0, mB=1.0, sB=0.5, rho=0.4, k=0.3, sigma=0.3)


def lin_table(n=8, seed=3):
    rng = np.random.default_rng(seed)
    t = np.array(TIMES)
    C = np.array([[1, TRUE["rho"]], [TRUE["rho"], 1]])
    rows = []
    for i in range(1, n + 1):
        za, zb = rng.multivariate_normal([0, 0], C)
        a, b = TRUE["mA"] + TRUE["sA"] * za, TRUE["mB"] + TRUE["sB"] * zb
        y = a * np.exp(-TRUE["k"] * t) + b + TRUE["sigma"] * rng.normal(size=t.size)
        rows += [dict(ID=i, TIME=float(tt), DV=float(v), WT=60 + 5 * i)
                 for tt, v in zip(t, y, strict=True)]  # fmt: skip
    return pd.DataFrame(rows)


def lin_problem(tmp_path, *, correlate=True, extra="", options="", estimate=None):
    f = tmp_path / "lin.csv"
    lin_table().to_csv(f, index=False)
    est = estimate or (
        "estimate A ~ normal(mean=8, sd=1)\nestimate B ~ normal(mean=0.5, sd=1)\nestimate k = 0.2\n"
    )
    corr = "correlate A, B\n" if correlate else ""
    return tf.Problem.from_text(
        LIN + f'fit lin\ndata "{f}"\nunit ID\nobserve y = DV\n{est}{corr}{extra}'
        f"error y ~ additive(0.5)\nmethod laplace\n"
        f"options rtol=1e-10 atol=1e-12 progress=none {options}\n"
    )


def exact_m2ll(v, df):
    """-2LL of the linear model: each unit's y is normal, mean mA e + mB, covariance
    sA^2 e e' + sB^2 1 1' + rho sA sB (e 1' + 1 e') + sigma^2 I."""
    mA, sA, mB, sB, k, sigma, rho = v
    total = 0.0
    for _, d in df.groupby("ID"):
        e = np.exp(-k * d.TIME.to_numpy())
        one = np.ones_like(e)
        cov = (sA**2 * np.outer(e, e) + sB**2 * np.outer(one, one)
               + rho * sA * sB * (np.outer(e, one) + np.outer(one, e))
               + sigma**2 * np.eye(len(e)))  # fmt: skip
        total += -2 * multivariate_normal(mA * e + mB, cov).logpdf(d.DV.to_numpy())
    return total


def as_oracle(obj, x):
    d = obj.unpack(x)
    return [d["A.mean"], d["A.sd"], d["B.mean"], d["B.sd"], d["k"], d["y.additive"],
            d.get("corr(A,B)", 0.0)]  # fmt: skip


def test_laplace_is_exact_for_a_linear_model(tmp_path):
    p = lin_problem(tmp_path)
    obj = tf.objective(p)
    assert isinstance(obj, tf.Laplace)
    assert obj.names == ("A.mean", "A.sd", "B.mean", "B.sd", "k", "corr(A,B)", "y.additive")
    df = lin_table()
    for x in (obj.x0, np.array([10.5, 1.8, 1.2, 0.6, 0.31, 0.3, 0.35])):
        assert float(obj(x)) == pytest.approx(exact_m2ll(as_oracle(obj, x), df), rel=1e-9)
    assert "Laplace objective" in repr(obj) and "A, B varying" in repr(obj)


@pytest.mark.parametrize(
    "dist, x",
    [
        ("lognormal(median=8, sdlog=0.2)", [9.0, 0.25, 0.8, 0.6, 0.28, 0.2, 0.4]),
        ("weibull(shape=5, scale=10)", [6.0, 9.0, 0.8, 0.6, 0.28, 0.2, 0.4]),  # a shape
    ],
)
def test_gradient_by_the_implicit_function_theorem(tmp_path, dist, x):
    """Against central differences of -2LL (each with its own inner fit)."""
    est = f"estimate A ~ {dist}\nestimate B ~ normal(mean=0.5, sd=1)\nestimate k = 0.2\n"
    p = lin_problem(tmp_path, estimate=est)
    obj = tf.objective(p)
    sc = scales(obj.names, obj.x0, obj.bounds)
    u = to_u(np.array(x), sc)
    Z = obj.modes(to_x(jnp.asarray(u), sc))
    g = np.asarray(obj._gradient(jnp.asarray(u), jnp.asarray(Z), lambda w: to_x(w, sc)))
    h, E = 1e-5, np.eye(len(u))

    def f(w):
        return float(obj._jit_value(to_x(jnp.asarray(w), sc), jnp.asarray(Z))[0])

    fd = np.array([(f(u + h * E[i]) - f(u - h * E[i])) / (2 * h) for i in range(len(u))])
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-5)


def test_fit_reaches_the_maximum_likelihood(tmp_path):
    """The fit against the exact likelihood maximized by scipy."""
    p = lin_problem(tmp_path)
    r = tf.fit(p)
    df = lin_table()

    def ex(w):  # mA, log sA, mB, log sB, log k, log sigma, atanh rho
        v = [w[0], np.exp(w[1]), w[2], np.exp(w[3]), np.exp(w[4]), np.exp(w[5]), np.tanh(w[6])]
        return exact_m2ll(v, df)

    ref = minimize(ex, [8, 0, 0.5, 0, np.log(0.2), np.log(0.5), 0], method="BFGS",
                   options={"gtol": 1e-7})  # fmt: skip
    assert r.method == "laplace" and r.status == "converged"
    assert r.m2ll == pytest.approx(ref.fun, abs=1e-4)
    w = ref.x
    expect = {
        "A.mean": w[0], "A.sd": np.exp(w[1]), "B.mean": w[2], "B.sd": np.exp(w[3]),
        "k": np.exp(w[4]), "y.additive": np.exp(w[5]), "corr(A,B)": np.tanh(w[6]),
    }  # fmt: skip
    for k, v in expect.items():
        assert r.estimates[k] == pytest.approx(v, rel=2e-3, abs=2e-3), k
    # standard errors against the Hessian of the exact -2LL (central differences)
    t = r.table()
    assert list(t.columns) == ["quantity", "estimate", "SE", "RSE %", "CI 2.5 %", "CI 97.5 %",
                               "also"]  # fmt: skip
    h, E = 1e-3, np.eye(7) * 1e-3

    def d2(i, j):
        a, b = E[i], E[j]
        return (ex(w + a + b) - ex(w + a - b) - ex(w - a + b) + ex(w - a - b)) / (4 * h * h)

    H = np.array([[d2(i, j) for j in range(7)] for i in range(7)])
    se_w = np.sqrt(np.diag(2 * np.linalg.inv(H)))
    for name, i in (("k", 4), ("y.additive", 5), ("A.sd", 1)):  # log scale: SE x value
        se = float(t.loc[t.quantity == name, "SE"].iloc[0])
        assert se == pytest.approx(se_w[i] * r.estimates[name], rel=0.02), name
    se_mean = float(t.loc[t.quantity == "A.mean", "SE"].iloc[0])
    assert se_mean == pytest.approx(se_w[0], rel=0.02)
    assert np.all(np.isfinite(t["SE"])) and r.covariance_status == "ok"
    lo, hi = t["CI 2.5 %"].to_numpy(), t["CI 97.5 %"].to_numpy()
    assert np.all(lo < t["estimate"]) and np.all(t["estimate"] < hi)
    assert t.loc[t.quantity == "A.sd", "also"].iloc[0] == "var " + f"{r.estimates['A.sd'] ** 2:.4g}"
    ind = r.individual()
    assert list(ind.columns) == ["unit", "z(A)", "z(B)", "A", "B", "k"]
    assert len(ind) == 8 and np.allclose(ind.k, r.estimates["k"])
    assert set(r.shrinkage) == {"A", "B"}
    text = repr(r)
    assert "method laplace (BFGS, FOCE-I), converged" in text and "8 units" in text
    assert "shrinkage (1 - SD of the units' scores): A" in text


def test_modes_are_the_conditional_means(tmp_path):
    """For the linear model the conditional mode is the conditional mean (BLUP)."""
    p = lin_problem(tmp_path, correlate=False)
    obj = tf.objective(p)
    x = np.array([10.0, 2.0, 1.0, 0.5, 0.3, 0.3])
    Z = obj.modes(x)
    d = lin_table()[lambda df: df.ID == 1]
    e = np.exp(-0.3 * d.TIME.to_numpy())
    G = np.stack([2.0 * e, 0.5 * np.ones_like(e)], axis=1)  # dy/dz
    resid = d.DV.to_numpy() - (10 * e + 1)
    blup = np.linalg.solve(G.T @ G / 0.09 + np.eye(2), G.T @ resid / 0.09)
    np.testing.assert_allclose(Z[0], blup, rtol=1e-8)


def test_covariate_in_a_distribution(tmp_path):
    est = ("estimate A_ref = 8\nestimate A ~ lognormal(median=A_ref*(WT/70)^0.75, sdlog=0.2)\n"
           "estimate B ~ normal(mean=0.5, sd=1)\nestimate k = 0.2\n")  # fmt: skip
    p = lin_problem(tmp_path, correlate=False, estimate=est)
    obj = tf.objective(p)
    x = np.array(obj.x0)
    ind = obj.individual(x, np.zeros((8, 2)))
    wt = 60 + 5 * np.arange(1, 9)
    np.testing.assert_allclose(ind.A, 8 * (wt / 70) ** 0.75, rtol=1e-12)


def test_correlation_must_be_a_correlation_matrix(tmp_path):
    f = tmp_path / "lin.csv"
    lin_table().to_csv(f, index=False)
    p = tf.Problem.from_text(
        LIN.replace("observable y = A*e + B", "param C=0\nobservable y = A*e + B + C")
        + f'fit lin\ndata "{f}"\nunit ID\nobserve y = DV\n'
        "estimate A ~ normal(mean=8, sd=1)\nestimate B ~ normal(mean=0.5, sd=1)\n"
        "estimate C ~ normal(mean=fix(0), sd=0.1)\nestimate k = 0.2\n"
        "correlate A, B, C\nerror y ~ additive(0.5)\nmethod laplace\n"
    )
    obj = tf.objective(p)
    i = {n: j for j, n in enumerate(obj.names)}
    x = np.array(obj.x0)
    assert np.isfinite(float(obj(x)))
    x[i["corr(A,B)"]], x[i["corr(A,C)"]], x[i["corr(B,C)"]] = 0.9, 0.9, -0.9  # not PD
    assert np.isnan(float(obj(x)))


def test_censored_values_with_laplace(tmp_path):
    """M3 with a varying parameter: one unit's measurement is censored; -2LL against
    integration over the scores (Laplace is not exact with a censored value, but close)."""
    df = lin_table(n=3)
    df["CENS"] = 0
    df.loc[4, ["DV", "CENS"]] = [2.0, 1]  # unit 1, t = 6: below 2.0
    f = tmp_path / "lin.csv"
    df.to_csv(f, index=False)
    p = tf.Problem.from_text(
        LIN + f'fit lin\ndata "{f}"\nunit ID\nobserve y = DV\n'
        "estimate A ~ normal(mean=10, sd=2)\nestimate B = 1, k = 0.3\n"
        "censor y where CENS == 1 below 2\nerror y ~ additive(0.3)\nmethod laplace\n"
        "options rtol=1e-10 atol=1e-12\n"
    )
    obj = tf.objective(p)
    assert obj.n_censored == 1
    x = np.array(obj.x0)
    zg = np.linspace(-10, 10, 40001)  # a fine grid: each unit's posterior is narrow
    total = 0.0
    for _, d in df.groupby("ID"):
        e = np.exp(-0.3 * d.TIME.to_numpy())
        f_ = (10 + 2 * zg[:, None]) * e + 1
        meas = d.CENS.to_numpy() == 0
        ll = norm.logpdf(d.DV.to_numpy()[meas], f_[:, meas], 0.3).sum(axis=1)
        ll = ll + norm.logcdf((2.0 - f_[:, ~meas]) / 0.3).sum(axis=1) + norm.logpdf(zg)
        top = ll.max()
        total += -2 * (top + np.log(np.trapezoid(np.exp(ll - top), zg)))
    assert float(obj(x)) == pytest.approx(total, abs=0.02)


@pytest.mark.parametrize(
    "name, args, ref",
    [
        ("lognormal", {"median": 2.0, "sdlog": 0.5}, lambda q: 2 * np.exp(0.5 * norm.ppf(q))),
        ("lognormal", {"mean": 2.0, "cv": 0.5},
         lambda q: 2 / np.sqrt(1.25) * np.exp(np.sqrt(np.log(1.25)) * norm.ppf(q))),
        ("normal", {"mean": 1.0, "cv": 0.2}, lambda q: 1 + 0.2 * norm.ppf(q)),
        ("exponential", {"rate": 4.0}, lambda q: -np.log1p(-q) / 4),
        ("logitnormal", {"median": 0.3, "sdlogit": 0.7},
         lambda q: 1 / (1 + np.exp(-(np.log(0.3 / 0.7) + 0.7 * norm.ppf(q))))),
        ("weibull", {"shape": 1.7, "scale": 3.0}, lambda q: weibull_min(1.7, scale=3).ppf(q)),
    ],
)  # fmt: skip
def test_quantiles(name, args, ref):
    z = np.linspace(-3, 3, 13)
    jax.config.update("jax_enable_x64", True)
    got = np.asarray(quantile(name, {k: jnp.asarray(v) for k, v in args.items()}, jnp.asarray(z)))
    np.testing.assert_allclose(got, ref(norm.cdf(z)), rtol=1e-9)


def test_method_and_problem_must_agree(tmp_path):
    p = lin_problem(tmp_path)
    with pytest.raises(ValueError, match="use method laplace"):
        tf.fit(p, method="optimize")


def test_a_time_limit_stops_laplace_and_keeps_the_work(tmp_path):
    from tenqsp_fit import optimize as O

    r = tf.fit(lin_problem(tmp_path, options="timeout=0.001"))
    assert r.status == "stopped" and np.isfinite(r.m2ll)
    assert r.stopped_by.startswith("the time limit")
    assert np.isnan(r.se).all() and r.covariance_status == O.STOPPED_SE


def test_ctrl_c_keeps_the_work_with_laplace(tmp_path, monkeypatch):
    from tenqsp_fit import optimize as O

    def update(self, steps, *a, **k):
        if steps.max() >= 2 and not k.get("last"):
            raise KeyboardInterrupt

    monkeypatch.setattr(O._Progress, "update", update)
    r = tf.fit(lin_problem(tmp_path))
    assert r.status == "stopped"
    assert len(r.trace) >= 2 and np.isfinite(r.m2ll)
    assert np.isnan(r.se).all() and r.covariance_status == O.STOPPED_SE
    assert r.individual().shape[0] == 8  # the units' values at the point reached


def test_a_coefficient_that_does_not_vary_takes_any_sign(tmp_path):
    """beta ~ normal(mean=0.5, sd=0): one value, estimated as it is (not on the log scale,
    though its start is above 0), with the covariate formula written first. It reaches the
    fit started from 0, whatever the sign of the answer."""
    head = "A = A_ref*(WT/70)^beta\nestimate A_ref ~ normal(mean=8, sd=1)\n"
    tail = "estimate B ~ normal(mean=0.5, sd=1)\nestimate k = 0.2\n"
    one = "estimate beta ~ normal(mean=0.5, sd=0)\n"
    r1 = tf.fit(lin_problem(tmp_path, correlate=False, estimate=head + one + tail))
    zero = "estimate beta = 0\n"
    r0 = tf.fit(lin_problem(tmp_path, correlate=False, estimate=head + zero + tail))
    assert r1.status == r0.status == "converged"
    i = list(r1.names).index("beta")
    assert r1.scales[i] == ("as is",) and "beta" in r1.table().quantity.to_list()
    assert abs(r1.m2ll - r0.m2ll) < 1e-4
    assert abs(r1.estimates["beta"] - r0.estimates["beta"]) < 1e-3
    ind = r1.individual()
    assert {"A_ref", "A"} <= set(ind.columns)
