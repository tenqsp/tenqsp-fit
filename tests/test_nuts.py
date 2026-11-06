# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Priors, ``method nuts`` and the prior predictive check.

Oracles: SciPy's densities (each prior family); the log posterior of the linear model of
``test_laplace`` written out by hand (the potential NUTS samples, at any point); the posterior
mode by SciPy on closed forms (written priors with ``optimize`` and ``laplace``); and an
independent NumPyro NUTS run on the closed-form one-compartment model with the same priors
(the posterior ``method nuts`` gives)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st
from scipy.optimize import minimize
from test_laplace import TIMES as LIN_TIMES
from test_laplace import exact_m2ll, lin_problem, lin_table
from test_optimize import conc, problem, table

import tenqsp_fit as tf
from tenqsp_fit.distributions import make_distribution
from tenqsp_fit.priors import draw, lkj_draw, log_density

FAMILIES = [  # (name, arguments, the same distribution in SciPy)
    ("normal", dict(mean=1.0, sd=2.0), st.norm(1, 2)),
    ("normal", dict(mean=2.0, cv=0.5), st.norm(2, 1)),
    ("lognormal", dict(median=2.0, sdlog=0.5), st.lognorm(0.5, scale=2)),
    (
        "lognormal",
        dict(mean=2.0, cv=0.5),
        st.lognorm(np.sqrt(np.log1p(0.25)), scale=2 / np.sqrt(1.25)),
    ),  # fmt: skip
    ("exponential", dict(mean=3.0), st.expon(scale=3)),
    ("logitnormal", dict(median=0.3, sdlogit=0.8), None),
    ("weibull", dict(shape=1.7, scale=2.0), st.weibull_min(1.7, scale=2)),
    ("gamma", dict(shape=3.0, rate=2.0), st.gamma(3, scale=0.5)),
    ("gamma", dict(mean=2.0, cv=0.5), st.gamma(4, scale=0.5)),
    ("beta", dict(alpha=2.0, beta=5.0), st.beta(2, 5)),
    ("beta", dict(mean=0.25, concentration=8.0), st.beta(2, 6)),
    ("uniform", dict(lower=1.0, upper=4.0), st.uniform(1, 3)),
]


def _dist(name, args):
    return make_distribution(name, args, between_units=False)


@pytest.mark.parametrize("name,args,ref", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_prior_densities_and_draws(name, args, ref):
    d = _dist(name, args)
    if ref is None:  # logitnormal: normal on the logit scale, with the change of variable
        mu, s = np.log(0.3 / 0.7), 0.8
        v = np.array([0.1, 0.3, 0.8])
        want = st.norm(mu, s).logpdf(np.log(v / (1 - v))) - np.log(v * (1 - v))
        mean = None
    else:
        v = np.array([0.15, 0.4, 0.9]) if name == "beta" else np.array([1.2, 2.0, 3.5])
        want = ref.logpdf(v)
        mean = ref.mean()
    got = np.array([float(log_density(d, jnp.asarray(x))) for x in v])
    np.testing.assert_allclose(got, want, rtol=1e-10)
    x = draw(d, np.random.default_rng(0), 40_000)
    if mean is not None:
        assert abs(x.mean() - mean) < 4 * ref.std() / np.sqrt(len(x))


def test_lkj_draws():
    """LKJ(2): each correlation's marginal is 2 Beta(a, a) - 1 with a = eta - 1 + d/2."""
    for d in (2, 3):
        R = lkj_draw(np.random.default_rng(1), d, 20_000)
        assert np.all(np.linalg.eigvalsh(R) > 0)
        r = R[:, 0, d - 1]
        a = 2 - 1 + d / 2
        assert abs(r.mean()) < 0.02
        assert abs(r.var() - 1 / (2 * a + 1)) < 0.01


# ----------------------------------------------------------------------------- the rules
def test_default_priors_are_printed(tmp_path):
    p = lin_problem(tmp_path, extra="prior k ~ lognormal(median=0.3, sdlog=0.5)\n")
    p._method = None
    p.method("nuts", chains=2)
    s = p.summary()
    assert "k ~ lognormal(median=0.3, sdlog=0.5)\n" in s + "\n"  # written: no mark
    assert "A.sd ~ lognormal(median=1, sdlog=2) (default)" in s
    assert "A.mean ~ normal(mean=8, sd=16) (default)" in s
    assert "corr(A, B) ~ LKJ(2) (default)" in s
    assert "chains=2 warmup=1000 (default) samples=1000 (default)" in s


def test_a_start_of_0_needs_a_written_prior(tmp_path):
    est = "estimate A ~ normal(mean=8, sd=1)\nestimate B ~ normal(mean=0, sd=1)\nestimate k = 0.2\n"
    p = lin_problem(tmp_path, correlate=False, estimate=est)
    p.check()  # laplace: no default needed
    p._method = None
    p.method("nuts")
    with pytest.raises(tf.ProblemError, match=r"B.mean starts at 0.*prior B.mean ~ normal"):
        p.check()


@pytest.mark.parametrize(
    "line,match",
    [
        ("prior B.mean ~ lognormal(median=1, sdlog=1)", "of any sign, but lognormal"),
        ("prior k ~ uniform(lower=0, upper=1)", r"estimate k = <start> in \(0, 1\)"),
        ("prior corr(A,B) ~ normal(mean=0, sd=0.3)", "LKJ"),
        ("prior k ~ beta(alpha=2, beta=2)", "density only in"),
    ],
)
def test_a_written_prior_must_suit_its_quantity(tmp_path, line, match):
    with pytest.raises(tf.ProblemError, match=match):
        lin_problem(tmp_path, extra=line + "\n").check()


# ----------------------------------------------------------------------------- MAP
def test_written_priors_give_the_posterior_mode_optimize(tmp_path):
    """-2LL plus -2 log prior, minimized by SciPy on the closed form."""
    prior = "prior ke ~ lognormal(median=0.05, sdlog=0.2)\nprior ka ~ normal(mean=1, sd=0.5)\n"
    r = tf.fit(problem(tmp_path, options="starts=1", extra=prior))
    df = table()
    t, dose, y = df.TIME.to_numpy(), df.Dose.to_numpy(), df.DV.to_numpy()

    def objective(lx):
        ka, CL, ke = np.exp(lx)
        m2ll = np.sum(((y - conc(t, dose, ka, CL, ke)) / 0.3) ** 2)
        lp = st.lognorm(0.2, scale=0.05).logpdf(ke) + st.norm(1, 0.5).logpdf(ka)
        return m2ll - 2 * lp

    ref = minimize(objective, np.log([1.0, 0.05, 0.1]), method="Nelder-Mead",
                   options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))  # fmt: skip
    np.testing.assert_allclose([r.estimates[k] for k in ("ka", "CL", "ke")], np.exp(ref.x),
                               rtol=2e-4)  # fmt: skip
    e = r.estimates  # -2LL and the prior terms, reported apart
    lp = st.lognorm(0.2, scale=0.05).logpdf(e["ke"]) + st.norm(1, 0.5).logpdf(e["ka"])
    assert r.with_priors == pytest.approx(r.m2ll - 2 * lp, abs=1e-8)
    assert "posterior mode" in repr(r)
    assert r.optimizer == "BFGS"  # least squares cannot carry a prior


def test_written_priors_give_the_posterior_mode_laplace(tmp_path):
    """The linear model's exact -2LL (Laplace is exact there) plus -2 log prior."""
    r = tf.fit(lin_problem(tmp_path, extra="prior k ~ lognormal(median=0.2, sdlog=0.1)\n"))
    df = lin_table()

    def objective(v):
        mA, lsA, mB, lsB, lk, ls, z = v
        x = [mA, np.exp(lsA), mB, np.exp(lsB), np.exp(lk), np.exp(ls), np.tanh(z)]
        return exact_m2ll(x, df) - 2 * st.lognorm(0.1, scale=0.2).logpdf(x[4])

    v0 = [10, np.log(2), 1, np.log(0.5), np.log(0.25), np.log(0.3), 0.3]
    ref = minimize(objective, v0, method="Nelder-Mead",
                   options=dict(xatol=1e-9, fatol=1e-10, maxiter=40000))  # fmt: skip
    assert r.with_priors == pytest.approx(ref.fun, abs=1e-3)
    assert r.estimates["k"] == pytest.approx(np.exp(ref.x[4]), rel=2e-3)


# ----------------------------------------------------------------------------- the potential
def test_the_potential_is_the_log_posterior(tmp_path):
    """At random points: -log posterior of the linear model written out by hand (data, the
    units' scores, written and default priors, LKJ(2), the scales' Jacobian)."""
    from tenqsp_fit.laplace import Laplace
    from tenqsp_fit.nuts import Posterior

    p = lin_problem(tmp_path, extra="prior A.mean ~ normal(mean=10, sd=3)\n")
    p._method = None
    p.method("nuts")
    lap = Laplace(p)
    post = Posterior(p, lap, lap._cond)
    assert post.names == ("A.mean", "A.sd", "B.mean", "B.sd", "k", "corr(A,B)", "y.additive")
    df = lin_table()
    units = [d for _, d in df.groupby("ID")]
    rng = np.random.default_rng(5)

    def by_hand(u, E):
        mA, sA, mB, sB, k = u[0], np.exp(u[1]), u[2], np.exp(u[3]), np.exp(u[4])
        s5 = 1 / (1 + np.exp(-u[5]))
        rho, sig = -1 + 2 * s5, np.exp(u[6])
        L = np.linalg.cholesky([[1, rho], [rho, 1]])
        Z = E @ L.T
        nll = 0.0
        for i, d in enumerate(units):
            f = (mA + sA * Z[i, 0]) * np.exp(-k * d.TIME.to_numpy()) + mB + sB * Z[i, 1]
            nll -= st.norm(f, sig).logpdf(d.DV.to_numpy()).sum()
        lp = (st.norm(10, 3).logpdf(mA) + st.lognorm(2, scale=1).logpdf(sA)
              + st.norm(0.5, 1).logpdf(mB) + st.lognorm(2, scale=1).logpdf(sB)
              + st.lognorm(2, scale=0.2).logpdf(k) + st.lognorm(2, scale=0.5).logpdf(sig)
              + np.log(1 - rho**2))  # fmt: skip
        logjac = u[1] + u[3] + u[4] + u[6] + np.log(2 * s5 * (1 - s5))
        return nll + 0.5 * np.sum(E**2) - lp - logjac

    vals = []
    for _ in range(4):
        u = np.array([9, np.log(1.8), 0.8, np.log(0.6), np.log(0.28), 0.4, np.log(0.35)])
        u = u + 0.2 * rng.normal(size=7)
        E = rng.normal(size=(8, 2))
        flat = jnp.asarray(np.concatenate([u, E.ravel()]))
        vals.append((float(post.potential(flat)), by_hand(u, E)))
    a, b = np.array(vals).T  # equal, to the solver's tolerance (both keep every constant)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


# ----------------------------------------------------------------------------- sampling
PRIORS_PK = (
    "prior ka ~ lognormal(median=1.5, sdlog=0.5)\n"
    "prior CL ~ lognormal(median=0.04, sdlog=0.5)\n"
    "prior ke ~ lognormal(median=0.08, sdlog=0.5)\n"
)


@pytest.fixture(scope="module")
def sampled(tmp_path_factory):
    p = problem(tmp_path_factory.mktemp("nuts"), options="seed=2", extra=PRIORS_PK)
    p._options.update(rtol=1e-8, atol=1e-10)  # the closed form's accuracy is not needed here
    p._method = None
    p.method("nuts", chains=4, warmup=200, samples=250)
    return tf.fit(p)


@pytest.fixture(scope="module")
def reference():
    return _reference()


def _reference():
    """NumPyro's NUTS on the closed form with the same priors (no TenQSP, no tenqsp-fit)."""
    import numpyro
    import numpyro.distributions as nd
    from numpyro.infer import MCMC, NUTS

    df = table()
    t, dose, y = (jnp.asarray(df[c].to_numpy()) for c in ("TIME", "Dose", "DV"))

    def model():
        ka = numpyro.sample("ka", nd.LogNormal(np.log(1.5), 0.5))
        CL = numpyro.sample("CL", nd.LogNormal(np.log(0.04), 0.5))
        ke = numpyro.sample("ke", nd.LogNormal(np.log(0.08), 0.5))
        f = dose * ka * ke / (CL * (ka - ke)) * (jnp.exp(-ke * t) - jnp.exp(-ka * t))
        numpyro.sample("y", nd.Normal(f, 0.3), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=4000, num_chains=1,
                progress_bar=False)  # fmt: skip
    mcmc.run(jax.random.PRNGKey(0))
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_nuts_draws_the_posterior(sampled, reference):
    t = sampled.table().set_index("quantity")
    for k in ("ka", "CL", "ke"):
        ref = reference[k]
        sd = ref.std()
        # Monte Carlo error with ~700 effective draws: about 0.04 SD for a median, 0.1 SD
        # for a 2.5 % quantile (both runs)
        assert abs(t.loc[k, "median"] - np.median(ref)) < 0.2 * sd, k
        assert abs(t.loc[k, "SD"] / sd - 1) < 0.1, k
        for q, col in ((2.5, "CI 2.5 %"), (97.5, "CI 97.5 %")):
            assert abs(t.loc[k, col] - np.percentile(ref, q)) < 0.4 * sd, (k, q)
    assert (t["R-hat"] < 1.02).all()
    assert int(sampled.convergence().divergences.sum()) == 0


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_the_nuts_result(sampled, tmp_path):
    r = sampled
    assert r.method == "nuts" and r.status == "sampled"
    assert list(r.draws.columns) == ["chain", "draw", "ka", "CL", "ke", "divergent"]
    assert len(r.draws) == 4 * 250
    np.testing.assert_allclose(list(r.estimates.values()),
                               r.draws[["ka", "CL", "ke"]].median().to_numpy())  # fmt: skip
    conv = r.convergence()
    assert list(conv.chain) == [1, 2, 3, 4] and (conv.draws == 250).all()
    np.testing.assert_allclose(r.corr.to_numpy(), r.draws[["ka", "CL", "ke"]].corr().to_numpy())
    text = repr(r)
    assert "4 chains x 250 draws (after 200 warmup)" in text and "prior" in text
    fig = r.plot_trace()
    assert len([a for a in fig.axes if a.get_visible()]) == 4
    with pytest.raises(ValueError, match="profile is for likelihood fits"):
        r.profile("ka")
    out = r.save(tmp_path / "run")
    assert (out / "draws.csv").exists() and (out / "convergence.csv").exists()
    # the VPC takes each simulated study from a different draw: wider than at one value
    v = r.vpc(n=100, seed=1).table
    assert np.isfinite(v["sim p50"]).all()


def test_fit_method_nuts_overrides(tmp_path):
    from tenqsp_fit.fit import _with_method

    p = problem(tmp_path, options="starts=1", extra=PRIORS_PK)
    q = _with_method(p, "nuts")  # what tf.fit(p, method="nuts") fits
    assert q._method.name == "nuts" and q._method.options == {}
    assert p._method.name == "optimize"  # the problem itself is unchanged


# ----------------------------------------------------------------------------- prior predictive
def test_prior_predictive(tmp_path):
    p = lin_problem(tmp_path)
    pp = tf.prior_predictive(p, n=300, seed=4)
    d = pp.draws
    assert list(d.columns) == ["A.mean", "A.sd", "B.mean", "B.sd", "k", "corr(A,B)",
                               "y.additive", "ran"]  # fmt: skip
    assert d.ran.all() and pp.failed == 0
    # the defaults: k ~ lognormal(median=0.2, sdlog=2); A.mean ~ normal(8, 16)
    assert abs(np.median(np.log(d.k)) - np.log(0.2)) < 4 * 2 / np.sqrt(300)
    assert abs(d["A.mean"].mean() - 8) < 4 * 16 / np.sqrt(300)
    assert d["corr(A,B)"].between(-1, 1).all()
    assert len(pp.table) == len(LIN_TIMES)
    assert "prior predictive check" in pp.figure._suptitle.get_text()
    again = tf.prior_predictive(p, n=300, seed=4)
    np.testing.assert_allclose(again.table["sim p50"], pp.table["sim p50"])
    assert "prior predictive check" in repr(pp)


def test_prior_predictive_needs_priors_it_can_set(tmp_path):
    est = "estimate A ~ normal(mean=8, sd=1)\nestimate B ~ normal(mean=0, sd=1)\nestimate k = 0.2\n"
    with pytest.raises(tf.ProblemError, match="B.mean starts at 0"):
        tf.prior_predictive(lin_problem(tmp_path, correlate=False, estimate=est), n=10)


def test_a_study_that_does_not_run_is_left_out(tmp_path):
    """One set of values the model cannot run with (NaN): left out of the bands, counted."""
    from tenqsp_fit.laplace import Laplace
    from tenqsp_fit.results import simulate_vpc

    lap = Laplace(lin_problem(tmp_path))
    xs = np.tile(np.asarray(lap.x0, float), (20, 1))
    xs[3, 4] = np.nan  # k
    v = simulate_vpc("lin", lap, xs, np.random.default_rng(0))
    assert v.failed == 1 and not v.ran[3] and v.ran.sum() == 19
    assert np.isfinite(v.table["sim p50"]).all()
    assert "1 of 20 studies did not run" in v.figure._suptitle.get_text()


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_the_doctor_and_passport_after_nuts(sampled):
    d = sampled.doctor()
    found = {f.check: f for f in d.findings}
    assert {"convergence", "prior and data", "tolerance check", "gradient check"} <= set(found)
    assert found["tolerance check"].severity == "note"  # rtol 1e-8 is enough here
    assert found["gradient check"].severity == "note"
    # the priors (sdlog 0.5) are much wider than what 24 measurements allow: the posterior SD
    # of each quantity, on the log scale, is a small part of the prior's (0.5)
    ratio = found["prior and data"].evidence
    sd = np.log(sampled.draws[["ka", "CL", "ke"]].to_numpy()).std(axis=0, ddof=1)
    np.testing.assert_allclose([ratio[n] for n in ("ka", "CL", "ke")], sd / 0.5, rtol=0.05)
    assert max(ratio.values()) < 0.2
    q = sampled.passport().to_dict()["quantities"]
    assert [r["quantity"] for r in q] == ["ka", "CL", "ke"]
    assert all(r["determined"] == "yes" and "lognormal" in r["prior"] for r in q)
    assert "doctor:" in repr(sampled)


@pytest.mark.slow
@pytest.mark.timeout(1200)  # the first to ask for the samples runs them
def test_to_arviz(sampled):
    """The draws, the sampler's statistics and the data, by chain and draw."""
    dt = sampled.to_arviz()
    post = dt["posterior"]
    assert dict(post.sizes) == {"chain": 4, "draw": 250}
    d = sampled.draws
    for name in sampled.names:
        np.testing.assert_array_equal(post[name].values.ravel(), d[name].to_numpy())
    ss = dt["sample_stats"]
    assert int(ss["diverging"].values.sum()) == int(sampled.per_chain.divergences.sum())
    assert {"energy", "lp", "n_steps", "step_size"} <= set(ss.data_vars)
    np.testing.assert_allclose(dt["observed_data"]["conc"].values, table().DV.to_numpy())


@pytest.mark.slow
@pytest.mark.timeout(1200)  # the first to ask for the samples runs them
def test_a_prediction_after_nuts_is_a_posterior_interval(sampled):
    import tenqsp as tq

    from tenqsp_fit import predict as P

    t = np.linspace(0.5, 24, 12)
    pr = sampled.predict(regimen=tq.bolus("depot", 5.0, times=[0]), times=t, n=1000)
    assert pr.kind == P.POSTERIOR
    # the oracle: the closed form at each posterior draw, its percentiles
    d = sampled.draws[["ka", "CL", "ke"]].to_numpy()
    curves = np.array([conc(t, 5.0, *v) for v in d])
    for col, q in (("median", 50), ("2.5 %", 2.5), ("97.5 %", 97.5)):
        np.testing.assert_allclose(pr.table[col], np.percentile(curves, q, axis=0), rtol=0.03)
    fig = sampled.plot_diagnostics()
    assert "nuts" in fig.subfigs[2]._suptitle.get_text()


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_to_arviz_with_units(tmp_path):
    """Each unit's normal score per draw; their medians are the result's modes."""
    p = lin_problem(tmp_path, correlate=False, options="seed=1 checks=quick")
    p._method = None
    p.method("nuts", chains=2, warmup=150, samples=100)
    r = tf.fit(p)
    dt = r.to_arviz()
    z = dt["posterior"]["z(A)"]
    assert z.dims == ("chain", "draw", "unit") and list(z.unit.values) == list(r.objective.units)
    np.testing.assert_allclose(np.median(z.values.reshape(-1, z.shape[2]), axis=0), r.modes[:, 0])
