# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The doctor and the passport. Oracles: a model where only a*b^2 is determined (the ridge's
direction known exactly), a well-determined control (the oral PK model: no ridge, no weak
quantity), the rules on hand-made results, TenQSP's own reports."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_optimize import PK, problem, table

import tenqsp_fit as tf
from tenqsp_fit import doctor as Dr

PROD = """model prod
param a=1, b=2, k=0.3
d_t(e) = -k*e
e(0) = 1
observable y = a*b^2*e
"""


def prod_problem(tmp_path):
    t = np.array([0.5, 1, 2, 4, 6, 8])
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"TIME": t, "DV": 4 * np.exp(-0.3 * t) + 0.05 * rng.normal(size=t.size)})
    f = tmp_path / "prod.csv"
    df.to_csv(f, index=False)
    return tf.Problem.from_text(
        PROD + f'fit prod\ndata "{f}"\nobserve y = DV\nestimate a = 1.5, b = 1.5, k = 0.2\n'
        "error y ~ additive(fix(0.05))\nmethod optimize\n"
        "options rtol=1e-10 atol=1e-12 progress=none checks=quick\n"
    )


@pytest.fixture(scope="module")
def pk_fit(tmp_path_factory):
    """The well-determined control, every check run."""
    return tf.fit(problem(tmp_path_factory.mktemp("doctor")))


def checks(result, severity=None):
    return {f.check for f in result.doctor().findings if severity in (None, f.severity)}


def test_a_ridge_is_named_with_its_determined_product(tmp_path):
    r = tf.fit(prod_problem(tmp_path))
    ridge = [f for f in r.doctor().findings if f.check == "ridge"]
    assert len(ridge) == 1 and ridge[0].severity == "problem"
    # only a*b^2 is determined: the ridge runs along d(log a) = -2 d(log b)
    v = np.array([ridge[0].evidence["directions"][0][n] for n in ("a", "b", "k")])
    exact = np.array([2.0, -1.0, 0.0]) / np.sqrt(5.0)
    assert abs(v @ exact) > 0.999
    assert "only a^0.5 x b is" in ridge[0].message  # (a b^2)^0.5
    assert ridge[0].evidence["quantities"] == ["a", "b"]
    assert "doctor: " in repr(r) and "ridge" in repr(r)
    # the passport marks them, and k (determined) not
    q = r.passport().to_dict()["quantities"]
    det = {row["quantity"]: row["determined"] for row in q}
    assert det == {"a": "not (ridge)", "b": "not (ridge)", "k": "yes"}


def test_the_control_is_well_determined_and_passes_the_checks(pk_fit):
    d = pk_fit.doctor()
    assert not d.count("problem") and not d.count("warning"), d
    assert {"determination", "optima", "tolerance check", "gradient check"} <= checks(pk_fit)
    tol = next(f for f in d.findings if f.check == "tolerance check")
    assert tol.evidence["change of -2LL"] < 1e-6 and tol.evidence["shift in SE"] < 1e-3
    grad = next(f for f in d.findings if f.check == "gradient check")
    assert max(grad.evidence["relative differences"]) < 1e-3
    assert "all 3 starts reached the same fit" in repr(d)


@pytest.mark.timeout(600)
def test_a_loose_tolerance_is_a_problem(tmp_path):
    p = problem(tmp_path, options="starts=1")
    p._options.update(rtol=1e-2, atol=1e-4)
    r = tf.fit(p)
    tol = next(f for f in r.doctor().findings if f.check == "tolerance check")
    assert tol.severity == "problem", tol
    assert "options rtol=0.0001" in tol.remedy


def test_quick_leaves_out_the_costly_checks(tmp_path):
    p = prod_problem(tmp_path)
    assert "checks=quick" in p.to_text()
    r = tf.fit(p)
    assert "tolerance check" not in checks(r) and "gradient check" not in checks(r)
    assert "tolerance and gradient checks" in checks(r, "note")
    with pytest.raises(ValueError, match="checks is 'full' or 'quick'"):
        p.options(checks="fast")


def test_weakness_rules():
    log, logit, plain = ("log",), ("logit", -1.0, 1.0), ("as is",)
    se = np.log(10) / (2 * 1.96)  # a factor 10 exactly
    assert Dr._weakness("k", log, 1, 10, 0.99 * se, 0) is None
    assert "spans a factor" in Dr._weakness("k", log, 1, 10, 1.01 * se, 0)
    assert Dr._weakness("r", logit, -0.4, 0.5, 0, 0) is None  # 45 % of (-1, 1)
    assert "65 % of its range" in Dr._weakness("r", logit, -0.6, 0.7, 0, 0)
    assert Dr._weakness("m", plain, 0, 1, 0, 49) is None
    assert "95 % interval" in Dr._weakness("m", plain, -0.4, 1.4, 0, 101)


def test_combinations_read_as_products():
    sc = [("log",)] * 3
    names = ["k", "beta", "tau"]
    text, who = Dr._combination(np.array([0.7, -0.7, 0.01]), names, sc)
    assert who == ["k", "beta"] and "only k x beta is" in text
    text, _ = Dr._combination(np.array([0.7, 0.7, 0.0]), names, sc)
    assert "only k / beta is" in text
    text, who = Dr._combination(np.array([0.0, 0.0, 1.0]), names, sc)
    assert who == ["tau"] and "tau is not determined" in text
    text, who = Dr._combination(np.array([0.55, -0.82, -0.31]), names, sc)
    assert len(who) == 3 and "move together as" in text


def test_optima_bounds_spreads_shrinkage():
    def starts(*v):
        return SimpleNamespace(starts=pd.DataFrame({"-2LL": v}))

    assert Dr._optima(starts(1.0))[0].severity == "note"
    assert "same fit" in Dr._optima(starts(1.0, 1.05, 1.08))[0].message
    f = Dr._optima(starts(1.0, 1.05, 3.0, 3.02, np.nan))[0]
    assert f.severity == "warning" and f.evidence["starts"] == [2, 2] and "1 failed" in f.message
    lap = SimpleNamespace(objective=SimpleNamespace(varying=[("CL", None), ("ka", None)]),
                          estimates={"CL.sdlog": 0.004, "ka.cv": 0.3})  # fmt: skip
    (s,) = Dr._spreads(lap)
    assert "CL.sdlog is 0.004" in s.message and "sdlog=0" in s.remedy
    (sh,) = Dr._shrinkage(SimpleNamespace(shrinkage={"ka": 0.45, "CL": 0.05}))
    assert "ka 45 %" in sh.message and "CL" not in sh.message


def test_jumps_in_the_model(tmp_path):
    model = PK.replace("observable", "lag depot = tlag\nextinction centr below 1e-6\n"
                       "observable").replace("ke=0.086", "ke=0.086, tlag=0.5")  # fmt: skip
    model = model.replace("ka*depot - ke*centr", "ka*depot - ke*floor(centr)")
    f = tmp_path / "pk.csv"
    table().to_csv(f, index=False)
    p = tf.Problem.from_text(
        model + f'fit pk\ndata "{f}"\nunit ID\ndoses Dose at t=0 into depot\nobserve conc = DV\n'
        "estimate ka = 1, CL = 0.05, ke = 0.1\nerror conc ~ additive(fix(0.3))\nmethod optimize\n"
    )
    found = Dr._jumps(SimpleNamespace(problem=p))
    text = " | ".join(f.message for f in found)
    assert {f.check for f in found} == {"jumps in the model"}
    assert "extinction group" in text and "lag time tlag" in text and "floor jumps" in text


def test_passport(pk_fit, tmp_path):
    prof = pk_fit.profile("ka", points=4)
    assert np.isfinite(prof.interval).all()
    pp = pk_fit.passport()
    titles = [t for t, _ in pp.sections]
    assert titles == ["The run", "Each quantity", "Ridges and determined combinations",
                      "The doctor's findings", "Numerical checks", "Named predictions",
                      "TenQSP's reports"]  # fmt: skip
    q = pp.to_dict()["quantities"]
    # ka's set has a second piece at ke's value: in oral one-compartment PK, swapping ka and
    # ke gives the same curve
    assert [r["profile"] for r in q] == ["2 pieces, closed", "", ""]
    assert all(r["determined"] == "yes" for r in q)
    d = pp.to_dict()
    json.dumps(d, default=str)  # plain data
    assert d["tenqsp"]["ledger closure"] < 1e-8
    assert d["run"]["hashes"]["problem.tq"] and d["run"]["versions"]["tenqsp"]
    text = repr(pp)
    assert "validate(): 2 issue(s)" in text and "no initial condition given" in text
    assert "Numerical safeguards" in text
    md = pp.markdown()
    assert md.startswith("# Passport: pk") and "quantity | estimate | SE" in md
    folder = pk_fit.save(tmp_path / "run")
    assert (folder / "passport.md").read_text().startswith("# Passport: pk")
    assert json.loads((folder / "passport.json").read_text())["run"]["problem"] == "pk"


def test_flat_directions_sharing_quantities_are_one_finding():
    a = (0.0, None, ["k", "beta"])
    b = (0.0, None, ["beta", "k"])
    c = (0.0, None, ["s"])
    groups = Dr._groups([a, c, b])
    assert sorted(len(g) for g in groups) == [1, 2]
    d = (0.0, None, ["s", "tau"])
    assert len(Dr._groups([a, c, b, d])) == 2  # s joins tau; k, beta apart


def test_optima_are_measured_from_each_best():
    f = Dr._optima(SimpleNamespace(starts=pd.DataFrame({"-2LL": [1.0, 1.08, 1.15]})))[0]
    assert f.check == "several optima" and f.evidence["starts"] == [2, 1]
    assert "result.profile(name)" in f.remedy  # 0.15 apart: well within the cut-off 3.84


def test_profile_advised_only_when_another_optimum_is_close():
    def advice(m2ll):
        return Dr._optima(SimpleNamespace(starts=pd.DataFrame({"-2LL": m2ll})))[0].remedy

    assert "the next fit is within 3.84 of the best" in advice([-34.5, -32.0, -34.5])
    assert "profile" not in advice([-34.5, -24.8, -34.5])  # the QSP quickstart's two fits


def test_profile_shapes():
    from tenqsp_fit.passport import _shape

    def prof(rises, interval):
        t = pd.DataFrame({"value": np.arange(len(rises), dtype=float), "-2LL": rises})
        return SimpleNamespace(table=t, m2ll=0.0, interval=interval)

    nan = float("nan")
    assert _shape(prof([5.0, 0.0, 5.0], (0.5, 1.5))) == "closed"
    assert _shape(prof([0.2, 0.0, 5.0], (nan, 1.5))) == "flat below"
    assert _shape(prof([5.0, 0.0, 2.0], (0.5, nan))) == "not reached above (profile further)"


def test_a_fit_that_did_not_end_at_a_minimum():
    (f,) = Dr._converged(SimpleNamespace(status="max iterations"))
    assert f.severity == "problem" and "'max iterations'" in f.message
    assert Dr._converged(SimpleNamespace(status="converged")) == []
    r = SimpleNamespace(hessian_u=-np.eye(2), covariance_status="failed: not positive definite",
                        names=["a", "b"], scales=[("log",)] * 2)  # fmt: skip
    assert [f.check for f in Dr._determination(r)] == ["standard errors", "not at a minimum"]
