# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Parameters with one value per category: estimating, fixing and setting their entries
(k[S], k[R]) or the whole table (k[strain]), in fit text and in Python."""

import numpy as np
import pandas as pd
import pytest
import tenqsp as tq

import tenqsp_fit as tf

GROWTH = """model growth
axis strain categorical [S, R]
field B(strain)
param r[strain] = {S: 0.5, R: 0.1}
param f = 1
d_t(B) = r[strain]*f*B
B[strain=S](0) = 100
B[strain=R](0) = 1e4
observable bact = sum_strain(B)
"""

ORGANS = """model organs
axis organ categorical [liver, gut]
axis sub categorical [vascular, tissue]
field A(organ, sub)
param kp[organ, sub] = {liver: {vascular: 1, tissue: 2}, gut: {vascular: 3, tissue: 4}}
d_t(A) = -kp[organ, sub]*A
A(0) = 1
observable bact = sum_organ(sum_sub(A))
"""

TIMES = np.arange(1.0, 13.0)


def growth_data(rS=0.5, rR=0.1, units=1, sd=0.0, seed=0):
    """log10 of the total count, from the closed form; ``units`` repeats it per ID."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, units + 1):
        y = np.log10(100 * np.exp(rS * TIMES) + 1e4 * np.exp(rR * TIMES))
        y = y + sd * rng.standard_normal(len(TIMES))
        rows += [dict(ID=i, TIME=t, DV=v) for t, v in zip(TIMES, y, strict=True)]
    return pd.DataFrame(rows)


def fit_text(body: str, method: str = "optimize") -> str:
    return f"""fit g
data "d.csv"
observe log10(bact) = DV
{body}
error log10(bact) ~ additive(0.05)
method {method}
"""


@pytest.fixture
def growth():
    return tq.Model.from_text(GROWTH)


@pytest.fixture
def organs():
    return tq.Model.from_text(ORGANS)


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    growth_data().to_csv("d.csv", index=False)
    return tmp_path


def read(text, model):
    return tf.Problem.from_text(text, model=model)


def problems(text, model) -> str:
    with pytest.raises((tf.FitTextError, ValueError, TypeError)) as e:
        read(text, model)
    return str(e.value)


# ----------------------------------------------------------------------------- writing it
def test_the_forms_name_entries(folder, growth):
    p = read(fit_text("estimate r[strain] = {S: 0.4, R: 0.2}"), growth)
    assert p.estimated()[:2] == ["r[S]", "r[R]"]
    p = read(fit_text("estimate r[strain] = 0.3"), growth)
    assert p._starts()[1][:2].tolist() == [0.3, 0.3]
    p = read(fit_text("estimate r[ R ] = 0.2"), growth)  # spacing made standard
    assert p.estimated()[0] == "r[R]"
    # bounds hold for each entry on the line
    p = read(fit_text("estimate r[strain] = {S: 0.4, R: 0.2} in (0.01, 2)"), growth)
    assert p._starts()[2][:2] == [(0.01, 2.0), (0.01, 2.0)]


def test_text_and_python_give_the_same_problem(folder, growth):
    text = read(
        fit_text(
            "estimate r[strain] = {S: 0.4, R: 0.2} in (0.01, 2)\nfix f = 1\n"
            "prior r[strain] ~ lognormal(median=0.3, sdlog=2)"
        ),
        growth,
    )
    p = tf.Problem(growth, name="g")
    p.data("d.csv")
    p.observe("log10(bact)", "DV")
    p.estimate("r[strain]", {"S": 0.4, "R": 0.2}, bounds=(0.01, 2))
    p.fix("f", 1)
    p.prior("r[strain]", tf.lognormal(median=0.3, sdlog=2))
    p.error("log10(bact)", tf.additive(0.05))
    p.method("optimize")
    assert p == text
    assert read(text.to_text(), growth) == text  # the round trip


def test_entries_and_tables_in_python(growth):
    p = tf.Problem(growth)
    p.estimate("r[R]", 0.2)
    p.fix("r[S]", 0.5)
    assert p.estimated() == ["r[R]"]
    p = tf.Problem(growth)
    p.fix("r[strain]", {"S": 0.5, "R": 0.1})
    assert [d.name for d in p._params] == ["r[S]", "r[R]"]


@pytest.mark.parametrize(
    "line, message",
    [
        ("estimate r = 0.3", "r has one value per category of 'strain' (S, R); estimate "
         "r[strain] = {S: …, R: …}, or one category, e.g. estimate r[R] = …."),
        ("estimate r[strain] = {S: 0.3}", "R is missing; give every category, or estimate "
         "r[S] = … alone."),
        ("estimate r[strain] = {S: 0.3, R: 0.1, X: 1}", "'strain' has no category X"),
        ("estimate r[X] = 0.3", "'strain' has no category X; it has S, R."),
        ("estimate r[S, R] = 0.3", "r is over 'strain'; name one category, e.g. r[S]."),
        ("estimate f[S] = 0.3", "f has one value, not one per category; write estimate f."),
        ("estimate q[S] = 0.3", "the model has no parameter 'q' with one value per category"),
        ("estimate r[strain] ~ lognormal(median=0.3, sdlog=0.5)",
         "varies between units one category at a time: estimate r[S] ~"),
        ("estimate f = {S: 1, R: 2}", "braces give one value per category"),
        ("estimate r[S] = 0.3\nestimate r[strain] = 0.3", "r[S] is already estimated"),
        ("fix r[S] = 0.5\nestimate r[S] = 0.3", "r[S] is already fixed"),
        ("r = f\nestimate r[R] = 0.3", "r[R] is already set by r[R] = f"),
        ("correlate r[strain], f", "r[strain] names every category"),
    ],
)  # fmt: skip
def test_messages(folder, growth, line, message):
    assert message in problems(fit_text(line), growth)


def test_a_table_over_two_axes(folder, organs):
    p = read(
        fit_text(
            "estimate kp[organ, sub] = {liver: {vascular: 1, tissue: 2}, "
            "gut: {vascular: 3, tissue: 4}}"
        ),
        organs,
    )
    assert p.estimated()[:4] == [
        "kp[liver, vascular]", "kp[liver, tissue]", "kp[gut, vascular]", "kp[gut, tissue]",
    ]  # fmt: skip
    p = read(fit_text("estimate kp[liver,tissue] = 2"), organs)
    assert p.estimated()[0] == "kp[liver, tissue]"
    msg = problems(fit_text("estimate kp[organ, sub] = {liver: {vascular: 1}, gut: 3}"), organs)
    assert "tissue is missing (under liver)" in msg
    msg = problems(fit_text("estimate kp[liver] = 2"), organs)
    assert "kp is over 'organ' and 'sub'; name one category of each" in msg
    msg = problems(fit_text("estimate kp[liver, organ] = 2"), organs)
    assert "kp is over 'organ' and 'sub', not 'organ'" in msg


# ----------------------------------------------------------------------------- formulas
def test_formulas_set_entries(folder, growth):
    p = read(fit_text("r = r_all\nestimate r_all = 0.3"), growth)
    assert [d.name for d in p._defines] == ["r[S]", "r[R]"]
    assert p == read(fit_text("r[strain] = r_all\nestimate r_all = 0.3"), growth)
    p = read(fit_text("r[R] = r[S]*frac\nestimate r[S] = 0.4, frac = 0.3"), growth)
    assert [d.name for d in p._defines] == ["r[R]"]
    p.check()
    assert read(p.to_text(), growth) == p


# ----------------------------------------------------------------------------- fits
def test_entries_are_estimated(folder, growth):
    r = tf.fit(read(fit_text("estimate r[strain] = {S: 0.3, R: 0.3}\noptions progress=none"),
                    growth))  # fmt: skip
    est = dict(zip(r.names, r.x, strict=True))
    assert est["r[S]"] == pytest.approx(0.5, rel=1e-3)
    assert est["r[R]"] == pytest.approx(0.1, rel=1e-3)
    assert "r[R]" in r.table().index or "r[R]" in r.table().iloc[:, 0].tolist()


def test_one_estimate_shared_by_every_category(tmp_path, monkeypatch, growth):
    monkeypatch.chdir(tmp_path)
    growth_data(rS=0.3, rR=0.3).to_csv("d.csv", index=False)
    r = tf.fit(read(fit_text("r = r_all\nestimate r_all = 0.5\noptions progress=none"), growth))
    assert dict(zip(r.names, r.x, strict=True))["r_all"] == pytest.approx(0.3, rel=1e-3)


def test_a_fixed_entry_is_used_in_the_fit_and_in_predictions(tmp_path, monkeypatch, growth):
    monkeypatch.chdir(tmp_path)
    growth_data(rS=0.6, rR=0.1).to_csv("d.csv", index=False)  # S differs from the model's 0.5
    r = tf.fit(read(fit_text("fix r[S] = 0.6\nestimate r[R] = 0.3\noptions progress=none"),
                    growth))  # fmt: skip
    assert dict(zip(r.names, r.x, strict=True))["r[R]"] == pytest.approx(0.1, rel=1e-3)
    pr = r.predict(times=[6.0], outputs=["bact"])
    expect = 100 * np.exp(0.6 * 6) + 1e4 * np.exp(0.1 * 6)
    assert float(pr.table["median"].iloc[0]) == pytest.approx(expect, rel=1e-4)


def test_an_entry_varies_between_units_with_a_shared_spread(tmp_path, monkeypatch, growth):
    """NONMEM's SAME: two entries vary between units with one estimated spread."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(1, 9):
        rS = 0.5 * np.exp(0.2 * rng.standard_normal())
        rR = 0.1 * np.exp(0.2 * rng.standard_normal())
        y = np.log10(100 * np.exp(rS * TIMES) + 1e4 * np.exp(rR * TIMES))
        rows += [dict(ID=i, TIME=t, DV=v + 0.02 * rng.standard_normal())
                 for t, v in zip(TIMES, y, strict=True)]  # fmt: skip
    pd.DataFrame(rows).to_csv("d.csv", index=False)
    text = fit_text(
        "unit ID\nestimate r[S] ~ lognormal(median=0.4, sdlog=w)\n"
        "estimate r[R] ~ lognormal(median=0.2, sdlog=w)\nestimate w = 0.3\n"
        "options progress=none checks=quick",
        method="laplace",
    )
    r = tf.fit(read(text, growth))
    est = dict(zip(r.names, r.x, strict=True))
    assert est["r[S].median"] == pytest.approx(0.5, rel=0.15)
    assert est["r[R].median"] == pytest.approx(0.1, rel=0.15)
    assert 0.08 < est["w"] < 0.4


# ----------------------------------------------------------------------------- observables
def test_an_observable_reading_a_parameter_per_category_does_not_stop_the_fit(folder):
    """An observable (or definition) that reads r[strain] makes no difference to a fit that
    links another observable."""
    m = tq.Model.from_text(
        GROWTH + "rate = r[strain]*B\nobservable growth = rate\nobservable both = B*r[strain]\n"
    )
    r = tf.fit(read(fit_text("estimate r[strain] = {S: 0.3, R: 0.3}\noptions progress=none"), m))
    est = dict(zip(r.names, r.x, strict=True))
    assert est["r[S]"] == pytest.approx(0.5, rel=1e-3)


@pytest.mark.parametrize(
    "lhs, message",
    [
        ("both", "the observable both has one value per category of 'strain'; a measurement is "
         "one value. Write one observable per category in the model"),
        ("cells", "(e.g. 'observable cells_S = B[strain=S]')"),
        ("log10(cells)", "the observable cells has one value per category"),
        ("cells[S]", "the left side of observe is a formula of the model's observables; an "
         "observable cannot be sliced there. Write one observable per category"),
        ("cells[strain=S]", "an observable cannot be sliced there"),
        ("bact[S]", "cannot be sliced there. Write an observable with one value in the model"),
    ],
)  # fmt: skip
def test_an_observable_per_category_is_refused_at_observe(growth, lhs, message):
    m = tq.Model.from_text(GROWTH + "observable both = B*r[strain]\nobservable cells = B\n")
    with pytest.raises(ValueError) as e:
        tf.Problem(m).observe(lhs, "DV")
    assert message in str(e.value)
    tf.Problem(m).observe("bact", "DV")  # one value: accepted


def test_a_model_reading_entries_by_label(tmp_path):
    """The model reads V[organ=spleen] (one entry) and K[phage, organ=liver] (one per phage):
    estimating V[spleen] and K[p2, liver] reaches them; the other entries keep their values."""
    model = tq.Model.from_text("""model s
axis organ categorical [liver, spleen]
axis phage categorical [p1, p2]
param V[organ] = {liver: 2, spleen: 0.5}
param K[phage, organ] = {p1: {liver: 1, spleen: 9}, p2: {liver: 0.5, spleen: 9}}
field C(phage)
d_t(C) = -K[phage, organ=liver]*C
C(0) = 1
observable y1 = V[organ=spleen]*C[phage=p1]
observable y2 = C[phage=p2]
""")
    t = np.array([0.5, 1, 2, 4])
    f = tmp_path / "s.csv"
    df = pd.DataFrame({"TIME": np.r_[t, t], "DVID": [1] * 4 + [2] * 4,
                       "DV": np.r_[0.8 * np.exp(-t), np.exp(-0.3 * t)]})  # fmt: skip
    df.sort_values(["TIME", "DVID"]).to_csv(f, index=False)
    r = tf.fit(tf.Problem.from_text(
        f'fit s\ndata "{f}"\nobserve y1 = DV where DVID == 1\nobserve y2 = DV where DVID == 2\n'
        "estimate V[spleen] = 0.5, K[p2, liver] = 0.5\nerror y1 ~ additive(fix(0.01))\n"
        "error y2 ~ additive(fix(0.01))\nmethod optimize\noptions progress=none checks=quick\n",
        model=model,
    ))  # fmt: skip
    assert abs(r.estimates["V[spleen]"] - 0.8) < 1e-4
    assert abs(r.estimates["K[p2, liver]"] - 0.3) < 1e-4
    text = r.model().to_text()
    assert "liver: 1, spleen: 9" in text and "K[phage, organ=liver]" in text
