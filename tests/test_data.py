# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The data as the fit reads them: units and their TenQSP scenarios, the row checks,
and provenance."""

import hashlib
import warnings

import numpy as np
import pandas as pd
import pytest
import tenqsp as tq

import tenqsp_fit as tf

PK = """model pk
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""
IV = """model iv
param CL=4, V=30
d_t(C) = -CL/V*C
observable conc = C/V
"""


def nonmem(n=3, times=(0.5, 1, 2, 4)):
    """One dose (EVID 1, CMT 1) and samples (EVID 0, CMT 2) per unit, a WT column."""
    rows = []
    for i in range(1, n + 1):
        rows.append(dict(ID=i, TIME=0.0, AMT=4.0 + i, DV=np.nan, EVID=1, CMT=1, WT=60 + 10 * i))
        for t in times:
            rows.append(
                dict(
                    ID=i, TIME=t, AMT=np.nan, DV=i / t if t else 0.0, EVID=0, CMT=2, WT=60 + 10 * i
                )
            )
    return pd.DataFrame(rows)


def r_layout(n=3, times=(0.5, 1, 2, 4)):
    """The same data as R lays them out: no dose rows, the dose as a column."""
    return pd.DataFrame(
        [
            dict(Subject=i, Wt=60 + 10 * i, Dose=4.0 + i, Time=t, conc=i / t)
            for i in range(1, n + 1)
            for t in times
        ]
    )


def problem(table, doses=None, model=PK, **columns):
    p = tf.Problem(tq.Model.from_text(model))
    p.data(table, **columns)
    p.unit("ID")
    p.doses(**(doses or {"cmt": {1: "depot"}}))
    p.observe("conc", "DV")
    p.estimate("ka" if model == PK else "CL", 1.0)
    p.error("conc", tf.additive(0.5))
    p.method("optimize")
    return p


def problems(p):
    with pytest.raises(tf.ProblemError) as e:
        p.check()
    return str(e.value)


# ----------------------------------------------------------------------------- units
def test_units_carry_scenarios_observations_and_covariates():
    p = problem(nonmem())
    p.define("CL", "CL_pop*(WT/70)^0.75")
    p.estimate("CL_pop", 0.04)
    units = p.units()
    assert [u.id for u in units] == [1, 2, 3]
    u = units[1]
    assert isinstance(u, tf.Unit) and isinstance(u.scenario, tq.Scenario)
    assert u.scenario.regimen == tq.bolus("depot", 6.0, times=[0])
    assert u.scenario.times == (0.5, 1.0, 2.0, 4.0)
    assert u.scenario.params == {}
    obs = u.observations["conc"]
    assert isinstance(obs, tf.Observations)
    np.testing.assert_allclose(obs.values, [4, 2, 1, 0.5])
    assert list(obs.rows) == [7, 8, 9, 10]  # data rows, counted from 1
    assert u.covariates == {"WT": 80.0}
    assert "unit 2: 1 dose, 4 times" in repr(u)


def test_r_layout_and_nonmem_layout_read_the_same():
    a = problem(nonmem())
    b = problem(
        r_layout(),
        doses={"column": "Dose", "at": 0, "into": "depot"},
        ID="Subject",
        TIME="Time",
        DV="conc",
    )
    assert a != b  # different declarations ...
    assert a.units() == b.units()  # ... the same data
    pd.testing.assert_frame_equal(a.table(), b.table())


def test_table_is_sorted_with_doses_first():
    t = problem(nonmem(n=2)).table()
    assert list(t.columns) == ["ID", "TIME", "EVID", "AMT", "CMT", "DVID", "DV"]
    assert list(t.EVID[:5]) == [1, 0, 0, 0, 0]
    assert t.CMT[0] == "depot" and t.DVID[1] == "conc"
    assert len(t) == 10


def test_doses_by_into_cmt_infusion_and_addl():
    df = nonmem(n=1)
    df.loc[0, "ADDL"], df.loc[0, "II"] = 2, 12
    df.loc[0, "CMT"] = 7  # ignored by "doses into"
    u = problem(df, doses={"into": "depot"}).units()[0]
    assert [e["time"] for e in u.scenario.regimen.to_list()] == [0, 12, 24]
    df = nonmem(n=1).assign(RATE=[2.5, 0, 0, 0, 0])
    p = problem(df, doses={"cmt": {1: "C"}}, model=IV)
    (u,) = p.units()
    assert u.scenario.regimen == tq.infusion("C", rate=2.5, amount=5.0, start=0)
    t = p.table()
    assert t.RATE[0] == 2.5 and t.AMT[0] == pytest.approx(5.0)


def test_units_without_measurements_are_named_in_the_summary():
    df = nonmem()
    df = df[~((df.ID == 3) & (df.EVID == 0))]
    p = problem(df)
    assert [u.id for u in p.units()] == [1, 2]
    assert "3 units, 1 without linked measurements (not fitted)" in p.summary()


def test_censor_limits_per_measurement():
    df = nonmem(n=1).assign(LLOQ=0.3)
    p = problem(df)
    p.censor("conc", below="LLOQ")
    (u,) = p.units()
    np.testing.assert_allclose(u.observations["conc"].below, [0.3] * 4)
    assert "LLOQ" in p.table().columns
    q = problem(nonmem(n=1))
    q.censor("conc", below=0.1)
    assert q.units()[0].observations["conc"].below[0] == 0.1
    assert q.units()[0].observations["conc"] != u.observations["conc"]


# ----------------------------------------------------------------------------- row checks
def test_unsorted_times():
    df = nonmem(n=2)
    df.loc[[2, 3], "TIME"] = [2.0, 1.0]
    assert "times go back on row 4 " in problems(problem(df))


def test_refused_records_name_their_rows_and_a_fix():
    df = nonmem(n=2).assign(SS=0, RATE=0)
    df.loc[0, "SS"] = 1
    df.loc[5, "RATE"] = -2
    df.loc[3, "EVID"] = 4
    msg = problems(problem(df))
    assert "steady-state doses (SS) on row 1 are not supported yet" in msg
    assert "RATE -1 or -2 (a rate or duration set by the model) on row 6" in msg
    assert "EVID 3 or 4 (a reset) on row 4" in msg


def test_inconsistent_evid_mdv_amt_dv():
    df = nonmem(n=1).assign(MDV=0)
    df.loc[1, "AMT"] = 3.0
    df.loc[2, "MDV"] = 1
    df.loc[3, "DV"] = np.nan
    msg = problems(problem(df))
    assert "row 2 has EVID 0 (a measurement) and a dose amount" in msg
    assert "row 3 has EVID 0 (a measurement) and MDV 1" in msg
    assert "row 4 has EVID 0 (a measurement) but no DV" in msg


def test_a_dose_of_zero_is_a_dose_a_missing_amount_is_not():
    """AMT 0 on a dose row is a placebo record (NONMEM allows it); a missing AMT is refused."""
    df = nonmem(n=2)
    df.loc[0, "AMT"] = 0.0
    problem(df).check()
    df.loc[0, "AMT"] = np.nan
    assert "row 1 has EVID 1 (a dose) but no amount (AMT)" in problems(problem(df))


def test_missing_and_negative_times_and_text_in_numbers():
    df = nonmem(n=1)
    df["TIME"] = df["TIME"].astype(object)
    df.loc[1, "TIME"] = -1.0
    df.loc[2, "TIME"] = np.nan
    df.loc[3, "TIME"] = "two"
    msg = problems(problem(df))
    assert "TIME: row 4 is not numbers" in msg
    assert "TIME is missing on rows 3 and 4" in msg
    assert "TIME is below 0 on row 2" in msg


def test_no_time_column():
    df = nonmem(n=1).rename(columns={"TIME": "TIM"})
    msg = problems(problem(df))
    assert "the data have no TIME column. Did you mean 'TIM'?" in msg


def test_dose_column_and_covariates_are_one_value_per_unit():
    df = r_layout()
    df.loc[1, "Dose"] = 9.0
    df.loc[5, "Wt"] = 99.0
    p = problem(
        df,
        doses={"column": "Dose", "at": 0, "into": "depot"},
        ID="Subject",
        TIME="Time",
        DV="conc",
    )
    p.define("CL", "CL_pop*(Wt/70)^0.75")
    p.estimate("CL_pop", 0.04)
    msg = problems(p)
    assert "Dose (the dose) changes within units 1;" in msg
    assert "Wt (a covariate) changes within units 2;" in msg


def test_measurement_at_a_dose_time_into_what_it_reads():
    df = nonmem(n=1, times=(0, 1, 2))
    df = df.iloc[[1, 0, 2, 3]].reset_index(drop=True)  # the t=0 sample before the dose
    p = problem(df, doses={"into": "C"}, model=IV)
    msg = problems(p)
    assert "row 1: a measurement of conc at the time of a dose into C" in msg
    assert "TenQSP gives the value just after the dose" in msg
    # the dose row first: the value after the dose, as NONMEM reads it too
    assert problem(df.iloc[[1, 0, 2, 3]].reset_index(drop=True), {"into": "C"}, IV).units()
    # a dose into a compartment the observable does not read: no question
    assert problem(df, doses={"into": "depot"}).units()


def test_repeated_rows_warn():
    df = pd.concat([nonmem(n=1), nonmem(n=1).iloc[[2]]]).sort_values("TIME", kind="stable")
    with pytest.warns(UserWarning, match="repeat earlier rows exactly at row 4"):
        problem(df.reset_index(drop=True)).check()


def test_tenqsp_refusal_is_reported_not_reinterpreted():
    df = nonmem(n=1)
    df.loc[0, "ADDL"], df.loc[0, "II"] = 1.5, 12
    msg = problems(problem(df))
    assert "TenQSP cannot read the dose records" in msg


# ----------------------------------------------------------------------------- provenance
def test_provenance_records_the_file_hash_and_filters(tmp_path):
    df = nonmem().assign(DVID=1)
    df.loc[df.ID == 3, "DVID"] = 2
    f = tmp_path / "pk.csv"
    df.to_csv(f, index=False, na_rep=".")
    p = tf.Problem(tq.Model.from_text(PK))
    p.data(str(f))
    p.unit("ID")
    p.doses(cmt={1: "depot"})
    p.observe("conc", "DV", where="DVID == 1")
    p.ignore("DV", where="DVID == 2")
    p.estimate("ka", 1.0)
    p.error("conc", tf.additive(0.5))
    p.method("optimize")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prov = p.provenance()
    assert prov["data"]["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert prov["data"]["file"] == str(f.resolve())
    assert prov["rows"] == {
        "measurements": 12,
        "linked": 8,
        "left out": [{"rule": "ignore DV where DVID == 2", "rows": 4}],
        "doses": 3,
        "neither": 0,
    }
    assert prov["units"] == {"column": "ID", "in the data": 3, "with measurements": 2}
    from tenqsp_fit.passport import _data_lines  # as the passport says it

    lines = _data_lines(prov)
    assert lines[0].startswith("data file ") and lines[0].endswith(prov["data"]["sha256"])
    assert "left out: 4 row(s) by ignore DV where DVID == 2" in lines
    assert prov["model"]["hash"] == tq.Model.from_text(PK).to_ir().content_hash()
    # a table given in Python is hashed by its contents
    a, b = problem(nonmem()), problem(nonmem())
    assert a.provenance()["data"]["sha256"] == b.provenance()["data"]["sha256"]
    c = nonmem()
    c.loc[3, "DV"] = 0.1
    assert problem(c).provenance()["data"]["sha256"] != a.provenance()["data"]["sha256"]


def test_spaces_around_names_and_values(tmp_path):
    f = tmp_path / "pk.csv"
    f.write_text(nonmem(n=1).to_csv(index=False, na_rep=".").replace(",", ", "))
    p = problem(str(f))
    assert p.units() == problem(nonmem(n=1)).units()


def test_a_link_on_another_column_reads_observation_records_only():
    """observe log(conc) = LNDV on a NONMEM file: a row with EVID 2 (or MDV 1) holding 0 in
    LNDV is not a measurement."""
    df = pd.DataFrame(
        dict(ID=1, TIME=[0, 1, 2, 3], AMT=[100, 0, 0, 0], LNDV=[0, 1.13, 1.39, 0],
             EVID=[1, 0, 0, 2], MDV=[1, 0, 0, 1], CMT=[1, 2, 2, 2])
    )  # fmt: skip
    for data in (df, df.drop(columns="EVID")):
        p = tf.Problem(tq.Model.from_text(PK))
        p.data(data)
        p.doses(cmt={1: "depot"})
        p.observe("log(conc)", "LNDV")
        p.error("log(conc)", tf.additive(0.2))
        p.estimate("ke", 0.1)
        p.method("optimize")
        o = p.units()[0].observations["log(conc)"]
        np.testing.assert_array_equal(o.times, [1, 2])


def test_ignore_lines_leave_their_rows_out():
    """An ignore line on any column leaves its rows out of the fit (not only of the summary);
    with observe DV, ignore DV where ... is the documented way (no 'linked twice')."""
    wide = pd.DataFrame(dict(TIME=[1, 2, 3, 4], conc=[1.0, 2, 3, 4], BLQ=[0, 0, 1, 1]))
    long = wide.rename(columns={"conc": "DV"})
    for df, column in ((wide, "conc"), (long, "DV")):
        p = tf.Problem(tq.Model.from_text(PK))
        p.data(df)
        p.observe("conc", column)
        p.ignore(column, where="BLQ == 1")
        p.error("conc", tf.additive(1))
        p.estimate("ke", 0.1)
        p.method("optimize")
        p.check()
        np.testing.assert_array_equal(p.units()[0].observations["conc"].values, [1, 2])
        assert "conc <- " + column + " (all rows)   2 rows" in p.summary()
        assert p.provenance()["rows"]["linked"] == 2


def test_a_missing_evid_a_missing_limit_and_no_data_are_reported():
    df = nonmem(n=1)
    df.loc[2, "EVID"] = np.nan
    with pytest.raises(tf.ProblemError, match=r"EVID is missing on row 3"):
        problem(df).check()
    df = nonmem(n=1).assign(LLOQ=np.nan, BLQ=0)
    df.loc[4, ["BLQ", "LLOQ"]] = [1, np.nan]
    p = problem(df)
    p.censor("conc", below="LLOQ", where="BLQ == 1")
    with pytest.raises(tf.ProblemError, match=r"LLOQ is missing \(or not a number\) on cens"):
        p.check()
    p = tf.Problem(tq.Model.from_text(PK))
    p.estimate("ke", 0.1)
    p.method("optimize")
    p.summary()  # a preview needs no data; what reads them does
    for read in (p.units, p.table, p.provenance, lambda: tf.fit(p)):
        with pytest.raises(tf.ProblemError, match='no data; add: data "file.csv"'):
            read()


def test_a_bom_trailing_spaces_and_dur(tmp_path):
    """Excel's "CSV UTF-8" starts with a BOM; spaces after values are dropped; DUR maps."""
    f = tmp_path / "bom.csv"
    f.write_bytes("﻿ID,TIME,DV,DVID,Duration\n1,0,1.5,bact ,1\n1,1,2.5,bact,1\n"
                  .encode())  # fmt: skip
    from tenqsp_fit.data import apply_map, read_table

    df, _, _ = read_table(str(f))
    assert list(df.columns)[0] == "ID" and list(df.DVID) == ["bact", "bact"]
    assert "DUR" in apply_map(df, {"DUR": "Duration"}).columns


def test_a_measurement_at_the_start_of_an_infusion_is_read():
    """An infusion does not jump the value, so a sample at its start is not ambiguous."""
    df = pd.DataFrame(dict(ID=1, TIME=[0, 1, 6, 6], AMT=[100, np.nan, 50, np.nan],
                           RATE=[np.nan, np.nan, 25, np.nan], EVID=[1, 0, 1, 0],
                           DV=[np.nan, 3.1, np.nan, 4.0], CMT=[1, 2, 2, 2]))  # fmt: skip
    p = problem(df, doses={"cmt": {1: "depot", 2: "centr"}})
    np.testing.assert_array_equal(p.units()[0].observations["conc"].times, [1, 6])


TWO = """model two
axis phage categorical [a, b]
param k[phage] = {a: 0.3, b: 0.1}
field C(phage)
compartment B
d_t(C) = -k[phage]*C
d_t(B) = -0.1*B
observable Ca = C[phage=a]
observable Cb = C[phage=b]
"""


def two_phages(rows: bool):
    """Two units, each dosed into both elements of a field at t=2 (as two phages into a
    plasma field): exact data, from the closed form. As two dose columns, or as dose rows
    with CMT 1 and 2."""
    out = []
    for i, (a, b) in enumerate([(10.0, 4.0), (20.0, 8.0)], 1):
        if rows:
            out += [dict(ID=i, TIME=2.0, AMT=a, CMT=1, EVID=1, DVID=0),
                    dict(ID=i, TIME=2.0, AMT=b, CMT=2, EVID=1, DVID=0)]  # fmt: skip
        for t in (3.0, 4.0, 6.0, 8.0):
            more = {} if rows else {"AMTA": a, "AMTB": b}
            ya, yb = a * np.exp(-0.3 * (t - 2)), b * np.exp(-0.1 * (t - 2))
            out += [dict(ID=i, TIME=t, DV=ya, EVID=0, DVID=1, **more),
                    dict(ID=i, TIME=t, DV=yb, EVID=0, DVID=2, **more)]  # fmt: skip
    return pd.DataFrame(out)


def fit_two(df, doses):
    p = tf.Problem(tq.Model.from_text(TWO))
    p.data(df)
    p.unit("ID")
    doses(p)
    p.observe("Ca", "DV", where="DVID == 1")
    p.observe("Cb", "DV", where="DVID == 2")
    for lab in ("a", "b"):
        p.estimate(f"k[{lab}]", 0.2)
    p.error("Ca", tf.proportional(0.1))
    p.error("Cb", tf.proportional(0.1))
    p.method("optimize")
    p.options(progress="none", checks="quick")
    return p


@pytest.mark.parametrize("rows", [False, True])
def test_doses_into_elements_of_a_field(rows):
    """``doses AMTA at t=2 into C[phage=a]`` (one statement per element), or ``doses CMT 1
    into C[phage=a], 2 into C[phage=b]``: each unit's regimen doses both elements, and the
    fit recovers both rates exactly."""
    if rows:

        def doses(p):
            p.doses(cmt={1: "C[phage=a]", 2: "C[ phage = b ]"})
    else:

        def doses(p):
            p.doses(column="AMTA", at=2, into="C[phage = a]")
            p.doses(column="AMTB", at=2, into="C[phage=b]")

    p = fit_two(two_phages(rows), doses)
    reg = p.units()[1].scenario.regimen
    assert [(e.time, e.target, e.where, e.amount) for e in reg.events] == [
        (2.0, "C", (("phage", "a"),), 20.0),
        (2.0, "C", (("phage", "b"),), 8.0),
    ]
    text = p.to_text()
    if rows:
        assert "doses CMT 1 into C[phage=a], 2 into C[phage=b]" in text
    else:
        assert "doses AMTA at t=2 into C[phage=a]\ndoses AMTB at t=2 into C[phage=b]" in text
        assert "2 doses (one per unit)" in p.summary()
    r = tf.fit(p)
    assert r.estimates["k[a]"] == pytest.approx(0.3, rel=1e-5)
    assert r.estimates["k[b]"] == pytest.approx(0.1, rel=1e-5)


def test_doses_into_a_field_as_fit_text(tmp_path):
    df = two_phages(rows=False)
    df.to_csv(tmp_path / "two.csv", index=False)
    (tmp_path / "two.tq").write_text(
        TWO + '\nfit two\ndata "two.csv"\nunit ID\n'
        "doses AMTA at t=2 into C[phage=a]\ndoses AMTB at t=2 into C[phage=b]\n"
        "observe Ca = DV where DVID == 1\nobserve Cb = DV where DVID == 2\n"
        "estimate k[a] = 0.2\nestimate k[b] = 0.2\n"
        "error Ca ~ proportional(0.1)\nerror Cb ~ proportional(0.1)\nmethod optimize\n"
    )
    p = tf.load(tmp_path / "two.tq")
    assert [d.into for d in p._doses] == ["C[phage=a]", "C[phage=b]"]
    # each statement keeps its own line, for the messages about it
    assert "doses AMTA" in p._doses[0].origin and "doses AMTB" in p._doses[1].origin
    assert tf.fit(p).estimates["k[b]"] == pytest.approx(0.1, rel=1e-5)


@pytest.mark.parametrize(
    "calls, message",
    [
        ([dict(column="AMTA", at=2, into="C")], r"say which element.*C\[phage=a\]"),
        ([dict(column="AMTA", at=2, into="C[phage=c]")], "no label 'c'"),
        ([dict(column="AMTA", at=2, into="C[organ=a]")], "write the axis and its label"),
        ([dict(column="AMTA", at=2, into="B[phage=a]")], "'B' is a compartment"),
        ([dict(column="AMTA", at=2, into="D")], "no compartment or field 'D'"),
        ([dict(column="AMTA", at=2, into="C[phage=a]"),
          dict(column="AMTB", at=2, into="C[phage=a]")], "declared twice; give one per place"),
        ([dict(into="C[phage=a]"), dict(column="AMTB", at=2, into="C[phage=b]")],
         "declared twice; give one doses statement"),
        ([dict(column="AMTB", at=2, into="C[phage=b]"), dict(into="C[phage=a]")],
         "declared twice; give one doses statement"),
    ],
)  # fmt: skip
def test_doses_into_a_field_mistakes(calls, message):
    p = tf.Problem(tq.Model.from_text(TWO))
    with pytest.raises(ValueError, match=message):
        for c in calls:
            p.doses(**c)


def test_doses_into_a_field_over_a_continuous_axis_arrive_later():
    m = tq.Model.from_text(
        "model z\naxis z in [0, 1] n=4 kind=one_way\nparam v = 1\nfield u(z)\n"
        "d_t(u) = -d_z(v*u)\n    at z=1: outflow\nobservable total = u(z=1)\n"
    )
    with pytest.raises(ValueError, match="continuous axis z"):
        tf.Problem(m).doses(into="u")


def test_a_model_reading_a_field_at_a_boundary_can_be_fitted():
    """A model formula in the model's own notation (a field at a boundary, ``P(w=1)``, as a
    transit delay): the units build, the dose-time check knows the observable reads ``P``,
    and the summary, the fit (exact data: the true values back) and what reads the model
    afterwards all work."""
    text = (
        "model edge\naxis w in [0, 1] n=8 kind=one_way\nparam v = 2, k = 0.5, CL = 1\n"
        "field P(w)\ncompartment A\ncompartment C\nd_t(P) = -d_w(v*P)\n"
        "    at w=0: P = k*A\n    at w=1: outflow\nd_t(A) = -k*A\nS = P(w=1)\n"
        "d_t(C) = v*S - CL*C\nobservable out = C\n"
    )
    m = tq.Model.from_text(text)
    times = [0.5, 1, 2, 3, 4, 6, 8, 12]
    rows = []
    for i, dose in enumerate([10.0, 40.0], 1):
        sol = m.solve((0, 12), times=times, regimen=tq.bolus("A", dose, times=[0]),
                      rtol=1e-10, atol=1e-12)  # fmt: skip
        out = zip(times, sol["out"], strict=True)
        rows += [dict(ID=i, TIME=t, DV=float(c), D=dose) for t, c in out]
    p = tf.Problem(m)
    p.data(pd.DataFrame(rows))
    p.unit("ID")
    p.doses(column="D", at=0, into="A")
    p.observe("out", "DV")
    p.estimate("k", 0.3)
    p.estimate("CL", 0.7)
    p.fix("v", 2)
    p.error("out", tf.proportional(0.1))
    p.method("optimize")
    p.options(rtol=1e-10, atol=1e-12, progress="none", checks="quick")
    assert p._m.reads("out") == {"C"} and p._m.reads("S") == {"P"}
    assert p._m.param_names("S") == set()  # a field and an axis: no parameter
    assert "fixed" in p.summary()
    r = tf.fit(p)
    assert r.estimates["k"] == pytest.approx(0.5, rel=1e-4)
    assert r.estimates["CL"] == pytest.approx(1.0, rel=1e-4)
    assert len(r.tables()["sdtab"]) == len(rows)
    assert "S = P(w=1)" in r.model().to_text()
    with pytest.raises(ValueError, match=r"a formula of the model \(S = P\(w=1\)\)"):
        p2 = tf.Problem(m)
        p2.estimate("S", 1.0)
