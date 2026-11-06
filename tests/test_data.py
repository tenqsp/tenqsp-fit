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
