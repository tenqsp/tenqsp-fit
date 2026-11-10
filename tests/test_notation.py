# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Every notation a TenQSP formula may use, in the model's named formulas and observables,
through a fit and everything that reads the model afterwards. Oracle: the data are the
model's own solution (exact), so the fit returns the true values."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
import tenqsp as tq

import tenqsp_fit as tf

matplotlib.use("Agg")
pytestmark = pytest.mark.timeout(600)  # one fit of 8 links, compiled once: about 2 min

MODEL = """model every_notation
axis w in [0, 1] n=6 kind=one_way
axis a in [0, 2] n=8
axis s categorical [S, R]
param v = 2, k = 0.5, CL = 1, ktr = 0.3, d = 0.1, Ka = 0.5
param kd[s] = {S: 0.2, R: 0.05}
param V[s] = {S: 1, R: 2}
field P(w)
field B(s)
field Q(a)
compartment A
compartment C
d_t(P) = -d_w(v*P)
    at w=0: P = k*A
    at w=1: outflow
d_t(A) = -k*A
d_t(B) = ktr*C - kd[s]*B
d_t(Q) = -d*Q
Q(0) = 1 + 0*a
d_t(C) = v*S_end - CL*C - ktr*C
S_end = P(w=1)
S_up = P(w=upper)
B_R = B[s=R]
B_tot = sum_s(B)
Q_tot = int_a(Q)
mean_a = int_a(a*Q)/int_a(Q)
h(a) = exp(-a)*Q
h_tot = int_a(h(a))
VR = V[s=R]
gate = 1 if t > 1 else 0.5
sat = C/(Ka + C)   if t > 3
      0            otherwise
mix = log10(1 + S_up) + max(B_R, 0) + sigmoid(C - 1) + VR*gate + sat
observable o_end = S_end
observable o_end_direct = v*P(w=1)
observable o_slice = B_R
observable o_slice_direct = B[s=S]*V[s=S]
observable o_total = B_tot + Q_tot + h_tot + sum_s(B*V[s])
observable o_mean = mean_a
observable o_mix = mix + C
observable o_inline = int_a(exp(-a)*Q) + P(w=lower)*0
"""
TIMES = [0.5, 1, 2, 4, 6, 8, 12]


@pytest.fixture(scope="module")
def every(tmp_path_factory):
    m = tq.Model.from_text(MODEL)
    names = [o.name for o in m.to_ir().observables]
    rows = []
    for i, dose in enumerate([5.0, 10.0, 20.0, 40.0], 1):
        sol = m.solve((0, 12), times=TIMES, regimen=tq.bolus("A", dose, times=[0]),
                      rtol=1e-10, atol=1e-12)  # fmt: skip
        for j, o in enumerate(names, 1):
            for t, c in zip(TIMES, np.asarray(sol[o]), strict=True):
                rows.append(dict(ID=i, TIME=t, DVID=j, DV=float(c), D=dose))
    df = pd.DataFrame(rows).sort_values(["ID", "TIME", "DVID"], kind="stable")
    folder = tmp_path_factory.mktemp("every")
    df.to_csv(folder / "all.csv", index=False)
    text = ["fit every", f'data "{folder / "all.csv"}"', "unit ID", "doses D at t=0 into A"]
    text += [f"observe {o} = DV where DVID == {j}" for j, o in enumerate(names, 1)]
    text += ["estimate k = 0.3", "estimate CL = 0.7", "fix v = 2"]
    text += [f"error {o} ~ proportional(0.1)" for o in names]
    text += ["method optimize", "options rtol=1e-10 atol=1e-12 progress=none checks=quick"]
    p = tf.Problem.from_text("\n".join(text) + "\n", model=m)
    return m, p, tf.fit(p), folder


def test_a_fit_reads_every_notation(every):
    m, p, r, folder = every
    assert "fixed" in p.summary()
    assert r.estimates["k"] == pytest.approx(0.5, rel=1e-4)
    assert r.estimates["CL"] == pytest.approx(1.0, rel=1e-4)
    r.doctor()
    assert len(r.tables()["sdtab"]) == len(TIMES) * 4 * 8
    r.plot_gof()
    r.plot_individual()
    r.vpc(n=5)
    r.predict(regimen=tq.bolus("A", 5, times=[0]), times=[1, 2])
    back = tq.Model.from_text(r.model().to_text())
    assert {p.name: p.value for p in back.to_ir().params}["k"] == pytest.approx(0.5, rel=1e-4)
    r.save(folder / "saved")


@pytest.mark.parametrize(
    "name, message",
    [("S_end", r"a formula of the model \(S_end = P\(w=1\)\)"),
     ("mean_a", "a formula of the model"), ("gate", "a formula of the model"),
     ("sat", "estimate Ka instead"), ("VR", r"estimate V\[R\] instead"),
     ("B_R", "a value the model takes from its fields"),
     ("B_tot", "a value the model takes from its fields"),
     ("h_tot", "a value the model takes from its fields")],
)  # fmt: skip
def test_what_the_model_computes_is_not_estimated(every, name, message):
    m = every[0]
    with pytest.raises(ValueError, match=message):
        tf.Problem(m).estimate(name, 1.0)
