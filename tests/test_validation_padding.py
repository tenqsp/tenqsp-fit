# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Padding (PRD F4.9, Section 11): tenqsp-fit runs every unit in one batched TenQSP function,
the units padded to the largest numbers of doses and samples. The padding must not change the
likelihood: on B1 made ragged (subjects keep 7 to 11 samples, some get a second dose), -2LL of
all units at once equals the sum over the units fitted alone, within 0.01 at the default rtol
(acceptance), at the reference estimates, for the population (Laplace) and pooled objectives."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tenqsp_fit as tf

B1 = Path(__file__).parents[1] / "validation" / "b1"
MODEL = """model theoph
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""
FIT = """fit {name}
data "{file}" with ID=Subject, TIME=Time, DV=conc, AMT=AMT, EVID=EVID
unit ID
doses into depot
observe conc = DV
{estimates}
error conc ~ additive(1)
method {method}
options progress=none checks=quick
"""
POP = ("estimate ka ~ lognormal(median=1.5, sdlog=0.5)\n"
       "estimate CL ~ lognormal(median=0.04, sdlog=0.5)\nestimate ke = 0.08")  # fmt: skip
POOLED = "estimate ka = 1.5, CL = 0.04, ke = 0.08"
# the reference (nlme) estimates, in the order of each objective's names
X_POP = np.array([1.593188, 0.643574, 0.039668, 0.166928, 0.085889, 0.709255])
X_POOLED = np.array([1.593188, 0.039668, 0.085889, 0.709255])


def ragged():
    """B1 in NONMEM form (dose rows EVID 1), subject i keeping 11 - (i mod 5) samples (the
    first and a spread of the rest); every third subject a second dose at 12 h."""
    d = pd.read_csv(B1 / "theoph.csv")
    rows = []
    for s, g in d.groupby("Subject"):
        g = g.sort_values("Time")
        keep = np.unique(np.linspace(0, len(g) - 1, len(g) - s % 5).round().astype(int))
        doses = [0.0, 12.0] if s % 3 == 0 else [0.0]
        for t in doses:
            rows.append(dict(Subject=s, Time=t, conc=np.nan, AMT=g.Dose.iloc[0], EVID=1))
        for _, r in g.iloc[keep].iterrows():
            rows.append(dict(Subject=s, Time=r.Time, conc=r.conc, AMT=np.nan, EVID=0))
    return pd.DataFrame(rows).sort_values(["Subject", "Time", "EVID"], ascending=[1, 1, 0])


CASES = [("laplace", POP, X_POP), ("optimize", POOLED, X_POOLED)]


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(("method", "estimates", "x"), CASES)
def test_padding_does_not_change_the_likelihood(tmp_path, method, estimates, x):
    data = ragged()
    shapes = {
        (int((g.EVID == 1).sum()), int((g.EVID == 0).sum())) for _, g in data.groupby("Subject")
    }
    assert len(shapes) > 3  # the units do differ in shape

    def m2ll(d, name):
        file = tmp_path / f"{name}.csv"
        d.to_csv(file, index=False)
        p = tf.Problem.from_text(MODEL + FIT.format(name=name, file=file, estimates=estimates,
                                                    method=method))  # fmt: skip
        return float(tf.objective(p)(x))

    together = m2ll(data, "all")
    alone = sum(m2ll(g, f"s{s}") for s, g in data.groupby("Subject"))
    print(
        f"\n{method}: all units at once {together:.6f}, one by one {alone:.6f}, "
        f"difference {together - alone:.2e}"
    )
    assert abs(together - alone) < 0.01
