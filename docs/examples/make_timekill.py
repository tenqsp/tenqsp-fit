# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Simulate the time-kill data of the quickstart: log10 counts of bacteria and phage every
half hour for 6 h, with normal error (SD 0.15), bacteria below 100/mL reported as censored."""

import numpy as np
import pandas as pd
import tenqsp as tq

model = tq.load("infection_age.tq")  # the model part; the fit part is ignored here
times = np.arange(0.5, 6.01, 0.5)
sim = model.simulator(outputs=["bact", "phage"], rtol=1e-10, atol=1e-8)  # tight: counts near 0
run = sim(tq.Scenario(times=times))  # at the model's own values
rng = np.random.default_rng(7)
rows = []
for kind in ("bact", "phage"):
    for t, y in zip(times, np.log10(run[kind]) + 0.15 * rng.normal(size=times.size)):
        below = kind == "bact" and y <= 2
        rows.append(dict(TIME=t, DV=2.0 if below else round(y, 3), DVID=kind, BLQ=int(below)))
pd.DataFrame(rows).sort_values("TIME", kind="stable").to_csv("timekill.csv", index=False)
