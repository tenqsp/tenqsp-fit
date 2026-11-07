# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Profile-interval coverage: one subject of the theophylline design
(subject 1's dose and sampling times), simulated many times from fixed true values (ka 1.5,
CL 0.04, ke 0.08, additive residual SD 0.7, near the population fit), each replicate fitted
(method optimize, the residual SD estimated too) and profiled for ka, CL and ke at the 90 %
level. Acceptance: each interval covers the truth in 0.845-0.955 of 200 replicates (the 99 %
binomial band); the Wald intervals (log scale) are recorded alongside.

    python validation/theophylline/profile_coverage.py 200 --workers 12 --summary"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import jax
import numpy as np
import pandas as pd

import tenqsp_fit as tf

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).parent
NAMES = ["ka", "CL", "ke"]
TRUE = dict(ka=1.5, CL=0.04, ke=0.08)
SD = 0.7
LEVEL = 0.90
MODEL = """model theoph
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""


def design():
    d = pd.read_csv(HERE / "theoph.csv")
    s1 = d[d.Subject == 1]
    return float(s1.Dose.iloc[0]), s1.Time.to_numpy()


def conc(t, dose, ka, CL, ke):
    return dose * ka * ke / (CL * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def replicate(seed: int) -> dict:
    dose, times = design()
    rng = np.random.default_rng(seed)
    y = conc(times, dose, **TRUE) + SD * rng.normal(size=times.size)
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "one.csv"
        pd.DataFrame(dict(Subject=1, Dose=dose, Time=times, conc=y)).to_csv(file, index=False)
        p = tf.Problem.from_text(
            MODEL
            + f"""
fit theoph_one
data "{file}" with ID=Subject, TIME=Time, DV=conc
unit ID
doses Dose at t=0 into depot
observe conc = DV
estimate ka = 1.5, CL = 0.04, ke = 0.08
error conc ~ additive(1)
method optimize
options starts=2 checks=quick progress=none
"""
        )
        t0 = time.perf_counter()
        r = tf.fit(p)
        tab = r.table().set_index("quantity")
        row = dict(seed=seed, status=r.status)
        z = 1.6448536269514722
        for n in NAMES:
            prof = r.profile(n, level=LEVEL)
            lo, hi = prof.interval
            se = float(tab.SE[n] / r.estimates[n])
            row.update({n: r.estimates[n], f"{n} lower": lo, f"{n} upper": hi,
                        f"{n} wald lower": r.estimates[n] * np.exp(-z * se),
                        f"{n} wald upper": r.estimates[n] * np.exp(z * se)})  # fmt: skip
        row["seconds"] = time.perf_counter() - t0
    return row


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in NAMES:
        t = TRUE[n]
        prof = (df[f"{n} lower"] <= t) & (t <= df[f"{n} upper"])
        wald = (df[f"{n} wald lower"] <= t) & (t <= df[f"{n} wald upper"])
        closed = np.isfinite(df[f"{n} lower"]) & np.isfinite(df[f"{n} upper"])
        rows.append(dict(quantity=n, replicates=len(df), closed=int(closed.sum()),
                         profile_coverage=float(prof.mean()), wald_coverage=float(wald.mean()),
                         median_ratio=float(np.median(df[n] / t)),
                         accept=bool(0.845 <= prof.mean() <= 0.955)))  # fmt: skip
    return pd.DataFrame(rows)


def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or HERE / "results" / "profile_coverage.csv")
    out.parent.mkdir(exist_ok=True)
    done = set(pd.read_csv(out).seed) if out.exists() else set()
    todo = [s for s in range(a.n) if s not in done]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for job in as_completed([pool.submit(replicate, s) for s in todo]):
            row = job.result()
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"replicate {row['seed']}: {row['status']}, {row['seconds']:.0f} s", flush=True)
    if a.summary:
        print(summary(pd.read_csv(out)).to_string(index=False))


if __name__ == "__main__":
    main()
