# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Simulation-based calibration of ``method nuts`` (PRD F13.1, small replicate counts in 0.1):
one subject of the theophylline design (subject 1's dose and times), one compartment with
first-order absorption, lognormal priors on ka, CL and ke, the residual SD known (0.7).

Each replicate draws the truth from the prior, simulates data, samples the posterior (2 chains
x (300 warmup + 250 draws)) and records the rank of the truth among 99 evenly spaced draws. If
the sampler draws the posterior, every rank is uniform on 0..99 (Talts et al. 2018).
Acceptance (PRD Section 11): a chi-square test of uniformity over 10 bins, p > 0.01 / 3 per
parameter (Bonferroni); divergences and R-hat recorded.

    python validation/sbc/sbc_pk.py 100 --workers 12 --summary"""

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
PRIOR = dict(ka=(1.5, 0.3), CL=(0.04, 0.3), ke=(0.08, 0.3))  # median, sdlog
SD = 0.7
L = 99  # draws ranked against the truth
MODEL = """model theoph
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
"""


def design():
    d = pd.read_csv(HERE.parent / "b1" / "theoph.csv")
    s1 = d[d.Subject == 1]
    return float(s1.Dose.iloc[0]), s1.Time.to_numpy()


def conc(t, dose, ka, CL, ke):
    return dose * ka * ke / (CL * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def replicate(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    truth = {n: m * np.exp(s * rng.normal()) for n, (m, s) in PRIOR.items()}
    dose, times = design()
    y = conc(times, dose, **truth) + SD * rng.normal(size=times.size)
    priors = "".join(f"prior {n} ~ lognormal(median={m}, sdlog={s})\n"
                     for n, (m, s) in PRIOR.items())  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "one.csv"
        pd.DataFrame(dict(Subject=1, Dose=dose, Time=times, conc=y)).to_csv(file, index=False)
        p = tf.Problem.from_text(
            MODEL
            + f"""
fit sbc
data "{file}" with ID=Subject, TIME=Time, DV=conc
unit ID
doses Dose at t=0 into depot
observe conc = DV
estimate ka = 1.5, CL = 0.04, ke = 0.08
{priors}error conc ~ additive(fix({SD}))
method nuts chains=2 warmup=300 samples=250
options seed={seed} checks=quick progress=none
"""
        )
        t0 = time.perf_counter()
        r = tf.fit(p)
        seconds = time.perf_counter() - t0
    d = r.draws
    pick = np.linspace(0, len(d) - 1, L).round().astype(int)
    tab = r.table().set_index("quantity")
    row = dict(seed=seed, seconds=seconds, divergences=int(d.divergent.sum()),
               rhat_max=float(tab["R-hat"].max()))  # fmt: skip
    for n in NAMES:
        row[f"true {n}"] = truth[n]
        row[f"rank {n}"] = int((d[n].to_numpy()[pick] < truth[n]).sum())
    return row


def summary(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import chisquare

    rows = []
    for n in NAMES:
        counts = np.bincount(np.minimum(df[f"rank {n}"].to_numpy() * 10 // (L + 1), 9),
                             minlength=10)  # fmt: skip
        p = float(chisquare(counts).pvalue)
        rows.append(dict(quantity=n, replicates=len(df), bins=" ".join(map(str, counts)),
                         p=p, accept=p > 0.01 / len(NAMES)))  # fmt: skip
    out = pd.DataFrame(rows)
    print(
        f"divergences: {int(df.divergences.sum())} in all; R-hat at most "
        f"{df.rhat_max.max():.3f}; median {df.seconds.median():.0f} s per replicate"
    )
    return out


def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or HERE / "results" / "sbc_pk.csv")
    out.parent.mkdir(exist_ok=True)
    done = set(pd.read_csv(out).seed) if out.exists() else set()
    todo = [s for s in range(a.n) if s not in done]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for job in as_completed([pool.submit(replicate, s) for s in todo]):
            row = job.result()
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"replicate {row['seed']}: {row['seconds']:.0f} s", flush=True)
    if a.summary:
        print(summary(pd.read_csv(out)).to_string(index=False))


if __name__ == "__main__":
    main()
