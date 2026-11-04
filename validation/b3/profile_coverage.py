# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""B3 profile-interval coverage, design ``both`` (s held at its true value): each replicate is
simulated and fitted as in ``replicates.py`` (same data, same four starts), then, for k and
tau at the 90 % level:

- the profile interval ``result.profile`` walks out from the estimate;
- the profile-likelihood test at the truth: the quantity held at its true value and the rest
  refitted (four starts); covered when -2LL there is within 2.71 of the best fit (chi-square,
  1 df, 90 %). The walk cannot cross a hump in the profile higher than the cut-off, so where
  the set has two pieces only this test sees the far one;
- the Wald interval (log scale).

Why: on this design the likelihood can have two optima, near the truth and at tau ~0.9 with
k ~6x, often within 1-2 of each other in -2LL. A Wald interval sits at one of them; the
profile-likelihood set (-2LL within 2.71 of the best) can contain both. Acceptance as for the
Wald intervals: coverage within the 99 % binomial band for the number of replicates (50:
0.78-1.00).

    python validation/b3/profile_coverage.py 50 --workers 12 --summary"""

from __future__ import annotations

import argparse
import os
import sys
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
from scipy.stats import binom

import tenqsp_fit as tf

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import replicates as R  # noqa: E402

PROFILED = ["k", "tau"]
LEVEL = 0.90
CUT = 2.705543454095404  # chi-square, 1 df, 90 %


def replicate(seed: int) -> dict:
    names = R.NAMES[:-1]
    true = R.TRUE[: len(names)]
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "b3.csv"
        R.dataset(seed).to_csv(file, index=False)
        start = true * np.exp(0.2 * np.random.default_rng(10_000 + seed).normal(size=true.size))
        t0 = time.perf_counter()
        r = tf.fit(R.problem(file, "both", start, "starts=4 checks=quick", names))
        tab = r.table().set_index("quantity")
        row = dict(seed=seed, status=r.status, m2ll=r.m2ll)
        z = 1.6448536269514722
        for n in names:
            row[n] = r.estimates[n]
        for n in PROFILED:
            # the truth held, the rest refitted from the same starts
            rest = [m for m in names if m != n]
            held = tf.fit(R.problem(file, "both", start[[names.index(m) for m in rest]],
                                    "starts=4 checks=quick", rest))  # fmt: skip
            row[f"{n} held m2ll"] = held.m2ll
            row[f"{n} lr"] = held.m2ll - min(r.m2ll, held.m2ll)
            lo, hi = r.profile(n, level=LEVEL).interval
            se = float(tab.SE[n] / r.estimates[n])
            row.update({f"{n} lower": lo, f"{n} upper": hi,
                        f"{n} wald lower": r.estimates[n] * np.exp(-z * se),
                        f"{n} wald upper": r.estimates[n] * np.exp(z * se)})  # fmt: skip
        row["seconds"] = time.perf_counter() - t0
    return row


def summary(df: pd.DataFrame) -> pd.DataFrame:
    n_rep = len(df)
    band = (binom.ppf(0.005, n_rep, LEVEL) / n_rep, binom.ppf(0.995, n_rep, LEVEL) / n_rep)
    true = dict(zip(R.NAMES, R.TRUE, strict=True))
    rows = []
    for n in PROFILED:
        t = true[n]
        prof = (df[f"{n} lower"] <= t) & (t <= df[f"{n} upper"])
        test = df[f"{n} lr"] <= CUT
        wald = (df[f"{n} wald lower"] <= t) & (t <= df[f"{n} wald upper"])
        closed = np.isfinite(df[f"{n} lower"]) & np.isfinite(df[f"{n} upper"])
        rows.append(dict(quantity=n, replicates=n_rep, closed=int(closed.sum()),
                         profile_walk=float(prof.mean()), profile_test=float(test.mean()),
                         wald=float(wald.mean()),
                         band=f"{band[0]:.3f}-{band[1]:.3f}",
                         second_optimum=int((df.tau > 0.75).sum()),
                         accept=bool(band[0] <= test.mean() <= band[1])))  # fmt: skip
    return pd.DataFrame(rows)


def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", help="these seeds instead of 0..n-1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or HERE / "results" / "profile_coverage_both.csv")
    out.parent.mkdir(exist_ok=True)
    done = set(pd.read_csv(out).seed) if out.exists() else set()
    todo = [s for s in (a.seeds or range(a.n)) if s not in done]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for job in as_completed([pool.submit(replicate, s) for s in todo]):
            row = job.result()
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"replicate {row['seed']}: {row['status']}, {row['seconds']:.0f} s", flush=True)
    if a.summary:
        print(summary(pd.read_csv(out)).to_string(index=False))


if __name__ == "__main__":
    main()
