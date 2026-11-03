# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""B3 over replicates: a generic phage infection-age toy (logistic bacteria, infected cells by
infection age, free phage) with synthetic true values. Each replicate simulates a new dataset
from the truth, fits it (method optimize, 4 starts around a point near the truth: B3 has
local optima, and one start from 20 % off can stop at one) and records the estimates, their
SEs and the doctor's findings, one row per replicate, appended to a CSV (so a run can stop and
resume).

Two designs, as the doctor's Fisher analysis at the truth showed (the phage curve is what
tells k and beta apart):

- ``both``: bacteria (counts below 100/mL kept as censored) and phage at 12 times;
- ``bacteria``: bacteria only, the same times.

The sharpness of the lysis hazard, s, is held at its true value (owner, 2026-11-03): samples
every 0.5 h cannot resolve it, and with s free most fits slide towards a step (s -> 0) until
the iteration limit, which leaves nothing to measure coverage on. ``--s-free`` estimates it
too, to check that the doctor flags it.

    python validation/b3/replicates.py both 200 --workers 12 --summary
    python validation/b3/replicates.py bacteria 100 --workers 12 --summary
    python validation/b3/replicates.py both 50 --s-free --workers 12 --summary

Acceptance (PRD Section 11, B3, two designs): with bacteria only, the k-beta ridge is reported
in at least 95 % of replicates; with both, no k-beta ridge, and the 90 % Wald intervals (log
scale) cover the truth within the binomial band (200 replicates: 0.845-0.955); with s free, s
is flagged (weakly determined or a ridge) in at least 95 %."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# several replicates run at once (--workers): a few threads each
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tenqsp as tq

import tenqsp_fit as tf

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).parent
MODEL = """model phage_toy
axis a in [0, 2] n=100 kind=one_way
param g=1, k=5e-8, beta=50, tau=0.5, s=0.05
param K=1e9, delta=0.1, hmax=30
h(a) = hmax/(1 + exp(-(a - tau)/s))
field I(a)
d_t(S) = g*S*(1 - S/K) - k*S*P
d_t(I) = -d_a(I) - h*I        using sharp
    at a=0: I = k*S*P
    at a=2: outflow
d_t(P) = beta*int_a(h*I) - k*S*P - delta*P
S(0) = 1e6
P(0) = 1e4
observable bact = S + int_a(I)
observable phage = P
"""
NAMES = ["g", "k", "beta", "tau", "s"]
TRUE = np.array([1.0, 5e-8, 50.0, 0.5, 0.05])
SIGMA, LOQ = 0.15, 100.0
T_OBS = np.arange(0.5, 6.01, 0.5)
OBSERVE = {
    "both": 'observe log10(bact) = DV where DVID == "bact"\n'
    'observe log10(phage) = DV where DVID == "phage"\n'
    "censor log10(bact) where BLQ == 1 below 2\n"
    "error log10(phage) ~ additive(fix(0.15))",
    "bacteria": 'observe log10(bact) = DV where DVID == "bact"\n'
    'ignore DV where DVID == "phage"\n'
    "censor log10(bact) where BLQ == 1 below 2",
}
_truth = None


def truth():
    global _truth
    if _truth is None:
        sim = tq.Model.from_text(MODEL).simulator(outputs=["bact", "phage"], rtol=1e-10,
                                                  atol=1e-8)  # fmt: skip
        out = sim.function(tq.Scenario(times=T_OBS), wrt=NAMES)(jnp.asarray(TRUE))
        _truth = np.asarray(out["bact"]), np.asarray(out["phage"])
    return _truth


def dataset(seed: int) -> pd.DataFrame:
    """One replicate: log10 counts with normal error (SD 0.15); bacteria below 100/mL as
    censored rows."""
    bact, phage = truth()
    rng = np.random.default_rng(seed)
    yb = np.log10(bact) + SIGMA * rng.normal(size=T_OBS.size)
    yp = np.log10(phage) + SIGMA * rng.normal(size=T_OBS.size)
    rows = [dict(TIME=t, DV=v if v > 2 else 2.0, DVID="bact", BLQ=int(v <= 2))
            for t, v in zip(T_OBS, yb, strict=True)]  # fmt: skip
    rows += [dict(TIME=t, DV=v, DVID="phage", BLQ=0) for t, v in zip(T_OBS, yp, strict=True)]
    return pd.DataFrame(rows).sort_values("TIME", kind="stable")


def problem(file, design: str, start, options: str, names) -> tf.Problem:
    est = ", ".join(f"{n} = {v:.6g}" for n, v in zip(names, start, strict=True))
    return tf.Problem.from_text(
        MODEL
        + f"""
fit b3_{design}
data "{file}"
{OBSERVE[design]}
estimate {est}
error log10(bact) ~ additive(fix(0.15))
method optimize
options {options} rtol=1e-8 progress=none
"""
    )


def replicate(design: str, seed: int, options: str, s_free: bool = False) -> dict:
    names = NAMES if s_free else NAMES[:-1]
    true = TRUE[: len(names)]
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "b3.csv"
        dataset(seed).to_csv(file, index=False)
        # started near the truth: each value off by about 20 % (log-normal), its own draw
        start = true * np.exp(0.2 * np.random.default_rng(10_000 + seed).normal(size=true.size))
        t0 = time.perf_counter()
        r = tf.fit(problem(file, design, start, options, names))
        seconds = time.perf_counter() - t0
    tab = r.table().set_index("quantity")
    row = dict(design=design, s_free=s_free, seed=seed, status=r.status, m2ll=r.m2ll,
               seconds=seconds, covariance=r.covariance_status)  # fmt: skip
    for n in names:
        row[n] = r.estimates[n]
        row[f"se_log {n}"] = float(tab.SE[n] / r.estimates[n])
    row["findings"] = json.dumps([dict(check=f.check, severity=f.severity,
                                       quantities=f.evidence.get("quantities", []))
                                  for f in r.doctor().findings])  # fmt: skip
    return row


def _ridge(pair):
    def found(findings: str) -> bool:
        return any(
            f["check"] == "ridge" and set(pair) <= set(f["quantities"])
            for f in json.loads(findings)
        )

    return found


_k_beta_ridge = _ridge(("k", "beta"))
_k_tau_ridge = _ridge(("k", "tau"))  # a delay traded against the infection rate


def _several_optima(findings: str) -> bool:
    return any(f["check"] == "several optima" for f in json.loads(findings))


def _flags_s(findings: str) -> bool:
    return any(
        f["check"] in ("ridge", "weakly determined") and "s" in f["quantities"]
        for f in json.loads(findings)
    )


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per design (and s free or held): replicates, converged, with standard errors, k-beta
    and k-tau ridges reported, the correlation of log k and log tau over the replicates,
    several optima reported, s flagged; per quantity the median log ratio to the truth and the
    90 % Wald coverage (over the replicates with a standard error)."""
    out = []
    for (design, s_free), d in df.groupby(["design", "s_free"]):
        names = NAMES if s_free else NAMES[:-1]
        row = dict(
            design=design,
            s_free=bool(s_free),
            replicates=len(d),
            converged=int((d.status == "converged").sum()),
            with_se=int((d.covariance == "ok").sum()),
            ridge_k_beta=float(d.findings.map(_k_beta_ridge).mean()),
            ridge_k_tau=float(d.findings.map(_k_tau_ridge).mean()),
            corr_log_k_tau=float(np.corrcoef(np.log(d.k), np.log(d.tau))[0, 1]),
            several_optima=float(d.findings.map(_several_optima).mean()),
            s_flagged=float(d.findings.map(_flags_s).mean()) if s_free else np.nan,
            median_seconds=float(d.seconds.median()),
        )
        for n, v in zip(names, TRUE, strict=False):
            lr = np.log(d[n] / v)
            se = d[f"se_log {n}"]
            ok = np.isfinite(se)
            row[f"bias {n}"] = float(np.median(lr))
            row[f"cover90 {n}"] = (
                float((np.abs(lr[ok]) <= 1.645 * se[ok]).mean()) if ok.any() else np.nan
            )
        out.append(row)
    return pd.DataFrame(out)


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("design", choices=sorted(OBSERVE))
    ap.add_argument("n", type=int)
    ap.add_argument("--s-free", action="store_true")
    ap.add_argument("--options", default="starts=4 checks=quick")  # B3 has local optima
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    tag = f"{a.design}{'_s_free' if a.s_free else ''}"
    out = Path(a.out or HERE / "results" / f"{tag}.csv")
    out.parent.mkdir(exist_ok=True)
    done = set(pd.read_csv(out).seed) if out.exists() else set()
    todo = [seed for seed in range(a.n) if seed not in done]
    with ProcessPoolExecutor(a.workers, mp_context=_spawn()) as pool:
        jobs = {pool.submit(replicate, a.design, seed, a.options, a.s_free): seed for seed in todo}
        for job in as_completed(jobs):
            row = job.result()
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{tag} {row['seed']}: {row['status']}, {row['seconds']:.0f} s", flush=True)
    if a.summary:
        print(summary(pd.read_csv(out)).T.to_string())


def _spawn():
    import multiprocessing

    return multiprocessing.get_context("spawn")  # JAX is not fork-safe


if __name__ == "__main__":
    main()
