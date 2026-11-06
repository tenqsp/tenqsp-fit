# ADR 0004 — Profile likelihood: the cut-off and the scan

- **Status:** Accepted by the project lead, 2026-11-05.
- **Code:** `src/tenqsp_fit/results.py` (`Profile`), `src/tenqsp_fit/optimize.py`
  (`Result.profile`), `src/tenqsp_fit/doctor.py`.

## Context

`result.profile(name)` holds one quantity at values on each side of its estimate, refits the
rest, and gives the interval where −2LL stays within a cut-off of its minimum. Two problems
showed in validation over synthetic replicates:

1. **Coverage too low with few measurements.** Theophylline, `method optimize` per subject
   (11 measurements, 3 parameters, residual SD estimated), 200 replicates, nominal 0.90:
   ka 0.805, CL 0.845, ke 0.855, the same with plain SciPy least squares. The χ²₁ cut-off is
   an asymptotic result.
2. **A walk misses separate pieces.** On the infection-age model the likelihood has a second
   optimum (tau about 0.9, k about 6 times the truth). The likelihood-ratio set at the truth
   covers k 0.80 and tau 0.86, but a walk outward from the estimate stops at the hump between
   the optima: 0.64 and 0.76.

## Options considered

- For 1: keep χ²₁; the F-based small-sample cut-off of nonlinear regression (Bates & Watts
  1988); a bootstrap (expensive).
- For 2: walk only and warn; **scan past the walk** on a wide grid; a global search per point.

## Decision

- **Cut-off:** for `method optimize` with an error-model quantity estimated and no written
  likelihood, `n log(1 + F(level; 1, n − p)/(n − p))`, with n the measured (not censored)
  values and p the estimated quantities other than the error model's; χ²₁ otherwise. With
  it, the theophylline coverage is 0.885 / 0.90 / 0.92. `Profile.cut` and `Profile.rule` say
  which applied.
- **Scan:** after the walk (default 8 points each way, step SE/2), scan past each end to 1/100
  and 100 times the estimate on the log scale (8 points), refitting each point from the
  nearest point already fitted and from other starts; where the scan dips, golden-section search finds the bottom, and each
  end of a piece is refined by four halvings. The result lists the `pieces`; `interval` spans
  all of them. On by default; `scan=False` leaves it out.
- **Doctor:** "profile in pieces" (warning) when the set is not one interval; "better fit in
  the profile" (problem) when a point of the profile is lower than the fit itself.

## Consequences

- A profile with the scan costs about 3–4 times one without (theophylline ke: about
  3 minutes against 50 s).
- A second optimum within the cut-off is reported instead of silently missed; one outside it
  is drawn on the profile plot (theophylline's flip-flop fit at ke ≈ 1.5).
