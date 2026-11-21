# ADR 0006 — Bounds kept as bounds: L-BFGS-B and trust-region least squares

- **Status:** Accepted 2026-11-21 (owner).
- **Supersedes:** ADR 0001's optimizers (Optimistix Levenberg–Marquardt and BFGS), its
  estimation scales for bounded quantities (the logit within `(lo, hi)`, `lo + exp(u)` above
  one bound) and its step cap; ADR 0002's outer optimizer (SciPy BFGS on the same scales).
- **Code:** `src/tenqsp_fit/optimize.py` (`fit_scales`, `box`, `minimize_bounded`,
  `run_bounded`, `fit_optimize`), `laplace.py` (`fit_laplace`), `held.py` (profiles),
  `priors.py` (default priors on the statistical scales), `doctor.py` (ridges).

## Context

Profiling slow population and pooled fits of stiff models found three causes in the
optimizers, none in the model runs:

1. **Fits ended at false bounds.** On the logit scale a quantity pushed near a bound has a
   derivative near 0 there and cannot come back. In one pooled fit (3 bounded quantities, 2
   error terms) the fit ended with a fraction on its upper bound 1, with a near-bound warning;
   with the bounds kept as bounds (L-BFGS-B, the same objective) the optimum was 4.3 lower in
   −2LL with the fraction at 0.7, confirmed at rtol 1e-6, 1e-8 and 1e-10 and by a likelihood
   written independently of tenqsp-fit. In a larger version of the same fit, 15 of 16 starts
   ended at the same false optimum, 20.7 above the one a single bounded start reached.
2. **The stopping rule could not be met.** Tolerances of 1e-8 on −2LL and on the parameters
   (estimation scale), against an objective that a stiff model at rtol 1e-6 carries an error
   of order 1 in: a quantity drifting toward a bound (its estimation-scale value toward
   infinity) or along a nearly flat direction kept the fit going. In the larger fit, 65 % of
   the time lowered −2LL by less than 1.
3. **Rejected steps cost two runs:** Optimistix's backtracking (halving only, with no scaling
   of the first inverse Hessian) rejected 51 of 89 steps in the smaller fit, and each rejected
   point was run again to ask whether the model ran there (18 % of the fit).

## Options considered

1. Keep the logit scale, add a stopping rule: the false bounds remain.
2. A projected or bounded quasi-Newton method keeping the bounds: SciPy's L-BFGS-B (a
   standard, long-used implementation with a line search that interpolates), and for sums of
   squares SciPy's trust-region reflective method (`least_squares`, `method="trf"`), which
   keeps bounds the same way.
3. A quasi-Newton method of our own: no gain over (2) measured.

## Decision

1. **Fit scales keep bounds as bounds** (`fit_scales`): a quantity with bounds is estimated
   on the log scale when its lower bound is 0 or above (a distribution's family may set the
   scale), else as it is, with the bounds (on that scale) given to the optimizer. Quantities
   without bounds: as before (log for a positive start, else as is).
2. **`method optimize`:** L-BFGS-B on −2LL (projected gradient 1e-3), or trust-region least
   squares where −2LL is a sum of squares (tolerances 1e-8). **`method laplace`** (the outer
   fit) and **profiles** use the same function (`minimize_bounded`).
3. **A start also ends when its last 5 steps lowered −2LL by less than 1e-4 in all**: finer
   than any comparison of fits needs, and the integration error of −2LL is a bias that
   changes little between nearby values, so the rule ends where −2LL stops moving.
4. **No second run of a rejected point.** A point whose run fails, or whose derivatives are
   not finite, is counted as a failed run where it is evaluated. A start whose derivatives
   are not finite is drawn again, as one whose model does not run.
5. **No step cap** (L-BFGS-B's line search and the trust region bound the steps).
6. **Default priors and NUTS keep the statistical scales** (`statistical`: the logit within
   bounds), so a prior means what it meant.
7. **The doctor's ridges leave out quantities at a bound**: an active bound is a result, not
   a measurement gap (a fraction at its bound 1 had been reported as "not determined,
   curvature ratio 0", where −2LL rose by 10 over the last 30 % of its range).

## Measurements

| | Optimistix BFGS, logit scales | L-BFGS-B, bounds kept |
|---|---:|---:|
| pooled fit, −2LL | the fraction on its bound 1 | 4.3 lower, the fraction at 0.7 |
| larger fit, 16 starts | 15 at a false optimum, 11 h | one start reached the optimum in 8 min |

## Consequences

- Fits may end at a different (lower) −2LL than before, and estimates near a bound change.
  The CHANGELOG says so; the tutorials' printed results were rerun.
- `result.scales` and the summary's `scales` line show the fit scales
  (`log scale, bounded: ...`).
- The result's optimizer is `"L-BFGS-B"`, `"trust-region least squares"`, or
  `"L-BFGS-B, FOCE-I"`.
- tenqsp-fit needs SciPy 1.16 or later (the least-squares callback).
