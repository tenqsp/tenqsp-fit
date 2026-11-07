# ADR 0001 — The defaults of `method optimize`

- **Status:** Accepted 2026-11-06; in use since 0.1 (2026-11-01).
- **Code:** `src/tenqsp_fit/optimize.py`, `src/tenqsp_fit/likelihood.py`.

## Context

`method optimize` fits one set of parameters to all the data, by maximum likelihood (least
squares when the error model is fixed). The models are ODEs and PDEs solved numerically,
often stiff, with parameters spanning orders of magnitude (an adsorption rate of 10⁻⁸, a burst
size of 50). TenQSP gives exact first derivatives of every output. The choices: the
optimizer, the scale each quantity is estimated on, and how many starts.

## Options considered

1. **One optimizer for everything** (BFGS on −2LL), simple.
2. **Levenberg–Marquardt** where −2LL is a sum of squares, BFGS elsewhere.
3. A global optimizer (differential evolution, CMA-ES) first, then a local one.

## Measurements

- Theophylline, pooled (one set of parameters for 12 subjects): with the residual SD fixed,
  Levenberg–Marquardt converges in 18 steps; with the SD estimated, 48–61 steps, because the
  SD's terms of −2LL do not vanish at the minimum, so the Gauss–Newton model of the curvature
  is poor. BFGS with the exact gradient does not have that problem.
- The infection-age model of the QSP quickstart, 8 starts: one thread per start ran 7 times faster than one
  batched (`jax.vmap`) call, which on the CPU runs the starts one after the other.
- The same model over 200 synthetic replicates: in 33 the best fit found sat at a second optimum; in 1 of
  6 refitted from the truth, the starts missed a lower optimum near the truth. Several starts
  matter; one start is never enough to know.

## Decision

- **Levenberg–Marquardt** (Optimistix) with exact Jacobians when every error model is
  additive with a known sd: −2LL is then a sum of squares up to a constant, censored values
  included as `sqrt(−2 log Φ((limit − f)/sd))`. **BFGS** with the exact gradient otherwise:
  when an error-model quantity is estimated, and when the sd changes with the prediction
  (proportional, combined, power), whose `2 log sd` terms change with the parameters.
  *Amended 2026-11-07:* the first version used Levenberg–Marquardt for every fixed error
  model, which for a proportional one minimized a weighted sum of squares, not −2LL (on a
  3-subject oral PK example, CL 8 % low and −2LL 1.5 above the minimum).
- **Scales:** a quantity with two bounds on the logit scale within them, one with a bound
  on one side as `lo + exp(u)` or `hi - exp(u)`, a positive start on the log scale, otherwise
  as written. The summary lists each scale.
- **Starts:** start 1 is the values written; the others are drawn around them (log scale:
  95 % within a factor 5 either way), with a seed; a start whose model run fails is drawn
  again (up to 20 times). Default 1 start (the user's values), with the doctor's note that
  other optima were not looked for, and `options starts=` to look.
- **Iterations:** at most 200 by default (`options maxiter=`); stopping there is reported as
  a problem, not hidden.
- No global stage in 0.1 (option 3): its cost on an ODE model is many thousands of solves,
  and the multi-start with the doctor's "several optima" finding covers the common case.

## Consequences

- Fits with a fixed error model converge in few steps; with an estimated one, BFGS is the
  safer choice at a modest cost.
- Results always report −2LL, whichever optimizer ran.
- A global stage remains a candidate for a later version.
- Where two optima are close in −2LL (within the profile cut-off, 3.84), the doctor's
  "several optima" warning advises a profile interval: in simulation studies of the
  infection-age model, Wald intervals at the best fit covered the truth less often than
  nominal when the data narrowly favoured a second optimum.
