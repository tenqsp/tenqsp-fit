# ADR 0003 — The defaults of `method nuts`

- **Status:** Accepted 2026-11-06; in use since 0.1 (2026-11-02).
- **Code:** `src/tenqsp_fit/nuts.py`, `src/tenqsp_fit/priors.py`.

## Context

`method nuts` draws from the posterior with the No-U-Turn Sampler (NumPyro), using TenQSP's
exact gradients. Each gradient is an ODE or PDE solve with sensitivities, so the number of
steps per draw sets the cost. Population models put each unit's random effects in the
posterior too.

## Options considered

1. NUTS with NumPyro's default adaptation (diagonal mass matrix learned in warmup windows).
2. **A dense mass matrix from the curvature at the fast fit**, held through the run.
3. Centred or **non-centred** random effects.

## Measurements

- Theophylline: CL and ke move together, 7 times narrower together than apart. With the
  default adaptation NUTS took 15 times more steps per draw than with the mass matrix from
  the fast fit.
- Simulation-based calibration on the same model (200 runs): rank uniformity p 0.32 / 0.52 /
  0.46, no divergences, R-hat at most 1.035.

## Decision

- **First the fast fit** (`laplace` when parameters vary, else `optimize`) with the written
  priors; then the curvature of the potential there (central differences of the exact
  gradient). Its inverse is the **mass matrix**, held through the run; the chains start one
  curvature-SD from the fast fit in random directions.
- **Warmup** tunes only the step size (dual averaging, target acceptance 0.8); maximum tree
  depth 10.
- **Non-centred** random effects: standard-normal scores per unit, correlated through the
  Cholesky factor of the correlation matrix.
- All chains run as one batched computation, in chunks of 10 draws, between which progress
  is shown and Ctrl-C keeps the draws so far.
- **Convergence reported** per quantity: rank-normalized split R-hat, bulk and tail ESS
  (ArviZ), divergences, energy BFMI, and how often the tree depth was reached.

## Consequences

- Runs are much shorter on correlated posteriors; a posterior very different in shape from the
  curvature at the fast fit (strongly non-Gaussian, several modes) is sampled less well, and
  the convergence report is where that shows.
- The fast fit's result is available before sampling starts.
