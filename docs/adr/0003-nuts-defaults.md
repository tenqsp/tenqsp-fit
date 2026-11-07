# ADR 0003 — The defaults of `method nuts`

- **Status:** Accepted 2026-11-06; amended 2026-11-07 (chains in threads, acceptance target
  0.9 and the `target_accept` option).
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
4. The chains batched into one computation (`jax.vmap`), or **each chain in its own thread**.
5. Acceptance target 0.8 (Stan's and NumPyro's default), **0.9**, or 0.95.
6. Mass matrix held from the fast fit, or adapted further during warmup.

## Measurements

- Theophylline: CL and ke move together, 7 times narrower together than apart. With the
  default adaptation NUTS took 15 times more steps per draw than with the mass matrix from
  the fast fit.
- Simulation-based calibration on the same model (200 runs): rank uniformity p 0.32 / 0.52 /
  0.46, no divergences, R-hat at most 1.035.
- The theophylline population (12 subjects, 4 chains of 1000 warmup and 1000 draws), all runs
  giving the same posterior:

  | Chains | Target | Time | Divergences | R-hat max | ESS min |
  |---|---|---|---|---|---|
  | batched | 0.8 | 46 min | 41 | 1.012 | 252 |
  | threads | 0.8 | 19 min | 18 | 1.009 | 512 |
  | threads | 0.9 | 21-22 min | 3 | 1.005 | 948 |

  Batched, the chains share one core and every step waits for the chain with the longest
  trajectory; the steps per draw were the same (about 15) either way. Shorter runs (300 warmup
  and 300 draws): target 0.95 gave no divergences but half the effective sample size of 0.9
  (more steps per draw); adapting the mass matrix during warmup took 2.5 times as long for
  no clear gain.

## Decision

- **First the fast fit** (`laplace` when parameters vary, else `optimize`) with the written
  priors; then the curvature of the potential there (central differences of the exact
  gradient). Its inverse is the **mass matrix**, held through the run; the chains start one
  curvature-SD from the fast fit in random directions.
- **Warmup** tunes only the step size (dual averaging) to the acceptance target
  `target_accept`, **0.9** by default; the user can raise it (`method nuts target_accept=0.95`),
  the first remedy the doctor gives for divergences. Maximum tree depth 10.
- **Non-centred** random effects: standard-normal scores per unit, correlated through the
  Cholesky factor of the correlation matrix.
- **Each chain runs in its own thread**, in chunks of 10 draws; between chunks progress is
  shown, and a stop (Ctrl-C or the time limit) keeps the draws every chain has.
- **Convergence reported** per quantity: rank-normalized split R-hat, bulk and tail ESS
  (ArviZ), divergences, energy BFMI, and how often the tree depth was reached.

## Consequences

- Runs are much shorter on correlated posteriors; a posterior very different in shape from the
  curvature at the fast fit (strongly non-Gaussian, several modes) is sampled less well, and
  the convergence report is where that shows.
- The fast fit's result is available before sampling starts.
- A run uses as many cores as chains. The default target is stricter than Stan's, so steps
  are smaller and a run takes about 10 % longer than at 0.8, for fewer divergences and
  a larger effective sample size.
