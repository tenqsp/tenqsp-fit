# ADR 0002 — The defaults of `method laplace`

- **Status:** Proposed, in use since 0.1 (2026-11-01).
- **Code:** `src/tenqsp_fit/laplace.py`.

## Context

`method laplace` fits populations: parameters vary between units with an estimated
distribution. Pharmacometricians expect NONMEM's FOCE-I objective and its standard errors,
and results that agree with NONMEM, nlme and Monolix on standard data. TenQSP gives exact
first derivatives of the solution, not yet second derivatives.

## Options considered

1. **FOCE-I by the Laplace approximation at the conditional modes**, outer fit by BFGS.
2. FO (linearization at zero random effects): fast, known bias.
3. SAEM (Monolix's method): stochastic, needs a different set of diagnostics.
4. For the outer gradient: the first-order "FOCE gradient" with the solver's sensitivities
   held fixed, or the exact gradient through the implicit function theorem.

## Measurements

- Theophylline (12 subjects): the first-order gradient stops at −2LL 354.04, the
  Lindstrom–Bates point of R's `nlme`; the exact gradient reaches 353.98, with estimates
  within 2 % of `nlme`'s.
- Profile-interval coverage on the same model over 200 synthetic replicates (CL 0.845,
  ke 0.855, ka 0.79 at a nominal 0.90): the ka shortfall is that of the likelihood-ratio
  interval itself with 11 measurements per subject, not of the fit (see ADR 0004).

## Decision

- **Every distribution through normal scores:** a varying parameter is a transform of a
  standard-normal score per unit (lognormal: `median·exp(sdlog·z)`), so one method serves all
  distributions, and correlated lognormals are NONMEM's OMEGA block.
- **The objective:** −2LL per unit by Laplace's method at the conditional mode (NONMEM's EBE),
  with the Fisher information of each prediction including the error model's dependence on it
  (FOCE with interaction); M3 terms for censored values.
- **The inner fit:** Newton steps for all units' modes at once, warm-started from the last
  modes, falling back to a Fisher-scoring step when the Hessian is not positive definite.
- **The outer fit:** BFGS (SciPy) on the estimation scales of `method optimize`, with the
  **exact** gradient: implicit-function-theorem terms, each a difference of exact first
  derivatives at fixed modes (option 4, second part).
- **Standard errors** from the Hessian of −2LL by central differences of that gradient
  (NONMEM's R matrix); intervals on the estimation scale carried over. **Shrinkage:**
  1 − SD of the modes' scores (NONMEM's SD-based ETA shrinkage).
- No FO, no SAEM in 0.1.

## Consequences

- −2LL is comparable with NONMEM's OFV after adding n log 2π.
- Each outer step costs 2p + 2q extra model evaluations (p estimated quantities, q varying
  parameters), batched into one call.
- When TenQSP gains second derivatives of the solution, the differences can be replaced.
