# ADR 0005 — A faster `method laplace`: one compile, an exact gradient, one run per inner step

- **Status:** Accepted 2026-11-21 (owner).
- **Amends:** ADR 0002 (the inner Newton step with a difference Hessian; the outer gradient
  by central differences).
- **Code:** `src/tenqsp_fit/laplace.py`, `doctor.py` (`_tolerance`), `held.py`,
  `likelihood.py`, `_jax_setup.py`, `problem.py` (option `cache`).
- **Related:** TenQSP ADR 0010 (stacked batches), which makes each run of the units cheaper.

## Context

A Friberg myelosuppression population problem (maturation as an axis; 20 simulated
patients, 160 measurements, 7 estimates, 2 varying parameters, lognormal variability,
proportional error) took 276 s with tenqsp-fit 0.2 and TenQSP 0.8.3 (one start, standard
errors and checks included), far more than a model of 6 states should need. Measured causes:

1. **Four compiles of the same objective**, 22–24 s each: the value, the gradient, the modes
   (`Laplace.modes`), and the doctor's tolerance check, which built a new `jax.jit` of the
   gradient although the tolerances are an input so that the check can reuse the compiled
   function (a bug).
2. **Many runs per evaluation:** the inner Newton step took 3 runs of the units (the gradient
   by the scores and 2 for its difference Hessian), and the outer gradient 2p + 2q = 18
   batched evaluations.
3. **XLA's threads:** spreading these small array operations over many threads made each run
   slower (1.2–1.7 times, measured on one core against many).

## Options considered

1. Second derivatives of the predictions for the gradient: by central differences of the
   sensitivities (each perturbed copy of the units a run with its own steps), or exact,
   forward mode over the solver's forward sensitivities.
2. The inner fit: exact Newton (no fewer runs in total: measured), a chord method (fails far
   from the modes: measured), Fisher scoring with one run per iteration.
3. The compile: one function for value, modes and gradient, with the gradient's part run only
   when asked (`lax.cond`), against separate functions.

## Measurements (Friberg, this machine, released TenQSP 0.8.x behaviour, vmap batches)

| | tenqsp-fit 0.2 | this record |
|---|---:|---:|
| Whole fit (one start, standard errors, checks) | 272 s | 203 s |
| Compiles of the objective | 4 (~90 s) | 1 (~23 s; ~2 s from the cache) |
| BFGS evaluations | 45 | 47 |
| −2LL | 296.638328 | 296.638328 |

Estimates and 95 % intervals agree to the digits printed (MTT 102.692, 98.907–106.622).

- Second derivatives by differences of the sensitivities: the gradient at the optimum was off
  by up to 2e-3 at rtol 1e-6 (each copy takes its own steps, so the differences carry the
  solver's error divided by the step), above BFGS's tolerance (1e-3): BFGS then repeated
  ~20 evaluations at the optimum. Exact second derivatives: 445 ms against 838 ms for the
  copies, and BFGS ended cleanly.
- Fisher scoring takes 10–22 iterations far from the optimum and 1–3 near it, each one run
  of the units (~70 ms here). An inner tolerance of 1e-6 instead of 1e-8 took a third fewer
  iterations but the fit failed (−2LL 442): kept at 1e-8.
- What remains is mostly runs of the units: one run of the 20 patients costs ~70 ms in a
  `jax.vmap` batch, against ~10 ms for the same patients as one stacked system (TenQSP ADR
  0010; prototype outside the repository: the whole fit in 26 s, 14.5 s with the cache).

## Decision

1. **The inner fit: Fisher scoring, one run per iteration** (`H = J'WJ + R^-1`, the trial
   point's run giving its value, its gradient by the scores and H together), step halving per
   unit; at most 100 iterations. The difference Hessian goes.
2. **The inner fit starts from the modes the implicit function theorem predicts**,
   `z^ + (dz^/du) du`, in the fit, the standard errors and the profiles.
3. **The exact gradient.** At the modes, one run carries the units' sensitivities and their
   derivatives (forward over forward), so each unit's predictions are an exact quadratic in
   its model parameters; −2LL's gradient by the implicit function theorem is computed by
   automatic differentiation through that quadratic. Where the solver gives no second
   derivatives (through events; TenQSP's sparse tier) or the model has many parameters set
   by the fit (w + 1 > 2p + 2q), the gradient by differences of ADR 0002 stays.
4. **One compiled function** gives −2LL, the modes and, when asked, the gradient and the
   modes' derivative, with the tolerances as an input; the fit, the standard errors,
   `obj(x)`, `obj.modes`, the doctor and the profiles call it. (`Laplace.evaluate`.)
5. **JAX's persistent compilation cache**, on by default in `~/.cache/tenqsp-fit/jax`
   (owner-only permissions: JAX runs what it finds there) or in `TENQSP_FIT_CACHE`; off with
   `options cache=off` or the variable set to empty. A cache directory the user has set for
   JAX is used as it is.
6. **One XLA thread**, set on import when `XLA_FLAGS` is unset and JAX has not started;
   documented, with the way out (set `XLA_FLAGS`).
7. **Standard errors** as before (central differences, step 1e-3, of the gradient), each
   gradient from the predicted modes.

The start budget and narrower start draws for variability terms (also agreed) are a separate
change.

## Consequences

- Estimates change in the last digits (exact gradient; Fisher inner fit to the same
  tolerance). The CHANGELOG says so.
- ADR 0002's "each outer step costs 2p + 2q extra model evaluations" becomes one run with
  w + w² derivatives, where the solver allows it.
- A cache directory on the user's disk (a few compiled programs per problem shape), and one
  process-wide XLA setting; both described in the guide.
- The gain on its own is modest (272 → 203 s here); the larger part comes from TenQSP's
  stacked batches (TenQSP ADR 0010), which make each run of the units several times cheaper
  and which tenqsp-fit uses through TenQSP with no change here: with them the same fit took
  43 s, 29 s with the cache (−2LL 296.638416: the shared steps move the last digits).
