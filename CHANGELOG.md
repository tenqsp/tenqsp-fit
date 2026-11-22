# Changelog

All notable changes to this project are documented here. The project follows semantic
versioning; `0.x` releases may change the API.

## [Unreleased]

### Added
- **Models given by formulas of t** (NONMEM's `$PRED`): a model with no equations and
  outputs written as formulas, fitted by every method with no solver. A dose enters as a
  parameter set from a data column (`D = DOSE`); `doses` lines and dose rows are refused,
  with that way out in the message. The summary and `doctor()` say that no solver runs.

- **`options cache`** (`on` by default): JAX's compilation cache on disk, in
  `~/.cache/tenqsp-fit/jax` or the directory `TENQSP_FIT_CACHE` names, so a problem fitted
  again compiles nothing (a Friberg myelosuppression fit: ~23 s of compiling becomes ~2 s).
  `cache=off`, or `TENQSP_FIT_CACHE` set to empty, turns it off.

### Changed
- **TenQSP's native solver** (`options backend=auto`, the default): the units' simulations
  run on TenQSP's BDF compiled by numba wherever the model allows it (forward derivatives, no
  events or conditions on time), else on JAX; estimates agree with the JAX solver to
  round-off. A population fit of Friberg's model to 20 patients: 44 s with JAX, 9 s native.
  The first fit of a model compiles its solver once (cached on disk). `backend=jax` or
  `backend=native` chooses. Needs TenQSP 0.9.1 (0.9 brings it; 0.9.1 fixes a ZeroDivisionError in it); numba is now a dependency.
- **The setup compiles less**: the fit's one compiled function (value, gradient and the
  run's steps, with the tolerances and the step limit as inputs) also serves the standard
  errors and the doctor's checks, and the step limit is set from the fit's first run instead
  of a separate run and build.
- **Profiles stay within the bounds** of the quantity profiled (walk and scan).
- **Bounds are kept as bounds** (ADR 0006): `method optimize` uses L-BFGS-B on −2LL (or
  trust-region least squares where −2LL is a sum of squares), and `method laplace` and
  profiles use the same optimizer, all from SciPy, instead of Optimistix and BFGS on scales
  that mapped bounds away (the logit). On those scales a quantity pushed near a bound could
  not come back: fits ended on a bound, with a near-bound warning, above the optimum (in one
  fit 4.3 above it in −2LL). **Fits can end at a lower −2LL than before, and estimates near
  a bound change.** A bounded quantity is now estimated on the log scale when its lower bound
  is 0 or above, else as it is; the summary says `log scale, bounded` or `as is, bounded`.
  Default priors and NUTS keep their scales.
- **A start ends when −2LL stops moving**: also when its last 5 steps lowered −2LL by less
  than 1e-4 in all (before, 1e-8 on −2LL and on the parameters, which a stiff model's −2LL,
  carrying an integration error of order 1 at the default tolerance, kept from being met:
  one fit spent most of its time lowering −2LL by less than 1).
- **A rejected step costs one run**, not two (the point was run again to ask whether the
  model ran there). A start whose derivatives are not finite is drawn again.
- **The doctor's ridges leave out quantities at a bound**: an active bound is a result, not
  something the data fail to determine.
- **Needs SciPy 1.16 or later; no longer needs Optimistix.**
- **`method laplace` is faster** (ADR 0005): the objective is compiled once for the fit, the
  standard errors, the modes and the doctor (it was compiled four times); the inner fit takes
  one run of the units per step (Fisher scoring) and starts from the modes predicted at the
  new point; the gradient is exact (second derivatives from the solver, forward over
  forward) instead of central differences. Estimates can change in the last digits.
- **One XLA thread**: on import, tenqsp-fit sets `XLA_FLAGS` to one thread unless it is
  already set (to anything) or JAX has started: the many small operations of a fit run
  faster that way. Set `XLA_FLAGS` yourself to choose otherwise.
- **Needs TenQSP 0.8.2 or later** (models given by formulas came in 0.8.1; 0.8.2 fixes
  derivatives that 0.8.1 resolved too coarsely, which could make standard errors fail).

### Fixed
- The doctor's tolerance check after `method laplace` compiled the gradient again (some 20 s
  on a Friberg myelosuppression problem), although the tolerances are an input of the compiled
  function so that it can be rerun at tighter ones.
- A fit whose estimates have two or more directions in which -2LL does not change (or
  hardly) stopped with "The truth value of an array with more than one element is
  ambiguous" when the doctor grouped them, instead of reporting them.

## [0.2.1] — 2026-11-10

### Added
- **Tutorials**, run by the test suite: 1, theophylline (one subject by `optimize` and
  `nuts`, each subject alone, then the population by `laplace`, each checked against R's
  `nls` and `nlme`); 2, a transit delay and absorption (what a sparse and a rich design
  determine; profile intervals against standard-error intervals over ten simulated studies);
  3, phenobarbital in newborns (covariates from sparse routine data, checked against
  NONMEM's run of the same model; shrinkage); 4, myelosuppression (a simulated
  dose-ranging study, a prediction for a new dose, and the dose a virtual population of the
  fitted model picks); 5, warfarin (checked against Monolix's fits; a sex effect that is a
  weight effect).
- **Example data:** phenobarbital (`nlme::Phenobarb`) and warfarin (the Monolix demo set),
  with `docs/examples/DATA.md` giving the source of every dataset.
- **Doses into one element of a field** over categorical axes: `doses DAMT at t=2 into
  Cpla[phage=mxp1001]`, and in the CMT form `doses CMT 1 into Cpla[phage=mxp1001]`. Doses
  given as a column may take several `doses` lines, one per place (two phages given together).
- **`result.vpc(rtol=, atol=)`**: the VPC's simulations at other solver tolerances than the
  fit's, without a rebuild. A fit's tolerances set for the last digits of −2LL can make the
  VPC far slower than it needs to be.
- **A VPC by group:** `result.vpc(by="GROUP")` draws one row of panels per value of a data
  column (dose groups, studies), each with its own time bins and percentiles; also
  `tf.prior_predictive(problem, by=...)`.

### Fixed
- **Standard errors with a wide box.** An estimate counted as "at a bound" when it lay within
  1 % of its box's width from it, so in a box several decades wide (`in (100, 1e8)`, as
  written for rate constants and burdens) an estimate well inside it (23,479) was "at the
  lower bound 100". It then got no standard error, and the others' were computed with it held
  fixed: **too small**, under a status "ok". Near a bound now means within 1 % of the
  estimate's or the bound's size (or of a narrower box's width); an estimate held at a bound
  is named in `result.covariance_status`, which then begins with "conditional".
- `tf.fit(problem, method="optimize")` refused a problem written without a `method` line
  ("no method"); `method=` now stands for it.
- A profile whose end was not reached printed it as `nan`; it now says "not reached".
- A profile that reached a bound went on refitting points next to it,
  and a value on a bound gave a divide-by-zero warning.
- A model whose formulas use the model's own notation, such as a field at a boundary
  (`Psen = Pdl(w=1)`, a transit delay), could not be fitted: building the units, the
  summary, and fixing or estimating a name the model defines all stopped at that formula.
  A test now fits a model using every notation TenQSP's formulas allow.
- `result.model()` after a population fit (method laplace) failed; it now writes the model
  with each varying parameter at its typical value.
- Estimating a named total or slice of the model's fields (`B_tot = sum_s(B)`) is refused
  with a message saying what it is, as for the model's other named formulas.

## [0.2.0] — 2026-11-09

### Added
- **Parameters per category.** Each entry of a model parameter with one value per category
  (`param k[strain]`) is a parameter of the fit: `estimate k[R] = 3e-11`, `fix k[S] = 1e-8`,
  or the whole table, `estimate k[strain] = {S: 3e-8, R: 3e-11}` (nested braces over several
  axes). Entries take bounds, priors, distributions between units and correlations, and
  formulas of the fit set them (`k = k_all` gives every category one estimate;
  `k[R] = k[S]*f`). The results, the doctor and the tables name entries as `k[R]`.
- **Models that read a table's entry by label** (TenQSP 0.8.0's `Vvas[organ=spleen]`,
  `iKp[phage, organ=liver]`): estimating or fixing `Vvas[spleen]` reaches every place the model
  reads it.
- **One residual SD for several links** (NONMEM's `SAME` for residual error): an error
  model's number may be a name estimated or fixed once, `error conc ~ additive(s)` and
  `error conc2 ~ additive(s)` with `estimate s = 0.2`; in Python `tf.additive("s")`. Any part
  of `combined` and `power` can be shared this way.
- **Estimates at a bound** are marked in `result.table()` (column `bound`, and "at its lower
  bound" in the printout). They have no standard error or interval, and the other quantities'
  standard errors are computed with them held at the bound.
- **`result.model()`**: the TenQSP model with the fitted values in (estimated, fixed and
  formula-set parameters, table entries included), written at full precision by
  `.to_text()`; `result.save()` writes it as `fitted_model.tq`.
- **A doctor note on scales:** a quantity held above 0 by the log scale only because its start
  was positive is named when its written prior puts weight below 0, or when quantities that
  share its prior (or its table) were estimated as they are.

### Changed
- **Needs TenQSP 0.8.0 or later** (for models that read a table's entry by label).
- **Shorter steps for `method optimize` and profiles:** one step moves no quantity by more than
  a factor 100 (on its log or logit scale). Along a nearly flat direction, Levenberg–Marquardt
  and BFGS could propose steps of hundreds of powers of ten, where the model does not run;
  each was refused and halved about ten times. In a two-strain phage fit this removes the 12 %
  of failed runs, makes the fit 2–4 times faster and, from poor starts, keeps a weakly
  determined rate from being thrown to 1e-195 and left there.
- **The doctor's failed-runs check** tells failed runs near the estimates from those far from
  them: when every failed run checked was far (an overshoot the fit came back from) it is a
  note, not a warning.

### Fixed
- A model parameter declared without a value (`param k`, `param k[s]`), its start given by the
  fit, made `tf.objective` and `tf.fit` fail with "No value for k".
- A dose row with `AMT` 0 (a placebo record) was refused as having no amount.
- `k = k_all` for a parameter per category was accepted when read and failed when the fit
  started; it now sets every category.
- A model with an observable or a named formula that reads a parameter per category
  (`observable whole = C*Vo[organ]`) could not be fitted at all, even when that observable was
  not linked: reading what each observable depends on refused `Vo[organ]`.
- `observe` refuses an observable with one value per category (or per point of a continuous
  axis) at that line, and says to write one observable per category; such an observable used
  to be accepted and the fit then failed inside the library. `observe conc[phage=p1]` says
  that an observable cannot be sliced there (it suggested `==`).
- The docs said to write the censored part of the data into a written likelihood, which cannot
  read the data's columns; they now say to use an error model with `censor`, and so does the
  message for a written likelihood given with `censor`. The estimates
  chapter says that a quantity which may be negative needs bounds spanning zero, and what an
  additive error on a log10 link means (lognormal measurements; the SD on the log10 scale).

## [0.1.0] — first release

tenqsp-fit's first release: fitting TenQSP models to data, with a diagnosis of what the data
can and cannot determine. It needs TenQSP 0.7.0 or later.

### Writing a fit
- **A fit as text** after the TenQSP model (`fit`, `data`, `observe`, `estimate`, `error`,
  `prior`, `method`, `options`), in the same file or apart, or **built in Python** with one
  call per line; the two give the same problem. Unknown words and impossible values are
  refused with a message that names the line.
- **Data** in the NONMEM layout (dose rows with `EVID`, `AMT`, `CMT`, `RATE`/`DUR`, `ADDL`/`II`,
  `MDV`) or the R layout (the dose as a column), long or wide; column names mapped in the
  `data` line; rows left out with `ignore`; measurements below a limit of quantification (M3);
  covariates read once per unit.
- **Estimates** with one value, or varying between units (`lognormal`, `logitnormal`,
  `normal`, `exponential`, `weibull`), with covariates and formulas, correlations
  (`correlate`), fixed values and bounds (`in (lo, hi)`).
- **Residual error models** (additive, proportional, combined, power; also on a transformed
  output such as `log(conc)`), or a likelihood written out.
- **Priors** (normal, lognormal, logitnormal, exponential, weibull, gamma, beta, uniform;
  LKJ(2) for correlations), used by `nuts` and, as penalties, by `optimize` and `laplace`.

### Methods
- `method optimize`: maximum likelihood for individual models and for parameters common to
  all units; Levenberg–Marquardt when every error is additive with a known size, BFGS on −2LL
  otherwise; several starts on request; standard errors from the curvature.
- `method laplace`: population models by the Laplace approximation (FOCE-I), with standard
  errors, each unit's estimates and shrinkage.
- `method nuts`: Bayesian fits by the No-U-Turn sampler (NumPyro), non-centred, chains in
  parallel, with R-hat, effective sample sizes and divergences.
- Exact derivatives from TenQSP throughout; BDF by default, with Radau as fallback.
- Ctrl-C stops a fit, a profile or a sampler and keeps what was done.

### Results
- A printout, the same as data, each unit's values, NONMEM-style tables (`sdtab`, `patab`),
  goodness-of-fit plots, individual plots and visual predictive checks.
- **Profile likelihood** for any quantity, with its shape (closed, flat, not reached).
- **Predictions for a new regimen**, with their uncertainty and, on request, a new unit's
  range.
- **The doctor**: ridges (quantities the data cannot tell apart), weakly determined
  quantities, the condition number, several optima, convergence, bounds, shrinkage, a
  tolerance check, a gradient check, jumps in the model and failed runs, each with what to
  do.
- **The passport**: one document per fit with what was run (hashes of the problem text and
  the data, versions, platform) and what the data determine. `result.save(folder)` writes the
  fit, its tables, plots and passport.

### Documentation
- Two quickstarts (a population PK fit; a QSP model with a structured population), a guide
  in seven chapters (including one for NONMEM users), a page per method, a page listing what
  0.1 does not do yet, and decision records for the defaults.
