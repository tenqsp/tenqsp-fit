# Changelog

All notable changes to this project are documented here. The project follows semantic
versioning; `0.x` releases may change the API.

## [Unreleased]

### Added
- **Parameters per category.** Each entry of a model parameter with one value per category
  (`param k[strain]`) is a parameter of the fit: `estimate k[R] = 3e-11`, `fix k[S] = 1e-8`,
  or the whole table, `estimate k[strain] = {S: 3e-8, R: 3e-11}` (nested braces over several
  axes). Entries take bounds, priors, distributions between units and correlations, and
  formulas of the fit set them (`k = k_all` gives every category one estimate;
  `k[R] = k[S]*f`). The results, the doctor and the tables name entries as `k[R]`.

### Fixed
- `k = k_all` for a parameter per category was accepted when read and failed when the fit
  started; it now sets every category.

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
