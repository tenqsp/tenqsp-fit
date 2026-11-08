# Not yet

What tenqsp-fit 0.1 does not do. Where a problem needs one of these, tenqsp-fit stops with a
message that says so and, where there is one, what to do instead; it does not quietly do
something else.

## Methods

| Not yet | Instead, for now |
|---|---|
| FO, SAEM, importance sampling (NONMEM's `METHOD=0`, `SAEM`, `IMP`) | [`laplace`](methods/laplace.md) (FOCE-I), or [`nuts`](methods/nuts.md) |
| a written likelihood (`likelihood` lines) with parameters that vary between units | an error model; a written likelihood works with [`optimize`](methods/optimize.md), and with [`nuts`](methods/nuts.md) when no parameter varies between units |
| a profile likelihood after `nuts` | not needed: the posterior gives each quantity's interval |
| the centred parameterisation under `nuts` | the non-centred one, always (each varying parameter from a standard-normal score) |
| rebuilding TenQSP's derivative function when a run needs more room than its record has, and retrying | such a run counts as a failed run (the doctor counts them, with their causes); start values nearer the estimates help |
| centring and scaling each quantity to order one before optimizing or sampling | quantities are estimated on the log or logit scale; `nuts` adapts its step to the curvature at the start |

## The population model

| Not yet | Instead, for now |
|---|---|
| variability between occasions (IOV) | — |
| covariates that change with time within a unit | a covariate is read once per unit |
| Box-Cox and other distributions with a shape parameter (`boxcox`), and fitting a shape parameter by profile | `lognormal`, `logitnormal`, `normal`, `exponential`, `weibull` |
| `gamma` and `beta` as distributions between units | `lognormal` (gamma(mean, cv=1) is the `exponential`); `logitnormal` for values in (0, 1). Both can be priors already |
| mixture models | — |
| priors on correlations other than LKJ(2) | — |
| `nuts` sampling a `correlate` block of three or more without an edge (by the Cholesky factor of its correlation matrix) | blocks of two, or `laplace`; see [the limits of `nuts`](methods/nuts.md#limits) |

## Data

| Not yet | Instead, for now |
|---|---|
| `EVID` 3 or 4 (a reset) | split the unit at the reset into two units |
| `SS` (steady-state doses) | the dosing history with `ADDL` and `II` before the first measurement |
| `RATE` −1 or −2 (rate or duration set by the model) | `RATE` or `DUR` in the data |
| measurements above an upper limit (censoring from above) | censoring below a limit (M3) only |

## Results

| Not yet | Instead, for now |
|---|---|
| CWRES, NPDE | IWRES and the VPC (`result.vpc()`) |
| a prediction-corrected or stratified VPC | the VPC over time |
| a VPC of a link with a written likelihood | — |
| a conditional VPC (each unit's own parameters) | `result.plot_individual()` |
| the prior predictive of named predictions | `tf.prior_predictive(problem)` simulates the observables |
| ArviZ data after `optimize` and `laplace` | `result.table()`, `result.cov`, `result.corr`; `result.to_arviz()` after `nuts` |
| the expected standard errors of a planned design, and comparing designs | simulate the design (`result.predict`, or TenQSP) and fit the simulated data |
| confidence intervals of predictions after `optimize` and `laplace` beyond the standard errors | `result.predict()` gives intervals from the estimates' normal approximation, labelled approximate; after `nuts` they are posterior intervals |
| the sandwich covariance (NONMEM's default `$COV`) | standard errors from the Hessian (`MATRIX=R`); a [profile likelihood](guide/results.md#profile-likelihood) where the curvature is not enough |
| bootstrap, stochastic simulation and estimation | — |

See also [for NONMEM users](guide/nonmem.md).
