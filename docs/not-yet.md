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

## The population model

| Not yet | Instead, for now |
|---|---|
| variability between occasions (IOV) | — |
| covariates that change with time within a unit | a covariate is read once per unit |
| `gamma` and `beta` as distributions between units | `lognormal` (gamma(mean, cv=1) is the `exponential`); `logitnormal` for values in (0, 1). Both can be priors already |
| mixture models | — |
| estimating a model parameter that has one value per category (`param k[strain]`) | write one parameter per category in the model |
| priors on correlations other than LKJ(2) | — |

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
| confidence intervals of predictions after `optimize` and `laplace` beyond the standard errors | `result.predict()` gives intervals from the estimates' normal approximation, labelled approximate; after `nuts` they are posterior intervals |
| the sandwich covariance (NONMEM's default `$COV`) | standard errors from the Hessian (`MATRIX=R`); a [profile likelihood](guide/results.md#profile-likelihood) where the curvature is not enough |
| bootstrap, stochastic simulation and estimation | — |

See also [for NONMEM users](guide/nonmem.md).
