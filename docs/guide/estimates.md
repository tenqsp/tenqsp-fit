# Estimates, distributions and error models

This page covers the lines that say what is estimated: `estimate`, `fix`, `correlate`,
formulas, `error`, `likelihood` and `prior`. A model parameter that none of them names keeps
its value from the model; the summary lists those under `from model`.

| Line | Means |
|---|---|
| `estimate ke = 0.08` | one value for all units, estimated from this start |
| `estimate ka ~ lognormal(median=1.5, sdlog=0.5)` | ka varies between units; its median and sdlog are estimated |
| `estimate F = 0.5 in (0, 1)` | one value, kept inside the bounds |
| `estimate n = 2 in (1, inf)` | one value, kept above 1 (`in (-inf, 0)`: below 0) |
| `fix V = 30` | held at 30 (NONMEM's FIX) |
| `correlate ka, CL` | the correlation of ka and CL between units is estimated |
| `CL = CL_pop*(WT/70)^beta` | a model parameter set by a formula, for this fit (NONMEM's $PK) |
| `error conc ~ combined(0.5, 0.1)` | the residual error model, its quantities estimated from these starts |
| `prior ke ~ lognormal(median=0.08, sdlog=0.5)` | a prior on an estimated quantity |

## One value, or varying between units

`estimate ke = 0.08` estimates one value of `ke`, the same for every unit, from the start 0.08.
Several can share a line: `estimate g = 0.8, k = 1e-7`.

`estimate ka ~ lognormal(median=1.5, sdlog=0.5)` makes `ka` vary between units (it needs a
`unit` line and `method laplace` or `nuts`). Each number in the distribution is the start of
an estimated quantity, named after the parameter: `ka.median` (NONMEM's THETA, Monolix's
`ka_pop`) and `ka.sdlog` (the SD of log ka between units; NONMEM's OMEGA is its square). The
results show the spread also as a CV and a variance.

Every argument is named. The families and the ways of writing them are TenQSP's:

| Family | Arguments | Between units | As a prior |
|---|---|---|---|
| `lognormal` | `(median, sdlog)`, `(median, cv)`, `(meanlog, sdlog)`, `(meanlog, cv)`, `(mean, cv)`, `(mean, sdlog)` | yes | yes |
| `normal` | `(mean, sd)`, `(mean, var)`, `(mean, cv)` | yes | yes |
| `logitnormal` | `(median, sdlogit)`: a value in (0, 1), normal on the logit scale | yes | yes |
| `exponential` | `(mean)` or `(rate)` | yes | yes |
| `weibull` | `(shape, scale)` | yes | yes |
| `gamma` | `(shape, rate)`, `(shape, scale)`, `(mean, sd)`, `(mean, var)`, `(mean, cv)` | in 0.2 | yes |
| `beta` | `(alpha, beta)`, `(mean, concentration)`, `(mean, var)`, `(mean, sd)` | in 0.2 | yes |
| `uniform` | `(lower, upper)` | no | yes |

In a lognormal, `median` is the typical value; `mean` is the arithmetic mean, higher by a
factor exp(sdlog²/2). A family that cannot be used between units yet says so:

```text
line 7: estimate g ~ gamma(shape=2, rate=2)
gamma between units arrives in tenqsp-fit 0.2; lognormal is close for most data, and gamma(mean, cv=1) is the exponential.
```

### The scale of each estimate

Every quantity is estimated on a scale where it can take any value, so an optimizer cannot
step outside its range:

| Quantity | Estimated on |
|---|---|
| one value with a positive start, a median, an SD, a CV | the log scale (stays positive) |
| `normal`'s mean, `lognormal`'s meanlog, one value with a start of 0 or below | as is (any sign) |
| a value with bounds `in (lo, hi)`, a logitnormal's median | within the bounds (a logit scale) |
| a value with one bound, `in (lo, inf)` or `in (-inf, hi)` | above lo as lo + e<sup>u</sup>, below hi as hi − e<sup>u</sup> |
| a correlation | within (−1, 1) |

The summary lists the scales, and writes each one-value estimate in the long form that shows
its scale: `estimate ke = 0.08` is `ke ~ lognormal(median=0.08, sdlog=0)`, a lognormal that
does not vary. The long form can also be written, to choose the scale:

```text
estimate g ~ normal(mean=0.8, sd=0)
estimate k = 1e-7
estimate tau = 0.4 in (0, 2)
```

```text
estimated  3: g k tau
scales     as is: g; log scale: k; within bounds: tau
one value  g ~ normal(mean=0.8, sd=0)
           k ~ lognormal(median=1e-7, sdlog=0)
           tau = 0.4 in (0, 2)
```

The 95 % intervals of the results are symmetric on these scales: a positive quantity's
interval is wider above the estimate than below.

## Fixed values

- `fix V = 30` holds a model parameter (or a quantity used by a formula) at a value.
- `fix(...)` holds one argument of a distribution or of an error model:
  `lognormal(median=1.5, sdlog=fix(0.6))`, `additive(fix(0.01))`.

`estimate V = fix(30)` is refused with the form to write (`fix V = value`).

## Covariates and formulas

A model parameter can be set by a formula for this fit only, leaving the model unchanged, as
in NONMEM's $PK. The formula uses estimated or fixed quantities, model parameters and data
columns (covariates, read once per unit; see [the data](data.md#covariates)):

```text
CL = CL_pop*(WT/70)^beta
estimate CL_pop = 0.04
fix beta = 0.75
```

```text
estimated  5: CL_pop ka.median ka.sdlog ke conc.proportional
one value  CL_pop ~ lognormal(median=0.04, sdlog=0)
           ke ~ lognormal(median=0.08, sdlog=0)
fixed      beta=0.75
from model none
formulas   CL = CL_pop*(WT/70)^beta (this fit)
```

A quantity such as `CL_pop` that is not a model parameter is allowed only when a formula of
the fit uses it. A formula the model itself has (`ka = mu` in the model) is listed as
`(model)`; its parameters are estimated directly (`estimate mu = 1`).

A parameter that varies between units can have a formula as its location: the covariate model
of a population fit.

```text
estimate CL_pop = 0.04
estimate CL ~ lognormal(median=CL_pop*(WT/70)^0.75, cv=0.3)
```

Each unit's CL is then lognormal around `CL_pop*(WT/70)^0.75` for its own weight.

## Correlations

`correlate ka, CL` estimates the correlation of ka and CL between units, from 0 (or from
`correlate ka, CL = -0.3`). Three or more names correlate every pair, as NONMEM's BLOCK.

## A worked example

The theophylline data in NONMEM layout ([the data](data.md#the-nonmem-layout-dose-rows)),
with a covariate on CL, a fixed spread of ka, a correlation, a bounded ke with a prior, and a
combined error:

```text
estimate ka ~ lognormal(median=1.5, sdlog=fix(0.6))
estimate CL_pop = 0.04
estimate CL ~ lognormal(median=CL_pop*(WT/70)^0.75, cv=0.3)
correlate ka, CL
estimate ke = 0.08 in (0.01, 1)
error conc ~ combined(0.5, 0.1)
prior ke ~ lognormal(median=0.08, sdlog=0.5)

method laplace
```

```text
errors     conc ~ combined(0.5, 0.1)   (sd = sqrt(a^2 + (b*f)^2), f = prediction)
varying    ka ~ lognormal(median=1.5, sdlog=fix(0.6))
           CL ~ lognormal(median=CL_pop*(WT/70)^0.75, cv=0.3)
correlated ka, CL (start 0)
estimated  7: ka.median CL_pop CL.cv ke corr(ka,CL) conc.additive conc.proportional
scales     log scale: ka.median CL_pop CL.cv conc.additive conc.proportional; within bounds: ke corr(ka,CL)
one value  CL_pop ~ lognormal(median=0.04, sdlog=0)
           ke = 0.08 in (0.01, 1)
fixed      ka.sdlog=0.6
...
prior      ke ~ lognormal(median=0.08, sdlog=0.5)  (used: the estimates are the posterior mode)
```

The fit (about 40 seconds):

```text
theoph_cov: method laplace (BFGS, FOCE-I), converged; -2LL 353.6949 (132 measurements, 12 units)
starts: 1; 1 at the best fit (within 0.1); redrawn 0; failed runs during the fit 0; 40 s

quantity                  estimate         SE  RSE %   95 % CI                 also
ka.median                    1.548     0.2839   18.3   (1.08, 2.218)
CL_pop                    0.040212    0.00295   7.34   (0.03483, 0.04643)
CL.cv                      0.23644    0.05199     22   (0.1537, 0.3638)        sdlog 0.2332, var 0.0544
ke                        0.085082   0.003922   4.61   (0.07775, 0.09314)
corr(ka,CL)                -0.1659      0.278    168   (-0.6217, 0.3739)
conc.additive              0.46948     0.1105   23.5   (0.296, 0.7446)
conc.proportional          0.10163    0.02663   26.2   (0.06081, 0.1698)

with the written priors: the estimates are the posterior mode; -2LL plus the priors' terms 349.2334
...
```

## Residual error models

`error conc ~ additive(1)` gives the residual error of a linked measurement. The numbers are
the starts of the estimated quantities, named `conc.additive`, `conc.proportional`,
`conc.exponent`; `fix(...)` holds one. With `f` the prediction:

| Model | SD of a measurement | Quantities |
|---|---|---|
| `additive(a)` | a | `.additive` |
| `proportional(b)` | b·f | `.proportional` |
| `combined(a, b)` | sqrt(a² + (b·f)²): the variances add, as NONMEM's `Y = F + F*EPS(1) + EPS(2)` | `.additive`, `.proportional` |
| `power(b, k)` | b·f^k (k = 1 is proportional) | `.proportional`, `.exponent` |

The error is normal on the link's scale: on `observe log10(bact) = DV`, an additive error is
an additive error on log10 counts (about a constant CV on the counts). In the proportional,
combined and power models, `f` is kept above 0 by a smooth floor at the solver's absolute
tolerance, so the SD cannot reach 0 (the summary lists it under `near zero`).

**A log-normal (exponential) error** has no name of its own: it is an additive error on the
log scale. Put the logarithm of the measurements in a column of the data and link it on that
scale, `observe log(conc) = LNDV` with `error log(conc) ~ additive(0.2)`: the measurement is
then lognormal around the prediction, with sdlog 0.2 (NONMEM's `Y = LOG(F) + EPS(1)` with
log-transformed data).

### A likelihood written out

For a residual distribution that is not one of these, the log-density of one measurement can
be written in place of the `error` line, as NONMEM's `$ERROR` with `-2LL`. `DV` is the
measurement, the observable the prediction, and the other names estimated or fixed
quantities:

```text
estimate k = 0.3
estimate s = 0.05
likelihood amount: -0.5*((DV - amount)/s)^2 - log(s) - 0.5*log(2*3.141592653589793)
method optimize
```

```text
check_t: method optimize (BFGS), converged; -2LL -30.6710 (3 measurements)
...
quantity                  estimate         SE  RSE %   95 % CI                 also
k                          0.49487   0.001316  0.266   (0.4923, 0.4975)
s                        0.0014579  0.0005952   40.8   (0.000655, 0.003245)
```

−2LL is −2 times the sum of what is written, so constants left out of the formula are left out
of −2LL. A written likelihood cannot be combined with `censor` (write the censored part into
the formula) and is not available with `method laplace` yet.

## Priors

`prior ke ~ lognormal(median=0.08, sdlog=0.5)` puts a prior on an estimated quantity, by its
name in the results: `ke`, `CL.median`, `ka.sdlog`, `conc.additive`. The arguments are
numbers. With `method optimize` and `laplace` the written priors are used, and the estimates
are the posterior mode (the printout says so, and gives −2LL with and without the priors'
terms). With `method nuts` every quantity has a prior: written, or by default:

| Quantity | Default prior |
|---|---|
| positive (log scale) | `lognormal(median=start, sdlog=2)`: 95 % within a factor 50 of the start |
| any sign | `normal(mean=start, sd=2*abs(start))`; a start of 0 needs a written prior |
| bounded `in (lo, hi)` | `uniform(lower=lo, upper=hi)` |
| one bound, `in (lo, inf)` or `in (-inf, hi)` | the positive rule for x − lo or hi − x |
| correlations | LKJ(2) on each `correlate` block |

A prior must give density to every value the quantity can take: a lognormal prior on a
quantity of any sign is refused, with what to write instead. The defaults are listed by
`summary()` under `method nuts`. What data the priors expect can be checked before fitting
with the [prior predictive check](python.md#the-prior-predictive-check).
