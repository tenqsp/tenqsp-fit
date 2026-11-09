# For NONMEM users

tenqsp-fit reads NONMEM's data layout and fits NONMEM's population model by NONMEM's FOCE-I.
What differs is mostly in the writing: the model is a TenQSP model (equations, not
`$PK`/`$DES` code), and the fit part says what each column means and how each parameter
varies, in one statement each.

## A control stream, translated

```text
model pk
param ka=1.5, CL=2.8, V=32
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
observable conc = centr/V

fit pk_pop
data "pk.csv"
unit ID
doses CMT 1 into depot
observe conc = DV
estimate CL_pop = 2.8
estimate CL ~ lognormal(median=CL_pop*(WT/70)^0.75, sdlog=0.3)
estimate V ~ lognormal(median=30, sdlog=0.2)
correlate CL, V
estimate ka = 1.5 in (0.1, 10)
error conc ~ proportional(0.1)
method laplace
```

is, in NONMEM,

```text
$INPUT ID TIME AMT DV EVID CMT WT
$DATA pk.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=6
$MODEL COMP=(DEPOT,DEFDOSE) COMP=(CENTR,DEFOBS)
$PK
  CL = THETA(1)*(WT/70)**0.75*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)
$DES
  DADT(1) = -KA*A(1)
  DADT(2) =  KA*A(1) - CL/V*A(2)
$ERROR
  IPRED = A(2)/V
  Y = IPRED*(1 + EPS(1))
$THETA (0, 2.8) (0, 30) (0.1, 1.5, 10)
$OMEGA BLOCK(2) 0.09 0 0.04
$SIGMA 0.01
$EST METHOD=COND INTER
$COV
```

`problem.summary()` lists how tenqsp-fit read each line:

```text
pk_pop: model pk (2 states), data "pk.csv": 3 units, 18 observations, 3 doses
units      ID: 3 units
links      conc <- DV (all rows)   18 rows, 3 units
doses      AMT by CMT: 1 -> depot   3 rows
errors     conc ~ proportional(0.1)   (sd = b*f, f = prediction)
varying    CL ~ lognormal(median=CL_pop*(WT/70)^0.75, sdlog=0.3)
           V ~ lognormal(median=30, sdlog=0.2)
correlated CL, V (start 0)
estimated  7: CL_pop CL.sdlog V.median V.sdlog ka corr(CL,V) conc.proportional
scales     log scale: CL_pop CL.sdlog V.median V.sdlog conc.proportional; within bounds: ka corr(CL,V)
one value  CL_pop ~ lognormal(median=2.8, sdlog=0)
           ka = 1.5 in (0.1, 10)
fixed      none
from model none
formulas   none
method     laplace (population, FOCE-I)
options    rtol 1e-6 (default), starts 1 (default), seed 0 (default), maxiter 200 (default), checks full (default)
near zero  kept above 0 by a smooth floor at 1e-6 (the solver's atol): f in conc ~ proportional
```

## Translation table

| NONMEM | tenqsp-fit |
|---|---|
| `$INPUT`, `$DATA` | `data "file.csv"`; columns with other names are mapped: `data "f.csv" with ID=Subject, TIME=Time` ([the data](data.md)) |
| `ID` | `unit ID` |
| `TIME`, `DV`, `AMT`, `EVID`, `MDV`, `CMT`, `DVID`, `RATE`, `II`, `ADDL` | read as in NONMEM; also a `DUR` column for an infusion's duration |
| `IGNORE=`, `ACCEPT=` | `ignore DV where TIME == 0` |
| doses into a compartment (`CMT`, `DEFDOSE`) | `doses CMT 1 into depot, 2 into centr`, or `doses into depot` for every dose row |
| several observation types (`DVID`, `CMT` on observation rows) | one `observe` line each: `observe conc = DV where DVID == 1` |
| `$PK`, `$DES`, `$ERROR`'s prediction | the TenQSP model above the `fit` line: parameters, equations, observables |
| a covariate relation in `$PK` | a formula in the distribution, `lognormal(median=CL_pop*(WT/70)^0.75, ...)`, or a formula line ([estimates and distributions](estimates.md)) |
| `THETA` | `estimate ke = 0.08`; a typical value of a varying parameter is its distribution's `median` (lognormal) or `mean` (normal) |
| bounds `(lo, init, hi)` | `estimate ka = 1.5 in (0.1, 10)` |
| `ETA` with `EXP(ETA)` | `estimate CL ~ lognormal(median=..., sdlog=...)`: \(\eta = \text{sdlog} \cdot z\) |
| `OMEGA` (diagonal) | the spread: `sdlog` (\(\omega = \text{sdlog}\), OMEGA = sdlog²), or `cv` |
| `$OMEGA BLOCK` | `correlate CL, V` (three or more names: every pair) |
| `$OMEGA ... SAME` (one variance for several ETAs) | name the spread and estimate it once: `sdlog=w` in each, `estimate w = 0.3` |
| `FIX` | `fix(0.3)` inside a distribution or error model, or `fix V = 32` |
| `$SIGMA`, `EPS` | `error conc ~ additive(a)`, `proportional(b)`, `combined(a, b)`, `power(b, k)`: the SD itself, not a variance |
| `$SIGMA ... SAME` (one residual SD for several outputs) | name it and estimate it once: `error conc ~ additive(s)` and `error conc2 ~ additive(s)`, `estimate s = 0.2` |
| BLQ, M3 | `censor conc below 0.05`; with a flag and a limit column, `censor conc where BLQ == 1 below LLOQ` |
| BLQ, M1 (rows dropped) | `ignore DV where BLQ == 1`; the passport lists the rows left out |
| an exponential error on log-transformed data (`Y = LOG(F) + EPS(1)`) | a column of the logs, `observe log(conc) = LNDV`, and `error log(conc) ~ additive(sd)` |
| `$EST METHOD=COND INTER` | `method laplace` |
| `$EST METHOD=NUTS` (or `BAYES`) | `method nuts` |
| `$COV` | always run: standard errors with every fit (see below) |
| `$COV PRINT=E`, the correlation matrix | `result.corr`; its eigenvalues are in the passport |
| a covariance step that fails, or a skewed interval | `result.profile("CL")`: a profile likelihood interval ([results](results.md#profile-likelihood)) |
| CWRES, NPDE | not yet: IWRES in `sdtab` and the goodness-of-fit plots, and the VPC |
| `$EST METHOD=0` (FO), `SAEM`, `IMP` | not yet: `method laplace` (FOCE-I) or `method nuts` |
| `$SIM`, a VPC by PsN | `result.vpc()`, `result.predict(regimen=...)` |
| `$TABLE` (sdtab, patab) | `result.tables()`: `sdtab` and `patab` ([results](results.md)) |
| EBEs, `ETA` in tables | `result.individual()`: each unit's scores `z(CL)` (= ETA / its SD) and values |
| PsN's `sse`, bootstrap | not yet |

## What differs

**−2LL and OFV.** tenqsp-fit reports the full −2 log-likelihood, with the constant \(n \log
2\pi\) (\(n\): measured values, not counting censored ones). NONMEM's objective function value
leaves it out:

$$
\text{OFV} = -2LL - n \log 2\pi .
$$

Theophylline ([the population quickstart](../start/quickstart-population.md)): −2LL 353.9837
with 132 measurements is an OFV of 111.38. Differences of −2LL between two models of the same
data are differences of OFV.

**Spreads, not variances.** A lognormal's spread is `sdlog`, the SD of \(\eta\); the table
shows NONMEM's OMEGA (its square) and the CV beside it. Residual errors are SDs too
(`additive(0.7)` is \(\sigma = 0.7\), `$SIGMA 0.49`).

**Scores.** Every varying parameter is a transform of a standard-normal score \(z\), so
`z(CL)` in `result.individual()` is NONMEM's ETA divided by its SD. Parameters can vary by
other families than the lognormal (normal, logitnormal, exponential, Weibull); see
[`laplace`](../methods/laplace.md).

**Estimation scales.** Every quantity is estimated on a scale where any value is allowed:
positive quantities on the log scale, bounded ones on the logit scale. The 95 % intervals are
symmetric on that scale and carried back, so they never cross 0 or a bound.

**The covariance step.** Standard errors come from the Hessian of −2LL alone (NONMEM's
`$COV MATRIX=R`), not NONMEM's default sandwich \(R^{-1} S R^{-1}\). They are computed after
every fit; when the Hessian is not positive definite, the result and the doctor say so.

**Shrinkage** is NONMEM's SD-based ETA shrinkage: \(1 - \mathrm{SD}(\hat\eta)/\omega\), here
\(1 - \mathrm{SD}(\hat z)\).

**Starts and the doctor.** `options starts=8` runs eight fits from different starts and
reports how many reached the best one. Every fit ends with [the doctor](doctor.md): ridges
(quantities the data cannot tell apart), several optima, estimates at a bound, shrinkage, and
whether the solver's tolerance was tight enough.

**Tolerances.** The ODE solver's relative tolerance is `options rtol=1e-6` (the default);
NONMEM's `TOL=6` is the same. The doctor checks whether a tighter one changes the fit.

## Data rows not read yet

| Rows | Instead |
|---|---|
| `EVID` 3 or 4 (a reset) | split the unit at the reset into two units (a new ID for each period) |
| `SS` (steady-state dose) | give the dosing history (`ADDL`, `II`) before the first measurement |
| `RATE` −1 or −2 (rate or duration set by the model) | give `RATE` or `DUR` in the data |
| a covariate that changes within a unit | not yet: a covariate is read once per unit |

tenqsp-fit says so when it reads such rows, naming the rows. See [not yet](../not-yet.md).
