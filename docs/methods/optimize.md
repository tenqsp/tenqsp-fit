# method optimize

One set of parameters for all the data, by maximum likelihood (least squares when the
residual error is known), from one or several starts.

Use it for one experiment, for data pooled across subjects, or for a model whose parameters
do not vary between units. When parameters vary between units, use
[`laplace`](laplace.md); `optimize` refuses such a problem and says so.

## What it computes

The model's predictions \(f_j(\theta)\), one per measurement \(y_j\), with the residual SD
\(s_j\) from the error model (`additive`: \(s = a\); `proportional`: \(s = b f\);
`combined`: \(s = \sqrt{a^2 + (b f)^2}\); `power`: \(s = b f^k\)). Minus twice the
log-likelihood is

$$
-2LL = \sum_{j \text{ measured}} \left[ \left(\frac{y_j - f_j}{s_j}\right)^2 + \log s_j^2 + \log 2\pi \right]
\;-\; 2 \sum_{j \text{ censored}} \log \Phi\!\left(\frac{L_j - f_j}{s_j}\right)
$$

The second sum is for measurements below a limit \(L_j\) (`censor`): the probability that the
measurement fell below the limit (Beal's M3 method). The constant \(n \log 2\pi\) is included,
so −2LL is the full −2 log-likelihood; NONMEM's objective function leaves it out (see
[for NONMEM users](../guide/nonmem.md)).

With written priors (`prior` lines), the fit minimizes −2LL plus −2 log of the priors: the
posterior mode.

## The optimizer

| The error model | Optimizer | Why |
|---|---|---|
| known: every quantity of the error model fixed (`additive(fix(0.02))`) | Levenberg–Marquardt | −2LL is a sum of squares up to a constant; Levenberg–Marquardt uses the Jacobian of the residuals and converges in few steps |
| estimated | BFGS | the \(\log s_j^2\) terms do not vanish at the minimum, and Levenberg–Marquardt converges slowly there |

Both are [Optimistix](https://docs.kidger.site/optimistix/)'s, with relative and absolute
tolerances 10⁻⁸ on the parameters and on −2LL, and with TenQSP's exact derivatives of the
model's outputs (the Jacobian of the residuals, or the gradient of −2LL). The result's first
line says which was used.

**Estimation scales.** Each quantity is estimated on a scale where any value is allowed:
between bounds `in (lo, hi)` on the logit scale, above a single bound `in (lo, inf)` as
\(lo + e^u\), below one `in (-inf, hi)` as \(hi - e^u\), else
on the log scale when its start is above 0, else as it is. A positive start therefore keeps
the quantity positive. `problem.summary()` lists each quantity's scale.

## Starts

`options starts=4` runs four starts. Start 1 is the values written; the others are drawn
around them, with `seed`, on the estimation scales:

| Scale | Spread of the drawn starts |
|---|---|
| log | within a factor of about 5 either way (95 %) |
| logit (bounded) | 1 on the logit scale |
| as is | half the value, at least 0.5 |

A start at which the model does not run is drawn again. The starts run in parallel, one per
core. The best fit (lowest −2LL) is reported; `result.starts` has every start's values, final
−2LL, steps and status, and `result.plot_waterfall()` draws the final −2LL of each. When the
starts end at different fits the doctor warns: see
[a QSP model with a structured population](../start/quickstart-qsp.md).

## Standard errors

From the Hessian \(H\) of −2LL at the estimates on the estimation scale, by central
differences of the exact gradient: covariance \(= 2H^{-1}\) (NONMEM's R matrix). They are
carried to the reported scale by the derivative of the scale (the delta method). The 95 %
intervals are the estimation-scale intervals carried over, so the interval of a quantity on
the log scale is symmetric on the log scale and stays positive. When \(H\) is not positive
definite there are no standard errors, and the result says so (see the example in
[the QSP quickstart](../start/quickstart-qsp.md)).

## An example

An exponential decay, \(A' = -kA\), \(A(0) = 1\), measured six times:

```text
model decay
param k = 0.5
d_t(A) = -k*A
A(0) = 1
observable amount = A

fit decay
data "decay.csv"
observe amount = DV
estimate k = 0.3
error amount ~ additive(fix(0.02))
method optimize
options starts=3
```

```text
decay: method optimize (Levenberg-Marquardt), converged; -2LL -34.8751 (6 measurements)
starts: 3; 3 at the best fit (within 0.1); redrawn 0; failed runs during the fit 0; 8 s

quantity                  estimate         SE  RSE %   95 % CI                 also
k                          0.49354    0.01426   2.89   (0.4664, 0.5223)

doctor: 4 notes
note     determination: condition number of the curvature 1; no ridge, every quantity determined within a factor 10.
note     optima: all 3 starts reached the same fit.
note     tolerance check: at rtol 1e-08 the -2LL changes by 3.1e-05 and the estimates by less than 0.001 SE; rtol 1e-06 is enough.
note     gradient check: exact gradients agree with differences (at rtol 1e-08) to 0.017 % (3 directions).
```

With the residual SD estimated (`error amount ~ additive(0.1)`) the optimizer is BFGS:

```text
decay: method optimize (BFGS), converged; -2LL -40.4212 (6 measurements)
starts: 3; 3 at the best fit (within 0.1); redrawn 0; failed runs during the fit 0; 7 s

quantity                  estimate         SE  RSE %   95 % CI                 also
k                          0.49354   0.005941    1.2   (0.482, 0.5053)
amount.additive          0.0083343   0.002406   28.9   (0.004733, 0.01468)

doctor: 4 notes
note     determination: condition number of the curvature 575; no ridge, every quantity determined within a factor 10.
note     optima: all 3 starts reached the same fit.
note     tolerance check: at rtol 1e-08 the -2LL changes by 0.00018 and the estimates by less than 0.001 SE; rtol 1e-06 is enough.
note     gradient check: exact gradients agree with differences (at rtol 1e-08) to 0.071 % (3 directions).
```

The estimate of k is the same; its SE is smaller because the estimated SD (0.008) is smaller
than the 0.02 assumed in the first fit.

## Options

| Option | Default | Means |
|---|---|---|
| `starts` | 1 | the number of starts |
| `seed` | 0 | the seed of the drawn starts |
| `maxiter` | 200 | the most steps per start; a start that reaches it ends with status "max iterations", and the doctor reports it |
| `rtol`, `atol` | 10⁻⁶, (TenQSP's) | the ODE solver's tolerances; the doctor checks whether a tighter `rtol` changes the fit |
| `timeout` | off | a time limit (`90s`, `30min`, `2h`); the fit stops as with Ctrl-C and keeps what it has |
| `progress` | `text` (`plot` in a notebook) | `text`, `plot` or `none` |
| `checks` | `full` | `quick` leaves out the doctor's tolerance and gradient checks |

These are written on the `options` line; see [running a fit](../guide/running.md).

## What it reports

Estimates, standard errors, RSE %, 95 % intervals, the doctor's findings
([the doctor](../guide/doctor.md)), and, through the result: `profile` (profile likelihood),
`plot_gof`, `plot_individual`, `vpc`, `predict`, `tables`, `passport`, `save`; see
[results](../guide/results.md).

## Limits

- The standard errors and intervals rest on the curvature at the estimates; when the
  likelihood is far from quadratic (a quantity weakly determined, or near a bound), a
  [profile likelihood](../guide/results.md#profile-likelihood) gives a better interval.
- One set of parameters: no parameters that vary between units (use [`laplace`](laplace.md)
  or [`nuts`](nuts.md)).

## References

- Bates DM, Watts DG. *Nonlinear Regression Analysis and Its Applications.* Wiley, 1988.
- Beal SL. Ways to fit a PK model with some data below the quantification limit.
  *J Pharmacokinet Pharmacodyn* 2001;28:481–504 (the M3 method).
- Nocedal J, Wright SJ. *Numerical Optimization*, 2nd ed. Springer, 2006 (BFGS,
  Levenberg–Marquardt).
