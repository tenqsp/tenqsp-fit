# method laplace

A population fit: parameters that vary between units (subjects, animals, flasks), by the
Laplace approximation of the likelihood. With an additive or other error model it is
NONMEM's FOCE with interaction (`METHOD=COND INTER`).

Use it whenever a parameter is written with a distribution
(`estimate ka ~ lognormal(median=1.5, sdlog=0.5)`). A problem without one is refused, with a
pointer to [`optimize`](optimize.md).

## Parameters that vary: normal scores

Each parameter that varies is a transform of a standard-normal **score** \(z_i\) per unit
\(i\): its distribution's quantile function applied to \(\Phi(z_i)\). For a lognormal,

$$
ka_i = \text{median} \cdot e^{\,\text{sdlog} \cdot z_i},
$$

which is NONMEM's \(\theta \, e^{\eta_i}\) with \(\eta_i = \text{sdlog} \cdot z_i\) and
\(\omega^2 = \text{sdlog}^2\). The distributions that can vary between units:

| Family | Written as |
|---|---|
| `lognormal` | `median, sdlog`, `median, cv`, `meanlog, sdlog`, `mean, cv`, ... |
| `normal` | `mean, sd`, `mean, var`, `mean, cv` |
| `logitnormal` | `median, sdlogit` (values in (0, 1)) |
| `exponential` | `mean` or `rate` |
| `weibull` | `shape, scale` |

Whatever the family, the scores are normal, so the method is the same for all of them.
`correlate ka, CL` estimates the correlation between the two parameters' scores (three or
more names: every pair, NONMEM's `BLOCK`); the scores then have a correlation matrix \(R\);
otherwise \(R = I\).

A distribution's arguments may be formulas of covariates
(`estimate CL ~ lognormal(median=CL_pop*(WT/70)^0.75, sdlog=0.3)`); see
[estimates and distributions](../guide/estimates.md).

## What it computes

For unit \(i\) with measurements \(y_i\), \(-2\log p(y_i)\) is approximated at the scores
\(\hat z_i\) that maximize the unit's joint density (its conditional mode; NONMEM's empirical
Bayes estimate, EBE):

$$
-2LL_i \approx L_i(\hat z_i) + \hat z_i^\top R^{-1} \hat z_i + \log|R| + \log|H_i|,
\qquad H_i = J_i^\top W_i J_i + R^{-1},
$$

and \(-2LL = \sum_i -2LL_i\). Here \(L_i\) is the unit's −2 log-likelihood given its scores
(as in [`optimize`](optimize.md): the error model, \(n \log 2\pi\), the M3 terms of censored
measurements), \(J_i\) the derivative of the unit's predictions by its scores, and \(W_i\) the
information of each measurement about its prediction: \(1/s^2 + 2\,(\partial \log s / \partial
f)^2\) for a measured value. The second term makes an error model that depends on the
prediction (proportional, combined, power) "with interaction".

## The fit

**Inner fit.** The conditional modes of all units at once, by Newton steps (from differences
of the exact gradient by the scores), with a Fisher-scoring step when Newton's would not go
downhill, and step halving. Each evaluation starts from the modes found last.

**Outer fit.** BFGS ([SciPy](https://scipy.org)) on the estimation scales of
[`optimize`](optimize.md) (log, logit within bounds; a correlation within (−1, 1)), as NONMEM
minimizes its FOCE objective. Its gradient includes how the modes move with the parameters
(the implicit function theorem), so the fit reaches the minimum of the Laplace −2LL. The
terms of this gradient are central differences (step 10⁻⁴ on the estimation scale) of the
objective and of its exact derivative by the scores, at fixed modes: not exact, but close
enough for the fit to reach the minimum (theophylline: −2LL 353.98, against 354.04 with the
first-order "FOCE gradient", which holds the solver's sensitivities fixed). The fit
ends when the largest derivative of −2LL on the estimation scale is below 10⁻³.

**Starts** as in [`optimize`](optimize.md): `options starts=4` draws three more around the
values written, with `seed`, and runs them in parallel.

## Standard errors and shrinkage

**Standard errors** from the Hessian of −2LL at the estimates, by central differences of the
gradient: covariance \(= 2H^{-1}\) on the estimation scale (NONMEM's R matrix), carried to the
reported scale. The 95 % intervals are carried over from the estimation scale, so a positive
quantity's interval stays positive.

**Shrinkage** per varying parameter: \(1 - \mathrm{SD}(\hat z)\), the SD of the units' modes
(with \(n-1\)). This is NONMEM's SD-based ETA shrinkage: near 0, each unit's own data
determine its value; near 1, its value is the typical value.

The table also shows each spread in its other usual forms: for a lognormal's `sdlog`, the CV
(\(\sqrt{e^{\text{sdlog}^2} - 1}\)) and the variance (NONMEM's OMEGA).

## An example

Theophylline in 12 subjects ([the population quickstart](../start/quickstart-population.md)):

```text
estimate ka ~ lognormal(median=1.5, sdlog=0.5)
estimate CL ~ lognormal(median=0.04, sdlog=0.5)
estimate ke = 0.08
error conc ~ additive(1)

method laplace
```

```text
theoph_pop: method laplace (BFGS, FOCE-I), converged; -2LL 353.9837 (132 measurements, 12 units)
starts: 1; 1 at the best fit (within 0.1); redrawn 0; failed runs during the fit 0; 40 s

quantity                  estimate         SE  RSE %   95 % CI                 also
ka.median                   1.6195     0.3207   19.8   (1.099, 2.388)
ka.sdlog                   0.65643       0.15   22.9   (0.4194, 1.027)         cv 73.4 %, var 0.4309
CL.median                 0.039544   0.002353   5.95   (0.03519, 0.04444)
CL.sdlog                   0.16751    0.03643   21.7   (0.1094, 0.2566)        cv 16.9 %, var 0.02806
ke                        0.084964   0.004342   5.11   (0.07687, 0.09392)
conc.additive              0.70777    0.04815    6.8   (0.6194, 0.8087)

shrinkage (1 - SD of the units' scores): ka 0.8 %, CL -0.7 %
```

`result.individual()` gives each unit's scores and parameter values (its EBEs).

## Options

The options of [`optimize`](optimize.md#options): `starts`, `seed`, `maxiter` (200 steps of
the outer fit), `rtol`, `atol`, `timeout`, `progress`, `checks`.

## Limits

- The Laplace approximation is accurate when each unit's likelihood is close to normal in its
  scores; with few, very nonlinear or mostly censored measurements per unit it can be off.
  [`nuts`](nuts.md) does not approximate.
- A written likelihood (`likelihood` lines) cannot be combined with parameters that vary
  between units yet; use an error model.
- See [not yet](../not-yet.md) for what the population model does not have yet (variability
  between occasions, covariates that change with time, gamma and beta distributions between
  units).

## References

- Beal SL, Sheiner LB. *NONMEM Users Guides.* (FOCE and the interaction option.)
- Wang Y. Derivation of various NONMEM estimation methods. *J Pharmacokinet Pharmacodyn*
  2007;34:575–593.
- Lindstrom MJ, Bates DM. Nonlinear mixed effects models for repeated measures data.
  *Biometrics* 1990;46:673–687.
- Savic RM, Karlsson MO. Importance of shrinkage in empirical Bayes estimates for diagnostics.
  *AAPS J* 2009;11:558–569.
- Beal SL. Ways to fit a PK model with some data below the quantification limit.
  *J Pharmacokinet Pharmacodyn* 2001;28:481–504.
