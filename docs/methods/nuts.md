# method nuts

The posterior distribution of the estimated quantities, by draws from the No-U-Turn Sampler
(NUTS, [NumPyro](https://num.pyro.ai)'s), with convergence checks from
[ArviZ](https://python.arviz.org).

Use it when the curvature at the estimates is a poor summary of what the data allow (weakly
determined quantities, skewed or bounded posteriors), when prior knowledge should enter the
fit, or when the Laplace approximation is in doubt. It works with and without parameters
that vary between units.

## What it samples

The estimated quantities on their estimation scales \(u\) (log, logit or as is; as in
[`optimize`](optimize.md)) and, when parameters vary between units, each unit's standard-normal
scores (non-centred; see [`laplace`](laplace.md) for the scores). The potential, minus the log
posterior, is

$$
U = \tfrac12\,(-2LL) + \tfrac12 \lVert E \rVert^2 - \log p(x) - \log \left|\frac{dx}{du}\right|,
$$

with \(-2LL\) the likelihood given the units' values (error models, censored measurements),
\(E\) the units' independent scores (correlated through `correlate` by the Cholesky factor of
\(R\)), \(p(x)\) the priors and the last term the Jacobian of the estimation scales.

## Priors

Every estimated quantity has a prior: the one written (`prior ke ~ gamma(mean=0.08, cv=0.5)`)
or a default, centred on the starting value, by its estimation scale:

| Quantity | Default prior |
|---|---|
| positive (log scale) | `lognormal(median=start, sdlog=2)`: 95 % within a factor 50 of the start |
| any sign | `normal(mean=start, sd=...)`, the SD twice the start's size; a start of 0 needs a written prior |
| bounded, `in (lo, hi)` | `uniform(lower=lo, upper=hi)` |
| the correlations of a `correlate` block | LKJ(2) on the block's correlation matrix |

`problem.summary()` lists every prior and marks the defaults; the result's table does too.
Written priors are also used by [`optimize`](optimize.md) and [`laplace`](laplace.md) (which
then give the posterior mode); the defaults by `nuts` only.

## How a run goes

1. **The starting point**: the fast fit, [`laplace`](laplace.md) when parameters vary,
   otherwise [`optimize`](optimize.md), with the written priors.
2. **The shape of the posterior there**: the curvature of \(U\) (central differences of its
   exact gradient). Its inverse is NUTS's mass matrix, held fixed through the run; the chains
   start one curvature-SD away from the starting point in random directions. Starting with the
   right shape saves the sampler from learning it during warmup, which is slow when quantities
   are correlated.
3. **Warmup**: the step size is tuned (dual averaging, target acceptance 0.8).
4. **The draws**. All chains run as one batched computation. Ctrl-C (or `options timeout=`)
   stops the run and keeps the draws so far.

The maximum tree depth is 10.

## Convergence

Per quantity: rank-normalized split R-hat and bulk and tail effective sample sizes (ArviZ).
Per chain: divergences after warmup, energy BFMI, how often a trajectory reached the maximum
tree depth, and the mean steps per draw (`result.convergence()`). The doctor reports a problem
when

| Check | Limit |
|---|---|
| R-hat | above 1.01 |
| effective sample size (bulk or tail) | below 400 |
| divergent transitions after warmup | any |
| energy BFMI | below 0.3 in a chain |
| maximum tree depth reached | any |

and notes how much the data narrowed each prior (a posterior almost as wide as its prior
means the prior, not the data, determines the quantity).

## An example

The exponential decay of [`optimize`](optimize.md#an-example), with the residual SD estimated
and a prior on k:

```text
estimate k = 0.3
error amount ~ additive(0.1)
prior k ~ lognormal(median=0.5, sdlog=1)
method nuts chains=4 warmup=500 samples=500
```

```text
decay: method nuts, sampled; 4 chains x 500 draws (after 500 warmup), 0 divergences, R-hat <= 1.005, ESS >= 662; 41 s
started from optimize: -2LL -40.4211 (6 measurements); -2LL at the posterior medians -39.8713

quantity                   median        SD   95 % interval           R-hat    ESS  prior
k                         0.49349   0.00914   (0.4761, 0.5136)        1.005    662  lognormal(median=0.5, sdlog=1)
amount.additive          0.010495   0.00489   (0.005892, 0.02345)     1.003    874  lognormal(median=0.1, sdlog=2) (default)

doctor: 4 notes
note     convergence: R-hat, effective sample sizes, divergences, BFMI and tree depth: all within their limits.
note     prior and data: the data narrowed every prior to less than 50 % of its width.
note     tolerance check: at rtol 1e-08, -2 log posterior changes by at most 0.00034 over 20 draws; rtol 1e-06 is enough.
note     gradient check: exact gradients agree with differences (at rtol 1e-08) to 0.014 % (3 directions).
```

The median of k agrees with the maximum-likelihood estimate (0.49354). The residual SD's
posterior is wider and higher than its estimate (0.0083, SE 0.0024): six measurements say
little about an SD, and the posterior shows the skew that a standard error cannot.

`result.draws` holds every draw after warmup; `result.to_arviz()` gives them as ArviZ data for
ArviZ's own plots and summaries; `result.plot_trace()` draws each chain's path.

## Options

On the `method` line:

| Option | Default | Means |
|---|---|---|
| `chains` | 4 | the number of chains |
| `warmup` | 1000 | warmup draws per chain (discarded) |
| `samples` | 1000 | draws per chain kept |

On the `options` line: `seed` (the chains' random numbers and starting points), `rtol`, `atol`,
`timeout`, `progress`, `checks`; see [running a fit](../guide/running.md).

## What it reports

`result.table()`: the posterior median, mean, SD, 95 % interval (2.5th and 97.5th
percentiles), R-hat, effective sample sizes and the prior of each quantity. `result.estimates`
are the posterior medians; `individual()`, `plot_gof()`, `plot_individual()` and `tables()`
are at the posterior medians. `vpc()` simulates each study from a different draw, and
`predict()` gives posterior intervals. There is no profile likelihood after `nuts`: the
posterior itself gives each quantity's interval.

## Limits

- Slower than [`optimize`](optimize.md) and [`laplace`](laplace.md): each draw runs the model
  several times (the mean steps per draw), and a population adds every unit's scores to what
  is sampled.
- The mass matrix is fixed from the curvature at the starting point; a posterior whose shape
  changes much across its range (a funnel, a banana) can still give divergences, and the
  doctor reports them.

## References

- Hoffman MD, Gelman A. The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian
  Monte Carlo. *J Mach Learn Res* 2014;15:1593–1623.
- Phan D, Pradhan N, Jankowiak M. Composable effects for flexible and accelerated probabilistic
  programming in NumPyro. arXiv:1912.11554, 2019.
- Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C. Rank-normalization, folding, and
  localization: an improved R-hat for assessing convergence of MCMC. *Bayesian Anal*
  2021;16:667–718.
- Betancourt M. Diagnosing suboptimal cotangent disintegrations in Hamiltonian Monte Carlo.
  arXiv:1604.00695, 2016 (energy BFMI).
- Lewandowski D, Kurowicka D, Joe H. Generating random correlation matrices based on vines and
  extended onion method. *J Multivar Anal* 2009;100:1989–2001 (the LKJ distribution).
