# The doctor and the passport

Every `tf.fit` ends by examining the fit: what the data determine, what they do not, and
whether the numbers can be trusted. The findings end the printout of the result;
`result.doctor()` returns them, and `result.doctor().table()` gives them as a DataFrame
(check, severity, finding, what to do).

```text
doctor: 3 notes
note     determination: condition number of the curvature 39.1; no ridge, every quantity determined within a factor 10.
note     optima: one start: other optima were not looked for.
         what to do: options starts=8 looks from 8 starts.
note     tolerance check: at rtol 1e-08 the -2LL changes by 2.2e-05 and the estimates by less than 0.001 SE; rtol 1e-06 is enough.
```

Each finding has a check, a severity, what was found with its numbers, and often what to do.

| Severity | Means |
|---|---|
| **problem** | the results are wrong or meaningless as they stand: do not report them before acting on it |
| **warning** | read the results with care |
| **note** | for the record: what was checked, and that it passed |

The findings are listed most severe first.

## What the doctor looks at

### Ridges: quantities the data cannot tell apart

The standard errors come from the curvature of −2LL at the estimates, on the scale each
quantity was estimated on (log for a positive quantity). The doctor looks at that curvature in
every direction. A direction whose curvature (as a ratio of singular values) is below 1/1000 of
the strongest is a **ridge**: along it the fit does not change, so the data do not say where on
it the estimates should be. A ridge is a **problem**.

The finding names the quantities that move along the ridge (those with a weight of at least
0.3 in it):

- one quantity: it is not determined at all;
- two quantities on the log scale: which combination *is* determined, as a product with
  powers. From the [QSP quickstart](../start/quickstart-qsp.md), with bacteria counts only:

  ```text
  problem  ridge: beta and k are not determined apart, only beta^0.88 x k is (curvature ratio 8.3e-04).
           what to do: measure what tells them apart, write a prior on some of them, or estimate the determined combination (a formula after fit).
  ```

- more quantities: how they move together along the ridge, on their estimation scales. From
  a bacteria-only fit stopped early:

  ```text
  problem  ridge: g, log10(bact).additive, k and beta are not determined apart: the fit does not change when log g, log log10(bact).additive, log k, log beta move together as +1.00 : -0.50 : -0.41 : -0.38 (curvature ratio 0.0e+00).
  ```

Several flat directions that share quantities are one finding.

What to do: measure what tells the quantities apart (another output, another design), write a
prior on one of them from other knowledge (see [estimates and distributions](estimates.md)), or
fix one and say so.

When the curvature is not positive in any direction, the fit did not reach a minimum
(**problem**, "not at a minimum"). When the standard errors could not be computed at all, the
doctor says why (**problem**, "standard errors").

### Weakly determined quantities

A **warning** for each quantity, outside a ridge, whose 95 % interval

- spans more than a factor 10, for a quantity on the log scale;
- covers more than half of its range, for a quantity within bounds (a correlation, say);
- or, for others, whose RSE is above 50 %.

Quantities correlated with it above 0.9 are named as its partners.

### The condition number

A **note**: the largest curvature over the smallest. Without a ridge or a weak quantity it says
"no ridge, every quantity determined within a factor 10".

### Several optima

With several starts, the starts' final −2LL are grouped: starts within 0.1 of each other
reached the same fit. Two or more groups are a **warning**, with each group's −2LL and how
many starts reached it. From the QSP quickstart:

```text
warning  several optima: the starts ended at 2 different fits: -2LL -34.5421 (3 starts), -2LL -24.8131 (1 start); the best is reported.
         what to do: if few starts reached the best, run more starts (options starts=); compare the fits in result.starts.
```

When the next fit is within 3.84 of the best (the profile's cut-off), the data hardly prefer
one fit: the standard errors describe the best fit only, and the advice adds that a
[profile](results.md#profile-likelihood) gives an interval covering the values of both.

With one start, a note says other optima were not looked for. `result.plot_waterfall()` shows
the same.

### Convergence

A fit that ended at the iteration limit is a **problem**: the estimates are where it stopped,
not a minimum.

```text
problem  convergence: the fit ended with status 'max iterations': the estimates are where it stopped, not a minimum.
         what to do: more iterations (options maxiter=), better starts, or bounds; check result.plot_trace().
```

A fit stopped by Ctrl-C or `options timeout` gets a note instead, and the other findings are
about where it stopped.

### At a bound

A **warning** when an estimate ended within 1 % of the width of its interval from a bound
written for it (`estimate k = 0.3 in (0.01, 0.4)`), as NONMEM's "near its boundary"; for a
bound on one side (`in (1, inf)`), within 1 % of the bound's or the estimate's size. It is
also a Python warning when the fit ends.

```text
warning  at a bound: k ended at 0.4, near its upper bound 0.4: the bound may be too tight, or the data do not determine k. Widen the bound, or fix k.
```

### A spread near 0, shrinkage

With parameters that vary between units:

- **spread near 0** (warning): a spread (`sd`, `sdlog`, `cv`, `sdlogit`) below 0.01, or a
  variance below 0.0001. The parameter hardly varies between units, and its interval is
  unreliable this close to the boundary. Consider a spread of 0.
- **shrinkage** (warning): above 30 % for a parameter. Each unit's own value is pulled
  towards the typical value, so plots of the units' values, and of their relation to
  covariates, are unreliable. Judge the model by the VPC instead.

### Profiles

When `result.profile(name)` has run, its findings join the doctor's:

- **profile in pieces** (warning): the 95 % set has more than one piece, so the likelihood has
  optima the data cannot tell apart. Report the whole set.
- **better fit in the profile** (problem): a profile point fits better than the estimate; the
  fit missed an optimum. Refit with more starts, or from that point.

See [profile likelihood](results.md#profile-likelihood).

### The tolerance check

The solver's tolerance decides how exactly the model is solved, and so −2LL and its gradient.
The doctor computes −2LL again at the estimates with `rtol` divided by 100 (and `atol`, when
one is written). It is a **problem** if −2LL changes by more than 0.1, or if the change of its
gradient would move an estimate by more than 0.1 SE; otherwise a note:

```text
note     tolerance check: at rtol 1e-08 the -2LL changes by 2.2e-05 and the estimates by less than 0.001 SE; rtol 1e-06 is enough.
```

What to do when it fails: fit again with a tighter `rtol`, as the finding says.

### The gradient check

The optimizers and the sampler use TenQSP's exact derivatives. The doctor checks them against
central differences, at `rtol` / 100, along 3 random directions, at 3 points drawn from the
estimates' approximate distribution where −2LL is within the 99 % chi-square bound of the best
(the region the data allow). A difference above 1 % is a **warning**: not consistent enough for
NUTS, which needs exact gradients. Not run after `laplace`, whose gradient is itself computed
by differences.

The tolerance and gradient checks solve the model again: a few seconds for small models,
a minute or two for large ones. `options checks=quick` leaves them out, and a note says so.

### Jumps in the model

A **warning** for each part of the model that can make −2LL jump: an extinction group (a
population set to 0 when it crosses its threshold), a lag time on doses (−2LL jumps when a
dose time plus the lag passes a measurement time), and the jumps TenQSP's `validate()` notes
(`floor`, `ceil`, `rem`, conditions that switch cell by cell). The gradient methods are safe
only if nothing switches near the estimate: check a profile of the quantities involved.

### Failed runs

A **note** with the number of model runs that failed during the fit (steps the optimizer
refused). A few are normal: an optimizer tries values far out of range.

### After `nuts`

- **convergence**: R-hat above 1.01 (problem), divergent transitions after warmup (problem),
  effective sample size below 400, energy BFMI below 0.3, draws at the maximum tree depth (10)
  (warnings); a note when all are within their limits.
- **prior and data** (warning): a quantity whose posterior SD is above half its prior's (on the
  estimation scale): the data say little about it, the prior decides it.
- **weakly determined** (warning), as above, from the posterior.
- the tolerance check at 20 draws, and the gradient check at 3 draws.

## The passport

```python
pp = result.passport()
print(pp)                     # as text
pp.markdown()                 # as Markdown
pp.to_dict()                  # as plain data
pp.save("passport.md")        # or "passport.json"
```

One document per fit: what was run, and what the data determine. `result.save(folder)` writes
it as `passport.md` and `passport.json`. Its sections:

| Section | Holds |
|---|---|
| The run | method, optimizer, status, −2LL, options, the SHA-256 of the problem text and of the data as read, versions, platform, time |
| Each quantity | estimate, SE, 95 % interval, whether it is determined (`yes`, `weakly`, `not (ridge)`), its profile's shape when one was computed, its prior |
| Ridges and determined combinations | the ridges, and the condition number |
| The doctor's findings | the findings above, other than the numerical ones |
| Numerical checks | the tolerance and gradient checks, failed runs, starts redrawn |
| Named predictions | each `result.predict(..., name=...)` |
| TenQSP's reports | the model's `validate()`, `diagnostics()` of a run at the estimates with how well the ledger closed, and `explain()` with its list of numerical safeguards |

A profile's shape is `closed` (past the cut-off on both sides), or for a side that is not:
`flat` (−2LL rose by less than 1) or `not reached` (rising, but the profile stopped short:
profile further). A set in pieces says how many, and a better fit found is added.

The passport of the theophylline fit, after a named prediction (start and end):

```text
Passport: theoph_pop
====================

The run
-------
problem theoph_pop: method laplace (BFGS, FOCE-I), converged
-2LL 353.9837 (132 measurements, 12 units)
options: none written (rtol 1e-06 unless written)
problem text (model and fit) SHA-256 722cbb35c100a7808acc33040fe26d8ef91eace6d5ee787667b7777ee36f38eb
data as read SHA-256 a1d839ceef82514b7de1befce5e24e811ebcc9b2868a5dd238b68e2431a608f9
versions: tenqsp_fit 0.0.1, tenqsp 0.6.1, jax 0.11.2, numpy 2.5.3, pandas 3.1.0, python 3.12.16, scipy 1.18.1
platform Linux x86_64; 41 s
all units run as one build of the model (TenQSP pads the shorter ones exactly, with slots at t = +inf)

Each quantity
-------------
     quantity  estimate       SE  CI 2.5 %  CI 97.5 % determined profile prior
    ka.median  1.619518 0.320713  1.098549   2.387548        yes
     ka.sdlog  0.656425 0.150018  0.419421   1.027354        yes
    CL.median  0.039544 0.002353  0.035190   0.044436        yes
     CL.sdlog  0.167511 0.036433  0.109372   0.256554        yes
           ke  0.084964 0.004342  0.076865   0.093916        yes
conc.additive  0.707770 0.048153  0.619412   0.808733        yes

Ridges and determined combinations
----------------------------------
no ridge: no direction weaker than 1e-3 of the strongest
condition number of the curvature 39.1; no ridge, every quantity determined within a factor 10.

The doctor's findings
---------------------
note, optima: one start: other optima were not looked for. What to do: options starts=8 looks from 8 starts.

Numerical checks
----------------
note, tolerance check: at rtol 1e-08 the -2LL changes by 2.2e-05 and the estimates by less than 0.001 SE; rtol 1e-06 is enough.
failed runs during the fit: 0; starts redrawn: 0
gradient check: not for laplace (its gradient is itself a difference)

Named predictions
-----------------
4 mg/kg q12h: conc from 0 to 48 (bolus 4 into depot at t = 0; bolus 4 into depot at t = 12; bolus 4 into depot at t = 24); approximate (from the standard errors), 200 draws; with a new unit's 90 % range

TenQSP's reports
----------------
validate(): 2 issue(s):
  [note] compartment 'depot': no initial condition given; starts at 0.
  [note] compartment 'centr': no initial condition given; starts at 0.
...
```

The hashes identify the run: the same problem text, the same data and the same versions give
the same fit. Keep the passport with a report of the results; a reviewer can see from it what
was estimated, what was fixed, what the data could and could not determine, and which checks
passed.
