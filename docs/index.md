# tenqsp-fit

**Fitting [TenQSP](https://tenqsp.github.io/tenqsp/) models to data, with a diagnosis of what
the data can and cannot determine.**

A fit is written as text after the model, in the way NONMEM, Monolix and nlmixr2 users write
one, or as the same statements in Python:

```text
model theoph
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL

fit theoph_pop
data "theoph.csv" with ID=Subject, TIME=Time, DV=conc
unit ID
doses Dose at t=0 into depot
observe conc = DV

estimate ka ~ lognormal(median=1.5, sdlog=0.5)     # varies between subjects
estimate CL ~ lognormal(median=0.04, sdlog=0.5)
estimate ke = 0.08                                  # one value for all subjects
error conc ~ additive(1)

method laplace
```

```python
import tenqsp_fit as tf

result = tf.fit(tf.load("theoph_pop.tq"))
print(result)          # estimates, standard errors, and what the doctor found
```

## What it does

- **Three methods.** `optimize` fits one set of parameters (one experiment, or the data
  pooled) by least squares or maximum likelihood, from several starts. `laplace` fits a
  population, with parameters that vary between units, by the Laplace approximation
  (FOCE-I). `nuts` samples the posterior with the No-U-Turn Sampler.
- **The model is any TenQSP model**: compartments, or structured populations over age, size
  or depth, with doses, events and categories. TenQSP gives exact derivatives of every
  output: `optimize` and `nuts` use exact gradients, and `laplace` builds its gradient from
  them by central differences.
- **A diagnosis with every fit.** The doctor reports quantities the data cannot tell apart
  (ridges), quantities that are only weakly determined, several optima, estimates at a
  bound, shrinkage, and whether the solver's tolerance was tight enough. The passport writes
  all of it, with every default and version, into one document.
- **Results** as pharmacometricians expect them: a parameter table with standard errors,
  individual estimates, goodness-of-fit plots, a visual predictive check, profile
  likelihoods, NONMEM-style tables, predictions for new regimens, and a results folder.

## Where to start

- [Install](start/install.md)
- [Quickstart: a population PK fit](start/quickstart-population.md), theophylline in
  12 subjects, about a minute
- [Quickstart: a QSP model with a structured population](start/quickstart-qsp.md), bacteria,
  phage and infected cells by infection age, and what the data cannot determine

tenqsp-fit is in early development; its first release is 0.1.
