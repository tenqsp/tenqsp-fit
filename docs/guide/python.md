# The same problems in Python

Every statement of the fit text is one method of `tf.Problem`, with the same words. A problem
built in Python prints as fit text, and fit text reads back to the same problem, so either
can be kept.

## One call per line

| Fit text | Python |
|---|---|
| `fit theoph_pop` (after the model) | `p = tf.Problem(model, name="theoph_pop")` |
| `data "theoph.csv" with ID=Subject, TIME=Time` | `p.data("theoph.csv", ID="Subject", TIME="Time")`, or `p.data(df)` for a pandas DataFrame |
| `unit ID` | `p.unit("ID")` |
| `doses into depot` | `p.doses(into="depot")` |
| `doses CMT 1 into depot, 2 into centr` | `p.doses(cmt={1: "depot", 2: "centr"})` |
| `doses Dose at t=0 into depot` | `p.doses(column="Dose", at=0, into="depot")` |
| `observe conc = DV where CMT == 2` | `p.observe("conc", "DV", where="CMT == 2")` |
| `ignore DV where DVID == 3` | `p.ignore("DV", where="DVID == 3")` |
| `censor conc where BLQ == 1 below LLOQ` | `p.censor("conc", below="LLOQ", where="BLQ == 1")` |
| `estimate ke = 0.08` | `p.estimate("ke", 0.08)` |
| `estimate F = 0.5 in (0, 1)` | `p.estimate("F", 0.5, bounds=(0, 1))` |
| `estimate n = 2 in (1, inf)` | `p.estimate("n", 2, bounds=(1, None))` (or `np.inf`) |
| `estimate ka ~ lognormal(median=1.5, sdlog=0.5)` | `p.estimate("ka", tf.lognormal(median=1.5, sdlog=0.5))` |
| `... sdlog=fix(0.6)` | `... sdlog=tf.fix(0.6)` |
| `... median=CL_pop*(WT/70)^0.75` | `... median="CL_pop*(WT/70)^0.75"` (a formula is text) |
| `fix V = 30` | `p.fix("V", 30)` |
| `correlate ka, CL = -0.3` | `p.correlate("ka", "CL", start=-0.3)` |
| `CL = CL_pop*(WT/70)^beta` | `p.define("CL", "CL_pop*(WT/70)^beta")` |
| `error conc ~ combined(0.5, 0.1)` | `p.error("conc", tf.combined(0.5, 0.1))` |
| `likelihood conc: <log-density>` | `p.likelihood("conc", "<log-density>")`, or a Python function (below) |
| `prior ke ~ lognormal(median=0.08, sdlog=0.5)` | `p.prior("ke", tf.lognormal(median=0.08, sdlog=0.5))` |
| `method nuts chains=4` | `p.method("nuts", chains=4)` |
| `options starts=8 seed=1` | `p.options(starts=8, seed=1)` |

The model is a TenQSP model: `tq.load("model.tq")`, `tq.Model.from_text(...)`, or one built
with TenQSP's Python API. Unknown keywords raise, with the nearest name.

```python
import tenqsp as tq
import tenqsp_fit as tf

m = tq.Model.from_text("""
model check
param k = 0.5
d_t(A) = -k*A
A(0) = 1
observable amount = A
""")
p = tf.Problem(m, name="check")
p.data("check.csv")
p.observe("amount", "DV")
p.estimate("k", 0.3)
p.error("amount", tf.additive(tf.fix(0.01)))
p.method("optimize")
print(p.to_text())
```

```text
fit check
data "check.csv"
observe amount = DV

estimate k = 0.3
error amount ~ additive(fix(0.01))

method optimize
```

`p.to_text(model=True)` puts the model text first, so the result is one file for `tf.load`.
A table given as a DataFrame, or a likelihood given as a Python function, is not text; it is
written as a comment.

## Reading text

- `tf.load("theoph_pop.tq")`: a model followed by its fit part. The data file is found next to
  the `.tq` file.
- `tf.load("theoph_fit.tq", model="theoph_model.tq")`: the fit part alone, with the model in
  its own file (or a TenQSP model).
- `tf.Problem.from_text(text)`, `tf.Problem.from_text(text, model=...)`: the same from a
  string; data files are found relative to the working directory.

Two problems that say the same thing are equal (`==`), whether they were read from one file,
from two, or built in Python.

## Fitting

`tf.fit(p)` fits by the problem's method. `tf.fit(p, method="nuts")` uses another method
without changing the problem, and `progress="text"`, `"plot"` or `"none"` sets what is shown
while it runs (see [running a fit](running.md)).

## The likelihood as a function

`tf.objective(p)` is −2 log-likelihood of the problem as a JAX function of its estimated
quantities, for an optimizer or a sampler of your own:

```python
import jax
import numpy as np

obj = tf.objective(p)
print(obj)
print(obj.names, obj.x0)
print(obj(obj.x0))
print(jax.grad(obj)(obj.x0))
print(obj.predictions(np.array([0.5])))
```

```text
<tenqsp_fit objective (-2 log-likelihood) of ['k']: 1 unit, 3 measurements>
('k',) [0.3]
728.589863380331
[-9747.68976485]
{'amount': Array([[0.60653042, 0.36787929, 0.13533564]], dtype=float64)}
```

- `x` is in `obj.names` order, on the quantities' own scale (`k` itself, not log k);
  `obj.x0` are the starts and `obj.bounds` the bounds.
- `jax.grad`, `jax.jit` and `jax.jacfwd` work on it. The gradient is exact: TenQSP gives the
  derivatives of the solution.
- `obj.predictions(x)`: each link's predictions at its measurements, one row per unit, on the
  link's scale. `obj.terms(x)`: each measurement's contribution to −2LL.
  `obj.curves(x, times)`: each link's prediction on a time grid.
- A run that fails gives NaN; `obj.status(x)` says why, per unit. `obj.unpack(x)` gives `x`
  by name.

With SciPy:

```python
from scipy.optimize import minimize

res = minimize(lambda x: float(obj(x)), obj.x0,
               jac=lambda x: np.asarray(jax.grad(obj)(x)), method="BFGS")
print(res.x, res.fun)
```

```text
[0.49487271] -22.05362656350535
```

When parameters vary between units, `tf.objective(p)` is the Laplace approximation (FOCE-I)
of −2LL, the objective of `method laplace`. Its `obj.modes(x)` gives each unit's
conditional modes (normal scores) and `obj.individual(x)` each unit's parameter values. Its
derivative needs second derivatives of the solution, which TenQSP does not give, so
`jax.grad` of it is not its gradient; `method laplace` computes the gradient another way.

## A likelihood as a Python function

`p.likelihood(link, function)` takes a function `density(dv, prediction, values)` of JAX
arrays that returns the log-density of each measurement; `values` holds the estimated and
fixed quantities by name. A Student t with 4 degrees of freedom:

```python
import jax.numpy as jnp
import pandas as pd

def t_density(dv, prediction, values):
    z = (dv - prediction) / values["s"]
    return -2.5 * jnp.log1p(z**2 / 4) - jnp.log(values["s"])

q = tf.Problem(m, name="check_t4")
q.data(pd.DataFrame({"TIME": [1, 2, 4], "DV": [0.61, 0.37, 0.14]}))
q.observe("amount", "DV")
q.estimate("k", 0.3)
q.estimate("s", 0.05)
q.likelihood("amount", t_density)
q.method("optimize")
print(tf.fit(q, progress="none").table())
```

```text
  quantity  estimate        SE      RSE %  CI 2.5 %  CI 97.5 % also
0        k  0.494858  0.001567   0.316678  0.491796   0.497939
1        s  0.001374  0.000664  48.349696  0.000533   0.003545
```

As with a written formula, −2LL is −2 times the sum of what the function returns.

## The data as read

`p.units()` gives one `tf.Unit` per unit: its `id`, a TenQSP `scenario` (doses and
measurement times), its `observations` by link (`tf.Observations`: `times`, `values`, the
data `rows`, and `below` and `censored` for a censored link) and its `covariates`.
`p.table()` gives the same as one NONMEM-style table, and `p.provenance()` what is recorded
about the data with every fit. See [the data](data.md#the-data-as-the-fit-reads-them).

## The prior predictive check

Before fitting: what data do the priors expect? `tf.prior_predictive(p, n=200)` simulates
`n` studies with the data's doses, times and limits, each with its quantities drawn from the
priors (written, or the defaults of `method nuts`; see [priors](estimates.md#priors)), new
units from their distributions, and residual error. It shows the data's percentiles over
the simulated ones, as a VPC. For the problem `check` above, with the default prior of `k`:

```python
v = tf.prior_predictive(p, n=100)
print(v)
print(v.draws.head(3))
```

```text
<prior predictive check: 100 simulations, percentiles [5.0, 50.0, 95.0], 3 bins; .table, .figure, .draws>
          k   ran
0  0.385771  True
1  0.230344  True
2  1.079904  True
```

`.draws` are the quantities drawn, with `ran` False where the model did not run with them;
such studies are left out of the bands and counted in `.failed`. `.figure` is the plot and
`.table` the numbers.
