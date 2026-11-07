# Install

tenqsp-fit needs Python 3.12 or later and TenQSP. Until its first release it is installed from
a clone of the repository:

```bash
git clone https://github.com/tenqsp/tenqsp-fit
cd tenqsp-fit
python -m pip install -e ".[dev,plot]"
pytest -m "not slow"               # about 20 minutes
```

The `pip install` line also installs TenQSP (0.7.0 or later) from PyPI.

## Check the installation

```python
import tenqsp_fit as tf

p = tf.Problem.from_text("""
model check
param k = 0.5
d_t(A) = -k*A
A(0) = 1
observable amount = A

fit check
data "check.csv"
observe amount = DV
estimate k = 0.3
error amount ~ additive(fix(0.01))
method optimize
""")
```

with a `check.csv` of two columns, for example:

```text
TIME,DV
1,0.61
2,0.37
4,0.14
```

`tf.fit(p).estimates["k"]` is close to 0.5.

## What gets installed

tenqsp-fit is built on [TenQSP](https://tenqsp.github.io/tenqsp/) (the models and their
exact derivatives), [JAX](https://jax.readthedocs.io), [Optimistix](https://docs.kidger.site/optimistix/)
(Levenberg–Marquardt), [SciPy](https://scipy.org) (BFGS and statistics),
[NumPyro](https://num.pyro.ai) (the NUTS sampler), [ArviZ](https://python.arviz.org)
(posterior summaries) and [pandas](https://pandas.pydata.org) (data and tables). Plots need
[matplotlib](https://matplotlib.org), installed with the `plot` extra (as above, or
`pip install matplotlib`).

It runs on the CPU. On a machine with a GPU but no CUDA build of `jaxlib`, JAX prints a
one-line note that it falls back to the CPU.
