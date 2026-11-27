# Install

tenqsp-fit needs Python 3.12 or later. It installs TenQSP (0.9.2 or later) with it, with
TenQSP's native solver (numba): one command installs everything a fit needs.

```bash
pip install tenqsp-fit                 # the library
pip install "tenqsp-fit[plot]"         # with plots (matplotlib)
```

Keep the quotes (zsh, the default shell on macOS, reads the brackets otherwise).
`pip install -U tenqsp-fit` upgrades an installed tenqsp-fit, and TenQSP with it when the new
version needs a newer TenQSP. For TenQSP's labelled array output, add `"tenqsp[xarray]"`.

To work on tenqsp-fit itself, from a clone of the repository:

```bash
git clone https://github.com/tenqsp/tenqsp-fit
cd tenqsp-fit
python -m pip install -e ".[dev,plot]"
pytest -m "not slow"                   # about 20 minutes
```

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
exact derivatives), [JAX](https://jax.readthedocs.io), [SciPy](https://scipy.org) (the
optimizers, L-BFGS-B and trust-region least squares, and statistics),
[NumPyro](https://num.pyro.ai) (the NUTS sampler), [ArviZ](https://python.arviz.org)
(posterior summaries), [pandas](https://pandas.pydata.org) (data and tables) and
[numba](https://numba.pydata.org) (TenQSP's native solver, which `options backend=auto`
uses wherever the model allows). Plots need
[matplotlib](https://matplotlib.org), installed with the `plot` extra (as above, or
`pip install matplotlib`).

It runs on the CPU. On a machine with a GPU but no CUDA build of `jaxlib`, JAX prints a
one-line note that it falls back to the CPU.
