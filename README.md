# tenqsp-fit

[![CI](https://github.com/tenqsp/tenqsp-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/tenqsp/tenqsp-fit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tenqsp-fit.svg)](https://pypi.org/project/tenqsp-fit/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24040995.svg)](https://doi.org/10.5281/zenodo.24040995)

**Fitting [TenQSP](https://github.com/tenqsp/tenqsp) models to data, with a diagnosis of what
the data can and cannot determine.**

Version 0.1, the first release. It fits individual and population models by optimization
(`method optimize`), the Laplace approximation (`method laplace`, FOCE-I) and NUTS
(`method nuts`); reports ridges and what the data cannot determine (the doctor, profile
likelihoods); and gives predictions with their uncertainty and visual predictive checks.
A fit is written as text after the TenQSP model, or built in Python.

```bash
pip install tenqsp-fit
```

Documentation: <https://tenqsp.github.io/tenqsp-fit/>

## Licence and credit

tenqsp-fit is free and open source under the [Apache License 2.0](LICENSE). It was created by
Dongwoo Chae, MD, PhD, Yonsei University College of Medicine
([ORCID 0000-0002-7675-3821](https://orcid.org/0000-0002-7675-3821)); copyright 2026 Yonsei
University Health System (see [`NOTICE`](NOTICE)). Its code is written with AI assistance
(Claude Code, Anthropic) under the author's design, review and testing.
