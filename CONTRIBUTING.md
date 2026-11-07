# Contributing to tenqsp-fit

Thank you for helping. Contributions of every size are welcome: a report of something that
confused you, a fix to the documentation, a new example, or a change to the library.

## Reporting a problem or asking for something

Open an issue. For a problem, include the smallest problem that shows it (the model and fit
text, or the Python code, and a few rows of data), what you expected, what happened, and the
output of
`python -c "import tenqsp, tenqsp_fit, jax; print(tenqsp.__version__, tenqsp_fit.__version__, jax.__version__)"`.

## Setting up

```bash
git clone https://github.com/tenqsp/tenqsp-fit
cd tenqsp-fit
python -m pip install -e ".[dev,plot,docs]"
pytest -m "not slow"
```

## Before you open a pull request

```bash
ruff check src tests validation && ruff format --check src tests validation
pytest                       # all tests, about 35 minutes
mkdocs build --strict        # if you changed the documentation
```

- **Every change to a method comes with a test against something known**: a closed-form
  answer, another program's result on the same data (R's `nlme`, SciPy), or a simulation
  where the truth is known. A test that only repeats what the code computes does not count.
- **Every output shown in the documentation comes from a real run.** The scripts in
  `docs/examples` run the quickstarts and make their figures.
- **Messages are for modellers.** Say what is wrong in the problem's own terms, on which line,
  and what to write instead. Unknown options and keywords raise; nothing is silently ignored.
- **Nothing hidden.** Every default the fit used appears in `problem.summary()` and in the
  passport.
- New source files start with the licence header:
  ```python
  # SPDX-FileCopyrightText: 2026 Yonsei University Health System
  # SPDX-License-Identifier: Apache-2.0
  ```
- Something missing in TenQSP is fixed in TenQSP, not worked around here.

## Sign your commits (Developer Certificate of Origin)

tenqsp-fit uses the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) instead of a contributor licence agreement. By adding a `Signed-off-by` line to a
commit, you certify that you wrote the change, or otherwise have the right to submit it,
under the project's licence (Apache-2.0). Git adds the line for you:

```bash
git commit -s -m "Explain what the change does"
```

which ends the message with `Signed-off-by: Your Name <your.email@example.org>`, using your
real name. Pull requests whose commits are not all signed off cannot be merged. To sign off
commits you already made: `git rebase --signoff main`, then push again.

## Licence and credit

Contributions are licensed under the Apache License 2.0, like the rest of tenqsp-fit.
Contributors are credited in the changelog and in the git history. tenqsp-fit was created by
Dongwoo Chae; that credit, in the `NOTICE` file, stays with every copy.

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
