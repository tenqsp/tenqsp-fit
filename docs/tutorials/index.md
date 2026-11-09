# Tutorials

Each tutorial fits one model and answers one question a modeller would ask of the data. The
models follow [TenQSP's tutorials](https://tenqsp.github.io/tenqsp/tutorials/) where they can:
the same model, now fitted. Every section ends with checks: against a published or reference
fit for real data, and against the values the data were simulated from otherwise, so you can
see that each answer is right.

| | Tutorial | Question |
|---|---|---|
| 1 | [Theophylline](01_theophylline.ipynb) | does one subject, then all twelve, give the textbook values; what does a population fit add? |
| 2 | [Transit delay and absorption](02_transit_absorption.ipynb) | can the data tell a transit delay from slow absorption; which interval to trust? |

Each tutorial is a plain Python file in the repository (`docs/tutorials/`), run by the test
suite, so it cannot go out of date. To open one as a notebook:
`python docs/tutorials/build.py 01` writes `docs/tutorials/01_theophylline.ipynb` (it needs
the `docs` extra: `pip install -e ".[docs]"`).
