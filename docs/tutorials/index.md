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
| 3 | [Phenobarbital in newborns](03_phenobarbital.ipynb) | what do weight and Apgar score explain in sparse routine data; does the fit match NONMEM's? |
| 4 | [Myelosuppression](04_myelosuppression.ipynb) | from a dose-ranging study to a dose: what decides the fraction of patients with grade 4 neutropenia? |
| 5 | [Warfarin](05_warfarin.ipynb) | the fit next to Monolix's; is a smaller volume in women about sex, or about weight? |

Each tutorial is a plain Python file in the repository (`docs/tutorials/`), run by the test
suite, so it cannot go out of date. To open one as a notebook:
`python docs/tutorials/build.py 01` writes `docs/tutorials/01_theophylline.ipynb` (it needs
the `docs` extra: `pip install -e ".[docs]"`).
