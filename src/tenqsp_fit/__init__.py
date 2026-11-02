# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""tenqsp-fit: fitting TenQSP models to data, with diagnosis of what the data determine.

>>> import tenqsp_fit as tf
>>> problem = tf.load("theoph_pop.tq")      # a TenQSP model followed by its fit part
>>> print(problem.summary())
"""

from .distributions import (
    Distribution,
    ErrorModel,
    additive,
    beta,
    combined,
    exponential,
    fix,
    gamma,
    logitnormal,
    lognormal,
    normal,
    power,
    proportional,
    uniform,
    weibull,
)
from .errors import FitTextError, ProblemError
from .fit import fit
from .laplace import Laplace
from .likelihood import Objective, objective
from .optimize import Result
from .problem import Problem
from .results import prior_predictive
from .text import load
from .units import Observations, Unit

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "Distribution",
    "ErrorModel",
    "FitTextError",
    "Laplace",
    "Objective",
    "Observations",
    "Problem",
    "ProblemError",
    "Result",
    "Unit",
    "additive",
    "beta",
    "combined",
    "exponential",
    "fit",
    "fix",
    "gamma",
    "load",
    "logitnormal",
    "lognormal",
    "normal",
    "objective",
    "power",
    "prior_predictive",
    "proportional",
    "uniform",
    "weibull",
]
