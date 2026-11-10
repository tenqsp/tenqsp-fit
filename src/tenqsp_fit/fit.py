# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""``tf.fit(problem)``: fit a problem by its method (``method optimize`` in the text, or
``method=`` here to override)."""

from __future__ import annotations

from .errors import nearest
from .likelihood import Objective
from .problem import METHODS

__all__ = ["fit"]


def fit(problem, *, method: str | None = None, progress: str | None = None):
    """Fit ``problem``: ``result = tf.fit(problem)``. ``method`` overrides the problem's
    (``"optimize"``, ``"laplace"``, ``"nuts"``); ``progress``
    (``"text"``, ``"plot"`` or ``"none"``) the problem's ``options progress`` (default: the
    live plot in a notebook, the text elsewhere). Returns a ``Result``; the doctor's
    findings (``result.doctor()``) end its printout."""
    from .doctor import examine

    result = _fit(problem, method=method, progress=progress)
    shown = progress or problem._options.get("progress")
    say = (lambda text: None) if shown == "none" else (lambda text: print(text, flush=True))
    result._doctor = examine(result, say=say)
    return result


def _fit(problem, *, method=None, progress=None, with_se: bool = True):
    if method is not None and method not in METHODS:
        raise ValueError(f"unknown method '{method}'.{nearest(method, METHODS)} Methods: "
                         f"{', '.join(METHODS)}.")  # fmt: skip
    if method is not None and problem._method is None:  # method= stands for the method line
        problem = _with_method(problem, method)
    problem.check()
    name = method or problem._method.name
    if name not in METHODS:
        raise ValueError(f"unknown method '{name}'.{nearest(name, METHODS)} Methods: "
                         f"{', '.join(METHODS)}.")  # fmt: skip
    if progress is not None and progress not in ("text", "plot", "none"):
        raise ValueError(f"progress is 'text', 'plot' or 'none', got {progress!r}.")
    varying = [d.name for d in problem._varying()]
    if name == "optimize" and varying:
        raise ValueError(
            f"method optimize fits one set of parameters, but {', '.join(varying)} vary "
            "between units; use method laplace."
        )
    if name == "laplace" and not varying:
        raise ValueError("method laplace is for parameters that vary between units, and none "
                         "does; use method optimize.")  # fmt: skip
    if name == "nuts":
        from .nuts import fit_nuts

        if method is not None and method != problem._method.name:
            problem = _with_method(problem, "nuts")
        return fit_nuts(problem, progress=progress)
    if name == "laplace":
        from .laplace import Laplace, fit_laplace

        return fit_laplace(problem, Laplace(problem), progress=progress, with_se=with_se)
    if name != "optimize":
        raise NotImplementedError(f"method {name} is not available yet.")
    from .optimize import fit_optimize

    return fit_optimize(problem, Objective(problem), progress=progress, with_se=with_se)


def _with_method(problem, name):
    """A copy of ``problem`` whose method is ``name`` (its own options dropped)."""
    from .results import _copy

    p = _copy(problem)
    p._method = None
    p.method(name)
    p.check()
    return p
