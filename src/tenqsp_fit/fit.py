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
    (``"optimize"``; ``"laplace"`` and ``"nuts"`` are not available yet); ``progress``
    (``"text"`` or ``"none"``) the problem's ``options progress``. Returns a ``Result``."""
    problem.check()
    name = method or problem._method.name
    if name not in METHODS:
        raise ValueError(f"unknown method '{name}'.{nearest(name, METHODS)} Methods: "
                         f"{', '.join(METHODS)}.")  # fmt: skip
    if progress is not None and progress not in ("text", "plot", "none"):
        raise ValueError(f"progress is 'text' or 'none', got {progress!r}.")
    if name != "optimize":
        raise NotImplementedError(f"method {name} is not available yet; method optimize is.")
    from .optimize import fit_optimize

    return fit_optimize(problem, Objective(problem), progress=progress)
