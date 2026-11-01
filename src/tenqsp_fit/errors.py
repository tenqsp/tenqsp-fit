# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The errors tenqsp-fit raises."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


class ProblemError(ValueError):
    """A fitting problem is not complete or not consistent. The message lists every problem
    found, each with what to write instead."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        if len(self.problems) == 1:
            msg = self.problems[0]
        else:
            msg = f"The fitting problem has {len(self.problems)} problems:\n\n" + "\n\n".join(
                self.problems
            )
        super().__init__(msg)


class FitTextError(ProblemError):
    """A fit text cannot be read. The message lists every problem with its line."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        ValueError.__init__(
            self,
            f"The fit text has {len(self.problems)} problem(s):\n\n" + "\n\n".join(self.problems),
        )


def nearest(word: str, choices: Iterable[str]) -> str:
    """' Did you mean 'x'?' for the closest choice, or ''."""
    hit = difflib.get_close_matches(word, sorted(choices), n=1)
    return f" Did you mean '{hit[0]}'?" if hit else ""
