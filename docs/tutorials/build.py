# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Turn the tutorials into executed notebooks, for reading.

    python docs/tutorials/build.py            # every tutorial
    python docs/tutorials/build.py 02         # those whose file name starts with 02

The tutorials are plain Python files in "percent" format (`# %%` starts a code cell,
`# %% [markdown]` a text cell), so the test suite runs them as scripts
(tests/test_tutorials.py) and they cannot go stale. This script converts each to a notebook
(jupytext), runs it (nbclient) and writes it to docs/tutorials/ (not in git).
Needs the `docs` extra: pip install -e ".[docs]".
"""

from __future__ import annotations

import sys
from pathlib import Path

import jupytext
import nbclient

HERE = Path(__file__).parent
OUT = HERE


def tutorials(prefix: str = "") -> list[Path]:
    return sorted(p for p in HERE.glob("[0-9][0-9]_*.py") if p.name.startswith(prefix))


def build(path: Path) -> Path:
    nb = jupytext.read(path)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nbclient.NotebookClient(nb, timeout=1800, resources={"metadata": {"path": str(HERE)}}).execute()
    OUT.mkdir(exist_ok=True)
    out = OUT / path.with_suffix(".ipynb").name
    jupytext.write(nb, out)
    return out


if __name__ == "__main__":
    for p in tutorials(sys.argv[1] if len(sys.argv) > 1 else ""):
        print("built", build(p))
