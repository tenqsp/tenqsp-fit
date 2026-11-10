# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Every tutorial (docs/tutorials/NN_*.py) runs as a script and its own checks pass. Each
ends its sections with checks against reference fits or the values the data were simulated
from, so running them is a test (slow: minutes each)."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

TUTORIALS = sorted((Path(__file__).parents[1] / "docs" / "tutorials").glob("[0-9][0-9]_*.py"))


@pytest.mark.slow
@pytest.mark.timeout(3600)  # a tutorial is many fits (the per-test limit is for one)
@pytest.mark.parametrize("path", TUTORIALS, ids=[p.stem for p in TUTORIALS])
def test_tutorial_runs(path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(path.parent)
    runpy.run_path(str(path), run_name="__main__")
    plt.close("all")
