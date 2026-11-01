# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Live plots of a fit (``options progress=plot``; the default in a notebook): -2LL against
the step, and one small panel per quantity showing its path, one line per start, as Monolix
shows. In a notebook (Jupyter, Positron) the figure is redrawn in place; elsewhere it is a page
that refreshes itself, opened in the browser. ``result.plot_trace()`` draws the same figure
afterwards from ``result.trace``."""

from __future__ import annotations

import base64
import html
import io
import math
import os
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

REFRESH = 2.0  # seconds between redraws
NCOLS = 4


def in_notebook() -> bool:
    """True in a Jupyter-protocol kernel (Jupyter, Positron's console, VS Code)."""
    ip = sys.modules.get("IPython")
    if ip is None:
        return False
    try:
        shell = ip.get_ipython()
    except Exception:  # pragma: no cover
        return False
    return shell is not None and "IPKernelApp" in getattr(shell, "config", {})


def default_progress() -> str:
    return "plot" if in_notebook() else "text"


def _pyplot():
    try:
        import matplotlib

        if not in_notebook() and "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError(
            "plots need matplotlib: pip install matplotlib (or tenqsp-fit[plot])."
        ) from None
    return plt


def draw_trace(trace: pd.DataFrame, names, *, title: str = "", fig=None):
    """The figure: -2LL against the step (top left), then one panel per quantity. One line
    per start, the best start (lowest -2LL so far) dark and thick, the others light."""
    plt = _pyplot()
    n_panels = 1 + len(names)
    ncols = min(NCOLS, n_panels)
    nrows = math.ceil(n_panels / ncols)
    if fig is None:
        fig = plt.figure(figsize=(3.1 * ncols, 2.3 * nrows + 0.4))
    fig.clear()
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    if len(trace):
        last = trace.groupby("start")["-2LL"].min()
        best = int(last.idxmin())
        order = [s for s in last.index if s != best] + [best]
        for s in order:
            d = trace[trace.start == s]
            style = (dict(color="#1f3b73", lw=1.8, zorder=3) if s == best
                     else dict(color="#9bb3d6", lw=0.9, zorder=2))  # fmt: skip
            axes[0].plot(d.iteration, d["-2LL"], **style)
            for ax, name in zip(axes[1:], names, strict=False):
                ax.plot(d.iteration, d[name], **style)
        v = trace["-2LL"].to_numpy()
        lo = float(np.nanmin(v))
        # leave out each start's first steps, which can be far above the rest
        later = trace[trace.iteration > trace.groupby("start").iteration.transform("min") + 1]
        hi = float(np.nanpercentile(later["-2LL"], 95)) if len(later) else float(np.nanmax(v))
        pad = max(hi - lo, 1e-3) * 0.08
        axes[0].set_ylim(lo - pad, hi + pad)
        axes[0].set_title(f"-2LL (best {lo:.4f})", fontsize=9)
    else:
        axes[0].set_title("-2LL", fontsize=9)
    for ax, name in zip(axes[1:], names, strict=False):
        ax.set_title(name, fontsize=9)
    from matplotlib.ticker import MaxNLocator

    for ax in axes[:n_panels]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    for ax in axes[max(0, n_panels - ncols) : n_panels]:
        ax.set_xlabel("step", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


class LivePlot:
    """Redraws the trace every ``REFRESH`` seconds: in a notebook in place, elsewhere as a
    page in the browser (``path``)."""

    def __init__(self, names, label: str, problem_name: str, *, open_browser: bool = True):
        self.names, self.label, self.problem_name = list(names), label, problem_name
        self.notebook = in_notebook()
        self.drawn = -1e9
        self.handle = None
        self.fig = None
        self.path = None
        if not self.notebook:
            folder = Path(tempfile.mkdtemp(prefix="tenqsp-fit-"))
            self.path = folder / f"{problem_name}.html"
            self._write_page(None, "starting", final=False)
            print(f"live plot: {self.path.as_uri()}", flush=True)
            if open_browser:
                webbrowser.open(self.path.as_uri())

    def _title(self, status: str) -> str:
        return f"{self.problem_name}: {self.label}, {status}"

    def update(self, trace_rows, columns, status: str, seconds: float, last: bool = False):
        if not last and time.perf_counter() - self.drawn < REFRESH:
            return
        self.drawn = time.perf_counter()
        trace = pd.DataFrame(sorted(trace_rows), columns=columns)
        clock = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
        title = self._title(f"{status}, {clock}")
        plt = _pyplot()
        if self.fig is None:
            n_panels = 1 + len(self.names)
            ncols = min(NCOLS, n_panels)
            self.fig = plt.figure(figsize=(3.1 * ncols, 2.3 * math.ceil(n_panels / ncols) + 0.4))
        draw_trace(trace, self.names, title=title, fig=self.fig)
        if self.notebook:
            from IPython.display import display

            if self.handle is None:
                self.handle = display(self.fig, display_id=True)
            else:
                self.handle.update(self.fig)
            if last:
                plt.close(self.fig)
        else:
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", dpi=90)
            self._write_page(buf.getvalue(), title, final=last)
            if last:
                plt.close(self.fig)

    def _write_page(self, png: bytes | None, title: str, final: bool):
        refresh = "" if final else f'<meta http-equiv="refresh" content="{REFRESH:g}">'
        img = "<p>waiting for the first steps</p>"
        if png:
            data = base64.b64encode(png).decode()
            img = f'<img src="data:image/png;base64,{data}" style="max-width:100%">'
        page = (
            f"<!doctype html><html><head><meta charset='utf-8'>{refresh}"
            f"<title>{html.escape(self.problem_name)}</title></head>"
            f"<body style='font-family:sans-serif;background:#fff;margin:16px'>"
            f"<p>{html.escape(title)}</p>{img}</body></html>"
        )
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(page, encoding="utf-8")
        os.replace(tmp, self.path)  # the browser never reads a half-written page
