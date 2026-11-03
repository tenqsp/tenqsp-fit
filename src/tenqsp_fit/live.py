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
    """True in a Jupyter-protocol kernel (Jupyter, Positron's console, VS Code): its shell is
    a ZMQInteractiveShell (Positron's is a subclass, with its own kernel app)."""
    ip = sys.modules.get("IPython")
    if ip is None:
        return False
    try:
        shell = ip.get_ipython()
    except Exception:  # pragma: no cover
        return False
    return any(c.__name__ == "ZMQInteractiveShell" for c in type(shell).__mro__)


def default_progress() -> str:
    return "plot" if in_notebook() else "text"


def _matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:  # pragma: no cover
        raise ImportError(
            "plots need matplotlib: pip install matplotlib (or tenqsp-fit[plot])."
        ) from None


def _figsize(names) -> tuple[float, float]:
    n_panels = 1 + len(names)
    ncols = min(NCOLS, n_panels)
    return 3.1 * ncols, 2.3 * math.ceil(n_panels / ncols) + 0.4


def draw_trace(trace: pd.DataFrame, names, *, title: str = "", fig=None, value: str = "-2LL",
               line: str = "start", xlabel: str = "step", warmup: int | None = None):  # fmt: skip
    """The figure: -2LL (the column ``value``) against the step (top left), then one panel per
    quantity. One line per start (``line``: per chain for NUTS), the best (lowest value so
    far) dark and thick, the others light; ``warmup``: a dotted line where NUTS's draws
    begin."""
    _matplotlib()
    n_panels = 1 + len(names)
    ncols = min(NCOLS, n_panels)
    nrows = math.ceil(n_panels / ncols)
    if fig is None:  # the user's own figure, shown by their backend (Plots pane, inline)
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=_figsize(names))
    fig.clear()
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    if len(trace):
        last = trace.groupby(line)[value].min()
        best = int(last.idxmin())
        order = [s for s in last.index if s != best] + [best]
        for s in order:
            d = trace[trace[line] == s]
            style = (dict(color="#1f3b73", lw=1.8, zorder=3) if s == best
                     else dict(color="#9bb3d6", lw=0.9, zorder=2))  # fmt: skip
            axes[0].plot(d.iteration, d[value], **style)
            for ax, name in zip(axes[1:], names, strict=False):
                ax.plot(d.iteration, d[name], **style)
        v = trace[value].to_numpy()
        v = v[np.isfinite(v)]
        if len(v):
            lo, hi = float(np.min(v)), float(np.max(v))
            pad = max(hi - lo, 1e-3) * 0.05
            axes[0].set_ylim(lo - pad, hi + pad)
            axes[0].set_title(f"{value} (lowest {lo:.4f})", fontsize=9)
        if warmup is not None and trace.iteration.max() >= warmup:
            for ax in axes[:n_panels]:
                ax.axvline(warmup, ls=":", color="0.5", lw=1)
    else:
        axes[0].set_title(value, fontsize=9)
    for ax, name in zip(axes[1:], names, strict=False):
        ax.set_title(name, fontsize=9)
    from matplotlib.ticker import MaxNLocator

    for ax in axes[:n_panels]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    for ax in axes[max(0, n_panels - ncols) : n_panels]:
        ax.set_xlabel(xlabel, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    if hasattr(fig, "tight_layout"):  # a subfigure takes its parent's layout
        fig.tight_layout()
    return fig


class LivePlot:
    """Redraws the trace every ``REFRESH`` seconds: in a notebook in place, elsewhere as a
    page in the browser (``path``)."""

    def __init__(self, names, label: str, problem_name: str, *, open_browser: bool = True,
                 **trace_options):  # fmt: skip
        self.names, self.label, self.problem_name = list(names), label, problem_name
        self.trace_options = trace_options  # draw_trace's value, line, xlabel, warmup
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
        if self.fig is None:
            # a figure of its own, outside pyplot: the live plot never touches the session's
            # backend or its figures (switching the backend stops Positron's Plots pane)
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self.fig = Figure(figsize=_figsize(self.names))
            FigureCanvasAgg(self.fig)
        draw_trace(trace, self.names, title=title, fig=self.fig, **self.trace_options)
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=90)
        png = buf.getvalue()
        if self.notebook:
            from IPython.display import Image, display

            if self.handle is None:
                self.handle = display(Image(data=png), display_id=True)
            else:
                self.handle.update(Image(data=png))
        else:
            self._write_page(png, title, final=last)

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
