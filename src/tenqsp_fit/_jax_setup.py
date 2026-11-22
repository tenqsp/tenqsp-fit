# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Two process-wide JAX settings tenqsp-fit makes, both described in the guide (Running a fit,
Speed).

**One XLA thread.** The solves of a fit are many small array operations; spreading each over
the CPU's threads costs more than it gains (Friberg's model, 20 patients: 1.7 times faster per
solve on one thread), and the starts of a multi-start fit already run on separate cores. So
on import, before JAX starts, tenqsp-fit sets ``XLA_FLAGS`` to one thread, unless
``XLA_FLAGS`` is already set (to anything, also empty) or JAX has already started (then it is
too late, and nothing is changed).

**The compilation cache.** A fit compiles its likelihood once, which can take tens of
seconds; JAX's persistent compilation cache keeps the compiled code on disk, so the same
problem fitted again (in a new session, with other data of the same shape) starts at once.
On by default, in ``~/.cache/tenqsp-fit/jax`` (made readable by its owner only: JAX runs
what it finds there), or in the directory ``TENQSP_FIT_CACHE`` names; off with
``options cache=off`` or ``TENQSP_FIT_CACHE`` set to empty. A cache directory already set
for JAX (``jax_compilation_cache_dir``) is used as it is.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ONE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
CACHE_ENV = "TENQSP_FIT_CACHE"
_state = {"dir": None, "off": False}  # what tenqsp-fit itself set


def one_thread() -> None:
    """Set XLA to one thread, if nobody has set XLA_FLAGS and JAX has not started."""
    if "XLA_FLAGS" in os.environ:
        return
    if "jax" in sys.modules:
        try:
            from jax._src import xla_bridge

            if xla_bridge.backends_are_initialized():
                return
        except (ImportError, AttributeError):  # pragma: no cover - another JAX: leave it
            return
    os.environ["XLA_FLAGS"] = ONE_THREAD


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tenqsp-fit" / "jax"


def use_cache(setting: str) -> None:
    """Turn JAX's persistent compilation cache on or off for the fits that follow
    (``setting``: the problem's ``cache`` option, "on" or "off")."""
    import jax
    from jax.experimental.compilation_cache import compilation_cache as cc

    if setting == "off":
        if jax.config.jax_enable_compilation_cache:
            jax.config.update("jax_enable_compilation_cache", False)
            _state["off"] = True
            cc.reset_cache()
        return
    if _state["off"]:  # turned off by an earlier fit of this session: on again
        jax.config.update("jax_enable_compilation_cache", True)
        _state["off"] = False
        cc.reset_cache()
    if jax.config.jax_compilation_cache_dir:  # the user's, or ours from an earlier fit
        return
    env = os.environ.get(CACHE_ENV)
    if env == "":
        return
    path = Path(env) if env else default_cache_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError:  # a directory that cannot be made: no cache, the fit goes on
        return
    jax.config.update("jax_compilation_cache_dir", str(path))
    _state["dir"] = str(path)
    cc.reset_cache()  # JAX checks for a cache once, at its first compile


one_thread()  # on import (tenqsp_fit imports this module first)
