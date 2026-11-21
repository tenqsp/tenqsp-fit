# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""What keeps a fit fast: one compile of the Laplace objective, the compilation cache on disk,
and XLA on one thread."""

import os
import subprocess
import sys
import textwrap

import pytest
from test_laplace import lin_problem

import tenqsp_fit as tf
from tenqsp_fit import _jax_setup


def test_laplace_fit_and_checks_compile_the_objective_once(tmp_path):
    """The fit, the standard errors, the modes and the doctor's tolerance check (which runs
    at tighter tolerances) all call one compiled function."""
    r = tf.fit(lin_problem(tmp_path))  # the doctor's checks run at its end
    assert r.status == "converged"
    r.objective.modes(r.x)
    float(r.objective(r.x))
    assert r.objective._compiled._cache_size() == 1
    # the exact gradient through the quadratic predictions needs second derivatives of the
    # solve: the JAX backend's; the native one gives first derivatives (differences then)
    assert r.objective._quadratic == (r.objective._cond.backend == "jax")


@pytest.mark.parametrize("value, stored", [("on", "on"), ("off", "off"), (True, "on"),
                                           (False, "off")])  # fmt: skip
def test_cache_option(tmp_path, value, stored):
    p = lin_problem(tmp_path)
    p.options(cache=value)
    assert p._options["cache"] == stored


def test_cache_option_refuses_other_values(tmp_path):
    p = lin_problem(tmp_path)
    with pytest.raises(ValueError, match="cache is 'on' or 'off'"):
        p.options(cache="yes")


def _fit_in_new_process(tmp_path, cache_dir, option=""):
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(os.path.dirname(__file__))!r})
        import pathlib
        from test_laplace import lin_problem
        import tenqsp_fit as tf
        p = lin_problem(pathlib.Path({str(tmp_path)!r}), options={option!r})
        r = tf.fit(p)
        print(r.status)
    """)
    env = {**os.environ, "TENQSP_FIT_CACHE": str(cache_dir),
           "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}  # fmt: skip
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                         text=True, timeout=600)  # fmt: skip
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.slow
def test_cache_writes_compiled_code_and_off_writes_nothing(tmp_path):
    on, off = tmp_path / "on", tmp_path / "off"
    assert "converged" in _fit_in_new_process(tmp_path, on)
    assert any(on.iterdir())
    if sys.platform != "win32":
        assert (on.stat().st_mode & 0o777) == 0o700  # its owner's only: JAX runs what is there
    assert "converged" in _fit_in_new_process(tmp_path, off, "cache=off")
    assert not off.exists() or not any(off.iterdir())


def test_one_xla_thread_on_import_unless_set():
    code = "import os, tenqsp_fit; print(os.environ.get('XLA_FLAGS'))"
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == _jax_setup.ONE_THREAD
    out = subprocess.run([sys.executable, "-c", code], env={**env, "XLA_FLAGS": ""},
                         capture_output=True, text=True)  # fmt: skip
    assert out.stdout.strip() == ""  # set by the user (even empty): left as it is
