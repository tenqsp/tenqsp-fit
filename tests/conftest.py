# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Shared test settings."""

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_functions():
    """Each compiled model holds memory mappings, and Linux allows a process 65,530 of them;
    the whole suite in one process passes that, and the compiler then aborts. Release them
    after each test file."""
    yield
    gc.collect()
    jax.clear_caches()
