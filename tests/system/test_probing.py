"""Behaviour of ``pyutilz.system.system.probing._resolve_binary``: a binary that is on PATH resolves to an absolute path, and
one that is not raises the dedicated not-found error, so a probe never falls back to a bare name that the current directory
could shadow."""

from __future__ import annotations

import os
import sys

import pytest

from pyutilz.system.system import probing


def test_a_binary_on_path_resolves_to_an_absolute_path():
    resolved = probing._resolve_binary(os.path.basename(sys.executable))
    assert os.path.isabs(resolved)
    assert os.path.exists(resolved)


def test_a_missing_binary_raises_the_not_found_error():
    with pytest.raises(probing._BinaryNotFoundError):
        probing._resolve_binary("nosuch_binary_zz")
