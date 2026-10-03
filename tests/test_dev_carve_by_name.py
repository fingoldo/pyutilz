"""Behavioural tests for :mod:`pyutilz.dev.carve_by_name`."""

from __future__ import annotations

import importlib
import sys

import pytest

from pyutilz.dev.carve_by_name import carve_by_name

SOURCE = '''"""Module doc."""
import math

LIMIT = 3


def _keep():
    return 1


def _step1_a(x):
    return _step1_helper(x) + LIMIT


def _step1_helper(x):
    return math.floor(x)


def public(x):
    return _step1_a(x) + _keep()
'''


def _fingerprint(path):
    """Size and modification time of a file: unchanged means it was not rewritten."""
    st = path.stat()
    return st.st_size, st.st_mtime_ns


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    root = tmp_path / "carvepkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "big.py").write_text(SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield root
    for name in [m for m in sys.modules if m.startswith("carvepkg")]:
        del sys.modules[name]


def test_carved_module_pulls_dependencies_and_source_reexports(pkg):
    moved = carve_by_name(pkg / "big.py", "big_steps", r"_step1_a")
    assert moved == ["LIMIT", "_step1_a", "_step1_helper"]
    steps = importlib.import_module("carvepkg.big_steps")
    assert "carvepkg.big" not in sys.modules, "the carved sibling must import without its source module"
    big = importlib.import_module("carvepkg.big")
    assert big.public(2.7) == 2 + 3 + 1
    assert big._step1_a is steps._step1_a


def test_dry_run_writes_nothing_and_existing_sibling_is_refused(pkg):
    before = _fingerprint(pkg / "big.py")
    assert carve_by_name(pkg / "big.py", "big_steps", r"_step1_a", apply=False)
    assert _fingerprint(pkg / "big.py") == before and not (pkg / "big_steps.py").exists()
    (pkg / "taken.py").write_text("x = 1\n")
    with pytest.raises(FileExistsError):
        carve_by_name(pkg / "big.py", "taken", r"_step1_a")
    assert _fingerprint(pkg / "big.py") == before


def test_no_match_raises_and_leaves_source_untouched(pkg):
    before = _fingerprint(pkg / "big.py")
    with pytest.raises(ValueError):
        carve_by_name(pkg / "big.py", "big_steps", r"_nothing_.*")
    assert _fingerprint(pkg / "big.py") == before


def test_sibling_may_be_given_as_a_path(pkg):
    carve_by_name(pkg / "big.py", pkg / "elsewhere.py", r"_step1_a")
    assert (pkg / "elsewhere.py").exists()
