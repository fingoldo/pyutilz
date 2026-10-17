"""The on-disk code-audit scan memo: a hit must equal a fresh scan, and a change to the tree must never be served stale."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pyutilz.dev.code_audit.registry import run_all_checked

BAD = "def f(x=[]):\n    return x\n"
GOOD = "def f(x=None):\n    return x\n"


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    """Point the cache at a throwaway home so the test never writes to the developer's real cache."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return tmp_path / "home"


def test_a_cached_scan_equals_a_fresh_scan(tmp_path, cache_home):
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "mod.py").write_text(BAD, encoding="utf-8")
    first, first_failed = run_all_checked(tmp_path / "tree")
    second, second_failed = run_all_checked(tmp_path / "tree")
    assert [(f.check, f.file, f.line) for f in second] == [(f.check, f.file, f.line) for f in first]
    assert second_failed == first_failed
    assert list(cache_home.rglob("*.json")), "the second run should have read the memo this one wrote"


def test_editing_a_file_is_never_served_from_the_memo(tmp_path, cache_home):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "mod.py").write_text(BAD, encoding="utf-8")
    assert run_all_checked(tree)[0], "the defective file must be reported"
    time.sleep(0.01)  # mtime granularity: make the rewrite visibly newer
    (tree / "mod.py").write_text(GOOD, encoding="utf-8")
    assert run_all_checked(tree)[0] == [], "the fixed file must not be answered from the stale entry"


def test_a_runtime_registered_scanner_is_not_memoised(tmp_path, cache_home, monkeypatch):
    """A scanner the process registered itself is not a function of the tree alone, so its results are never stored."""
    from pyutilz.dev.code_audit import registry

    def _extra(root, exclude_dirs=frozenset()):
        return []

    monkeypatch.setitem(registry._SCANNERS, "runtime_only_probe", _extra)
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "mod.py").write_text(GOOD, encoding="utf-8")
    run_all_checked(tmp_path / "tree", checks=["runtime_only_probe"])
    assert not list(cache_home.rglob("*.json"))


def test_a_corrupt_memo_entry_falls_back_to_a_fresh_scan(tmp_path, cache_home):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "mod.py").write_text(BAD, encoding="utf-8")
    run_all_checked(tree)
    for entry in cache_home.rglob("*.json"):
        entry.write_bytes(b"not json")
    assert run_all_checked(tree)[0], "a corrupt entry must not hide the finding"
