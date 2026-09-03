"""Scanner tests for raising_stub_swallowed, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.raising_stub_swallowed import scan_raising_stub_swallowed

from ._helpers import _write

# ---- raising_stub_swallowed ---------------------------------------------
#
# A test says "this must never be called" by raising, and a broad handler downstream turns the
# raise into a benign path. Confirmed: a cache was re-probed on every run behind a green test.


def test_raising_stub_swallowed_flags_the_shape(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def part_has_upwork(part):
    try:
        return ParquetFile(part).probe()
    except Exception:
        return None
""")
    _write(tmp_path, "test_cache.py", """
from unittest.mock import patch

def test_locate_uses_cache_when_not_stale():
    def _boom(*a, **k):
        raise AssertionError("must not be called")
    with patch("prod.ParquetFile", _boom) as spy:
        locate()
    assert spy.called is False
""")
    findings = scan_raising_stub_swallowed(tmp_path)
    assert len(findings) == 1
    assert "ParquetFile" in findings[0].detail


def test_raising_stub_swallowed_scopes_stub_names_to_the_test(tmp_path: Path):
    """`_gql` is a name every test in a file defines for itself. Collected module-wide, one
    raising definition tainted five harmless ones and produced the rule's only false positive on a
    real repository."""
    _write(tmp_path, "prod.py", """
def go():
    try:
        return gql(1)
    except Exception:
        return None
""")
    _write(tmp_path, "test_x.py", """
from unittest.mock import patch

def test_harmless():
    def _gql(*a, **k):
        return {"ok": True}
    with patch("prod.gql", _gql) as spy:
        go()
    assert spy.called

def test_raising():
    def _gql(*a, **k):
        raise AssertionError("no")
    with patch("prod.gql", _gql) as spy:
        go()
    assert spy.called is False
""")
    findings = scan_raising_stub_swallowed(tmp_path)
    assert len(findings) == 1, [f.line for f in findings]


def test_raising_stub_swallowed_ignores_a_test_expecting_the_raise(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def go():
    try:
        return gql(1)
    except Exception:
        return None
""")
    _write(tmp_path, "test_x.py", """
import pytest
from unittest.mock import patch

def test_it_propagates():
    def _gql(*a, **k):
        raise ValueError("boom")
    with patch("prod.gql", _gql) as spy:
        with pytest.raises(ValueError):
            go()
    assert spy.called
""")
    assert scan_raising_stub_swallowed(tmp_path) == []


def test_raising_stub_swallowed_ignores_a_narrow_handler(tmp_path: Path):
    """A handler that catches a specific type is not the swallow this rule is about."""
    _write(tmp_path, "prod.py", """
def go():
    try:
        return gql(1)
    except KeyError:
        return None
""")
    _write(tmp_path, "test_x.py", """
from unittest.mock import patch

def test_x():
    def _gql(*a, **k):
        raise AssertionError("no")
    with patch("prod.gql", _gql) as spy:
        go()
    assert spy.called is False
""")
    assert scan_raising_stub_swallowed(tmp_path) == []


# ---- F13: raising stub swallowed, under a tests ancestor directory ------------------


def _raising_stub_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write(root, "prod.py", """
def probe():
    return 1


def run():
    try:
        return probe()
    except Exception:
        return None
""")
    _write(root, "test_x.py", """
from unittest.mock import patch
import prod


def test_never_called():
    def boom(*a):
        raise AssertionError("no")

    with patch("prod.probe", boom):
        prod.run()
    assert not boom.called
""")


def test_raising_stub_swallowed_survives_a_tests_ancestor_directory(tmp_path: Path):
    root = tmp_path / "tests" / "proj"
    _raising_stub_tree(root)
    assert len(scan_raising_stub_swallowed(root)) == 1


def test_raising_stub_swallowed_ignores_a_production_only_tree(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def probe():
    return 1


def run():
    try:
        return probe()
    except Exception:
        return None
""")
    assert scan_raising_stub_swallowed(tmp_path) == []
