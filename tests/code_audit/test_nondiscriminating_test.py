"""Scanner tests for nondiscriminating_test, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.nondiscriminating_test import scan_nondiscriminating_test_functions

from ._helpers import _write

# ---- F18/F74/F168: nondiscriminating test -------------------------------------------


def test_nondiscriminating_test_accepts_the_declarative_xfail_marker(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
import pytest

@pytest.mark.xfail(reason="known")
def test_thing():
    assert 1 + 1 == 2
""")
    assert scan_nondiscriminating_test_functions(tmp_path) == []


def test_nondiscriminating_test_still_flags_an_imperative_xfail(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
import pytest

def test_thing():
    assert 1 + 1 == 2
    pytest.xfail("gap")
""")
    assert len(scan_nondiscriminating_test_functions(tmp_path)) == 1


def test_nondiscriminating_test_accepts_a_handler_that_calls_pytest_fail(tmp_path: Path):
    _write(tmp_path, "test_c.py", """
import pytest

def test_x():
    try:
        assert compute() == 3
    except Exception as e:
        pytest.fail(str(e))
""")
    assert scan_nondiscriminating_test_functions(tmp_path) == []


def test_nondiscriminating_test_reads_the_suffix_naming_convention(tmp_path: Path):
    _write(tmp_path, "widget_test.py", """
def test_x():
    do_work()
""")
    assert len(scan_nondiscriminating_test_functions(tmp_path)) == 1


def test_nondiscriminating_test_accepts_a_PRIVATE_assertion_helper(tmp_path: Path):
    """A shared checker named `_assert_...` carries the assertion just as `assert_...` does.

    Factoring the check into one helper is the refactor this scanner should encourage; matching only the
    public spelling reported every delegating test as checking nothing.
    """
    _write(
        tmp_path,
        "test_d.py",
        """
def _assert_traceback_preserved(msg):
    assert msg in collect()

def test_x():
    _assert_traceback_preserved("could not save %s")
""",
    )
    assert scan_nondiscriminating_test_functions(tmp_path) == []
