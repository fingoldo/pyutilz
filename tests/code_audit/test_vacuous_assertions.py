"""Scanner tests for vacuous_assertions, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_tautological_is_not_none_only_tests,
    scan_vacuous_assertions,
)

from ._helpers import _write

# ---- vacuous_assertion ------------------------------------------------


def test_vacuous_assertion_bare_true_flagged(tmp_path: Path):
    _write(tmp_path, "test_bad.py", """
def test_thing():
    result = compute()
    assert True
""")
    findings = scan_vacuous_assertions(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "vacuous_assertion"


def test_vacuous_assertion_full_domain_or_flagged(tmp_path: Path):
    _write(tmp_path, "test_bad.py", """
def test_thing(result):
    assert result is None or result == {} or isinstance(result, dict)
""")
    findings = scan_vacuous_assertions(tmp_path)
    assert len(findings) == 1


def test_vacuous_assertion_real_check_is_clean(tmp_path: Path):
    _write(tmp_path, "test_ok.py", """
def test_thing():
    result = compute()
    assert result == 42
""")
    findings = scan_vacuous_assertions(tmp_path)
    assert findings == []


# ---- tautological_is_not_none_only_test ----------------------------------


def test_tautological_is_not_none_only_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_bad.py",
        """
def test_returns_something():
    result = compute()
    assert result is not None
""",
    )
    findings = scan_tautological_is_not_none_only_tests(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "tautological_is_not_none_only_test"
    assert "test_returns_something" in findings[0].detail


def test_tautological_is_not_none_with_stronger_assert_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_ok.py",
        """
def test_returns_something():
    result = compute()
    assert result is not None
    assert result.value == 42
""",
    )
    findings = scan_tautological_is_not_none_only_tests(tmp_path)
    assert findings == []


def test_tautological_is_not_none_nested_in_if_not_flagged(tmp_path: Path):
    """A bare is-not-None inside a conditional branch isn't the function's only unconditional
    check -- scanner is conservative and skips nested asserts entirely."""
    _write(
        tmp_path,
        "test_ok.py",
        """
def test_conditional():
    result = compute()
    if result:
        assert result is not None
    assert result.status == "ok"
""",
    )
    findings = scan_tautological_is_not_none_only_tests(tmp_path)
    assert findings == []


def test_tautological_is_not_none_non_test_function_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_ok.py",
        """
def helper():
    result = compute()
    assert result is not None
""",
    )
    findings = scan_tautological_is_not_none_only_tests(tmp_path)
    assert findings == []


# ---- F203: the test_glob parameter is a glob ----------------------------------------


def test_vacuous_assertions_uses_test_glob_as_a_glob(tmp_path: Path):
    _write(tmp_path, "check_a.py", "def test_x():\n    assert True\n")
    _write(tmp_path, "prod.py", "def test_y():\n    assert True\n")
    findings = scan_vacuous_assertions(tmp_path, test_glob="check_*.py")
    assert [f.file for f in findings] == ["check_a.py"]


def test_vacuous_assertions_default_glob_skips_production_modules(tmp_path: Path):
    _write(tmp_path, "prod.py", "def test_y():\n    assert True\n")
    assert scan_vacuous_assertions(tmp_path) == []
