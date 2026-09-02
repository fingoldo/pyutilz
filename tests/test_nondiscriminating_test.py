"""Tests for the nondiscriminating-test checker, distilled from the 2026-09-01 wide audit.

Every shape the detector claims to catch is injected, and healthy code it must not flag is asserted clean --
a checker for tests that cannot fail would otherwise be an instance of the very defect it exists to find.
Tests use tmp_path so each scan runs against a hermetic tree.
"""

from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_nondiscriminating_test_functions


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Write ``source`` to ``tmp_path/name`` and return the path."""
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


# --- nondiscriminating test functions ------------------------------------


def test_nondiscriminating_zero_assertion_flagged(tmp_path: Path):
    """A test whose body checks nothing cannot fail for the reason it names."""
    _write(tmp_path, "test_bad.py", "def test_thing():\n    result = compute()\n    print(result)\n")
    findings = scan_nondiscriminating_test_functions(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "nondiscriminating_test"
    assert "no-assert" in findings[0].detail


def test_nondiscriminating_swallowed_assertion_flagged(tmp_path: Path):
    """`except AssertionError: pass` means no assertion inside can ever fail the suite."""
    src = "def test_thing():\n    try:\n        assert compute() == 1\n    except AssertionError:\n        pass\n"
    _write(tmp_path, "test_bad.py", src)
    findings = scan_nondiscriminating_test_functions(tmp_path)
    assert len(findings) == 1
    assert "swallows-assertionerror" in findings[0].detail


def test_nondiscriminating_all_assertions_behind_if_flagged(tmp_path: Path):
    """When every assertion is guarded, the stronger failure skips the check instead of failing it."""
    src = "def test_thing():\n    result = compute()\n    if result is not None:\n        assert result > 0\n"
    _write(tmp_path, "test_bad.py", src)
    findings = scan_nondiscriminating_test_functions(tmp_path)
    assert len(findings) == 1
    assert "all-asserts-conditional" in findings[0].detail


def test_nondiscriminating_pass_body_if_flagged(tmp_path: Path):
    """A documented expectation written as a no-op permits exactly the state it calls wrong."""
    src = "def test_thing():\n    result = compute()\n    assert result is not None\n    if result.beta:\n        pass\n"
    _write(tmp_path, "test_bad.py", src)
    findings = scan_nondiscriminating_test_functions(tmp_path)
    assert len(findings) == 1
    assert "pass-body-if" in findings[0].detail


def test_nondiscriminating_imperative_xfail_flagged(tmp_path: Path):
    """An imperative xfail discards the measurement just taken, concealing a gap that has closed."""
    src = "import pytest\n\n\ndef test_thing():\n    gap = measure()\n    pytest.xfail(f'still open: {gap}')\n"
    _write(tmp_path, "test_bad.py", src)
    findings = scan_nondiscriminating_test_functions(tmp_path)
    assert len(findings) == 1
    assert "imperative-xfail" in findings[0].detail


def test_nondiscriminating_reports_every_shape_not_just_the_first(tmp_path: Path):
    """A function exhibiting two shapes is described by both, so fixing one does not hide the other."""
    src = "def test_thing():\n    result = compute()\n    if result:\n        pass\n"
    _write(tmp_path, "test_bad.py", src)
    findings = scan_nondiscriminating_test_functions(tmp_path)
    assert len(findings) == 1
    assert "no-assert" in findings[0].detail
    assert "pass-body-if" in findings[0].detail


def test_nondiscriminating_healthy_test_is_clean(tmp_path: Path):
    """An unconditional assertion on a real result is exactly what the checker must leave alone."""
    _write(tmp_path, "test_ok.py", "def test_thing():\n    assert compute() == 42\n")
    assert scan_nondiscriminating_test_functions(tmp_path) == []


def test_nondiscriminating_pytest_raises_counts_as_a_check(tmp_path: Path):
    """A `pytest.raises` block is a real check even with no `assert` statement."""
    src = "import pytest\n\n\ndef test_thing():\n    with pytest.raises(ValueError):\n        compute()\n"
    _write(tmp_path, "test_ok.py", src)
    assert scan_nondiscriminating_test_functions(tmp_path) == []


def test_nondiscriminating_handler_asserting_the_failure_is_clean(tmp_path: Path):
    """A handler that asserts on the caught error is checking the failure, not swallowing it."""
    src = "def test_thing():\n    try:\n        compute()\n    except AssertionError as e:\n        assert 'expected' in str(e)\n"
    _write(tmp_path, "test_ok.py", src)
    assert scan_nondiscriminating_test_functions(tmp_path) == []


def test_nondiscriminating_ignores_non_test_helpers(tmp_path: Path):
    """Fixtures and helpers legitimately assert nothing, so only `test_*` functions are in scope."""
    _write(tmp_path, "test_ok.py", "def make_frame():\n    return build()\n\n\ndef test_thing():\n    assert make_frame() is not None\n")
    assert scan_nondiscriminating_test_functions(tmp_path) == []


def test_nondiscriminating_ignores_non_test_files(tmp_path: Path):
    """Production modules are not tests; a function there asserting nothing is normal."""
    _write(tmp_path, "helpers.py", "def test_connection():\n    open_socket()\n")
    assert scan_nondiscriminating_test_functions(tmp_path) == []


def test_nondiscriminating_nested_helper_assertions_do_not_count_for_the_parent(tmp_path: Path):
    """An assertion inside a nested function is that helper's, and does not make the caller discriminating."""
    src = "def test_thing():\n    def check(v):\n        assert v > 0\n\n    compute()\n"
    _write(tmp_path, "test_bad.py", src)
    findings = scan_nondiscriminating_test_functions(tmp_path)
    assert len(findings) == 1
    assert "no-assert" in findings[0].detail
