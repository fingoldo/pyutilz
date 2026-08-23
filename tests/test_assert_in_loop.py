"""Unit tests for ``scan_assert_in_loop_reports_only_the_first``.

Positive: an assert inside a loop over a real data source. Negative: the shapes where first-failure
reporting is fine - a hand-written literal, a `range`/`zip` wrapper, and a loop that already collects.
"""

from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_assert_in_loop_reports_only_the_first


def _write(tmp_path: Path, body: str) -> Path:
    (tmp_path / "test_it.py").write_text(body, encoding="utf-8")
    return tmp_path


def test_assert_inside_a_loop_over_a_data_source_is_flagged(tmp_path: Path):
    root = _write(tmp_path, "def test_rows():\n    for row in build():\n        assert row.ok, row\n")
    findings = scan_assert_in_loop_reports_only_the_first(root)
    assert len(findings) == 1, findings
    assert findings[0].check == "assert_in_loop_first_failure_only"


def test_the_detail_names_the_site_so_two_loops_in_one_file_stay_distinct(tmp_path: Path):
    """A consumer keying a baseline on (check, file, detail) must not suppress a whole file at once."""
    root = _write(
        tmp_path,
        "def test_a():\n    for row in build():\n        assert row.ok\n\n\ndef test_b():\n    for row in build():\n        assert row.fine\n",
    )
    details = {f.detail for f in scan_assert_in_loop_reports_only_the_first(root)}
    assert len(details) == 2, details


def test_a_short_literal_loop_is_not_flagged(tmp_path: Path):
    """Three hand-written cases: the first failure IS the report."""
    root = _write(tmp_path, "def test_x():\n    for value in (1, 2, 3):\n        assert value > 0\n")
    assert scan_assert_in_loop_reports_only_the_first(root) == []


def test_a_range_loop_is_not_flagged(tmp_path: Path):
    """A `range` bound is visible at the call site, which is the property that matters."""
    root = _write(tmp_path, "def test_x():\n    for i in range(100):\n        assert i >= 0\n")
    assert scan_assert_in_loop_reports_only_the_first(root) == []


def test_a_loop_that_already_accumulates_is_not_flagged(tmp_path: Path):
    """It is already reporting the whole set somewhere; the assert is a per-item guard, not the verdict."""
    root = _write(
        tmp_path,
        "def test_x():\n    bad = []\n    for row in build():\n        bad.append(row)\n        assert row is not None\n",
    )
    assert scan_assert_in_loop_reports_only_the_first(root) == []
