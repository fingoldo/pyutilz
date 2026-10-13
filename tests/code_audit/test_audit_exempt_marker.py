"""Inline ``# audit-exempt: <check_id> -- <reason>`` markers: the shared helper and its scanner wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyutilz.dev.code_audit import scan_except_skip_masks_call_under_test, scan_hardcoded_absolute_path_in_test
from pyutilz.dev.code_audit._base import is_line_exempt

from ._helpers import _write

_CHECK = "except_skip_masks_call_under_test"

_MASKED_TRY = """
import pytest

def test_masked():
    try:  {marker}
        value = int("3")
    except ValueError:
        pytest.skip("environment")
    assert value == 3
"""


def _lines(marker: str) -> list[str]:
    return _MASKED_TRY.format(marker=marker).splitlines()


# ---- helper: is_line_exempt -----------------------------------------------------------------------


def test_helper_valid_marker_exempts_its_line():
    lines = _lines(f"# audit-exempt: {_CHECK} -- optional guard, reviewed")
    assert is_line_exempt(lines, 5, _CHECK) is True


@pytest.mark.parametrize("marker", [
    f"# audit-exempt: {_CHECK}",
    f"# audit-exempt: {_CHECK} --",
    f"# audit-exempt: {_CHECK} --    ",
])
def test_helper_marker_without_reason_is_not_an_exemption(marker: str):
    assert is_line_exempt(_lines(marker), 5, _CHECK) is False


def test_helper_marker_for_a_different_check_does_not_suppress():
    lines = _lines("# audit-exempt: hardcoded_absolute_path_in_test -- unrelated reason")
    assert is_line_exempt(lines, 5, _CHECK) is False


def test_helper_marker_on_another_line_does_not_suppress():
    lines = _lines("") + [f"# audit-exempt: {_CHECK} -- reason on a different line"]
    assert is_line_exempt(lines, 5, _CHECK) is False


def test_helper_out_of_range_line_is_not_exempt():
    assert is_line_exempt(["x = 1"], 0, _CHECK) is False
    assert is_line_exempt(["x = 1"], 5, _CHECK) is False


# ---- scanner wiring: except_skip_masks_call_under_test --------------------------------------------


def test_scanner_exempt_line_is_not_reported(tmp_path: Path):
    _write(tmp_path, "test_masked.py", _MASKED_TRY.format(marker=f"# audit-exempt: {_CHECK} -- optional dep guard"))
    assert scan_except_skip_masks_call_under_test(tmp_path) == []


def test_scanner_unexempted_line_is_still_reported(tmp_path: Path):
    _write(tmp_path, "test_masked.py", _MASKED_TRY.format(marker=""))
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert [f.check for f in findings] == [_CHECK]


def test_scanner_marker_without_reason_is_still_reported(tmp_path: Path):
    _write(tmp_path, "test_masked.py", _MASKED_TRY.format(marker=f"# audit-exempt: {_CHECK} --"))
    assert [f.check for f in scan_except_skip_masks_call_under_test(tmp_path)] == [_CHECK]


def test_scanner_marker_for_another_check_is_still_reported(tmp_path: Path):
    _write(tmp_path, "test_masked.py", _MASKED_TRY.format(marker="# audit-exempt: wall_clock_assertion -- wrong check"))
    assert [f.check for f in scan_except_skip_masks_call_under_test(tmp_path)] == [_CHECK]


# ---- scanner wiring: hardcoded_absolute_path_in_test ----------------------------------------------


def test_hardcoded_path_exempt_line_is_not_reported(tmp_path: Path):
    src = 'PATH = "C:/data/table"  # audit-exempt: hardcoded_absolute_path_in_test -- classifier input\n'
    _write(tmp_path, "test_paths.py", src)
    assert scan_hardcoded_absolute_path_in_test(tmp_path) == []


def test_hardcoded_path_without_marker_is_reported(tmp_path: Path):
    _write(tmp_path, "test_paths.py", 'PATH = "C:/data/table"\n')
    assert [f.check for f in scan_hardcoded_absolute_path_in_test(tmp_path)] == ["hardcoded_absolute_path_in_test"]
