"""Scanner tests for measurement_hygiene, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_regex_integer_parse,
    scan_thresholds_below_documented_result,
)

from ._helpers import _write

# ---- measurement hygiene -------------------------------------------------


def test_regex_integer_parse_truncation_flags_a_bare_digit_class(tmp_path: Path):
    _write(tmp_path, "bad.py", r'''
import re


def count(text):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0
''')
    findings = scan_regex_integer_parse(tmp_path)
    assert len(findings) == 1, findings


def test_regex_integer_parse_truncation_accepts_a_real_number_pattern(tmp_path: Path):
    _write(tmp_path, "ok.py", r'''
import re


def count(text):
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return float(m.group().replace(",", "")) if m else 0.0
''')
    assert scan_regex_integer_parse(tmp_path) == []


def test_threshold_below_documented_result_flags_a_weakened_gate(tmp_path: Path):
    _write(tmp_path, "test_cards.py", '''
def test_cards_recover_the_expected_cause():
    """7 of 8 demonstration cards recover the expected cause."""
    decided = 7
    assert decided >= 6
''')
    findings = scan_thresholds_below_documented_result(tmp_path)
    assert len(findings) == 1, findings


def test_threshold_below_documented_result_accepts_a_matching_gate(tmp_path: Path):
    _write(tmp_path, "test_cards.py", '''
def test_cards_recover_the_expected_cause():
    """7 of 8 demonstration cards recover the expected cause."""
    decided = 7
    assert decided >= 7
''')
    assert scan_thresholds_below_documented_result(tmp_path) == []


# ---- F95: the documented threshold is compared against ASSERT bounds only -----------


def test_thresholds_below_documented_result_ignores_a_loop_guard(tmp_path: Path):
    _write(tmp_path, "test_x.py", '''
def test_cards(values):
    """Recovers 7 of 8 demonstration cards."""
    for i in range(3):
        if i > 0:
            pass
    assert len(values) >= 7
''')
    assert scan_thresholds_below_documented_result(tmp_path) == []


def test_thresholds_below_documented_result_still_flags_a_weak_assert(tmp_path: Path):
    _write(tmp_path, "test_x.py", '''
def test_cards(values):
    """Recovers 7 of 8 demonstration cards."""
    assert len(values) >= 3
''')
    findings = scan_thresholds_below_documented_result(tmp_path)
    assert len(findings) == 1 and findings[0].line == 3
