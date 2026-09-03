"""Scanner tests for tautological_guard, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_tautological_guards,
)

from ._helpers import _write

# ---- tautological_guard --------------------------------------------------


def test_tautological_guard_flags_threshold_anded_with_identity_pin(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def label(causes, lead):
    for c in causes:
        if c.mean >= 0.75 * lead and c is causes[0]:
            return "strongly supported"
    return "weak"
""")
    findings = scan_tautological_guards(tmp_path)
    assert len(findings) == 1 and findings[0].check == "tautological_guard", findings


def test_tautological_guard_ignores_none_checks_and_distinct_targets(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def label(a, b, lead):
    if a.mean >= 0.75 * lead and a is not None:
        return 1
    if a.mean >= lead and b is lead:
        return 2
    return 0
""")
    assert scan_tautological_guards(tmp_path) == []


# ---- F32: tautological guard, the pin must fix the thresholded value ----------------


def test_tautological_guards_ignores_a_pin_on_a_different_attribute(tmp_path: Path):
    _write(tmp_path, "a.py", """
def go(item):
    if item.score > 0.5 and item.label == "ok":
        return 1
    return 0
""")
    assert scan_tautological_guards(tmp_path) == []


def test_tautological_guards_still_flags_a_pin_on_the_whole_object(tmp_path: Path):
    _write(tmp_path, "a.py", """
GOLD = object()


def go(item):
    if item.score > 0.5 and item == GOLD:
        return 1
    return 0
""")
    assert len(scan_tautological_guards(tmp_path)) == 1
