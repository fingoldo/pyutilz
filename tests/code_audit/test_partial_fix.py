"""Scanner tests for partial_fix, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_inconsistent_filter,
    scan_partial_guard_across_siblings,
)

from ._helpers import _write

# ---- partial_guard_across_siblings / inconsistent_filter -----------------


def test_partial_guard_across_siblings_flags_the_odd_one_out(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def score_a(rows):
    if not rows:
        return 0.0
    return len(rows)


def score_b(rows):
    if not rows:
        return 0.0
    return len(rows) * 2


def score_c(rows):
    return len(rows) * 3
""")
    findings = scan_partial_guard_across_siblings(tmp_path)
    assert len(findings) == 1 and "score_c" in findings[0].detail, findings


def test_partial_guard_across_siblings_is_silent_when_the_whole_family_guards(tmp_path: Path):
    """The same family as the positive case with the hole closed: three ``score_*`` siblings over
    the same first parameter, each guarding it, and each spelling the guard differently so the
    silence is not an artefact of one template matching every member. The asymmetry IS the
    finding, so a fully-guarded family must produce nothing."""
    _write(tmp_path, "ok.py", """
def score_a(rows):
    if not rows:
        return 0.0
    return len(rows)


def score_b(rows):
    if rows:
        return len(rows) * 2
    return 0.0


def score_c(rows):
    return len(rows) * 3 if bool(rows) else 0.0
""")
    assert scan_partial_guard_across_siblings(tmp_path) == []


def test_inconsistent_filter_is_silent_until_configured(tmp_path: Path):
    _write(tmp_path, "mod.py", """
def rank(graph):
    return graph.causes()
""")
    assert scan_inconsistent_filter(tmp_path) == []
    findings = scan_inconsistent_filter(tmp_path, filter_pairs=(("causes", "postmortem_events"),))
    assert len(findings) == 1 and "rank" in findings[0].detail, findings
