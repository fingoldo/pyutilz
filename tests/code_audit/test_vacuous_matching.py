"""Scanner tests for vacuous_matching, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_partial_guard_across_siblings,
    scan_vacuous_empty_pattern_match,
)

from ._helpers import _write

# ---- vacuous_empty_pattern_match ----------------------------------------


def test_vacuous_empty_pattern_match_flags_unguarded_all(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def covers(text, stems):
    return all(stem in text for stem in stems)
""")
    findings = scan_vacuous_empty_pattern_match(tmp_path)
    assert len(findings) == 1 and findings[0].check == "vacuous_empty_pattern_match", findings


def test_vacuous_empty_pattern_match_accepts_a_guard_and_ignores_any(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def covers(text, stems):
    return bool(stems) and all(stem in text for stem in stems)


def touches(text, stems):
    return any(stem in text for stem in stems)
""")
    assert scan_vacuous_empty_pattern_match(tmp_path) == []


# ---- F67: guard matching is word-bounded --------------------------------------------


def test_vacuous_empty_pattern_match_does_not_take_xs_as_a_guard_for_x(tmp_path: Path):
    _write(tmp_path, "a.py", """
def f(x, xs):
    if xs:
        pass
    return all(i in "abc" for i in x)
""")
    assert len(scan_vacuous_empty_pattern_match(tmp_path)) == 1


def test_partial_guard_across_siblings_word_bounds_the_parameter(tmp_path: Path):
    _write(tmp_path, "a.py", """
def score_triples(pred_triples, pred_triples_raw):
    if pred_triples:
        return 1
    return 0


def score_junk(pred_triples, pred_triples_raw):
    if pred_triples:
        return 1
    return 0


def score_grounding(pred_triples, pred_triples_raw):
    if pred_triples_raw:
        return 1
    return 0
""")
    findings = scan_partial_guard_across_siblings(tmp_path)
    assert len(findings) == 1 and "score_grounding" in findings[0].detail
