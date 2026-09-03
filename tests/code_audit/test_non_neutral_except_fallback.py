"""Scanner tests for non_neutral_except_fallback, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.non_neutral_except_fallback import scan_non_neutral_except_fallback

from ._helpers import _write

# ---- F19/F75/F167: non-neutral except fallback --------------------------------------


def test_non_neutral_except_fallback_ignores_a_nested_callback(tmp_path: Path):
    _write(tmp_path, "m.py", """
def f():
    try:
        g()
    except ValueError:
        def cb():
            return 0.0
        register(cb)
        raise
""")
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_non_neutral_except_fallback_sees_annotated_and_tuple_fallbacks(tmp_path: Path):
    _write(tmp_path, "m.py", """
def a():
    try:
        g()
    except ValueError:
        v: float = 0.0
    return v


def b():
    try:
        g()
    except ValueError:
        p, q = 0.0, 0.0
    return p
""")
    assert len(scan_non_neutral_except_fallback(tmp_path)) == 2


def test_non_neutral_except_fallback_names_the_first_substitution(tmp_path: Path):
    _write(tmp_path, "m.py", """
def f(k):
    try:
        g()
    except ValueError:
        if k:
            return 1.0
        return 2.0
""")
    findings = scan_non_neutral_except_fallback(tmp_path)
    assert len(findings) == 1 and "returns 1.0" in findings[0].detail
