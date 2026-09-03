"""Scanner tests for return_annotation, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_return_annotation_mismatch,
)

from ._helpers import _write

# ---- return_annotation_mismatch --------------------------------------------


def test_return_annotation_mismatch_tuple_literal_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(x) -> float:
    if x < 0:
        return (0.0, 1.0)
    return x
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P2"


def test_return_annotation_mismatch_bare_return_none_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(x) -> str:
    if not x:
        return
    return x
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert len(findings) == 1


def test_return_annotation_mismatch_consistent_scalar_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x) -> float:
    if x < 0:
        return 0.0
    return x
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert findings == []


def test_return_annotation_mismatch_optional_annotation_is_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
from typing import Optional

def f(x) -> Optional[float]:
    if x < 0:
        return None
    return x
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert findings == []


def test_return_annotation_mismatch_nested_function_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x) -> float:
    def helper():
        return [1, 2, 3]
    return x + len(helper())
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert findings == []
