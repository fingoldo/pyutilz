"""Scanner tests for bare_except, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_bare_except,
)

from ._helpers import _write

# ---- bare_except -----------------------------------------------------------


def test_bare_except_bare_colon_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f():
    try:
        risky()
    except:
        pass
""")
    findings = scan_bare_except(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "bare_except"


def test_bare_except_base_exception_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f():
    try:
        risky()
    except BaseException:
        pass
""")
    findings = scan_bare_except(tmp_path)
    assert len(findings) == 1, findings


def test_bare_except_base_exception_reraise_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    try:
        risky()
    except BaseException:
        cleanup()
        raise
""")
    assert scan_bare_except(tmp_path) == []


def test_bare_except_narrow_exception_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    try:
        risky()
    except Exception:
        pass
""")
    assert scan_bare_except(tmp_path) == []
