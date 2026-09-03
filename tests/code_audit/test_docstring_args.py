"""Scanner tests for docstring_args, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_docstring_args_completeness,
)

from ._helpers import _write

# ---- docstring_args_incomplete ---------------------------------------------


def test_docstring_args_incomplete_missing_param_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
def f(a, b):
    """Do a thing.

    Args:
        a: the first thing.
    """
    return a + b
''')
    findings = scan_docstring_args_completeness(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "docstring_args_incomplete"
    assert "b" in findings[0].detail


def test_docstring_args_incomplete_all_documented_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(a, b):
    """Do a thing.

    Args:
        a: the first thing.
        b: the second thing.
    """
    return a + b
''')
    findings = scan_docstring_args_completeness(tmp_path)
    assert findings == []


def test_docstring_args_incomplete_no_args_section_is_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(a, b):
    """Do a thing."""
    return a + b
''')
    findings = scan_docstring_args_completeness(tmp_path)
    assert findings == []
