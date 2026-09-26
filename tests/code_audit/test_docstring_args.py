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


def test_a_prose_colon_in_a_description_is_not_an_argument():
    """``see :func:`other``` inside a description must not be read as a documented argument named ``see``."""
    from pyutilz.dev.code_audit.docstring_args import _documented_arg_names

    doc = "Do it.\n\nArgs:\n    a: the first one --\n        see :func:`other`.\n    b: the second one.\n\nReturns:\n    x."
    assert _documented_arg_names(doc) == {"a", "b"}


def test_every_entry_after_a_wrapped_description_is_read():
    """An entry right after a wrapped description line is still found: the indent may not start on the previous line."""
    from pyutilz.dev.code_audit.docstring_args import _documented_arg_names

    doc = "Do it.\n\nArgs:\n    a: long\n        wrapped text.\n    _b: private but documented.\n\nReturns:\n    x."
    assert _documented_arg_names(doc) == {"a", "_b"}
