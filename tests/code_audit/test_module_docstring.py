"""Scanner tests for module_docstring, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_duplicate_module_docstring,
)

from ._helpers import _write

# ---- duplicate_module_docstring ------------------------------------------


def test_duplicate_module_docstring_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
"""First docstring."""
"""Second docstring, silently discarded."""

def f():
    pass
''')
    findings = scan_duplicate_module_docstring(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Low"


def test_duplicate_module_docstring_single_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
"""Only docstring."""

def f():
    pass
''')
    findings = scan_duplicate_module_docstring(tmp_path)
    assert findings == []
