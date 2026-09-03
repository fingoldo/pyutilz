"""Scanner tests for locals_get, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_locals_get_fragile_lookup,
)

from ._helpers import _write

# ---- locals_get_fragile_lookup --------------------------------------------


def test_locals_get_fragile_lookup_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(flag):
    if flag:
        cached_result = compute()
    return locals().get("cached_result", None)
""")
    findings = scan_locals_get_fragile_lookup(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "locals_get_fragile_lookup"
    assert findings[0].severity == "P1"


def test_globals_get_fragile_lookup_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f():
    return globals().get("some_name", None)
""")
    findings = scan_locals_get_fragile_lookup(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "locals_get_fragile_lookup"


def test_locals_get_normal_variable_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(flag):
    cached_result = None
    if flag:
        cached_result = compute()
    return cached_result
""")
    findings = scan_locals_get_fragile_lookup(tmp_path)
    assert findings == []


def test_locals_dict_other_method_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    return list(locals().keys())
""")
    findings = scan_locals_get_fragile_lookup(tmp_path)
    assert findings == []
