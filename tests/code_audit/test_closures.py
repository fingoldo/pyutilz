"""Scanner tests for closures, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_late_binding_closures,
)

from ._helpers import _write

# ---- late_binding_closure ----------------------------------------------


def test_late_binding_lambda_in_for_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def build():
    callbacks = []
    for x in range(5):
        callbacks.append(lambda: x * 2)
    return callbacks
""")
    findings = scan_late_binding_closures(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].severity == "P1"
    assert findings[0].check == "late_binding_closure"


def test_late_binding_lambda_with_default_arg_safe(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def build():
    callbacks = []
    for x in range(5):
        callbacks.append(lambda x=x: x * 2)
    return callbacks
""")
    findings = scan_late_binding_closures(tmp_path)
    assert findings == [], findings


def test_sync_lambda_in_sorted_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def order(items):
    for group in items:
        sorted(group, key=lambda k: group[k])
""")
    # The lambda doesn't escape the iteration (sorted is synchronous).
    findings = scan_late_binding_closures(tmp_path)
    assert findings == []
