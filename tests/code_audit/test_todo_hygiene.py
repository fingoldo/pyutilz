"""Scanner tests for todo_hygiene, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_todo_hygiene,
)

from ._helpers import _write

# ---- todo_hygiene -------------------------------------------------------------


def test_todo_hygiene_unattributed_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", "# TODO: handle empty list case\nx = 1\n")
    findings = scan_todo_hygiene(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "todo_hygiene"


def test_todo_hygiene_dated_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", "# TODO 2026-04-28: handle empty list case\nx = 1\n")
    assert scan_todo_hygiene(tmp_path) == []


def test_todo_hygiene_assignee_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", "# TODO(alice): handle empty list case\nx = 1\n")
    assert scan_todo_hygiene(tmp_path) == []
