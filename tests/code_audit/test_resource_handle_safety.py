"""Scanner tests for resource_handle_safety, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_resource_handle_safety,
)

from ._helpers import _write

# ---- resource_handle_safety --------------------------------------------------


def test_resource_handle_safety_bare_open_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(path):
    fh = open(path, "w")
    fh.write("x")
""")
    findings = scan_resource_handle_safety(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "resource_handle_safety"


def test_resource_handle_safety_popen_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import subprocess

def f():
    proc = subprocess.Popen(["ls"])
    return proc
""")
    findings = scan_resource_handle_safety(tmp_path)
    assert len(findings) == 1, findings


def test_resource_handle_safety_with_block_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(path):
    with open(path, "w") as fh:
        fh.write("x")
""")
    assert scan_resource_handle_safety(tmp_path) == []


# ---- F204: resource handle safety, attribute-form open ------------------------------


def test_resource_handle_safety_sees_path_open(tmp_path: Path):
    _write(tmp_path, "a.py", """
from pathlib import Path


def f(p):
    fh = Path(p).open()
    return fh.read()
""")
    assert len(scan_resource_handle_safety(tmp_path)) == 1


def test_resource_handle_safety_accepts_path_open_in_a_with(tmp_path: Path):
    _write(tmp_path, "a.py", """
from pathlib import Path


def f(p):
    with Path(p).open() as fh:
        return fh.read()
""")
    assert scan_resource_handle_safety(tmp_path) == []
