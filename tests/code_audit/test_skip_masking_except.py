"""Scanner tests for skip_masking_except, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_except_skip_masks_call_under_test,
)

from ._helpers import _write

# ---- except_skip_masks_call_under_test -----------------------------------


def test_except_skip_masks_real_call_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_bad.py",
        """
import pytest

def test_something():
    try:
        result = train_model(x=1, y=2)
    except Exception:
        pytest.skip("environment issue")
    assert result is not None
""",
    )
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "except_skip_masks_call_under_test"


def test_except_skip_import_guard_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_ok.py",
        """
import pytest

def test_something():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
""",
    )
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert findings == []


def test_except_no_skip_call_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_ok.py",
        """
def test_something():
    try:
        result = train_model(x=1, y=2)
    except Exception:
        raise
""",
    )
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert findings == []


def test_except_skip_non_test_file_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "helper.py",
        """
import pytest

def something():
    try:
        result = train_model(x=1, y=2)
    except Exception:
        pytest.skip("bad")
""",
    )
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert findings == []


# ---- F189: skip_masking_except reads the suffix naming convention -------------------


def test_except_skip_masks_call_under_test_reads_the_suffix_convention(tmp_path: Path):
    _write(tmp_path, "widget_test.py", """
import pytest


def test_x():
    try:
        result = do_work(1)
    except TypeError:
        pytest.skip("not supported")
""")
    assert len(scan_except_skip_masks_call_under_test(tmp_path)) == 1


def test_except_skip_masks_call_under_test_ignores_a_production_module(tmp_path: Path):
    _write(tmp_path, "widget.py", """
import pytest


def check_x():
    try:
        result = do_work(1)
    except TypeError:
        pytest.skip("not supported")
""")
    assert scan_except_skip_masks_call_under_test(tmp_path) == []
