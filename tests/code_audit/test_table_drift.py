"""Scanner tests for table_drift, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_table_header_row_drift,
)

from ._helpers import _write

# ---- table_header_row_drift ---------------------------------------------


def test_table_header_row_drift_flags_dictwriter_key_mismatch(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import csv


def dump(fh, rows):
    writer = csv.DictWriter(fh, fieldnames=["a", "b", "c"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"a": row[0], "b": row[1]})
""")
    findings = scan_table_header_row_drift(tmp_path)
    assert any(f.severity == "P1" for f in findings), findings


def test_table_header_row_drift_accepts_matching_keys(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import csv


def dump(fh, rows):
    writer = csv.DictWriter(fh, fieldnames=["a", "b"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"a": row[0], "b": row[1]})
""")
    assert scan_table_header_row_drift(tmp_path) == []


# ---- F98: table drift, per writer variable ------------------------------------------


def test_table_header_row_drift_keys_the_header_by_writer(tmp_path: Path):
    _write(tmp_path, "a.py", """
import csv


def go(f1, f2):
    w1 = csv.DictWriter(f1, fieldnames=["a", "b"])
    w1.writerow({"a": 1, "b": 2})
    w2 = csv.DictWriter(f2, fieldnames=["x", "y"])
    w2.writerow({"x": 1, "y": 2})
""")
    assert scan_table_header_row_drift(tmp_path) == []


def test_table_header_row_drift_still_flags_a_real_drift(tmp_path: Path):
    _write(tmp_path, "a.py", """
import csv


def go(f1):
    w1 = csv.DictWriter(f1, fieldnames=["a", "b"])
    w1.writerow({"a": 1, "c": 2})
""")
    assert len(scan_table_header_row_drift(tmp_path)) == 1
