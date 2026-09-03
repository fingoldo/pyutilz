"""Scanner tests for provenance_flow, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_record_field_flow,
)

from ._helpers import _write

# ---- record_field_flow ---------------------------------------------------


def test_record_field_flow_flags_defaulted_read_of_an_unwritten_key(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def build(hedge):
    return {"mechanism_modality_source": hedge}


def weight(triple):
    return triple.get("mechanism_modality_sources", "unstated")
""")
    findings = scan_record_field_flow(tmp_path)
    assert [f.check for f in findings if f.severity == "P1"] == ["field_read_never_written"], findings


def test_record_field_flow_near_miss_only_ignores_a_foreign_schema_key(tmp_path: Path):
    """A key of somebody else's JSON resembles nothing this tree writes, and must not be reported."""
    _write(tmp_path, "client.py", """
def parse(response):
    return response.get("esearchresult", {})
""")
    assert scan_record_field_flow(tmp_path) == []
    assert scan_record_field_flow(tmp_path, near_miss_only=False), "the exhaustive form must still see it"


def test_record_field_flow_ignores_a_key_with_both_sides(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def build():
    return {"modality_source": "code"}


def read(row):
    return row.get("modality_source", "")
""")
    assert scan_record_field_flow(tmp_path) == []


# ---- F22/F170/F171: provenance flow -------------------------------------------------


def test_record_field_flow_ignores_a_read_without_a_default(tmp_path: Path):
    _write(tmp_path, "a.py", """
def w():
    return {"modality_source": 1}


def r(d):
    return d["modality_sources"]
""")
    assert scan_record_field_flow(tmp_path) == []


def test_record_field_flow_still_flags_a_defaulted_read(tmp_path: Path):
    _write(tmp_path, "a.py", """
def w():
    return {"modality_source": 1}


def r(d):
    return d.get("modality_sources", 0)
""")
    assert len(scan_record_field_flow(tmp_path)) == 1


def test_record_field_flow_reports_a_dict_key_at_its_own_line(tmp_path: Path):
    _write(tmp_path, "a.py", """
def w():
    return {
        "modality_source": 1,
        "quote_tier": 2,
    }
""")
    findings = scan_record_field_flow(tmp_path, report_written_never_read=True)
    assert sorted(f.line for f in findings) == [3, 4]


def test_record_field_flow_skips_a_private_key():
    from pyutilz.dev.code_audit.provenance_flow import _is_field_like

    assert _is_field_like("modality_source") is True
    assert _is_field_like("_secret") is False
