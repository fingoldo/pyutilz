"""Scanner tests for sql_sibling_missing_time_bound, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.sql_sibling_missing_time_bound import scan_sql_sibling_missing_time_bound

from ._helpers import _write

# ---- sql_sibling_missing_time_bound --------------------------------------


def test_sql_sibling_missing_time_bound_flags_the_unbounded_one(tmp_path: Path):
    """Three statements over one scan table are bounded; the fourth reads every row ever written."""
    _write(
        tmp_path,
        "queries.py",
        '''
RECENT_SQL = "SELECT uid FROM new_upwork.scans WHERE ts > now() - interval \'7 days\'"

TODAY_SQL = "SELECT uid FROM new_upwork.scans WHERE ts >= current_date"

WINDOW_SQL = "SELECT uid FROM new_upwork.scans WHERE ts BETWEEN %s AND %s"

ALL_SQL = "SELECT uid, payload FROM new_upwork.scans ORDER BY uid"
''',
    )
    findings = scan_sql_sibling_missing_time_bound(tmp_path)
    assert len(findings) == 1, findings
    assert "ALL_SQL" in findings[0].detail


def test_sql_sibling_missing_time_bound_accepts_a_key_predicate(tmp_path: Path):
    """A predicate against a caller-supplied parameter bounds the scan to what the caller passed.

    This rule's only hit on real code was `WHERE fj.client_team_uid IN %s` -- a fallback that
    retries a handful of named clients and reads nothing else however large the table gets.
    """
    _write(
        tmp_path,
        "queries.py",
        '''
RECENT_SQL = "SELECT uid FROM new_upwork.scans WHERE ts > now() - interval \'7 days\'"

TODAY_SQL = "SELECT uid FROM new_upwork.scans WHERE ts >= current_date"

WINDOW_SQL = "SELECT uid FROM new_upwork.scans WHERE ts BETWEEN %s AND %s"

RETRY_SQL = "SELECT uid, payload FROM new_upwork.scans WHERE cl_uid IN %s ORDER BY uid"
''',
    )
    assert scan_sql_sibling_missing_time_bound(tmp_path) == []


def test_sql_sibling_missing_time_bound_accepts_a_limit(tmp_path: Path):
    """A bounded fetch is bounded, whatever bounds it."""
    _write(
        tmp_path,
        "queries.py",
        '''
RECENT_SQL = "SELECT uid FROM new_upwork.scans WHERE ts > now() - interval \'7 days\'"

TODAY_SQL = "SELECT uid FROM new_upwork.scans WHERE ts >= current_date"

WINDOW_SQL = "SELECT uid FROM new_upwork.scans WHERE ts BETWEEN %s AND %s"

SAMPLE_SQL = "SELECT uid, payload FROM new_upwork.scans ORDER BY uid LIMIT 100"
''',
    )
    assert scan_sql_sibling_missing_time_bound(tmp_path) == []


def test_sql_sibling_missing_time_bound_needs_two_bounded_siblings(tmp_path: Path):
    """One bounded neighbour is not a convention, and a convention is the only evidence here."""
    _write(
        tmp_path,
        "queries.py",
        '''
RECENT_SQL = "SELECT uid FROM new_upwork.scans WHERE ts > now() - interval \'7 days\'"

ALL_SQL = "SELECT uid, payload FROM new_upwork.scans ORDER BY uid"

OTHER_SQL = "SELECT uid FROM new_upwork.clients ORDER BY uid"
''',
    )
    assert scan_sql_sibling_missing_time_bound(tmp_path) == []


def test_sql_sibling_missing_time_bound_ignores_a_different_table(tmp_path: Path):
    """The convention belongs to one table; a query over another table is not in the family."""
    _write(
        tmp_path,
        "queries.py",
        '''
RECENT_SQL = "SELECT uid FROM new_upwork.scans WHERE ts > now() - interval \'7 days\'"

TODAY_SQL = "SELECT uid FROM new_upwork.scans WHERE ts >= current_date"

WINDOW_SQL = "SELECT uid FROM new_upwork.scans WHERE ts BETWEEN %s AND %s"

CLIENTS_SQL = "SELECT uid, name FROM new_upwork.clients ORDER BY uid"
''',
    )
    assert scan_sql_sibling_missing_time_bound(tmp_path) == []
