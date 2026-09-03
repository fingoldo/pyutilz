"""Scanner tests for count_then_fetch_same_table, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.count_then_fetch_same_table import scan_count_then_fetch_same_table

from ._helpers import _write

# ---- count_then_fetch_same_table -----------------------------------------


def test_count_then_fetch_same_table_flags_the_canonical_case(tmp_path: Path):
    """Two round trips for one answer: `len(rows)` already is the count."""
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur):
    cur.execute("SELECT COUNT(*) FROM jobs WHERE stale")
    total = cur.fetchone()[0]
    cur.execute("SELECT id, uid FROM jobs WHERE stale")
    return total, cur.fetchall()
''',
    )
    findings = scan_count_then_fetch_same_table(tmp_path)
    assert len(findings) == 1, findings
    assert "jobs" in findings[0].detail


def test_count_then_fetch_same_table_accepts_a_paginated_fetch(tmp_path: Path):
    """A LIMIT is the one legitimate reason to ask twice: the page does not carry the total."""
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur):
    cur.execute("SELECT COUNT(*) FROM jobs WHERE stale")
    total = cur.fetchone()[0]
    cur.execute("SELECT id, uid FROM jobs WHERE stale LIMIT 100")
    return total, cur.fetchall()
''',
    )
    assert scan_count_then_fetch_same_table(tmp_path) == []


def test_count_then_fetch_same_table_accepts_a_grouped_count(tmp_path: Path):
    """A GROUP BY answers a breakdown the fetched rows do not contain."""
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur):
    cur.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
    by_status = cur.fetchall()
    cur.execute("SELECT id, uid FROM jobs")
    return by_status, cur.fetchall()
''',
    )
    assert scan_count_then_fetch_same_table(tmp_path) == []


def test_count_then_fetch_same_table_accepts_a_different_table(tmp_path: Path):
    """Counting one table and fetching another is two answers, not one asked twice."""
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur):
    cur.execute("SELECT COUNT(*) FROM clients")
    total = cur.fetchone()[0]
    cur.execute("SELECT id, uid FROM jobs")
    return total, cur.fetchall()
''',
    )
    assert scan_count_then_fetch_same_table(tmp_path) == []


def test_count_then_fetch_same_table_declines_an_interpolated_table(tmp_path: Path):
    """An interpolated table name renders as `?`, not as nothing.

    Dropping it spliced the surrounding text together, the table was read as `where`, and that
    supplied both of this rule's first two hits against real code.
    """
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur, table):
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE stale")
    total = cur.fetchone()[0]
    cur.execute(f"SELECT id FROM {table} WHERE stale")
    return total, cur.fetchall()
''',
    )
    assert scan_count_then_fetch_same_table(tmp_path) == []
