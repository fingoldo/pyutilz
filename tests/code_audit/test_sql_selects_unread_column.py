"""Scanner tests for sql_selects_unread_column, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.sql_selects_unread_column import scan_sql_selects_unread_column

from ._helpers import _write

# ---- sql_selects_unread_column -------------------------------------------


def test_sql_selects_unread_column_flags_the_canonical_case(tmp_path: Path):
    """Four columns fetched, four bound, one never read -- the quiet shape that ships."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL = "SELECT id, uid, payload, updated_at FROM jobs WHERE ts > 1"

def load(cur):
    cur.execute(SQL)
    for job_id, uid, payload, updated_at in cur:
        handle(job_id, uid, payload)
''',
    )
    findings = scan_sql_selects_unread_column(tmp_path)
    assert len(findings) == 1, findings
    assert "updated_at" in findings[0].detail


def test_sql_selects_unread_column_accepts_an_underscored_binding(tmp_path: Path):
    """`_` is how a deliberately-ignored column is spelled, and it is not a defect."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL = "SELECT id, uid, payload, updated_at FROM jobs"

def load(cur):
    cur.execute(SQL)
    for job_id, uid, payload, _updated_at in cur:
        handle(job_id, uid, payload)
''',
    )
    assert scan_sql_selects_unread_column(tmp_path) == []


def test_sql_selects_unread_column_declines_a_star_select(tmp_path: Path):
    """`SELECT *` names no columns, so there is nothing to compare the unpacking against."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL = "SELECT * FROM jobs"

def load(cur):
    cur.execute(SQL)
    for job_id, uid, payload, updated_at in cur:
        handle(job_id, uid, payload)
''',
    )
    assert scan_sql_selects_unread_column(tmp_path) == []


def test_sql_selects_unread_column_declines_two_queries_in_one_function(tmp_path: Path):
    """With two SELECTs it cannot say which unpacking belongs to which, and a coin flip here
    would be worse than silence."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL_A = "SELECT id, uid, payload, updated_at FROM jobs"
SQL_B = "SELECT id, uid FROM clients"

def load(cur):
    cur.execute(SQL_A)
    for job_id, uid, payload, updated_at in cur:
        handle(job_id, uid, payload)
    cur.execute(SQL_B)
''',
    )
    assert scan_sql_selects_unread_column(tmp_path) == []


def test_sql_selects_unread_column_accepts_every_column_read(tmp_path: Path):
    """The correct form, which must stay silent."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL = "SELECT id, uid, payload FROM jobs"

def load(cur):
    cur.execute(SQL)
    for job_id, uid, payload in cur:
        handle(job_id, uid, payload)
''',
    )
    assert scan_sql_selects_unread_column(tmp_path) == []


# ---- F82: sql_selects_unread_column needs a cursor link -----------------------------


def test_sql_selects_unread_column_ignores_an_unrelated_unpacking(tmp_path: Path):
    _write(tmp_path, "a.py", """
def q(cur):
    cur.execute("SELECT id, name FROM t")
    lo, hi = compute()
    return lo
""")
    assert scan_sql_selects_unread_column(tmp_path) == []


def test_sql_selects_unread_column_still_flags_a_cursor_unpacking(tmp_path: Path):
    _write(tmp_path, "a.py", """
def q(cur):
    cur.execute("SELECT id, name FROM t")
    rows = cur.fetchall()
    for a, b in rows:
        print(a)
""")
    assert len(scan_sql_selects_unread_column(tmp_path)) == 1
