"""Scanner tests for sql_lint, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_sql_aggregate_before_cast,
    scan_sql_limit_without_order_by,
    scan_sql_offset_pagination,
)

from ._helpers import _write

# ---- sql_limit_without_order_by -----------------------------------------


def test_sql_limit_without_order_by_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
from sqlalchemy import text

def f(session):
    return session.execute(text("""
        SELECT id FROM widgets WHERE flag IS NULL LIMIT :n
    """))
''')
    findings = scan_sql_limit_without_order_by(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "sql_limit_without_order_by"
    assert findings[0].severity == "P2"


def test_sql_limit_with_order_by_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(session):
    return session.execute("""
        SELECT id FROM widgets WHERE flag IS NULL ORDER BY id LIMIT :n
    """)
''')
    assert scan_sql_limit_without_order_by(tmp_path) == []


def test_sql_limit_1_exempted(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(session):
    return session.execute("SELECT id FROM widgets LIMIT 1")
''')
    assert scan_sql_limit_without_order_by(tmp_path) == []


def test_sql_limit_non_sql_string_ignored(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
LIMIT_MSG = "please stay under the LIMIT of 10 items"
''')
    assert scan_sql_limit_without_order_by(tmp_path) == []


# ---- sql_offset_pagination ------------------------------------------------


def test_sql_offset_pagination_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
def f(session, offset):
    return session.execute("""
        SELECT id FROM widgets WHERE flag IS NULL
        ORDER BY id LIMIT :n OFFSET :offset
    """)
''')
    findings = scan_sql_offset_pagination(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "sql_offset_pagination"
    assert findings[0].severity == "Low"


def test_sql_limit_without_offset_not_flagged_by_offset_scanner(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(session):
    return session.execute("SELECT id FROM widgets ORDER BY id LIMIT :n")
''')
    assert scan_sql_offset_pagination(tmp_path) == []


def test_sql_offset_pagination_ignores_module_docstring_mentioning_sql_keywords(tmp_path: Path):
    _write(tmp_path, "pkg_init.py", '''
"""Documents this package's scanners.

``scan_sql_offset_pagination``: a SQL literal combining ``LIMIT`` and ``OFFSET``. Advisory --
flags the pattern so a reviewer can confirm the query is a SELECT with a stable filtered set.
"""
''')
    assert scan_sql_offset_pagination(tmp_path) == []
    assert scan_sql_limit_without_order_by(tmp_path) == []


def test_sql_offset_pagination_ignores_class_and_function_docstrings(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
class Foo:
    """A SELECT with LIMIT and OFFSET is discussed here, not executed."""

    def bar(self):
        """Same SELECT/LIMIT/OFFSET vocabulary, still just prose."""
        return 1
''')
    assert scan_sql_offset_pagination(tmp_path) == []


def test_sql_offset_pagination_still_flags_real_sql_after_a_docstring(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
"""This module talks about SELECT, LIMIT and OFFSET in prose."""

def f(session, offset):
    return session.execute("""
        SELECT id FROM widgets WHERE flag IS NULL
        ORDER BY id LIMIT :n OFFSET :offset
    """)
''')
    findings = scan_sql_offset_pagination(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "sql_offset_pagination"


# ---- sql_aggregate_before_cast --------------------------------------------


def test_sql_aggregate_before_cast_json_extract_no_cast_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
def latest_count(cur):
    cur.execute("SELECT MAX(data->>'count') FROM events")
''')
    findings = scan_sql_aggregate_before_cast(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "sql_aggregate_before_cast"
    assert findings[0].severity == "P2"


def test_sql_aggregate_before_cast_with_cast_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def latest_count(cur):
    cur.execute("SELECT MAX((data->>'count')::int) FROM events")
''')
    findings = scan_sql_aggregate_before_cast(tmp_path)
    assert findings == []


def test_sql_aggregate_before_cast_no_json_extract_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def latest(cur):
    cur.execute("SELECT MAX(created_at) FROM events")
''')
    findings = scan_sql_aggregate_before_cast(tmp_path)
    assert findings == []


# ---- F101: sql_lint sees an f-string query ------------------------------------------


def test_sql_limit_without_order_by_sees_an_fstring_query(tmp_path: Path):
    _write(tmp_path, "a.py", "def q(user_id):\n    sql = f'SELECT id, name FROM users WHERE owner = {user_id} LIMIT 50'\n    return sql\n")
    assert len(scan_sql_limit_without_order_by(tmp_path)) == 1


def test_sql_limit_without_order_by_accepts_an_ordered_fstring_query(tmp_path: Path):
    _write(tmp_path, "a.py", "def q(user_id):\n    sql = f'SELECT id, name FROM users WHERE owner = {user_id} ORDER BY id LIMIT 50'\n    return sql\n")
    assert scan_sql_limit_without_order_by(tmp_path) == []
