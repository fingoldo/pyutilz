"""Scanner tests for column_no_write_path, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.column_no_write_path import scan_column_no_write_path

from ._helpers import _write

# ---- column_no_write_path ------------------------------------------------

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS new_upwork.scraper_runs (
    id          bigserial PRIMARY KEY,
    scraper     text NOT NULL,
    started_at  timestamp DEFAULT (now() AT TIME ZONE 'utc'),
    duration_s  double precision,
    retries     integer
);
"""

_DASHBOARD = """
PANEL = "SELECT scraper, duration_s, retries FROM new_upwork.scraper_runs ORDER BY started_at DESC"
"""


def _sql(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / "sql").mkdir(exist_ok=True)
    (tmp_path / "sql" / name).write_text(text, encoding="utf-8")


def test_column_no_write_path_flags_a_column_nothing_fills(tmp_path: Path):
    """`retries` is read by the dashboard and written by nothing: NULL forever."""
    _sql(tmp_path, "create_runs.sql", _RUNS_DDL)
    _write(
        tmp_path,
        "writer.py",
        """
SQL = "INSERT INTO new_upwork.scraper_runs (scraper, duration_s) VALUES (%s, %s)"
""",
    )
    _write(tmp_path, "dashboard.py", _DASHBOARD)
    findings = scan_column_no_write_path(tmp_path)
    assert len(findings) == 1, findings
    assert "retries" in findings[0].detail


def test_column_no_write_path_accepts_a_written_column(tmp_path: Path):
    """The correct form: every read column has an INSERT that names it."""
    _sql(tmp_path, "create_runs.sql", _RUNS_DDL)
    _write(
        tmp_path,
        "writer.py",
        """
SQL = "INSERT INTO new_upwork.scraper_runs (scraper, duration_s, retries) VALUES (%s, %s, %s)"
""",
    )
    _write(tmp_path, "dashboard.py", _DASHBOARD)
    assert scan_column_no_write_path(tmp_path) == []


def test_column_no_write_path_accepts_a_column_with_a_default(tmp_path: Path):
    """`started_at` fills itself, so no statement needs to name it."""
    _sql(tmp_path, "create_runs.sql", _RUNS_DDL)
    _write(
        tmp_path,
        "writer.py",
        """
SQL = "INSERT INTO new_upwork.scraper_runs (scraper, duration_s, retries) VALUES (%s, %s, %s)"
""",
    )
    _write(
        tmp_path,
        "dashboard.py",
        """
PANEL = "SELECT scraper, started_at FROM new_upwork.scraper_runs"
""",
    )
    assert scan_column_no_write_path(tmp_path) == []


def test_column_no_write_path_ignores_a_column_nobody_reads(tmp_path: Path):
    """A table this tree only declares, and neither writes nor reads, is not this rule's business."""
    _sql(tmp_path, "create_runs.sql", _RUNS_DDL)
    assert scan_column_no_write_path(tmp_path) == []


def test_column_no_write_path_counts_a_bare_update_as_a_write(tmp_path: Path):
    """A column filled by an UPDATE, never by the INSERT, still has a writing path.

    An earlier version of this test used an upsert whose `DO UPDATE SET retries = EXCLUDED.retries`
    sat in the same statement as its `INSERT INTO` -- so it passed on the INSERT alone and proved
    nothing about the UPDATE half of the pattern.
    """
    _sql(tmp_path, "create_runs.sql", _RUNS_DDL)
    _write(
        tmp_path,
        "writer.py",
        """
INSERT_SQL = "INSERT INTO new_upwork.scraper_runs (scraper) VALUES (%s)"

FINISH_SQL = "UPDATE new_upwork.scraper_runs SET retries = %s WHERE id = %s"
""",
    )
    _write(tmp_path, "dashboard.py", _DASHBOARD)
    assert [f for f in scan_column_no_write_path(tmp_path) if "retries" in f.detail] == []


def test_column_no_write_path_ignores_a_commented_out_table(tmp_path: Path):
    """A block-commented CREATE TABLE is not a declaration, and reporting one is pure phantom.

    An earlier version commented out a single COLUMN line instead, which the column pattern
    rejects anyway -- so it passed with comment-stripping removed and proved nothing.
    """
    _sql(
        tmp_path,
        "create_runs.sql",
        """
/*
CREATE TABLE IF NOT EXISTS new_upwork.scraper_runs (
    scraper     text NOT NULL,
    retries     integer
);
*/
""",
    )
    _write(tmp_path, "dashboard.py", _DASHBOARD)
    assert scan_column_no_write_path(tmp_path) == []
