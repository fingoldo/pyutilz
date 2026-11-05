"""Schema loading, commit, verbosity, logging and rollback pins for database.db.sqlite (mutation survivors)."""

from __future__ import annotations

import logging
import sqlite3

import pytest

from pyutilz.database.db import sqlite as sq
from pyutilz.database.db.sql_helpers import SQLValidationError


@pytest.fixture
def file_db(tmp_path):
    path = str(tmp_path / "db.sqlite")
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    yield path, conn, cur
    conn.close()


class TestEnsureTables:
    def test_the_default_schema_path_is_database_schema_sql(self, file_db, tmp_path, monkeypatch):
        path, conn, cur = file_db
        (tmp_path / "database").mkdir()
        (tmp_path / "database" / "schema.sql").write_text("CREATE TABLE made_here (a INTEGER);", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert sq.ensure_db_tables_created(conn, cur) is True
        names = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "made_here" in names

    def test_an_explicit_schema_path_is_used(self, file_db, tmp_path):
        path, conn, cur = file_db
        schema = tmp_path / "custom.sql"
        schema.write_text("CREATE TABLE explicit (a INTEGER);", encoding="utf-8")
        assert sq.ensure_db_tables_created(conn, cur, str(schema)) is True

    def test_a_missing_schema_file_logs_and_returns_false(self, file_db, tmp_path, caplog):
        path, conn, cur = file_db
        caplog.set_level(logging.ERROR)
        assert sq.ensure_db_tables_created(conn, cur, str(tmp_path / "nope.sql")) is False
        assert "DB Schema file not found." in caplog.text

    def test_an_empty_schema_file_logs_and_returns_false(self, file_db, tmp_path, caplog):
        path, conn, cur = file_db
        caplog.set_level(logging.ERROR)
        schema = tmp_path / "empty.sql"
        schema.write_text("", encoding="utf-8")
        assert sq.ensure_db_tables_created(conn, cur, str(schema)) is False
        assert "DB Schema empty." in caplog.text

    def test_a_one_character_schema_is_not_empty(self, file_db, tmp_path):
        path, conn, cur = file_db
        schema = tmp_path / "one.sql"
        schema.write_text(";", encoding="utf-8")
        assert sq.ensure_db_tables_created(conn, cur, str(schema)) is True

    def test_the_schema_is_committed_for_another_connection(self, file_db, tmp_path):
        path, conn, cur = file_db
        schema = tmp_path / "c.sql"
        schema.write_text("CREATE TABLE committed (a INTEGER);", encoding="utf-8")
        sq.ensure_db_tables_created(conn, cur, str(schema))
        other = sqlite3.connect(path)
        try:
            names = [r[0] for r in other.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        finally:
            other.close()
        assert "committed" in names


class TestInsert:
    def test_the_identifiers_are_validated(self, file_db):
        path, conn, cur = file_db
        cur.execute("CREATE TABLE t (a INTEGER)")
        with pytest.raises(SQLValidationError):
            sq.insert_sqllite_data("t", [{"a": 1}], ["bad col;--"], cur, conn)

    def test_verbose_logs_the_row_count_and_table(self, file_db, caplog):
        path, conn, cur = file_db
        cur.execute("CREATE TABLE t (a INTEGER)")
        caplog.set_level(logging.INFO)
        assert sq.insert_sqllite_data("t", [{"a": 1}, {"a": 2}], ["a"], cur, conn, verbose=1) == 2
        assert "Inserted 2 row(s) into t table." in caplog.text

    def test_quiet_mode_logs_nothing_on_success(self, file_db, caplog):
        path, conn, cur = file_db
        cur.execute("CREATE TABLE t (a INTEGER)")
        caplog.set_level(logging.INFO)
        assert sq.insert_sqllite_data("t", [{"a": 1}], ["a"], cur, conn, verbose=0) == 1
        assert "Inserted" not in caplog.text

    def test_a_successful_insert_is_committed_for_another_connection(self, file_db):
        path, conn, cur = file_db
        cur.execute("CREATE TABLE t (a INTEGER)")
        conn.commit()
        sq.insert_sqllite_data("t", [{"a": 7}], ["a"], cur, conn)
        other = sqlite3.connect(path)
        try:
            assert list(other.execute("SELECT a FROM t")) == [(7,)]
        finally:
            other.close()

    def test_a_failing_insert_logs_the_table_and_the_error(self, file_db, caplog):
        path, conn, cur = file_db
        cur.execute("CREATE TABLE t (a INTEGER NOT NULL)")
        caplog.set_level(logging.ERROR)
        assert sq.insert_sqllite_data("t", [{"a": None}], ["a"], cur, conn) == 0
        assert "Could not insert data into t table:" in caplog.text

    def test_a_failing_insert_logs_the_last_ten_values(self, file_db, caplog):
        path, conn, cur = file_db
        cur.execute("CREATE TABLE t (a INTEGER NOT NULL)")
        caplog.set_level(logging.ERROR)
        rows = [{"a": i} for i in range(12)] + [{"a": None}]
        assert sq.insert_sqllite_data("t", rows, ["a"], cur, conn) == 0
        last_ten = [(i,) for i in range(3, 12)] + [(None,)]
        assert f"Data sample: {last_ten}" in caplog.text

    def test_a_failing_insert_rolls_back_the_earlier_rows(self, file_db):
        path, conn, cur = file_db
        cur.execute("CREATE TABLE t (a INTEGER NOT NULL)")
        conn.commit()
        rows = [{"a": 1}, {"a": 2}, {"a": None}]
        sq.insert_sqllite_data("t", rows, ["a"], cur, conn, verbose=0)
        assert list(cur.execute("SELECT a FROM t")) == []
