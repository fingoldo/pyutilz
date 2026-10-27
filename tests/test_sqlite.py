"""Behaviour of ``pyutilz.database.db.sqlite``: schema creation from a file, the bulk insert over explicit
connection and cursor, identifier validation, and the rollback that keeps a failed batch from leaking into a
later commit. An in-memory SQLite database is used throughout, so no server or file is needed."""

from __future__ import annotations

import sqlite3

import pytest

from pyutilz.database.db.sql_helpers import SQLValidationError
from pyutilz.database.db.sqlite import ensure_db_tables_created, insert_sqllite_data


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    yield connection
    connection.close()


class TestSchemaCreation:
    def test_a_schema_file_creates_its_tables(self, conn, tmp_path):
        schema = tmp_path / "schema.sql"
        schema.write_text("CREATE TABLE items (id INTEGER, name TEXT);", encoding="utf-8")
        cur = conn.cursor()
        assert ensure_db_tables_created(conn, cur, str(schema)) is True
        assert cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='items'").fetchone() is not None

    def test_a_missing_schema_file_returns_false(self, conn, tmp_path):
        assert ensure_db_tables_created(conn, conn.cursor(), str(tmp_path / "absent.sql")) is False

    def test_an_empty_schema_file_returns_false(self, conn, tmp_path):
        schema = tmp_path / "empty.sql"
        schema.write_text("", encoding="utf-8")
        assert ensure_db_tables_created(conn, conn.cursor(), str(schema)) is False


class TestInsert:
    @pytest.fixture
    def items_table(self, conn):
        conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
        return conn

    def test_rows_are_inserted_and_their_count_returned(self, items_table):
        rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        assert insert_sqllite_data("items", rows, ["id", "name"], items_table.cursor(), items_table, verbose=0) == 2
        assert items_table.execute("SELECT id, name FROM items ORDER BY id").fetchall() == [(1, "a"), (2, "b")]

    def test_a_generator_of_columns_is_consumed_once(self, items_table):
        rows = [{"id": 3, "name": "c"}]
        columns = (c for c in ["id", "name"])
        assert insert_sqllite_data("items", rows, columns, items_table.cursor(), items_table, verbose=0) == 1
        assert items_table.execute("SELECT name FROM items").fetchall() == [("c",)]

    def test_a_missing_key_is_stored_as_null(self, items_table):
        assert insert_sqllite_data("items", [{"id": 4}], ["id", "name"], items_table.cursor(), items_table, verbose=0) == 1
        assert items_table.execute("SELECT name FROM items WHERE id=4").fetchone() == (None,)

    def test_an_invalid_table_name_is_refused_before_any_sql_runs(self, items_table):
        with pytest.raises(SQLValidationError):
            insert_sqllite_data("items; DROP TABLE items", [{"id": 1}], ["id"], items_table.cursor(), items_table, verbose=0)
        assert items_table.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("items",)]

    def test_a_failing_batch_inserts_nothing_and_returns_zero(self, conn):
        conn.execute("CREATE TABLE strict (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        rows = [{"id": 1, "name": "ok"}, {"id": 2, "name": None}]
        assert insert_sqllite_data("strict", rows, ["id", "name"], conn.cursor(), conn, verbose=0) == 0
        assert conn.execute("SELECT COUNT(*) FROM strict").fetchone() == (0,)
