"""SQLite helpers that take conn/cursor as explicit arguments.

These do NOT read the module-level Postgres connection globals; carved out of db.py and
re-exported by the db.__init__ facade.
"""
# ----------------------------------------------------------------------------------------------------------------------------
# SQLITE
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

from typing import Any, Dict, Iterable, Optional
import sqlite3

from os.path import join, exists

from pyutilz.database.db.sql_helpers import validate_sql_identifier


def ensure_db_tables_created(conn: sqlite3.Connection, cursor: sqlite3.Cursor, schema_fpath: Optional[str] = None) -> bool:
    """Create DB tables by executing the SQL script at ``schema_fpath`` (default ``database/schema.sql``).

    Returns True on success, False if the schema file is missing or empty (logging an error in either case).
    """

    if not schema_fpath:
        schema_fpath = join("database", "schema.sql")

    if not exists(schema_fpath):
        logger.error("DB Schema file not found.")
        return False

    with open(schema_fpath, encoding="utf-8") as f:
        schema_string = f.read()

    if len(schema_string) > 0:
        cursor.executescript(schema_string)
        conn.commit()

        return True
    else:
        logger.error("DB Schema empty.")
        return False


def insert_sqllite_data(table_name: str, data: Iterable[Dict[str, Any]], columns: Iterable, cursor: sqlite3.Cursor, conn: sqlite3.Connection, verbose: int = 1):
    """Самый быстрый способ для массовых вставок"""

    # Regression fix: `columns` is iterated 3 separate times below (validate, build
    # placeholders, build columns_str); a genuine one-shot iterator/generator (which the
    # `Iterable` type hint explicitly invites) was exhausted after the first pass, silently
    # producing an empty `INSERT INTO t () VALUES ()` -- caught by the broad `except Exception`
    # below and reported as "0 rows inserted", indistinguishable from a real failure.
    columns = list(columns)

    # Validate table/column names to prevent SQL injection
    validate_sql_identifier(table_name)
    for col in columns:
        validate_sql_identifier(col)

    # Создаем SQL запрос
    placeholders = ", ".join(["?" for _ in columns])
    # Regression fix: only the exact literal string "GROUP" was quoted -- SQLite's reserved-word
    # matching is case-insensitive (a column literally named "group" was NOT caught) and there
    # are many other reserved words besides GROUP (ORDER, TABLE, SELECT, WHERE, INDEX, etc.).
    # Quoting is always valid in SQLite regardless of whether the name happens to be reserved,
    # so quote every identifier unconditionally instead of special-casing one keyword.
    columns_str = ", ".join(f'"{col}"' for col in columns)
    sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"  # nosec B608 - table_name/columns validated above

    # Преобразуем словари в кортежи в правильном порядке
    values_list = []
    for row in data:
        values = tuple(row.get(col) for col in columns)
        values_list.append(values)

    # Вставляем данные
    try:
        cursor.executemany(sql, values_list)
        conn.commit()
        n = len(values_list)
        if verbose:
            logger.info("Inserted %s row(s) into %s table.", n, table_name)
        # Regression fix: `return n` was previously nested INSIDE `if verbose:`, so a fully
        # successful insert with verbose=0 (the function's own documented way to silence
        # per-call logging) fell off the end of the try block and implicitly returned None --
        # a caller doing `total += insert_sqllite_data(..., verbose=0)` crashed with TypeError
        # on a completely successful insert.
        return n
    except Exception as e:
        logger.exception("Could not insert data into %s table: %s.", table_name, e)
        logger.error("Data sample: %s", values_list[-10:])
        # Regression fix: cursor.executemany() executes each parameter set sequentially: if row
        # k violates a constraint, rows 1..k-1 already executed and remained pending in an open,
        # never-rolled-back transaction on this connection -- indistinguishable from "nothing was
        # inserted" (the return value this except branch reports) until an unrelated later
        # conn.commit() on the same connection silently committed them for real.
        conn.rollback()
        return 0
