# ----------------------------------------------------------------------------------------------------------------------------
# SQLLITE
# ----------------------------------------------------------------------------------------------------------------------------
# These helpers take conn/cursor as explicit arguments and do NOT read the
# module-level Postgres connection globals; carved out of db.py and
# re-exported by the db.__init__ facade.
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

from typing import Any, Dict, Iterable

from os.path import join, exists

from pyutilz.database.db.sql_helpers import validate_sql_identifier


def ensure_db_tables_created(conn: object, cursor: object, schema_fpath: str = None) -> bool:

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


def insert_sqllite_data(table_name: str, data: Iterable[Dict[str, Any]], columns: Iterable, cursor: object, conn: object, verbose: int = 1):
    """Самый быстрый способ для массовых вставок"""

    # Validate table/column names to prevent SQL injection
    validate_sql_identifier(table_name)
    for col in columns:
        validate_sql_identifier(col)

    # Создаем SQL запрос
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join([f'"{col}"' if col == "GROUP" else col for col in columns])
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
            return n
    except Exception as e:
        logger.error(f"Could not insert data into {table_name} table: {e}.")
        logger.error(f"Data sample: {data[-10:]}")
        return 0
