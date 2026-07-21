# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
import threading
import warnings

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed

# ensure_installed("sqlalchemy psycopg2 pandas pymysql")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Dict, Iterable, Optional, Union, cast
import json
import pandas as pd

from time import sleep
from enum import Enum

from os.path import join, exists

from pyutilz.core.pythonlib import weekofmonth, datetime_to_utc_timestamp, lookup_in_stack

import sqlalchemy
from datetime import datetime, date, timezone

from dateutil.relativedelta import relativedelta

import psycopg2
from psycopg2 import sql
from psycopg2.errors import DuplicateTable
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import execute_values, NamedTupleCursor, DictCursor
from psycopg2 import OperationalError, InternalError, InterfaceError

# psycopg2.InterfaceError: cursor already closed
# psycopg2.InternalError: current transaction is aborted, commands ignored until end of transaction block
# ----------------------------------------------------------------------------------------------------------------------------
# SQL Injection Protection
# ----------------------------------------------------------------------------------------------------------------------------

import re

# ----------------------------------------------------------------------------------------------------------------------------
# Re-exported PURE / STATELESS helpers carved into cohesive submodules.
# These do NOT read the module-level connection globals (conn/cur/cursors),
# so they live in submodules; the connection-stateful functions below stay
# in this parent module so they all share the SAME global state.
# ``_SQL_IDENTIFIER_RE`` / ``validate_sql_identifier`` are imported here so
# every ``from pyutilz.database.db import X`` and ``pyutilz.db`` alias usage
# keeps resolving exactly as before.
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.database.db.sql_helpers import (
    _SQL_IDENTIFIER_RE,
    validate_sql_identifier,
    validate_sql_qualified_identifier,
    construct_templates_and_values,
    u,
    nu,
    make_set_excluded_clause,
    MakeSetExcludedClause,
    update_if_now,
)
from pyutilz.database.db.upsert import build_upsert_query
from pyutilz.database.db.sqlite import ensure_db_tables_created, insert_sqllite_data

# ----------------------------------------------------------------------------------------------------------------------------
# Global variables
# ----------------------------------------------------------------------------------------------------------------------------
last_db_settings_read_at = None
conn_alchemy = None
PAGE_SIZE: int = 1_000_000
# Bound for real by connect_to_db(); declared here so module-level references resolve
# before that first call (and so static analysis can see these names exist at all).
db_flavor: Optional[str] = None
conn: Optional[Any] = None
cur: Optional[Any] = None
db_name: Optional[str] = None
db_host: Optional[str] = None
db_port: Optional[int] = None
db_schema: Optional[str] = None
username: Optional[str] = None
pwd: Optional[str] = None
init_params_fn: Optional[Any] = None
db_sslmode: Optional[str] = None

# Per-thread cursor cache (regression fix, 2026-07-21 audit round 2): psycopg2 cursors are NOT
# safe to share across threads -- only the underlying connection (`conn`, still module-global) is.
# The cache used to be one shared module-global dict, so two threads calling e.g. safe_execute()
# concurrently could interleave onto the SAME cursor object: thread B's execute() could land
# between thread A's own execute() and fetchall(), so thread A silently received thread B's query
# results. Each thread now gets its own independent cache, keyed the same way (by cursor_type).
_thread_local = threading.local()


def _get_thread_cursors() -> Dict[str, Any]:
    """Return the CALLING thread's own cursor cache (see module-level comment above)."""
    if not hasattr(_thread_local, "cursors"):
        _thread_local.cursors = {}
    return cast(Dict[str, Any], _thread_local.cursors)


# Optional override for log_to_db()'s "node" column -- callers may set
# pyutilz.database.db.node_id = "..." directly; falls back to lookup_in_stack()
# when unset (the pre-existing default, unchanged).
node_id: Optional[str] = None
# ----------------------------------------------------------------------------------------------------------------------------
# sqlalchemy tricks
# ----------------------------------------------------------------------------------------------------------------------------
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Insert
from sqlalchemy.dialects.postgresql.dml import OnConflictDoNothing


@compiles(Insert, "postgresql")
def prefix_inserts(insert, compiler, **kw):

    if insert._post_values_clause is None:
        insert._post_values_clause = OnConflictDoNothing()
    return compiler.visit_insert(insert, **kw)


# ----------------------------------------------------------------------------------------------------------------------------


def get_table_fields(table, alias, prefix="", suffix="", excluding=""):
    if isinstance(excluding, str):
        excluding = excluding.split(",")
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table)
    assert cur is not None, "get_table_fields() requires connect_to_db() to have been called first"
    cur.execute("select * from " + table + " where 0=1")  # nosec B608 - table validated by validate_sql_identifier above
    cur.fetchall()
    if cur.description is not None:
        return ",".join([alias + "." + col.name + " " + prefix + col.name + suffix for col in cur.description if col.name not in excluding])


def connect_to_db(
    m_db_name,
    m_db_host,
    m_db_port,
    m_db_username,
    m_db_pwd,
    m_init_params_fn=None,
    m_db_flavor: str = "postgres",
    m_db_schema: Optional[str] = None,
    m_db_sslmode: Optional[str] = None,
    max_retries: Optional[int] = None,
):
    """Connect to a Postgres or MySQL database, retrying (5s backoff) on failure.

    ``max_retries=None`` (default) preserves the original retry-forever behavior for
    long-running services that should keep waiting out a DB outage. Pass an int to bound it
    (e.g. for a one-shot script/CLI where "wrong host/credentials" should fail fast instead of
    looping forever) -- once exhausted, the last connection exception is re-raised rather than
    silently retrying past a permanent (not transient) failure like bad credentials.

    ``m_db_flavor="mysql"`` is PARTIAL support: it only builds ``conn_alchemy`` (SQLAlchemy
    engine), so only the SQLAlchemy-based helpers (``select()``, ``execute_alchemy()``,
    ``explain_table()``) work against MySQL. It never assigns the module-level ``conn``/``cur``
    globals, so the entire cursor-based API this module otherwise exposes (``get_cursor``,
    ``basic_db_execute``, ``safe_execute``, ``get_table_fields``, ``db_command``, ``log_to_db``,
    ``EnsurePgTableExists``, ``ReadTableIntoDic*``, ``GetIdByKeyFieldAndInsertIfNeeded``,
    ``create/delete_postgres_range_partitions``, etc.) raises ``AssertionError`` on first use
    against a MySQL connection, even though ``connect_to_db()`` completed without error. Building
    a real ``pymysql`` conn/cur pair mirroring the Postgres branch is a real feature gap, not
    fixed here (pymysql's cursor semantics -- named/withhold server-side cursors in particular --
    don't map 1:1 onto psycopg2's, and this repo has no MySQL instance available to verify
    against); this docstring exists so the gap is discoverable up front instead of via a
    confusing assert deep in an unrelated helper.
    """
    global db_flavor, conn, cur, conn_alchemy
    global db_name, db_host, db_port, db_schema, username, pwd, init_params_fn, db_sslmode
    db_flavor = m_db_flavor
    db_name = m_db_name
    db_host = m_db_host
    db_port = m_db_port
    db_schema = m_db_schema
    username = m_db_username
    pwd = m_db_pwd
    init_params_fn = m_init_params_fn
    db_sslmode = m_db_sslmode

    assert db_flavor in ("postgres", "mysql")  # nosec B101 - db_flavor only steers if/elif branching below, never spliced into SQL

    attempt = 0
    while True:
        attempt += 1
        conn_opened_this_iteration = False
        try:
            logger.info("Connecting to the DB %s...", db_name)

            if db_flavor == "postgres":
                #!pip install psycopg2

                params = dict(dbname=db_name, host=db_host, port=db_port, user=username, password=pwd)
                if db_sslmode:
                    params["sslmode"] = db_sslmode
                if db_schema:
                    params["options"] = f"--search_path={db_schema},public"

                conn = psycopg2.connect(**params)
                conn_opened_this_iteration = True
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

                conn_string = "postgresql+psycopg2://%s:%s@%s:%s/%s?" % (username, pwd, db_host, db_port, db_name)

                if db_sslmode:
                    conn_string += f"&sslmode={db_sslmode}"
                if db_schema:
                    conn_string += f"&options=--search_path%3D{db_schema},public"

                conn_alchemy = sqlalchemy.create_engine(conn_string, paramstyle="format")
                # https://stackoverflow.com/questions/25917741/sqlalchemy-with-postgres-insert-into-a-table-whose-columns-have-parentheses

                # Only clears THIS (the connecting) thread's own cursor cache -- any other thread
                # already holding cached cursors against a stale/prior connection keeps them
                # until it next touches get_cursor() and (if the old connection is truly dead)
                # gets a real error from psycopg2 rather than a silent cross-thread mix-up.
                thread_cursors = _get_thread_cursors()
                thread_cursors.clear()

                cur = conn.cursor()

                # if db_schema: cur.execute(f"set search_path to '{db_schema}'")

                thread_cursors["cursor"] = cur
            elif db_flavor == "mysql":
                #!pip install pymysql
                import pymysql

                conn_alchemy = sqlalchemy.create_engine(f"mysql+pymysql://{username}:{pwd}@{db_host}:{db_port}/{db_name}")

            if init_params_fn is not None:
                init_params_fn()
        except Exception as e:
            logger.exception(e)
            if conn_opened_this_iteration and conn is not None:
                try:
                    conn.close()
                except Exception as close_exc:
                    logger.exception(close_exc)
            if max_retries is not None and attempt >= max_retries:
                logger.error("connect_to_db: giving up after %d attempts", attempt)
                raise
            sleep(5)
        else:
            logger.info("Connected to the DB %s", db_name)
            return


def get_cursor_type(cursor_factory: Any, cursor_name: Optional[str] = None) -> str:
    if cursor_factory is None:
        cursor_factory = psycopg2.extensions.cursor
    cursor_type = cursor_factory.__name__ + ("" if cursor_name is None else "_named")

    return cursor_type  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def get_cursor(cursor_type: str, cursor_factory: Any = None, cursor_name: Optional[str] = None, itersize: Optional[int] = None) -> Any:
    thread_cursors = _get_thread_cursors()
    if cursor_type in thread_cursors and "_named" not in cursor_type:
        cur = thread_cursors[cursor_type]
    else:
        assert conn is not None, "get_cursor() requires connect_to_db() to have been called first"
        cur = conn.cursor(cursor_factory=cursor_factory, name=cursor_name, withhold=(False if cursor_name is None else True))
        if itersize:
            if str(itersize).isnumeric():
                cur.itersize = itersize
        if "_named" not in cursor_type:
            thread_cursors[cursor_type] = cur
    return cur


def mogrify(operation, parameters=None):
    cursor_type = get_cursor_type(None, None)
    cur = get_cursor(cursor_type)
    return cur.mogrify(operation, parameters).decode("utf-8")


def basic_db_execute(
    ex_type,
    statement,
    data=None,
    auto_commit=True,
    cursor_factory=None,
    cursor_name=None,
    return_cursor=False,
    itersize: Optional[int] = None,
    page_size: int = PAGE_SIZE,
    max_retries: int = 5,
):
    global cur

    cursor_type = get_cursor_type(cursor_factory, cursor_name)
    stmt_preview = str(statement)[:500]

    # Add circuit breaker to prevent infinite retry loops
    retry_count = 0
    while retry_count < max_retries:
        # `local_cur` (a genuine Python local, NOT the module-global `cur`) is what the
        # execute()/fetchall() critical section below operates on -- regression fix (2026-07-21
        # audit round 2): with `global cur` covering this whole function, a concurrent thread's
        # own `cur = get_cursor(...)` reassignment could swap the module-global `cur` out from
        # under this thread BETWEEN its own execute() and fetchall() calls, silently handing this
        # thread another thread's cursor/result buffer. `cur` is still updated below (after the
        # critical section completes) purely as a best-effort "last cursor used" introspection
        # convenience for external callers/tests, never read back for this function's own logic.
        local_cur = None
        try:
            local_cur = get_cursor(cursor_type=cursor_type, cursor_factory=cursor_factory, cursor_name=cursor_name, itersize=itersize)

            if ex_type == "execute":
                local_cur.execute(statement, data)
            elif ex_type == "execute_values":
                execute_values(local_cur, statement, data, page_size=page_size)

            # if '_named' not in cursor_type:
            # if auto_commit: conn.commit()

        except (OperationalError, InterfaceError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Max retries (%s) exceeded for database operation (statement=%s)", max_retries, stmt_preview)
                raise
            logger.exception("Database operation failed (statement=%s): %s", stmt_preview, e)
            logger.info("Retrying database operation (%s/%s)...", retry_count, max_retries)
            sleep(1)
            connect_to_db(
                m_db_name=db_name,
                m_db_host=db_host,
                m_db_port=db_port,
                m_db_username=username,
                m_db_pwd=pwd,
                m_init_params_fn=init_params_fn,
                m_db_flavor=db_flavor if db_flavor is not None else "postgres",
                m_db_schema=db_schema,
                m_db_sslmode=db_sslmode,
            )
            continue  # Retry the operation
        except DuplicateTable as e:
            logger.warning("DuplicateTable (statement=%s): %s", stmt_preview, e)
            # conn.commit()
            return
        except InternalError as e:
            logger.exception("InternalError (statement=%s): %s", stmt_preview, e)
            # logger.warning("rolling back operation...")
            # conn.rollback()
            # InternalError indicates real corruption/state issues; do not silently
            # fall through to an implicit None (indistinguishable from "no rows") or
            # spin forever in the retry loop - propagate it to the caller.
            if cursor_name is not None and local_cur is not None:
                try:
                    local_cur.close()
                except Exception as close_exc:
                    logger.exception(close_exc)
            raise
        except Exception as e:
            logger.exception("Database operation failed (statement=%s): %s", stmt_preview, e)
            if "cursor" in str(e) and "already exists" in str(e):
                # Regression fix: this branch previously never touched retry_count, so the
                # circuit breaker documented at the top of this function ("prevent infinite
                # retry loops") never applied here -- a named/server-side cursor collision that
                # keeps recurring (e.g. left open by a prior failed transaction) looped forever.
                # Also, `cur = get_cursor(...)` above raised BEFORE completing its assignment, so
                # the module-level `cur` still held whatever the PREVIOUS call left it as -- an
                # unrelated cursor, not the one that actually collided -- so `cur.close()` was
                # closing the wrong object. Only the stale cache entry is cleared here now;
                # nothing is closed that this function can't positively identify.
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error("Max retries (%s) exceeded for database operation (cursor collision, statement=%s)", max_retries, stmt_preview)
                    raise
                _get_thread_cursors().pop(cursor_type, None)
                sleep(1)
                continue
            else:
                if cursor_name is not None and local_cur is not None:
                    try:
                        local_cur.close()
                    except Exception as close_exc:
                        logger.exception(close_exc)
                raise
        else:
            cur = local_cur
            # A named/server-side cursor (withhold=True) corresponds to a real Postgres
            # `DECLARE ... CURSOR ... WITH HOLD`, kept alive across commits until explicitly
            # closed. When this function isn't handing the cursor back to the caller
            # (return_cursor=False), IT is the only thing that can still close it -- otherwise
            # the server-side cursor leaks for the lifetime of the connection.
            if local_cur.description is not None:
                # cols_names = [desc[0] for desc in cur.description]
                if return_cursor:
                    return local_cur
                else:
                    rows = local_cur.fetchall()
                    if cursor_name is not None:
                        local_cur.close()
                    return rows
            else:
                if return_cursor:
                    return local_cur
                else:
                    if cursor_name is not None:
                        local_cur.close()
                    return []


def safe_execute(statement, data=None, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize: Optional[int] = None):
    return basic_db_execute("execute", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize)


def safe_execute_values(
    statement, data, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize: Optional[int] = None, page_size: int = PAGE_SIZE
):
    return basic_db_execute("execute_values", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize, page_size=page_size)


def fetch_db_elements(self, elements, fields, indices=None, prefix=""):
    if isinstance(fields, str):
        fields = fields.split(",")
    if elements is not None:
        if fields == ["*"]:
            assert cur is not None, "fetch_db_elements() requires connect_to_db() to have been called first"
            fields = [col.name for col in cur.description]
        if indices is None:
            indices = range(len(fields))
        for element in elements:
            for i, field in enumerate(fields):
                setattr(self, prefix + field, element[indices[i]])


def db_command(mode, table_name, where_fields=None, set_fields=None, replace_values=None, returning="*", source=None, jsonize=True, fetch_into=None, prefix=""):
    """
    Executes DML commands easily, looking up sql fields in main and replacement dictionaries, optionally fetching returned values into separate class or dictionary instance[prefixed, if needed]
    This procedure is convenient when you have variables in local or global scope which names match exactly SQL table fields names. Using this sub, in your code you only mention fields names once, pass locals()
    and do not worry about duplicated code! Also it allows fetching returned data into some object in one go. ;-)
    """
    if replace_values is None:
        replace_values = {}
    if source is None:
        source = {}

    # ----------------------------------------------------------------------------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------------------------------------------------------------------------

    if where_fields is None and set_fields is None:
        logger.error("Either where_fields or set_fields have to be defined")
        return
    if mode not in ["select", "insert", "update"]:
        logger.error("Unknown mode: %s", mode)
        return
    # Regression fix: "select"/"update" only ever consult where_fields to build the WHERE
    # clause, but the check above only required ONE of where_fields/set_fields to be non-None --
    # where_fields=None with set_fields=[...] passed silently, then crashed inside
    # construct_templates_and_values() with an opaque `TypeError: 'NoneType' object is not
    # iterable`. Separately, where_fields=[] (an empty list, not None) also passed the check
    # silently, producing a WHERE clause with no condition ("... where ") that Postgres itself
    # rejects with a syntax error at execution time, rather than failing fast with a clear
    # validation message here.
    if mode in ("select", "update") and not where_fields:
        logger.error("mode=%r requires a non-empty where_fields (got %r)", mode, where_fields)
        return

    # ----------------------------------------------------------------------------------------------------------------------------
    # Smart params parsing
    # ----------------------------------------------------------------------------------------------------------------------------

    if isinstance(where_fields, str):
        where_fields = where_fields.split(",")
    if isinstance(set_fields, str):
        set_fields = set_fields.split(",")

    if mode in ["select", "update"]:
        where_values, sql_where_templates = construct_templates_and_values(mode, where_fields, replace_values, source, jsonize)
    if mode in ["insert", "update"]:
        set_values, sql_set_templates = construct_templates_and_values(mode, set_fields, replace_values, source, jsonize)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Carrying out exact sql text
    # ----------------------------------------------------------------------------------------------------------------------------

    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)

    if mode == "select":
        # table_name validated above; where_fields/set_fields validated per-item in construct_templates_and_values;
        # `returning` is an accepted raw fragment (may be "*" or a comma list), not an identifier
        sql = "select " + returning + " from  " + table_name + " where " + " and ".join(sql_where_templates)  # nosec B608
        sql_fields_values = where_values
    elif mode == "insert":
        sql = "insert into " + table_name + " (" + ",".join(set_fields) + ") values (" + ",".join(sql_set_templates) + ")"  # nosec B608
        sql_fields_values = set_values
    elif mode == "update":
        # Fixed: SET clause must use comma separator, not "and"
        sql = "update  " + table_name + " set " + ", ".join(sql_set_templates) + " where " + " and ".join(sql_where_templates)  # nosec B608
        sql_fields_values = set_values + where_values

    if mode in ["insert", "update"]:
        if len(returning) > 0:
            sql += " returning " + returning

    res = safe_execute(sql, sql_fields_values)
    if fetch_into is not None:
        if prefix == "":
            if table_name.endswith("s"):
                prefix = table_name[:-1] + "_"
        fetch_db_elements(fetch_into, res, returning, prefix=prefix)
    return res


def read_db_settings(g, interval_minutes=10, settings_names_contains=None):
    global last_db_settings_read_at
    do_update = False
    if last_db_settings_read_at is None:
        do_update = True
    else:
        now_time = datetime.now(timezone.utc)
        if (now_time - last_db_settings_read_at).total_seconds() / 60 >= interval_minutes:
            do_update = True
    if do_update:
        sql = "select name,value,type from settings"
        sql_params = None
        if settings_names_contains:
            # Use parameterized query to prevent SQL injection
            sql += " where strpos(name,%s)>0"
            sql_params = (settings_names_contains,)

        for setting_name, val, typename in safe_execute(sql, sql_params):
            if typename is None:
                typename = "string"
            ltypename = typename.lower()
            if val is not None:
                if ltypename == "int":
                    val = int(val)
                elif ltypename in ["float", "real", "double", "numeric"]:
                    val = float(val)
                elif ltypename in ["str", "string"]:
                    val = str(val)
                elif ltypename in ["json", "jsonb"]:
                    raw_val = val if isinstance(val, (str, bytes, bytearray)) else str(val)
                    try:
                        # orjson is faster than stdlib json; falls back if missing (kept optional -- core has no
                        # hard requirements).
                        import orjson  # type: ignore

                        try:
                            val = orjson.loads(raw_val)
                        except orjson.JSONDecodeError:
                            logger.warning("Setting %r has json/jsonb type but value is not valid JSON: %r", setting_name, val)
                    except ImportError:
                        try:
                            val = json.loads(raw_val)
                        except json.JSONDecodeError:
                            # Leave non-JSON values untouched rather than crashing settings load.
                            logger.warning("Setting %r has json/jsonb type but value is not valid JSON: %r", setting_name, val)
                elif ltypename in ["bool", "boolean"]:
                    val = val.lower() in ["true", "1", "t", "y", "yes"]
            g[setting_name] = val
        last_db_settings_read_at = datetime.now(timezone.utc)


_LOG_TO_DB_KNOWN_LEVELS = frozenset({"info", "warning", "warn", "error", "critical", "fatal"})


def log_to_db(message, details=None, more_details=None, level="info", append_severity=False, application=None, table_name="logs"):
    import inspect

    cInfo = 1
    cWarning = 2
    cError = 3

    if level:
        if level not in _LOG_TO_DB_KNOWN_LEVELS:
            # Regression fix: an unrecognized level (typo, case mismatch, a caller-invented
            # string like "critical" before it was added to the map below) used to fall through
            # to the `else` branch SILENTLY -- logged and persisted as plain "info" severity with
            # no signal the requested level wasn't honored. Now at least a warning marks the
            # mismatch, even though the message itself still degrades to info (log_to_db must not
            # raise over a bad level string and abort the caller's actual operation).
            logger.warning("log_to_db: unrecognized level %r, treating as 'info' (known levels: %s)", level, sorted(_LOG_TO_DB_KNOWN_LEVELS))
        s = message
        if details:
            if more_details:
                s = s + " [" + str(details) + "/" + str(more_details) + "]"
            else:
                s = s + " [" + str(details) + "]"
        else:
            if more_details:
                s = s + " [" + str(more_details) + "]"

        if level in ["warning", "warn"]:
            logger.warning(s)
            severity = cWarning
            if append_severity:
                message = "[Warning] " + message
        elif level in ("error", "critical", "fatal"):
            logger.error(s)
            severity = cError
            if append_severity:
                message = "[ERROR] " + message
        else:
            logger.info(s)
            severity = cInfo

        node = node_id
        if node is None or application is None:
            if node is None:
                node = lookup_in_stack("node_id")
            if application is None:
                application = lookup_in_stack("app_name")

        # Validate table name to prevent SQL injection
        validate_sql_identifier(table_name)
        safe_execute(
            "insert into " + table_name + "(node,message,details,more_details,severity,application) values (%s,%s,%s,%s,%s,%s)",  # nosec B608 - table_name validated above
            (node, message, details, more_details, severity, application),
        )


def check_if_pg_table_exists(table_name: str, schema_name: Optional[str] = "public"):
    # table_name/schema_name are compared against information_schema metadata columns as VALUES,
    # not identifiers -- parameterized via %s placeholders (correctness/consistency fix: this
    # previously used u()'s manual quote-doubling escape instead of driver-level parameter
    # binding, unlike log_to_db()'s equivalent query a few lines above in the same file, which
    # already used %s correctly for the same class of comparison).
    res = safe_execute(
        """
    SELECT EXISTS (
       SELECT FROM information_schema.tables
       WHERE  table_schema = %s AND table_name = %s
   )""",
        (schema_name, table_name),
    )
    if res:
        return res[0][0]


def ensure_pg_table_exists(
    table: str, key_field_name: Optional[str] = "name", id_field_name: Optional[str] = "id", autocreate_id_type_name: Optional[str] = None
) -> None:
    """Create ``table`` (with ``key_field_name``/``id_field_name`` columns) if it doesn't exist yet.

    Also importable as :func:`EnsurePgTableExists` -- a deprecated alias, same function, kept for
    backward compatibility with the legacy PascalCase/Hungarian-notation name.
    """
    # Validate identifiers to prevent SQL injection
    validate_sql_identifier(table)
    validate_sql_identifier(key_field_name)
    validate_sql_identifier(id_field_name)
    if not check_if_pg_table_exists(table):
        if autocreate_id_type_name:
            if autocreate_id_type_name.lower() not in ("smallserial serial bigserial uuid".split()):
                # autocreate_id_type_name is spliced verbatim into the CREATE TABLE statement below with no other
                # validation, so under `python -O` a skipped assert would let arbitrary SQL be injected via this arg.
                raise ValueError(f"Invalid autocreate_id_type_name: {autocreate_id_type_name!r}")
            # Regression fix: the validation above (2 lines up) correctly lowercases before
            # comparing against the whitelist, but this comparison didn't -- autocreate_id_type_name="UUID"
            # passed validation ("UUID".lower() == "uuid" is in the whitelist) but silently failed
            # THIS check ("UUID" == "uuid" is False), so `default_gen` stayed empty and the
            # generated DDL had no `default gen_random_uuid()` despite auto-generation having
            # been explicitly requested -- any insert omitting the id column (the whole point of
            # autocreate_id_type_name) would then fail with a NOT NULL constraint violation.
            default_gen = " default gen_random_uuid()" if autocreate_id_type_name.lower() == "uuid" else ""
            safe_execute(
                f"create table {table} ({id_field_name} {autocreate_id_type_name} primary key {default_gen},{key_field_name} text, added_at timestamp without time zone DEFAULT (now() at time zone 'utc'))"
            )


def EnsurePgTableExists(sTable: str, sKeyFieldName: Optional[str] = "name", sIdFieldName: Optional[str] = "id", sAutocreateIdTypeName: Optional[str] = None):
    """Deprecated alias for :func:`ensure_pg_table_exists` -- kept for backward compatibility."""
    warnings.warn(
        "EnsurePgTableExists is deprecated and will be removed in a future release; use ensure_pg_table_exists instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ensure_pg_table_exists(table=sTable, key_field_name=sKeyFieldName, id_field_name=sIdFieldName, autocreate_id_type_name=sAutocreateIdTypeName)


def read_table_into_dict(
    dict_enums: dict,
    table: str,
    key_field_name: Optional[str] = "name",
    condition: Optional[str] = "",
    id_field_name: Optional[str] = "id",
    autocreate_id_type_name: Optional[str] = None,
) -> None:
    """
    Reads id->value mapping into a dictionary
    if autocreate_id_type_name is specified, if table does not exist, it gets created with specified key type

    Also importable as :func:`ReadTableIntoDic` -- a deprecated alias, same function, kept for
    backward compatibility with the legacy PascalCase/Hungarian-notation name.
    """

    dict_enums.clear()
    ensure_pg_table_exists(table=table, key_field_name=key_field_name, id_field_name=id_field_name, autocreate_id_type_name=autocreate_id_type_name)
    # table/key_field_name/id_field_name validated inside ensure_pg_table_exists above; condition is an accepted raw WHERE fragment
    res = safe_execute(f"select {id_field_name},{key_field_name} from {table} {condition}")  # nosec B608
    for rs in res:
        if rs[1] is not None:
            if rs[0] is not None:
                the_id = rs[0]
                key = rs[1]
                dict_enums[key] = the_id


def ReadTableIntoDic(
    dicEnums: dict,
    sTable: str,
    sKeyFieldName: Optional[str] = "name",
    sCondition: Optional[str] = "",
    sIdFieldName: Optional[str] = "id",
    sAutocreateIdTypeName: Optional[str] = None,
) -> None:
    """Deprecated alias for :func:`read_table_into_dict` -- kept for backward compatibility."""
    warnings.warn(
        "ReadTableIntoDic is deprecated and will be removed in a future release; use read_table_into_dict instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_table_into_dict(
        dict_enums=dicEnums, table=sTable, key_field_name=sKeyFieldName, condition=sCondition, id_field_name=sIdFieldName, autocreate_id_type_name=sAutocreateIdTypeName
    )


def read_table_into_dict_reversed(
    dict_enums: dict,
    table: str,
    key_field_name: Optional[str] = "name",
    condition: Optional[str] = "",
    id_field_name: Optional[str] = "id",
    autocreate_id_type_name: Optional[str] = None,
) -> None:
    """Reads value->id mapping into a dictionary.

    WARNING: condition is spliced verbatim into the SQL statement (raw WHERE fragment,
    unvalidated). This function executes raw, unvalidated SQL - condition must NEVER
    be built from external/user-controlled input directly; only pass trusted, hard-coded
    or internally-constructed condition strings.

    Also importable as :func:`ReadTableIntoDicReversed` -- a deprecated alias, same function,
    kept for backward compatibility with the legacy PascalCase/Hungarian-notation name.
    """

    dict_enums.clear()
    ensure_pg_table_exists(table=table, key_field_name=key_field_name, id_field_name=id_field_name, autocreate_id_type_name=autocreate_id_type_name)
    # table/key_field_name/id_field_name validated inside ensure_pg_table_exists above; condition is an accepted raw WHERE fragment
    res = safe_execute(f"select {id_field_name},{key_field_name} from {table} {condition}")  # nosec B608
    for rs in res:
        if rs[1] is not None:
            if rs[0] is not None:
                the_id = rs[0]
                key = rs[1]
                dict_enums[the_id] = key


def ReadTableIntoDicReversed(
    dicEnums: dict,
    sTable: str,
    sKeyFieldName: Optional[str] = "name",
    sCondition: Optional[str] = "",
    sIdFieldName: Optional[str] = "id",
    sAutocreateIdTypeName: Optional[str] = None,
) -> None:
    """Deprecated alias for :func:`read_table_into_dict_reversed` -- kept for backward compatibility."""
    warnings.warn(
        "ReadTableIntoDicReversed is deprecated and will be removed in a future release; use read_table_into_dict_reversed instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_table_into_dict_reversed(
        dict_enums=dicEnums, table=sTable, key_field_name=sKeyFieldName, condition=sCondition, id_field_name=sIdFieldName, autocreate_id_type_name=sAutocreateIdTypeName
    )


def read_unique_table_field(table_name: str, field_name: str, container: Union[set, dict], clear: bool = True, placeholder_value=None) -> Union[set, dict]:
    if clear:
        container.clear()
    res = safe_execute(
        sql.SQL("select distinct {field_name} from {table_name}").format(field_name=sql.Identifier(field_name), table_name=sql.Identifier(table_name))
    )
    if isinstance(container, set):
        for rec in res:
            container.add(rec[0])
    elif isinstance(container, dict):
        for rec in res:
            container[rec[0]] = placeholder_value
    return container


def get_id_by_key_field_and_insert_if_needed(
    dict_enums: dict,
    table: str,
    key_field_value: str,
    key_field_name: Optional[str] = "name",
    key_is_not_string: Optional[bool] = False,
    alternate_fields_names: Optional[str] = "",
    alternate_fields_values: Optional[str] = "",
    unique_constraint_fields: Optional[str] = "",
    use_alternate_fields_only: Optional[bool] = False,
    id_field_name: Optional[str] = "id",
    add_updated_at_timestamp: Optional[str] = None,
) -> str:
    """Look up ``key_field_value``'s id in ``dict_enums`` (or the DB), inserting a new row if needed.

    Also importable as :func:`GetIdByKeyFieldAndInsertIfNeeded` -- a deprecated alias, same
    function, kept for backward compatibility with the legacy PascalCase/Hungarian-notation name.
    """

    if key_field_value == "null":
        return "null"

    if key_field_value in dict_enums:
        return dict_enums[key_field_value]  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
    else:
        key_field_name = key_field_name if key_field_name is not None else "name"
        alternate_fields_names = alternate_fields_names if alternate_fields_names is not None else ""
        unique_constraint_fields = unique_constraint_fields if unique_constraint_fields is not None else ""
        id_field_name = id_field_name if id_field_name is not None else "id"

        # Validate identifiers to prevent SQL injection
        validate_sql_identifier(table)
        validate_sql_identifier(key_field_name)
        validate_sql_identifier(id_field_name)
        for _name in [n for n in alternate_fields_names.split(",") if n]:
            validate_sql_identifier(_name)
        if unique_constraint_fields == "":
            unique_constraint_fields = key_field_name
        for _name in [n for n in unique_constraint_fields.split(",") if n]:
            validate_sql_identifier(_name)

        if key_is_not_string:
            Data = key_field_value
        else:
            Data = u(key_field_value)
        # All identifiers below (table, key_field_name, id_field_name, alternate_fields_names, unique_constraint_fields) are validated above
        if len(alternate_fields_names) > 0:
            if not use_alternate_fields_only:
                rs = safe_execute(
                    f"insert into {table} ({key_field_name} , {alternate_fields_names}) values ({Data},{alternate_fields_values}) on conflict ({unique_constraint_fields}) do update set {make_set_excluded_clause(key_field_name, add_updated_at_timestamp)} returning {id_field_name}"  # nosec B608
                )
            else:
                rs = safe_execute(
                    f"insert into {table} ({alternate_fields_names}) values ({alternate_fields_values}) on conflict ({unique_constraint_fields}) do update set {make_set_excluded_clause(alternate_fields_names, add_updated_at_timestamp)} returning {id_field_name}"  # nosec B608
                )
        else:
            rs = safe_execute(
                f"insert into {table} ({key_field_name}) values ({Data}) on conflict ({unique_constraint_fields}) do update set {make_set_excluded_clause(key_field_name, add_updated_at_timestamp)} returning {id_field_name}"  # nosec B608
            )

        the_id = rs[0][0]
        dict_enums[key_field_value] = the_id

        return the_id  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def GetIdByKeyFieldAndInsertIfNeeded(
    dicEnums: dict,
    sTable: str,
    sKeyFieldValue: str,
    sKeyFieldName: Optional[str] = "name",
    bKeyIsNotString: Optional[bool] = False,
    sAlternateFieldsNames: Optional[str] = "",
    sAlternateFieldsValues: Optional[str] = "",
    sUniqueConstraintFields: Optional[str] = "",
    bUseAlternateFieldsOnly: Optional[bool] = False,
    sIdFieldName: Optional[str] = "id",
    bAddUpdatedAtTimestamp: Optional[str] = None,
) -> str:
    """Deprecated alias for :func:`get_id_by_key_field_and_insert_if_needed` -- kept for backward compatibility."""
    warnings.warn(
        "GetIdByKeyFieldAndInsertIfNeeded is deprecated and will be removed in a future release; use get_id_by_key_field_and_insert_if_needed instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_id_by_key_field_and_insert_if_needed(
        dict_enums=dicEnums,
        table=sTable,
        key_field_value=sKeyFieldValue,
        key_field_name=sKeyFieldName,
        key_is_not_string=bKeyIsNotString,
        alternate_fields_names=sAlternateFieldsNames,
        alternate_fields_values=sAlternateFieldsValues,
        unique_constraint_fields=sUniqueConstraintFields,
        use_alternate_fields_only=bUseAlternateFieldsOnly,
        id_field_name=sIdFieldName,
        add_updated_at_timestamp=bAddUpdatedAtTimestamp,
    )


def create_postgres_range_partitions(table_name: str, from_date: date, to_date: date, partition_size: str, bigint_degree: int = 0):

    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)
    assert partition_size in ("day", "week", "month", "year")  # nosec B101 - partition_size only selects a relativedelta branch below, never spliced into SQL
    d = from_date
    while d <= to_date:
        if partition_size == "day":
            n = d + relativedelta(days=1)
        elif partition_size == "week":
            n = d + relativedelta(weeks=1)
        elif partition_size == "month":
            n = d + relativedelta(months=1)
        elif partition_size == "year":
            n = d + relativedelta(years=1)
        if bigint_degree is None or bigint_degree == 0:
            cmd = f"CREATE TABLE z_{table_name}_y{d.year:04d}m{d.month:02d}w{weekofmonth(d):02d}d{d.day:02d} PARTITION OF {table_name} FOR VALUES FROM ('{d:%Y-%m-%d %H:%M:%S}') TO ('{n:%Y-%m-%d %H:%M:%S}')"
        else:
            cmd = f"CREATE TABLE z_{table_name}_y{d.year:04d}m{d.month:02d}w{weekofmonth(d):02d}d{d.day:02d} PARTITION OF {table_name} FOR VALUES FROM ('{datetime_to_utc_timestamp(d)*int(10**bigint_degree)}') TO ('{datetime_to_utc_timestamp(n)*int(10**bigint_degree)}')"
        # print(cmd)
        safe_execute(cmd)
        d = n


def delete_postgres_range_partitions(table_name: str, from_date: date, to_date: date, partition_size: str) -> None:

    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)
    assert partition_size in ("day", "week", "month", "year")  # nosec B101 - partition_size only selects a relativedelta branch below, never spliced into SQL
    d = from_date
    while d <= to_date:
        if partition_size == "day":
            n = d + relativedelta(days=1)
        elif partition_size == "week":
            n = d + relativedelta(weeks=1)
        elif partition_size == "month":
            n = d + relativedelta(months=1)
        elif partition_size == "year":
            n = d + relativedelta(years=1)

        cmd = f"drop table z_{table_name}_y{d.year:04d}m{d.month:02d}w{weekofmonth(d):02d}d{d.day:02d}"
        safe_execute(cmd)
        d = n


def explain_table(table_name: str) -> Optional[object]:
    """Read table names along with comments from a DB table, return as Pandas dataframe"""
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)
    if db_flavor == "mysql":
        return pd.read_sql(f"SHOW FULL COLUMNS FROM {table_name}", con=conn_alchemy)["Field Type Comment".split()]  # type: ignore[no-any-return]  # untyped upstream source (pandas read_sql); return value verified correct at runtime
    return None


def showcase_table(table_name: str, condition: str = "", limit: int = 5) -> object:
    """Read a sample from a DB table, return as Pandas dataframe.

    WARNING: This function executes raw, unvalidated SQL. The 'condition' parameter is
    spliced verbatim into the query and must NEVER be built from external/user-controlled
    input directly - only pass trusted, hard-coded or internally-constructed condition
    strings. For user input, use parameterized queries instead of passing raw SQL conditions.
    """
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)
    # Note: condition parameter should ideally be deprecated in favor of parameterized queries
    # but kept for backward compatibility with warning
    if condition and not condition.strip().lower().startswith("where"):
        condition = "WHERE " + condition
    return pd.read_sql(f"SELECT * FROM {table_name} {condition} LIMIT {int(limit)}", con=conn_alchemy)  # nosec B608 - table_name validated above; condition is an accepted raw WHERE fragment


def select(sql: str) -> object:
    """Execute arbitrary SQL against DB table, return results as Pandas dataframe.

    WARNING: `sql` is executed verbatim, with zero validation or parameterization.
    This function is an intentional raw-SQL escape hatch - `sql` must NEVER be built
    from external/user-controlled input directly; only pass trusted, hard-coded or
    internally-constructed statements.
    """
    return pd.read_sql(sql, con=conn_alchemy)


def execute_alchemy(sql: str, max_retries: int = 3) -> None:
    """Execute arbitrary SQL against DB table using Alchemy directly.

    WARNING: `sql` is executed verbatim, with zero validation or parameterization.
    This function is an intentional raw-SQL escape hatch - `sql` must NEVER be built
    from external/user-controlled input directly; only pass trusted, hard-coded or
    internally-constructed statements.
    """
    assert conn_alchemy is not None, "execute_alchemy() requires conn_alchemy to be configured first"
    n = 0
    last_exc: Optional[Exception] = None
    while n < max_retries:
        try:
            n += 1
            with conn_alchemy.connect() as connection:
                connection.execute(sqlalchemy.text(sql))
                connection.commit()
            return
        except Exception as e:  # noqa: PERF203 -- per-attempt retry loop; the try/except IS the retry mechanism
            logger.exception(e)
            last_exc = e
            sleep(3)
    # Regression fix: previously fell off the end of the function with no return/raise once
    # `n < max_retries` became false -- a permanently-broken statement (SQL syntax error,
    # permission denied) failed identically on every attempt (retried as if transient) and then
    # returned None silently, with no way for the caller to know the statement never executed.
    logger.error("execute_alchemy: giving up after %d attempts", max_retries)
    if last_exc is not None:
        raise last_exc


def enable_tables_sizes_approximation():
    """
    Example usage after the function is created:

    select table_name,nrows
        FROM
            cozmee.get_approximate_tables_sizes(
                array['public.amazon_products','public.amazon_users_reviews','public.amazon_products_reviews']
                )

    """

    safe_execute("""

CREATE or replace FUNCTION get_approximate_tables_sizes(tables_names text[])
RETURNS  TABLE (table_name text,nrows bigint)  AS
$func$

	SELECT relname::text,reltuples::bigint AS estimate FROM pg_class WHERE  oid in (select unnest($1)::regclass)
$func$ LANGUAGE sql;
""")


def create_enum_from_table(enum_name: str, table_name: str, id_field_name: str, desc_field_name: str) -> object:
    """Uses values and ids from a PG database to fill in Python enum.

    LinksTypes=create_enum_from_table("LinksTypes",table_name="links_types",id_field_name="id",desc_field_name="name")
    """
    dct = {}
    for id_field, desc_field in safe_execute(
        sql.SQL("select {id_field},{desc_field} from {table}").format(
            table=sql.Identifier(table_name), id_field=sql.Identifier(id_field_name), desc_field=sql.Identifier(desc_field_name)
        )
    ):
        dct[desc_field.upper()] = id_field
    return Enum(enum_name, dct)


def suggest_json_optimization(table: str, table_field: str, path: str = "", fields: Optional[list] = None, min_occurence_percent: float = 0.5, max_vals: int = 5) -> dict:
    """
    Aim is to remove lengthy fields that mostly holds a default value from the JSON.
    Absence of some JSON field is interpreted as null in PostGres.
    Result will be an optimization of an existing table.
    """
    if fields is None:
        fields = []

    # Validate identifiers to prevent SQL injection
    validate_sql_identifier(table)
    validate_sql_identifier(table_field)

    # Кандидат должен быть лидером по встречаемости. Также должен покрывать как минимум 90% вариантов. Кандидат не должн быть не-None, если поле имеет None в уже существующих значениях:
    # minHoursWeek [(None, 1244198), ('0', 150675)] - OK
    # risingTalent [(None, 97558), ('true', 441321), ('false', 856003)] - FAIL. false мог бы быть хорошим кандидатом, но уже встречается None.

    res = {}

    for field in fields:
        # Validate the JSON field key: it is spliced (quoted) directly into the query below
        validate_sql_identifier(field)

        # Prepare params
        if path == "" or path is None:
            full_path = table_field
        else:
            full_path = table_field + "->" + path

        # Ask DB
        # table/table_field/field validated above (field is validated even though quoted, since it's spliced directly)
        vals = safe_execute(f"""
                    select {full_path}->>'{field}' as val,count(*) as qty from {table} where {full_path} is not null group by {full_path}->>'{field}' order by qty desc
            """)  # nosec B608
        if not vals:  # safe_execute can return None/empty; nothing to analyse for this field
            continue

        # Compute distributiuon %
        has_none = False
        max_occurences = -1
        total_occurences = 0

        for val, occs in vals:
            if val is None:
                has_none = True
            total_occurences += occs
            if occs > max_occurences:
                max_occurences = occs
                best_val = val

        # Print native values, if not too many
        if len(vals) <= max_vals:
            logger.info("%s %s", field, vals)
        else:
            logger.info("%s %s ...", field, vals[:max_vals])

        # Print opt suggestions for that field
        if total_occurences > 0 and max_occurences / total_occurences >= min_occurence_percent:
            if not (has_none and best_val is not None):
                logger.info(
                    "Suggestion: make %s as default for %s. That will save %.2f%% of space.",
                    best_val,
                    field,
                    max_occurences / total_occurences * 100,
                )
                res[field] = best_val

    return res


def regjobs_create_table(table_name: str = "regular_jobs"):
    safe_execute(sql.SQL("""
create table {table_name} (
	name citext primary key,
	description text,
	interval_minutes int,
	singleton boolean default true, --do not start new instance if prev one is still running (unless ping has timed out)
	ping_timeout_minutes int,

	taken_by text, --node ip

	started_at timestamp without time zone,

	last_ping_at timestamp without time zone,
	last_result jsonb,

	finished_at timestamp without time zone,
	result jsonb

	)
    """).format(table_name=sql.Identifier(table_name)))


def regjobs_poll(job_name: str, taken_by: str, table_name: str = "regular_jobs"):
    return safe_execute(
        sql.SQL("""
        with base as (select name from {table_name} where name=%(job_name)s
            and
                (
                    (started_at is null) --never started before
                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        (finished_at>started_at)
                            and
                        singleton=true
                    ) -- not finished long enough after successful finishing for a singleton
                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        (finished_at is null or finished_at<started_at)
                            and
                        (ping_timeout_minutes is not null and (last_ping_at is null or last_ping_at<(now() at time zone 'utc'-ping_timeout_minutes * interval '1 minute')))
                            and
                        singleton=true
                    ) -- not finished long enough WITHOUT successful finishing yet AND with a missed ping, for a singleton


                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        singleton=false
                    ) -- not finished long enough for NOT a singleton

                )
                limit 1 for update skip locked)

            update {table_name} rj set taken_by=%(taken_by)s,started_at=now() at time zone 'utc' from base where rj.name=base.name returning rj.name
    """).format(table_name=sql.Identifier(table_name)),
        {"job_name": job_name, "taken_by": taken_by},
    )


def regjobs_progress(job_name: str, result: dict, table_name: str = "regular_jobs"):
    return safe_execute(
        sql.SQL("""
        update {table_name} set last_ping_at=now() at time zone 'utc',last_result=%(result)s where name=%(job_name)s
    """).format(table_name=sql.Identifier(table_name)),
        {"job_name": job_name, "result": result},
    )


def regjobs_finalize(job_name: str, result: dict, table_name: str = "regular_jobs"):
    return safe_execute(
        sql.SQL("""
        update {table_name} set finished_at=now() at time zone 'utc',result=%(result)s where name=%(job_name)s
    """).format(table_name=sql.Identifier(table_name)),
        {"job_name": job_name, "result": result},
    )
