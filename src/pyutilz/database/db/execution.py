"""Query execution for ``pyutilz.database.db``.

The cursor-based execution core (``basic_db_execute`` and its ``safe_execute*`` wrappers), the
``db_command`` DML convenience layer, the settings/logging helpers built on them, and the
SQLAlchemy-engine raw-SQL escape hatches (``select``/``execute_alchemy``/``showcase_table``/
``explain_table``).
"""

from typing import Optional

from ._common import (
    logger,
    json,
    pd,
    re,
    datetime,
    timezone,
    execute_values,
    DuplicateTable,
    OperationalError,
    InternalError,
    InterfaceError,
    sqlalchemy,
    PAGE_SIZE,
)
from .sql_helpers import construct_templates_and_values, validate_sql_identifier

# See the PROJECT IDIOM comment in connection.py: attribute access on the facade is what keeps the
# connection globals live and every cross-function call (safe_execute, get_cursor, ...) patchable.
import pyutilz.database.db as _facade


def get_table_fields(table, alias, prefix="", suffix="", excluding=""):
    """Returns a comma-separated ``<alias>.<column> <prefix><column><suffix>`` select list for a table.

    Column names are read from the driver's cursor description of an empty (``where 0=1``) probe query,
    so the list always follows the live schema. ``excluding`` accepts a comma-separated string or an
    iterable of column names to leave out.
    """
    if isinstance(excluding, str):
        excluding = excluding.split(",")
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table)
    # Must use the CALLING thread's own cursor: the module-global `cur` can be rebound by another
    # thread between execute() and fetchall(), silently returning another table's description.
    local_cur = _facade.get_cursor(_facade.get_cursor_type(None, None))
    local_cur.execute("select * from " + table + " where 0=1")  # nosec B608 - table validated by validate_sql_identifier above
    local_cur.fetchall()
    if local_cur.description is not None:
        return ",".join([alias + "." + col.name + " " + prefix + col.name + suffix for col in local_cur.description if col.name not in excluding])


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
    """Runs one statement (``ex_type`` selects ``execute`` or psycopg2's batched ``execute_values``) and
    returns its rows, or the cursor itself when ``return_cursor`` is set.

    Reconnects and retries up to ``max_retries`` times on a dropped connection or a named-cursor name
    collision; a statement with no result set yields an empty list. Server-side cursors this function
    does not hand back to the caller are closed here so they cannot leak for the connection's lifetime.
    """
    if not auto_commit:
        # The connection runs in ISOLATION_LEVEL_AUTOCOMMIT, so every statement commits on its own;
        # a caller batching statements under auto_commit=False has no transaction and no rollback.
        logger.warning("basic_db_execute: auto_commit=False is not honoured (connection is in autocommit mode); each statement commits independently")

    cursor_type = _facade.get_cursor_type(cursor_factory, cursor_name)
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
            local_cur = _facade.get_cursor(cursor_type=cursor_type, cursor_factory=cursor_factory, cursor_name=cursor_name, itersize=itersize)

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
            _facade.sleep(1)
            _facade.connect_to_db(
                m_db_name=_facade.db_name,
                m_db_host=_facade.db_host,
                m_db_port=_facade.db_port,
                m_db_username=_facade.username,
                m_db_pwd=_facade.pwd,
                m_init_params_fn=_facade.init_params_fn,
                m_db_flavor=_facade.db_flavor if _facade.db_flavor is not None else "postgres",
                m_db_schema=_facade.db_schema,
                m_db_sslmode=_facade.db_sslmode,
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
                _facade._get_thread_cursors().pop(cursor_type, None)
                _facade.sleep(1)
                continue
            else:
                if cursor_name is not None and local_cur is not None:
                    try:
                        local_cur.close()
                    except Exception as close_exc:
                        logger.exception(close_exc)
                raise
        else:
            _facade.cur = local_cur
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
    """Runs a single statement through ``basic_db_execute`` in plain ``execute`` mode, with its
    reconnect-and-retry handling; returns the rows, or the cursor when ``return_cursor`` is set."""
    return _facade.basic_db_execute("execute", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize)


def safe_execute_values(
    statement, data, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize: Optional[int] = None, page_size: int = PAGE_SIZE
):
    """Runs a statement over many rows at once via psycopg2's ``execute_values``, sending them in
    batches of ``page_size``; otherwise behaves exactly like ``safe_execute``."""
    return _facade.basic_db_execute(
        "execute_values", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize, page_size=page_size
    )


def fetch_db_elements(self, elements, fields, indices=None, prefix=""):
    """Copies fetched rows onto ``self`` as attributes named ``<prefix><field>``.

    ``fields`` is a comma-separated string or a list of names; ``"*"`` resolves them from the last
    cursor's description. ``indices`` maps each field to its position within a row, defaulting to the
    field order. Every row in ``elements`` is applied in turn, so the last one wins.
    """
    if isinstance(fields, str):
        fields = fields.split(",")
    if elements is not None:
        if fields == ["*"]:
            # Per-thread cursor, for the same cross-thread-rebinding reason as get_table_fields().
            local_cur = _facade.get_cursor(_facade.get_cursor_type(None, None))
            assert local_cur is not None, "fetch_db_elements() requires connect_to_db() to have been called first"
            fields = [col.name for col in local_cur.description]
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
    # "insert" consults only set_fields; without this guard set_fields=None crashes two frames down
    # inside construct_templates_and_values, and set_fields=[] builds `insert into t () values ()`.
    if mode == "insert" and not set_fields:
        logger.error("mode='insert' requires a non-empty set_fields (got %r)", set_fields)
        return
    # `returning` is spliced into the SQL as a string; None is the natural way to say "no RETURNING".
    if returning is None:
        returning = ""
    if mode == "select" and not returning:
        returning = "*"

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

    res = _facade.safe_execute(sql, sql_fields_values)
    if fetch_into is not None:
        if prefix == "":
            if table_name.endswith("s"):
                prefix = table_name[:-1] + "_"
        _facade.fetch_db_elements(fetch_into, res, returning, prefix=prefix)
    return res


def read_db_settings(g, interval_minutes=10, settings_names_contains=None):
    """Loads the ``settings`` table into the namespace dict ``g``, keyed by setting name.

    Each value is cast according to the row's declared type (int/float/str/json/bool), leaving it
    untouched when it fails to parse. Re-reads at most once per ``interval_minutes``; passing
    ``settings_names_contains`` restricts the load to settings whose name contains that substring.
    """
    do_update = False
    if _facade.last_db_settings_read_at is None:
        do_update = True
    else:
        now_time = datetime.now(timezone.utc)
        if (now_time - _facade.last_db_settings_read_at).total_seconds() / 60 >= interval_minutes:
            do_update = True
    if do_update:
        sql = "select name,value,type from settings"
        sql_params = None
        if settings_names_contains:
            # Use parameterized query to prevent SQL injection
            sql += " where strpos(name,%s)>0"
            sql_params = (settings_names_contains,)

        for setting_name, val, typename in _facade.safe_execute(sql, sql_params):
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
        _facade.last_db_settings_read_at = datetime.now(timezone.utc)


_LOG_TO_DB_KNOWN_LEVELS = frozenset({"info", "warning", "warn", "error", "critical", "fatal"})

# Accepts only quoted JSON keys and array indices chained with -> / ->> , e.g. 'a'->'b'->>2
_JSON_PATH_RE = re.compile(r"^(?:'[A-Za-z0-9_ .-]+'|[0-9]+)(?:->>?(?:'[A-Za-z0-9_ .-]+'|[0-9]+))*$")


def log_to_db(message, details=None, more_details=None, level="info", append_severity=False, application=None, table_name="logs"):
    """Emits a message to the Python logger AND inserts it as a row into the ``logs`` table.

    ``details``/``more_details`` are appended to the logged text in brackets and stored in their own
    columns. ``level`` picks the logger method and the numeric severity, degrading to ``info`` (with a
    warning) when unrecognized; ``append_severity`` also prefixes the stored message with the level.
    The originating node and application default to the values found on the facade or in the caller's
    stack when not supplied.
    """
    cInfo = 1
    cWarning = 2
    cError = 3

    # A falsy level (None, "") used to skip the entire body: no Python log record, no DB row, no
    # warning -- a wrapper forwarding an absent config key silently dropped every message.
    if not level:
        level = "info"

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

        node = _facade.node_id
        if node is None or application is None:
            if node is None:
                node = _facade.lookup_in_stack("node_id")
            if application is None:
                application = _facade.lookup_in_stack("app_name")

        # Validate table name to prevent SQL injection
        validate_sql_identifier(table_name)
        _facade.safe_execute(
            "insert into " + table_name + "(node,message,details,more_details,severity,application) values (%s,%s,%s,%s,%s,%s)",  # nosec B608 - table_name validated above
            (node, message, details, more_details, severity, application),
        )


def explain_table(table_name: str) -> Optional[object]:
    """Read table names along with comments from a DB table, return as Pandas dataframe"""
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)
    if _facade.db_flavor == "mysql":
        return pd.read_sql(f"SHOW FULL COLUMNS FROM {table_name}", con=_facade.conn_alchemy)["Field Type Comment".split()]  # type: ignore[no-any-return]  # untyped upstream source (pandas read_sql); return value verified correct at runtime
    # Returning a bare None here produced an unattributable TypeError at the call site; name the flavor.
    logger.warning("explain_table is only implemented for db_flavor='mysql' (current flavor: %r); returning None", _facade.db_flavor)
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
    return pd.read_sql(f"SELECT * FROM {table_name} {condition} LIMIT {int(limit)}", con=_facade.conn_alchemy)  # nosec B608 - table_name validated above; condition is an accepted raw WHERE fragment


def select(sql: str) -> object:
    """Execute arbitrary SQL against DB table, return results as Pandas dataframe.

    WARNING: `sql` is executed verbatim, with zero validation or parameterization.
    This function is an intentional raw-SQL escape hatch - `sql` must NEVER be built
    from external/user-controlled input directly; only pass trusted, hard-coded or
    internally-constructed statements.
    """
    return pd.read_sql(sql, con=_facade.conn_alchemy)


def execute_alchemy(sql: str, max_retries: int = 3) -> None:
    """Execute arbitrary SQL against DB table using Alchemy directly.

    WARNING: `sql` is executed verbatim, with zero validation or parameterization.
    This function is an intentional raw-SQL escape hatch - `sql` must NEVER be built
    from external/user-controlled input directly; only pass trusted, hard-coded or
    internally-constructed statements.
    """
    conn_alchemy = _facade.conn_alchemy
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
            _facade.sleep(3)
    # Regression fix: previously fell off the end of the function with no return/raise once
    # `n < max_retries` became false -- a permanently-broken statement (SQL syntax error,
    # permission denied) failed identically on every attempt (retried as if transient) and then
    # returned None silently, with no way for the caller to know the statement never executed.
    logger.error("execute_alchemy: giving up after %d attempts", max_retries)
    if last_exc is not None:
        raise last_exc
