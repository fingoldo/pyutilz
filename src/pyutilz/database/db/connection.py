"""Connection / engine setup and cursor management for ``pyutilz.database.db``.

Holds ``connect_to_db`` plus the per-thread cursor cache and the cursor accessors built on it. The
mutable connection state itself (``conn``/``cur``/``conn_alchemy``/credentials) lives on the
``pyutilz.database.db`` facade so every submodule reads and writes ONE namespace.
"""

from typing import Any, Dict, Optional, cast

from ._common import (
    logger,
    psycopg2,
    threading,
    ISOLATION_LEVEL_AUTOCOMMIT,
    compiles,
    Insert,
    OnConflictDoNothing,
)

# PROJECT IDIOM for a re-export package's submodules (see the comment block in
# pyutilz/data/pandaslib/frames.py):
#   import <parent> as _facade    -- ALLOWED, and load-bearing.
#   from <parent> import <name>   -- FORBIDDEN at module top level.
# The facade imports this submodule, so importing the facade back is a genuine cycle: plain
# ``import x`` binds the PARTIALLY-INITIALISED sys.modules entry and defers every attribute lookup
# to call time, which both survives the cycle and keeps the connection globals LIVE (a from-import
# would snapshot ``conn``/``cur`` at import time, so a caller's -- or a test's -- later rebinding of
# ``pyutilz.database.db.conn`` would never be seen here).
import pyutilz.database.db as _facade


@compiles(Insert, "postgresql")
def prefix_inserts(insert, compiler, **kw):
    """SQLAlchemy compilation hook making every PostgreSQL INSERT default to ``ON CONFLICT DO NOTHING``.

    Only applies when the statement carries no post-VALUES clause of its own, so an explicit
    ``on_conflict_*`` written by the caller is left alone.
    """

    if insert._post_values_clause is None:
        insert._post_values_clause = OnConflictDoNothing()
    return compiler.visit_insert(insert, **kw)


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
    # The connection state lives on the facade (see the _facade comment above), so these are
    # facade attribute writes rather than the pre-split ``global`` statement.
    _facade.db_flavor = db_flavor = m_db_flavor
    _facade.db_name = db_name = m_db_name
    _facade.db_host = db_host = m_db_host
    _facade.db_port = db_port = m_db_port
    _facade.db_schema = db_schema = m_db_schema
    _facade.username = username = m_db_username
    _facade.pwd = pwd = m_db_pwd
    _facade.init_params_fn = init_params_fn = m_init_params_fn
    _facade.db_sslmode = db_sslmode = m_db_sslmode

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

                conn = _facade.psycopg2.connect(**params)
                _facade.conn = conn
                conn_opened_this_iteration = True
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

                conn_string = "postgresql+psycopg2://%s:%s@%s:%s/%s?" % (username, pwd, db_host, db_port, db_name)  # pragma: allowlist secret

                if db_sslmode:
                    conn_string += f"&sslmode={db_sslmode}"
                if db_schema:
                    conn_string += f"&options=--search_path%3D{db_schema},public"

                _facade.conn_alchemy = _facade.sqlalchemy.create_engine(conn_string, paramstyle="format")
                # https://stackoverflow.com/questions/25917741/sqlalchemy-with-postgres-insert-into-a-table-whose-columns-have-parentheses

                # Only clears THIS (the connecting) thread's own cursor cache -- any other thread
                # already holding cached cursors against a stale/prior connection keeps them
                # until it next touches get_cursor() and (if the old connection is truly dead)
                # gets a real error from psycopg2 rather than a silent cross-thread mix-up.
                thread_cursors = _get_thread_cursors()
                thread_cursors.clear()

                cur = conn.cursor()
                _facade.cur = cur

                # if db_schema: cur.execute(f"set search_path to '{db_schema}'")

                thread_cursors["cursor"] = cur
            elif db_flavor == "mysql":
                #!pip install pymysql
                import pymysql  # noqa: F401 -- the import IS the availability check for the engine URL built below

                _facade.conn_alchemy = _facade.sqlalchemy.create_engine(f"mysql+pymysql://{username}:{pwd}@{db_host}:{db_port}/{db_name}")

            if init_params_fn is not None:
                init_params_fn()
        except Exception as e:
            logger.exception(e)
            if conn_opened_this_iteration and _facade.conn is not None:
                try:
                    _facade.conn.close()
                except Exception as close_exc:
                    logger.exception(close_exc)
                # Leaving the global bound to a CLOSED connection makes get_cursor()'s
                # "call connect_to_db() first" assert pass and fail later with an opaque
                # InterfaceError instead.
                _facade.conn = None
            if max_retries is not None and attempt >= max_retries:
                logger.error("connect_to_db: giving up after %d attempts", attempt)
                raise
            _facade.sleep(5)
        else:
            logger.info("Connected to the DB %s", db_name)
            return


def get_cursor_type(cursor_factory: Any, cursor_name: Optional[str] = None) -> str:
    """Returns the cache key identifying a cursor flavour: the cursor factory's class name, suffixed
    with ``_named`` when a server-side cursor name is requested (named cursors are never cached)."""
    if cursor_factory is None:
        cursor_factory = psycopg2.extensions.cursor
    cursor_type = str(cursor_factory.__name__) + ("" if cursor_name is None else "_named")  # str(): cursor_factory is Any, so its __name__ is too

    return cursor_type


def get_cursor(cursor_type: str, cursor_factory: Any = None, cursor_name: Optional[str] = None, itersize: Optional[int] = None) -> Any:
    """Returns a cursor of the requested type, reusing the calling thread's cached one when possible.

    Unnamed cursors are cached per thread and per ``cursor_type``; a named (server-side) cursor is
    created fresh every time, ``WITH HOLD``, and honours ``itersize`` as its fetch batch size.
    """
    thread_cursors = _get_thread_cursors()
    if cursor_type in thread_cursors and "_named" not in cursor_type:
        cur = thread_cursors[cursor_type]
    else:
        assert _facade.conn is not None, "get_cursor() requires connect_to_db() to have been called first"
        cur = _facade.conn.cursor(cursor_factory=cursor_factory, name=cursor_name, withhold=(False if cursor_name is None else True))
        if itersize:
            if str(itersize).isnumeric():
                cur.itersize = itersize
        if "_named" not in cursor_type:
            thread_cursors[cursor_type] = cur
    return cur


def mogrify(operation, parameters=None):
    """Returns the statement as the driver would actually send it, with ``parameters`` bound and
    escaped, decoded to ``str`` -- for logging and for building a statement out of a template."""
    cursor_type = _facade.get_cursor_type(None, None)
    cur = _facade.get_cursor(cursor_type)
    return cur.mogrify(operation, parameters).decode("utf-8")
