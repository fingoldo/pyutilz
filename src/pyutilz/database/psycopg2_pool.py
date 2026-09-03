"""Thread-safe psycopg2 connection pool with retry, staleness-tolerant health-checking, and
context-manager helpers.

Complements ``pyutilz.database.db`` (a single global connection, retried on connect) with a real
``psycopg2.pool.ThreadedConnectionPool`` -- ``ThreadedConnectionPool`` is thread-safe internally,
so concurrent access from multiple threads (or ``asyncio.to_thread()`` workers) needs no extra
locking beyond what this module already does around pool creation/teardown.

Ported from a downstream project's own pool module (independently duplicated, in slightly
different shapes, across at least two more downstream projects) -- generalized here by dropping
app-specific bits (a JSON-field parser, an unused client-id-prefix constant) and swapping its
private DSN-masking regex pair for the canonical :mod:`pyutilz.text.secrets_scrub`.

Module-level singleton design (mirrors ``pyutilz.database.db``'s own ``conn``/``cur`` global
pattern): call :func:`get_connection`/:func:`get_connection_from_pool` to borrow a connection,
:func:`release_connection` to return it, or use :func:`managed_connection` as a context manager.
Only ONE pool (one DSN) is supported per process; call :func:`close_pool` before switching DSNs.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None  # type: ignore[assignment]

from pyutilz.text.secrets_scrub import sanitize_dsn

logger = logging.getLogger(__name__)

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pool_dsn: Optional[str] = None
_pool_lock = threading.Lock()  # protects pool creation/teardown only
_conn_last_used: Dict[int, float] = {}  # backend_pid(conn) -> monotonic timestamp
_conn_last_used_lock = threading.Lock()  # guards _conn_last_used reads/writes/sweep
_vector_registered: "weakref.WeakSet[Any]" = weakref.WeakSet()  # auto-expires when conn is GC'd
_vector_registered_lock = threading.Lock()  # guards WeakSet check-and-add
_HEALTH_CHECK_IDLE_SEC = 30.0  # skip SELECT 1 if used within this window
_CONN_USED_MAX_AGE = 600.0  # 10 minutes


def _conn_key(conn: Any) -> int:
    """Return a stable per-backend key for ``conn``.

    Uses ``conn.info.backend_pid`` -- unique per server-side backend process and immune to
    Python-object GC reuse (unlike ``id(conn)``). Falls back to ``id(conn)`` if the attribute is
    unavailable (e.g. connection closed before we could read it), with a warning.
    """
    try:
        pid = int(conn.info.backend_pid)
    except (AttributeError, psycopg2.Error) as e:
        logger.warning("backend_pid unavailable, falling back to id(conn): %s", e)
        return id(conn)
    if not pid:
        # psycopg2 2.9.x returns 0 (rather than raising) for a closed connection; keying every
        # closed connection under 0 would share one last_used timestamp between all of them.
        return id(conn)
    return pid


def _ensure_pool(dsn: str, pool_max: int = 8) -> psycopg2.pool.ThreadedConnectionPool:
    """Lazily create the connection pool (thread-safe)."""
    global _pool, _pool_dsn
    # The DSN comparison MUST happen before returning an existing pool, otherwise a second DSN
    # silently receives connections to the first database.
    if _pool is not None:
        if _pool_dsn != dsn:
            raise ValueError("Connection pool already created with a different DSN. Close the existing pool first.")
        return _pool
    with _pool_lock:
        if _pool is not None:  # double-checked locking
            if _pool_dsn != dsn:  # type: ignore[unreachable]  # mypy narrows the module global to None after the outer check and has no model for another thread setting it in between
                raise ValueError("Connection pool already created with a different DSN. Close the existing pool first.")
            return _pool
        if _pool_dsn is not None and _pool_dsn != dsn:
            raise ValueError("Connection pool already created with a different DSN. Close the existing pool first.")
        _pool = psycopg2.pool.ThreadedConnectionPool(1, pool_max, dsn)
        _pool_dsn = dsn
        return _pool


def _reset_pool() -> None:
    """Close and discard the pool so the next access recreates it."""
    global _pool, _pool_dsn
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
            except psycopg2.Error:
                pass
            _pool = None
            _pool_dsn = None
        with _conn_last_used_lock:
            _conn_last_used.clear()
        with _vector_registered_lock:
            _vector_registered.clear()


def _ensure_connection(conn: Any) -> None:
    """Verify a connection is alive by running a simple query."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    except psycopg2.Error as orig:
        try:
            conn.rollback()
        except Exception:  # nosec B110 - best-effort cleanup; the original error is what matters
            pass
        try:
            conn.reset()
        except Exception as reset_err:
            logger.warning("conn.reset() also failed: %s", reset_err)
        raise orig


def _maybe_register_vector(pool: psycopg2.pool.ThreadedConnectionPool, conn: Any) -> None:
    """Best-effort ``pgvector`` type registration, once per connection (check-and-add under lock
    to prevent two threads both seeing "not registered" and double-registering -- harmless but
    wastes a syscall and spams logs). On failure, returns ``conn`` to the pool (closed) and
    re-raises so the caller doesn't hand out a half-initialized connection."""
    if register_vector is None:
        return
    try:
        with _vector_registered_lock:
            if conn not in _vector_registered:
                register_vector(conn)
                _vector_registered.add(conn)
    except Exception:
        pool.putconn(conn, close=True)
        raise


def get_connection(dsn: str, pool_max: int = 8, max_retries: int = 3) -> Any:
    """Return a psycopg2 connection from a pool (created on first call).

    Retries on ``OperationalError``/``InterfaceError`` with exponential backoff (1s, 2s, 4s, ...).
    Thread-safe: ``ThreadedConnectionPool.getconn()`` handles its own locking.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    for attempt in range(max_retries):
        conn = None
        pool = None
        try:
            pool = _ensure_pool(dsn, pool_max)
            conn = pool.getconn()
            _ensure_connection(conn)
            _maybe_register_vector(pool, conn)
            return conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # If getconn() succeeded but _ensure_connection() raised, the connection was checked
            # out of the pool and must be returned -- otherwise it leaks one slot of pool capacity
            # on every terminal health-check failure. On non-final attempts _reset_pool() below
            # closes everything anyway; putconn here is the correct release and is harmless
            # (close=True discards the bad conn).
            if conn is not None and pool is not None:
                try:
                    pool.putconn(conn, close=True)
                except Exception as put_err:
                    logger.debug("putconn of failed conn raised: %s", put_err)
            if attempt < max_retries - 1:
                delay = 2**attempt
                logger.warning("DB connection failed (attempt %d/%d): %s -- retrying in %ds", attempt + 1, max_retries, sanitize_dsn(str(e)), delay)
                time.sleep(delay)
                _reset_pool()
            else:
                raise
    raise AssertionError("unreachable")  # loop always returns or raises


def get_pool_dsn() -> Optional[str]:
    """Return the DSN used by the connection pool, or None if not initialized."""
    return _pool_dsn


def get_connection_from_pool(dsn: str, pool_max: int = 8, _recovering: bool = False) -> Any:
    """Return a connection from the pool with a lightweight stale-check.

    Use this when the pool is already initialized and you need a connection for a short-lived
    operation. Falls back to :func:`get_connection` if the pool doesn't exist yet. On a stale
    connection, resets once and retries before raising. Thread-safe: ``ThreadedConnectionPool``
    handles its own locking.
    """
    pool = _pool
    if pool is None:
        return get_connection(dsn, pool_max)
    if _pool_dsn != dsn:
        raise ValueError("Connection pool already created with a different DSN. Close the existing pool first.")
    conn = pool.getconn()
    key = _conn_key(conn)
    with _conn_last_used_lock:
        last_used = _conn_last_used.get(key, 0.0)
    if time.monotonic() - last_used > _HEALTH_CHECK_IDLE_SEC:
        try:
            _ensure_connection(conn)
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            logger.warning("Stale connection detected in pool, resetting")
            try:
                conn.reset()
            except Exception:
                pool.putconn(conn, close=True)
                with _conn_last_used_lock:
                    _conn_last_used.pop(key, None)
                if _recovering:
                    raise
                return get_connection_from_pool(dsn, pool_max, _recovering=True)
    with _conn_last_used_lock:
        _conn_last_used[key] = time.monotonic()
    try:
        _maybe_register_vector(pool, conn)
    except Exception:
        with _conn_last_used_lock:
            _conn_last_used.pop(key, None)
        raise
    return conn


def release_connection(conn: Any) -> None:
    """Return a connection to the pool.

    Always rolls back before returning to the pool so an aborted transaction state is not
    inherited by the next borrower (rollback on a clean connection is a harmless no-op).
    """
    key = _conn_key(conn)
    with _conn_last_used_lock:
        _conn_last_used[key] = time.monotonic()
        # Sweep stale entries on every release. Cost is O(pool_size) (typically tiny), and removes
        # the class of bug where a threshold-gated sweep would never fire because the pool is
        # smaller than the threshold. Snapshot (k, original_ts) pairs so that if another thread
        # re-inserts a fresh timestamp between snapshot and delete, we skip it.
        cutoff = time.monotonic() - _CONN_USED_MAX_AGE
        stale = [(k, v) for k, v in _conn_last_used.items() if v < cutoff]
        for k, original_ts in stale:
            if _conn_last_used.get(k) == original_ts:
                del _conn_last_used[k]
    try:
        conn.rollback()
    except Exception as e:
        logger.warning("rollback on release failed: %s", e)
    pool = _pool
    returned = False
    if pool is not None:
        try:
            pool.putconn(conn)
            returned = True
        except psycopg2.pool.PoolError as e:
            # The pool was recreated (or closed) while this connection was checked out, so the
            # current pool object does not know this connection: psycopg2 raises PoolError
            # ("trying to put unkeyed connection" / "connection pool is closed"). Letting that
            # escape would mask whatever exception the caller was already unwinding AND leak the
            # server-side backend; fall through to the orphaned-connection close below instead.
            logger.warning("putconn refused the connection (pool recreated while it was checked out?): %s", e)
    if not returned:
        # The pool was closed (or reset) while this connection was checked out; dropping the
        # reference here would leak the server-side backend until GC or process exit.
        try:
            conn.close()
        except Exception as e:
            logger.warning("closing an orphaned connection (pool already gone) failed: %s", e)


@contextmanager
def managed_connection(dsn: str, pool_max: int = 8) -> Iterator[Any]:
    """Context manager that yields a connection, COMMITS on clean exit, and auto-releases it.

    Pooled connections are not in autocommit mode, so every statement runs inside an implicit
    transaction. Leaving the block normally commits that transaction; leaving it via an exception
    rolls it back (via release_connection) and re-raises. This matches psycopg2's own
    ``with connection:`` semantics -- and
    without it the unconditional rollback in :func:`release_connection` silently discarded every
    write made through this context manager.

    Usage::

        with managed_connection(dsn) as conn:
            conn.cursor().execute("insert into t values (1)")
        # committed here, and the connection is returned to the pool
        # (on an exception the transaction is rolled back and the connection still returned)
    """
    conn = get_connection(dsn, pool_max)
    try:
        yield conn
        # Reached only when the body left the block cleanly (an exception propagates out of the
        # yield instead). The exception path needs no explicit rollback: release_connection()
        # below unconditionally rolls back before returning the connection to the pool, which is
        # exactly the abort semantics wanted here.
        conn.commit()
    finally:
        release_connection(conn)


def close_pool() -> None:
    """Close all connections in the pool and reset module state."""
    _reset_pool()
