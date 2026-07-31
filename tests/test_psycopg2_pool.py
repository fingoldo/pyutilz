"""Tests for pyutilz.database.psycopg2_pool. All psycopg2 I/O is mocked -- no real DB needed."""

from __future__ import annotations

import threading
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from pyutilz.database import psycopg2_pool as pool_mod
from pyutilz.database.psycopg2_pool import (
    close_pool,
    get_connection,
    get_connection_from_pool,
    managed_connection,
    release_connection,
)


@pytest.fixture(autouse=True)
def _clean_pool_state():
    close_pool()
    yield
    close_pool()


def test_get_connection_retries_on_operational_error():
    """get_connection retries on OperationalError and succeeds on second attempt."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = lambda s, *a: None

    call_count = 0

    def getconn_side_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise psycopg2.OperationalError("connection refused")
        return mock_conn

    with ExitStack() as stack:
        MockPool = stack.enter_context(patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool"))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.register_vector", None))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.time.sleep"))

        pool_instance = MagicMock()
        pool_instance.getconn.side_effect = getconn_side_effect
        MockPool.return_value = pool_instance

        conn = get_connection("postgresql://test:test@localhost/test", max_retries=3)  # pragma: allowlist secret

    assert conn == mock_conn


def test_get_connection_raises_after_max_retries():
    with ExitStack() as stack:
        MockPool = stack.enter_context(patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool"))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.time.sleep"))

        pool_instance = MagicMock()
        pool_instance.getconn.side_effect = psycopg2.OperationalError("down")
        MockPool.return_value = pool_instance

        with pytest.raises(psycopg2.OperationalError):
            get_connection("postgresql://test:test@localhost/test", max_retries=2)  # pragma: allowlist secret


def test_get_connection_rejects_zero_max_retries():
    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        get_connection("postgresql://test:test@localhost/test", max_retries=0)  # pragma: allowlist secret


def test_get_connection_from_pool_stale_recovery():
    """A stale connection is reset and retried."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.getconn.return_value = mock_conn

    call_count = [0]

    def fake_ensure(conn):
        call_count[0] += 1
        if call_count[0] == 1:
            raise psycopg2.OperationalError("stale")

    with ExitStack() as stack:
        stack.enter_context(patch.object(pool_mod, "_pool", mock_pool))
        stack.enter_context(patch.object(pool_mod, "_pool_dsn", "test"))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool._ensure_connection", side_effect=fake_ensure))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.register_vector", None))

        conn = get_connection_from_pool("test")
        assert conn is mock_conn
        mock_conn.reset.assert_called_once()


def test_get_connection_from_pool_no_infinite_recursion():
    """Stale-connection recovery does not infinitely recurse."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    mock_conn.reset.side_effect = RuntimeError("reset failed")

    with ExitStack() as stack:
        stack.enter_context(patch.object(pool_mod, "_pool", mock_pool))
        stack.enter_context(patch.object(pool_mod, "_pool_dsn", "test"))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool._ensure_connection", side_effect=psycopg2.OperationalError("stale")))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.register_vector", None))

        with pytest.raises(RuntimeError, match="reset failed"):
            get_connection_from_pool("test", _recovering=True)


def test_connection_leak_on_register_vector_failure():
    """A connection is returned to the pool if register_vector raises -- not leaked."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = lambda s, *a: None

    with ExitStack() as stack:
        MockPool = stack.enter_context(patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool"))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.register_vector", side_effect=RuntimeError("vector ext broken")))

        pool_instance = MagicMock()
        pool_instance.getconn.return_value = mock_conn
        MockPool.return_value = pool_instance

        with pytest.raises(RuntimeError, match="vector ext broken"):
            get_connection("postgresql://test:test@localhost/test")  # pragma: allowlist secret

        pool_instance.putconn.assert_called_once_with(mock_conn, close=True)


def test_managed_connection_releases_on_exception():
    """managed_connection releases the connection even when the body raises."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = lambda s, *a: None

    with ExitStack() as stack:
        MockPool = stack.enter_context(patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool"))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.register_vector", None))

        pool_instance = MagicMock()
        pool_instance.getconn.return_value = mock_conn
        MockPool.return_value = pool_instance

        with pytest.raises(RuntimeError, match="boom"):
            with managed_connection("postgresql://test:test@localhost/test"):  # pragma: allowlist secret
                raise RuntimeError("boom")

        pool_instance.putconn.assert_called_once_with(mock_conn)
        mock_conn.rollback.assert_called_once()


def test_threaded_pool_concurrent_access():
    """ThreadedConnectionPool handles concurrent getconn/putconn without races."""
    mock_conns = [MagicMock() for _ in range(4)]
    for mc in mock_conns:
        mc.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mc.cursor.return_value.__exit__ = MagicMock(return_value=False)

    conn_idx = [0]
    pool_lock = threading.Lock()

    def fake_getconn():
        with pool_lock:
            idx = conn_idx[0] % len(mock_conns)
            conn_idx[0] += 1
            return mock_conns[idx]

    with ExitStack() as stack:
        MockPool = stack.enter_context(patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool"))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.register_vector", None))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool._ensure_connection"))

        pool_instance = MagicMock()
        pool_instance.getconn.side_effect = fake_getconn
        MockPool.return_value = pool_instance

        results = []
        errors = []

        def worker():
            try:
                conn = get_connection("postgresql://test:test@localhost/test")  # pragma: allowlist secret
                results.append(conn)
                release_connection(conn)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive(), f"Thread {t.name} did not finish within timeout"

        assert not errors, f"Errors in concurrent access: {errors}"
        assert len(results) == 4


def test_ensure_pool_rejects_different_dsn_after_torn_down_pool():
    """``_pool_dsn`` can outlive ``_pool`` (e.g. mid-teardown race); a new DSN must still be rejected."""
    with patch.object(pool_mod, "_pool", None), patch.object(pool_mod, "_pool_dsn", "dsn-a"):
        with pytest.raises(ValueError, match="already created with a different DSN"):
            pool_mod._ensure_pool("dsn-b")


def test_close_pool_clears_module_state():
    with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool") as MockPool:
        MockPool.return_value = MagicMock()
        pool_mod._ensure_pool("dsn-a")
        assert pool_mod.get_pool_dsn() == "dsn-a"
        close_pool()
        assert pool_mod.get_pool_dsn() is None
