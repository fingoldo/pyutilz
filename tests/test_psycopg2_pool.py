"""Tests for pyutilz.database.psycopg2_pool. All psycopg2 I/O is mocked -- no real DB needed."""

from __future__ import annotations

import itertools
import threading
import time
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

    with ExitStack() as stack:
        MockPool = stack.enter_context(patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool"))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.register_vector", None))
        stack.enter_context(patch("pyutilz.database.psycopg2_pool.time.sleep"))

        pool_instance = MagicMock()
        # First getconn fails, every later one hands out the same connection.
        pool_instance.getconn.side_effect = itertools.chain([psycopg2.OperationalError("connection refused")], itertools.repeat(mock_conn))
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


DSN = "postgresql://test:test@localhost/test"  # pragma: allowlist secret


def _conn(pid=7):
    conn = MagicMock(name="conn")
    conn.info.backend_pid = pid
    return conn


class TestReleaseAndStaleChecks:
    def test_a_released_connection_is_rolled_back_and_returned(self):
        pool = MagicMock(name="pool")
        conn = _conn()
        pool_mod._pool = pool
        pool_mod._pool_dsn = "dsn-a"
        release_connection(conn)
        conn.rollback.assert_called_once_with()
        pool.putconn.assert_called_once_with(conn)

    def test_a_failed_rollback_on_release_is_logged(self, caplog):
        pool = MagicMock(name="pool")
        conn = _conn()
        conn.rollback.side_effect = RuntimeError("no rollback")
        pool_mod._pool = pool
        pool_mod._pool_dsn = "dsn-a"
        with caplog.at_level("WARNING", logger=pool_mod.logger.name):
            release_connection(conn)
        assert "rollback on release failed: no rollback" in [r.getMessage() for r in caplog.records]
        pool.putconn.assert_called_once_with(conn)

    def test_a_connection_refused_by_the_pool_is_closed_and_logged(self, caplog):
        pool = MagicMock(name="pool")
        conn = _conn()
        pool.putconn.side_effect = psycopg2.pool.PoolError("unkeyed")
        pool_mod._pool = pool
        pool_mod._pool_dsn = "dsn-a"
        with caplog.at_level("WARNING", logger=pool_mod.logger.name):
            release_connection(conn)
        conn.close.assert_called_once_with()
        assert "putconn refused the connection (pool recreated while it was checked out?): unkeyed" in [r.getMessage() for r in caplog.records]

    def test_a_connection_released_with_no_pool_is_closed(self):
        conn = _conn()
        release_connection(conn)
        conn.close.assert_called_once_with()

    def test_a_failed_close_of_an_orphan_is_logged(self, caplog):
        conn = _conn()
        conn.close.side_effect = RuntimeError("no close")
        with caplog.at_level("WARNING", logger=pool_mod.logger.name):
            release_connection(conn)
        assert "closing an orphaned connection (pool already gone) failed: no close" in [r.getMessage() for r in caplog.records]

    def test_a_release_stamps_the_connection_as_recently_used(self):
        conn = _conn(pid=99)
        release_connection(conn)
        assert 99 in pool_mod._conn_last_used

    def test_a_release_drops_only_entries_older_than_the_window(self):
        now = time.monotonic()
        pool_mod._conn_last_used[1] = now - 700
        pool_mod._conn_last_used[2] = now - 5
        release_connection(_conn(pid=3))
        assert 1 not in pool_mod._conn_last_used
        assert 2 in pool_mod._conn_last_used

    def test_a_connection_idle_for_longer_than_the_window_is_probed_on_borrow(self):
        pool = MagicMock(name="pool")
        conn = _conn(pid=5)
        pool.getconn.return_value = conn
        pool_mod._pool = pool
        pool_mod._pool_dsn = DSN
        pool_mod._conn_last_used[5] = time.monotonic() - 31
        with patch("pyutilz.database.psycopg2_pool._ensure_connection") as ensure, patch("pyutilz.database.psycopg2_pool.register_vector", None):
            get_connection_from_pool(DSN)
        ensure.assert_called_once_with(conn)

    def test_a_connection_used_within_the_window_is_not_probed(self):
        pool = MagicMock(name="pool")
        conn = _conn(pid=5)
        pool.getconn.return_value = conn
        pool_mod._pool = pool
        pool_mod._pool_dsn = DSN
        pool_mod._conn_last_used[5] = time.monotonic() - 10
        with patch("pyutilz.database.psycopg2_pool._ensure_connection") as ensure, patch("pyutilz.database.psycopg2_pool.register_vector", None):
            get_connection_from_pool(DSN)
        ensure.assert_not_called()

    def test_without_a_pool_the_borrow_creates_one_with_the_default_size(self):
        with patch("pyutilz.database.psycopg2_pool.get_connection") as get:
            pool_mod.get_connection_from_pool(DSN)
        get.assert_called_once_with(DSN, 8)

    def test_a_stale_connection_that_cannot_be_reset_is_discarded_and_replaced_once(self):
        pool = MagicMock(name="pool")
        stale = _conn(pid=5)
        stale.reset.side_effect = RuntimeError("no reset")
        fresh = _conn(pid=6)
        pool.getconn.side_effect = [stale, fresh]
        pool_mod._pool = pool
        pool_mod._pool_dsn = DSN
        pool_mod._conn_last_used[5] = 0.0
        with patch("pyutilz.database.psycopg2_pool._ensure_connection", side_effect=psycopg2.OperationalError("stale")), patch(
            "pyutilz.database.psycopg2_pool.register_vector", None
        ):
            conn = get_connection_from_pool(DSN)
        pool.putconn.assert_called_once_with(stale, close=True)
        assert conn is fresh

    def test_a_second_stale_connection_raises_instead_of_looping(self):
        pool = MagicMock(name="pool")
        stale = _conn(pid=5)
        stale.reset.side_effect = RuntimeError("no reset")
        pool.getconn.return_value = stale
        pool_mod._pool = pool
        pool_mod._pool_dsn = DSN
        pool_mod._conn_last_used[5] = 0.0
        with patch("pyutilz.database.psycopg2_pool._ensure_connection", side_effect=psycopg2.OperationalError("stale")), patch(
            "pyutilz.database.psycopg2_pool.register_vector", None
        ), pytest.raises(RuntimeError, match="no reset"):
            get_connection_from_pool(DSN, _recovering=True)

    def test_a_stale_connection_is_logged(self, caplog):
        pool = MagicMock(name="pool")
        stale = _conn(pid=5)
        stale.reset.side_effect = RuntimeError("no reset")
        pool.getconn.side_effect = [stale, _conn(pid=6)]
        pool_mod._pool = pool
        pool_mod._pool_dsn = DSN
        pool_mod._conn_last_used[5] = 0.0
        with caplog.at_level("WARNING", logger=pool_mod.logger.name), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection", side_effect=psycopg2.OperationalError("stale")
        ), patch("pyutilz.database.psycopg2_pool.register_vector", None):
            get_connection_from_pool(DSN)
        assert "Stale connection detected in pool, resetting" in [r.getMessage() for r in caplog.records]


class TestManagedConnection:
    def test_a_clean_exit_commits_then_releases(self):
        conn = _conn()
        with patch("pyutilz.database.psycopg2_pool.get_connection", return_value=conn), patch("pyutilz.database.psycopg2_pool.release_connection") as release:
            with managed_connection(DSN) as borrowed:
                assert borrowed is conn
        conn.commit.assert_called_once_with()
        release.assert_called_once_with(conn)

    def test_an_exception_skips_the_commit_and_still_releases(self):
        conn = _conn()
        with patch("pyutilz.database.psycopg2_pool.get_connection", return_value=conn), patch("pyutilz.database.psycopg2_pool.release_connection") as release:
            with pytest.raises(RuntimeError):
                with managed_connection(DSN):
                    raise RuntimeError("body failed")
        conn.commit.assert_not_called()
        release.assert_called_once_with(conn)


class TestPoolCreationAndReset:
    def test_a_new_pool_is_sized_from_one_to_eight_by_default(self):
        with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool") as MockPool:
            pool_mod._ensure_pool("dsn-a")
        MockPool.assert_called_once_with(1, 8, "dsn-a", keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)

    def test_a_second_dsn_is_refused_with_the_close_hint(self):
        with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool"):
            pool_mod._ensure_pool("dsn-a")
        with pytest.raises(ValueError) as info:
            pool_mod._ensure_pool("dsn-b")
        assert str(info.value) == "Connection pool already created with a different DSN. Close the existing pool first."

    def test_reset_for_an_outdated_pool_is_a_no_op(self):
        current = MagicMock(name="current")
        stale = MagicMock(name="stale")
        pool_mod._pool = current
        pool_mod._pool_dsn = "dsn-a"
        pool_mod._reset_pool(expected=stale)
        current.closeall.assert_not_called()
        assert pool_mod.get_pool_dsn() == "dsn-a"

    def test_reset_is_skipped_while_a_connection_is_checked_out(self):
        pool = MagicMock(name="pool")
        pool._used = {1: "conn"}
        pool_mod._pool = pool
        pool_mod._pool_dsn = "dsn-a"
        pool_mod._reset_pool(expected=pool)
        pool.closeall.assert_not_called()

    def test_reset_closes_an_idle_pool_and_forgets_its_bookkeeping(self):
        pool = MagicMock(name="pool")
        pool._used = None
        pool_mod._pool = pool
        pool_mod._pool_dsn = "dsn-a"
        pool_mod._conn_last_used[1] = 1.0
        pool_mod._vector_registered.add(_conn())
        pool_mod._reset_pool(expected=pool)
        pool.closeall.assert_called_once_with()
        assert pool_mod.get_pool_dsn() is None
        assert pool_mod._conn_last_used == {}
        assert len(pool_mod._vector_registered) == 0

    def test_a_failing_closeall_still_tears_the_pool_down(self):
        pool = MagicMock(name="pool")
        pool.closeall.side_effect = psycopg2.Error("boom")
        pool_mod._pool = pool
        pool_mod._pool_dsn = "dsn-a"
        pool_mod._reset_pool()
        assert pool_mod.get_pool_dsn() is None


class TestConnectionProbe:
    def test_the_probe_runs_select_one(self):
        conn = MagicMock()
        pool_mod._ensure_connection(conn)
        conn.cursor.return_value.__enter__.return_value.execute.assert_called_once_with("SELECT 1")

    def test_a_failed_probe_rolls_back_resets_and_reraises_the_original(self):
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.execute.side_effect = psycopg2.OperationalError("gone")
        with pytest.raises(psycopg2.OperationalError, match="gone"):
            pool_mod._ensure_connection(conn)
        conn.rollback.assert_called_once_with()
        conn.reset.assert_called_once_with()

    def test_a_failed_reset_is_logged_and_the_original_still_raised(self, caplog):
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.execute.side_effect = psycopg2.OperationalError("gone")
        conn.reset.side_effect = RuntimeError("no reset")
        with caplog.at_level("WARNING", logger=pool_mod.logger.name), pytest.raises(psycopg2.OperationalError):
            pool_mod._ensure_connection(conn)
        assert "conn.reset() also failed: no reset" in [r.getMessage() for r in caplog.records]

    def test_a_failed_rollback_is_swallowed(self):
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.execute.side_effect = psycopg2.OperationalError("gone")
        conn.rollback.side_effect = RuntimeError("no rollback")
        with pytest.raises(psycopg2.OperationalError, match="gone"):
            pool_mod._ensure_connection(conn)
        conn.reset.assert_called_once_with()


class TestVectorRegistration:
    def test_no_registration_without_pgvector(self):
        with patch("pyutilz.database.psycopg2_pool.register_vector", None):
            pool_mod._maybe_register_vector(MagicMock(), _conn())

    def test_a_connection_is_registered_only_once(self):
        conn = _conn()
        register = MagicMock()
        with patch("pyutilz.database.psycopg2_pool.register_vector", register):
            pool_mod._maybe_register_vector(MagicMock(), conn)
            pool_mod._maybe_register_vector(MagicMock(), conn)
        register.assert_called_once_with(conn)

    def test_a_failed_registration_returns_the_connection_closed_and_reraises(self):
        conn = _conn()
        pool = MagicMock()
        register = MagicMock(side_effect=RuntimeError("no vector"))
        with patch("pyutilz.database.psycopg2_pool.register_vector", register), pytest.raises(RuntimeError, match="no vector"):
            pool_mod._maybe_register_vector(pool, conn)
        pool.putconn.assert_called_once_with(conn, close=True)


class TestConnectionKeys:
    def test_the_key_is_the_backend_pid(self):
        assert pool_mod._conn_key(_conn(pid=42)) == 42

    def test_a_zero_backend_pid_falls_back_to_the_object_identity(self):
        conn = _conn(pid=0)
        assert pool_mod._conn_key(conn) == id(conn)

    def test_a_missing_backend_pid_is_logged_and_falls_back(self, caplog):
        conn = MagicMock(spec=[])
        with caplog.at_level("WARNING", logger=pool_mod.logger.name):
            key = pool_mod._conn_key(conn)
        assert key == id(conn)
        messages = [r.getMessage() for r in caplog.records]
        assert any(m.startswith("backend_pid unavailable, falling back to id(conn): ") for m in messages)


class TestGetConnectionRetries:
    def _run(self, *, fail_times, max_retries, sleeps, caplog=None):
        conn = _conn()
        pool = MagicMock(name="pool")
        pool._used = None
        pool.getconn.return_value = conn
        ensure = MagicMock(side_effect=[psycopg2.OperationalError("down")] * fail_times + [None])
        with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool", return_value=pool), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection", ensure
        ), patch("pyutilz.database.psycopg2_pool.register_vector", None), patch("pyutilz.database.psycopg2_pool.time.sleep", side_effect=sleeps.append):
            return get_connection(DSN, max_retries=max_retries), pool

    def test_the_wait_doubles_between_attempts(self):
        sleeps = []
        conn, _ = self._run(fail_times=3, max_retries=4, sleeps=sleeps)
        assert sleeps == [1, 2, 4]
        assert conn is not None

    def test_a_failed_attempt_is_logged_with_its_number_and_delay(self, caplog):
        sleeps = []
        with caplog.at_level("WARNING", logger=pool_mod.logger.name):
            self._run(fail_times=1, max_retries=3, sleeps=sleeps)
        assert "DB connection failed (attempt 1/3): down -- retrying in 1s" in [r.getMessage() for r in caplog.records]

    def test_a_failed_return_of_the_connection_is_logged_at_debug(self, caplog):
        conn = _conn()
        pool = MagicMock(name="pool")
        pool._used = None
        pool.getconn.return_value = conn
        pool.putconn.side_effect = RuntimeError("cannot put")
        ensure = MagicMock(side_effect=[psycopg2.OperationalError("down"), None])
        with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool", return_value=pool), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection", ensure
        ), patch("pyutilz.database.psycopg2_pool.register_vector", None), patch("pyutilz.database.psycopg2_pool.time.sleep"), caplog.at_level(
            "DEBUG", logger=pool_mod.logger.name
        ):
            get_connection(DSN, max_retries=3)
        assert "putconn of failed conn raised: cannot put" in [r.getMessage() for r in caplog.records]

    def test_the_pool_is_reset_between_attempts_when_it_is_idle(self):
        conn, pool = self._run(fail_times=1, max_retries=3, sleeps=[])
        pool.closeall.assert_called_once_with()

    def test_the_last_failure_is_raised_after_the_last_attempt(self):
        sleeps = []
        with pytest.raises(psycopg2.OperationalError, match="down"):
            self._run(fail_times=3, max_retries=3, sleeps=sleeps)
        assert sleeps == [1, 2]


def _use_pool(pool, dsn=DSN):
    pool_mod._pool = pool
    pool_mod._pool_dsn = dsn


class TestPoolBoundaries:
    def test_the_idle_window_is_thirty_seconds_and_the_stale_window_six_hundred(self):
        assert pool_mod._HEALTH_CHECK_IDLE_SEC == 30.0
        assert pool_mod._CONN_USED_MAX_AGE == 600.0

    def test_a_connection_idle_for_exactly_the_window_is_not_probed(self):
        pool = MagicMock()
        conn = _conn(pid=5)
        pool.getconn.return_value = conn
        _use_pool(pool)
        pool_mod._conn_last_used[5] = 0.0
        with patch("pyutilz.database.psycopg2_pool.time.monotonic", return_value=30.0), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection"
        ) as ensure, patch("pyutilz.database.psycopg2_pool.register_vector", None):
            get_connection_from_pool(DSN)
        ensure.assert_not_called()

    def test_a_connection_idle_just_past_the_window_is_probed(self):
        pool = MagicMock()
        conn = _conn(pid=5)
        pool.getconn.return_value = conn
        _use_pool(pool)
        pool_mod._conn_last_used[5] = 0.0
        with patch("pyutilz.database.psycopg2_pool.time.monotonic", return_value=30.05), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection"
        ) as ensure, patch("pyutilz.database.psycopg2_pool.register_vector", None):
            get_connection_from_pool(DSN)
        ensure.assert_called_once_with(conn)

    def test_a_connection_never_seen_is_probed_once_the_clock_passes_thirty_seconds(self):
        pool = MagicMock()
        conn = _conn(pid=9)
        pool.getconn.return_value = conn
        _use_pool(pool)
        with patch("pyutilz.database.psycopg2_pool.time.monotonic", return_value=30.05), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection"
        ) as ensure, patch("pyutilz.database.psycopg2_pool.register_vector", None):
            get_connection_from_pool(DSN)
        ensure.assert_called_once_with(conn)

    def test_a_never_seen_connection_at_exactly_thirty_seconds_is_not_probed(self):
        pool = MagicMock()
        conn = _conn(pid=9)
        pool.getconn.return_value = conn
        _use_pool(pool)
        with patch("pyutilz.database.psycopg2_pool.time.monotonic", return_value=30.0), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection"
        ) as ensure, patch("pyutilz.database.psycopg2_pool.register_vector", None):
            get_connection_from_pool(DSN)
        ensure.assert_not_called()

    def test_an_entry_exactly_at_the_stale_cutoff_is_kept(self):
        pool_mod._conn_last_used[1] = 0.0
        with patch("pyutilz.database.psycopg2_pool.time.monotonic", return_value=600.0):
            release_connection(_conn(pid=3))
        assert 1 in pool_mod._conn_last_used

    def test_an_entry_just_past_the_stale_cutoff_is_dropped(self):
        pool_mod._conn_last_used[1] = 0.0
        with patch("pyutilz.database.psycopg2_pool.time.monotonic", return_value=600.05):
            release_connection(_conn(pid=3))
        assert 1 not in pool_mod._conn_last_used

    def test_the_default_pool_and_retry_sizes_are_eight_and_three(self):
        import inspect

        params = inspect.signature(pool_mod.get_connection).parameters
        assert params["pool_max"].default == 8
        assert params["max_retries"].default == 3

    def test_one_retry_is_the_smallest_accepted_setting(self):
        conn = _conn()
        pool = MagicMock()
        pool.getconn.return_value = conn
        with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool", return_value=pool), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection"
        ), patch("pyutilz.database.psycopg2_pool.register_vector", None):
            assert get_connection(DSN, max_retries=1) is conn

    def test_the_pool_is_created_with_minimum_one_then_the_size_then_the_dsn(self):
        with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool") as MockPool, patch(
            "pyutilz.database.psycopg2_pool._ensure_connection"
        ), patch("pyutilz.database.psycopg2_pool.register_vector", None):
            MockPool.return_value.getconn.return_value = _conn()
            get_connection(DSN, pool_max=5)
        MockPool.assert_called_once_with(1, 5, DSN, keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)

    def test_a_failed_checkout_is_not_returned_to_the_pool(self):
        pool = MagicMock()
        pool._used = None
        pool.getconn.side_effect = psycopg2.OperationalError("down")
        with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool", return_value=pool), patch(
            "pyutilz.database.psycopg2_pool.time.sleep"
        ), patch("pyutilz.database.psycopg2_pool.register_vector", None), pytest.raises(psycopg2.OperationalError):
            get_connection(DSN, max_retries=1)
        pool.putconn.assert_not_called()

    def test_a_failed_health_check_returns_the_connection_closed(self):
        conn = _conn()
        pool = MagicMock()
        pool._used = None
        pool.getconn.return_value = conn
        with patch("pyutilz.database.psycopg2_pool.psycopg2.pool.ThreadedConnectionPool", return_value=pool), patch(
            "pyutilz.database.psycopg2_pool._ensure_connection", side_effect=psycopg2.OperationalError("bad")
        ), patch("pyutilz.database.psycopg2_pool.register_vector", None), pytest.raises(psycopg2.OperationalError):
            get_connection(DSN, max_retries=1)
        pool.putconn.assert_called_once_with(conn, close=True)

    def test_a_different_dsn_on_borrow_is_refused_with_the_close_hint(self):
        _use_pool(MagicMock(), dsn="dsn-a")
        with pytest.raises(ValueError) as info:
            get_connection_from_pool("dsn-b")
        assert str(info.value) == "Connection pool already created with a different DSN. Close the existing pool first."

    def test_a_stale_connection_after_recovery_is_raised_not_retried_forever(self):
        pool = MagicMock()
        stale = _conn(pid=5)
        stale.reset.side_effect = RuntimeError("no reset")
        pool.getconn.return_value = stale
        _use_pool(pool)
        pool_mod._conn_last_used[5] = 0.0
        with patch("pyutilz.database.psycopg2_pool._ensure_connection", side_effect=psycopg2.OperationalError("stale")), patch(
            "pyutilz.database.psycopg2_pool.register_vector", None
        ), pytest.raises(RuntimeError, match="no reset"):
            get_connection_from_pool(DSN)

    def test_the_stale_connection_key_is_forgotten_after_its_replacement(self):
        pool = MagicMock()
        stale = _conn(pid=5)
        stale.reset.side_effect = RuntimeError("no reset")
        pool.getconn.side_effect = [stale, _conn(pid=6)]
        _use_pool(pool)
        pool_mod._conn_last_used[5] = 0.0
        with patch("pyutilz.database.psycopg2_pool._ensure_connection", side_effect=psycopg2.OperationalError("stale")), patch(
            "pyutilz.database.psycopg2_pool.register_vector", None
        ):
            get_connection_from_pool(DSN)
        assert 5 not in pool_mod._conn_last_used

    def test_a_vector_registration_failure_returns_the_connection_and_forgets_it(self):
        pool = MagicMock()
        conn = _conn(pid=8)
        pool.getconn.return_value = conn
        _use_pool(pool)
        with patch("pyutilz.database.psycopg2_pool.register_vector", side_effect=RuntimeError("no vector")), pytest.raises(RuntimeError, match="no vector"):
            get_connection_from_pool(DSN)
        pool.putconn.assert_called_once_with(conn, close=True)
        assert 8 not in pool_mod._conn_last_used

    def test_a_fresh_borrow_registers_the_vector_type_once(self):
        pool = MagicMock()
        conn = _conn(pid=4)
        pool.getconn.return_value = conn
        _use_pool(pool)
        pool_mod._conn_last_used[4] = time.monotonic()
        register = MagicMock()
        with patch("pyutilz.database.psycopg2_pool.register_vector", register):
            assert get_connection_from_pool(DSN) is conn
        register.assert_called_once_with(conn)

    def test_a_clean_release_is_put_back_and_kept_open(self):
        pool = MagicMock()
        conn = _conn()
        _use_pool(pool)
        release_connection(conn)
        pool.putconn.assert_called_once_with(conn)
        conn.close.assert_not_called()

    def test_the_managed_connection_asks_for_the_default_pool_size_by_its_dsn(self):
        conn = _conn()
        with patch("pyutilz.database.psycopg2_pool.get_connection", return_value=conn) as get, patch("pyutilz.database.psycopg2_pool.release_connection"):
            with managed_connection(DSN):
                pass
        get.assert_called_once_with(DSN, 8)
