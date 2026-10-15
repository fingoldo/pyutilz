"""Regression tests for the 2026-10-15 audit findings in config, psycopg2_pool, disk_cache,
resilience and the db connection/execution modules (C-01, C-02, C-07, C-08, C-09, P-05, R-01,
R-02, R-03, S-10, X-07, X-18).
"""

from __future__ import annotations

import os
import threading
from collections import deque
from pathlib import Path
from typing import Any, List

import pytest

from pyutilz.core.disk_cache import DiskCache
from pyutilz.database import exceptions as db_exceptions
from pyutilz.database.db import connection as connection_mod
from pyutilz.database.db import execution as execution_mod
from pyutilz.database import db as db_facade
from pyutilz.database import psycopg2_pool as pool_mod
from pyutilz.system import resilience
from pyutilz.system.config import TomlLiveConfig

# ── C-01: mtime must be recorded before the read ───────────────────────────────


def test_c01_edit_during_reload_is_picked_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "c.toml"
    p.write_text("[s]\nk = 1\n", encoding="utf-8")
    os.utime(p, (1000, 1000))
    cfg = TomlLiveConfig(p, check_interval=0)
    assert cfg.get("s", "k") == 1

    original_read_bytes = Path.read_bytes
    fired: List[int] = []

    def racing_read_bytes(self: Path) -> bytes:
        data = original_read_bytes(self)
        if self == p and not fired:
            # An editor writes the new content after this reload has read the old bytes.
            fired.append(1)
            p.write_text("[s]\nk = 2\n", encoding="utf-8")
            os.utime(p, (2000, 2000))
        return data

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    cfg._reload()  # reads k=1, the edit lands during the read
    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)

    assert fired, "the race was not simulated"
    # The next debounced check must see the newer mtime and reload to k=2.
    assert cfg.get("s", "k") == 2


# ── C-08: data setter stores a copy; get() reads under the lock ─────────────────


def test_c08_data_setter_copies_caller_dict(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    p.write_text("[s]\nk = 1\n", encoding="utf-8")
    cfg = TomlLiveConfig(p, check_interval=1e9)
    incoming: dict = {"s": {"k": 1}}
    cfg.data = incoming
    # The copy is shallow (same as the data getter), so the top level is what must be isolated.
    incoming["s"] = {"k": 99}
    incoming["t"] = {"x": 1}
    assert cfg.get("s", "k") == 1
    assert "t" not in cfg.data


# ── C-09: CircuitBreaker.state is a locked snapshot, not the live object ────────


def test_c09_state_snapshot_is_detached_from_breaker() -> None:
    cb = resilience.CircuitBreaker("audit-c09")
    snap = cb.state
    snap.is_open = True
    snap.failure_count = 42
    assert cb.state.is_open is False
    assert cb.state.failure_count == 0


def test_c09_state_is_not_assignable() -> None:
    cb = resilience.CircuitBreaker("audit-c09-assign")
    with pytest.raises(AttributeError):
        cb.state = resilience.CircuitState()  # type: ignore[misc]


# ── P-05: dead-letter queue is a bounded deque ─────────────────────────────────


def test_p05_dead_letter_queue_is_bounded_deque() -> None:
    q = resilience.DeadLetterQueue(max_size=3)
    for i in range(5):
        q.add(f"op{i}", {}, ValueError("boom"))
    assert isinstance(q.queue, deque)
    assert q.queue.maxlen == 3
    assert q.size() == 3
    assert [e["operation"] for e in q.get_recent(10)] == ["op4", "op3", "op2"]
    assert [e["operation"] for e in q.get_recent(2)] == ["op4", "op3"]
    assert q.get_recent(0) == []


# ── C-02: a per-call retry must not tear down a pool with checked-out connections ──


class _FakePool:
    def __init__(self, used: dict) -> None:
        self._used = used
        self.closed = False

    def closeall(self) -> None:
        self.closed = True


def test_c02_reset_skipped_while_connections_checked_out(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePool(used={1: object()})
    monkeypatch.setattr(pool_mod, "_pool", fake)
    monkeypatch.setattr(pool_mod, "_pool_dsn", "dsn-a")

    pool_mod._reset_pool(expected=fake)

    assert fake.closed is False
    assert pool_mod._pool is fake
    assert pool_mod.get_pool_dsn() == "dsn-a"


def test_c02_reset_runs_when_no_connection_checked_out(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePool(used={})
    monkeypatch.setattr(pool_mod, "_pool", fake)
    monkeypatch.setattr(pool_mod, "_pool_dsn", "dsn-a")

    pool_mod._reset_pool(expected=fake)

    assert fake.closed is True
    assert pool_mod._pool is None


def test_c02_reset_skipped_when_pool_already_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    current = _FakePool(used={})
    stale = _FakePool(used={})
    monkeypatch.setattr(pool_mod, "_pool", current)
    monkeypatch.setattr(pool_mod, "_pool_dsn", "dsn-a")

    pool_mod._reset_pool(expected=stale)

    assert current.closed is False
    assert pool_mod._pool is current


# ── C-07: DiskCache.get waits for the per-key lock held by put ──────────────────


def test_c07_get_waits_for_key_lock(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.put("k1", 123) is True
    results: List[Any] = []
    with cache._get_key_lock("k1"):
        reader = threading.Thread(target=lambda: results.append(cache.get("k1")))
        reader.start()
        reader.join(timeout=0.3)
        assert reader.is_alive(), "get() must block while the per-key lock is held"
    reader.join(timeout=5)
    assert results == [123]


# ── S-10: put reports whether the value was stored ─────────────────────────────


def test_s10_put_returns_false_on_failure_and_true_on_success(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.put("bad", lambda: 1) is False  # unpicklable
    assert cache.put("good", {"a": 1}) is True
    assert cache.get("good") == {"a": 1}


# ── R-01: reconnect closes the connection it replaces ──────────────────────────


class _FakeConn:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakePsycopg2:
    def __init__(self) -> None:
        self.made: List[_FakeConn] = []

    def connect(self, **params: Any) -> _FakeConn:
        conn = _FakeConn()
        self.made.append(conn)
        return conn


def test_r01_reconnect_closes_replaced_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    old = _FakeConn()
    fake_pg = _FakePsycopg2()
    monkeypatch.setattr(db_facade, "psycopg2", fake_pg, raising=False)
    monkeypatch.setattr(db_facade, "conn", old, raising=False)
    monkeypatch.setattr(connection_mod, "_setup_postgres_session", lambda *a, **k: None)

    connection_mod.connect_to_db("db", "host", 5432, "user", "pwd", max_retries=1)

    new = fake_pg.made[-1]
    assert db_facade.conn is new
    assert old.closed is True
    assert new.closed is False


# ── R-02: the replaced SQLAlchemy engine is disposed ───────────────────────────


class _FakeEngine:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


def test_r02_previous_engine_is_disposed(monkeypatch: pytest.MonkeyPatch) -> None:
    old_engine = _FakeEngine()
    monkeypatch.setattr(db_facade, "conn_alchemy", old_engine, raising=False)
    connection_mod._dispose_previous_engine()
    assert old_engine.disposed is True


def test_r02_dispose_failure_is_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken:
        def dispose(self) -> None:
            raise RuntimeError("socket already gone")

    monkeypatch.setattr(db_facade, "conn_alchemy", _Broken(), raising=False)
    connection_mod._dispose_previous_engine()  # must not raise


# ── R-03: named cursor closed on the OperationalError retry branch ─────────────


class _FakeCursor:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_r03_named_cursor_closed_on_operational_error_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    cursors: List[_FakeCursor] = []

    def fake_get_cursor(**kwargs: Any) -> _FakeCursor:
        cur = _FakeCursor()
        cursors.append(cur)
        return cur

    def failing_execute(local_cur: Any, ex_type: str, statement: Any, data: Any, page_size: int) -> None:
        raise execution_mod.OperationalError("connection dropped")

    monkeypatch.setattr(db_facade, "get_cursor", fake_get_cursor, raising=False)
    monkeypatch.setattr(db_facade, "get_cursor_type", lambda *a, **k: "named", raising=False)
    monkeypatch.setattr(db_facade, "sleep", lambda s: None, raising=False)
    monkeypatch.setattr(execution_mod, "_execute_on_cursor", failing_execute)
    monkeypatch.setattr(execution_mod, "_reconnect_for_retry", lambda: None)

    with pytest.raises(execution_mod.OperationalError):
        execution_mod.basic_db_execute("execute", "select 1", cursor_name="c1", max_retries=2)

    assert len(cursors) == 2
    assert all(cur.closed == 1 for cur in cursors)


# ── X-07: returning must be a plain column list ────────────────────────────────


def test_x07_returning_injection_is_refused() -> None:
    with pytest.raises(db_exceptions.SQLValidationError):
        execution_mod._build_dml_statement("select", "t", ["a"], None, {}, {"a": 1}, False, "id; drop table t")


def test_x07_plain_column_lists_still_build() -> None:
    sql, _ = execution_mod._build_dml_statement("select", "t", ["a"], None, {}, {"a": 1}, False, "id, t.name")
    assert sql.startswith("select id, t.name from")
    sql, _ = execution_mod._build_dml_statement("select", "t", ["a"], None, {}, {"a": 1}, False, "*")
    assert sql.startswith("select * from")


# ── X-18: alias, prefix and suffix of get_table_fields are validated ───────────


def test_x18_prefix_with_sql_is_refused() -> None:
    with pytest.raises(db_exceptions.SQLValidationError):
        execution_mod.get_table_fields("t", "o", prefix="x, (select 1) as y --")


def test_x18_alias_with_sql_is_refused() -> None:
    with pytest.raises(db_exceptions.SQLValidationError):
        execution_mod.get_table_fields("t", "o; drop")
