"""Regression tests for pyutilz.database (db.py, deltalakes.py, redislib.py).

Each test targets a specific fix and is written to fail on the pre-fix source.
"""

import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# db.validate_sql_identifier  (regex hoisted to module level; behavior preserved)
# ---------------------------------------------------------------------------

from pyutilz.database.db import validate_sql_identifier, construct_templates_and_values


def test_validate_sql_identifier_valid():
    assert validate_sql_identifier("my_table") == "my_table"
    assert validate_sql_identifier("_col1") == "_col1"


def test_validate_sql_identifier_rejects_injection():
    for bad in ["1abc", "a-b", "drop table x", "a;b", "a b", ""]:
        with pytest.raises(ValueError):
            validate_sql_identifier(bad)
    with pytest.raises(ValueError):
        validate_sql_identifier(123)  # non-string


def test_construct_templates_jsonize_sorts_keys():
    """orjson with OPT_SORT_KEYS -> stable serialization for hashing/dedup."""
    values, templates = construct_templates_and_values(
        mode="insert",
        fields=["payload"],
        replace_values={"payload": {"b": 1, "a": 2}},
        source={},
        jsonize=True,
    )
    assert values == ['{"a":2,"b":1}']
    assert templates == ["%s"]


# ---------------------------------------------------------------------------
# deltalakes.safe_delta_write  (#6: no hardcoded /tmp on Windows; non-local re-raises)
# ---------------------------------------------------------------------------

from pyutilz.database import deltalakes


def test_safe_delta_write_uses_platform_tempdir(monkeypatch, tmp_path):
    """Lock file must live under tempfile.gettempdir(), not a hardcoded '/tmp'.

    Pre-fix this raised on Windows because '/tmp' does not exist.
    """
    captured = {}

    class FakeLock:
        def __init__(self, path):
            captured["path"] = path

        def acquire(self, timeout=None):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(deltalakes, "FileLock", FakeLock)

    result = deltalakes.safe_delta_write(str(tmp_path / "mytable"), lambda: "ok")
    assert result == "ok"
    assert captured["path"].startswith(tempfile.gettempdir())
    assert "/tmp" not in captured["path"] or tempfile.gettempdir() == "/tmp"


def test_safe_delta_write_nonlocal_reraises():
    """#6: the non-local branch must re-raise (previously it swallowed the exception)."""

    def boom():
        raise ValueError("delta failed")

    with pytest.raises(ValueError, match="delta failed"):
        deltalakes.safe_delta_write("s3://bucket/table", boom)


# ---------------------------------------------------------------------------
# redislib.rexecute  (#7: no busy-loop on permanent errors; rc-None guard)
# ---------------------------------------------------------------------------

import pyutilz.database.redislib as redislib


def test_rexecute_raises_when_not_connected(monkeypatch):
    monkeypatch.setattr(redislib, "rc", None)
    with pytest.raises(RuntimeError, match="not established"):
        redislib.rexecute("get", "key")


def test_rexecute_permanent_error_does_not_busy_loop(monkeypatch):
    """A non-ConnectionError must propagate immediately, not spin forever."""

    call_count = {"n": 0}

    class FakeRC:
        def get(self, *a, **k):
            call_count["n"] += 1
            raise KeyError("permanent")

    monkeypatch.setattr(redislib, "rc", FakeRC())
    with pytest.raises(KeyError):
        redislib.rexecute("get", "somekey")
    assert call_count["n"] == 1  # not retried


def test_rexecute_transient_then_success(monkeypatch):
    from redis.exceptions import ConnectionError as RedisConnError

    state = {"n": 0}

    class FakeRC:
        def get(self, *a, **k):
            state["n"] += 1
            if state["n"] < 3:
                raise RedisConnError("temporary")
            return "value"

    monkeypatch.setattr(redislib, "rc", FakeRC())
    # sleep uses random backoff; keep it fast by monkeypatching sleep.
    monkeypatch.setattr(redislib, "sleep", lambda *_: None)
    assert redislib.rexecute("get", "k") == "value"
    assert state["n"] == 3


def test_rexecute_unknown_method_raises(monkeypatch):
    class FakeRC:
        pass

    monkeypatch.setattr(redislib, "rc", FakeRC())
    with pytest.raises(AttributeError):
        redislib.rexecute("no_such_method")


def test_rclose_resets_connection(monkeypatch):
    closed = {"v": False}

    class FakeRC:
        def close(self):
            closed["v"] = True

    monkeypatch.setattr(redislib, "rc", FakeRC())
    redislib.rclose()
    assert closed["v"] is True


# ---------------------------------------------------------------------------
# db subpackage split sensor
#
# db.py (>1000 LOC) was split into a subpackage: db/__init__.py keeps the
# connection-stateful functions + module-level mutable globals (conn/cur/
# cursors/conn_alchemy) so they all share ONE global namespace, while pure /
# stateless helpers were carved into cohesive submodules. This sensor asserts
# the public import surface still resolves through the facade AND that the
# carved helpers are the SAME objects re-exported by the facade (single source
# of truth — no divergent copies).
# ---------------------------------------------------------------------------


def test_db_facade_reexports_representative_public_symbols():
    import pyutilz.database.db as db

    for name in (
        # pure helpers carved into submodules
        "validate_sql_identifier",
        "construct_templates_and_values",
        "u",
        "nu",
        "MakeSetExcludedClause",
        "update_if_now",
        "build_upsert_query",
        "ensure_db_tables_created",
        "insert_sqllite_data",
        # connection-stateful functions kept in the parent facade
        "connect_to_db",
        "safe_execute",
        "db_command",
        "regjobs_poll",
        # module-level globals shared across the parent
        "conn_alchemy",
        "PAGE_SIZE",
    ):
        assert hasattr(db, name), f"facade lost public symbol {name!r}"

    assert db.PAGE_SIZE == 1_000_000
    assert db.validate_sql_identifier("ok_col") == "ok_col"


def test_db_facade_helpers_are_same_objects_as_submodules():
    """The facade must re-export the ACTUAL submodule objects, not copies."""
    import pyutilz.database.db as db
    from pyutilz.database.db import sql_helpers, upsert, sqlite

    assert db.validate_sql_identifier is sql_helpers.validate_sql_identifier
    assert db._SQL_IDENTIFIER_RE is sql_helpers._SQL_IDENTIFIER_RE
    assert db.construct_templates_and_values is sql_helpers.construct_templates_and_values
    assert db.MakeSetExcludedClause is sql_helpers.MakeSetExcludedClause
    assert db.build_upsert_query is upsert.build_upsert_query
    assert db.insert_sqllite_data is sqlite.insert_sqllite_data


def test_db_connection_globals_live_in_parent_facade():
    """Connection state (mutable module globals) must exist on the facade so
    every stateful function reads/writes the SAME namespace after a connect."""
    import pyutilz.database.db as db

    # conn_alchemy is defined at import time; conn/cur/cursors are created by
    # connect_to_db. Simulate a connect by assigning the globals and confirm a
    # stateful reader (get_cursor) sees them via the shared module namespace.
    class _FakeCur:
        description = None

        def execute(self, *a, **k):
            pass

    db.conn = object()
    db.cursors = {"cursor": _FakeCur()}
    assert db.get_cursor("cursor") is db.cursors["cursor"]

    # lazy alias pyutilz.db must resolve to the same module object.
    import pyutilz

    assert pyutilz.db is db
    assert redislib.rc is None
