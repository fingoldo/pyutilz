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


def test_construct_templates_select_mode_builds_assignment_clauses():
    values, templates = construct_templates_and_values(
        mode="select",
        fields=["a", "b"],
        replace_values={"a": 1},
        source={"b": 2},
        jsonize=False,
    )
    assert values == [1, 2]
    assert templates == ["a=%s", "b=%s"]


def test_construct_templates_jsonize_false_leaves_dict_value_untouched():
    values, _ = construct_templates_and_values(
        mode="insert",
        fields=["payload"],
        replace_values={"payload": {"a": 1}},
        source={},
        jsonize=False,
    )
    assert values == [{"a": 1}]


# ---------------------------------------------------------------------------
# db.validate_sql_qualified_identifier
# ---------------------------------------------------------------------------

from pyutilz.database.db import validate_sql_qualified_identifier


def test_validate_sql_qualified_identifier_valid():
    assert validate_sql_qualified_identifier("schema.table") == "schema.table"
    assert validate_sql_qualified_identifier("table") == "table"


def test_validate_sql_qualified_identifier_rejects_invalid_part():
    for bad in ["schema.bad-name", "bad-name.table", "a..b", "a.", "1abc.table"]:
        with pytest.raises(ValueError):
            validate_sql_qualified_identifier(bad)


def test_validate_sql_qualified_identifier_rejects_non_string():
    with pytest.raises(ValueError):
        validate_sql_qualified_identifier(123)


# ---------------------------------------------------------------------------
# db.u / db.nu  (SQL string-literal quoting helpers)
# ---------------------------------------------------------------------------

from pyutilz.database.db import u, nu


def test_u_quotes_and_escapes():
    assert u(None) == "null"
    assert u("abc") == "'abc'"
    assert u("a'b") == "'a''b'"


def test_nu_treats_empty_string_as_null():
    assert nu(None) == "null"
    assert nu("") == "null"
    assert nu("abc") == "'abc'"


# ---------------------------------------------------------------------------
# db.MakeSetExcludedClause / db.update_if_now
# ---------------------------------------------------------------------------

from pyutilz.database.db import MakeSetExcludedClause, update_if_now


def test_make_set_excluded_clause_basic():
    assert MakeSetExcludedClause("a,b") == "a=excluded.a,b=excluded.b"


def test_make_set_excluded_clause_with_updated_at_timestamp():
    clause = MakeSetExcludedClause("a,b", bAddUpdatedAtTimestamp="updated_at")
    assert clause == "a=excluded.a,b=excluded.b,updated_at=(now() at time zone 'utc')"


def test_make_set_excluded_clause_rejects_invalid_field():
    with pytest.raises(ValueError):
        MakeSetExcludedClause("good,bad-name")


def test_update_if_now_do_update_branch():
    result = update_if_now("2026-07-13 now()", "a=excluded.a")
    assert "do update set" in result
    assert "a=excluded.a" in result


def test_update_if_now_do_nothing_branch():
    assert update_if_now("2026-07-13", "a=excluded.a") == "do nothing"


def test_update_if_now_case_insensitive():
    result = update_if_now("NOW()", "a=excluded.a")
    assert "do update set" in result


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


def test_safe_delta_write_lock_key_is_case_insensitive_on_windows(monkeypatch, tmp_path):
    """Regression (2026-07-21 audit round 2): os.path.abspath() never changes case, but NTFS
    (Windows' default filesystem) is case-insensitive-but-preserving -- "C:\\Data\\orders" and
    "c:\\data\\ORDERS" are the SAME table on disk but used to hash to two DIFFERENT lock files,
    letting two writers both acquire "their own" lock and commit concurrently to the same
    underlying Delta log. Both spellings must now resolve to the identical lock file."""
    captured_paths = []

    class FakeLock:
        def __init__(self, path):
            captured_paths.append(path)

        def acquire(self, timeout=None):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(deltalakes, "FileLock", FakeLock)

    base = str(tmp_path / "region-us" / "orders")
    variant = str(tmp_path / "region-us" / "orders").upper() if os.name == "nt" else str(tmp_path / "region-us" / "orders")

    deltalakes.safe_delta_write(base, lambda: "ok")
    deltalakes.safe_delta_write(variant, lambda: "ok")

    if os.name == "nt":
        assert captured_paths[0] == captured_paths[1], "same table under different case must map to the SAME lock file on Windows"


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

    # conn_alchemy is defined at import time; conn/cur are created by connect_to_db.
    # `cursors` is deliberately NOT a plain shared global anymore (see
    # test_get_cursor_cache_is_thread_local below) -- get_cursor()'s cache lives per-thread.
    class _FakeCur:
        description = None

        def execute(self, *a, **k):
            pass

    db.conn = object()
    db._get_thread_cursors()["cursor"] = _FakeCur()
    assert db.get_cursor("cursor") is db._get_thread_cursors()["cursor"]

    # lazy alias pyutilz.db must resolve to the same module object.
    import pyutilz

    assert pyutilz.db is db
    assert redislib.rc is None


def test_get_cursor_cache_is_thread_local():
    """Regression (2026-07-21 audit round 2, CRITICAL): the cursor cache used to be one shared
    module-global dict, so two threads calling get_cursor()/basic_db_execute() concurrently could
    interleave onto the SAME psycopg2 cursor object -- a cursor's execute()/fetchall() pair is
    not safe to share across threads even though the connection itself is. Each thread must now
    get an independently-cached cursor object for the same cursor_type."""
    import threading

    import pyutilz.database.db as db

    class _FakeCur:
        description = None

        def execute(self, *a, **k):
            pass

    db.conn = object()
    db._get_thread_cursors().clear()

    results: dict = {}

    def worker(name):
        # Each thread populates its OWN cache entry for the same cursor_type key.
        db._get_thread_cursors()["cursor"] = _FakeCur()
        results[name] = db.get_cursor("cursor")

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t1.join()
    t2.start()
    t2.join()

    assert results["t1"] is not results["t2"], "threads must not observe each other's cached cursor object"


# ---------------------------------------------------------------------------
# db.execute_alchemy  (Engine.execute() was removed in SQLAlchemy 2.0; the
# function used to call conn_alchemy.execute(sql) directly on the Engine,
# which raises AttributeError on every call with the installed SQLAlchemy
# version. Fixed to open a Connection via conn_alchemy.connect() and commit.)
# ---------------------------------------------------------------------------


def test_execute_alchemy_engine_has_no_execute_method():
    # Pins the actual SQLAlchemy 2.0 API shape this bug depended on.
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    assert not hasattr(engine, "execute")


def test_execute_alchemy_runs_sql_against_the_engine():
    from sqlalchemy import create_engine
    import pyutilz.database.db as db

    engine = create_engine("sqlite:///:memory:")
    db.conn_alchemy = engine
    try:
        db.execute_alchemy("CREATE TABLE t (x INTEGER)")
        db.execute_alchemy("INSERT INTO t VALUES (1)")
        with engine.connect() as conn:
            rows = conn.exec_driver_sql("SELECT x FROM t").fetchall()
        assert rows == [(1,)]
    finally:
        db.conn_alchemy = None


# ---------------------------------------------------------------------------
# db.basic_db_execute  (InternalError branch used to log-and-fall-through with
# no return/raise/retry-increment, so callers got an implicit None indistinguishable
# from "no rows" - or, inside a caller-added retry wrapper, an infinite silent loop.
# Fixed to re-raise InternalError so it propagates to the caller.)
# ---------------------------------------------------------------------------


def test_basic_db_execute_reraises_internal_error(monkeypatch):
    from psycopg2 import InternalError
    import pyutilz.database.db as db

    def fake_get_cursor(*args, **kwargs):
        raise InternalError("current transaction is aborted")

    monkeypatch.setattr(db, "get_cursor", fake_get_cursor)

    with pytest.raises(InternalError):
        db.basic_db_execute("execute", "select 1")


# ---------------------------------------------------------------------------
# db.basic_db_execute  (a named/server-side cursor -- withhold=True on Postgres --
# was never closed on the return_cursor=False success path, leaking a real
# DECLARE ... CURSOR ... WITH HOLD on the server. Fixed to close it right after
# fetchall() whenever the caller isn't taking ownership of the cursor itself.)
# ---------------------------------------------------------------------------


def test_basic_db_execute_closes_named_cursor_when_not_returned(monkeypatch):
    import pyutilz.database.db as db

    class _FakeCur:
        description = [("col",)]
        closed = False

        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return [(1,)]

        def close(self):
            self.closed = True

    fake_cur = _FakeCur()
    monkeypatch.setattr(db, "get_cursor", lambda *a, **k: fake_cur)

    rows = db.basic_db_execute("execute", "select 1", cursor_name="page1", return_cursor=False)

    assert rows == [(1,)]
    assert fake_cur.closed is True


def test_basic_db_execute_does_not_close_cursor_when_returned(monkeypatch):
    """When return_cursor=True the caller takes ownership -- basic_db_execute must not close it
    out from under them."""
    import pyutilz.database.db as db

    class _FakeCur:
        description = [("col",)]
        closed = False

        def execute(self, *a, **k):
            pass

        def close(self):
            self.closed = True

    fake_cur = _FakeCur()
    monkeypatch.setattr(db, "get_cursor", lambda *a, **k: fake_cur)

    result = db.basic_db_execute("execute", "select 1", cursor_name="page1", return_cursor=True)

    assert result is fake_cur
    assert fake_cur.closed is False


def test_basic_db_execute_does_not_close_unnamed_cursor(monkeypatch):
    """Only NAMED (server-side) cursors need explicit closing here -- basic_db_execute's cache
    reuses unnamed cursors across calls (see get_cursor), so closing one after every call would
    break that reuse."""
    import pyutilz.database.db as db

    class _FakeCur:
        description = [("col",)]
        closed = False

        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return [(1,)]

        def close(self):
            self.closed = True

    fake_cur = _FakeCur()
    monkeypatch.setattr(db, "get_cursor", lambda *a, **k: fake_cur)

    db.basic_db_execute("execute", "select 1", cursor_name=None, return_cursor=False)

    assert fake_cur.closed is False


# ---------------------------------------------------------------------------
# db.basic_db_execute  (exception log lines never included the failing SQL
# statement, so a production incident from one specific bad query was logged
# as an anonymous traceback indistinguishable from every other call site.)
# ---------------------------------------------------------------------------


def test_basic_db_execute_logs_the_failing_statement(monkeypatch, caplog):
    import logging

    import pyutilz.database.db as db

    def fake_get_cursor(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(db, "get_cursor", fake_get_cursor)

    with caplog.at_level(logging.ERROR, logger="pyutilz.database.db"):
        with pytest.raises(RuntimeError):
            db.basic_db_execute("execute", "select * from very_specific_table where x=1")

    assert any("very_specific_table" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# db.connect_to_db  (retry loop leaked the just-opened psycopg2 connection
# when a later step in the same try block - set_isolation_level, create_engine,
# init_params_fn() - raised after psycopg2.connect() had already succeeded.
# Fixed to close() the freshly-opened conn before sleeping/retrying.)
# ---------------------------------------------------------------------------


def test_connect_to_db_closes_leaked_conn_on_later_failure(monkeypatch):
    import pyutilz.database.db as db

    closed = {"n": 0}

    class FakeConn:
        def __init__(self):
            self.closed_flag = False

        def set_isolation_level(self, level):
            # simulate a failure AFTER the connection was already opened
            raise RuntimeError("isolation level failed")

        def close(self):
            closed["n"] += 1
            self.closed_flag = True

    attempts = {"n": 0}

    def fake_psycopg2_connect(**params):
        attempts["n"] += 1
        if attempts["n"] >= 3:
            # stop the infinite retry loop by making the whole connect succeed
            # on the 3rd attempt (no set_isolation_level failure this time)
            class WorkingConn(FakeConn):
                def set_isolation_level(self, level):
                    pass

                def cursor(self):
                    return object()

            return WorkingConn()
        return FakeConn()

    monkeypatch.setattr(db.psycopg2, "connect", fake_psycopg2_connect)
    monkeypatch.setattr(db.sqlalchemy, "create_engine", lambda *a, **k: object())
    monkeypatch.setattr(db, "sleep", lambda *_: None)

    db.connect_to_db(
        m_db_name="testdb",
        m_db_host="localhost",
        m_db_port=5432,
        m_db_username="u",
        m_db_pwd="p",
        m_db_flavor="postgres",
    )

    # Both failed attempts must have had their leaked connection closed.
    assert closed["n"] == 2
    assert attempts["n"] == 3


def test_connect_to_db_close_failure_does_not_prevent_retry(monkeypatch):
    """If conn.close() itself raises (broken connection), the retry loop must
    still continue rather than propagating that secondary exception."""
    import pyutilz.database.db as db

    class BrokenCloseConn:
        def set_isolation_level(self, level):
            raise RuntimeError("isolation level failed")

        def close(self):
            raise RuntimeError("close also failed")

    attempts = {"n": 0}

    def fake_psycopg2_connect(**params):
        attempts["n"] += 1
        if attempts["n"] >= 2:
            class WorkingConn:
                def set_isolation_level(self, level):
                    pass

                def close(self):
                    pass

                def cursor(self):
                    return object()

            return WorkingConn()
        return BrokenCloseConn()

    monkeypatch.setattr(db.psycopg2, "connect", fake_psycopg2_connect)
    monkeypatch.setattr(db.sqlalchemy, "create_engine", lambda *a, **k: object())
    monkeypatch.setattr(db, "sleep", lambda *_: None)

    # Must not raise, despite close() itself failing on the first attempt.
    db.connect_to_db(
        m_db_name="testdb",
        m_db_host="localhost",
        m_db_port=5432,
        m_db_username="u",
        m_db_pwd="p",
        m_db_flavor="postgres",
    )


# ---------------------------------------------------------------------------
# upsert.build_upsert_query  hash_fields contract (was type-hinted str with
# default '', but iterated element-by-element in several places; a caller
# passing a single str got it split character-by-character into the SQL).
# ---------------------------------------------------------------------------

from pyutilz.database.db.upsert import build_upsert_query


def test_build_upsert_query_hash_fields_bare_str_treated_as_single_field():
    """A bare str hash_fields must be treated as ONE field name, not iterated
    char-by-char (pre-fix: 'md5hash' -> validate_sql_identifier('m'), ('d'), ...
    which either raises on the first single-char token or corrupts the SQL)."""
    query = build_upsert_query(
        fields_names=["id", "name", "md5hash"],
        table_name="mytable",
        conflict_fields=["id"],
        history_table_name="mytable_history",
        history_fields=["id", "name"],
        hash_fields="md5hash",
    )
    # the whole field name must appear intact in the generated hash-changed condition
    assert "c.md5hash" in query and "u.md5hash" in query
    # and must NOT have been shredded into single-character conditions
    assert "c.m " not in query and "u.m<" not in query


def test_build_upsert_query_hash_fields_list_still_works():
    query = build_upsert_query(
        fields_names=["id", "name", "hash_a", "hash_b"],
        table_name="mytable",
        conflict_fields=["id"],
        history_table_name="mytable_history",
        history_fields=["id", "name"],
        hash_fields=["hash_a", "hash_b"],
    )
    assert "c.hash_a" in query and "c.hash_b" in query


def test_build_upsert_query_hash_fields_default_empty_still_works():
    query = build_upsert_query(
        fields_names=["id", "name"],
        table_name="mytable",
        conflict_fields=["id"],
    )
    assert "insert into mytable" in query


# ---------------------------------------------------------------------------
# db.log_to_db  (an unrecognized `level` silently downgraded to INFO severity
# with no signal, and "critical"/"fatal" -- reasonable caller-invented level
# names -- fell into the same silent-INFO bucket instead of ERROR.)
# ---------------------------------------------------------------------------


def test_log_to_db_unrecognized_level_warns_and_degrades_to_info(monkeypatch, caplog):
    import logging

    import pyutilz.database.db as db

    captured = {}
    monkeypatch.setattr(db, "safe_execute", lambda stmt, params: captured.setdefault("params", params))

    with caplog.at_level(logging.WARNING, logger="pyutilz.database.db"):
        db.log_to_db("something happened", level="Critical!!!")

    assert any("unrecognized level" in r.message for r in caplog.records)
    assert captured["params"][4] == 1  # cInfo -- still degrades safely, doesn't raise


def test_log_to_db_critical_maps_to_error_severity(monkeypatch):
    import pyutilz.database.db as db

    captured = {}
    monkeypatch.setattr(db, "safe_execute", lambda stmt, params: captured.setdefault("params", params))

    db.log_to_db("disk full", level="critical")

    assert captured["params"][4] == 3  # cError


def test_log_to_db_fatal_maps_to_error_severity(monkeypatch):
    import pyutilz.database.db as db

    captured = {}
    monkeypatch.setattr(db, "safe_execute", lambda stmt, params: captured.setdefault("params", params))

    db.log_to_db("disk full", level="fatal")

    assert captured["params"][4] == 3  # cError


def test_log_to_db_warning_maps_to_warning_severity(monkeypatch):
    import pyutilz.database.db as db

    captured = {}
    monkeypatch.setattr(db, "safe_execute", lambda stmt, params: captured.setdefault("params", params))

    db.log_to_db("disk getting full", level="warning")

    assert captured["params"][4] == 2  # cWarning


# ---------------------------------------------------------------------------
# db.EnsurePgTableExists / db.ReadTableIntoDic / db.ReadTableIntoDicReversed /
# db.GetIdByKeyFieldAndInsertIfNeeded  (PascalCase, Hungarian-notation names
# mixed in with ~28 modern snake_case functions in the same module's public
# surface -- undiscoverable via IDE autocomplete conventions the rest of the
# module trains the user to expect. Added modern snake_case primaries; the
# old names are now thin, deprecated delegating wrappers.)
# ---------------------------------------------------------------------------


def test_ensure_pg_table_exists_pascalcase_alias_warns_and_delegates(monkeypatch):
    import pyutilz.database.db as db

    captured = {}
    monkeypatch.setattr(db, "check_if_pg_table_exists", lambda table_name, schema_name="public": True)

    def fake_ensure(table, key_field_name=None, id_field_name=None, autocreate_id_type_name=None):
        captured.update(table=table, key_field_name=key_field_name, id_field_name=id_field_name, autocreate_id_type_name=autocreate_id_type_name)

    monkeypatch.setattr(db, "ensure_pg_table_exists", fake_ensure)

    with pytest.warns(DeprecationWarning, match="ensure_pg_table_exists"):
        db.EnsurePgTableExists(sTable="t", sKeyFieldName="k", sIdFieldName="i")

    assert captured == {"table": "t", "key_field_name": "k", "id_field_name": "i", "autocreate_id_type_name": None}


def test_ensure_pg_table_exists_snake_case_is_the_real_implementation(monkeypatch):
    """The snake_case function must not itself warn -- it's the primary, non-deprecated entry point."""
    import warnings

    import pyutilz.database.db as db

    monkeypatch.setattr(db, "check_if_pg_table_exists", lambda table_name, schema_name="public": True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        db.ensure_pg_table_exists(table="t")  # must not raise


def test_read_table_into_dict_pascalcase_alias_warns_and_delegates(monkeypatch):
    import pyutilz.database.db as db

    calls = []
    monkeypatch.setattr(
        db,
        "read_table_into_dict",
        lambda dict_enums, table, key_field_name=None, condition=None, id_field_name=None, autocreate_id_type_name=None: calls.append((table, key_field_name)),
    )

    with pytest.warns(DeprecationWarning, match="read_table_into_dict"):
        db.ReadTableIntoDic({}, sTable="t", sKeyFieldName="k")

    assert calls == [("t", "k")]


def test_read_table_into_dict_reversed_pascalcase_alias_warns_and_delegates(monkeypatch):
    import pyutilz.database.db as db

    calls = []
    monkeypatch.setattr(
        db,
        "read_table_into_dict_reversed",
        lambda dict_enums, table, key_field_name=None, condition=None, id_field_name=None, autocreate_id_type_name=None: calls.append((table, key_field_name)),
    )

    with pytest.warns(DeprecationWarning, match="read_table_into_dict_reversed"):
        db.ReadTableIntoDicReversed({}, sTable="t", sKeyFieldName="k")

    assert calls == [("t", "k")]


def test_get_id_by_key_field_and_insert_if_needed_pascalcase_alias_warns_and_delegates(monkeypatch):
    import pyutilz.database.db as db

    calls = []

    def fake(dict_enums, table, key_field_value, **kwargs):
        calls.append((table, key_field_value))
        return "the-id"

    monkeypatch.setattr(db, "get_id_by_key_field_and_insert_if_needed", fake)

    with pytest.warns(DeprecationWarning, match="get_id_by_key_field_and_insert_if_needed"):
        result = db.GetIdByKeyFieldAndInsertIfNeeded({}, sTable="t", sKeyFieldValue="v")

    assert result == "the-id"
    assert calls == [("t", "v")]


def test_get_id_by_key_field_and_insert_if_needed_null_shortcut():
    import pyutilz.database.db as db

    assert db.get_id_by_key_field_and_insert_if_needed({}, table="t", key_field_value="null") == "null"


def test_get_id_by_key_field_and_insert_if_needed_cache_hit():
    import pyutilz.database.db as db

    dict_enums = {"existing": 42}
    assert db.get_id_by_key_field_and_insert_if_needed(dict_enums, table="t", key_field_value="existing") == 42
