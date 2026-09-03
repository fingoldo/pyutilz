"""Regression tests for the database findings of the 2026-09-03 domain audit
(09-domain-db-web-cloud-llm-text.md): F04, F15, F16, F17, F18, F24, F25, F48..F56.

No database is touched: the SQL builders are pure (assertions are on the generated text) and every
function that executes goes through a monkeypatched facade / fake cursor.
"""

import logging
from datetime import date, datetime, timezone

import pytest

import pyutilz.database.db as db
from pyutilz.database.db.upsert import build_upsert_query

# ---------------------------------------------------------------------------
# F15 - fresh_data's RETURNING must project every conflict field, because the
# history CTE joins `fresh_data u` on it (`u.<key>=c.<key>`).
# ---------------------------------------------------------------------------


def test_fresh_data_returns_conflict_key_even_when_not_in_history_fields():
    query = build_upsert_query(
        fields_names=["id", "name", "h"],
        table_name="t",
        conflict_fields=["id"],
        on_conflict_update_fields=["name"],
        history_table_name="t_hist",
        history_fields=["name"],
        hash_fields="h",
    )
    fresh_returning = query[query.index(" returning ") : query.index(",changed_data")]
    # pre-fix: "returning name,h" -- then "left join t c on u.id=c.id" referenced a column
    # fresh_data never projected -> ERROR: column u.id does not exist.
    assert "id" in [f.strip() for f in fresh_returning.replace(" returning ", "").replace(")", "").split(",")]
    assert "u.id=c.id" in query


def test_conflict_key_already_in_history_fields_is_not_duplicated():
    query = build_upsert_query(
        fields_names=["id", "name", "h"],
        table_name="t",
        conflict_fields=["id"],
        history_table_name="t_hist",
        history_fields=["id", "name"],
        hash_fields="h",
    )
    fresh_returning = query[query.index(" returning ") : query.index(",changed_data")]
    assert fresh_returning.count("id") == 1


# ---------------------------------------------------------------------------
# F16 - an aliased conflict field must stay in changed_data's RETURNING under
# its ORIGINAL name, since the tail UPDATE joins `changed_data as c` on it.
# ---------------------------------------------------------------------------


def test_aliased_conflict_field_is_projected_back_under_its_original_name():
    query = build_upsert_query(
        fields_names=["id", "name"],
        table_name="t",
        conflict_fields=["id"],
        history_table_name="t_hist",
        history_fields=["id", "name"],
        history_fields_aliases={"id": "hist_id"},
        timestamp_check_fields=["checked_at"],
        timestamp_update_fields=["updated_at"],
    )
    # pre-fix: changed_data projected only `checked_at`, yet the tail statement said
    # "... from changed_data as c where u.id=c.id" -> ERROR: column c.id does not exist.
    assert "hist_id as id" in query
    assert "update t AS u set updated_at=c.checked_at from changed_data as c where u.id=c.id" in query


# ---------------------------------------------------------------------------
# F48 - default_timestamp is a raw SQL fragment and must be documented as one.
# ---------------------------------------------------------------------------


def test_default_timestamp_is_documented_as_a_raw_fragment():
    doc = build_upsert_query.__doc__ or ""
    assert "default_timestamp" in doc
    assert "WARNING" in doc and "verbatim" in doc


# ---------------------------------------------------------------------------
# F04 - release_connection must not let a PoolError escape (pool recreated while
# the connection was checked out) and must close the orphaned backend instead.
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self):
        self.closed_count = 0
        self.committed = 0
        self.rolled_back = 0

    def rollback(self):
        self.rolled_back += 1

    def commit(self):
        self.committed += 1

    def close(self):
        self.closed_count += 1

    @property
    def info(self):
        raise AttributeError("no info")


def test_release_connection_survives_a_foreign_pool_and_closes_the_connection(monkeypatch):
    import psycopg2.pool

    from pyutilz.database import psycopg2_pool as pool_mod

    class _HostilePool:
        def putconn(self, conn, close=False):
            raise psycopg2.pool.PoolError("trying to put unkeyed connection")

    conn = _FakeConn()
    monkeypatch.setattr(pool_mod, "_pool", _HostilePool())

    # pre-fix: PoolError propagated out of release_connection (masking whatever the caller was
    # unwinding) and the backend leaked.
    pool_mod.release_connection(conn)

    assert conn.closed_count == 1


# ---------------------------------------------------------------------------
# F25 - managed_connection commits on clean exit, rolls back on exception.
# ---------------------------------------------------------------------------


def test_managed_connection_commits_on_clean_exit(monkeypatch):
    from pyutilz.database import psycopg2_pool as pool_mod

    conn = _FakeConn()
    monkeypatch.setattr(pool_mod, "get_connection", lambda dsn, pool_max=8: conn)
    monkeypatch.setattr(pool_mod, "release_connection", lambda c: None)

    with pool_mod.managed_connection("dsn") as c:
        assert c is conn

    # pre-fix: nothing committed, and release_connection's unconditional rollback discarded the work.
    assert conn.committed == 1
    assert conn.rolled_back == 0


def test_managed_connection_does_not_commit_on_exception(monkeypatch):
    from pyutilz.database import psycopg2_pool as pool_mod

    conn = _FakeConn()
    monkeypatch.setattr(pool_mod, "get_connection", lambda dsn, pool_max=8: conn)
    monkeypatch.setattr(pool_mod, "_pool", None)

    with pytest.raises(RuntimeError):
        with pool_mod.managed_connection("dsn"):
            raise RuntimeError("boom")

    assert conn.committed == 0
    # release_connection() is what aborts the transaction on the error path.
    assert conn.rolled_back == 1


# ---------------------------------------------------------------------------
# F17 - read_db_settings must not abort the whole load on one unparseable value.
# ---------------------------------------------------------------------------


def test_read_db_settings_skips_unparseable_values_and_keeps_going(monkeypatch, caplog):
    rows = [("a", "12", "int"), ("b", "oops", "int"), ("c", True, "bool"), ("d", "1.5", "float"), ("e", "nope", "float")]
    monkeypatch.setattr(db, "safe_execute", lambda sql, params=None: rows)
    monkeypatch.setattr(db, "last_db_settings_read_at", None, raising=False)

    g = {}
    with caplog.at_level(logging.WARNING):
        db.read_db_settings(g)  # pre-fix: ValueError on 'oops', then AttributeError on the bool

    assert g["a"] == 12
    assert g["b"] == "oops"  # left untouched, as the docstring promises
    assert g["c"] is True
    assert g["d"] == 1.5
    assert g["e"] == "nope"
    assert db.last_db_settings_read_at is not None


# ---------------------------------------------------------------------------
# F18 - the bigint_degree partition branch must accept the date type its own
# signature declares.
# ---------------------------------------------------------------------------


def test_create_range_partitions_bigint_degree_accepts_date(monkeypatch):
    statements = []
    monkeypatch.setattr(db, "safe_execute", lambda sql, params=None: statements.append(sql))

    # pre-fix: AttributeError: 'datetime.date' object has no attribute 'utctimetuple'
    db.create_postgres_range_partitions("orders", date(2024, 1, 5), date(2024, 1, 6), "day", bigint_degree=3)

    assert len(statements) == 2
    expected_from = int(datetime(2024, 1, 5, tzinfo=timezone.utc).timestamp()) * 1000
    assert f"FROM ('{expected_from}')" in statements[0]


def test_create_range_partitions_bigint_degree_still_accepts_datetime(monkeypatch):
    statements = []
    monkeypatch.setattr(db, "safe_execute", lambda sql, params=None: statements.append(sql))

    db.create_postgres_range_partitions("orders", datetime(2024, 1, 5, tzinfo=timezone.utc), datetime(2024, 1, 5, tzinfo=timezone.utc), "day", bigint_degree=0)

    assert "PARTITION OF orders" in statements[0]


# ---------------------------------------------------------------------------
# F24 - the key value is bound through a placeholder, not spliced.
# ---------------------------------------------------------------------------


def test_get_id_by_key_field_binds_the_key_value(monkeypatch):
    captured = {}

    def fake_safe_execute(sql, data=None):
        captured["sql"] = sql
        captured["data"] = data
        return [(7,)]

    monkeypatch.setattr(db, "safe_execute", fake_safe_execute)

    enums = {}
    the_id = db.get_id_by_key_field_and_insert_if_needed(enums, "colors", "1),(2", key_is_not_string=True)

    assert the_id == 7
    # pre-fix: "values (1),(2)" -- the caller's value executed as SQL.
    assert "values (%s)" in captured["sql"]
    assert "1),(2" not in captured["sql"]
    assert captured["data"] == ("1),(2",)


def test_get_id_by_key_field_binds_the_key_value_with_alternate_fields(monkeypatch):
    captured = {}

    def fake_safe_execute(sql, data=None):
        captured["sql"] = sql
        captured["data"] = data
        return [(9,)]

    monkeypatch.setattr(db, "safe_execute", fake_safe_execute)

    the_id = db.get_id_by_key_field_and_insert_if_needed({}, "colors", "red'; drop table t--", alternate_fields_names="shade", alternate_fields_values="'dark'")

    assert the_id == 9
    assert "values (%s,'dark')" in captured["sql"]
    assert "drop table" not in captured["sql"]
    assert captured["data"] == ("red'; drop table t--",)


# ---------------------------------------------------------------------------
# F49 - read_table_into_dict must carry the trusted-input warning its twin has.
# ---------------------------------------------------------------------------


def test_read_table_into_dict_documents_the_raw_condition():
    doc = db.read_table_into_dict.__doc__ or ""
    assert "WARNING" in doc and "condition" in doc


# ---------------------------------------------------------------------------
# F50 - the existence probe must use the connection's configured schema.
# ---------------------------------------------------------------------------


def test_ensure_pg_table_exists_probes_the_configured_schema(monkeypatch):
    seen = {}

    def fake_check(table_name, schema_name="public"):
        seen["schema_name"] = schema_name
        return True

    monkeypatch.setattr(db, "check_if_pg_table_exists", fake_check)
    monkeypatch.setattr(db, "db_schema", "myschema", raising=False)

    db.ensure_pg_table_exists("enums")

    assert seen["schema_name"] == "myschema"  # pre-fix: "public"


def test_ensure_pg_table_exists_falls_back_to_public(monkeypatch):
    seen = {}
    monkeypatch.setattr(db, "check_if_pg_table_exists", lambda table_name, schema_name="public": seen.setdefault("schema_name", schema_name) or True)
    monkeypatch.setattr(db, "db_schema", None, raising=False)

    db.ensure_pg_table_exists("enums")

    assert seen["schema_name"] == "public"


# ---------------------------------------------------------------------------
# F51 / F52 - fetch_db_elements input handling.
# ---------------------------------------------------------------------------


class _Target:
    pass


def test_fetch_db_elements_star_without_a_description_reports_the_cause(monkeypatch):
    class _NoDescCur:
        description = None

    monkeypatch.setattr(db, "get_cursor", lambda *a, **k: _NoDescCur())

    # pre-fix: TypeError: 'NoneType' object is not iterable
    with pytest.raises(ValueError, match="description"):
        db.fetch_db_elements(_Target(), [(1,)], "*")


def test_fetch_db_elements_strips_whitespace_from_field_names():
    target = _Target()
    db.fetch_db_elements(target, [(1, "new")], "id, status", prefix="order_")

    assert target.order_id == 1
    assert target.order_status == "new"  # pre-fix: attribute was named "order_ status"
    assert not any(name.strip() != name for name in vars(target))


def test_fetch_db_elements_skips_non_identifier_fields_keeping_alignment(caplog):
    target = _Target()
    with caplog.at_level(logging.WARNING):
        db.fetch_db_elements(target, [(1, 42, "new")], "id, count(*), status")

    assert target.id == 1
    assert target.status == "new"  # the expression did not shift the remaining columns
    assert "count(*)" in caplog.text


# ---------------------------------------------------------------------------
# F53 / F54 - a non-positive retry budget must not be a silent no-op.
# ---------------------------------------------------------------------------


def test_basic_db_execute_rejects_non_positive_max_retries(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("must not reach the database")

    monkeypatch.setattr(db, "get_cursor", explode)

    with pytest.raises(ValueError, match="max_retries"):
        db.basic_db_execute("execute", "select 1", max_retries=0)


def test_execute_alchemy_rejects_non_positive_max_retries(monkeypatch):
    monkeypatch.setattr(db, "conn_alchemy", object(), raising=False)

    with pytest.raises(ValueError, match="max_retries"):
        db.execute_alchemy("create table t (x int)", max_retries=0)


# ---------------------------------------------------------------------------
# F55 - the named-cursor collision retry must actually close the colliding
# server-side cursor instead of popping a cache entry that is never populated.
# ---------------------------------------------------------------------------


def test_named_cursor_collision_closes_the_colliding_cursor(monkeypatch):
    executed = []

    class _PlainCur:
        description = None

        def execute(self, statement, data=None):
            executed.append(statement)

        def close(self):
            pass

    calls = {"n": 0}

    def fake_get_cursor(cursor_type=None, cursor_factory=None, cursor_name=None, itersize=None):
        if cursor_name is not None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception('cursor "page1" already exists')
            return _PlainCur()
        return _PlainCur()

    monkeypatch.setattr(db, "get_cursor", fake_get_cursor)
    monkeypatch.setattr(db, "sleep", lambda s: None)

    db.basic_db_execute("execute", "select 1", cursor_name="page1")

    # pre-fix: the recovery only popped a cache entry that get_cursor never writes for named
    # cursors, so nothing was closed and the retry re-declared the same colliding name.
    assert 'CLOSE "page1"' in executed


# ---------------------------------------------------------------------------
# F56 - rconnect's redis db argument is an index, validated before any state
# is touched.
# ---------------------------------------------------------------------------


def test_rconnect_rejects_a_non_numeric_db_without_touching_the_global(monkeypatch):
    redis = pytest.importorskip("redis")
    from pyutilz.database import redislib

    class _FakeRedis:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            raise AssertionError("the existing connection must not be disturbed")

    monkeypatch.setattr(redis, "Redis", _FakeRedis)
    existing = _FakeRedis()
    monkeypatch.setattr(redislib, "rc", existing)

    with pytest.raises(ValueError, match="redis_db_name"):
        redislib.rconnect("localhost", 6379, "sessions", "pwd")

    assert redislib.rc is existing


def test_rconnect_accepts_an_int_index(monkeypatch):
    redis = pytest.importorskip("redis")
    from pyutilz.database import redislib

    class _FakeRedis:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            pass

    monkeypatch.setattr(redis, "Redis", _FakeRedis)
    monkeypatch.setattr(redislib, "rc", None)

    rc = redislib.rconnect("localhost", 6379, 3, "pwd")
    try:
        assert rc.kwargs["db"] == 3
    finally:
        monkeypatch.setattr(redislib, "rc", None)
