"""Regression tests for pyutilz.database.db / redislib / deltalakes (2026-07-21 audit fixes)."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from pyutilz.database.db.sql_helpers import (
    MakeSetExcludedClause,
    construct_templates_and_values,
    u,
    validate_sql_identifier,
)
from pyutilz.database.db.upsert import build_upsert_query
from pyutilz.database.db.sqlite import insert_sqllite_data


class TestConstructTemplatesUpdateMode:
    def test_update_mode_builds_real_templates(self):
        values, templates = construct_templates_and_values("update", ["name", "age"], {}, {"name": "bob", "age": 30}, jsonize=True)
        assert values == ["bob", 30]
        assert templates == ["name=%s", "age=%s"]  # previously [] -- the bug this regresses against


class TestMakeSetExcludedClauseInjection:
    def test_rejects_malicious_bAddUpdatedAtTimestamp(self):
        with pytest.raises(ValueError):
            MakeSetExcludedClause("name", "updated_at=now(); DROP TABLE users;--")

    def test_accepts_valid_identifier(self):
        result = MakeSetExcludedClause("name", "updated_at")
        assert result == "name=excluded.name,updated_at=(now() at time zone 'utc')"


class TestValidateSqlIdentifierTrailingNewline:
    def test_rejects_trailing_newline(self):
        # Regression test: $ (not \Z) matched before a trailing \n, silently accepting an
        # identifier the docstring promises is rejected.
        with pytest.raises(ValueError):
            validate_sql_identifier("users\n")

    def test_accepts_plain_identifier(self):
        assert validate_sql_identifier("users") == "users"


class TestUTypeCheck:
    def test_raises_on_non_string_non_none(self):
        with pytest.raises(TypeError):
            u(42)

    def test_none_returns_null(self):
        assert u(None) == "null"


class TestBuildUpsertQueryHistoryFieldsAliasing:
    def test_history_fields_list_not_mutated(self):
        caller_history_fields = ["id", "name"]
        build_upsert_query(
            fields_names=["id", "name", "md5hash"],
            table_name="mytable",
            conflict_fields=["id"],
            history_table_name="mytable_history",
            history_fields=caller_history_fields,
            hash_fields="md5hash",
        )
        # Regression test: previously mutated to ["id", "name", "md5hash"] via list aliasing.
        assert caller_history_fields == ["id", "name"]

    def test_hash_field_not_leaked_into_history_table_columns(self):
        query = build_upsert_query(
            fields_names=["id", "name", "md5hash"],
            table_name="mytable",
            conflict_fields=["id"],
            history_table_name="mytable_history",
            history_fields=["id", "name"],
            hash_fields="md5hash",
        )
        insert_clause = query[query.index("insert into mytable_history") : query.index("insert into mytable_history") + 60]
        assert "md5hash" not in insert_clause


class TestBuildUpsertQueryOnConflictTargeted:
    def test_conflict_fields_used_even_without_update_fields(self):
        query = build_upsert_query(
            fields_names=["id", "name"],
            table_name="mytable",
            conflict_fields=["id"],
        )
        # Regression test: previously produced a bare "on conflict do nothing" ignoring
        # conflict_fields entirely.
        assert "on conflict (id) do nothing" in query


class TestBuildUpsertQueryTestGapCoverage:
    """Test-gap-only findings (2026-07-21 audit round 2, HIGH/MEDIUM) -- these branches have no
    production-code bug, they were simply never exercised by any existing test."""

    def test_timestamp_update_fields_pairs_with_matching_check_field(self):
        """The history-table "update stale-check timestamp" path: timestamp_update_fields[i]
        must be paired with timestamp_check_fields[i] (via the zip() at upsert.py's
        `upd_fields_and_vals` construction), not misaligned."""
        query = build_upsert_query(
            fields_names=["id", "name"],
            table_name="mytable",
            conflict_fields=["id"],
            history_table_name="mytable_history",
            history_fields=["id", "name"],
            timestamp_check_fields=["checked_at"],
            timestamp_update_fields=["updated_at"],
        )
        assert "update mytable AS u set updated_at=c.checked_at" in query

    def test_multiple_timestamp_fields_pair_positionally(self):
        """Two timestamp fields: confirms position-i pairs with position-i, not a cross-wire."""
        query = build_upsert_query(
            fields_names=["id", "name"],
            table_name="mytable",
            conflict_fields=["id"],
            history_table_name="mytable_history",
            history_fields=["id", "name"],
            timestamp_check_fields=["checked_at", "verified_at"],
            timestamp_update_fields=["updated_at", "confirmed_at"],
        )
        assert "updated_at=c.checked_at" in query
        assert "confirmed_at=c.verified_at" in query

    def test_custom_onconflict_used_verbatim(self):
        """custom_onconflict is spliced in verbatim, bypassing the auto-generated
        ``on conflict (...) do update set ...`` construction entirely."""
        custom = "on conflict (id) do update set name = excluded.name || '_custom_marker'"
        query = build_upsert_query(
            fields_names=["id", "name"],
            table_name="mytable",
            conflict_fields=["id"],
            on_conflict_update_fields=["name"],  # would normally trigger auto-generation
            custom_onconflict=custom,
        )
        assert custom in query
        # The auto-generated form (no "_custom_marker") must NOT also appear.
        assert "excluded.name where" not in query
        assert query.count("on conflict") == 1

    def test_fields_types_cast_appears_in_insert_select_and_on_conflict_update(self):
        """fields_types casts must appear BOTH in the fresh-insert `select` list (`field::type`)
        AND in the `on conflict ... set` list (`field=excluded.field::type`)."""
        query = build_upsert_query(
            fields_names=["id", "amount"],
            table_name="mytable",
            conflict_fields=["id"],
            fields_types={"amount": "numeric"},
            on_conflict_update_fields=["amount"],
        )
        assert "select id,amount::numeric" in query
        assert "on conflict (id) do update set amount=excluded.amount::numeric" in query

    def test_skip_fields_excluded_from_insert_column_list(self):
        """A skipped field must not appear in the generated `insert into (...)` column list --
        even though it's still part of the `with data(...)` VALUES-source clause."""
        query = build_upsert_query(
            fields_names=["id", "name", "internal_flag"],
            table_name="mytable",
            conflict_fields=["id"],
            skip_fields=["internal_flag"],
        )
        insert_idx = query.index("insert into mytable (")
        insert_clause = query[insert_idx : insert_idx + 60]
        assert "internal_flag" not in insert_clause
        assert "id,name" in insert_clause


class TestInsertSqlliteData:
    def _make_db(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "t.db"))
        cur = conn.cursor()
        cur.execute("create table t (id integer, name text not null)")
        conn.commit()
        return conn, cur

    def test_returns_row_count_even_when_verbose_zero(self, tmp_path):
        conn, cur = self._make_db(tmp_path)
        result = insert_sqllite_data("t", [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], ["id", "name"], cur, conn, verbose=0)
        assert result == 2  # previously None

    def test_rolls_back_partial_insert_on_failure(self, tmp_path):
        conn, cur = self._make_db(tmp_path)
        result = insert_sqllite_data("t", [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": None}], ["id", "name"], cur, conn, verbose=0)
        assert result == 0
        # Previously rows 1/2 stayed pending in an open, uncommitted transaction.
        assert conn.in_transaction is False
        assert cur.execute("select count(*) from t").fetchone()[0] == 0

    def test_reserved_word_column_lowercase_group(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "t2.db"))
        cur = conn.cursor()
        cur.execute('create table t2 (id integer, "group" text)')
        conn.commit()
        # Regression test: only the exact string "GROUP" (uppercase) was quoted; lowercase
        # "group" (the natural spelling) previously produced a syntax error.
        result = insert_sqllite_data("t2", [{"id": 1, "group": "x"}], ["id", "group"], cur, conn, verbose=0)
        assert result == 1

    def test_one_shot_iterator_columns_does_not_silently_drop_rows(self, tmp_path):
        """Regression: `columns` is iterated 3 times (validate, placeholders, columns_str). A
        genuine one-shot iterator (which the `Iterable` type hint explicitly invites) was
        exhausted after the first pass, producing `INSERT INTO t () VALUES ()` -- a SQLite
        syntax error caught by the broad except and reported as "0 rows", indistinguishable
        from a real failure."""
        conn, cur = self._make_db(tmp_path)
        result = insert_sqllite_data("t", [{"id": 1, "name": "a"}], iter(["id", "name"]), cur, conn, verbose=0)
        assert result == 1
        assert cur.execute("select count(*) from t").fetchone()[0] == 1

    def test_failure_logs_traceback_not_just_message(self, tmp_path, caplog):
        """Regression: the failure branch used logger.error(e) (message only); switched to
        logger.exception so the full traceback is captured at effectively zero extra cost."""
        import logging

        conn, cur = self._make_db(tmp_path)
        with caplog.at_level(logging.ERROR, logger="pyutilz.database.db.sqlite"):
            insert_sqllite_data("t", [{"id": 1, "name": None}], ["id", "name"], cur, conn, verbose=0)
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR and "Could not insert" in r.message]
        assert error_records, "expected an ERROR log for the failed insert"
        assert error_records[0].exc_info is not None, "logger.exception must capture exc_info (traceback)"


class TestRexecuteAuthenticationError:
    def test_authentication_error_not_retried(self):
        from pyutilz.database import redislib

        redis = pytest.importorskip("redis")
        fake_rc = MagicMock()
        fake_rc.get.side_effect = redis.exceptions.AuthenticationError("wrong password")
        redislib.rc = fake_rc
        try:
            with patch("pyutilz.database.redislib.sleep") as mock_sleep:
                with pytest.raises(redis.exceptions.AuthenticationError):
                    redislib.rexecute("get", "key")
            mock_sleep.assert_not_called()  # must not retry
        finally:
            redislib.rc = None

    def test_max_retries_bounds_transient_connection_error(self):
        from pyutilz.database import redislib

        redis = pytest.importorskip("redis")
        fake_rc = MagicMock()
        fake_rc.get.side_effect = redis.exceptions.ConnectionError("blip")
        redislib.rc = fake_rc
        try:
            with patch("pyutilz.database.redislib.sleep"):
                with pytest.raises(redis.exceptions.ConnectionError):
                    redislib.rexecute("get", "key", max_retries=3)
            assert fake_rc.get.call_count == 3
        finally:
            redislib.rc = None

    def test_only_first_attempt_logs_full_traceback(self, caplog):
        """Regression: every retry attempt used to log a full ERROR-level traceback via
        logger.exception(e) -- with max_retries=None (retry forever, the documented default), a
        real outage produced roughly one traceback per second, forever. Only the first attempt
        of a streak should log the traceback; later attempts in the SAME streak log a one-line
        DEBUG summary instead."""
        import logging

        from pyutilz.database import redislib

        redis = pytest.importorskip("redis")
        fake_rc = MagicMock()
        fake_rc.get.side_effect = redis.exceptions.ConnectionError("blip")
        redislib.rc = fake_rc
        try:
            with patch("pyutilz.database.redislib.sleep"):
                with caplog.at_level(logging.DEBUG, logger="pyutilz.database.redislib"):
                    with pytest.raises(redis.exceptions.ConnectionError):
                        redislib.rexecute("get", "key", max_retries=3)
            error_records = [r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info is not None]
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG and "still failing" in r.message]
            assert len(error_records) == 1, "exactly one full-traceback ERROR log expected (the first attempt)"
            assert len(debug_records) == 2, "subsequent attempts within the same streak should log at DEBUG, not ERROR"
        finally:
            redislib.rc = None
