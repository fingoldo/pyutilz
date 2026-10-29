"""Behaviour of ``pyutilz.database.db.upsert``: a bare string names one hash field while ``""`` and ``None`` name none,
the insert list adds the timestamp-check fields and drops the skipped ones, a conflict update takes an explicit value when
one is given and casts to the declared type otherwise, and the generated upsert names the conflict target and the updated
columns. An identifier that is not a plain SQL name is refused before any SQL is built."""

from __future__ import annotations

import pytest

from pyutilz.database.db import upsert
from pyutilz.database.exceptions import SQLValidationError


class TestNormalizeHashFields:
    def test_a_bare_string_names_one_field(self):
        assert upsert._normalize_hash_fields("h") == ["h"]

    def test_an_empty_string_or_none_names_no_fields(self):
        assert upsert._normalize_hash_fields("") == []
        assert upsert._normalize_hash_fields(None) == []

    def test_a_sequence_is_listed(self):
        assert upsert._normalize_hash_fields(("a", "b")) == ["a", "b"]


class TestActualInsertFields:
    def test_timestamp_check_fields_are_added_and_skipped_fields_removed(self):
        assert upsert._actual_insert_fields(["a", "b"], ["ts"], ["b"]) == ["a", "ts"]

    def test_without_timestamp_fields_the_data_fields_are_kept(self):
        fields = ["a", "b"]
        assert upsert._actual_insert_fields(fields, [], []) == ["a", "b"]
        assert fields == ["a", "b"]


class TestConflictUpdateValues:
    def test_an_explicit_value_is_used_as_given(self):
        assert upsert._conflict_update_values({}, ["c"], {"c": "c + 1"}) == ["c=c + 1"]

    def test_a_typed_field_is_cast_from_the_excluded_row(self):
        assert upsert._conflict_update_values({"a": "int"}, ["a"], {}) == ["a=excluded.a::int"]

    def test_an_untyped_field_is_taken_from_the_excluded_row(self):
        assert upsert._conflict_update_values({}, ["b"], {}) == ["b=excluded.b"]


class TestBuildUpsertQuery:
    def test_the_query_names_the_conflict_target_and_the_updated_columns(self):
        sql = upsert.build_upsert_query(["a", "b"], "t", conflict_fields=["a"], on_conflict_update_fields=["b"])
        assert "insert into t (a,b)" in sql
        assert "on conflict (a) do update set b=excluded.b" in sql

    def test_an_identifier_that_is_not_a_plain_name_is_refused(self):
        with pytest.raises(SQLValidationError, match="Invalid SQL identifier"):
            upsert.build_upsert_query(["a b"], "t")
