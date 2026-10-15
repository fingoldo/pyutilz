"""Regression tests for audit finding X-08: alternate values in the insert builder are bound, not spliced."""

import pytest

import pyutilz.database.db as db


@pytest.fixture
def captured(monkeypatch):
    calls = []
    monkeypatch.setattr(db, "safe_execute", lambda stmt, params=None: calls.append((stmt, params)) or [(7,)])
    return calls


def test_string_fragment_is_refused_without_raw_sql(captured):
    with pytest.raises(TypeError, match="raw_sql=True"):
        db.get_id_by_key_field_and_insert_if_needed({}, "t", "k", alternate_fields_names="a", alternate_fields_values="'x'")
    assert captured == [], "nothing may reach the database when the fragment is refused"


def test_sequence_values_are_bound_as_parameters(captured):
    payload = "x'); drop table t; --"
    db.get_id_by_key_field_and_insert_if_needed({}, "t", "k", alternate_fields_names="a,b", alternate_fields_values=[payload, 3])
    stmt, params = captured[0]
    assert payload not in stmt, "a bound value must not appear in the SQL text"
    assert "values (%s,%s,%s)" in stmt
    assert params == ("k", payload, 3)


def test_alternate_only_branch_binds_values_without_the_key(captured):
    db.get_id_by_key_field_and_insert_if_needed({}, "t", "k", alternate_fields_names="a,b", alternate_fields_values=[1, 2], use_alternate_fields_only=True)
    stmt, params = captured[0]
    assert "values (%s,%s)" in stmt
    assert params == (1, 2)


def test_raw_sql_true_keeps_the_fragment_for_trusted_callers(captured):
    db.get_id_by_key_field_and_insert_if_needed({}, "t", "k", alternate_fields_names="a", alternate_fields_values="'x'", raw_sql=True)
    stmt, _ = captured[0]
    assert "values (%s,'x')" in stmt


def test_value_count_must_match_the_names(captured):
    with pytest.raises(ValueError, match="2 name"):
        db.get_id_by_key_field_and_insert_if_needed({}, "t", "k", alternate_fields_names="a,b", alternate_fields_values=[1])
    assert captured == []


def test_upsert_refuses_raw_fragments_without_raw_sql():
    from pyutilz.database.db.upsert import build_upsert_query

    with pytest.raises(TypeError, match="raw_sql=True"):
        build_upsert_query(["a"], "t", custom_onconflict="on conflict do nothing")
    with pytest.raises(TypeError, match="raw_sql=True"):
        build_upsert_query(["a", "b"], "t", conflict_fields=["a"], on_conflict_update_fields=["b"], on_conflict_update_values={"b": "excluded.b + 1"})


def test_upsert_builds_with_raw_sql_for_trusted_callers():
    from pyutilz.database.db.upsert import build_upsert_query

    sql = build_upsert_query(["a"], "t", custom_onconflict="on conflict do nothing", raw_sql=True)
    assert "on conflict do nothing" in sql
