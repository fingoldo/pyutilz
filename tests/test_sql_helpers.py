"""Behaviour of ``pyutilz.database.db.sql_helpers``: identifiers that could carry SQL are refused before they reach a
query, string literals are quoted with embedded quotes doubled, the parameter templates match the statement mode,
the ON CONFLICT clause is built from validated names only, and the deprecated PascalCase alias still works and says
so. Nothing here touches a database."""

from __future__ import annotations

import warnings

import pytest

from pyutilz.database.db.sql_helpers import (
    MakeSetExcludedClause,
    SQLValidationError,
    construct_templates_and_values,
    make_set_excluded_clause,
    nu,
    u,
    update_if_now,
    validate_sql_identifier,
    validate_sql_qualified_identifier,
)


class TestIdentifiers:
    @pytest.mark.parametrize("name", ["items", "_private", "Col_2"])
    def test_plain_identifiers_are_accepted_and_returned(self, name):
        assert validate_sql_identifier(name) == name

    @pytest.mark.parametrize("name", ["items; DROP TABLE x", "a b", "1leading", "x-y", "a'b", ""])
    def test_anything_else_is_refused(self, name):
        with pytest.raises(SQLValidationError):
            validate_sql_identifier(name)

    def test_a_non_string_is_refused_with_its_type_named(self):
        with pytest.raises(SQLValidationError, match="must be a string"):
            validate_sql_identifier(None)

    def test_a_schema_qualified_name_checks_each_part(self):
        assert validate_sql_qualified_identifier("public.items") == "public.items"

    def test_one_bad_part_refuses_the_whole_qualified_name(self):
        with pytest.raises(SQLValidationError):
            validate_sql_qualified_identifier("public.items; --")


class TestLiterals:
    def test_a_string_is_quoted_and_embedded_quotes_are_doubled(self):
        assert u("O'Brien") == "'O''Brien'"

    def test_none_becomes_the_null_keyword(self):
        assert u(None) == "null"

    def test_a_non_string_is_refused(self):
        with pytest.raises(TypeError):
            u(42)  # type: ignore[arg-type]

    def test_an_empty_string_stays_a_quoted_empty_literal_under_u(self):
        assert u("") == "''"

    def test_nu_turns_an_empty_string_into_null_too(self):
        assert nu("") == "null"
        assert nu("x") == "'x'"

    def test_a_custom_quote_symbol_is_used_and_escaped(self):
        assert u('say "hi"', symb='"') == '"say ""hi"""'


class TestTemplates:
    def test_insert_mode_gives_positional_placeholders(self):
        values, templates = construct_templates_and_values("insert", ["a", "b"], {}, {"a": 1, "b": 2}, jsonize=False)
        assert values == [1, 2]
        assert templates == ["%s", "%s"]

    def test_update_mode_gives_field_equals_placeholders(self):
        _, templates = construct_templates_and_values("update", ["a"], {}, {"a": 1}, jsonize=False)
        assert templates == ["a=%s"]

    def test_select_mode_gives_field_equals_placeholders(self):
        _, templates = construct_templates_and_values("select", ["a"], {}, {"a": 1}, jsonize=False)
        assert templates == ["a=%s"]

    def test_replace_values_take_precedence_over_the_source(self):
        values, _ = construct_templates_and_values("insert", ["a"], {"a": "override"}, {"a": "source"}, jsonize=False)
        assert values == ["override"]

    def test_a_missing_key_in_the_source_becomes_none(self):
        values, _ = construct_templates_and_values("insert", ["a"], {}, {}, jsonize=False)
        assert values == [None]

    def test_jsonize_serialises_dicts_with_sorted_keys(self):
        values, _ = construct_templates_and_values("insert", ["d"], {}, {"d": {"b": 1, "a": 2}}, jsonize=True)
        assert values == ['{"a":2,"b":1}']

    def test_a_hostile_field_name_is_refused_before_any_template_is_built(self):
        with pytest.raises(SQLValidationError):
            construct_templates_and_values("insert", ["a; DROP"], {}, {}, jsonize=False)


class TestOnConflict:
    def test_each_field_maps_to_its_excluded_value(self):
        assert make_set_excluded_clause("a,b") == "a=excluded.a,b=excluded.b"

    def test_a_timestamp_column_is_appended_as_the_final_clause(self):
        assert make_set_excluded_clause("a", "updated_at") == "a=excluded.a,updated_at=(now() at time zone 'utc')"

    def test_a_hostile_field_name_is_refused(self):
        with pytest.raises(SQLValidationError):
            make_set_excluded_clause("a,b; --")

    def test_a_hostile_timestamp_column_is_refused(self):
        with pytest.raises(SQLValidationError):
            make_set_excluded_clause("a", "x; DROP")

    def test_the_deprecated_alias_warns_and_gives_the_same_result(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = MakeSetExcludedClause("a,b")
        assert result == make_set_excluded_clause("a,b")
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


class TestUpdateIf:
    def test_a_now_timestamp_applies_the_given_clause(self):
        assert "do update set" in update_if_now("now()", "a=excluded.a")
        assert "a=excluded.a" in update_if_now("now()", "a=excluded.a")

    def test_an_ordinary_timestamp_does_nothing_on_conflict(self):
        assert update_if_now("2020-01-01", "a=excluded.a").strip() == "do nothing"
