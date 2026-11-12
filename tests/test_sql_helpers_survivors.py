"""Exact error, warning and clause text pins for database.db.sql_helpers (mutation survivors)."""

from __future__ import annotations

import re

import pytest

from pyutilz.database.db import sql_helpers as sh
from pyutilz.database.exceptions import SQLValidationError


class TestIdentifierMessages:
    def test_an_invalid_identifier_error_names_the_value_and_the_rule(self):
        with pytest.raises(SQLValidationError) as info:
            sh.validate_sql_identifier("a-b")
        assert str(info.value) == "Invalid SQL identifier: 'a-b'. Must contain only alphanumeric and underscore, start with letter or underscore."

    def test_a_non_string_identifier_error_names_its_type(self):
        with pytest.raises(SQLValidationError) as info:
            sh.validate_sql_identifier(5)
        assert str(info.value) == "SQL identifier must be a string, got <class 'int'>"

    def test_a_qualified_non_string_identifier_error_names_its_type(self):
        with pytest.raises(SQLValidationError) as info:
            sh.validate_sql_qualified_identifier(5)
        assert str(info.value) == "SQL identifier must be a string, got <class 'int'>"

    def test_a_digit_leading_identifier_is_rejected_even_when_a_suffix_matches(self):
        with pytest.raises(SQLValidationError):
            sh.validate_sql_identifier("1abc")

    def test_a_trailing_newline_is_rejected(self):
        with pytest.raises(SQLValidationError):
            sh.validate_sql_identifier("users\n")


class TestQuoteMessage:
    def test_a_non_string_quote_error_names_its_type(self):
        with pytest.raises(TypeError) as info:
            sh.u(5)
        assert str(info.value) == "u() requires a str or None, got <class 'int'>"


class TestDeprecatedAlias:
    def test_the_deprecation_message_names_the_replacement(self):
        with pytest.warns(DeprecationWarning, match=re.escape("MakeSetExcludedClause is deprecated and will be removed in a future release; use make_set_excluded_clause instead.")):
            sh.MakeSetExcludedClause("a")

    def test_the_warning_points_at_the_caller(self):
        with pytest.warns(DeprecationWarning) as record:
            sh.MakeSetExcludedClause("a")
        assert record[0].filename == __file__

    def test_the_alias_returns_the_replacement_result(self):
        assert sh.MakeSetExcludedClause("a") == sh.make_set_excluded_clause(fields="a")


class TestUpdateClauseText:
    def test_the_now_branch_builds_a_do_update_block(self):
        assert sh.update_if_now("now", "x = 1") == "\n            do update set\n                x = 1\n        "

    def test_the_other_branch_does_nothing(self):
        assert sh.update_if_now("later", "x = 1") == "do nothing"
