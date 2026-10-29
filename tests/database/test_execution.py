"""Behaviour of ``pyutilz.database.db.execution``: a statement's field arguments are checked before it is built (a where or
set list is needed, the mode must be select, insert or update, and an update needs a non-empty where list), and the retry
budget must be at least one while ``auto_commit=False`` is reported as not honoured. Nothing here touches a database."""

from __future__ import annotations

import logging

import pytest

from pyutilz.database.db import execution


class TestCommandArgs:
    def test_an_insert_with_set_fields_is_valid(self):
        assert execution._db_command_args_valid("insert", None, ["a"]) is True

    def test_no_where_and_no_set_fields_is_refused(self):
        assert execution._db_command_args_valid("select", None, None) is False

    def test_an_unknown_mode_is_refused(self):
        assert execution._db_command_args_valid("merge", ["id"], None) is False

    def test_an_update_with_a_where_list_is_valid(self):
        assert execution._db_command_args_valid("update", ["id"], ["a"]) is True

    def test_an_update_with_an_empty_where_list_is_refused(self):
        assert execution._db_command_args_valid("update", [], ["a"]) is False


class TestRetryArgs:
    def test_a_zero_budget_is_refused(self):
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            execution._validate_retry_args(0, True)

    def test_a_positive_budget_with_autocommit_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            execution._validate_retry_args(1, True)
        assert caplog.records == []

    def test_turning_autocommit_off_is_reported_as_not_honoured(self, caplog):
        with caplog.at_level(logging.WARNING):
            execution._validate_retry_args(1, False)
        assert any("auto_commit=False is not honoured" in rec.getMessage() for rec in caplog.records)
