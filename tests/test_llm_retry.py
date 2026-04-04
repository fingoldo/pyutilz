"""Tests for pyutilz.llm._retry."""

import logging
from unittest.mock import MagicMock

import pytest

from pyutilz.llm._retry import (
    log_retry,
    RETRY_WAIT,
    INFINITE_RETRY_KWARGS,
)


class TestLogRetry:
    def _make_retry_state(self, attempt=1, exc=None, sleep=5.0):
        state = MagicMock()
        state.attempt_number = attempt
        state.outcome.exception.return_value = exc or ValueError("test")
        state.next_action.sleep = sleep
        return state

    def test_logs_warning(self, caplog):
        state = self._make_retry_state(attempt=3, exc=RuntimeError("oops"))
        with caplog.at_level(logging.WARNING):
            log_retry(state)
        assert "attempt 3" in caplog.text
        assert "RuntimeError" in caplog.text
        assert "oops" in caplog.text

    def test_truncates_long_message(self, caplog):
        state = self._make_retry_state(exc=ValueError("x" * 300))
        with caplog.at_level(logging.WARNING):
            log_retry(state)
        assert "x" * 200 in caplog.text
        assert "x" * 201 not in caplog.text

    def test_handles_none_next_action(self, caplog):
        state = self._make_retry_state()
        state.next_action = None
        with caplog.at_level(logging.WARNING):
            log_retry(state)
        assert "attempt 1" in caplog.text


class TestRetryConfiguration:
    def test_infinite_retry_kwargs_keys(self):
        assert "wait" in INFINITE_RETRY_KWARGS
        assert "stop" in INFINITE_RETRY_KWARGS
        assert "before_sleep" in INFINITE_RETRY_KWARGS
        assert INFINITE_RETRY_KWARGS["reraise"] is True

    def test_before_sleep_is_log_retry(self):
        assert INFINITE_RETRY_KWARGS["before_sleep"] is log_retry

    def test_retry_wait_is_not_none(self):
        assert RETRY_WAIT is not None
