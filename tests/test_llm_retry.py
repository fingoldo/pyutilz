"""Tests for pyutilz.llm._retry."""

import logging
from unittest.mock import MagicMock

import pytest

pytest.importorskip("tenacity")

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


class TestIsRetryableHttpError:
    """Verify HTTP status classification: which errors trigger retry vs raise."""

    @staticmethod
    def _make_status_error(status_code: int):
        import httpx
        req = httpx.Request("POST", "https://api.example.com/v1/x")
        resp = httpx.Response(status_code=status_code, request=req)
        return httpx.HTTPStatusError(
            f"HTTP {status_code}", request=req, response=resp,
        )

    def test_402_is_retryable(self):
        """402 (insufficient balance — DeepSeek) must retry indefinitely."""
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        assert _is_retryable_http_error(self._make_status_error(402)) is True

    def test_429_is_retryable(self):
        """429 (rate limit) must retry."""
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        assert _is_retryable_http_error(self._make_status_error(429)) is True

    def test_500_is_retryable(self):
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        assert _is_retryable_http_error(self._make_status_error(500)) is True

    def test_503_is_retryable(self):
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        assert _is_retryable_http_error(self._make_status_error(503)) is True

    def test_400_is_not_retryable(self):
        """400 (bad request) is a permanent error — must raise, not retry."""
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        assert _is_retryable_http_error(self._make_status_error(400)) is False

    def test_401_is_not_retryable(self):
        """401 (auth) is permanent — retrying wastes time + may lock account."""
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        assert _is_retryable_http_error(self._make_status_error(401)) is False

    def test_403_is_not_retryable(self):
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        assert _is_retryable_http_error(self._make_status_error(403)) is False

    def test_transport_error_is_retryable(self):
        """Network-level errors (connection refused, timeout) must retry."""
        import httpx
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        req = httpx.Request("POST", "https://api.example.com/v1/x")
        assert _is_retryable_http_error(httpx.ConnectError("refused", request=req)) is True
        assert _is_retryable_http_error(httpx.ReadTimeout("timeout", request=req)) is True

    def test_value_error_is_not_retryable(self):
        """Unrelated exceptions (parse errors, etc.) must not be retried."""
        from pyutilz.llm.openai_compat import _is_retryable_http_error
        assert _is_retryable_http_error(ValueError("not http")) is False


class TestParseRetryAfter:
    """Verify Retry-After header parsing (used to honor server-provided backoff)."""

    @staticmethod
    def _make_resp(headers: dict):
        import httpx
        req = httpx.Request("POST", "https://api.example.com/v1/x")
        return httpx.Response(status_code=429, request=req, headers=headers)

    def test_retry_after_seconds(self):
        from pyutilz.llm.openai_compat import parse_retry_after
        assert parse_retry_after(self._make_resp({"Retry-After": "5"})) == 5.0

    def test_retry_after_ms(self):
        """Anthropic uses retry-after-ms; convert to seconds."""
        from pyutilz.llm.openai_compat import parse_retry_after
        assert parse_retry_after(self._make_resp({"retry-after-ms": "1500"})) == 1.5

    def test_ms_takes_precedence_over_seconds(self):
        """If both headers present, ms wins (more granular)."""
        from pyutilz.llm.openai_compat import parse_retry_after
        resp = self._make_resp({"retry-after-ms": "200", "Retry-After": "10"})
        assert parse_retry_after(resp) == 0.2

    def test_no_header_returns_none(self):
        from pyutilz.llm.openai_compat import parse_retry_after
        assert parse_retry_after(self._make_resp({})) is None

    def test_invalid_header_returns_none(self):
        """RFC-7231 also allows HTTP-date — we don't parse those, fall back to None."""
        from pyutilz.llm.openai_compat import parse_retry_after
        assert parse_retry_after(self._make_resp({"Retry-After": "Mon, 1 Jan 2027"})) is None

    def test_none_response_returns_none(self):
        from pyutilz.llm.openai_compat import parse_retry_after
        assert parse_retry_after(None) is None
