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


class TestMaxRetryAttemptsEnvFallback:
    """Malformed PYUTILZ_LLM_MAX_RETRIES must fall back to the default (50), not crash."""

    def _reload_retry_module(self):
        import importlib
        import pyutilz.llm._retry as retry_module

        return importlib.reload(retry_module)

    def test_invalid_env_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_LLM_MAX_RETRIES", "not-a-number")
        retry_module = self._reload_retry_module()
        assert retry_module.MAX_RETRY_ATTEMPTS == 50

    def test_valid_env_value_is_respected(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_LLM_MAX_RETRIES", "7")
        retry_module = self._reload_retry_module()
        assert retry_module.MAX_RETRY_ATTEMPTS == 7

    def test_missing_env_value_defaults_to_50(self, monkeypatch):
        monkeypatch.delenv("PYUTILZ_LLM_MAX_RETRIES", raising=False)
        retry_module = self._reload_retry_module()
        assert retry_module.MAX_RETRY_ATTEMPTS == 50


class TestRetryConfiguration:
    def test_infinite_retry_kwargs_keys(self):
        assert "wait" in INFINITE_RETRY_KWARGS
        assert "stop" in INFINITE_RETRY_KWARGS
        assert "before_sleep" in INFINITE_RETRY_KWARGS
        assert INFINITE_RETRY_KWARGS["reraise"] is True

    def test_before_sleep_is_log_retry(self):
        assert INFINITE_RETRY_KWARGS["before_sleep"] is log_retry

    def test_retry_wait_is_not_none(self):
        from tenacity.wait import wait_combine
        assert isinstance(RETRY_WAIT, wait_combine)


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


class TestBillingPausesGetAGraceWindow:
    """HTTP 402 is retryable on purpose, but not for hours.

    Measured against a live OpenRouter account with an empty balance: the shared policy retried a
    402 fifty times with waits of 5, 10, 20, 40, 80, 160 and then 300s each -- about 3.8 hours of
    wall clock for ONE call. A consumer running five concurrent workers wedges all five for most of
    a working day, on a condition no amount of waiting resolves.

    Keeping it retryable is right: a balance topped up mid-batch should resume rather than drop the
    batch. Only the tail is wrong -- the first six attempts already span five minutes, and if the
    balance was going to be restored it was restored long before attempt seven.
    """

    @staticmethod
    def _state(status_code, elapsed, attempt=1):
        class _Response:
            def __init__(self, code):
                self.status_code = code

        class _HttpError(Exception):
            def __init__(self, code):
                self.response = _Response(code)

        state = MagicMock()
        state.outcome.exception.return_value = _HttpError(status_code)
        state.seconds_since_start = elapsed
        state.attempt_number = attempt
        return state

    def test_a_402_inside_the_window_keeps_retrying(self):
        """The half of the old behaviour that was right: a top-up arriving in the first few minutes
        must resume the call, not find it already abandoned."""
        from pyutilz.llm._retry import _STOP, BILLING_GRACE_SECONDS

        assert _STOP(self._state(402, elapsed=BILLING_GRACE_SECONDS - 1, attempt=6)) is False

    def test_a_402_past_the_window_gives_up(self):
        from pyutilz.llm._retry import _STOP, BILLING_GRACE_SECONDS

        assert _STOP(self._state(402, elapsed=BILLING_GRACE_SECONDS + 1, attempt=7)) is True

    def test_the_old_behaviour_would_still_be_retrying_an_hour_later(self):
        """Pins the size of the problem, not just its direction. An hour into a billing pause the
        previous policy had 30+ attempts left."""
        from pyutilz.llm._retry import _STOP

        assert _STOP(self._state(402, elapsed=3600, attempt=20)) is True

    @pytest.mark.parametrize("status_code", [429, 500, 502, 503])
    def test_other_retryable_statuses_keep_the_full_schedule(self, status_code):
        """The counterweight, and the reason this is a window rather than a reclassification. A
        rate-limit window or a provider outage genuinely can outlast five minutes, and the long
        schedule exists for exactly those -- cutting them short would drop work that would have
        succeeded."""
        from pyutilz.llm._retry import _STOP

        assert _STOP(self._state(status_code, elapsed=3600, attempt=20)) is False

    def test_the_global_attempt_cap_still_applies_to_everything(self):
        """The window only ever makes the policy stop EARLIER. A 429 that exhausts the attempt cap
        must still stop, or this change would have turned a bounded retry into an unbounded one."""
        from pyutilz.llm._retry import _STOP, MAX_RETRY_ATTEMPTS

        if MAX_RETRY_ATTEMPTS == 0:
            pytest.skip("attempts are configured as infinite; there is no cap to assert")
        assert _STOP(self._state(429, elapsed=10, attempt=MAX_RETRY_ATTEMPTS + 1)) is True

    def test_a_non_http_exception_is_not_mistaken_for_a_billing_pause(self):
        """The predicate is duck-typed through `getattr` so this module keeps no httpx dependency.
        That makes it worth asserting that an ordinary exception -- no `.response` at all -- does
        not accidentally match."""
        from pyutilz.llm._retry import _is_billing_pause

        assert _is_billing_pause(ValueError("nothing to do with HTTP")) is False
        assert _is_billing_pause(None) is False

    def test_the_grace_window_is_configurable_and_defaults_sanely(self):
        from pyutilz.llm._retry import BILLING_GRACE_SECONDS

        assert BILLING_GRACE_SECONDS > 0
        assert BILLING_GRACE_SECONDS <= 3600, "a grace window longer than an hour is the tail this fix removed"
