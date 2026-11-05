"""Tests for pyutilz.system.resilience."""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from pyutilz.system import resilience

import pytest

from pyutilz.system.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    DeadLetterQueue,
    ErrorCategory,
    classify_error,
    retry_with_backoff,
)


class _HttpError(Exception):
    def __init__(self, msg, http_code=None):
        super().__init__(msg)
        self.http_code = http_code


class TestClassifyError:
    def test_429_is_rate_limit(self):
        assert classify_error(_HttpError("too many", http_code=429)) == ErrorCategory.RATE_LIMIT

    def test_503_is_transient(self):
        assert classify_error(_HttpError("down", http_code=503)) == ErrorCategory.TRANSIENT

    def test_404_is_validation(self):
        assert classify_error(_HttpError("missing", http_code=404)) == ErrorCategory.VALIDATION

    def test_401_is_permanent(self):
        assert classify_error(_HttpError("unauthorized", http_code=401)) == ErrorCategory.PERMANENT

    def test_connection_error_class_name_is_network(self):
        assert classify_error(ConnectionError("refused")) == ErrorCategory.NETWORK

    def test_timeout_message_is_network(self):
        assert classify_error(RuntimeError("request timed out")) == ErrorCategory.NETWORK

    def test_rate_limit_message_without_http_code(self):
        assert classify_error(RuntimeError("hit rate limit")) == ErrorCategory.RATE_LIMIT

    def test_unrecognized_error_defaults_to_validation(self):
        assert classify_error(ValueError("bad input")) == ErrorCategory.VALIDATION

    def test_custom_http_code_attr(self):
        class _StatusError(Exception):
            def __init__(self):
                self.status_code = 429

        assert classify_error(_StatusError(), http_code_attr="status_code") == ErrorCategory.RATE_LIMIT


class TestRetryWithBackoff:
    def test_succeeds_without_retry(self):
        calls = []

        @retry_with_backoff(max_retries=3, initial_backoff=0.001)
        def f():
            calls.append(1)
            return "ok"

        assert f() == "ok"
        assert len(calls) == 1

    def test_retries_transient_then_succeeds(self):
        attempts = {"n": 0}

        @retry_with_backoff(max_retries=3, initial_backoff=0.001, jitter=0)
        def f():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _HttpError("down", http_code=503)
            return "recovered"

        assert f() == "recovered"
        assert attempts["n"] == 3

    def test_exhausts_retries_and_raises_last_exception(self):
        @retry_with_backoff(max_retries=2, initial_backoff=0.001, jitter=0)
        def f():
            raise _HttpError("down", http_code=503)

        with pytest.raises(_HttpError, match="down"):
            f()

    def test_non_retryable_error_raises_immediately(self):
        attempts = {"n": 0}

        @retry_with_backoff(max_retries=5, initial_backoff=0.001)
        def f():
            attempts["n"] += 1
            raise _HttpError("bad request", http_code=400)

        with pytest.raises(_HttpError):
            f()
        assert attempts["n"] == 1

    def test_custom_classify_function(self):
        def classify_all_retryable(e):
            return ErrorCategory.TRANSIENT

        attempts = {"n": 0}

        @retry_with_backoff(max_retries=2, initial_backoff=0.001, jitter=0, classify=classify_all_retryable)
        def f():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ValueError("normally non-retryable")
            return "ok"

        assert f() == "ok"


class TestCircuitBreaker:
    def test_closed_circuit_allows_calls(self):
        cb = CircuitBreaker("test", failure_threshold=3)

        @cb.protect
        def f():
            return "ok"

        assert f() == "ok"
        assert cb.get_stats()["is_open"] is False

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout_sec=60)

        @cb.protect
        def f():
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                f()

        assert cb.get_stats()["is_open"] is True
        with pytest.raises(CircuitOpenError):
            f()

    def test_half_open_recovery_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout_sec=0, half_open_max_calls=1)

        @cb.protect
        def failing():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing()
        assert cb.get_stats()["is_open"] is True

        # recovery_timeout_sec=0, so the breaker is already past its recovery window

        @cb.protect
        def succeeding():
            return "recovered"

        assert succeeding() == "recovered"
        assert cb.get_stats()["is_open"] is False
        assert cb.get_stats()["failure_count"] == 0

    def test_manual_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1)

        @cb.protect
        def f():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            f()
        assert cb.get_stats()["is_open"] is True

        cb.reset()
        assert cb.get_stats()["is_open"] is False
        assert cb.get_stats()["failure_count"] == 0

    def test_picklable(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        restored = pickle.loads(pickle.dumps(cb))  # deserialize-ok: round-trips a pickle this test just produced in-process; the bytes never leave the test, no untrusted input
        assert restored.name == "test"
        assert restored.failure_threshold == 3

        @restored.protect
        def f():
            return "ok"

        assert f() == "ok"

    def test_get_stats_failure_rate(self):
        cb = CircuitBreaker("test", failure_threshold=10)

        @cb.protect
        def ok():
            return 1

        @cb.protect
        def bad():
            raise RuntimeError("x")

        ok()
        with pytest.raises(RuntimeError):
            bad()

        stats = cb.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_failures"] == 1
        assert stats["failure_rate"] == 0.5


class TestDeadLetterQueue:
    def test_add_and_get_all(self):
        dlq = DeadLetterQueue()
        dlq.add("create_order", {"symbol": "AAPL"}, ValueError("bad"), context={"account": "1"})
        entries = dlq.get_all()
        assert len(entries) == 1
        assert entries[0]["operation"] == "create_order"
        assert entries[0]["error_type"] == "ValueError"
        assert entries[0]["context"] == {"account": "1"}

    def test_newest_first_ordering(self):
        dlq = DeadLetterQueue()
        dlq.add("op1", {}, ValueError("a"))
        dlq.add("op2", {}, ValueError("b"))
        entries = dlq.get_all()
        assert entries[0]["operation"] == "op2"
        assert entries[1]["operation"] == "op1"

    def test_max_size_evicts_oldest(self):
        dlq = DeadLetterQueue(max_size=2)
        dlq.add("op1", {}, ValueError("a"))
        dlq.add("op2", {}, ValueError("b"))
        dlq.add("op3", {}, ValueError("c"))
        assert dlq.size() == 2
        ops = [e["operation"] for e in dlq.get_all()]
        assert ops == ["op3", "op2"]

    def test_get_recent(self):
        dlq = DeadLetterQueue()
        for i in range(5):
            dlq.add(f"op{i}", {}, ValueError("x"))
        recent = dlq.get_recent(2)
        assert [e["operation"] for e in recent] == ["op4", "op3"]

    def test_clear(self):
        dlq = DeadLetterQueue()
        dlq.add("op1", {}, ValueError("a"))
        dlq.clear()
        assert dlq.size() == 0

    def test_get_stats_empty(self):
        dlq = DeadLetterQueue()
        assert dlq.get_stats() == {"size": 0, "operations": {}, "error_types": {}}

    def test_picklable(self):
        dlq = DeadLetterQueue()
        dlq.add("op1", {}, ValueError("a"))
        restored = pickle.loads(pickle.dumps(dlq))  # deserialize-ok: round-trips a pickle this test just produced in-process; the bytes never leave the test, no untrusted input
        assert restored.size() == 1
        restored.add("op2", {}, ValueError("b"))
        assert restored.size() == 2

    def test_get_stats_counts_by_operation_and_error_type(self):
        dlq = DeadLetterQueue()
        dlq.add("op1", {}, ValueError("a"))
        dlq.add("op1", {}, ValueError("b"))
        dlq.add("op2", {}, TypeError("c"))
        stats = dlq.get_stats()
        assert stats["size"] == 3
        assert stats["operations"] == {"op1": 2, "op2": 1}
        assert stats["error_types"] == {"ValueError": 2, "TypeError": 1}


class TestSecondRoundBehaviour:
    @pytest.mark.parametrize("code", [500, 502, 503, 504])
    def test_each_server_error_status_is_transient(self, code):
        assert classify_error(_HttpError("down", http_code=code)) is ErrorCategory.TRANSIENT

    @pytest.mark.parametrize(
        "code, expected",
        [
            (400, ErrorCategory.VALIDATION),
            (404, ErrorCategory.VALIDATION),
            (409, ErrorCategory.VALIDATION),
            (401, ErrorCategory.PERMANENT),
            (403, ErrorCategory.PERMANENT),
            (429, ErrorCategory.RATE_LIMIT),
        ],
    )
    def test_each_status_keeps_its_category_even_with_a_network_message(self, code, expected):
        assert classify_error(_HttpError("connection reset", http_code=code)) is expected

    def test_an_unlisted_status_falls_through_to_the_message_checks(self):
        assert classify_error(_HttpError("boom", http_code=501)) is ErrorCategory.VALIDATION

    def test_category_values_are_their_lower_case_names(self):
        assert [c.value for c in ErrorCategory] == ["transient", "rate_limit", "network", "validation", "permanent"]

    def test_the_default_is_three_retries_after_the_first_call(self):
        calls = []

        @retry_with_backoff(jitter=0)
        def f():
            calls.append(1)
            raise _HttpError("down", http_code=503)

        with patch("time.sleep"), pytest.raises(_HttpError):
            f()
        assert len(calls) == 4

    def test_zero_retries_stops_after_the_first_failure(self):
        calls = []

        @retry_with_backoff(max_retries=0, jitter=0)
        def f():
            calls.append(1)
            raise _HttpError("down", http_code=503)

        with patch("time.sleep") as sleep, pytest.raises(_HttpError):
            f()
        assert len(calls) == 1
        sleep.assert_not_called()

    def test_the_default_backoff_doubles_from_one_second(self):
        sleeps = []

        @retry_with_backoff(max_retries=3, jitter=0)
        def f():
            raise _HttpError("down", http_code=503)

        with patch("time.sleep", side_effect=sleeps.append), pytest.raises(_HttpError):
            f()
        assert sleeps == [1.0, 2.0, 4.0]

    def test_the_backoff_multiplier_is_applied_per_attempt(self):
        sleeps = []

        @retry_with_backoff(max_retries=2, initial_backoff=2.0, backoff_multiplier=3.0, jitter=0)
        def f():
            raise _HttpError("down", http_code=503)

        with patch("time.sleep", side_effect=sleeps.append), pytest.raises(_HttpError):
            f()
        assert sleeps == [2.0, 6.0]

    def test_the_default_backoff_is_capped_at_sixty_seconds(self):
        sleeps = []

        @retry_with_backoff(max_retries=7, jitter=0)
        def f():
            raise _HttpError("down", http_code=503)

        with patch("time.sleep", side_effect=sleeps.append), pytest.raises(_HttpError):
            f()
        assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]

    def test_the_default_jitter_draws_up_to_a_quarter_of_the_backoff(self):
        draws = []

        @retry_with_backoff(max_retries=1)
        def f():
            raise _HttpError("down", http_code=503)

        def draw(lo, hi):
            draws.append((lo, hi))
            return 0.0

        with patch("random.uniform", side_effect=draw), patch("time.sleep"), pytest.raises(_HttpError):
            f()
        assert draws == [(0, 0.25)]

    def test_jitter_adds_the_drawn_amount_and_is_capped_at_the_maximum(self):
        sleeps = []

        @retry_with_backoff(max_retries=1, max_backoff=1.1)
        def f():
            raise _HttpError("down", http_code=503)

        with patch("random.uniform", side_effect=lambda lo, hi: hi), patch("time.sleep", side_effect=sleeps.append), pytest.raises(_HttpError):
            f()
        assert sleeps == [1.1]

    def test_without_jitter_no_amount_is_drawn(self):
        draws = []

        @retry_with_backoff(max_retries=1, jitter=0)
        def f():
            raise _HttpError("down", http_code=503)

        with patch("random.uniform", side_effect=lambda lo, hi: draws.append((lo, hi))), patch("time.sleep"), pytest.raises(_HttpError):
            f()
        assert draws == []

    def test_each_retry_is_logged_at_warning_with_its_attempt_and_delay(self, caplog):
        @retry_with_backoff(max_retries=2, jitter=0)
        def f():
            raise _HttpError("down", http_code=503)

        with patch("time.sleep"), caplog.at_level(logging.WARNING), pytest.raises(_HttpError):
            f()
        messages = [r.getMessage() for r in caplog.records]
        assert "f failed (attempt 1/3), retrying in 1.0s: down" in messages
        assert "f failed (attempt 2/3), retrying in 2.0s: down" in messages

    def test_the_final_failure_is_logged_at_error_with_the_retry_count(self, caplog):
        @retry_with_backoff(max_retries=2, jitter=0)
        def f():
            raise _HttpError("down", http_code=503)

        with patch("time.sleep"), caplog.at_level(logging.ERROR), pytest.raises(_HttpError):
            f()
        assert "f failed after 2 retries: down" in [r.getMessage() for r in caplog.records]

    def test_a_non_retryable_failure_is_logged_with_its_category(self, caplog):
        @retry_with_backoff()
        def f():
            raise _HttpError("nope", http_code=404)

        with patch("time.sleep") as sleep, caplog.at_level(logging.ERROR), pytest.raises(_HttpError):
            f()
        sleep.assert_not_called()
        assert "f failed with non-retryable error (validation): nope" in [r.getMessage() for r in caplog.records]

    def test_the_deprecated_alias_is_warned_about_by_name(self, caplog):
        with caplog.at_level(logging.WARNING):
            CircuitBreaker("svc", half_open_max_calls=5)
        assert "CircuitBreaker(svc): `half_open_max_calls` is a deprecated alias of `half_open_successes_to_close` -- it never capped concurrent admissions; please rename it." in [
            r.getMessage() for r in caplog.records
        ]

    def test_the_breaker_defaults_are_five_failures_sixty_seconds_and_three_successes(self):
        cb = CircuitBreaker("svc")
        assert (cb.failure_threshold, cb.recovery_timeout_sec, cb.half_open_successes_to_close) == (5, 60, 3)

    def test_the_breaker_opens_on_exactly_the_threshold(self):
        cb = CircuitBreaker("svc", failure_threshold=2)

        @cb.protect
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()
        assert cb.state.is_open is False
        with pytest.raises(ValueError):
            bad()
        assert cb.state.is_open is True

    def test_an_open_circuit_names_its_wait_in_seconds(self):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=60)

        @cb.protect
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()
        with pytest.raises(CircuitOpenError) as info:
            cb.protect(lambda: None)()
        assert str(info.value) == "Circuit svc is OPEN (too many failures). Will retry in 60s"

    def test_opening_is_logged_with_the_consecutive_failure_count(self, caplog):
        cb = CircuitBreaker("svc", failure_threshold=2)

        @cb.protect
        def bad():
            raise ValueError("boom")

        with caplog.at_level(logging.ERROR):
            for _ in range(2):
                with pytest.raises(ValueError):
                    bad()
        assert "Circuit svc OPEN after 2 consecutive failures" in [r.getMessage() for r in caplog.records]

    def test_a_half_open_trial_is_announced(self, caplog):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=0)

        @cb.protect
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()
        with caplog.at_level(logging.INFO):
            cb.protect(lambda: None)()
        assert "Circuit svc entering HALF_OPEN state" in [r.getMessage() for r in caplog.records]

    def test_half_open_needs_consecutive_successes_to_close(self):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=0, half_open_successes_to_close=3)

        @cb.protect
        def bad():
            raise ValueError("boom")

        ok = cb.protect(lambda: None)
        with pytest.raises(ValueError):
            bad()
        ok()
        ok()
        assert cb.state.is_half_open is True
        ok()
        assert cb.state.is_half_open is False
        assert cb.state.is_open is False

    def test_closing_after_recovery_is_logged(self, caplog):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=0, half_open_successes_to_close=1)

        @cb.protect
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()
        with caplog.at_level(logging.INFO):
            cb.protect(lambda: None)()
        assert "Circuit svc CLOSED (recovered)" in [r.getMessage() for r in caplog.records]

    def test_a_failed_probe_reopens_and_says_so(self, caplog):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=0)

        @cb.protect
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()
        with caplog.at_level(logging.ERROR), pytest.raises(ValueError):
            bad()
        assert cb.state.is_open is True
        assert "Circuit svc re-OPEN: probe call failed while HALF_OPEN" in [r.getMessage() for r in caplog.records]

    def test_a_success_clears_the_failure_run_while_closed(self):
        cb = CircuitBreaker("svc", failure_threshold=2)
        flip = {"fail": True}

        @cb.protect
        def sometimes():
            if flip["fail"]:
                raise ValueError("boom")
            return 1

        with pytest.raises(ValueError):
            sometimes()
        flip["fail"] = False
        assert sometimes() == 1
        flip["fail"] = True
        with pytest.raises(ValueError):
            sometimes()
        assert cb.state.is_open is False

    def test_a_manual_reset_is_logged(self, caplog):
        cb = CircuitBreaker("svc")
        with caplog.at_level(logging.INFO):
            cb.reset()
        assert "Circuit svc manually reset" in [r.getMessage() for r in caplog.records]

    def test_stats_have_exactly_the_documented_keys(self):
        stats = CircuitBreaker("svc").get_stats()
        assert set(stats) == {"name", "is_open", "failure_count", "total_calls", "total_failures", "failure_rate", "time_until_reset"}
        assert stats["name"] == "svc"

    def test_the_failure_rate_is_zero_before_any_call(self):
        assert CircuitBreaker("svc").get_stats()["failure_rate"] == 0.0

    def test_the_failure_rate_is_one_after_a_single_failed_call(self):
        cb = CircuitBreaker("svc", failure_threshold=5)

        @cb.protect
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()
        assert cb.get_stats()["failure_rate"] == 1.0

    def test_the_wait_is_reported_only_while_open(self):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=60)
        assert cb.get_stats()["time_until_reset"] == 0.0

        @cb.protect
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()
        assert cb.get_stats()["time_until_reset"] == pytest.approx(60, abs=1)

    def test_the_remaining_wait_shrinks_as_time_passes(self):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=60)
        cb._state.is_open = True
        cb._state.last_failure_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        assert cb.get_stats()["time_until_reset"] == pytest.approx(55, abs=0.5)

    def test_an_elapsed_window_reports_no_wait(self):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=0)
        cb._state.is_open = True
        cb._state.last_failure_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        assert cb.get_stats()["time_until_reset"] == 0.0

    def test_a_fresh_state_is_closed_with_no_counts(self):
        state = CircuitState()
        assert (state.is_open, state.is_half_open, state.half_open_successes, state.total_calls, state.failure_count) == (False, False, 0, 0, 0)

    def test_the_default_capacity_is_one_thousand(self):
        assert DeadLetterQueue().max_size == 1000

    def test_capacity_one_is_accepted_and_zero_is_refused_with_its_number(self):
        DeadLetterQueue(max_size=1)
        with pytest.raises(ValueError, match="DeadLetterQueue: max_size must be >= 1, got 0"):
            DeadLetterQueue(max_size=0)

    def test_an_added_failure_keeps_its_arguments_error_and_type(self):
        dlq = DeadLetterQueue()
        dlq.add("op", {"a": 1}, ValueError("bad"))
        entry = dlq.get_all()[0]
        assert entry["operation"] == "op"
        assert entry["args"] == {"a": 1}
        assert entry["error"] == "bad"
        assert entry["error_type"] == "ValueError"
        assert entry["context"] == {}

    def test_an_added_failure_is_logged_with_the_entry_attached(self, caplog):
        dlq = DeadLetterQueue()
        with caplog.at_level(logging.ERROR):
            dlq.add("op", {}, ValueError("bad"))
        record = [r for r in caplog.records if r.getMessage() == "Added to dead letter queue: op - bad"]
        assert len(record) == 1
        assert record[0].dlq_entry["operation"] == "op"

    def test_recent_returns_ten_by_default_newest_first(self):
        dlq = DeadLetterQueue()
        for i in range(11):
            dlq.add(f"op{i}", {}, ValueError("x"))
        recent = dlq.get_recent()
        assert len(recent) == 10
        assert recent[0]["operation"] == "op10"

    def test_recent_one_is_the_newest_entry(self):
        dlq = DeadLetterQueue()
        dlq.add("first", {}, ValueError("x"))
        dlq.add("second", {}, ValueError("x"))
        assert dlq.get_recent(1)[0]["operation"] == "second"

    def test_clearing_is_logged(self, caplog):
        dlq = DeadLetterQueue()
        with caplog.at_level(logging.INFO):
            dlq.clear()
        assert "Dead letter queue cleared" in [r.getMessage() for r in caplog.records]

    def test_a_single_entry_queue_reports_its_timestamp_as_both_oldest_and_newest(self):
        dlq = DeadLetterQueue()
        dlq.add("op", {}, ValueError("x"))
        stats = dlq.get_stats()
        timestamp = dlq.get_all()[0]["timestamp"]
        assert stats["oldest"] == timestamp
        assert stats["newest"] == timestamp

    def test_the_oldest_and_newest_timestamps_are_the_ends_of_the_queue(self):
        dlq = DeadLetterQueue()
        dlq.add("first", {}, ValueError("x"))
        dlq.add("second", {}, ValueError("x"))
        newest_first = dlq.get_all()
        stats = dlq.get_stats()
        assert stats["oldest"] == newest_first[1]["timestamp"]
        assert stats["newest"] == newest_first[0]["timestamp"]


class TestThirdRoundBehaviour:
    def test_keyword_arguments_reach_the_retried_function(self):
        @retry_with_backoff(jitter=0)
        def add(a, b=0):
            return a + b

        assert add(1, b=2) == 3

    def test_keyword_arguments_reach_the_protected_function(self):
        cb = CircuitBreaker("svc")

        @cb.protect
        def add(a, b=0):
            return a + b

        assert add(1, b=2) == 3

    def test_the_second_retry_draws_jitter_against_its_doubled_backoff(self):
        draws = []

        @retry_with_backoff(max_retries=2)
        def f():
            raise _HttpError("down", http_code=503)

        def draw(lo, hi):
            draws.append((lo, hi))
            return 0.0

        with patch("random.uniform", side_effect=draw), patch("time.sleep"), pytest.raises(_HttpError):
            f()
        assert draws == [(0, 0.25), (0, 0.5)]

    def test_total_failures_count_every_failed_call(self):
        cb = CircuitBreaker("svc", failure_threshold=10)

        @cb.protect
        def bad():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                bad()
        assert cb.get_stats()["total_failures"] == 2

    def test_a_failed_probe_leaves_the_circuit_open_but_not_half_open(self):
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=0)

        @cb.protect
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()
        with pytest.raises(ValueError):
            bad()
        assert cb.state.is_open is True
        assert cb.state.is_half_open is False

    def test_the_half_open_trial_is_allowed_at_exactly_the_timeout(self, monkeypatch):
        fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)

        class FixedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed

        monkeypatch.setattr(resilience, "datetime", FixedClock)
        cb = CircuitBreaker("svc", failure_threshold=1, recovery_timeout_sec=60)
        cb._state.is_open = True
        cb._state.last_failure_time = fixed - timedelta(seconds=60)
        assert cb._should_attempt_reset() is True
        cb._state.last_failure_time = fixed - timedelta(seconds=59)
        assert cb._should_attempt_reset() is False
