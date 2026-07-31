"""Tests for pyutilz.system.resilience."""

from __future__ import annotations

import pickle
import time

import pytest

from pyutilz.system.resilience import (
    CircuitBreaker,
    CircuitOpenError,
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

        time.sleep(0.01)  # recovery_timeout_sec=0, any elapsed time clears it

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
        restored = pickle.loads(pickle.dumps(cb))
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
        restored = pickle.loads(pickle.dumps(dlq))
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
