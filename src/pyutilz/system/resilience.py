"""Retry-with-backoff, circuit breaker, and dead-letter-queue primitives for calls to a flaky
external service (broker/exchange APIs, HTTP backends, anything with transient failure modes).

Ported from a downstream trading-bot project's own error-handling module -- fully generic once
decoupled from that project's own exception hierarchy (the only project-specific coupling was
reading ``BrokerError.http_code``; genericized here via ``http_code_attr``).
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

# ============================================================================
# Error Classification
# ============================================================================


class ErrorCategory(Enum):
    """Error classification for retry strategy."""

    TRANSIENT = "transient"  # Retry immediately (500, 502, 503)
    RATE_LIMIT = "rate_limit"  # Backoff required (429)
    NETWORK = "network"  # Connection issue, retry
    VALIDATION = "validation"  # Don't retry (400, 404)
    PERMANENT = "permanent"  # Don't retry (401, 403)


def classify_error(error: BaseException, http_code_attr: str = "http_code") -> ErrorCategory:
    """Classify ``error`` for a retry decision.

    Classification logic:
      - ``getattr(error, http_code_attr)`` 429 -> RATE_LIMIT
      - 500/502/503/504 -> TRANSIENT
      - 400/404/409 -> VALIDATION (don't retry)
      - 401/403 -> PERMANENT (reauth needed)
      - exception class name containing "Connection"/"Timeout", or message mentioning
        timeout/connection/rate limit -> NETWORK / RATE_LIMIT
      - anything else -> VALIDATION (the conservative default: don't retry an unrecognized error)

    ``http_code_attr`` lets callers point this at whatever attribute their own exception type
    uses for an HTTP status code (``http_code``, ``status_code``, ...); an error without that
    attribute (or without the attribute at all) falls through to the message-keyword checks.
    """
    http_code = getattr(error, http_code_attr, None)
    if http_code == 429:
        return ErrorCategory.RATE_LIMIT
    if http_code in (500, 502, 503, 504):
        return ErrorCategory.TRANSIENT
    if http_code in (400, 404, 409):
        return ErrorCategory.VALIDATION
    if http_code in (401, 403):
        return ErrorCategory.PERMANENT

    error_name = type(error).__name__
    if "Connection" in error_name or "Timeout" in error_name:
        return ErrorCategory.NETWORK

    error_msg = str(error).lower()
    if "timeout" in error_msg or "timed out" in error_msg:
        return ErrorCategory.NETWORK
    if "connection" in error_msg:
        return ErrorCategory.NETWORK
    if "rate limit" in error_msg:
        return ErrorCategory.RATE_LIMIT

    return ErrorCategory.VALIDATION


# ============================================================================
# Retry Decorator
# ============================================================================


def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter: float = 0.25,
    retryable_categories: Tuple[ErrorCategory, ...] = (
        ErrorCategory.TRANSIENT,
        ErrorCategory.NETWORK,
        ErrorCategory.RATE_LIMIT,
    ),
    classify: Callable[[BaseException], ErrorCategory] = classify_error,
) -> Callable[[_F], _F]:
    """Retry decorator with exponential backoff + jitter.

    Usage::

        @retry_with_backoff(max_retries=3, initial_backoff=1.0)
        def create_order(...):
            return broker_api.create_order(...)

    Backoff formula: ``min(initial * (multiplier ** attempt), max_backoff)``, then a random
    ``[0, jitter * backoff]`` amount is added (capped at ``max_backoff``) so many callers failing
    at once don't all retry on the same schedule and re-storm the flaky service.

    ``classify`` defaults to :func:`classify_error`; pass a project-specific classifier if the
    default HTTP-code/message heuristics don't fit.
    """

    def decorator(func: _F) -> _F:
        """Wrap ``func`` with the retry loop configured by the enclosing call."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Call ``func``, retrying on a retryable error category up to ``max_retries`` times."""
            last_exception: Optional[BaseException] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:  # noqa: PERF203 - per-iteration isolation is the point of a retry loop
                    last_exception = e
                    category = classify(e)

                    if category not in retryable_categories:
                        logger.error("%s failed with non-retryable error (%s): %s", func.__name__, category.value, e)
                        raise

                    if attempt >= max_retries:
                        logger.error("%s failed after %d retries: %s", func.__name__, max_retries, e)
                        break

                    backoff = min(initial_backoff * (backoff_multiplier**attempt), max_backoff)
                    if jitter > 0:
                        backoff = min(backoff + random.uniform(0, jitter * backoff), max_backoff)  # nosec B311 - jitter timing, not security-sensitive

                    logger.warning("%s failed (attempt %d/%d), retrying in %.1fs: %s", func.__name__, attempt + 1, max_retries + 1, backoff, e)
                    time.sleep(backoff)

            assert last_exception is not None  # loop only reaches here after catching an exception
            raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


# ============================================================================
# Circuit Breaker
# ============================================================================


class CircuitOpenError(RuntimeError):
    """Raised by a :class:`CircuitBreaker`-protected call while the circuit is OPEN."""


@dataclass
class CircuitState:
    """State of a circuit breaker."""

    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_successes: int = 0
    total_calls: int = 0
    total_failures: int = 0


class CircuitBreaker:
    """Circuit breaker pattern for a failing operation.

    States:
      - CLOSED: normal operation (failures < threshold)
      - OPEN: too many consecutive failures, reject immediately
      - HALF_OPEN: testing whether the service recovered

    Example::

        cb = CircuitBreaker("order_placement", failure_threshold=5)

        @cb.protect
        def create_order(...): ...
    """

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout_sec: int = 60, half_open_max_calls: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState()
        self._lock = threading.Lock()

    def __getstate__(self) -> Dict[str, Any]:
        # threading.Lock is unpicklable; drop it and rebuild on unpickle.
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def protect(self, func: _F) -> _F:
        """Decorator: wrap ``func`` so a tripped circuit raises :class:`CircuitOpenError` instead
        of attempting the call."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Call ``func`` unless the circuit is OPEN, tracking success/failure into the circuit's state."""
            with self._lock:
                if self.state.is_open:
                    if self._should_attempt_reset():
                        logger.info("Circuit %s entering HALF_OPEN state", self.name)
                        self.state.is_open = False
                        self.state.half_open_successes = 0
                    else:
                        raise CircuitOpenError(f"Circuit {self.name} is OPEN (too many failures). Will retry in {self._time_until_reset():.0f}s")

                self.state.total_calls += 1

            try:
                result = func(*args, **kwargs)

                with self._lock:
                    if not self.state.is_open and self.state.failure_count > 0:
                        self.state.half_open_successes += 1
                        if self.state.half_open_successes >= self.half_open_max_calls:
                            logger.info("Circuit %s CLOSED (recovered)", self.name)
                            self.state.failure_count = 0
                            self.state.half_open_successes = 0
                    else:
                        self.state.failure_count = 0

                return result

            except Exception:
                with self._lock:
                    self.state.failure_count += 1
                    self.state.total_failures += 1
                    self.state.last_failure_time = datetime.now(timezone.utc)

                    if self.state.failure_count >= self.failure_threshold:
                        self.state.is_open = True
                        logger.error("Circuit %s OPEN after %d consecutive failures", self.name, self.state.failure_count)

                raise

        return wrapper  # type: ignore[return-value]

    def _should_attempt_reset(self) -> bool:
        """True if enough time passed to attempt recovery."""
        if not self.state.last_failure_time:
            return True
        elapsed = datetime.now(timezone.utc) - self.state.last_failure_time
        return elapsed.total_seconds() >= self.recovery_timeout_sec

    def _time_until_reset(self) -> float:
        """Time remaining until a reset attempt (seconds)."""
        if not self.state.last_failure_time:
            return 0.0
        elapsed = datetime.now(timezone.utc) - self.state.last_failure_time
        remaining = self.recovery_timeout_sec - elapsed.total_seconds()
        return max(0.0, remaining)

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self.state = CircuitState()
            logger.info("Circuit %s manually reset", self.name)

    def get_stats(self) -> Dict[str, Any]:
        """Return circuit breaker statistics (is_open, failure_count, total_calls, ...)."""
        with self._lock:
            return {
                "name": self.name,
                "is_open": self.state.is_open,
                "failure_count": self.state.failure_count,
                "total_calls": self.state.total_calls,
                "total_failures": self.state.total_failures,
                "failure_rate": (self.state.total_failures / self.state.total_calls if self.state.total_calls > 0 else 0.0),
                "time_until_reset": self._time_until_reset() if self.state.is_open else 0.0,
            }


# ============================================================================
# Dead Letter Queue
# ============================================================================


class DeadLetterQueue:
    """Thread-safe bounded FIFO queue for operations that failed and could not be retried --
    useful for manual review/debugging, replay, and monitoring/alerting."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def __getstate__(self) -> Dict[str, Any]:
        # threading.Lock is unpicklable; drop it and rebuild on unpickle.
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def add(self, operation: str, args: Dict[str, Any], error: BaseException, context: Optional[Dict[str, Any]] = None) -> None:
        """Add a failed operation to the queue (oldest entries evicted once ``max_size`` is exceeded)."""
        with self._lock:
            entry = {
                "timestamp": datetime.now(timezone.utc),
                "operation": operation,
                "args": args,
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context or {},
            }
            self.queue.append(entry)
            if len(self.queue) > self.max_size:
                self.queue = self.queue[-self.max_size :]
            logger.error("Added to dead letter queue: %s - %s", operation, error, extra={"dlq_entry": entry})

    def get_all(self) -> List[Dict[str, Any]]:
        """Return every queued failure, newest first."""
        with self._lock:
            return list(reversed(self.queue))

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the ``n`` most recent failures, newest first."""
        with self._lock:
            return list(reversed(self.queue[-n:]))

    def clear(self) -> None:
        """Clear the queue."""
        with self._lock:
            self.queue.clear()
            logger.info("Dead letter queue cleared")

    def size(self) -> int:
        """Return the current queue size."""
        with self._lock:
            return len(self.queue)

    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics: size, per-operation counts, per-error-type counts, oldest/newest timestamps."""
        with self._lock:
            if not self.queue:
                return {"size": 0, "operations": {}, "error_types": {}}

            operations: Dict[str, int] = {}
            for entry in self.queue:
                op = entry["operation"]
                operations[op] = operations.get(op, 0) + 1

            error_types: Dict[str, int] = {}
            for entry in self.queue:
                err_type = entry["error_type"]
                error_types[err_type] = error_types.get(err_type, 0) + 1

            return {
                "size": len(self.queue),
                "operations": operations,
                "error_types": error_types,
                "oldest": self.queue[0]["timestamp"] if self.queue else None,
                "newest": self.queue[-1]["timestamp"] if self.queue else None,
            }
