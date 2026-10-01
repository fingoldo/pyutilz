"""Dead-man's-switch job monitoring (healthchecks.io / cronitor.io heartbeats) plus timeout and duration-logging decorators."""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


import time
import atexit
import random
import functools
import threading
import concurrent.futures
from functools import wraps
from datetime import datetime
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# requests lives under pyutilz's optional [web] extra -- a plain module-level `import requests`
# forced ANY use of pyutilz.system.monitoring (even functions that never touch job-completion
# heartbeats) to have it installed. Guarded like this file's other optional deps instead of a
# lazy per-call import: tests/test_monitoring_extra.py patches `pyutilz.system.monitoring.requests`
# directly, which needs a real module-level attribute to target.
requests = None
try:
    import requests as _requests
    requests = _requests
except Exception as e:  # nosec B110 - optional dependency probe; job_completed already fails loudly later if requests is actually needed but unset
    logger.debug("requests unavailable, job_completed's heartbeat send will fail if used: %s", e)

# ----------------------------------------------------------------------------------------------------------------------------
# INITS
# ----------------------------------------------------------------------------------------------------------------------------

API_TIMEOUT_SEC = 15

# Shared pool for job_completed's bounded (requests-timeout-capped) fire-and-forget heartbeat
# sends. NOT used by timeout_wrapper -- see that function's docstring for why a bounded shared
# pool is unsafe for wrapping arbitrary, potentially-unbounded caller functions.
_TIMEOUT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="timeout_wrapper")

# Shut the shared executor down at interpreter exit so its worker threads don't leak.
atexit.register(_TIMEOUT_EXECUTOR.shutdown, wait=False)

# ----------------------------------------------------------------------------------------------------------------------------
# HEARTBEAT DELIVERY: BACKGROUND RETRY
# ----------------------------------------------------------------------------------------------------------------------------
#
# A heartbeat that fails on a transient network error used to be logged once and forgotten. On a long-lived
# process that is the worst moment to forget it: the job HAS finished, the monitor is waiting for exactly this
# ping, and the next regular one may be hours away -- so one DNS blip ("Failed to resolve 'cronitor.link'")
# turned into a "Missed Event" alert for a healthy scraper. The failed send is now handed to a background
# thread that retries with exponential backoff for as long as the process lives (bounded by RETRY_MAX_AGE_SEC).

#: Delay before the first retry; doubles on every further failure of the same heartbeat.
RETRY_INITIAL_DELAY_SEC = 5.0
#: Ceiling of the backoff. Five minutes keeps a long outage cheap and still delivers within minutes of recovery.
RETRY_MAX_DELAY_SEC = 300.0
#: A heartbeat still undelivered after this long is dropped with one warning: the job's next regular ping is due
#: well before it, and a six-hour-old "complete" would only mislead the monitor's timeline.
RETRY_MAX_AGE_SEC = 6 * 3600.0
#: Best-effort budget for one last delivery pass at interpreter exit (see ``_HeartbeatRetrier.flush``).
EXIT_FLUSH_BUDGET_SEC = 10.0

#: Process-wide switch. ``job_completed(..., retry=False)`` opts one call out; tests switch the whole mechanism off
#: so no background thread outlives them.
_RETRY_ENABLED = True

#: HTTP statuses worth another attempt: the request timed out, was throttled, or the provider failed. Everything else
#: (200, 403 "blocked in your country", 404 unknown monitor, ...) is a verdict a retry cannot change.
_TRANSIENT_STATUSES = frozenset({408, 425, 429}) | frozenset(range(500, 600))

#: The body of a heartbeat: the dict ``monitored`` builds, or a plain string.
_HeartbeatData = Optional[Union[Dict[str, Any], str]]

#: One heartbeat stream: (endpoint, state). A newer ping for the same stream makes every older pending one obsolete.
_HeartbeatKey = Tuple[str, str]


def _backoff_delay(failures: int) -> float:
    """Seconds to wait after the *failures*-th consecutive failure: INITIAL, 2x, 4x, ... capped, with +-20% jitter."""
    base = min(RETRY_MAX_DELAY_SEC, RETRY_INITIAL_DELAY_SEC * (2 ** max(0, failures - 1)))
    return float(base * random.uniform(0.8, 1.2))  # nosec B311 - jitter for load spreading, not a security decision


def _attempt(
    endpoint: str,
    data: _HeartbeatData,
    params: Optional[Dict[str, Any]],
    provider: str,
    job_id: str,
    timeout: float = API_TIMEOUT_SEC,
) -> bool:
    """One POST of a heartbeat. True when there is nothing left to retry, False on a transient failure.

    Logging is unchanged from the single-shot sender this replaced: one warning on a non-OK status or a request
    error. 200, 403 (blocked in your country) and 429 (rate limited) stay silent; of those only 429 is retried.
    """
    try:
        if requests is None:
            raise ImportError("job_completed's heartbeat send requires requests, which failed to import (see earlier debug log for the reason)")
        res = requests.post(endpoint, data=data, params=params, timeout=timeout)

        if res.status_code not in (200, 403, 429):
            # 403=blocked in your country
            # 429=rate limit exceeded
            logger.warning("Problem %s while sending heartbeat to %s on job %s: %s", res.status_code, provider, job_id, res.text)
        return bool(res.status_code not in _TRANSIENT_STATUSES)
    except ImportError as e:
        logger.warning("Error while sending heartbeat to %s on monitor %s: %s", provider, job_id, e)
        return True  # no retry can install the missing package
    except Exception as e:
        logger.warning("Error while sending heartbeat to %s on monitor %s: %s", provider, job_id, e)
        return False


class _Pending:
    """A heartbeat awaiting redelivery, with its own backoff clock."""

    __slots__ = ("data", "endpoint", "failures", "first_failed_at", "job_id", "key", "next_at", "params", "provider")

    def __init__(
        self,
        key: _HeartbeatKey,
        endpoint: str,
        data: _HeartbeatData,
        params: Optional[Dict[str, Any]],
        provider: str,
        job_id: str,
    ) -> None:
        self.key = key
        self.endpoint = endpoint
        # Copied: the caller's dict (``monitored`` builds one per call) may be mutated after we return.
        self.data: _HeartbeatData = dict(data) if isinstance(data, dict) else data
        self.params: Optional[Dict[str, Any]] = dict(params) if params else params
        self.provider = provider
        self.job_id = job_id
        now = time.monotonic()
        self.first_failed_at = now
        self.failures = 1
        self.next_at = now + _backoff_delay(1)


class _HeartbeatRetrier:
    """Redelivers failed heartbeats from one daemon thread, which exists only while something is pending.

    At most one pending heartbeat per stream (see ``_HeartbeatKey``): a newer failure replaces the older one, and a
    successful send on a stream cancels whatever is still pending on it, so the queue is bounded by the number of
    distinct monitors in the process and a flapping network cannot grow it.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._pending: Dict[_HeartbeatKey, _Pending] = {}
        self._thread: Optional[threading.Thread] = None

    def __getstate__(self) -> Dict[str, Any]:
        """A retrier owns a lock and a live thread; pickling one is always a mistake, so say so."""
        raise TypeError("_HeartbeatRetrier holds a lock and a thread and cannot be pickled")

    def submit(self, item: _Pending) -> None:
        """Queue *item* for redelivery, replacing an older pending heartbeat on the same stream."""
        with self._cv:
            self._pending[item.key] = item
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="heartbeat-retry", daemon=True)
                self._thread.start()
            self._cv.notify_all()
        logger.info("Heartbeat to %s on monitor %s queued for background retry (first retry in %.0fs)", item.provider, item.job_id, item.next_at - item.first_failed_at)

    def forget(self, key: _HeartbeatKey) -> None:
        """A heartbeat on *key* was delivered (or refused for good): older pending ones on it are obsolete."""
        with self._cv:
            if self._pending.pop(key, None) is not None:
                self._cv.notify_all()

    @staticmethod
    def _log_gave_up(item: _Pending, now: float) -> None:
        """The one warning a dropped heartbeat gets (each is dropped exactly once, so there is nothing to throttle)."""
        logger.warning(
            "Giving up on heartbeat to %s on monitor %s after %d failed attempts over %.0f min",
            item.provider,
            item.job_id,
            item.failures,
            (now - item.first_failed_at) / 60,
        )

    def _next_due(self, now: float) -> Tuple[List[_Pending], Optional[float]]:
        """(pending items due now, seconds until the next one is due or None when none is waiting); drops expired ones."""
        due: List[_Pending] = []
        soonest: Optional[float] = None
        for key, item in list(self._pending.items()):
            if now - item.first_failed_at > RETRY_MAX_AGE_SEC:
                del self._pending[key]
                self._log_gave_up(item, now)
            elif item.next_at <= now:
                due.append(item)
            else:
                wait = item.next_at - now
                soonest = wait if soonest is None else min(soonest, wait)
        return due, soonest

    def _run(self) -> None:
        """Retry loop; returns (and lets ``submit`` start a fresh thread) once nothing is pending."""
        while True:
            with self._cv:
                while True:
                    due, soonest = self._next_due(time.monotonic())
                    if due:
                        break
                    if not self._pending:
                        self._thread = None
                        self._cv.notify_all()
                        return
                    self._cv.wait(timeout=soonest)
            for item in due:
                delivered = _attempt(item.endpoint, item.data, item.params, item.provider, item.job_id)
                with self._cv:
                    if self._pending.get(item.key) is not item:
                        continue  # replaced or cancelled while the request was in flight
                    if delivered:
                        del self._pending[item.key]
                        logger.info(
                            "Heartbeat to %s on monitor %s delivered on attempt %d after %.0fs",
                            item.provider,
                            item.job_id,
                            item.failures + 1,
                            time.monotonic() - item.first_failed_at,
                        )
                    else:
                        item.failures += 1
                        item.next_at = time.monotonic() + _backoff_delay(item.failures)
            with self._cv:
                self._cv.notify_all()

    def flush(self, budget: float = EXIT_FLUSH_BUDGET_SEC) -> None:
        """One last delivery attempt per pending heartbeat, within *budget* seconds. Registered ``atexit``.

        A one-shot job that finishes, hits a network blip on its final ping and exits would otherwise lose it: the
        retry thread is a daemon and dies with the interpreter. Never raises, never blocks past the budget.
        """
        deadline = time.monotonic() + budget
        try:
            with self._cv:
                items = list(self._pending.values())
            for item in items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if _attempt(item.endpoint, item.data, item.params, item.provider, item.job_id, timeout=min(API_TIMEOUT_SEC, remaining)):
                    self.forget(item.key)
        except Exception as e:  # nosec B110 - interpreter shutdown: nothing useful left to do with a failure here
            logger.debug("heartbeat flush at exit failed: %s", e)

    def wait_idle(self, timeout: float) -> bool:
        """Block until nothing is pending, or *timeout* seconds pass. True when idle (used by tests and diagnostics)."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
        return True

    def pending_count(self) -> int:
        """Number of heartbeats currently awaiting redelivery."""
        with self._cv:
            return len(self._pending)

    def clear(self) -> None:
        """Drop everything pending and let the thread wind down."""
        with self._cv:
            self._pending.clear()
            self._cv.notify_all()


_RETRIER = _HeartbeatRetrier()

# Registered after the executor's shutdown above, so it runs BEFORE it (atexit is LIFO).
atexit.register(_RETRIER.flush)

# ----------------------------------------------------------------------------------------------------------------------------
# 3RD PARTIES MONITORING
# ----------------------------------------------------------------------------------------------------------------------------


def job_completed(
    job_id: str,
    status: int = 0,
    data: _HeartbeatData = None,
    provider: str = "healthchecks.io",
    api_key: Optional[str] = None,
    blocking: bool = True,
    retry: bool = True,
) -> None:
    """Ping a dead-man's-switch monitoring provider (healthchecks.io / cronitor.io) that a job completed.

    ``blocking=True`` (default) sends the heartbeat inline and returns only after the request
    finishes (or times out after ``API_TIMEOUT_SEC``) -- preserves the historical synchronous
    contract every existing caller/test relies on. Pass ``blocking=False`` to submit the send to
    the module's shared ``_TIMEOUT_EXECUTOR`` and return immediately without waiting on the
    network round-trip; the request still fires (and still respects the same timeout + error
    logging), just off the calling thread. Fire-and-forget is not free of risk: a process that
    exits immediately after calling with ``blocking=False`` can beat the background send to the
    wire and the heartbeat is lost -- use ``blocking=True`` (the default) for a last-call-before-exit
    heartbeat, and ``blocking=False`` only when the caller keeps running long enough to let the
    executor's worker thread finish.

    ``data`` is a dict (the shape the ``monitored`` decorator builds and passes, and the form
    ``requests`` encodes as a form body) or a plain string; both are stringified for cronitor.io's
    ``msg`` param and passed through as the POST body for healthchecks.io.

    ``retry=True`` (default): a send that fails transiently -- a network error, a timeout, a 408/425/429 or a 5xx --
    is handed to a background thread that retries with exponential backoff (``RETRY_INITIAL_DELAY_SEC`` doubling up
    to ``RETRY_MAX_DELAY_SEC``) while the process lives, so a DNS blip at the moment a long-lived job reports in no
    longer turns into a "missed event" alert. This call still returns after the FIRST attempt. A newer heartbeat on the
    same monitor and state replaces an older pending one, and any successful send cancels it. A refusal a retry cannot
    change (403, 404, ...) is not retried. Pass ``retry=False`` for the old single-shot behaviour.
    """

    endpoint = ""
    params = None

    if provider == "healthchecks.io":
        if data:
            data = str(data)
        if api_key:
            endpoint = f"https://hc-ping.com/{api_key}/{job_id}/{status}"
        else:
            endpoint = f"https://hc-ping.com/{job_id}/{status}"
    elif provider == "cronitor.io":
        endpoint = f"https://cronitor.link/p/{api_key}/{job_id}"
        state: Any
        if status == 0:
            state = "complete"
        else:
            state = status

        params = dict(state=state)
        if data:
            params["msg"] = str(data)

    if endpoint:

        key: _HeartbeatKey = (endpoint, str(params.get("state")) if params else "")

        def _send() -> None:
            """Post the heartbeat, logging (not raising) on a non-OK status or request error; queue a transient failure for retry."""
            if _attempt(endpoint, data, params, provider, job_id):
                _RETRIER.forget(key)  # delivered, or refused for good: an older pending ping on this stream is obsolete
            elif retry and _RETRY_ENABLED:
                _RETRIER.submit(_Pending(key, endpoint, data, params, provider, job_id))

        if blocking:
            _send()
        else:
            _TIMEOUT_EXECUTOR.submit(_send)
    else:
        logger.info("No endpoint established for job %s. Check if monitoring credentials are properly configured.", job_id)


def monitored(
    job_id: Optional[str] = None,
    status: int = 0,
    log_data: bool = True,
    should_have_data: bool = False,
    duration_field: str = "duration",
    duration_rounding: int = 4,
    provider: str = "healthchecks.io",
    api_key: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory that runs the wrapped function, times it, logs and reports its dict result as a job_completed heartbeat.

    The wrapped function must return either ``None`` or a ``dict``. When ``duration_field`` is set,
    the elapsed wall-clock time is added to that key of the result dict before logging/reporting.

    ``should_have_data=True`` means a FALSY result (``None``, ``{}``) is treated as a failed run:
    the wrapper returns immediately and NO heartbeat is sent, so a dead-man's-switch monitor fires.
    Do not set it on a job that can legitimately produce an empty result -- a nightly job returning
    ``{}`` on a quiet day would raise a false alert.
    """
    def decorator_logged(func: Callable[..., Any]) -> Callable[..., Any]:
        """Wraps func so each call is timed, logged, and reported via job_completed."""
        @functools.wraps(func)
        def wrapper_logged(*args: Any, **kwargs: Any) -> Any:
            """Calls func, augments its dict result with duration, logs it, and pings the monitoring provider."""

            if duration_field:
                start_time = timer()

            data = func(*args, **kwargs)
            if should_have_data and not data:
                # Documented early return: a falsy result under `should_have_data` is treated as
                # "the job produced nothing", so NO heartbeat is sent and a dead-man's-switch
                # monitor will fire. A job that can legitimately return an empty result must not
                # set should_have_data.
                return data

            assert isinstance(data, (type(None), dict))  # nosec B101 - internal invariant on decorated func's own return type, not a security/permission gate

            if duration_field:
                if data is None:
                    data = {}
                data[duration_field] = round(timer() - start_time, duration_rounding)

            if log_data:
                logger.info(data)

            local_job_id: str = job_id or func.__name__
            job_completed(job_id=local_job_id, status=status, data=data, provider=provider, api_key=api_key)

            return data

        return wrapper_logged

    return decorator_logged

# ----------------------------------------------------------------------------------------------------------------------------
# TIMEOUTS & DURATIONS LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

def timeout_wrapper(timeout: float = API_TIMEOUT_SEC, report_actual_duration: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to enforce a timeout on function execution.

    Runs ``func`` on a dedicated per-call daemon thread, NOT the shared ``_TIMEOUT_EXECUTOR`` pool
    (that pool is bounded and is used elsewhere for genuinely bounded-duration work -- see
    ``job_completed``). A bounded pool is unsafe here: ``func`` is arbitrary caller code with no
    internal time bound, and a Python thread can never be forcibly killed once it is running past
    its timeout -- every real timeout permanently consumes one pool slot for the rest of that
    thread's (unbounded) lifetime. Under sustained real timeouts this silently exhausts the pool,
    so unrelated calls start queuing behind permanently-stuck workers and spuriously time out even
    though their own function completes instantly (surfaced as CI flakes in
    ``test_timeout_wrapper_parametrized`` / ``test_report_duration_logs``). A dedicated thread per
    call still leaks its own thread on a genuine timeout (unavoidable without process isolation),
    but that leak can never starve capacity for any OTHER call.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Wraps func so each call runs on its own timed daemon thread and is aborted (logged) past the timeout."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Runs func on a dedicated daemon thread, returning its result, or None (logged) on timeout/exception."""
            start_ts = time.time()
            outcome: dict[str, Any] = {}

            def _run() -> None:
                """Executes func in the dedicated thread, stashing its result or exception for the caller to collect."""
                try:
                    outcome["result"] = func(*args, **kwargs)
                except Exception as e:
                    outcome["error"] = e

            thread = threading.Thread(target=_run, name=f"timeout_wrapper-{func.__name__}", daemon=True)
            thread.start()
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.error("%s timed out after %ss at %s", func.__name__, timeout, datetime.now())  # noqa: DTZ005 -- intentional local wall-clock time for a human-facing log message
                # NOTE: a Python thread cannot be forcibly stopped, so it keeps running until func
                # returns; being a dedicated per-call thread (not shared pool capacity), it cannot
                # cause any OTHER call to spuriously time out.
                return None  # Or raise, depending on use case
            if "error" in outcome:
                # Regression fix: logger.exception() implicitly sets exc_info=True, which pulls
                # sys.exc_info() from the CURRENT thread -- but the exception was caught (and its
                # except block already exited) on the CHILD thread (_run, above); by the time
                # control reaches here (the main/wrapper thread), this thread was never inside an
                # except clause at all, so sys.exc_info() here is (None, None, None) and the
                # logged traceback is bogus ("NoneType: None") regardless of which real exception
                # occurred. Passing the actual exception object via exc_info= works regardless of
                # which thread originally raised it.
                logger.error("Error in %s: %s", func.__name__, outcome["error"], exc_info=outcome["error"])
                return None
            if report_actual_duration:
                logger.info("%s completed in %.2fs", func.__name__, time.time() - start_ts)
            return outcome.get("result")
        return wrapper
    return decorator

def log_duration(threshold: float = 1.0, logger_name: Optional[str] = None, max_arg_size: int = 1000) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to measure function execution time and log if it exceeds the threshold.
    Also logs the arguments passed to the function, truncating large ones for readability.

    Args:
        threshold (float): Time in seconds above which to log (default: 1.0).
        logger_name (str): Optional logger name; if None, uses the caller's module logger.
        max_arg_size (int): Max characters in repr() before truncating (default: 1000).
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Wraps func so its execution time (and args/kwargs, if it exceeds threshold) is logged."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Calls func, timing it, and logs a message with args/kwargs if the call exceeds threshold seconds."""
            start = timer()
            result = func(*args, **kwargs)
            dur = timer() - start
            if dur > threshold:
                logger_msg = logger if logger_name is None else logging.getLogger(logger_name)

                def safe_repr(obj: Any, max_size: int = max_arg_size) -> str:
                    """Safely repr large objects, truncating if needed."""
                    repr_str = repr(obj)
                    if len(repr_str) > max_size:
                        # Truncate and add ellipsis
                        half = max_size // 2
                        # `len - 2*half`, not `len - max_size`: only 2*half characters are KEPT, so
                        # for an odd max_arg_size the notice under-reported the drop by one.
                        return f"{repr_str[:half]}...[truncated {len(repr_str) - 2 * half} chars]...{repr_str[-half:]}"
                    return repr_str

                # Format args and kwargs safely with truncation
                args_str = ", ".join(safe_repr(arg) for arg in args) if args else ""
                kwargs_str = ", ".join(f"{k}={safe_repr(v)}" for k, v in kwargs.items()) if kwargs else ""
                args_kwargs = f"({args_str}{', ' if args and kwargs else ''}{kwargs_str})"

                logger_msg.info("%s%s took %.2f s.", func.__name__, args_kwargs, dur)
            return result
        return wrapper
    return decorator
