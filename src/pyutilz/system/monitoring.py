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
import functools
import threading
import concurrent.futures
from functools import wraps
from datetime import datetime
from timeit import default_timer as timer
from typing import Any, Optional

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
# 3RD PARTIES MONITORING
# ----------------------------------------------------------------------------------------------------------------------------


def job_completed(
    job_id: str, status: int = 0, data: Optional[str] = None, provider: str = "healthchecks.io", api_key: Optional[str] = None, blocking: bool = True
):
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

        def _send():
            """Post the heartbeat to the endpoint, logging (not raising) on a non-OK status or request error."""
            try:
                if requests is None:
                    raise ImportError("job_completed's heartbeat send requires requests, which failed to import (see earlier debug log for the reason)")
                res = requests.post(endpoint, data=data, params=params, timeout=API_TIMEOUT_SEC)

                if res.status_code not in (200, 403, 429):
                    # 403=blocked in your country
                    # 429=rate limit exceeded
                    logger.warning("Problem %s while sending heartbeat to %s on job %s: %s", res.status_code, provider, job_id, res.text)

            except Exception as e:
                logger.warning("Error while sending heartbeat to %s on monitor %s: %s", provider, job_id, e)

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
):
    """Decorator factory that runs the wrapped function, times it, logs and reports its dict result as a job_completed heartbeat.

    The wrapped function must return either ``None`` or a ``dict``. When ``duration_field`` is set,
    the elapsed wall-clock time is added to that key of the result dict before logging/reporting.
    """
    def decorator_logged(func):
        """Wraps func so each call is timed, logged, and reported via job_completed."""
        @functools.wraps(func)
        def wrapper_logged(*args, **kwargs):
            """Calls func, augments its dict result with duration, logs it, and pings the monitoring provider."""

            if duration_field:
                start_time = timer()

            data = func(*args, **kwargs)
            if should_have_data and not data:
                return data

            assert isinstance(data, (type(None), dict))  # nosec B101 - internal invariant on decorated func's own return type, not a security/permission gate

            if duration_field:
                if data is None:
                    data = {}
                data[duration_field] = round(timer() - start_time, duration_rounding)

            if log_data:
                logger.info(data)

            local_job_id = job_id
            if not local_job_id:
                local_job_id = func.__name__
            job_completed(job_id=local_job_id, status=status, data=data, provider=provider, api_key=api_key)

            return data

        return wrapper_logged

    return decorator_logged

# ----------------------------------------------------------------------------------------------------------------------------
# TIMEOUTS & DURATIONS LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

def timeout_wrapper(timeout: float = API_TIMEOUT_SEC, report_actual_duration: bool = False):
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
    def decorator(func):
        """Wraps func so each call runs on its own timed daemon thread and is aborted (logged) past the timeout."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Runs func on a dedicated daemon thread, returning its result, or None (logged) on timeout/exception."""
            start_ts = time.time()
            outcome: dict[str, Any] = {}

            def _run():
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

def log_duration(threshold=1.0, logger_name=None, max_arg_size=1000):
    """
    Decorator to measure function execution time and log if it exceeds the threshold.
    Also logs the arguments passed to the function, truncating large ones for readability.

    Args:
        threshold (float): Time in seconds above which to log (default: 1.0).
        logger_name (str): Optional logger name; if None, uses the caller's module logger.
        max_arg_size (int): Max characters in repr() before truncating (default: 1000).
    """
    def decorator(func):
        """Wraps func so its execution time (and args/kwargs, if it exceeds threshold) is logged."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Calls func, timing it, and logs a message with args/kwargs if the call exceeds threshold seconds."""
            start = timer()
            result = func(*args, **kwargs)
            dur = timer() - start
            if dur > threshold:
                logger_msg = logger if logger_name is None else logging.getLogger(logger_name)

                def safe_repr(obj, max_size=max_arg_size):
                    """Safely repr large objects, truncating if needed."""
                    repr_str = repr(obj)
                    if len(repr_str) > max_size:
                        # Truncate and add ellipsis
                        half = max_size // 2
                        return f"{repr_str[:half]}...[truncated {len(repr_str) - max_size} chars]...{repr_str[-half:]}"
                    return repr_str

                # Format args and kwargs safely with truncation
                args_str = ", ".join(safe_repr(arg) for arg in args) if args else ""
                kwargs_str = ", ".join(f"{k}={safe_repr(v)}" for k, v in kwargs.items()) if kwargs else ""
                args_kwargs = f"({args_str}{', ' if args and kwargs else ''}{kwargs_str})"

                logger_msg.info("%s%s took %.2f s.", func.__name__, args_kwargs, dur)
            return result
        return wrapper
    return decorator
