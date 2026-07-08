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
import requests
import functools
import concurrent.futures
from functools import wraps
from datetime import datetime
from timeit import default_timer as timer
from typing import Any, Optional

# ----------------------------------------------------------------------------------------------------------------------------
# INITS
# ----------------------------------------------------------------------------------------------------------------------------

API_TIMEOUT_SEC = 15

# Module-level ThreadPoolExecutor for timeout_wrapper to avoid creating new executors for each call
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
            try:
                res = requests.post(endpoint, data=data, params=params, timeout=API_TIMEOUT_SEC)

                if res.status_code not in (200, 403, 429):
                    # 403=blocked in your country
                    # 429=rate limit exceeded
                    logger.warning(f"Problem {res.status_code} while sending heartbeat to {provider} on job {job_id}: {res.text}")

            except Exception as e:
                logger.warning(f"Error while sending heartbeat to {provider} on monitor {job_id}: {e}")

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
    def decorator_logged(func):
        @functools.wraps(func)
        def wrapper_logged(*args, **kwargs):

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

def timeout_wrapper(timeout=API_TIMEOUT_SEC,report_actual_duration:bool=False,):
    """Decorator to enforce timeout on function execution.

    Uses module-level ThreadPoolExecutor to avoid creating new executor for each call.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_ts = time.time()
            # Use module-level executor instead of creating new one each call (performance)
            future = _TIMEOUT_EXECUTOR.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                if report_actual_duration:
                    logger.info("%s completed in %.2fs", func.__name__, time.time() - start_ts)
                return result
            except concurrent.futures.TimeoutError:
                logger.error(f"{func.__name__} timed out after {timeout}s at {datetime.now()}")
                # NOTE: future.cancel() cannot stop a thread that is already running
                # (which is always the case here since we waited on result), so we don't
                # attempt it; the worker thread keeps running until the wrapped call returns.
                return None  # Or raise, depending on use case
            except Exception as e:
                logger.exception(f"Error in {func.__name__}: {e}")
                return None
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
        @wraps(func)
        def wrapper(*args, **kwargs):
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
