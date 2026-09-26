"""Shared retry configuration for LLM providers.

All providers use retry on network/transient errors so that long-running
pipeline batches survive temporary outages, rate-limit windows, and billing
pauses without operator intervention.

By default retries up to ``MAX_RETRY_ATTEMPTS`` times (configurable via the
``PYUTILZ_LLM_MAX_RETRIES`` environment variable, 0 = infinite), within a total
per-call deadline (``PYUTILZ_LLM_MAX_CALL_SECONDS``, default 7200, 0 = none) and a cap on
consecutive read timeouts (``PYUTILZ_LLM_MAX_CONSECUTIVE_TIMEOUTS``, default 3, 0 = none).
"""

from __future__ import annotations  # PEP 604 in a runtime-evaluated annotation below; the floor is 3.8

import logging
import os
from typing import Any

from tenacity import (
    stop_after_attempt,
    stop_never,
    wait_exponential,
    wait_random,
)

logger = logging.getLogger(__name__)

# Configurable via environment variable; 0 means infinite.
_DEFAULT_MAX_RETRY_ATTEMPTS = 50
try:
    MAX_RETRY_ATTEMPTS: int = int(os.environ.get("PYUTILZ_LLM_MAX_RETRIES", str(_DEFAULT_MAX_RETRY_ATTEMPTS)))
except ValueError:
    logger.warning(
        "PYUTILZ_LLM_MAX_RETRIES=%r is not an integer; falling back to %d.",
        os.environ.get("PYUTILZ_LLM_MAX_RETRIES"), _DEFAULT_MAX_RETRY_ATTEMPTS,
    )
    MAX_RETRY_ATTEMPTS = _DEFAULT_MAX_RETRY_ATTEMPTS
if MAX_RETRY_ATTEMPTS < 0:
    # A negative value produced stop_after_attempt(-1), which stops after the FIRST attempt --
    # i.e. it silently disabled retries entirely, the exact opposite of the "0 = infinite" scale
    # this module documents (2026-09-03 audit F40). Refuse it loudly instead.
    logger.warning(
        "PYUTILZ_LLM_MAX_RETRIES=%d is negative, which would silently disable retries; falling back to %d. Use 0 for infinite retries.",
        MAX_RETRY_ATTEMPTS, _DEFAULT_MAX_RETRY_ATTEMPTS,
    )
    MAX_RETRY_ATTEMPTS = _DEFAULT_MAX_RETRY_ATTEMPTS


def log_retry(retry_state) -> None:
    """Log a warning before each retry sleep, escalating severity."""
    exc = retry_state.outcome.exception()
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    attempt = retry_state.attempt_number
    level = logging.WARNING if attempt < 10 else logging.ERROR
    logger.log(
        level,
        "LLM call attempt %d failed (%s: %s), retrying in %.0fs...",
        attempt,
        type(exc).__name__,
        str(exc)[:200],
        wait,
    )


# Shared wait strategy: roughly doubling backoff (attempts 1-9 with multiplier=2/min=5/max=300
# actually produce 5, 5, 8, 16, 32, 64, 128, 256, 300 -- the first two attempts both floor at 5s
# before the doubling pattern becomes visible, not a clean 5/10/20/... sequence), capped at 300s
# (5 min), plus random jitter to avoid thundering herd.
RETRY_WAIT = wait_exponential(multiplier=2, min=5, max=300) + wait_random(0, 5)

_STOP_ATTEMPTS = stop_never if MAX_RETRY_ATTEMPTS == 0 else stop_after_attempt(MAX_RETRY_ATTEMPTS)

# ── Billing pauses get a grace window, not the full schedule ────────────────────────────────────
#
# HTTP 402 is retryable on purpose: a balance that runs out mid-batch is often topped up within
# minutes, and dropping the batch would be worse than pausing it. The tail is what was wrong.
#
# Measured on a live run against an empty OpenRouter balance: the shared policy retried a 402 fifty
# times with waits of 5, 10, 20, 40, 80, 160 and then 300s each -- about 3.8 HOURS of wall clock for
# one call. A consumer with five concurrent workers wedges all five for most of a working day, on a
# condition no amount of waiting resolves. The first six attempts already span five minutes; if the
# balance was going to be restored, it was restored long before attempt seven.
#
# A window in ELAPSED TIME rather than an attempt count, because "wait five minutes for a top-up" is
# the actual intent: it stays true if anyone retunes the backoff curve, while an attempt count
# silently changes meaning the moment the waits change.
try:
    BILLING_GRACE_SECONDS: float = float(os.environ.get("PYUTILZ_LLM_BILLING_GRACE_SECONDS", "300"))
except ValueError:
    logger.warning(
        "PYUTILZ_LLM_BILLING_GRACE_SECONDS=%r is not a number; falling back to 300.",
        os.environ.get("PYUTILZ_LLM_BILLING_GRACE_SECONDS"),
    )
    BILLING_GRACE_SECONDS = 300.0
if BILLING_GRACE_SECONDS < 0:
    logger.warning("PYUTILZ_LLM_BILLING_GRACE_SECONDS=%s is negative; falling back to 300.", BILLING_GRACE_SECONDS)
    BILLING_GRACE_SECONDS = 300.0


# ── A total deadline per call, and a cap on back-to-back read timeouts ─────────────────────────────
#
# Without these the only elapsed-time stop was the 402 grace: 50 attempts, each allowed up to 3,000 s by the
# derived timeout, with up to 300 s of backoff between them, put the worst case for ONE call near 46 hours.
#
# The deadline is checked only BETWEEN attempts, never during one, so a long legitimate stream is never cut: an
# attempt that started before the deadline runs to its own timeout, and the deadline only refuses to start
# another. It is sized to let two full-length derived-timeout attempts (2 x 3,000 s) plus backoff fit.
_DEFAULT_MAX_CALL_SECONDS = 7200.0
_DEFAULT_MAX_CONSECUTIVE_TIMEOUTS = 3


def _env_number(name: str, default: float) -> float:
    """A non-negative float from the environment, warning and falling back to ``default`` on junk."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; falling back to %s.", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%s is negative; falling back to %s. Use 0 to disable.", name, value, default)
        return default
    return value


#: Seconds after the first attempt started past which no further attempt is started. 0 disables the deadline.
MAX_CALL_SECONDS: float = _env_number("PYUTILZ_LLM_MAX_CALL_SECONDS", _DEFAULT_MAX_CALL_SECONDS)
#: Consecutive read timeouts after which a call gives up. A silent route that times out at the derived ceiling
#: will do it again, and each such attempt may be billed upstream while it generated. 0 disables the cap.
MAX_CONSECUTIVE_TIMEOUTS: int = int(_env_number("PYUTILZ_LLM_MAX_CONSECUTIVE_TIMEOUTS", _DEFAULT_MAX_CONSECUTIVE_TIMEOUTS))

#: Exception class names that mean "the attempt waited for the model and gave up", across httpx and the SDKs.
_READ_TIMEOUT_NAMES = frozenset({"ReadTimeout", "APITimeoutError", "DeadlineExceeded", "TimeoutError"})


def _is_read_timeout(exc: BaseException | None) -> bool:
    """True for a read-side timeout (httpx ``ReadTimeout``, SDK timeout errors, builtin ``TimeoutError``).

    Matched by class name through the MRO rather than by import, for the same reason as ``_is_billing_pause``.
    A ``ConnectTimeout`` is not one: it fails fast and says nothing about a route that has gone silent.
    """
    if exc is None:
        return False
    return any(cls.__name__ in _READ_TIMEOUT_NAMES for cls in type(exc).__mro__)


def _is_billing_pause(exc: BaseException | None) -> bool:
    """True for an HTTP 402 from any provider.

    Duck-typed through ``getattr`` rather than importing httpx: this module is imported by every
    provider, including ones with no HTTP client at all, and a hard dependency here would be a new
    import edge for the sake of one status code.
    """
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 402


def _stop_policy(retry_state: Any) -> bool:
    """The shared attempt cap, plus a total deadline, a consecutive-timeout cap and a billing-pause cut-off."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    elapsed = retry_state.seconds_since_start or 0.0
    if MAX_CALL_SECONDS and elapsed >= MAX_CALL_SECONDS:
        logger.error(
            "Giving up after %.0fs across %d attempt(s): the call deadline is %.0fs (PYUTILZ_LLM_MAX_CALL_SECONDS, 0 = none).",
            elapsed, retry_state.attempt_number, MAX_CALL_SECONDS,
        )
        return True
    # The streak lives on the RetryCallState, which tenacity creates per call, so it never leaks between calls.
    streak = getattr(retry_state, "_pyutilz_timeout_streak", 0)
    streak = streak + 1 if _is_read_timeout(exc) else 0
    retry_state._pyutilz_timeout_streak = streak
    if MAX_CONSECUTIVE_TIMEOUTS and streak >= MAX_CONSECUTIVE_TIMEOUTS:
        logger.error(
            "Giving up after %d consecutive read timeouts: the route is not answering within the timeout "
            "(PYUTILZ_LLM_MAX_CONSECUTIVE_TIMEOUTS, 0 = no cap).",
            streak,
        )
        return True
    if _is_billing_pause(exc):
        if elapsed >= BILLING_GRACE_SECONDS:
            logger.error(
                "Giving up after %.0fs of HTTP 402 (account out of credits). "
                "Retrying further cannot help: top up the balance and re-run. "
                "Set PYUTILZ_LLM_BILLING_GRACE_SECONDS to change the wait.",
                elapsed,
            )
            return True
    return _STOP_ATTEMPTS(retry_state)


_STOP = _stop_policy

# Common tenacity kwargs for retry on transient errors.
# Each provider supplies its own `retry=` predicate.
INFINITE_RETRY_KWARGS = dict(
    wait=RETRY_WAIT,
    stop=_STOP,
    before_sleep=log_retry,
    reraise=True,
)
