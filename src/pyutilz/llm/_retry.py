"""Shared retry configuration for LLM providers.

All providers use retry on network/transient errors so that long-running
pipeline batches survive temporary outages, rate-limit windows, and billing
pauses without operator intervention.

By default retries up to ``MAX_RETRY_ATTEMPTS`` times (configurable via the
``PYUTILZ_LLM_MAX_RETRIES`` environment variable, 0 = infinite).
"""

import logging
import os

from tenacity import (
    retry_if_exception,
    stop_after_attempt,
    stop_never,
    wait_exponential,
    wait_random,
)

logger = logging.getLogger(__name__)

# Configurable via environment variable; 0 means infinite.
MAX_RETRY_ATTEMPTS: int = int(
    os.environ.get("PYUTILZ_LLM_MAX_RETRIES", "50")
)


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


# Shared wait strategy: exponential 5s → 10s → 20s → ... → 300s (5 min cap)
# plus random jitter to avoid thundering herd.
RETRY_WAIT = wait_exponential(multiplier=2, min=5, max=300) + wait_random(0, 5)

_STOP = stop_never if MAX_RETRY_ATTEMPTS == 0 else stop_after_attempt(MAX_RETRY_ATTEMPTS)

# Common tenacity kwargs for retry on transient errors.
# Each provider supplies its own `retry=` predicate.
INFINITE_RETRY_KWARGS = dict(
    wait=RETRY_WAIT,
    stop=_STOP,
    before_sleep=log_retry,
    reraise=True,
)
