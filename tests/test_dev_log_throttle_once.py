"""log_throttle with ONCE admits a key exactly once per process, independently per key, until the windows are reset."""

from __future__ import annotations

from pyutilz.dev.logginglib import ONCE, log_throttle, reset_log_throttles


def test_once_admits_a_key_a_single_time_and_other_keys_independently() -> None:
    """The first call per key is True, every later call with the same key is False, and a different key is unaffected."""
    reset_log_throttles()
    assert log_throttle("once.a", ONCE) is True
    assert log_throttle("once.a", ONCE) is False
    assert log_throttle("once.a", ONCE) is False
    assert log_throttle("once.b", ONCE) is True


def test_reset_reopens_every_window() -> None:
    """After reset_log_throttles the same key is admitted once more."""
    reset_log_throttles()
    assert log_throttle("once.c", ONCE) is True
    reset_log_throttles()
    assert log_throttle("once.c", ONCE) is True


def test_log_throttled_surfaces_the_first_occurrences_then_a_notice_then_silence(caplog) -> None:
    """The count-based wrapper logs max_count times, adds one suppression notice on the last allowed one, and nothing afterwards; keys are independent."""
    import logging

    from pyutilz.dev.logginglib import log_throttled

    reset_log_throttles()
    log = logging.getLogger("pyutilz.test_throttled")
    with caplog.at_level(logging.WARNING, logger=log.name):
        for i in range(5):
            log_throttled(log, "site.a", logging.WARNING, "problem %d", i, max_count=3)
        log_throttled(log, "site.b", logging.WARNING, "other")
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["problem 0", "problem 1", "problem 2", "site.a: further occurrences suppressed (throttled)", "other"]


def test_reset_with_a_key_clears_only_that_key_in_both_stores() -> None:
    """A keyed reset reopens that key's time window and count and leaves the others closed."""
    import logging

    from pyutilz.dev.logginglib import log_throttled

    reset_log_throttles()
    assert log_throttle("k.window", ONCE) is True
    assert log_throttle("k.other", ONCE) is True
    log = logging.getLogger("pyutilz.test_throttled_reset")
    log_throttled(log, "k.count", logging.DEBUG, "x", max_count=1)
    reset_log_throttles("k.window")
    assert log_throttle("k.window", ONCE) is True
    assert log_throttle("k.other", ONCE) is False
