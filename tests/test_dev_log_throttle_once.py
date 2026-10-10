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
