"""Behavioural tests for `pyutilz.dev.logginglib.log_throttle`.

Lifted from the Upwork scrapers, where three hot-loop log sites had each grown their own timestamp
dict. The properties below are the ones those copies got wrong or would have: independent keys must
not share a window, the check-and-stamp must be atomic, and the clock must be monotonic.
"""

from __future__ import annotations

import threading

import pytest

from pyutilz.dev.logginglib import log_throttle, reset_log_throttles


@pytest.fixture(autouse=True)
def _clean_windows():
    """No test inherits another's throttle clock."""
    reset_log_throttles()
    yield
    reset_log_throttles()


def test_the_first_call_passes_and_the_second_does_not():
    assert log_throttle("k", 60.0) is True
    assert log_throttle("k", 60.0) is False


def test_independent_keys_do_not_share_a_window():
    """Two errors throttled together means the second is invisible whenever the first is firing."""
    assert log_throttle("first", 60.0) is True
    assert log_throttle("second", 60.0) is True


def test_a_zero_interval_never_throttles():
    """`min_interval=0` is the documented way to disable a throttle without removing the guard."""
    assert log_throttle("k", 0.0) is True
    assert log_throttle("k", 0.0) is True


def test_reset_reopens_the_window():
    assert log_throttle("k", 60.0) is True
    assert log_throttle("k", 60.0) is False
    reset_log_throttles()
    assert log_throttle("k", 60.0) is True


def test_exactly_one_of_many_concurrent_callers_passes():
    """The check and the stamp happen under one lock.

    Doing them in two steps is how a "once a minute" guard fires N times a minute under load, and
    it is the bug a hand-rolled copy beside a call site is most likely to have. The barrier makes
    the threads contend on purpose rather than hoping the scheduler interleaves them.
    """
    passed: list[bool] = []
    lock = threading.Lock()
    workers = 16
    barrier = threading.Barrier(workers)

    def go() -> None:
        barrier.wait()
        allowed = log_throttle("contended", 60.0)
        with lock:
            passed.append(allowed)

    threads = [threading.Thread(target=go) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(passed) == workers
    assert sum(passed) == 1, f"expected exactly one caller through the gate, got {sum(passed)}"


def test_the_window_is_measured_on_a_monotonic_clock(monkeypatch):
    """A wall-clock step must not open the window early or hold it shut.

    Patching `time.monotonic` proves which clock is read: if the helper used `time.time` this test
    would see no effect at all and pass for the wrong reason, so it also asserts the throttle
    actually moved.
    """
    import pyutilz.dev.logginglib as logginglib

    fake = {"now": 1000.0}
    monkeypatch.setattr(logginglib.time, "monotonic", lambda: fake["now"])

    assert log_throttle("k", 60.0) is True
    fake["now"] += 59.0
    assert log_throttle("k", 60.0) is False
    fake["now"] += 2.0
    assert log_throttle("k", 60.0) is True

def test_the_first_call_passes_on_a_freshly_booted_machine(monkeypatch):
    """`time.monotonic()` is time since boot on some platforms, so a small value is normal.

    Encoding "never logged" as 0.0 made the very first call compare `now - 0.0` against the
    interval: on a runner up for less than a minute the first log was throttled away, and only
    there - the macOS legs failed while every other leg, on a long-lived host, passed.
    """
    import pyutilz.dev.logginglib as logginglib

    monkeypatch.setattr(logginglib.time, "monotonic", lambda: 12.0)
    assert log_throttle("fresh-boot", min_interval=60.0) is True
    assert log_throttle("fresh-boot", min_interval=60.0) is False
