"""Scanner tests for machine_dependent_test: tests whose verdict is the runner's, not the code's.

Each positive case is the shape of a real CI-only failure that was green on the dev box; each
negative case is the fix that was applied to it, so a regression in either direction shows up here.
"""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_deleted_attribute_read_unconditionally,
    scan_sleep_then_assert,
    scan_wall_clock_assertion,
)

from ._helpers import _write

# ---- wall_clock_assertion ------------------------------------------------


def test_wall_clock_upper_bound_through_an_elapsed_variable_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import time


def test_it_is_quick():
    start = time.perf_counter()
    work()
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
''')
    findings = scan_wall_clock_assertion(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "wall_clock_assertion"
    assert findings[0].severity == "P2"
    assert findings[0].line == 8


def test_wall_clock_difference_inline_in_the_assert_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import time


def test_no_backtracking():
    t0 = time.perf_counter()
    redact("x" * 20000)
    assert time.perf_counter() - t0 < 2.0
''')
    findings = scan_wall_clock_assertion(tmp_path)
    assert len(findings) == 1


def test_wall_clock_lower_bound_flagged(tmp_path: Path):
    """A lower bound fails where the clock is coarser than the interval, not where the code is wrong."""
    _write(tmp_path, "test_thing.py", '''
import time


def test_the_delay_happened():
    start = time.time()
    delay(0.01)
    elapsed = time.time() - start
    assert elapsed >= 0.01
''')
    findings = scan_wall_clock_assertion(tmp_path)
    assert len(findings) == 1


def test_a_timer_used_without_asserting_on_it_is_clean(tmp_path: Path):
    """Timing for a diagnostic message is not a verdict; only a comparison is."""
    _write(tmp_path, "test_thing.py", '''
import time


def test_it_works():
    start = time.perf_counter()
    result = work()
    print(f"took {time.perf_counter() - start:.3f}s")
    assert result == 42
''')
    assert scan_wall_clock_assertion(tmp_path) == []


def test_non_test_files_are_not_scanned_for_wall_clock(tmp_path: Path):
    _write(tmp_path, "production.py", '''
import time


def test_it_is_quick():
    start = time.perf_counter()
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
''')
    assert scan_wall_clock_assertion(tmp_path) == []


# ---- sleep_then_assert ---------------------------------------------------


def test_sleep_then_assert_on_a_background_count_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import time


def test_monitor_survives_a_failing_sample():
    m = Monitor(sleep_interval_seconds=0.01)
    m.start()
    time.sleep(1.5)
    m.stop(timeout=10)
    assert m.n_sampling_errors == 1
''')
    findings = scan_sleep_then_assert(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "sleep_then_assert"
    assert findings[0].severity == "P2"
    assert findings[0].line == 7


def test_a_sub_threshold_sleep_is_a_scheduling_yield_and_is_clean(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import time


def test_lock_serializes():
    t.start()
    time.sleep(0.05)  # give the other thread a chance to (wrongly) enter
    assert not entered.is_set()
''')
    assert scan_sleep_then_assert(tmp_path) == []


def test_a_sleep_inside_a_nested_workload_stub_is_clean(tmp_path: Path):
    """The sleep is the slow function under test, not the test waiting on a worker."""
    _write(tmp_path, "test_thing.py", '''
import time


def test_timeout_is_enforced():
    @timeout_wrapper(timeout=1)
    def slow_function():
        time.sleep(10)
        return "should not reach here"

    assert slow_function() is None
''')
    assert scan_sleep_then_assert(tmp_path) == []


def test_waiting_on_an_event_instead_of_sleeping_is_clean(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import threading


def test_monitor_survives_a_failing_sample():
    second_call_seen = threading.Event()
    m.start()
    assert second_call_seen.wait(30), "the sampling thread died on the failing sample"
    assert m.n_sampling_errors == 1
''')
    assert scan_sleep_then_assert(tmp_path) == []


def test_a_long_sleep_with_no_assertion_after_it_is_clean(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import time


def test_setup_only():
    assert m.started
    time.sleep(1.5)
    m.stop()
''')
    assert scan_sleep_then_assert(tmp_path) == []


# ---- deleted_attribute_read_unconditionally ------------------------------


def test_attribute_saved_unguarded_then_deleted_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import ast as _ast


def test_detail_without_ast_unparse():
    unparse = _ast.unparse
    del _ast.unparse
    try:
        details = scan(tmp_path)
    finally:
        _ast.unparse = unparse
    assert len(details) == 2
''')
    findings = scan_deleted_attribute_read_unconditionally(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "deleted_attribute_read_unconditionally"
    assert findings[0].severity == "P1"
    assert findings[0].line == 6


def test_monkeypatch_delattr_without_raising_false_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import ast as _ast


def test_fallback(monkeypatch):
    saved = _ast.unparse
    monkeypatch.delattr(_ast, "unparse")
    assert scan(saved) == []
''')
    assert len(scan_deleted_attribute_read_unconditionally(tmp_path)) == 1


def test_getattr_with_a_default_before_the_delete_is_clean(tmp_path: Path):
    """The correct form: the precondition is stated, so the absent case is handled."""
    _write(tmp_path, "test_thing.py", '''
import ast as _ast


def test_detail_without_ast_unparse():
    unparse = getattr(_ast, "unparse", None)
    if unparse is not None:
        del _ast.unparse
    try:
        details = scan(tmp_path)
    finally:
        if unparse is not None:
            _ast.unparse = unparse
    assert len(details) == 2
''')
    assert scan_deleted_attribute_read_unconditionally(tmp_path) == []


def test_monkeypatch_delattr_with_raising_false_is_clean(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import psutil


def test_missing_cpu_freq(monkeypatch):
    original = psutil.cpu_freq
    monkeypatch.delattr(psutil, "cpu_freq", raising=False)
    assert probe() is None
    assert original is not None
''')
    assert scan_deleted_attribute_read_unconditionally(tmp_path) == []


def test_a_delete_with_no_earlier_read_is_clean(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
import ast as _ast


def test_fallback():
    del _ast.unparse
    assert scan() == []
''')
    assert scan_deleted_attribute_read_unconditionally(tmp_path) == []
