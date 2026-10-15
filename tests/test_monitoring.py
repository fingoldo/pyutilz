"""Tests for monitoring.py - Phase 2 refactoring

Tests cover:
- ThreadPoolExecutor optimization (module-level reuse instead of per-call creation)
- Timeout enforcement functionality
"""

import pytest
import time
import threading


class TestTimeoutWrapper:
    """Test timeout_wrapper decorator - ThreadPoolExecutor optimization"""

    def test_executor_is_module_level_and_shared(self):
        """The heartbeat executor is ONE module-level pool, reused across calls.

        Behavioural replacement for an ``inspect.getsource(...)`` string match on
        ``"_TIMEOUT_EXECUTOR"``, which passed on any module merely mentioning the name and
        failed on a pure rename. Here the pool object identity is checked directly, and
        ``job_completed`` is driven twice to prove each non-blocking send is *submitted to
        that same pool* rather than to a freshly constructed executor per call.
        """
        import concurrent.futures
        from unittest.mock import patch

        import pyutilz.monitoring as monitoring_module

        executor = monitoring_module._TIMEOUT_EXECUTOR
        assert isinstance(executor, concurrent.futures.ThreadPoolExecutor)

        submitted = []
        with patch.object(executor, "submit", side_effect=lambda fn, *a, **kw: submitted.append(fn)) as spy, patch.object(
            concurrent.futures, "ThreadPoolExecutor", side_effect=AssertionError("a new ThreadPoolExecutor must not be built per call")
        ):
            monitoring_module.job_completed(job_id="j1", blocking=False)
            monitoring_module.job_completed(job_id="j2", blocking=False)

        assert spy.call_count == 2, "both non-blocking heartbeats must go through the shared module-level pool"
        assert len(submitted) == 2
        # And the pool object itself is still the same one after the calls -- not rebuilt.
        assert monitoring_module._TIMEOUT_EXECUTOR is executor

    def test_timeout_wrapper_executes_function(self):
        """Test that wrapped function executes correctly"""
        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def simple_function(x):
            return x * 2

        result = simple_function(10)
        assert result == 20

    def test_timeout_wrapper_enforces_timeout(self):
        """The wrapper returns None while the wrapped call is still running, i.e. it did not wait for completion.

        The wrapped function blocks on an event the test controls, so there is no wall-clock bound: if the timeout were
        not enforced, the wrapper could only return after the event is released, and ``finished`` would already be set.
        """
        from pyutilz.system.monitoring import timeout_wrapper

        release = threading.Event()
        finished = threading.Event()

        @timeout_wrapper(timeout=1)
        def slow_function():
            release.wait(timeout=30)  # safety net so a broken wrapper cannot hang the run forever
            finished.set()
            return "should not reach here"

        try:
            result = slow_function()
            assert result is None, "Should return None on timeout"
            assert not finished.is_set(), "wrapper returned only after the wrapped call completed: timeout not enforced"
        finally:
            release.set()

    def test_timeout_wrapper_does_not_block_fast_functions(self):
        """A fast function's result is returned, and the wrapper does not sit out the (large) timeout first.

        The timeout is 30s; a wrapper that waits for it returns the right value but takes ~30s. The check bounds the
        call at half the timeout, a margin that only a timeout-length wait can exceed.
        """
        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=30)
        def fast_function():
            return "completed"

        start = time.perf_counter()
        result = fast_function()
        elapsed = time.perf_counter() - start
        assert result == "completed"
        assert elapsed < 15.0, f"wrapper waited {elapsed:.2f}s for a function that returns immediately"

    def test_timeout_wrapper_with_arguments(self):
        """Test that wrapped function receives arguments correctly"""
        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def function_with_args(a, b, c=3):
            return a + b + c

        result = function_with_args(1, 2, c=4)
        assert result == 7

    def test_multiple_calls_reuse_executor(self):
        """Test that multiple calls reuse the same executor (not creating new ones)"""
        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def _echo(x):
            return x

        # Call multiple times
        results = [_echo(i) for i in range(10)]

        # Should succeed without creating multiple executors
        assert results == list(range(10))

    def test_timeout_wrapper_report_duration(self, caplog):
        """``report_actual_duration=True`` LOGS the measured duration; the return value is unchanged.

        Reframed against the real contract (src/pyutilz/system/monitoring.py:227-229): the flag
        emits ``"<func> completed in <N>s"`` at INFO and still returns the bare result -- it does
        NOT wrap it in a ``(result, duration)`` tuple. The previous version guarded on
        ``if isinstance(result, tuple)`` and, on the else branch, asserted only ``result ==
        "done"``, i.e. it passed identically whether the flag did anything at all.
        """
        import logging
        import re

        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5, report_actual_duration=True)
        def timed_function():
            time.sleep(0.1)
            return "done"

        with caplog.at_level(logging.INFO, logger="pyutilz.system.monitoring"):
            result = timed_function()

        assert result == "done"
        assert not isinstance(result, tuple)
        match = re.search(r"timed_function completed in ([0-9.]+)s", caplog.text)
        assert match is not None, f"report_actual_duration=True must log the duration; got: {caplog.text!r}"
        assert float(match.group(1)) >= 0.1

    def test_timeout_wrapper_without_report_duration_logs_nothing(self, caplog):
        """The counterpart: with the flag off, no duration line is emitted at all."""
        import logging

        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def quiet_function():
            time.sleep(0.1)
            return "done"

        with caplog.at_level(logging.INFO, logger="pyutilz.system.monitoring"):
            result = quiet_function()

        assert result == "done"
        assert "completed in" not in caplog.text


class TestConcurrentExecution:
    """Test concurrent execution capabilities"""

    def test_multiple_concurrent_timeouts(self):
        """Test that multiple functions can be wrapped and executed concurrently"""
        from pyutilz.system.monitoring import timeout_wrapper
        import concurrent.futures

        @timeout_wrapper(timeout=5)
        def task(n):
            time.sleep(0.1)
            return n * 2

        # Execute multiple tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(task, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert results == [i * 2 for i in range(10)]

    def test_thread_safety(self):
        """Test that timeout_wrapper is thread-safe"""
        from pyutilz.system.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def thread_safe_function(x):
            time.sleep(0.01)
            return x**2

        results = []
        threads = []

        def worker(value):
            result = thread_safe_function(value)
            results.append(result)

        # Create multiple threads
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All results should be present
        assert len(results) == 10
        assert set(results) == {i**2 for i in range(10)}


@pytest.mark.parametrize("timeout_value,sleep_time,should_timeout", [
    (2, 0.5, False),  # Completes before timeout
    (1, 2, True),     # Times out
    (5, 0.1, False),  # Completes quickly
])
def test_timeout_wrapper_parametrized(timeout_value, sleep_time, should_timeout):
    """Parametrized test for different timeout scenarios"""
    from pyutilz.system.monitoring import timeout_wrapper

    @timeout_wrapper(timeout=timeout_value)
    def sleepy_function():
        time.sleep(sleep_time)
        return "success"

    result = sleepy_function()

    # One unconditional assertion over both parametrised cases: the decorator swallows the
    # TimeoutError and returns None, and otherwise passes the real return value straight through.
    assert result == (None if should_timeout else "success"), f"unexpected result for timeout={timeout_value}, sleep={sleep_time}"


def test_monitoring_module_imports_successfully():
    """Test that monitoring module can be imported without errors"""
    try:
        import pyutilz.monitoring
        assert pyutilz.monitoring is not None
    except ImportError as e:
        pytest.fail(f"Failed to import monitoring module: {e}")
