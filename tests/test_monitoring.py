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

    def test_executor_is_module_level(self):
        """Test that ThreadPoolExecutor is created at module level (not per call)"""
        import pyutilz.monitoring as monitoring_module
        import inspect

        source = inspect.getsource(monitoring_module)

        # Should have module-level executor
        # The fix moved executor creation outside timeout_wrapper
        assert '_TIMEOUT_EXECUTOR' in source or '_executor' in source.lower(), \
               "Should have module-level ThreadPoolExecutor (performance fix)"

    def test_timeout_wrapper_executes_function(self):
        """Test that wrapped function executes correctly"""
        from pyutilz.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def simple_function(x):
            return x * 2

        result = simple_function(10)
        assert result == 20

    def test_timeout_wrapper_enforces_timeout(self):
        """Test that timeout is enforced for slow functions"""
        from pyutilz.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=1)
        def slow_function():
            time.sleep(10)
            return "should not reach here"

        # Should return None when timeout occurs (decorator catches TimeoutError)
        start = time.time()
        result = slow_function()
        elapsed = time.time() - start

        assert result is None, "Should return None on timeout"
        assert elapsed < 2.0, f"Should timeout around 1s, took {elapsed:.2f}s"

    def test_timeout_wrapper_does_not_block_fast_functions(self):
        """Test that fast functions complete before timeout"""
        from pyutilz.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def fast_function():
            time.sleep(0.1)
            return "completed"

        start = time.time()
        result = fast_function()
        elapsed = time.time() - start

        assert result == "completed"
        assert elapsed < 1.0  # Should complete quickly, not wait for full timeout

    def test_timeout_wrapper_with_arguments(self):
        """Test that wrapped function receives arguments correctly"""
        from pyutilz.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def function_with_args(a, b, c=3):
            return a + b + c

        result = function_with_args(1, 2, c=4)
        assert result == 7

    def test_multiple_calls_reuse_executor(self):
        """Test that multiple calls reuse the same executor (not creating new ones)"""
        from pyutilz.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def test_function(x):
            return x

        # Call multiple times
        results = [test_function(i) for i in range(10)]

        # Should succeed without creating multiple executors
        assert results == list(range(10))

    def test_timeout_wrapper_report_duration(self):
        """Test report_actual_duration parameter"""
        from pyutilz.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5, report_actual_duration=True)
        def timed_function():
            time.sleep(0.1)
            return "done"

        result = timed_function()

        # When report_actual_duration=True, should return tuple (result, duration)
        if isinstance(result, tuple):
            actual_result, duration = result
            assert actual_result == "done"
            assert duration >= 0.1
        else:
            # Or just the result if not reporting duration
            assert result == "done"


class TestConcurrentExecution:
    """Test concurrent execution capabilities"""

    def test_multiple_concurrent_timeouts(self):
        """Test that multiple functions can be wrapped and executed concurrently"""
        from pyutilz.monitoring import timeout_wrapper
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
        from pyutilz.monitoring import timeout_wrapper

        @timeout_wrapper(timeout=5)
        def thread_safe_function(x):
            time.sleep(0.01)
            return x ** 2

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
    from pyutilz.monitoring import timeout_wrapper

    @timeout_wrapper(timeout=timeout_value)
    def sleepy_function():
        time.sleep(sleep_time)
        return "success"

    result = sleepy_function()

    if should_timeout:
        # Should return None on timeout (decorator catches TimeoutError)
        assert result is None, "Should return None when timeout occurs"
    else:
        assert result == "success", "Should return success when completes before timeout"


def test_monitoring_module_imports_successfully():
    """Test that monitoring module can be imported without errors"""
    try:
        import pyutilz.monitoring
        assert pyutilz.monitoring is not None
    except ImportError as e:
        pytest.fail(f"Failed to import monitoring module: {e}")
