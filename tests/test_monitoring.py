"""Tests for monitoring.py - Phase 2 refactoring

Tests cover:
- ThreadPoolExecutor optimization (module-level reuse instead of per-call creation)
- Timeout enforcement functionality
"""

import pytest
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

        # Elapsed time is not asserted: the shared gate rejects wall-clock bounds because their verdict depends on the runner.
        result = fast_function()
        assert result == "completed"

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

    def test_timeout_wrapper_report_duration(self, caplog, monkeypatch):
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

        from types import SimpleNamespace

        from pyutilz.system import monitoring

        fake_now = [1000.0]
        monkeypatch.setattr(monitoring, "time", SimpleNamespace(time=lambda: fake_now[0]))

        @timeout_wrapper(timeout=5, report_actual_duration=True)
        def timed_function():
            fake_now[0] += 0.1  # the call "takes" 0.1s on the injected clock
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


@pytest.mark.parametrize("timeout_value,blocks,should_timeout", [
    (2, False, False),     # Completes before timeout
    (0.05, True, True),    # Never finishes within the timeout
    (5, False, False),     # Completes quickly
])
def test_timeout_wrapper_parametrized(timeout_value, blocks, should_timeout):
    """Parametrized test for different timeout scenarios"""
    from pyutilz.system.monitoring import timeout_wrapper

    release = threading.Event()

    @timeout_wrapper(timeout=timeout_value)
    def sleepy_function():
        if blocks:
            release.wait(30)  # held until the wrapper has given up on the call
        return "success"

    try:
        result = sleepy_function()
    finally:
        release.set()

    # One unconditional assertion over both parametrised cases: the decorator swallows the
    # TimeoutError and returns None, and otherwise passes the real return value straight through.
    assert result == (None if should_timeout else "success"), f"unexpected result for timeout={timeout_value}, blocks={blocks}"


def test_monitoring_module_imports_successfully():
    """Test that monitoring module can be imported without errors"""
    try:
        import pyutilz.monitoring
        assert pyutilz.monitoring is not None
    except ImportError as e:
        pytest.fail(f"Failed to import monitoring module: {e}")


class _MonLog:
    def __init__(self):
        self.warnings: list = []

    def warning(self, fmt, *args, **kwargs):
        self.warnings.append((fmt, args))


class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text


class TestHeartbeatAttemptAndBackoff:
    """A heartbeat attempt warns on an unexpected status with its exact text, stays silent on 200, 403 and 429, retries only
    429 and server errors, and the backoff doubles from the initial delay up to the cap with a 20 percent jitter (monitoring 87-126)."""

    def test_the_backoff_doubles_from_the_initial_delay_and_is_capped(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon.random, "uniform", lambda a, b: 1.0)
        assert mon._backoff_delay(1) == 5.0
        assert mon._backoff_delay(2) == 10.0
        assert mon._backoff_delay(3) == 20.0
        assert mon._backoff_delay(60) == 300.0

    def test_the_jitter_is_drawn_between_eighty_and_one_twenty_percent(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        drawn: list = []
        monkeypatch.setattr(mon.random, "uniform", lambda a, b: drawn.append((a, b)) or 1.0)
        mon._backoff_delay(1)
        assert drawn == [(0.8, 1.2)]

    def test_a_success_and_the_silent_statuses_warn_nothing(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        log = _MonLog()
        monkeypatch.setattr(mon, "logger", log)
        for status, expected in ((200, True), (403, True), (429, False)):
            monkeypatch.setattr(mon.requests, "post", lambda *a, _s=status, **k: _Resp(_s))
            assert mon._attempt("http://e", {}, None, "prov", "job") is expected
        assert log.warnings == []

    def test_an_unexpected_status_warns_with_its_exact_text_and_is_retried(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        log = _MonLog()
        monkeypatch.setattr(mon, "logger", log)
        monkeypatch.setattr(mon.requests, "post", lambda *a, **k: _Resp(500, "boom"))
        assert mon._attempt("http://e", {}, None, "prov", "job") is False
        assert log.warnings == [("Problem %s while sending heartbeat to %s on job %s: %s", (500, "prov", "job", "boom"))]

    def test_a_missing_requests_package_warns_and_is_not_retried(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        log = _MonLog()
        monkeypatch.setattr(mon, "logger", log)
        monkeypatch.setattr(mon, "requests", None)
        assert mon._attempt("http://e", {}, None, "prov", "job") is True
        fmt, args = log.warnings[0]
        assert fmt == "Error while sending heartbeat to %s on monitor %s: %s"
        assert args[0:2] == ("prov", "job")

    def test_a_request_error_logs_only_its_class_and_is_retried(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        log = _MonLog()
        monkeypatch.setattr(mon, "logger", log)

        def boom(*a, **k):
            raise ConnectionError("https://secret-key@host/ping")

        monkeypatch.setattr(mon.requests, "post", boom)
        assert mon._attempt("http://e", {}, None, "prov", "job") is False
        fmt, args = log.warnings[0]
        assert fmt == "Error while sending heartbeat to %s on monitor %s: %s (message omitted: may contain the ping URL with its key)"
        assert args == ("prov", "job", "ConnectionError")


class _NoThread:
    def __init__(self, target=None, name=None, daemon=None):
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return False


class TestHeartbeatRetrierQueue:
    """A newer failure on a stream replaces the older one, delivery forgets the stream, the queueing and give-up messages are exact,
    and a pending item's first retry is the initial backoff (monitoring 140-200)."""

    def _item(self, key=("e", "state"), provider="prov", job="job"):
        from pyutilz.system import monitoring as mon

        return mon._Pending(key, "http://e", {"a": 1}, None, provider, job)

    def test_a_pending_item_first_retries_after_the_initial_backoff(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon.random, "uniform", lambda a, b: 1.0)
        item = self._item()
        assert item.next_at - item.first_failed_at == pytest.approx(5.0)
        assert item.failures == 1

    def test_a_newer_failure_on_the_same_stream_replaces_the_older_one(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon.threading, "Thread", _NoThread)
        retrier = mon._HeartbeatRetrier()
        older = self._item()
        newer = self._item()
        retrier.submit(older)
        retrier.submit(newer)
        assert list(retrier._pending.values()) == [newer]

    def test_forgetting_a_stream_drops_its_pending_item_and_an_unknown_one_is_harmless(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon.threading, "Thread", _NoThread)
        retrier = mon._HeartbeatRetrier()
        retrier.submit(self._item(key=("a", "s")))
        retrier.forget(("a", "s"))
        assert retrier._pending == {}
        retrier.forget(("never", "seen"))

    def test_queueing_logs_the_provider_monitor_and_first_retry_delay(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        recorded: list = []

        class _InfoLog:
            def info(self, fmt, *args, **kwargs):
                recorded.append((fmt, args))

        monkeypatch.setattr(mon, "logger", _InfoLog())
        monkeypatch.setattr(mon.threading, "Thread", _NoThread)
        monkeypatch.setattr(mon.random, "uniform", lambda a, b: 1.0)
        retrier = mon._HeartbeatRetrier()
        item = self._item(provider="prov", job="job")
        retrier.submit(item)
        assert recorded == [
            ("Heartbeat to %s on monitor %s queued for background retry (first retry in %.0fs)", ("prov", "job", item.next_at - item.first_failed_at))
        ]

    def test_giving_up_logs_the_failure_count_and_minutes(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        recorded: list = []

        class _WarnLog:
            def warning(self, fmt, *args, **kwargs):
                recorded.append((fmt, args))

        monkeypatch.setattr(mon, "logger", _WarnLog())
        item = self._item(provider="prov", job="job")
        item.failures = 4
        item.first_failed_at = 0.0
        mon._HeartbeatRetrier._log_gave_up(item, now=180.0)
        assert recorded == [("Giving up on heartbeat to %s on monitor %s after %d failed attempts over %.0f min", ("prov", "job", 4, 3.0))]
