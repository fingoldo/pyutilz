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


class TestHeartbeatRetrierDelivery:
    """Due items are picked up, a late one is given up, a delivered one is forgotten with its exact log line, a failed one backs off, and
    flush makes one attempt per item within its budget (monitoring 205-262)."""

    def _item(self, first_failed_at, key=("e", "s")):
        from pyutilz.system import monitoring as mon

        item = mon._Pending(key, "http://e", {"a": 1}, None, "prov", "job")
        item.first_failed_at = first_failed_at
        item.next_at = first_failed_at
        return item

    def test_a_due_item_is_picked_and_a_future_one_sets_the_wait(self):
        from pyutilz.system import monitoring as mon

        retrier = mon._HeartbeatRetrier()
        now = mon.time.monotonic()
        due = self._item(now)
        later = self._item(now, key=("e", "later"))
        later.next_at = now + 30.0
        retrier._pending = {due.key: due, later.key: later}
        picked, soonest = retrier._next_due(now)
        assert picked == [due]
        assert soonest is None or soonest > 0

    def test_a_pending_item_older_than_the_retry_window_is_given_up(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        recorded: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"warning": lambda self, *a, **k: recorded.append(a[0])})())
        retrier = mon._HeartbeatRetrier()
        now = mon.time.monotonic()
        stale = self._item(now - mon.RETRY_MAX_AGE_SEC - 1)
        retrier._pending = {stale.key: stale}
        picked, _ = retrier._next_due(now)
        assert picked == []
        assert retrier._pending == {}
        assert recorded == ["Giving up on heartbeat to %s on monitor %s after %d failed attempts over %.0f min"]

    def test_a_delivered_item_is_forgotten_with_its_exact_log_line(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"info": lambda self, fmt, *a, **k: infos.append((fmt, a)), "warning": lambda self, *a, **k: None})())
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: True)
        # Bound the loop: a wrongly undelivered item must be given up within a second, not retried forever.
        monkeypatch.setattr(mon, "RETRY_MAX_AGE_SEC", 1.0)
        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.0)
        retrier = mon._HeartbeatRetrier()
        now = mon.time.monotonic()
        item = self._item(now - 0.5)
        retrier._pending = {item.key: item}
        retrier._run()
        assert retrier._pending == {}
        assert infos, "a delivered heartbeat must log its delivery"
        fmt, args = infos[0]
        assert fmt == "Heartbeat to %s on monitor %s delivered on attempt %d after %.0fs"
        assert args[0:3] == ("prov", "job", 2)
        assert args[3] == pytest.approx(0.5, abs=1.0)

    def test_flush_attempts_each_pending_item_once_within_its_budget(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: calls.append(1) or True)
        retrier = mon._HeartbeatRetrier()
        now = mon.time.monotonic()
        a = self._item(now, key=("e", "a"))
        b = self._item(now, key=("e", "b"))
        retrier._pending = {a.key: a, b.key: b}
        retrier.flush(budget=30.0)
        assert calls == [1, 1]
        assert retrier._pending == {}

    def test_flush_with_no_budget_attempts_nothing(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: calls.append(1) or True)
        retrier = mon._HeartbeatRetrier()
        item = self._item(mon.time.monotonic())
        retrier._pending = {item.key: item}
        retrier.flush(budget=0.0)
        assert calls == []
        assert retrier._pending == {item.key: item}


class TestHeartbeatTargets:
    """Each provider's endpoint, payload and query: healthchecks.io takes the key in the path and the data as text, cronitor.io takes
    the state (complete at status 0, else the status) and the message as a query, and an unknown provider has no endpoint
    (monitoring 282-322)."""

    def test_healthchecks_with_a_key_puts_it_in_the_path_and_the_data_as_text(self):
        from pyutilz.system import monitoring as mon

        endpoint, data, params = mon._heartbeat_target("job1", 0, {"x": 1}, "healthchecks.io", "KEY")
        assert endpoint == "https://hc-ping.com/KEY/job1/0"
        assert data == "{'x': 1}"
        assert params is None

    def test_healthchecks_without_a_key_omits_it_from_the_path(self):
        from pyutilz.system import monitoring as mon

        endpoint, data, params = mon._heartbeat_target("job1", 3, None, "healthchecks.io", None)
        assert endpoint == "https://hc-ping.com/job1/3"
        assert data is None
        assert params is None

    def test_cronitor_reports_complete_at_status_zero_and_the_status_otherwise(self):
        from pyutilz.system import monitoring as mon

        endpoint, data, params = mon._heartbeat_target("job1", 0, None, "cronitor.io", "KEY")
        assert endpoint == "https://cronitor.link/p/KEY/job1"
        assert params == {"state": "complete"}
        _, _, failing = mon._heartbeat_target("job1", 2, None, "cronitor.io", "KEY")
        assert failing == {"state": 2}

    def test_cronitor_carries_the_message_as_a_query_parameter(self):
        from pyutilz.system import monitoring as mon

        _, _, params = mon._heartbeat_target("job1", 0, {"m": 1}, "cronitor.io", "KEY")
        assert params == {"state": "complete", "msg": "{'m': 1}"}

    def test_an_unknown_provider_has_no_endpoint(self):
        from pyutilz.system import monitoring as mon

        assert mon._heartbeat_target("job1", 0, {"x": 1}, "nowhere", "KEY") == ("", {"x": 1}, None)


class _Retrier:
    def __init__(self):
        self.submitted: list = []
        self.forgotten: list = []

    def submit(self, item):
        self.submitted.append(item)

    def forget(self, key):
        self.forgotten.append(key)

    def clear(self):
        self.submitted.clear()
        self.forgotten.clear()


class TestJobCompletedDelivery:
    """A job_completed send with no endpoint logs and sends nothing; a delivered send forgets its stream; a transient failure is queued for
    retry unless retry is off; a refused one is not; and a non-blocking send goes through the executor (monitoring 325-365)."""

    def _setup(self, monkeypatch, attempt_result):
        from pyutilz.system import monitoring as mon

        retrier = _Retrier()
        monkeypatch.setattr(mon, "_RETRIER", retrier)
        monkeypatch.setattr(mon, "_RETRY_ENABLED", True)
        sent: list = []
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: sent.append(a) or attempt_result)
        return mon, retrier, sent

    def test_a_job_without_an_endpoint_logs_the_credentials_hint_and_sends_nothing(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"info": lambda self, fmt, *a, **k: infos.append((fmt, a)), "warning": lambda self, *a, **k: None, "debug": lambda self, *a, **k: None})())
        mon, retrier, sent = self._setup(monkeypatch, True)
        mon.job_completed("job1", provider="nowhere")
        assert sent == []
        assert infos == [("No endpoint established for job %s. Check if monitoring credentials are properly configured.", ("job1",))]

    def test_a_delivered_send_forgets_its_stream_and_queues_nothing(self, monkeypatch):
        mon, retrier, sent = self._setup(monkeypatch, True)
        mon.job_completed("job1", status=0, provider="healthchecks.io", api_key="K")
        assert len(sent) == 1
        assert retrier.submitted == []
        assert retrier.forgotten == [("https://hc-ping.com/K/job1/0", "")]

    def test_a_transient_failure_is_queued_for_retry(self, monkeypatch):
        mon, retrier, sent = self._setup(monkeypatch, False)
        mon.job_completed("job1", status=0, provider="healthchecks.io", api_key="K")
        assert len(retrier.submitted) == 1
        assert retrier.submitted[0].endpoint == "https://hc-ping.com/K/job1/0"
        assert retrier.forgotten == []

    def test_a_failure_is_not_queued_when_retry_is_off(self, monkeypatch):
        mon, retrier, sent = self._setup(monkeypatch, False)
        mon.job_completed("job1", status=0, provider="healthchecks.io", api_key="K", retry=False)
        assert retrier.submitted == []

    def test_the_stream_key_carries_the_cronitor_state(self, monkeypatch):
        mon, retrier, sent = self._setup(monkeypatch, True)
        mon.job_completed("job1", status=0, provider="cronitor.io", api_key="K")
        assert retrier.forgotten == [("https://cronitor.link/p/K/job1", "complete")]

    def test_a_non_blocking_send_goes_through_the_executor(self, monkeypatch):
        mon, retrier, sent = self._setup(monkeypatch, True)
        submitted: list = []
        monkeypatch.setattr(mon, "_TIMEOUT_EXECUTOR", type("E", (), {"submit": lambda self, fn: submitted.append(fn)})())
        mon.job_completed("job1", provider="healthchecks.io", api_key="K", blocking=False)
        assert sent == []
        assert len(submitted) == 1
        submitted[0]()
        assert len(sent) == 1


def _quiet_logger(sink=None):
    """A stand-in logger whose info() records its arguments into sink (or discards them)."""

    class _Logger:
        def info(self, *args, **kwargs):
            if sink is not None:
                sink.append(args)

        def warning(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    return _Logger()


class TestMonitoredDecorator:
    """monitored() times the call, adds the duration to a dict result (None becomes {}), refuses a non-dict/non-None result,
    skips the heartbeat for a falsy result when should_have_data is set, and reports via job_completed with the job id
    or the function name (monitoring.py 385-455)."""

    def _record(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []
        monkeypatch.setattr(mon, "job_completed", lambda **kw: calls.append(kw))
        monkeypatch.setattr(mon, "logger", _quiet_logger())
        return mon, calls

    def test_a_dict_result_gets_the_duration_and_is_reported(self, monkeypatch):
        mon, calls = self._record(monkeypatch)
        monkeypatch.setattr(mon, "timer", iter([10.0, 12.5]).__next__)

        @mon.monitored(job_id="nightly", status=0, provider="healthchecks.io", api_key="K")
        def job():
            return {"rows": 3}

        assert job() == {"rows": 3, "duration": 2.5}
        assert calls == [{"job_id": "nightly", "status": 0, "data": {"rows": 3, "duration": 2.5}, "provider": "healthchecks.io", "api_key": "K"}]

    def test_none_becomes_an_empty_dict_carrying_the_duration(self, monkeypatch):
        mon, calls = self._record(monkeypatch)
        monkeypatch.setattr(mon, "timer", iter([1.0, 1.0001]).__next__)

        @mon.monitored()
        def job():
            return None

        assert job() == {"duration": 0.0001}
        assert calls[0]["data"] == {"duration": 0.0001}

    def test_the_duration_is_rounded_to_the_requested_digits(self, monkeypatch):
        mon, calls = self._record(monkeypatch)
        monkeypatch.setattr(mon, "timer", iter([0.0, 1.23456]).__next__)

        @mon.monitored(duration_rounding=2)
        def job():
            return {}

        assert job() == {"duration": 1.23}

    def test_no_duration_field_leaves_the_result_untouched(self, monkeypatch):
        mon, calls = self._record(monkeypatch)

        @mon.monitored(duration_field="")
        def job():
            return {"a": 1}

        assert job() == {"a": 1}
        assert calls[0]["data"] == {"a": 1}

    def test_a_custom_duration_field_name_is_used(self, monkeypatch):
        mon, calls = self._record(monkeypatch)
        monkeypatch.setattr(mon, "timer", iter([0.0, 2.0]).__next__)

        @mon.monitored(duration_field="elapsed")
        def job():
            return {}

        assert job() == {"elapsed": 2.0}

    def test_should_have_data_skips_the_heartbeat_for_a_falsy_result(self, monkeypatch):
        mon, calls = self._record(monkeypatch)

        @mon.monitored(should_have_data=True)
        def job():
            return {}

        assert job() == {}
        assert calls == []

    def test_should_have_data_still_reports_a_truthy_result(self, monkeypatch):
        mon, calls = self._record(monkeypatch)

        @mon.monitored(should_have_data=True)
        def job():
            return {"n": 1}

        job()
        assert len(calls) == 1

    def test_without_should_have_data_a_falsy_result_is_still_reported(self, monkeypatch):
        mon, calls = self._record(monkeypatch)

        @mon.monitored()
        def job():
            return {}

        job()
        assert len(calls) == 1

    def test_a_non_dict_result_raises_assertion_error_and_sends_nothing(self, monkeypatch):
        mon, calls = self._record(monkeypatch)

        @mon.monitored()
        def job():
            return [1, 2]

        with pytest.raises(AssertionError, match="expected None or dict result, got list"):
            job()
        assert calls == []

    def test_the_job_id_defaults_to_the_function_name(self, monkeypatch):
        mon, calls = self._record(monkeypatch)

        @mon.monitored()
        def my_nightly_job():
            return {}

        my_nightly_job()
        assert calls[0]["job_id"] == "my_nightly_job"

    def test_log_data_false_does_not_log_the_result(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        mon_, calls = self._record(monkeypatch)
        logged: list = []
        monkeypatch.setattr(mon, "logger", _quiet_logger(logged))

        @mon.monitored(log_data=False)
        def job():
            return {"x": 1}

        job()
        assert logged == []
        assert len(calls) == 1

    def test_log_data_true_logs_the_result(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        mon_, calls = self._record(monkeypatch)
        logged: list = []
        monkeypatch.setattr(mon, "logger", _quiet_logger(logged))

        @mon.monitored()
        def job():
            return {"x": 1}

        job()
        assert logged and logged[0][0] == {"x": 1, "duration": logged[0][0]["duration"]}


class TestMonitoringConstantsAndImports:
    """Pins the timeouts, retry schedule and transient-status set (monitoring.py 36-95) as whole values, and loads a fresh copy
    of the module to check the optional requests import and the exit-time executor shutdown."""

    def test_the_request_timeout_and_pool_constants_are_pinned(self):
        from pyutilz.system import monitoring as mon

        assert mon.API_TIMEOUT_SEC == 15
        assert mon.EXIT_FLUSH_BUDGET_SEC == 10.0
        assert mon._TIMEOUT_EXECUTOR._max_workers == 10

    def test_the_retry_schedule_constants_are_pinned(self):
        from pyutilz.system import monitoring as mon

        assert mon.RETRY_INITIAL_DELAY_SEC == 5.0
        assert mon.RETRY_MAX_DELAY_SEC == 300.0
        assert mon.RETRY_MAX_AGE_SEC == 21600.0  # six hours

    def test_the_transient_statuses_are_exactly_timeouts_throttling_and_server_errors(self):
        from pyutilz.system import monitoring as mon

        assert mon._TRANSIENT_STATUSES == frozenset({408, 425, 429}) | frozenset(range(500, 600))
        for status in (407, 409, 424, 426, 428, 430, 499, 600):
            assert status not in mon._TRANSIENT_STATUSES

    def test_the_backoff_doubles_and_is_capped(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon.random, "uniform", lambda a, b: 1.0)
        assert mon._backoff_delay(1) == 5.0
        assert mon._backoff_delay(2) == 10.0
        assert mon._backoff_delay(3) == 20.0
        assert mon._backoff_delay(10) == 300.0

    def _fresh_module(self, monkeypatch, name):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "src" / "pyutilz" / "system" / "monitoring.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        return spec, module

    def test_a_missing_requests_package_is_logged_and_leaves_requests_none(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "requests", None)
        spec, module = self._fresh_module(monkeypatch, "monitoring_no_requests")
        debugs: list = []
        original_getlogger = __import__("logging").getLogger

        def _capture(name=None):
            real = original_getlogger(name)

            class _Probe:
                def debug(self, fmt, *args, **kwargs):
                    debugs.append((fmt, args))

                def __getattr__(self, attr):
                    return getattr(real, attr)

            return _Probe()

        monkeypatch.setattr("logging.getLogger", _capture)
        spec.loader.exec_module(module)
        assert module.requests is None
        assert debugs, "the optional-import probe must log at debug level"
        fmt, args = debugs[0]
        assert fmt == "requests unavailable, job_completed's heartbeat send will fail if used: %s"
        assert len(args) == 1 and isinstance(args[0], Exception)

    def test_the_exit_hook_shuts_the_executor_down_without_waiting(self, monkeypatch):
        import atexit

        registered: list = []
        monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append((fn, a, k)))
        spec, module = self._fresh_module(monkeypatch, "monitoring_atexit")
        spec.loader.exec_module(module)
        shutdowns = [(fn, kwargs) for fn, _args, kwargs in registered if getattr(fn, "__name__", "") == "shutdown"]
        assert len(shutdowns) == 1, "the shared executor must be registered for shutdown at exit"
        fn, kwargs = shutdowns[0]
        assert kwargs == {"wait": False}
        module._TIMEOUT_EXECUTOR.shutdown(wait=False)


class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class _FakeRequests:
    def __init__(self, status=None, exc=None):
        self.status = status
        self.exc = exc
        self.calls: list = []

    def post(self, endpoint, data=None, params=None, timeout=None):
        self.calls.append((endpoint, timeout))
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.status)


class TestAttemptStatusAndErrors:
    """_attempt's verdict (True = nothing left to retry, False = transient) per status, the missing-requests branch and the
    no-message logging of a transport error (monitoring.py 96-130)."""

    def _run(self, monkeypatch, fake):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "requests", fake)
        return mon._attempt("https://x/ping", None, None, "healthchecks.io", "job")

    def test_delivered_and_refused_statuses_do_not_retry(self, monkeypatch):
        for status in (200, 201, 403, 404, 409, 407):
            assert self._run(monkeypatch, _FakeRequests(status=status)) is True, status

    def test_timeouts_throttling_and_server_errors_retry(self, monkeypatch):
        for status in (408, 425, 429, 500, 503, 599):
            assert self._run(monkeypatch, _FakeRequests(status=status)) is False, status

    def test_a_non_ok_status_other_than_the_silent_ones_is_logged_with_its_body(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        warned: list = []
        monkeypatch.setattr(
            mon,
            "logger",
            type(
                "L",
                (),
                {
                    "warning": lambda self, fmt, *a: warned.append((fmt, a)),
                    "info": lambda self, *a, **k: None,
                    "debug": lambda self, *a, **k: None,
                    "error": lambda self, *a, **k: None,
                },
            )(),
        )
        self._run(monkeypatch, _FakeRequests(status=500))
        assert warned and warned[0][1][0] == 500 and "job" in warned[0][1]

    def test_a_silent_status_is_not_logged(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        warned: list = []
        monkeypatch.setattr(
            mon,
            "logger",
            type(
                "L",
                (),
                {
                    "warning": lambda self, *a, **k: warned.append(a),
                    "info": lambda self, *a, **k: None,
                    "debug": lambda self, *a, **k: None,
                    "error": lambda self, *a, **k: None,
                },
            )(),
        )
        for status in (200, 403, 429):
            self._run(monkeypatch, _FakeRequests(status=status))
        assert warned == []

    def test_a_missing_requests_package_is_not_retried_and_says_why(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        warned: list = []
        monkeypatch.setattr(
            mon,
            "logger",
            type(
                "L",
                (),
                {
                    "warning": lambda self, fmt, *a: warned.append(a),
                    "info": lambda self, *a, **k: None,
                    "debug": lambda self, *a, **k: None,
                    "error": lambda self, *a, **k: None,
                },
            )(),
        )
        monkeypatch.setattr(mon, "requests", None)
        assert mon._attempt("https://x/ping", None, None, "healthchecks.io", "job") is True
        assert warned and "requires requests" in str(warned[0][-1])

    def test_a_transport_error_is_retried_and_logged_by_class_name_only(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        warned: list = []
        monkeypatch.setattr(
            mon,
            "logger",
            type(
                "L",
                (),
                {
                    "warning": lambda self, fmt, *a: warned.append(a),
                    "info": lambda self, *a, **k: None,
                    "debug": lambda self, *a, **k: None,
                    "error": lambda self, *a, **k: None,
                },
            )(),
        )
        ping_url = "https://hc-ping.com/PLACEHOLDER-PING-KEY/job"
        assert self._run(monkeypatch, _FakeRequests(exc=OSError(ping_url))) is False
        assert warned and warned[0][-1] == "OSError"
        assert not any(ping_url in str(part) for part in warned[0])

    def test_the_timeout_is_passed_to_requests(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        fake = _FakeRequests(status=200)
        monkeypatch.setattr(mon, "requests", fake)
        mon._attempt("https://x/ping", None, None, "healthchecks.io", "job")
        assert fake.calls == [("https://x/ping", mon.API_TIMEOUT_SEC)]


class TestRetrierThreadAndQueue:
    """The retrier's own thread: named, daemon, started on submit, reused while alive, woken by forget, and refuses pickling
    (monitoring.py 165-185)."""

    def _retrier_with(self, monkeypatch, delay=60.0):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: delay)
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: False)
        return mon, mon._HeartbeatRetrier()

    def _item(self, mon, key_suffix="a"):
        key = ("https://x/" + key_suffix, "")
        return mon._Pending(key, key[0], None, None, "healthchecks.io", key_suffix), key

    def test_submit_starts_one_named_daemon_thread(self, monkeypatch):
        mon, retrier = self._retrier_with(monkeypatch)
        item, key = self._item(mon)
        retrier.submit(item)
        thread = retrier._thread
        assert thread is not None and thread.is_alive()
        assert thread.name == "heartbeat-retry"
        assert thread.daemon is True
        retrier.forget(key)

    def test_a_second_submit_reuses_the_live_thread(self, monkeypatch):
        mon, retrier = self._retrier_with(monkeypatch)
        first, key1 = self._item(mon, "a")
        second, key2 = self._item(mon, "b")
        retrier.submit(first)
        thread = retrier._thread
        retrier.submit(second)
        assert retrier._thread is thread
        retrier.forget(key1)
        retrier.forget(key2)

    def test_forget_wakes_the_waiting_thread_so_it_exits_promptly(self, monkeypatch):
        mon, retrier = self._retrier_with(monkeypatch, delay=60.0)
        item, key = self._item(mon)
        retrier.submit(item)
        thread = retrier._thread
        retrier.forget(key)
        thread.join(timeout=3.0)
        assert not thread.is_alive(), "forget must notify the retry thread; it was left sleeping out its 60s backoff"
        assert retrier._thread is None

    def test_a_pending_item_is_attempted_again_once_due(self, monkeypatch):
        import threading

        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.01)
        attempted = threading.Event()
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: attempted.set() or True)
        retrier = mon._HeartbeatRetrier()
        item, key = self._item(mon)
        retrier.submit(item)
        assert attempted.wait(timeout=3.0), "the retry thread never re-sent the pending heartbeat"

    def test_a_retrier_refuses_to_be_pickled(self, monkeypatch):
        import pickle

        mon, retrier = self._retrier_with(monkeypatch)
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(retrier)


class TestNextDueSchedule:
    """_next_due decides what is due, what waits and what expired, against an explicit clock (monitoring.py 200-214)."""

    def _pending(self, mon, suffix, first_failed_at, next_at):
        key = ("https://x/" + suffix, "")
        item = mon._Pending(key, key[0], None, None, "healthchecks.io", suffix)
        item.first_failed_at = first_failed_at
        item.next_at = next_at
        return item

    def test_an_item_exactly_at_the_age_limit_is_kept_and_one_past_it_is_dropped(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        gave_up: list = []
        monkeypatch.setattr(mon._HeartbeatRetrier, "_log_gave_up", staticmethod(lambda item, now: gave_up.append(item.job_id)))
        now = 100_000.0
        retrier = mon._HeartbeatRetrier()
        at_limit = self._pending(mon, "edge", now - mon.RETRY_MAX_AGE_SEC, now - 1)
        past_limit = self._pending(mon, "old", now - mon.RETRY_MAX_AGE_SEC - 1, now - 1)
        retrier._pending[at_limit.key] = at_limit
        retrier._pending[past_limit.key] = past_limit
        due, soonest = retrier._next_due(now)
        assert due == [at_limit]
        assert past_limit.key not in retrier._pending
        assert gave_up == ["old"]

    def test_only_items_whose_time_has_come_are_due(self):
        from pyutilz.system import monitoring as mon

        now = 1000.0
        retrier = mon._HeartbeatRetrier()
        ready = self._pending(mon, "ready", now, now)
        later = self._pending(mon, "later", now, now + 10)
        retrier._pending[ready.key] = ready
        retrier._pending[later.key] = later
        due, soonest = retrier._next_due(now)
        assert due == [ready]

    def test_the_wait_until_the_next_item_is_the_shortest_remaining_time(self):
        from pyutilz.system import monitoring as mon

        now = 1000.0
        retrier = mon._HeartbeatRetrier()
        slow = self._pending(mon, "slow", now, now + 10)
        fast = self._pending(mon, "fast", now, now + 3)
        retrier._pending[slow.key] = slow
        retrier._pending[fast.key] = fast
        due, soonest = retrier._next_due(now)
        assert due == []
        assert soonest == 3.0

    def test_nothing_pending_means_no_wait(self):
        from pyutilz.system import monitoring as mon

        due, soonest = mon._HeartbeatRetrier()._next_due(1000.0)
        assert due == [] and soonest is None


class TestRetrierRedeliveryAndFlush:
    """The retry thread re-sends each pending heartbeat with its own endpoint/data/params/provider/job id, drops one that was
    replaced while its request was in flight without stopping the batch, and flush() re-sends within the budget and never
    raises. wait_idle()/pending_count()/clear() report and reset the queue (monitoring.py 214-300)."""

    def _item(self, mon, suffix, data="payload", params=None):
        key = ("https://x/" + suffix, "complete")
        return mon._Pending(key, key[0], data, params, "healthchecks.io", suffix), key

    def test_the_retry_thread_resends_with_the_items_own_arguments_in_order(self, monkeypatch):
        import threading

        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.01)
        seen: list = []
        done = threading.Event()

        def _fake(*args, **kwargs):
            seen.append(args)
            done.set()
            return True

        monkeypatch.setattr(mon, "_attempt", _fake)
        retrier = mon._HeartbeatRetrier()
        item, key = self._item(mon, "order", data="payload", params={"state": "complete"})
        retrier.submit(item)
        assert done.wait(timeout=3.0)
        assert seen == [("https://x/order", "payload", {"state": "complete"}, "healthchecks.io", "order")]

    def test_a_failed_redelivery_is_attempted_again_and_success_clears_it(self, monkeypatch):
        import threading

        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.01)
        outcomes = iter([False, True])
        attempts: list = []
        done = threading.Event()

        def _fake(*args, **kwargs):
            attempts.append(args[-1])
            result = next(outcomes)
            if result:
                done.set()
            return result

        monkeypatch.setattr(mon, "_attempt", _fake)
        retrier = mon._HeartbeatRetrier()
        item, key = self._item(mon, "again")
        retrier.submit(item)
        assert done.wait(timeout=3.0)
        assert attempts == ["again", "again"]
        assert retrier.wait_idle(timeout=3.0)
        assert retrier.pending_count() == 0

    def test_an_item_replaced_during_its_request_is_left_alone_and_the_batch_continues(self, monkeypatch):
        import threading

        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.01)
        attempted: list = []
        second_done = threading.Event()
        retrier = mon._HeartbeatRetrier()
        first, key_a = self._item(mon, "a")
        second, key_b = self._item(mon, "b")
        replacement, _ = self._item(mon, "a")

        def _fake(endpoint, data, params, provider, job_id, **kwargs):
            attempted.append(job_id)
            if job_id == "a" and attempted.count("a") == 1:
                retrier.submit(replacement)
                return False
            if job_id == "b":
                second_done.set()
            return False

        monkeypatch.setattr(mon, "_attempt", _fake)
        retrier._pending[key_a] = first
        retrier._pending[key_b] = second
        retrier.submit(first)
        assert second_done.wait(timeout=3.0), "a replaced item must not stop the rest of the batch"
        retrier.forget(key_b)
        with retrier._cv:
            assert retrier._pending.get(key_a) is replacement

    def test_flush_gives_each_pending_heartbeat_one_attempt_within_the_budget(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: calls.append((a[-1], k["timeout"])) or True)
        retrier = mon._HeartbeatRetrier()
        item, key = self._item(mon, "flush")
        retrier._pending[key] = item
        retrier.flush(budget=100.0)
        assert calls and calls[0][0] == "flush" and calls[0][1] == mon.API_TIMEOUT_SEC
        assert retrier.pending_count() == 0

    def test_flush_passes_only_the_remaining_budget_as_timeout(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: calls.append(k["timeout"]) or False)
        retrier = mon._HeartbeatRetrier()
        item, key = self._item(mon, "tight")
        retrier._pending[key] = item
        retrier.flush(budget=2.0)
        assert len(calls) == 1
        assert 0 < calls[0] <= 2.0

    def test_flush_with_no_budget_attempts_nothing(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []
        monkeypatch.setattr(mon, "_attempt", lambda *a, **k: calls.append(1) or True)
        retrier = mon._HeartbeatRetrier()
        item, key = self._item(mon, "none")
        retrier._pending[key] = item
        retrier.flush(budget=0.0)
        assert calls == []
        assert retrier.pending_count() == 1

    def test_flush_never_raises_and_logs_the_failure_at_debug(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        def _boom(*a, **k):
            raise RuntimeError("interpreter going down")

        debugs: list = []
        monkeypatch.setattr(mon, "_attempt", _boom)
        monkeypatch.setattr(mon, "logger", type("L", (), {"debug": lambda self, fmt, *a: debugs.append((fmt, a)), "info": lambda self, *a, **k: None, "warning": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})())
        retrier = mon._HeartbeatRetrier()
        item, key = self._item(mon, "boom")
        retrier._pending[key] = item
        retrier.flush()
        assert debugs and debugs[0][0] == "heartbeat flush at exit failed: %s"
        assert "interpreter going down" in str(debugs[0][1][0])

    def test_wait_idle_is_false_while_pending_and_true_once_empty(self):
        from pyutilz.system import monitoring as mon

        retrier = mon._HeartbeatRetrier()
        item, key = self._item(mon, "idle")
        retrier._pending[key] = item
        assert retrier.wait_idle(timeout=0.05) is False
        retrier.clear()
        assert retrier.pending_count() == 0
        assert retrier.wait_idle(timeout=0.05) is True

    def test_clear_drops_everything_and_wakes_the_waiters(self):
        from pyutilz.system import monitoring as mon

        retrier = mon._HeartbeatRetrier()
        for suffix in ("x", "y"):
            item, key = self._item(mon, suffix)
            retrier._pending[key] = item
        assert retrier.pending_count() == 2
        retrier.clear()
        assert retrier.pending_count() == 0


class TestTimeoutWrapperLogCalls:
    """timeout_wrapper's logger calls: the exact timeout message and its arguments, the exception message with its exc_info,
    the per-call thread name, and the duration report (monitoring.py 455-500)."""

    def _recording_logger(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []

        class _Logger:
            def error(self, fmt, *args, **kwargs):
                calls.append(("error", fmt, args, kwargs))

            def info(self, fmt, *args, **kwargs):
                calls.append(("info", fmt, args, kwargs))

            def warning(self, *args, **kwargs):
                return None

            def debug(self, *args, **kwargs):
                return None

        monkeypatch.setattr(mon, "logger", _Logger())
        return mon, calls

    def test_a_timeout_logs_the_function_name_limit_and_time(self, monkeypatch):
        import threading

        mon, calls = self._recording_logger(monkeypatch)
        release = threading.Event()

        def slow_job():
            release.wait(timeout=5)

        wrapped = mon.timeout_wrapper(timeout=0.05)(slow_job)
        assert wrapped() is None
        release.set()
        errors = [c for c in calls if c[0] == "error"]
        assert len(errors) == 1
        _, fmt, args, _ = errors[0]
        assert fmt == "%s timed out after %ss at %s"
        assert args[0] == "slow_job" and args[1] == 0.05
        assert hasattr(args[2], "year"), "the third argument is the wall-clock time of the timeout"

    def test_an_exception_logs_the_message_and_passes_the_exception_as_exc_info(self, monkeypatch):
        mon, calls = self._recording_logger(monkeypatch)
        boom = ValueError("bad input")

        def failing_job():
            raise boom

        wrapped = mon.timeout_wrapper(timeout=5)(failing_job)
        assert wrapped() is None
        errors = [c for c in calls if c[0] == "error"]
        assert len(errors) == 1
        _, fmt, args, kwargs = errors[0]
        assert fmt == "Error in %s: %s"
        assert args[0] == "failing_job" and args[1] is boom
        assert kwargs == {"exc_info": boom}

    def test_the_worker_thread_is_named_after_the_function(self, monkeypatch):
        import threading

        mon, _calls = self._recording_logger(monkeypatch)
        names: list = []

        def named_job():
            names.append(threading.current_thread().name)
            return 1

        assert mon.timeout_wrapper(timeout=5)(named_job)() == 1
        assert names == ["timeout_wrapper-named_job"]

    def test_the_duration_report_carries_the_name_and_two_decimals(self, monkeypatch):
        mon, calls = self._recording_logger(monkeypatch)

        def quick():
            return 7

        assert mon.timeout_wrapper(timeout=5, report_actual_duration=True)(quick)() == 7
        infos = [c for c in calls if c[0] == "info"]
        assert len(infos) == 1
        _, fmt, args, _ = infos[0]
        assert fmt == "%s completed in %.2fs"
        assert args[0] == "quick" and isinstance(args[1], float) and args[1] >= 0


class TestLogDurationMessage:
    """log_duration logs only past its threshold, with the function name, its positional and keyword arguments, the
    duration to two decimals, and large arguments truncated to max_arg_size (monitoring.py 500-540)."""

    def _capture(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []

        class _Logger:
            def info(self, fmt, *args, **kwargs):
                infos.append((fmt, args))

            def debug(self, *args, **kwargs):
                return None

            def warning(self, *args, **kwargs):
                return None

            def error(self, *args, **kwargs):
                return None

        monkeypatch.setattr(mon, "logger", _Logger())
        monkeypatch.setattr(mon, "timer", iter([0.0, 2.5]).__next__)
        return mon, infos

    def test_a_slow_call_logs_name_arguments_and_duration(self, monkeypatch):
        mon, infos = self._capture(monkeypatch)

        @mon.log_duration(threshold=1.0)
        def compute(a, b, scale=1):
            return a + b

        assert compute(1, 2, scale=3) == 3
        assert infos == [("%s%s took %.2f s.", ("compute", "(1, 2, scale=3)", 2.5))]

    def test_a_call_with_no_arguments_logs_empty_parentheses(self, monkeypatch):
        mon, infos = self._capture(monkeypatch)

        @mon.log_duration(threshold=1.0)
        def idle():
            return None

        idle()
        assert infos == [("%s%s took %.2f s.", ("idle", "()", 2.5))]

    def test_a_fast_call_logs_nothing(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        mon_, infos = self._capture(monkeypatch)
        monkeypatch.setattr(mon, "timer", iter([0.0, 0.5]).__next__)

        @mon.log_duration(threshold=1.0)
        def quick():
            return None

        quick()
        assert infos == []

    def test_a_large_argument_is_truncated_in_the_middle(self, monkeypatch):
        mon, infos = self._capture(monkeypatch)

        @mon.log_duration(threshold=1.0, max_arg_size=10)
        def take(text):
            return None

        take("x" * 30)
        _, args = infos[0]
        assert args[1] == "('xxxx...[truncated 22 chars]...xxxx')"
