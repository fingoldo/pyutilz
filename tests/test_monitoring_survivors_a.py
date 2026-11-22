"""Mutation-survivor pins for monitoring: heartbeat, retrier and job-completion delivery, and the monitored decorator."""

import pytest


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
        flushes = [fn for fn, _args, _kwargs in registered if getattr(fn, "__name__", "") == "flush"]
        assert len(flushes) == 1, "the retry queue must be flushed at exit"
        fn, kwargs = shutdowns[0]
        assert kwargs == {"wait": False}
        module._TIMEOUT_EXECUTOR.shutdown(wait=False)

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
