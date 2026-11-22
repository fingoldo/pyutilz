"""Mutation-survivor pins for monitoring: retry scheduling, expiry wakeups, the timeout wrapper and the duration log message."""

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
        assert thread.is_alive()  # pending work keeps the thread in its wait, so forget() is what must wake it
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

        # items other tests left on the module-level retrier would reach this fake and consume its outcomes
        for stale in list(mon._RETRIER._pending):
            mon._RETRIER.forget(stale)
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
        assert args[0] == "quick" and isinstance(args[1], float) and 0 <= args[1] < 5

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

class TestSurvivorGapsFromSweep:
    """Pins what the monitoring sweep found unpinned: the defaults of the public entry points, the exact arguments handed to
    each attempt and queued retry, the failure and attempt counters, the wait and flush boundaries, the daemon flag, and the
    log_duration threshold, size limit and separators (monitoring.py 77-390, 412, 474, 493-537)."""

    def test_the_retry_switch_is_on_by_default(self):
        # The test session switches retries off through a fixture, so the default is read from a fresh copy of the module.
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "src" / "pyutilz" / "system" / "monitoring.py"
        spec = importlib.util.spec_from_file_location("monitoring_default_probe", path)
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        assert fresh._RETRY_ENABLED is True

    def test_backoff_jitter_scales_the_base_by_the_random_factor(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon.random, "uniform", lambda a, b: 0.8)
        assert mon._backoff_delay(1) == pytest.approx(4.0)
        monkeypatch.setattr(mon.random, "uniform", lambda a, b: 1.2)
        assert mon._backoff_delay(1) == pytest.approx(6.0)

    def test_each_failed_redelivery_raises_the_count_by_one_and_the_backoff_follows(self, monkeypatch):
        import threading

        from pyutilz.system import monitoring as mon

        delays: list = []
        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: delays.append(failures) or 0.01)
        outcomes = iter([False, False, True])
        finished = threading.Event()

        def _fake(*args, **kwargs):
            result = next(outcomes)
            if result:
                finished.set()
            return result

        monkeypatch.setattr(mon, "_attempt", _fake)
        retrier = mon._HeartbeatRetrier()
        key = ("https://x/count", "complete")
        retrier.submit(mon._Pending(key, key[0], None, None, "healthchecks.io", "count"))
        assert finished.wait(timeout=5)
        assert retrier.wait_idle(timeout=5)
        assert delays == [1, 2, 3]

    def test_the_delivered_log_reports_the_attempt_number(self, monkeypatch):
        import threading

        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.01)
        outcomes = iter([False, True])
        done = threading.Event()

        def _fake(*args, **kwargs):
            result = next(outcomes)
            if result:
                done.set()
            return result

        monkeypatch.setattr(mon, "_attempt", _fake)
        infos: list = []
        monkeypatch.setattr(
            mon,
            "logger",
            type("L", (), {"info": lambda self, fmt, *a, **k: infos.append((fmt, a)), "warning": lambda self, *a, **k: None, "debug": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})(),
        )
        retrier = mon._HeartbeatRetrier()
        key = ("https://x/att", "complete")
        retrier.submit(mon._Pending(key, key[0], None, None, "healthchecks.io", "att"))
        assert done.wait(timeout=5)
        assert retrier.wait_idle(timeout=5)
        delivered = [a for fmt, a in infos if fmt.startswith("Heartbeat to %s on monitor %s delivered")]
        assert len(delivered) == 1
        assert delivered[0][2] == 3  # the initial send, one failed redelivery, then the successful one

    def test_the_give_up_log_reports_minutes_since_the_first_failure(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        warned: list = []
        monkeypatch.setattr(
            mon,
            "logger",
            type("L", (), {"warning": lambda self, fmt, *a: warned.append((fmt, a)), "info": lambda self, *a, **k: None, "debug": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})(),
        )
        key = ("https://x/gave", "complete")
        item = mon._Pending(key, key[0], None, None, "healthchecks.io", "gave")
        item.first_failed_at = 1000.0
        item.failures = 4
        mon._HeartbeatRetrier._log_gave_up(item, 1000.0 + 600.0)
        assert len(warned) == 1
        _fmt, args = warned[0]
        assert args[2] == 4
        assert args[3] == 10.0

    def test_wait_idle_waits_for_the_queue_to_empty(self):
        import threading
        import time

        from pyutilz.system import monitoring as mon

        best = None
        for _ in range(5):
            retrier = mon._HeartbeatRetrier()
            key = ("https://x/wait", "complete")
            retrier._pending[key] = mon._Pending(key, key[0], None, None, "healthchecks.io", "wait")
            threading.Timer(0.1, retrier.clear).start()
            started = time.monotonic()
            assert retrier.wait_idle(timeout=5.0) is True
            elapsed = time.monotonic() - started
            best = elapsed if best is None else min(best, elapsed)
        assert best < 2.0

    def test_wait_idle_returns_promptly_once_a_redelivery_succeeds(self, monkeypatch):
        import time

        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.01)
        monkeypatch.setattr(mon, "_attempt", lambda *args, **kwargs: True)
        best = None
        for _ in range(5):
            retrier = mon._HeartbeatRetrier()
            key = ("https://x/woke", "complete")
            retrier.submit(mon._Pending(key, key[0], None, None, "healthchecks.io", "woke"))
            started = time.monotonic()
            assert retrier.wait_idle(timeout=5.0) is True
            elapsed = time.monotonic() - started
            best = elapsed if best is None else min(best, elapsed)
        assert best < 2.0

    def test_a_newer_short_item_is_attempted_while_an_older_long_one_waits(self, monkeypatch):
        import threading

        from pyutilz.system import monitoring as mon

        delays = iter([60.0, 0.01])
        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: next(delays))
        done = threading.Event()
        monkeypatch.setattr(mon, "_attempt", lambda *args, **kwargs: done.set() or True)
        retrier = mon._HeartbeatRetrier()
        slow_key = ("https://x/slow", "complete")
        fast_key = ("https://x/fast", "complete")
        retrier.submit(mon._Pending(slow_key, slow_key[0], None, None, "healthchecks.io", "slow"))
        retrier.submit(mon._Pending(fast_key, fast_key[0], None, None, "healthchecks.io", "fast"))
        assert done.wait(timeout=3.0), "a newer item must wake the retry thread"
        retrier.forget(slow_key)

    def test_flush_hands_each_attempt_its_items_arguments_and_the_remaining_budget(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []
        monkeypatch.setattr(mon, "_attempt", lambda *args, **kwargs: calls.append((args, kwargs)) or True)
        retrier = mon._HeartbeatRetrier()
        key = ("https://x/fl", "complete")
        retrier._pending[key] = mon._Pending(key, key[0], "payload", {"state": "complete"}, "healthchecks.io", "fl")
        retrier.flush(budget=0.5)
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == ("https://x/fl", "payload", {"state": "complete"}, "healthchecks.io", "fl")
        assert 0 < kwargs["timeout"] <= 0.5

    def test_a_blocking_send_passes_its_arguments_to_attempt_in_order(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        recorded: list = []
        monkeypatch.setattr(mon, "_attempt", lambda *args, **kwargs: recorded.append(args) or True)
        monkeypatch.setattr(mon, "_RETRIER", _Retrier())
        mon.job_completed("job9", status=0, data="payload", provider="healthchecks.io", api_key="K")
        assert recorded == [("https://hc-ping.com/K/job9/0", "payload", None, "healthchecks.io", "job9")]

    def test_a_queued_retry_keeps_every_argument_of_the_failed_send(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        retrier = _Retrier()
        monkeypatch.setattr(mon, "_RETRIER", retrier)
        monkeypatch.setattr(mon, "_RETRY_ENABLED", True)
        monkeypatch.setattr(mon, "_attempt", lambda *args, **kwargs: False)
        mon.job_completed("job9", status=0, data="payload", provider="cronitor.io", api_key="K")
        item = retrier.submitted[0]
        assert (item.endpoint, item.data, item.params, item.provider, item.job_id) == (
            "https://cronitor.link/p/K/job9",
            "payload",
            {"state": "complete", "msg": "payload"},
            "cronitor.io",
            "job9",
        )

    def test_the_public_defaults_are_healthchecks_status_zero_and_rounding_four(self):
        import inspect

        from pyutilz.system import monitoring as mon

        job = inspect.signature(mon.job_completed).parameters
        assert job["status"].default == 0
        assert job["provider"].default == "healthchecks.io"
        wrapped = inspect.signature(mon.monitored).parameters
        assert wrapped["status"].default == 0
        assert wrapped["provider"].default == "healthchecks.io"
        assert wrapped["duration_rounding"].default == 4

    def test_the_default_heartbeat_is_healthchecks_status_zero_with_a_rounded_duration(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        calls: list = []
        monkeypatch.setattr(mon, "job_completed", lambda **kwargs: calls.append(kwargs))
        monkeypatch.setattr(mon, "timer", iter([0.0, 1.23456789]).__next__)

        @mon.monitored(log_data=False)
        def job():
            return {}

        result = job()
        assert calls[0]["status"] == 0
        assert calls[0]["provider"] == "healthchecks.io"
        assert result == {"duration": 1.2346}

    def test_a_wrapped_job_accepts_keyword_arguments(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "job_completed", lambda **kwargs: None)

        @mon.monitored(duration_field="", log_data=False)
        def job(*, n):
            return {"n": n}

        assert job(n=4) == {"n": 4}

    def test_the_worker_thread_of_timeout_wrapper_is_a_daemon(self):
        import threading

        from pyutilz.system import monitoring as mon

        flags: list = []

        def probe():
            flags.append(threading.current_thread().daemon)

        mon.timeout_wrapper(timeout=5)(probe)()
        assert flags == [True]

    def test_the_shared_pool_threads_are_named_for_timeout_wrapper(self):
        from pyutilz.system import monitoring as mon

        assert mon._TIMEOUT_EXECUTOR._thread_name_prefix == "timeout_wrapper"

    def test_log_duration_defaults_to_a_one_second_threshold_and_logs_just_above_it(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"info": lambda self, fmt, *a, **k: infos.append(a), "debug": lambda self, *a, **k: None, "warning": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})())
        monkeypatch.setattr(mon, "timer", iter([0.0, 1.05]).__next__)

        @mon.log_duration()
        def slow():
            return None

        slow()
        assert len(infos) == 1

    def test_log_duration_does_not_log_a_call_exactly_at_the_threshold(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"info": lambda self, fmt, *a, **k: infos.append(a), "debug": lambda self, *a, **k: None, "warning": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})())
        monkeypatch.setattr(mon, "timer", iter([0.0, 1.0]).__next__)

        @mon.log_duration(threshold=1.0)
        def exact():
            return None

        exact()
        assert infos == []

    def test_log_duration_measures_elapsed_time_not_the_clock_reading(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"info": lambda self, fmt, *a, **k: infos.append(a), "debug": lambda self, *a, **k: None, "warning": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})())
        monkeypatch.setattr(mon, "timer", iter([100.0, 100.5]).__next__)

        @mon.log_duration(threshold=1.0)
        def quick():
            return None

        quick()
        assert infos == []

    def test_log_duration_keeps_an_argument_of_exactly_the_limit_whole(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"info": lambda self, fmt, *a, **k: infos.append(a), "debug": lambda self, *a, **k: None, "warning": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})())
        monkeypatch.setattr(mon, "timer", iter([0.0, 2.5]).__next__)

        @mon.log_duration(threshold=1.0, max_arg_size=10)
        def take(text):
            return None

        take("abcdefgh")  # repr is exactly 10 characters
        assert infos[0][1] == "('abcdefgh')"

    def test_log_duration_truncates_by_default_above_one_thousand_characters(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"info": lambda self, fmt, *a, **k: infos.append(a), "debug": lambda self, *a, **k: None, "warning": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})())
        monkeypatch.setattr(mon, "timer", iter([0.0, 2.5]).__next__)

        @mon.log_duration(threshold=1.0)
        def take(text):
            return None

        take("x" * 999)  # repr is 1001 characters, one over the default limit
        assert "[truncated" in infos[0][1]

    def test_log_duration_separates_keyword_arguments_with_comma_and_space(self, monkeypatch):
        from pyutilz.system import monitoring as mon

        infos: list = []
        monkeypatch.setattr(mon, "logger", type("L", (), {"info": lambda self, fmt, *a, **k: infos.append(a), "debug": lambda self, *a, **k: None, "warning": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})())
        monkeypatch.setattr(mon, "timer", iter([0.0, 2.5]).__next__)

        @mon.log_duration(threshold=1.0)
        def configure(**options):
            return None

        configure(a=1, b=2)
        assert infos[0][1] == "(a=1, b=2)"

class TestRetryTimingAndExpiryWakeups:
    """The pause before a redelivery is real, an expired item wakes a waiting caller at once, and wait_idle honours a short
    budget (monitoring.py 225-273)."""

    def test_a_failed_redelivery_waits_out_its_backoff_before_the_next_attempt(self, monkeypatch):
        import threading
        import time

        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.3 if failures >= 2 else 0.01)
        stamps: list = []
        done = threading.Event()
        outcomes = iter([False, True])

        def _fake(*args, **kwargs):
            stamps.append(time.monotonic())
            result = next(outcomes)
            if result:
                done.set()
            return result

        monkeypatch.setattr(mon, "_attempt", _fake)
        retrier = mon._HeartbeatRetrier()
        key = ("https://x/gap", "complete")
        retrier.submit(mon._Pending(key, key[0], None, None, "healthchecks.io", "gap"))
        assert done.wait(timeout=5)
        assert len(stamps) == 2
        assert stamps[1] - stamps[0] >= 0.25

    def test_an_item_that_expires_wakes_a_waiting_caller_promptly(self, monkeypatch):
        import time

        from pyutilz.system import monitoring as mon

        monkeypatch.setattr(mon, "_backoff_delay", lambda failures: 0.3)
        monkeypatch.setattr(mon, "_attempt", lambda *args, **kwargs: False)
        best = None
        for _ in range(5):
            retrier = mon._HeartbeatRetrier()
            key = ("https://x/expire", "complete")
            item = mon._Pending(key, key[0], None, None, "healthchecks.io", "expire")
            retrier.submit(item)
            item.first_failed_at = time.monotonic() - mon.RETRY_MAX_AGE_SEC - 1
            started = time.monotonic()
            assert retrier.wait_idle(timeout=5.0) is True
            elapsed = time.monotonic() - started
            best = elapsed if best is None else min(best, elapsed)
        assert best < 2.0

    def test_wait_idle_with_a_short_budget_still_returns_true_once_cleared(self):
        import threading

        from pyutilz.system import monitoring as mon

        retrier = mon._HeartbeatRetrier()
        key = ("https://x/short", "complete")
        retrier._pending[key] = mon._Pending(key, key[0], None, None, "healthchecks.io", "short")
        threading.Timer(0.1, retrier.clear).start()
        assert retrier.wait_idle(timeout=0.5) is True
