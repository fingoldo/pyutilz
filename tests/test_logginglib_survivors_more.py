"""OpenAI-compatible provider tests: logging setup continued."""

import logging
import pytest
from pyutilz.dev import logginglib as ll
from tests.logginglib_survivors_support import (
    _FakeRedis,
    _InfoLog,
    _ThrottleRecorder,
    _record,
)


class TestDebugged:
    def test_a_successful_call_returns_its_value(self):
        @ll.debugged()
        def f(x):
            return x + 1

        assert f(1) == 2

    def test_a_failure_is_reraised_when_not_interactive(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(ll.sys, "stdin", None)

        @ll.debugged(max_retries=3)
        def f():
            calls.append(1)
            raise ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            f()
        assert len(calls) == 1

    def test_an_interactive_failure_retries_up_to_the_cap(self, monkeypatch):
        calls: list = []
        breakpoints: list = []

        class _Tty:
            def isatty(self):
                return True

        monkeypatch.setattr(ll.sys, "stdin", _Tty())
        import pdb

        monkeypatch.setattr(pdb, "set_trace", lambda: breakpoints.append(1))

        @ll.debugged(max_retries=3)
        def f():
            calls.append(1)
            raise ValueError("bad")

        with pytest.raises(ValueError):
            f()
        assert len(calls) == 3
        assert len(breakpoints) == 2

    def test_an_interactive_failure_that_recovers_returns_the_value(self, monkeypatch):
        attempts: list = []

        class _Tty:
            def isatty(self):
                return True

        monkeypatch.setattr(ll.sys, "stdin", _Tty())
        import pdb

        monkeypatch.setattr(pdb, "set_trace", lambda: None)

        @ll.debugged(max_retries=3)
        def f():
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("first try")
            return "ok"

        assert f() == "ok"
        assert len(attempts) == 2

    def test_each_failure_is_logged_with_the_logger(self, monkeypatch):
        logged: list = []

        class _Recorder:
            def exception(self, e):
                logged.append(e)

        monkeypatch.setattr(ll, "logger", _Recorder())
        monkeypatch.setattr(ll.sys, "stdin", None)

        @ll.debugged()
        def f():
            raise ValueError("bad")

        with pytest.raises(ValueError):
            f()
        assert len(logged) == 1 and isinstance(logged[0], ValueError)


class TestLogThrottleWindow:
    def test_the_first_call_for_a_key_is_admitted(self, clock):
        assert ll.log_throttle("k") is True

    def test_a_second_call_inside_the_window_is_refused(self, clock):
        ll.log_throttle("k")
        clock.now += 59.0
        assert ll.log_throttle("k") is False

    def test_the_window_reopens_exactly_at_the_interval(self, clock):
        ll.log_throttle("k", min_interval=60.0)
        clock.now += 60.0
        assert ll.log_throttle("k", min_interval=60.0) is True

    def test_the_default_interval_is_sixty_seconds(self, clock):
        ll.log_throttle("k")
        clock.now += 59.9
        assert ll.log_throttle("k") is False
        clock.now += 0.1
        assert ll.log_throttle("k") is True

    def test_a_first_call_at_monotonic_zero_is_still_admitted(self, clock):
        clock.now = 0.0
        assert ll.log_throttle("fresh") is True

    def test_each_key_has_its_own_window(self, clock):
        ll.log_throttle("a")
        assert ll.log_throttle("b") is True

    def test_a_refused_call_does_not_restart_the_window(self, clock):
        ll.log_throttle("k", min_interval=60.0)
        clock.now += 30.0
        ll.log_throttle("k", min_interval=60.0)
        clock.now += 30.0
        assert ll.log_throttle("k", min_interval=60.0) is True


class TestLogThrottled:
    def test_the_first_max_count_occurrences_are_logged(self, clock):
        rec = _ThrottleRecorder()
        for i in range(5):
            ll.log_throttled(rec, "site", 30, "row %s", i, max_count=5)
        assert [c[1] for c in rec.calls[:5]] == ["row %s"] * 5
        assert [c[2] for c in rec.calls[:5]] == [(0,), (1,), (2,), (3,), (4,)]

    def test_the_last_allowed_line_is_followed_by_one_suppression_notice(self, clock):
        rec = _ThrottleRecorder()
        for _ in range(9):
            ll.log_throttled(rec, "site", 30, "row", max_count=5)
        notices = [c for c in rec.calls if c[1] == "%s: further occurrences suppressed (throttled)"]
        assert len(notices) == 1
        assert notices[0][2] == ("site",)
        assert len(rec.calls) == 6

    def test_the_level_and_exc_info_are_passed_through(self, clock):
        rec = _ThrottleRecorder()
        ll.log_throttled(rec, "site", 40, "boom", exc_info=True)
        assert rec.calls[0][0] == 40
        assert rec.calls[0][3] is True

    def test_the_count_includes_suppressed_calls(self, clock):
        rec = _ThrottleRecorder()
        for _ in range(7):
            ll.log_throttled(rec, "site", 30, "row", max_count=2)
        assert ll.log_throttle_count("site") == 7

    def test_distinct_keys_are_counted_independently(self, clock):
        rec = _ThrottleRecorder()
        ll.log_throttled(rec, "a", 30, "x")
        ll.log_throttled(rec, "a", 30, "x")
        ll.log_throttled(rec, "b", 30, "x")
        assert (ll.log_throttle_count("a"), ll.log_throttle_count("b")) == (2, 1)

    def test_an_unseen_key_has_a_zero_count(self, clock):
        assert ll.log_throttle_count("never") == 0


class TestResetLogThrottles:
    def test_a_named_reset_forgets_only_that_key(self, clock):
        ll.log_throttle("a")
        ll.log_throttle("b")
        ll.reset_log_throttles("a")
        assert ll.log_throttle("a") is True
        assert ll.log_throttle("b") is False

    def test_a_named_reset_clears_its_count(self, clock):
        rec = _ThrottleRecorder()
        ll.log_throttled(rec, "a", 30, "x")
        ll.reset_log_throttles("a")
        assert ll.log_throttle_count("a") == 0

    def test_a_reset_with_no_key_forgets_every_key(self, clock):
        ll.log_throttle("a")
        ll.log_throttle("b")
        rec = _ThrottleRecorder()
        ll.log_throttled(rec, "c", 30, "x")
        ll.reset_log_throttles()
        assert ll.log_throttle("a") is True
        assert ll.log_throttle("b") is True
        assert ll.log_throttle_count("c") == 0


class TestLogThrottledDefaults:
    def test_max_count_defaults_to_five_and_exc_info_to_false(self):
        import inspect

        params = inspect.signature(ll.log_throttled).parameters
        assert params["max_count"].default == 5
        assert params["exc_info"].default is False


class TestRedisTrimAnnouncementAndDebuggedForwarding:
    """The stdout notice on a trim, the debugged retry default, and keyword forwarding through debugged (logginglib 539-566)."""

    def test_a_trim_announces_itself_on_stdout(self, monkeypatch, capsys):
        monkeypatch.setattr(ll, "random", lambda: 0.0)
        ll.RedisHandler(_FakeRedis(), LOG_DEST="jobs", LOG_SIZE=7).emit(_record())
        assert capsys.readouterr().out == "logging list trimmed!\n"

    def test_no_announcement_when_nothing_is_trimmed(self, monkeypatch, capsys):
        monkeypatch.setattr(ll, "random", lambda: 0.5)
        ll.RedisHandler(_FakeRedis()).emit(_record())
        assert capsys.readouterr().out == ""

    def test_the_debugged_retry_cap_defaults_to_three(self):
        import inspect

        assert inspect.signature(ll.debugged).parameters["max_retries"].default == 3

    def test_keyword_arguments_reach_the_wrapped_function(self, monkeypatch):
        monkeypatch.setattr(ll.sys, "stdin", None)

        @ll.debugged()
        def add(a, b=2):
            return a + b

        assert add(1, b=5) == 6
        assert add(1) == 3


class TestInitLoggingUnrotatedAndDefaults:
    """The fallback file mode with rotation off, the stderr-reconfigure debug line, the thousands separator, and signature defaults (logginglib 86-459)."""

    @pytest.fixture
    def captured(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        calls: list = []
        monkeypatch.setattr(ll.logging, "basicConfig", lambda **kw: calls.append(kw))
        monkeypatch.setattr(ll, "logger", logging.getLogger("initial_probe"))
        yield calls
        for c in calls:
            for h in c.get("handlers", []):
                if isinstance(h, logging.FileHandler):
                    h.close()

    def test_the_fallback_file_mode_is_append_without_rotation(self, captured, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "concurrent_log_handler", None)
        ll.init_logging(forced_filename="svc.py", maxBytes=0)
        handler = next(h for h in captured[0]["handlers"] if isinstance(h, logging.FileHandler))
        assert handler.mode == "a"

    def test_a_stderr_that_refuses_reconfigure_is_reported_at_debug(self, captured, monkeypatch, caplog):
        monkeypatch.setitem(__import__("sys").modules, "concurrent_log_handler", None)

        class _Stderr:
            def reconfigure(self, **kwargs):
                raise ValueError("nope")

        monkeypatch.setattr(ll.sys, "stderr", _Stderr())
        with caplog.at_level(logging.DEBUG, logger="pyutilz.dev.logginglib"):
            ll.init_logging(forced_filename="svc.py", log_to_file=False)
        assert "Could not set errors='replace' on sys.stderr: nope" in caplog.text

    def test_the_default_format_is_pinned(self):
        import inspect

        assert inspect.signature(ll.init_logging).parameters["format"].default == (
            "%(asctime)s - %(levelname)s - %(processName)s[%(process)d]-%(threadName)s[%(thread)d]"
            "-%(filename)s-%(name)s - %(funcName)s-line:%(lineno)d - %(message)s"
        )

    def test_logged_includes_the_node_ip_by_default(self):
        import inspect

        assert inspect.signature(ll.logged).parameters["include_node_ip"].default is True

    def test_a_large_load_count_uses_underscore_thousands_separators(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows(list(range(1234)), "users", results_log={"results": {}}, lang="en", verbose=True)
        assert log_.info_calls == [("%s%s", "Loaded 1_234 rows from users DB table.", "")]
