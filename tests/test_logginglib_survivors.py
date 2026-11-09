"""External-IP resolution and log filename pins for dev.logginglib (mutation survivors, first group)."""

from __future__ import annotations

import logging
import os
import types
from pathlib import Path

import pytest

import pyutilz.web as web

from pyutilz.dev import logginglib as ll


class _Log:
    def __init__(self):
        self.warning_calls: list = []

    def warning(self, *args):
        self.warning_calls.append(args)


def _reset(monkeypatch):
    monkeypatch.setattr(ll, "EXTERNAL_IP", None)
    monkeypatch.setattr(ll, "_EXTERNAL_IP_RESOLVED", False)


class TestNodeExternalIp:
    def test_the_resolved_address_is_returned(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(web, "get_external_ip", lambda: "203.0.113.7")
        assert ll.get_node_external_ip() == "203.0.113.7"

    def test_resolution_happens_at_most_once_per_process(self, monkeypatch):
        _reset(monkeypatch)
        calls: list = []

        def fake():
            calls.append(1)
            return "203.0.113.7"

        monkeypatch.setattr(web, "get_external_ip", fake)
        ll.get_node_external_ip()
        ll.get_node_external_ip()
        assert len(calls) == 1

    def test_a_failed_lookup_gives_none_and_warns(self, monkeypatch):
        _reset(monkeypatch)
        log = _Log()
        monkeypatch.setattr(ll, "logger", log)

        def boom():
            raise OSError("no route")

        monkeypatch.setattr(web, "get_external_ip", boom)
        assert ll.get_node_external_ip() is None
        assert log.warning_calls == [("Could not resolve this node's external IP for logging: %s", log.warning_calls[0][1])]
        assert "no route" in str(log.warning_calls[0][1])

    def test_a_failed_lookup_is_not_retried(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(ll, "logger", _Log())
        calls: list = []

        def boom():
            calls.append(1)
            raise OSError("down")

        monkeypatch.setattr(web, "get_external_ip", boom)
        ll.get_node_external_ip()
        ll.get_node_external_ip()
        assert len(calls) == 1


class TestLogFilename:
    def test_the_trailing_py_becomes_log(self):
        assert ll._log_filename("run.py") == "run.log"

    def test_only_the_trailing_extension_is_replaced(self):
        assert ll._log_filename("run.pytest_smoke.py") == "run.pytest_smoke.log"

    def test_a_name_without_py_gets_log_appended(self):
        assert ll._log_filename("data") == "data.log"

    def test_a_py_prefix_that_is_not_an_extension_is_kept(self):
        assert ll._log_filename("a.pyx") == "a.pyx.log"


class TestInitLoggingDefaults:
    def test_the_documented_defaults_are_pinned(self):
        import inspect

        params = inspect.signature(ll.init_logging).parameters
        assert params["default_caller_name"].default == "app.py"
        assert params["log_to_file"].default is True
        assert params["maxBytes"].default == 5_000_000
        assert "%(asctime)s - %(levelname)s - %(processName)s[%(process)d]-%(threadName)s[%(thread)d]" in params["format"].default


class TestFreshProcessState:
    def test_the_external_ip_is_not_resolved_at_import(self):
        import subprocess
        import sys

        code = "import pyutilz.dev.logginglib as ll; " "print(ll._EXTERNAL_IP_RESOLVED, ll.EXTERNAL_IP)"
        src_root = str(Path(ll.__file__).resolve().parents[2])
        env = dict(os.environ, PYTHONPATH=src_root)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=110, check=True, env=env).stdout.strip()
        assert out == "False None"


class TestInitLoggingHandlers:
    @pytest.fixture
    def captured(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        calls: list = []

        def fake_basic_config(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(ll.logging, "basicConfig", fake_basic_config)
        monkeypatch.setattr(ll, "logger", logging.getLogger("initial_probe"))
        yield calls
        for handler in [h for c in calls for h in c.get("handlers", []) if isinstance(h, logging.FileHandler)]:
            handler.close()

    def test_the_file_handler_is_named_from_the_forced_caller(self, captured):
        ll.init_logging(forced_filename="svc.py")
        file_handlers = [h for h in captured[0]["handlers"] if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert os.path.basename(file_handlers[0].baseFilename) == "svc.log"

    def test_the_default_rotation_size_is_five_million_bytes(self, captured):
        ll.init_logging(forced_filename="svc.py")
        file_handler = next(h for h in captured[0]["handlers"] if isinstance(h, logging.FileHandler))
        assert file_handler.maxBytes == 5_000_000

    def test_a_custom_rotation_size_is_applied(self, captured):
        ll.init_logging(forced_filename="svc.py", maxBytes=1234)
        file_handler = next(h for h in captured[0]["handlers"] if isinstance(h, logging.FileHandler))
        assert file_handler.maxBytes == 1234

    def test_no_file_handler_when_file_logging_is_off(self, captured):
        ll.init_logging(forced_filename="svc.py", log_to_file=False)
        assert not any(isinstance(h, logging.FileHandler) for h in captured[0]["handlers"])
        assert any(isinstance(h, logging.StreamHandler) for h in captured[0]["handlers"])

    def test_the_format_and_level_reach_the_root_configuration(self, captured):
        ll.init_logging(forced_filename="svc.py", level=logging.DEBUG, format="%(message)s", log_to_file=False)
        assert captured[0]["format"] == "%(message)s"
        assert captured[0]["level"] == logging.DEBUG
        assert captured[0]["force"] is True

    def test_a_custom_logger_is_returned_and_set_to_the_level(self, captured):
        custom = logging.getLogger("custom_for_test_xyz")
        result = ll.init_logging(custom_logger=custom, level=logging.WARNING, log_to_file=False, forced_filename="svc.py")
        assert result is custom
        assert custom.level == logging.WARNING

    def test_without_a_custom_logger_the_caller_name_names_the_logger(self, captured):
        result = ll.init_logging(forced_filename="svc.py", log_to_file=False)
        assert result.name == "svc.py"


class TestInitLoggingFileHandlerDetails:
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

    def test_the_fallback_handler_keeps_two_backups_append_mode_and_utf8(self, captured, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "concurrent_log_handler", None)
        ll.init_logging(forced_filename="svc.py")
        handler = next(h for h in captured[0]["handlers"] if isinstance(h, logging.FileHandler))
        assert handler.backupCount == 2
        assert handler.mode == "a"
        assert handler.encoding == "utf-8"

    def test_the_caller_file_name_is_used_when_no_name_is_forced(self, captured, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "concurrent_log_handler", None)
        ll.init_logging()
        handler = next(h for h in captured[0]["handlers"] if isinstance(h, logging.FileHandler))
        assert os.path.basename(handler.baseFilename) == "test_logginglib_survivors.log"

    def test_an_ipython_input_caller_falls_back_to_the_default_name(self, captured, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "concurrent_log_handler", None)

        class _Frame:
            filename = "/tmp/ipython-input-42.py"

        monkeypatch.setattr(ll, "inspect", types.SimpleNamespace(stack=lambda: [None, _Frame()]))
        ll.init_logging(default_caller_name="app.py")
        handler = next(h for h in captured[0]["handlers"] if isinstance(h, logging.FileHandler))
        assert os.path.basename(handler.baseFilename) == "app.log"

    def test_an_ipykernel_caller_falls_back_to_the_default_name(self, captured, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "concurrent_log_handler", None)

        class _Frame:
            filename = "/tmp/ipykernel_launcher.py"

        monkeypatch.setattr(ll, "inspect", types.SimpleNamespace(stack=lambda: [None, _Frame()]))
        ll.init_logging(default_caller_name="app.py")
        handler = next(h for h in captured[0]["handlers"] if isinstance(h, logging.FileHandler))
        assert os.path.basename(handler.baseFilename) == "app.log"

    def test_a_concurrent_handler_is_used_when_installed(self, captured, monkeypatch):
        seen: list = []

        class _Concurrent:
            def __init__(self, **kwargs):
                seen.append(kwargs)
                self.level = 0

        monkeypatch.setitem(__import__("sys").modules, "concurrent_log_handler", types.SimpleNamespace(ConcurrentRotatingFileHandler=_Concurrent))
        ll.init_logging(forced_filename="svc.py")
        assert seen and seen[0]["backupCount"] == 2
        assert seen[0]["use_gzip"] is True
        assert seen[0]["encoding"] == "utf-8"
        assert seen[0]["mode"] == "a"
        assert os.path.basename(seen[0]["filename"]) == "svc.log"

    def test_the_stderr_is_reconfigured_to_replace_unencodable_text(self, captured, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "concurrent_log_handler", None)
        recorded: list = []

        class _Stderr:
            def reconfigure(self, **kwargs):
                recorded.append(kwargs)

        monkeypatch.setattr(ll.sys, "stderr", _Stderr())
        ll.init_logging(forced_filename="svc.py", log_to_file=False)
        assert recorded == [{"errors": "replace"}]


def _parent_for_log(alpha, beta=2, **kwargs):
    return ll.initialize_function_log()


def _parent_explicit(alpha, beta=2, **kwargs):
    return ll.initialize_function_log(explicit_only=True)


class TestInitializeFunctionLog:
    def test_the_caller_name_and_module_are_recorded(self):
        out = _parent_for_log(1)
        assert out["function"] == "_parent_for_log"
        assert out["module"] == "test_logginglib_survivors.py"

    def test_the_parameters_of_the_caller_are_recorded(self):
        out = _parent_for_log(1, beta=5, gamma="x")
        assert out["parameters"]["alpha"] == 1
        assert out["parameters"]["beta"] == 5

    def test_unsupported_argument_types_are_filtered_out(self):
        out = _parent_for_log(1, beta=[1, 2])
        assert "beta" not in out["parameters"]
        assert out["parameters"]["alpha"] == 1

    def test_explicit_only_keeps_just_the_keyword_arguments(self):
        out = _parent_explicit(1, beta=3, gamma="y")
        assert "alpha" not in out["parameters"]
        assert out["parameters"] == {"gamma": "y"}

    def test_each_entry_carries_a_timestamp(self):
        out = _parent_for_log(1)
        import datetime as _dt

        started = out["results"]["timing"]["started_at"]
        assert isinstance(started, _dt.datetime)
        assert started.tzinfo == _dt.timezone.utc


def _parent_kw_for_error(alpha, **kwargs):
    return ll.initialize_function_log()


def _parent_p_for_error(alpha):
    return ll.initialize_function_log()


class _ExcLog:
    def __init__(self):
        self.exceptions: list = []

    def exception(self, e):
        self.exceptions.append(e)


class TestInitializeFunctionLogErrors:
    def _run(self, monkeypatch, fake_inspect=None, fake_filter=None):
        log = _ExcLog()
        monkeypatch.setattr(ll, "logger", log)
        if fake_inspect is not None:
            monkeypatch.setattr(ll, "inspect", fake_inspect)
        if fake_filter is not None:
            monkeypatch.setattr(ll, "filter_elements_by_type", fake_filter)
        return ll.initialize_function_log(), log

    def test_a_missing_current_frame_is_logged_with_its_reason(self, monkeypatch):
        fake = types.SimpleNamespace(currentframe=lambda: None, getfile=lambda f: "x.py", getargvalues=lambda f: None)
        out, log = self._run(monkeypatch, fake_inspect=fake)
        assert str(log.exceptions[0]) == "no current frame available (some Python implementations don't provide one)"
        assert isinstance(log.exceptions[0], TypeError)
        assert out["function"] is None

    def test_a_missing_caller_frame_is_logged(self, monkeypatch):
        fake = types.SimpleNamespace(currentframe=lambda: types.SimpleNamespace(f_back=None), getfile=lambda f: "x.py", getargvalues=lambda f: None)
        out, log = self._run(monkeypatch, fake_inspect=fake)
        assert str(log.exceptions[0]) == "no caller frame available"

    def test_a_non_dict_keyword_set_is_flagged(self, monkeypatch):
        real = ll.filter_elements_by_type

        def fake_filter(obj, allowed_types):
            if isinstance(obj, dict) and "gamma" in obj:
                return ["not", "a", "dict"]
            return real(obj=obj, allowed_types=allowed_types)

        monkeypatch.setattr(ll, "filter_elements_by_type", fake_filter)
        with pytest.raises(AssertionError) as info:
            _parent_kw_for_error(1, gamma=2)
        assert str(info.value) == "expected dict kwargs, got list"

    def test_a_non_dict_parameter_set_is_flagged(self, monkeypatch):
        real = ll.filter_elements_by_type

        def fake_filter(obj, allowed_types):
            if "alpha" in obj:
                return ["not", "a", "dict"]
            return real(obj=obj, allowed_types=allowed_types)

        monkeypatch.setattr(ll, "filter_elements_by_type", fake_filter)
        with pytest.raises(AssertionError) as info:
            _parent_p_for_error(1)
        assert str(info.value) == "expected dict params, got list"


class TestInitializeFunctionLogSecondStage:
    def test_a_missing_frame_also_reports_the_missing_caller_in_the_second_stage(self, monkeypatch):
        log = _ExcLog()
        monkeypatch.setattr(ll, "logger", log)
        fake = types.SimpleNamespace(currentframe=lambda: None, getfile=lambda f: "x.py", getargvalues=lambda f: None)
        monkeypatch.setattr(ll, "inspect", fake)
        ll.initialize_function_log()
        assert [str(e) for e in log.exceptions] == [
            "no current frame available (some Python implementations don't provide one)",
            "no caller frame available",
        ]

    def test_an_unreadable_argument_frame_is_logged(self, monkeypatch):
        log = _ExcLog()
        monkeypatch.setattr(ll, "logger", log)
        fake = types.SimpleNamespace(
            currentframe=lambda: types.SimpleNamespace(f_back=types.SimpleNamespace(f_code=types.SimpleNamespace(co_name="f"))),
            getfile=lambda f: "x.py",
            getargvalues=lambda f: None,
        )
        monkeypatch.setattr(ll, "inspect", fake)
        ll.initialize_function_log()
        assert any(isinstance(e, AttributeError) for e in log.exceptions)
