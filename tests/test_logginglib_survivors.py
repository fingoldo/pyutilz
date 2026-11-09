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


class _InfoLog:
    def __init__(self):
        self.info_calls: list = []

    def info(self, *args):
        self.info_calls.append(args)


class TestClockAndMessage:
    def test_stopping_the_clock_records_finish_time_and_duration(self):
        from datetime import datetime, timedelta, timezone

        started = datetime.now(timezone.utc) - timedelta(seconds=10)
        obj = {"started_at": started}
        duration = ll._stop_clocks(obj)
        assert isinstance(duration, float)
        assert 9.9 < duration < 60
        assert obj["duration"] == duration
        assert obj["finished_at"].tzinfo == timezone.utc

    def test_a_start_in_the_future_gives_zero_duration(self):
        from datetime import datetime, timedelta, timezone

        obj = {"started_at": datetime.now(timezone.utc) + timedelta(hours=1)}
        assert ll._stop_clocks(obj) == 0.0

    def test_a_name_without_terminal_punctuation_gets_an_ellipsis(self, monkeypatch):
        log = _InfoLog()
        monkeypatch.setattr(ll, "logger", log)
        ll._message("Loading data")
        assert log.info_calls == [("%s%s", "Loading data", "...")]

    def test_a_name_already_ending_in_punctuation_gets_no_ellipsis(self, monkeypatch):
        log = _InfoLog()
        monkeypatch.setattr(ll, "logger", log)
        ll._message("Done!")
        assert log.info_calls == [("%s%s", "Done!", "")]

    def test_a_whitespace_only_name_is_not_logged(self, monkeypatch):
        log = _InfoLog()
        monkeypatch.setattr(ll, "logger", log)
        ll._message("   ")
        assert log.info_calls == []


class TestCloseOpenedActivities:
    def test_open_activities_are_stopped_and_closed_ones_are_left_alone(self):
        from datetime import datetime, timedelta, timezone

        t0 = datetime.now(timezone.utc) - timedelta(seconds=5)
        closed_at = t0 + timedelta(seconds=1)
        acts = {"done": {"started_at": t0, "finished_at": closed_at, "duration": 1.0}, "open": {"started_at": t0}}
        last = ll._close_opened_activities(acts)
        assert acts["done"]["finished_at"] == closed_at
        assert "finished_at" in acts["open"]
        assert isinstance(last, float)

    def test_no_open_activities_gives_none(self):
        from datetime import datetime, timezone

        acts = {"done": {"started_at": datetime.now(timezone.utc), "finished_at": datetime.now(timezone.utc)}}
        assert ll._close_opened_activities(acts) is None


class TestSubSecondDuration:
    def test_a_sub_second_duration_is_kept_as_measured(self):
        from datetime import datetime, timedelta, timezone

        obj = {"started_at": datetime.now(timezone.utc) - timedelta(milliseconds=500)}
        duration = ll._stop_clocks(obj)
        assert 0.4 < duration < 5


def _fresh_log(function="f"):
    from datetime import datetime, timezone

    return {
        "module": "m.py",
        "function": function,
        "parameters": {"a": 1},
        "results": {"timing": {"started_at": datetime.now(timezone.utc)}},
    }


class TestFinalizeFunctionLog:
    def test_finalizing_stamps_the_timing_and_returns_the_log(self):
        out = ll.finalize_function_log(_fresh_log())
        assert "finished_at" in out["results"]["timing"]
        assert out["results"]["timing"]["duration"] >= 0

    def test_verbose_without_a_database_prints_the_log(self, capsys):
        log = _fresh_log()
        ll.finalize_function_log(log, verbose=True)
        assert "'function': 'f'" in capsys.readouterr().out

    def test_quiet_by_default_prints_nothing(self, capsys):
        ll.finalize_function_log(_fresh_log())
        assert capsys.readouterr().out == ""

    def test_a_database_path_inserts_the_fields_in_order(self, monkeypatch):
        import pyutilz.database.db as dbpkg

        calls: list = []

        def fake_execute(query, rows):
            calls.append((query, rows))

        monkeypatch.setattr(dbpkg, "safe_execute_values", fake_execute, raising=False)
        ll.finalize_function_log(_fresh_log(), db_path="logging.api_calls")
        query, rows = calls[0]
        assert query == "insert into logging.api_calls (module,function,parameters,results,node,session) values %s"
        assert rows[0][0:2] == ["m.py", "f"]
        from pyutilz.text.strings.jsonutils import json_pg_dumps

        assert str(rows[0][2]) == str(json_pg_dumps({"a": 1}))


class TestLogResultAndResults:
    def test_log_result_stores_the_value_under_its_key(self):
        log = _fresh_log()
        ll.log_result(log, "rows", 3, verbose=False)
        assert log["results"]["rows"] == 3

    def test_log_result_announces_the_key_and_value(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_result(_fresh_log(), "rows", 3, verbose=True)
        assert log_.info_calls == [("%s%s", "rows: 3", "...")]

    def test_log_results_merges_every_key(self):
        log = _fresh_log()
        ll.log_results(log, {"a": 1, "b": 2}, verbose=False)
        assert log["results"]["a"] == 1 and log["results"]["b"] == 2

    def test_log_results_announces_the_merged_dict(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_results(_fresh_log(), {"a": 1}, verbose=True)
        assert log_.info_calls == [("%s%s", "{'a': 1}", "...")]


class TestFinalizeActivitiesAndDefaults:
    def test_finalizing_closes_the_activities_still_open(self):
        from datetime import datetime, timedelta, timezone

        log = _fresh_log()
        started = datetime.now(timezone.utc) - timedelta(seconds=2)
        log["results"]["activities"] = {"open": {"started_at": started}}
        ll.finalize_function_log(log)
        assert "finished_at" in log["results"]["activities"]["open"]

    def test_an_invalid_table_path_raises_the_sql_validation_error(self):
        from pyutilz.database.exceptions import SQLValidationError

        with pytest.raises(SQLValidationError):
            ll.finalize_function_log(_fresh_log(), db_path="bad table; drop")

    def test_log_result_announces_by_default(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_result(_fresh_log(), "rows", 3)
        assert log_.info_calls == [("%s%s", "rows: 3", "...")]

    def test_log_results_announces_by_default(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_results(_fresh_log(), {"a": 1})
        assert log_.info_calls == [("%s%s", "{'a': 1}", "...")]


class TestLogActivityDefaults:
    def test_starting_an_activity_announces_it_by_default(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_activity(_fresh_log(), "load")
        assert log_.info_calls == [("%s%s", "load", "...")]

    def test_starting_the_next_activity_returns_the_previous_duration(self, monkeypatch):
        monkeypatch.setattr(ll, "logger", _InfoLog())
        log = _fresh_log()
        assert ll.log_activity(log, "first", verbose=False) is None
        duration = ll.log_activity(log, "second", verbose=False)
        assert isinstance(duration, float)
        assert duration >= 0


class TestLogActivityAndLoadedRows:
    def test_an_activity_records_its_start_time(self, monkeypatch):
        import datetime as _dt

        monkeypatch.setattr(ll, "logger", _InfoLog())
        log = _fresh_log()
        ll.log_activity(log, "load", verbose=False)
        assert isinstance(log["results"]["activities"]["load"]["started_at"], _dt.datetime)

    def test_a_quiet_activity_is_not_announced(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_activity(_fresh_log(), "load", verbose=False)
        assert log_.info_calls == []

    def test_an_unknown_source_type_is_refused(self):
        with pytest.raises(AssertionError) as info:
            ll.log_loaded_rows([1], "users", source_type="sql")
        assert str(info.value) == "source_type must be 'db_table' or 'file', got 'sql'"

    def test_the_loaded_row_count_is_recorded_per_source(self):
        log = {"results": {}}
        ll.log_loaded_rows([1, 2, 3], "users", results_log=log, lang="en")
        assert log["results"]["loaded"]["db_table"]["users"] == {"rows": 3}

    def test_a_missing_results_log_is_created_and_nothing_is_announced(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        assert ll.log_loaded_rows([1], "users") is None
        assert log_.info_calls == []

    def test_english_database_message(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows([1, 2, 3], "users", source_type="db_table", results_log={"results": {}}, lang="en", verbose=True)
        assert log_.info_calls == [("%s%s", "Loaded 3 rows from users DB table.", "")]

    def test_english_file_message(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows([1, 2, 3], "users", source_type="file", results_log={"results": {}}, lang="en", verbose=True)
        assert log_.info_calls == [("%s%s", "Loaded 3 rows from users file.", "")]

    def test_russian_database_message(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows([1, 2, 3], "users", source_type="db_table", results_log={"results": {}}, lang="ru", verbose=True)
        assert log_.info_calls == [("%s%s", "Загружено 3 строк из таблицы БД users.", "")]

    def test_russian_file_message(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows([1, 2, 3], "users", source_type="file", results_log={"results": {}}, lang="ru", verbose=True)
        assert log_.info_calls == [("%s%s", "Загружено 3 строк из файла users.", "")]

    def test_an_unsupported_language_falls_back_to_english(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows([1, 2, 3], "users", source_type="file", results_log={"results": {}}, lang="fr", verbose=True)
        assert log_.info_calls == [("%s%s", "Loaded 3 rows from users file.", "")]

    def test_the_language_is_taken_from_the_callers_setting_when_unspecified(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        monkeypatch.setattr(ll, "lookup_in_stack", lambda name: "ru")
        ll.log_loaded_rows([1, 2, 3], "users", source_type="file", results_log={"results": {}}, verbose=True)
        assert log_.info_calls == [("%s%s", "Загружено 3 строк из файла users.", "")]

    def test_quiet_loads_are_not_announced(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows([1, 2, 3], "users", results_log={"results": {}}, lang="en", verbose=False)
        assert log_.info_calls == []


class TestLoadedRowsResultsKey:
    def test_a_log_without_results_gets_the_results_and_loaded_entries(self):
        log: dict = {}
        ll.log_loaded_rows([1, 2], "users", results_log=log, lang="en")
        assert log["results"]["loaded"]["db_table"]["users"] == {"rows": 2}

    def test_the_language_lookup_asks_for_the_reports_language_setting(self, monkeypatch):
        asked: list = []

        def fake_lookup(name):
            asked.append(name)
            return None

        monkeypatch.setattr(ll, "lookup_in_stack", fake_lookup)
        ll.log_loaded_rows([1], "users", results_log={"results": {}})
        assert asked == ["reports_language"]


class TestCredentialRedaction:
    def test_a_non_string_value_is_fully_masked(self):
        assert ll._redact_credential_shaped_value(5) == "***"
        assert ll._redact_credential_shaped_value(None) == "***"

    def test_a_proxy_url_loses_its_userinfo_and_keeps_scheme_and_host(self):
        assert ll._redact_credential_shaped_value("http://u:p@h:1") == "http://***@h:1"

    def test_a_scheme_url_with_only_a_userinfo_marker_is_kept_in_shape(self):
        assert ll._redact_credential_shaped_value("ab://noauth@") == "ab://***@"

    def test_an_at_sign_without_a_scheme_still_keeps_only_the_host(self):
        assert ll._redact_credential_shaped_value("ab@cd") == "***@cd"

    def test_a_long_plain_value_keeps_its_first_and_last_two_characters(self):
        assert ll._redact_credential_shaped_value("abcdefgh") == "ab***gh"
        assert ll._redact_credential_shaped_value("plain") == "pl***in"

    def test_a_value_of_four_characters_is_fully_masked(self):
        assert ll._redact_credential_shaped_value("abcd") == "***"

    def test_a_short_value_is_fully_masked(self):
        assert ll._redact_credential_shaped_value("ab") == "***"

    def test_a_scheme_without_userinfo_is_masked_as_a_plain_value(self):
        assert ll._redact_credential_shaped_value("http://h") == "ht***/h"


class TestLoadedRowsMessagesExact:
    def test_english_database_message_is_exact(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows([1, 2, 3], "users", source_type="db_table", results_log={"results": {}}, lang="en", verbose=True)
        assert log_.info_calls == [("%s%s", "Loaded 3 rows from users DB table.", "")]

    def test_russian_message_uses_the_thousands_separator_in_the_count(self, monkeypatch):
        log_ = _InfoLog()
        monkeypatch.setattr(ll, "logger", log_)
        ll.log_loaded_rows(list(range(12000)), "users", source_type="file", results_log={"results": {}}, lang="ru", verbose=True)
        assert log_.info_calls == [("%s%s", "Загружено 12_000 строк из файла users.", "")]


class TestRedactionRestAfterScheme:
    def test_an_at_sign_before_the_scheme_is_not_taken_for_userinfo(self):
        assert ll._redact_credential_shaped_value("a@b://c") == "a@***/c"


class TestRedactParametersAndCallParams:
    def test_the_session_variable_names_are_pinned(self):
        assert ll._SESSION_VARS == ("current_proxy", "current_proxy_resolved", "login")

    def test_a_credential_named_parameter_is_masked_by_value_shape(self):
        out = ll._redact_parameters({"password": "hunter2"})  # pragma: allowlist secret -- test fixture, not a credential
        assert out["password"] == "hu***r2"  # pragma: allowlist secret -- masked fixture, not a credential

    def test_a_dsn_inside_an_ordinary_argument_is_scrubbed(self):
        out = ll._redact_parameters({"url": "postgresql://u:pw@h/db"})
        assert out["url"] == "<dsn-redacted>"

    def test_numbers_and_ordinary_strings_pass_through(self):
        out = ll._redact_parameters({"n": 5, "name": "bob"})
        assert out == {"n": 5, "name": "bob"}

    def test_explicit_mode_keeps_only_the_keyword_arguments(self):
        assert ll._call_params((1, "x"), {"k": 2}, True, (int, str)) == {"k": 2}

    def test_default_mode_adds_positional_arguments_by_index(self):
        assert ll._call_params((1, "x"), {"k": 2}, False, (int, str)) == {0: 1, 1: "x", "k": 2}

    def test_unsupported_positional_types_are_dropped(self):
        assert ll._call_params(([1],), {}, False, (int, str)) == {}

    def test_a_non_dict_keyword_filter_result_is_refused(self, monkeypatch):
        monkeypatch.setattr(ll, "filter_elements_by_type", lambda obj, allowed_types: ["x"])
        with pytest.raises(AssertionError) as info:
            ll._call_params((), {"k": 1}, False, (int,))
        assert str(info.value) == "expected dict kwargs, got list"

    def test_a_non_dict_positional_filter_result_is_refused(self, monkeypatch):
        calls: list = []

        def fake(obj, allowed_types):
            calls.append(1)
            return {} if len(calls) == 1 else ["x"]

        monkeypatch.setattr(ll, "filter_elements_by_type", fake)
        with pytest.raises(AssertionError) as info:
            ll._call_params((1,), {}, False, (int,))
        assert str(info.value) == "expected dict params, got list"


class TestCallHeaderAndSession:
    def test_the_header_names_the_module_basename_and_the_function(self, monkeypatch):
        monkeypatch.setattr(ll, "get_node_external_ip", lambda: "10.0.0.1")
        log = _fresh_log()
        ll._fill_call_header(log, "/srv/app/jobs.py", "run", {"n": 1}, include_node_ip=False)
        assert log["module"] == "jobs.py"
        assert log["function"] == "run"

    def test_session_variables_are_left_out_of_the_parameters(self, monkeypatch):
        monkeypatch.setattr(ll, "get_node_external_ip", lambda: None)
        log = _fresh_log()
        ll._fill_call_header(log, "/a/b.py", "f", {"n": 1, "login": "bob", "current_proxy": "http://u:p@h:1"}, include_node_ip=False)
        assert log["parameters"] == {"n": 1}

    def test_the_node_ip_is_recorded_only_when_requested(self, monkeypatch):
        monkeypatch.setattr(ll, "get_node_external_ip", lambda: "10.0.0.1")
        with_ip = _fresh_log()
        ll._fill_call_header(with_ip, "/a/b.py", "f", {}, include_node_ip=True)
        without = _fresh_log()
        ll._fill_call_header(without, "/a/b.py", "f", {}, include_node_ip=False)
        assert with_ip["node"] == {"ip": "10.0.0.1"}
        assert "node" not in without

    def test_a_truthy_proxy_is_recorded_redacted_in_the_session(self):
        log = _fresh_log()
        ll._record_session(log, {"current_proxy": "http://u:p@h:1"})
        assert log["session"] == {"current_proxy": "http://***@h:1"}

    def test_a_truthy_login_is_recorded_masked_when_short(self):
        log = _fresh_log()
        ll._record_session(log, {"login": "bob"})
        assert log["session"] == {"login": "***"}

    def test_falsy_session_values_create_no_session_entry(self):
        log = _fresh_log()
        ll._record_session(log, {"login": "", "current_proxy": None})
        assert "session" not in log

    def test_non_session_keys_are_ignored(self):
        log = _fresh_log()
        ll._record_session(log, {"n": 1})
        assert "session" not in log
