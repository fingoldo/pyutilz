"""Extra tests for logginglib.py — covers uncovered lines."""

import pytest
import logging
import numbers
import functools
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock
from os.path import basename


# ---------------------------------------------------------------------------
# init_logging — lines 54-90
# ---------------------------------------------------------------------------

from pyutilz.logginglib import init_logging


def test_init_logging_forced_filename(tmp_path):
    log_file = str(tmp_path / "test")
    result = init_logging(forced_filename=log_file + ".py", log_to_file=False)
    assert result is not None


def test_init_logging_custom_logger():
    custom = logging.getLogger("custom_test_logger")
    result = init_logging(custom_logger=custom, log_to_file=False)
    assert result is custom


def test_init_logging_with_file(tmp_path, monkeypatch):
    # Force caller_name to something predictable
    monkeypatch.chdir(tmp_path)
    result = init_logging(forced_filename="testlog.py", log_to_file=True)
    assert result is not None


# ---------------------------------------------------------------------------
# initialize_function_log — lines 109-110, 120-121
# ---------------------------------------------------------------------------

from pyutilz.logginglib import initialize_function_log


def test_initialize_function_log():
    result = initialize_function_log()
    assert "module" in result
    assert "function" in result
    assert "results" in result
    assert "timing" in result["results"]


def test_initialize_function_log_explicit_only():
    result = initialize_function_log(explicit_only=True)
    assert "parameters" in result


# ---------------------------------------------------------------------------
# _init_clocks / _stop_clocks — lines 145, 152-153
# ---------------------------------------------------------------------------

from pyutilz.logginglib import _init_clocks, _stop_clocks


def test_init_and_stop_clocks():
    obj = {}
    _init_clocks(obj)
    assert "started_at" in obj
    duration = _stop_clocks(obj)
    assert duration >= 0
    assert "finished_at" in obj
    assert "duration" in obj


def test_stop_clocks_negative_duration():
    # line 145: duration < 0 -> clamp to 0
    from datetime import timedelta
    obj = {"started_at": datetime.utcnow() + timedelta(hours=1)}
    duration = _stop_clocks(obj)
    assert duration == 0


# ---------------------------------------------------------------------------
# _message — line 152-153
# ---------------------------------------------------------------------------

from pyutilz.logginglib import _message

# Need logger to be set for _message to work
import pyutilz.dev.logginglib as _logginglib_mod


def test_message_with_ellipsis():
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    _message("Loading data")
    mock_logger.info.assert_called_with("Loading data...")


def test_message_with_punctuation():
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    _message("Done!")
    mock_logger.info.assert_called_with("Done!")


def test_message_empty():
    # line 152: activity_name falsy
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    _message("")
    mock_logger.info.assert_not_called()


# ---------------------------------------------------------------------------
# finalize_function_log — lines 171, 176-177, 183
# ---------------------------------------------------------------------------

from pyutilz.logginglib import finalize_function_log


def test_finalize_function_log_no_db():
    results_log = {
        "results": {"timing": {"started_at": datetime.utcnow()}}
    }
    result = finalize_function_log(results_log)
    assert "duration" in result["results"]["timing"]


def test_finalize_function_log_with_activities():
    results_log = {
        "results": {
            "timing": {"started_at": datetime.utcnow()},
            "activities": {"step1": {"started_at": datetime.utcnow()}}
        }
    }
    result = finalize_function_log(results_log)
    assert "finished_at" in result["results"]["activities"]["step1"]


def test_finalize_function_log_verbose(capsys):
    results_log = {
        "results": {"timing": {"started_at": datetime.utcnow()}}
    }
    finalize_function_log(results_log, verbose=True)
    captured = capsys.readouterr()
    assert "results" in captured.out or "timing" in captured.out


# ---------------------------------------------------------------------------
# log_result / log_results — lines 190, 197
# ---------------------------------------------------------------------------

from pyutilz.logginglib import log_result, log_results


def test_log_result():
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    rl = {"results": {}}
    log_result(rl, "count", 42, verbose=True)
    assert rl["results"]["count"] == 42
    mock_logger.info.assert_called()


def test_log_results():
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    rl = {"results": {}}
    log_results(rl, {"a": 1, "b": 2}, verbose=True)
    assert rl["results"]["a"] == 1


# ---------------------------------------------------------------------------
# log_activity — line 210
# ---------------------------------------------------------------------------

from pyutilz.logginglib import log_activity


def test_log_activity():
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    rl = {"results": {}}
    log_activity(rl, "step1", verbose=True)
    assert "activities" in rl["results"]
    assert "step1" in rl["results"]["activities"]
    mock_logger.info.assert_called()


def test_log_activity_close_previous():
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    rl = {"results": {"activities": {"step1": {"started_at": datetime.utcnow()}}}}
    duration = log_activity(rl, "step2", verbose=True)
    assert duration is not None


# ---------------------------------------------------------------------------
# log_loaded_rows — lines 216-236
# ---------------------------------------------------------------------------

from pyutilz.logginglib import log_loaded_rows


def test_log_loaded_rows_en():
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    rl = {"results": {}}
    with patch("pyutilz.dev.logginglib.suffixize", return_value="rows"):
        log_loaded_rows([1, 2, 3], "my_table", source_type="db_table", results_log=rl, lang="en", verbose=True)
    assert rl["results"]["loaded"]["db_table"]["my_table"]["rows"] == 3


def test_log_loaded_rows_ru():
    mock_logger = MagicMock()
    _logginglib_mod.logger = mock_logger
    rl = {"results": {}}
    with patch("pyutilz.dev.logginglib.suffixize", return_value="rows"):
        log_loaded_rows([1], "my_file", source_type="file", results_log=rl, lang="ru", verbose=True)
    assert rl["results"]["loaded"]["file"]["my_file"]["rows"] == 1


# ---------------------------------------------------------------------------
# logged decorator — lines 251, 270-276
# ---------------------------------------------------------------------------

from pyutilz.logginglib import logged


def test_logged_decorator():
    _logginglib_mod.EXTERNAL_IP = "127.0.0.1"

    @logged()
    def my_func(x, y, results_log=None):
        return x + y

    result = my_func(1, 2)
    assert result == 3


def test_logged_decorator_with_special_vars():
    _logginglib_mod.EXTERNAL_IP = "127.0.0.1"

    @logged()
    def my_func(x, current_proxy=None, login=None, results_log=None):
        return x

    result = my_func(1, current_proxy="proxy1", login="admin")
    assert result == 1


def test_logged_decorator_explicit_only():
    _logginglib_mod.EXTERNAL_IP = None

    @logged(explicit_only=True, include_node_ip=False)
    def my_func(x, results_log=None):
        return x

    result = my_func(42)
    assert result == 42


# ---------------------------------------------------------------------------
# RedisHandler — lines 301-314
# ---------------------------------------------------------------------------

from pyutilz.logginglib import RedisHandler


def test_redis_handler_init():
    rc = MagicMock()
    handler = RedisHandler(rc=rc, LOG_DEST="TestLog", LOG_SIZE=500)
    assert handler.LOG_DEST == "TestLog"
    assert handler.LOG_SIZE == 500


def test_redis_handler_emit():
    rc = MagicMock()
    handler = RedisHandler(rc=rc)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("test", logging.INFO, "", 0, "test msg", (), None)
    with patch("pyutilz.dev.logginglib.random", return_value=0.01):
        handler.emit(record)
    rc.lpush.assert_called_once()
    rc.ltrim.assert_called_once()


def test_redis_handler_emit_no_trim():
    rc = MagicMock()
    handler = RedisHandler(rc=rc)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("test", logging.INFO, "", 0, "test msg", (), None)
    with patch("pyutilz.dev.logginglib.random", return_value=0.5):
        handler.emit(record)
    rc.lpush.assert_called_once()
    rc.ltrim.assert_not_called()


def test_redis_handler_emit_exception():
    rc = MagicMock()
    rc.lpush.side_effect = Exception("redis down")
    handler = RedisHandler(rc=rc)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("test", logging.INFO, "", 0, "test msg", (), None)
    handler.emit(record)  # should not raise


# ---------------------------------------------------------------------------
# debugged decorator — lines 327-329
# ---------------------------------------------------------------------------

from pyutilz.logginglib import debugged


def test_debugged_no_exception():
    @debugged()
    def good_func():
        return 42

    assert good_func() == 42
