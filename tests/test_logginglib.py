"""
Test suite for logginglib.py
Tests cover logging utility functions and decorators.
"""

import pytest


class TestFunctionLog:
    """Test function logging utilities"""

    def test_initialize_function_log(self):
        """Test initializing function log"""
        from pyutilz.logginglib import initialize_function_log

        log = initialize_function_log(explicit_only=False)

        # The documented shape: a "results" sub-dict (where log_result/log_results/log_activity
        # write), the calling function's name and module, and its captured parameters. The
        # previous `... or len(log) >= 0` tail made the whole assertion unfalsifiable.
        assert isinstance(log, dict)
        assert set(log) == {"function", "module", "parameters", "results"}
        assert log["function"] == "test_initialize_function_log"
        assert "started_at" in log["results"]["timing"]

    def test_initialize_function_log_explicit_only(self):
        """Test with explicit_only flag"""
        from pyutilz.logginglib import initialize_function_log

        log = initialize_function_log(explicit_only=True)

        assert isinstance(log, dict)

    def test_log_result(self):
        """Test logging single result"""
        from pyutilz.logginglib import initialize_function_log, log_result

        log = initialize_function_log()
        log_result(log, "test_key", 42, verbose=False)

        # log_result writes into the nested log["results"] dict -- pinned unconditionally.
        # The either-shape form this replaced ("results" in log or "test_key" in log) accepted
        # values moving between the nested dict and the top level, which is a breaking change
        # for every consumer that reads log["results"].
        assert log["results"]["test_key"] == 42
        assert "test_key" not in log

    def test_log_results(self):
        """Test logging multiple results"""
        from pyutilz.logginglib import initialize_function_log, log_results

        log = initialize_function_log()
        results = {"key1": "value1", "key2": 123}
        log_results(log, results, verbose=False)

        # Merged into the nested log["results"] dict, values intact -- pinned unconditionally.
        assert log["results"]["key1"] == "value1"
        assert log["results"]["key2"] == 123
        assert "key1" not in log and "key2" not in log

    def test_finalize_function_log(self):
        """Test finalizing function log"""
        from pyutilz.logginglib import initialize_function_log, finalize_function_log

        log = initialize_function_log()
        log["test"] = "data"

        result = finalize_function_log(log, db_path=None, verbose=False)

        assert isinstance(result, dict)
        assert "elapsed_time" in result or "duration" in result or result is not None


class TestLoggingActivities:
    """Test activity logging"""

    def test_log_activity(self):
        """Test logging activity"""
        from pyutilz.logginglib import initialize_function_log, log_activity
        import time

        log = initialize_function_log()

        # Start activity
        log_activity(log, "test_activity", verbose=False)

        # Do some work
        time.sleep(0.01)

        # Log activity again (should measure time)
        elapsed = log_activity(log, "test_activity", verbose=False)

        assert isinstance(elapsed, (int, float))
        assert elapsed >= 0

    def test_log_loaded_rows_records_row_count(self):
        """log_loaded_rows records the row count under results["loaded"][source_type][source].

        Runs everywhere: the recording contract (the part every consumer reads) needs no
        third-party dependency. Only the ENGLISH display message goes through
        ``suffixize()`` -> ``inflect``, so ``lang="ru"`` exercises the full function body
        without it -- see test_log_loaded_rows_english_message for the inflect-gated half.
        """
        from pyutilz.logginglib import initialize_function_log, log_loaded_rows

        log = initialize_function_log()
        log_loaded_rows(["a", "b", "c"], source="my_table", source_type="db_table", results_log=log, lang="ru", verbose=False)

        assert log["results"]["loaded"]["db_table"]["my_table"] == {"rows": 3}

        # A second source of a different type lands beside the first, not on top of it.
        log_loaded_rows([1] * 7, source="data.csv", source_type="file", results_log=log, lang="ru", verbose=False)
        assert log["results"]["loaded"]["file"]["data.csv"] == {"rows": 7}
        assert log["results"]["loaded"]["db_table"]["my_table"] == {"rows": 3}

    def test_log_loaded_rows_defaults_results_log(self):
        """results_log=None must not raise: the function substitutes its own {"results": {}}."""
        from pyutilz.logginglib import log_loaded_rows

        log_loaded_rows([1, 2], source="t", source_type="db_table", results_log=None, lang="ru", verbose=False)

    def test_log_loaded_rows_rejects_unknown_source_type(self):
        from pyutilz.logginglib import log_loaded_rows

        with pytest.raises(AssertionError):
            log_loaded_rows([1], source="t", source_type="carrier_pigeon", lang="ru", verbose=False)

    def test_log_loaded_rows_english_message(self, caplog):
        """The English message pluralises the noun via inflect and thousands-separates the count.

        ``pytest.importorskip`` rather than an unconditional ``@pytest.mark.skip``: inflect is
        declared in the ``nlp`` extra, so this runs wherever that extra is installed instead of
        never running anywhere.
        """
        pytest.importorskip("inflect")

        import logging

        from pyutilz.logginglib import initialize_function_log, log_loaded_rows

        log = initialize_function_log()
        with caplog.at_level(logging.INFO):
            log_loaded_rows([0] * 1234, source="my_table", source_type="db_table", results_log=log, lang="en", verbose=True)

        text = caplog.text
        assert "1_234" in text  # underscore thousands separator, per the f-string's `:_` spec
        assert "rows" in text  # pluralised -- "row" would mean suffixize() was bypassed
        assert "my_table" in text
        assert "DB table" in text
        assert log["results"]["loaded"]["db_table"]["my_table"] == {"rows": 1234}


class TestLoggingDecorators:
    """Test logging decorators"""

    def test_logged_decorator(self):
        """Test @logged decorator"""
        from pyutilz.logginglib import logged

        @logged(db_path=None)
        def sample_function(x, y, results_log=None):
            return x + y

        result = sample_function(2, 3)
        assert result == 5

    def test_debugged_decorator(self):
        """Test @debugged decorator"""
        from pyutilz.logginglib import debugged

        @debugged()
        def sample_function(x):
            return x * 2

        result = sample_function(5)
        assert result == 10


class TestInitLogging:
    """Test logging initialization"""

    def test_init_logging_console(self):
        """Test console-only logging initialization (no file handler)"""
        from pyutilz.logginglib import init_logging

        # Should not crash; pytest surfaces the real traceback if it does.
        init_logging(log_to_file=False, level="INFO")

    def test_init_logging_file(self, tmp_path):
        """Test file logging initialization"""
        from pyutilz.logginglib import init_logging

        log_file = tmp_path / "test.log"

        # pytest surfaces the real traceback if it raises.
        init_logging(log_to_file=True, forced_filename=str(log_file), level="DEBUG")


class TestEdgeCases:
    """Test edge cases"""

    def test_log_result_with_none(self):
        """Test logging None value"""
        from pyutilz.logginglib import initialize_function_log, log_result

        log = initialize_function_log()
        log_result(log, "none_key", None, verbose=False)

        # A None value must be STORED (key present, value None), not dropped -- pinned
        # unconditionally; the old else-branch only checked key presence at the top level.
        assert "none_key" in log["results"]
        assert log["results"]["none_key"] is None

    def test_log_results_empty_dict(self):
        """Test logging empty results"""
        from pyutilz.logginglib import initialize_function_log, log_results

        log = initialize_function_log()
        log_results(log, {}, verbose=False)

        # Should not crash
        assert isinstance(log, dict)
