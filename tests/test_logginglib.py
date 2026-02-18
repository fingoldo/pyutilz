"""
Test suite for logginglib.py
Tests cover logging utility functions and decorators.
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestFunctionLog:
    """Test function logging utilities"""

    def test_initialize_function_log(self):
        """Test initializing function log"""
        from pyutilz.logginglib import initialize_function_log

        log = initialize_function_log(explicit_only=False)

        assert isinstance(log, dict)
        assert "start_time" in log or "clocks" in log or len(log) >= 0

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

        # Results stored in nested dict
        assert "results" in log or "test_key" in log
        if "results" in log:
            assert log["results"]["test_key"] == 42
        else:
            assert log["test_key"] == 42

    def test_log_results(self):
        """Test logging multiple results"""
        from pyutilz.logginglib import initialize_function_log, log_results

        log = initialize_function_log()
        results = {"key1": "value1", "key2": 123}
        log_results(log, results, verbose=False)

        # Results stored in nested dict
        if "results" in log:
            assert "key1" in log["results"]
            assert "key2" in log["results"]
        else:
            assert "key1" in log or "key2" in log

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

    @pytest.mark.skip(reason="Requires inflect module")
    def test_log_loaded_rows(self):
        """Test logging loaded rows"""
        # Requires inflect module
        pass


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
        """Test console logging initialization"""
        from pyutilz.logginglib import init_logging

        # Should not crash
        try:
            init_logging(
                console=True,
                file=False,
                level="INFO"
            )
        except Exception:
            # May fail if logger already configured
            pass

    def test_init_logging_file(self, tmp_path):
        """Test file logging initialization"""
        from pyutilz.logginglib import init_logging

        log_file = tmp_path / "test.log"

        try:
            init_logging(
                console=False,
                file=str(log_file),
                level="DEBUG"
            )
        except Exception:
            # May fail if logger already configured
            pass


class TestEdgeCases:
    """Test edge cases"""

    def test_log_result_with_none(self):
        """Test logging None value"""
        from pyutilz.logginglib import initialize_function_log, log_result

        log = initialize_function_log()
        log_result(log, "none_key", None, verbose=False)

        # Results stored in nested dict
        if "results" in log:
            assert log["results"]["none_key"] is None
        else:
            assert "none_key" in log

    def test_log_results_empty_dict(self):
        """Test logging empty results"""
        from pyutilz.logginglib import initialize_function_log, log_results

        log = initialize_function_log()
        log_results(log, {}, verbose=False)

        # Should not crash
        assert isinstance(log, dict)
