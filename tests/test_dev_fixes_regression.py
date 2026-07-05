"""Behavioral regression tests for audit fixes in dev/* and system/* modules.

Each test exercises the actual code path and would FAIL on the pre-fix source.
Optional deps are guarded with importorskip.
"""

import logging
from datetime import timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# logginglib.py
# ---------------------------------------------------------------------------


class TestRedisHandlerEmit:
    """Finding 1: emit() must not silently drop records on error."""

    def test_emit_error_calls_handleError(self):
        from pyutilz.logginglib import RedisHandler

        rc = MagicMock()
        rc.lpush.side_effect = RuntimeError("redis down")
        handler = RedisHandler(rc=rc)
        handler.handleError = MagicMock()

        record = logging.LogRecord("n", logging.INFO, __file__, 1, "msg", None, None)
        handler.emit(record)

        # Pre-fix: `except Exception: pass` swallowed the error, handleError never called.
        handler.handleError.assert_called_once_with(record)


class TestInitializeFunctionLogNoNameError:
    """Finding 2: params/module_name must be initialised so no NameError fires."""

    def test_returns_dict_even_when_frame_introspection_fails(self):
        from pyutilz.logginglib import initialize_function_log

        # Force getargvalues to blow up -> pre-fix `params` stayed unbound -> NameError.
        with patch("pyutilz.logginglib.inspect.getargvalues", side_effect=TypeError("boom")):
            log = initialize_function_log()

        assert isinstance(log, dict)
        assert log["parameters"] == {}
        assert "module" in log and "function" in log


class TestClocksTimezoneAware:
    """Finding 3: timestamps must be timezone-aware (datetime.now(timezone.utc))."""

    def test_started_at_is_tz_aware(self):
        from pyutilz.logginglib import _init_clocks

        obj = {}
        _init_clocks(obj)
        # Pre-fix used naive utcnow() -> tzinfo is None.
        assert obj["started_at"].tzinfo is timezone.utc


class TestDebuggedNoInfiniteLoop:
    """Finding 4: debugged() must not loop into pdb forever in non-interactive envs."""

    def test_raises_instead_of_hanging_when_non_interactive(self):
        from pyutilz.logginglib import debugged

        calls = {"n": 0}

        @debugged()
        def always_fails():
            calls["n"] += 1
            raise ValueError("nope")

        # stdin.isatty() is False under pytest -> must re-raise, not call pdb / loop.
        with patch("sys.stdin") as stdin:
            stdin.isatty.return_value = False
            with pytest.raises(ValueError):
                always_fails()

        assert calls["n"] == 1  # exactly one attempt, no retry loop

    def test_success_path_returns_value(self):
        from pyutilz.logginglib import debugged

        @debugged()
        def ok(x):
            return x * 2

        assert ok(5) == 10


class TestLogLoadedRowsBadLang:
    """Finding 5: message must be bound (defaults to en) for unknown lang."""

    def test_unknown_lang_does_not_crash(self):
        from pyutilz.logginglib import log_loaded_rows

        pytest.importorskip("inflect")  # en fallback message uses suffixize -> inflect
        results_log = {"results": {}}
        # Pre-fix: lang='de' left `message` unbound -> UnboundLocalError when verbose.
        log_loaded_rows(obj=[1, 2, 3], source="tbl", results_log=results_log, lang="de", verbose=True)
        assert results_log["results"]["loaded"]["db_table"]["tbl"]["rows"] == 3


# ---------------------------------------------------------------------------
# notebook_init.py
# ---------------------------------------------------------------------------


class TestNotebookPolarsConfig:
    """Finding 6: setup_polars_config must guard psutil.cpu_count() -> None."""

    def test_cpu_count_none_does_not_crash(self):
        import pyutilz.dev.notebook_init as ni

        with patch.object(ni.psutil, "cpu_count", return_value=None):
            ni.setup_polars_config()

        assert int(ni.os.environ["POLARS_MAX_THREADS"]) >= 1


# ---------------------------------------------------------------------------
# meta_test_utils.py
# ---------------------------------------------------------------------------


class TestMetaAll:
    """Finding 7: safe_import must be exported in __all__."""

    def test_safe_import_in_all(self):
        from pyutilz.dev import meta_test_utils as m

        assert "safe_import" in m.__all__


# ---------------------------------------------------------------------------
# system/monitoring.py
# ---------------------------------------------------------------------------


class TestTimeoutExecutorAtexit:
    """Finding 8: shared executor shutdown must be registered at exit."""

    def test_atexit_registers_executor_shutdown(self):
        import atexit
        import importlib
        import sys

        # Re-import the module with atexit.register patched so we can observe
        # that the shared executor's shutdown IS registered at import time.
        # Pre-fix there was no atexit registration -> the executor leaked.
        registered = []
        real_register = atexit.register
        with patch.object(atexit, "register", side_effect=lambda f, *a, **k: registered.append((f, a, k)) or real_register(f, *a, **k)):
            sys.modules.pop("pyutilz.system.monitoring", None)
            mon = importlib.import_module("pyutilz.system.monitoring")

        assert any(f == mon._TIMEOUT_EXECUTOR.shutdown for f, _, _ in registered)


# ---------------------------------------------------------------------------
# system/hardware_monitor.py
# ---------------------------------------------------------------------------


class TestHardwareMonitorGuards:
    """Finding 9: cpu_freq()->None guard and safe-int gpu_module_id."""

    def test_cpu_freq_none_does_not_crash(self):
        psutil = pytest.importorskip("psutil")
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        mon = UtilizationMonitor()
        with patch.object(psutil, "cpu_freq", return_value=None), patch.object(psutil, "cpu_percent", return_value=1.0), patch.object(
            psutil, "virtual_memory", return_value=MagicMock(used=1, free=1)
        ), patch("pyutilz.system.hardware_monitor.get_own_memory_usage", return_value=1), patch(
            "pyutilz.system.hardware_monitor.get_nvidia_smi_info", return_value=None
        ), patch(
            "pyutilz.system.hardware_monitor.time.sleep", side_effect=lambda s: mon.stop_flag.set()
        ):
            # single loop iteration then stop; pre-fix crashed on None.current
            mon.query_utilization()

        assert mon.cpu_clocks[-1] == 0.0

    def test_bad_gpu_module_id_does_not_crash(self):
        psutil = pytest.importorskip("psutil")
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        gpu_stats = {"gpu": [{"gpu_module_id": "N/A"}]}
        mon = UtilizationMonitor(gpu_ids=[0])
        with patch.object(psutil, "cpu_freq", return_value=MagicMock(current=1000.0)), patch.object(psutil, "cpu_percent", return_value=1.0), patch.object(
            psutil, "virtual_memory", return_value=MagicMock(used=1, free=1)
        ), patch("pyutilz.system.hardware_monitor.get_own_memory_usage", return_value=1), patch(
            "pyutilz.system.hardware_monitor.get_nvidia_smi_info", return_value=gpu_stats
        ), patch(
            "pyutilz.system.hardware_monitor.time.sleep", side_effect=lambda s: mon.stop_flag.set()
        ):
            # Pre-fix: int("N/A") -> ValueError crashed the thread function.
            mon.query_utilization()

        # gpu_id coerced to 0, which isn't in gpu_ids=[0]? 0 IS in [0], so it is processed.
        # Either way no exception is what we assert.


# ---------------------------------------------------------------------------
# system/distributed.py
# ---------------------------------------------------------------------------


class TestRegisterScraperExplicitFailure:
    """Finding 10: register_scraper must raise (not silently return None) on info failure."""

    def test_raises_when_get_system_info_fails(self):
        import pyutilz.system.distributed as dist

        # Ensure node_id is unset so the try-block runs.
        dist._container.node_id = None
        with patch.object(dist.system, "get_system_info", side_effect=RuntimeError("no info")), patch.object(
            dist.web, "get_external_ip", return_value="1.2.3.4"
        ), patch.object(dist.pythonlib, "lookup_in_stack", return_value=None):
            with pytest.raises(RuntimeError):
                dist.register_scraper(scraper_name="s", version="v", app_name="a", ip="1.2.3.4")
