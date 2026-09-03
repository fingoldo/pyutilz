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


class TestInitializeFunctionLogUsesModuleLogger:
    """Regression (2026-07-21 audit round 2, LOW): both frame-introspection ``except`` blocks in
    ``initialize_function_log()`` used to call the module-level ``logging.exception(...)``
    function (always the ROOT logger) instead of this file's own configured ``logger`` object --
    invisible to a caller who reconfigured ``logger`` via ``init_logging()`` (custom handlers/
    format/level), unlike every other log call in this file."""

    def test_frame_getfile_failure_logs_via_module_logger(self):
        import pyutilz.logginglib as logginglib_module
        from pyutilz.logginglib import initialize_function_log

        fake_logger = MagicMock()
        with patch.object(logginglib_module, "logger", fake_logger):
            with patch("pyutilz.logginglib.inspect.getfile", side_effect=TypeError("boom")):
                initialize_function_log()
        fake_logger.exception.assert_called_once()

    def test_argvalues_failure_logs_via_module_logger(self):
        import pyutilz.logginglib as logginglib_module
        from pyutilz.logginglib import initialize_function_log

        fake_logger = MagicMock()
        with patch.object(logginglib_module, "logger", fake_logger):
            with patch("pyutilz.logginglib.inspect.getargvalues", side_effect=TypeError("boom")):
                initialize_function_log()
        fake_logger.exception.assert_called_once()


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
        # The pop+re-import builds a SECOND monitoring module with its own _TIMEOUT_EXECUTOR.
        # Both the sys.modules entry AND the parent package's attribute must be put back
        # afterwards, or every later test (and every already-imported consumer) sees a
        # different module object than `from pyutilz.system import monitoring` returns --
        # the identity split tests/test_meta/test_no_module_reload.py exists to prevent.
        import pyutilz.system as _pyutilz_system

        saved_mod = sys.modules.get("pyutilz.system.monitoring")
        saved_attr = getattr(_pyutilz_system, "monitoring", None)
        try:
            with patch.object(atexit, "register", side_effect=lambda f, *a, **k: registered.append((f, a, k)) or real_register(f, *a, **k)):
                sys.modules.pop("pyutilz.system.monitoring", None)
                mon = importlib.import_module("pyutilz.system.monitoring")

            assert any(f == mon._TIMEOUT_EXECUTOR.shutdown for f, _, _ in registered)
        finally:
            # Shut down the throwaway pool this test created, then restore both bindings.
            mon_created = sys.modules.get("pyutilz.system.monitoring")
            if mon_created is not None and mon_created is not saved_mod:
                mon_created._TIMEOUT_EXECUTOR.shutdown(wait=False)
            if saved_mod is not None:
                sys.modules["pyutilz.system.monitoring"] = saved_mod
            else:
                sys.modules.pop("pyutilz.system.monitoring", None)
            if saved_attr is not None:
                _pyutilz_system.monitoring = saved_attr


# ---------------------------------------------------------------------------
# system/hardware_monitor.py
# ---------------------------------------------------------------------------


class TestHardwareMonitorGuards:
    """cpu_freq() absence/None guards and safe-int gpu_module_id."""

    def test_cpu_freq_none_does_not_crash(self):
        psutil = pytest.importorskip("psutil")
        if not hasattr(psutil, "cpu_freq"):
            # Patching a name psutil never defined on this platform would only test the mock. The
            # platforms without it are covered by test_missing_cpu_freq_attribute_is_a_reported_absence.
            pytest.skip("this psutil build has no cpu_freq (platform-gated upstream); nothing to make return None")
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        mon = UtilizationMonitor()
        with patch.object(psutil, "cpu_freq", return_value=None), patch.object(psutil, "cpu_percent", return_value=1.0), patch.object(
            psutil, "virtual_memory", return_value=MagicMock(used=1, free=1)
        ), patch("pyutilz.system.hardware_monitor.get_own_memory_usage", return_value=1), patch(
            # Stopping from inside the sample keeps this to exactly one loop iteration; the loop
            # waits on stop_flag instead of sleeping, so patching sleep no longer terminates it.
            "pyutilz.system.hardware_monitor.get_nvidia_smi_info",
            side_effect=lambda **kwargs: mon.stop_flag.set(),
        ):
            # single loop iteration then stop; pre-fix crashed on None.current
            mon.query_utilization()

        assert mon.n_sampling_errors == 0, "an unreported CPU frequency is a known absence, not a sampling failure"
        # No clock is recorded rather than a fabricated 0.0: a zero would be averaged in later as a
        # genuine "the CPU ran at 0 MHz" measurement.
        assert mon.cpu_clocks == []
        assert mon.get_average_utilization()["cpu_clocks_mhz"] is None

    def test_missing_cpu_freq_attribute_is_a_reported_absence(self, monkeypatch):
        """A platform whose psutil has no cpu_freq at all (macOS) must still sample everything else.

        psutil defines cpu_freq only ``if hasattr(_psplatform, "cpu_freq")``, so on macOS the
        attribute is missing from the module and an unguarded call raised AttributeError on EVERY
        sample -- turning the whole monitor into a stream of swallowed exceptions with no data.
        Deleting the attribute here reproduces exactly that module surface.
        """
        psutil = pytest.importorskip("psutil")
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        monkeypatch.delattr(psutil, "cpu_freq", raising=False)

        mon = UtilizationMonitor()
        with patch.object(psutil, "cpu_percent", return_value=1.0), patch.object(psutil, "virtual_memory", return_value=MagicMock(used=1, free=1)), patch(
            "pyutilz.system.hardware_monitor.get_own_memory_usage", return_value=1
        ), patch(
            "pyutilz.system.hardware_monitor.get_nvidia_smi_info",
            side_effect=lambda **kwargs: mon.stop_flag.set(),
        ):
            mon.query_utilization()

        assert mon.n_sampling_errors == 0, "a missing platform capability must not be reported as a sampling error"
        assert mon.cpu_utilizaton == [1.0], "the rest of the sample must still be collected"
        assert mon.cpu_clocks == []
        stats = mon.get_average_utilization()
        assert stats["cpu_clocks_mhz"] is None
        assert "cpu_freq" in stats["unavailable_metrics"], stats["unavailable_metrics"]

    def test_bad_gpu_module_id_does_not_crash(self):
        psutil = pytest.importorskip("psutil")
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        gpu_stats = {"gpu": [{"gpu_module_id": "N/A"}]}
        mon = UtilizationMonitor(gpu_ids=[0])
        # cpu_freq is deliberately NOT patched here: this test is about gpu_module_id parsing, and
        # patching a platform-gated psutil name would make it fail on platforms lacking it (macOS)
        # for a reason unrelated to what it asserts. The real (or absent) cpu_freq is handled.
        with patch.object(psutil, "cpu_percent", return_value=1.0), patch.object(psutil, "virtual_memory", return_value=MagicMock(used=1, free=1)), patch(
            "pyutilz.system.hardware_monitor.get_own_memory_usage", return_value=1
        ), patch(
            "pyutilz.system.hardware_monitor.get_nvidia_smi_info",
            side_effect=lambda **kwargs: (mon.stop_flag.set(), gpu_stats)[1],
        ):
            # Pre-fix: int("N/A") -> ValueError crashed the thread function.
            mon.query_utilization()

        # A non-numeric gpu_module_id is a malformed READING, not a sampling failure, and it must
        # not be charged to the error counter (pre-fix it raised ValueError out of the thread).
        assert mon.n_sampling_errors == 0


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


class TestRegisterScraperAlreadyRegisteredReturnsNodeId:
    """register_scraper() must return the existing node_id when already registered
    (the DB-registration branch was previously only path that returned a value;
    a repeat call with node_id already set fell off the end and returned None)."""

    def test_returns_existing_node_id_without_reregistering(self):
        import pyutilz.system.distributed as dist

        dist._container.node_id = 42
        with patch.object(dist.system, "get_system_info") as mock_get_info, patch.object(dist.web, "get_external_ip", return_value="1.2.3.4"), patch.object(
            dist.pythonlib, "lookup_in_stack", return_value=None
        ):
            result = dist.register_scraper(scraper_name="s", version="v", app_name="a", ip="1.2.3.4")
        assert result == 42
        mock_get_info.assert_not_called()
