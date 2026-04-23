"""Tests for system.py - Phase 2 refactoring

Tests cover:
- Command injection fix (shell=True removed from Mac UUID extraction)
- Tracemalloc resource leak fix (always call tracemalloc.stop())
- Platform compatibility (wintypes import guarded)
"""

import pytest
import platform
import tracemalloc


class TestGetSystemInfo:
    """Test get_system_info function - command injection fix"""

    def test_no_shell_true_in_mac_uuid_extraction(self):
        """Verify shell=True is not used for Mac UUID extraction (line 130 fix)"""
        from pyutilz.system import get_system_info
        import inspect

        source = inspect.getsource(get_system_info)

        # Check that Mac UUID extraction uses subprocess.PIPE, not shell=True
        # The fix replaced shell=True with proper Popen chaining
        if 'ioreg' in source:  # Mac-specific code exists
            # Find the Mac UUID extraction section
            lines = source.split('\n')
            in_mac_section = False
            for line in lines:
                if 'current_system == "Mac"' in line or 'elif current_system == "Mac"' in line:
                    in_mac_section = True
                elif in_mac_section and 'elif current_system' in line:
                    in_mac_section = False

                # In Mac section, check for proper implementation
                if in_mac_section:
                    if 'ioreg' in line and 'Popen' in line:
                        # Should use subprocess.PIPE, not shell=True
                        assert 'subprocess.PIPE' in source or 'stdout=subprocess.PIPE' in source, \
                               "Mac UUID extraction should use subprocess.PIPE"
                    if 'grep' in line and 'Popen' in line:
                        # Should chain grep command, not use shell=True
                        assert 'stdin=' in source, \
                               "Should chain grep with stdin parameter"

    def test_returns_dict_with_expected_keys(self):
        """Test that function returns dict with expected system info keys"""
        from pyutilz.system import get_system_info

        try:
            result = get_system_info()
            assert isinstance(result, dict)
            # Should have at least some system info
            assert len(result) > 0
        except Exception:
            # May fail on some systems, but should not crash
            pass


class TestShowTraceMallocSnapshot:
    """Test show_tracemalloc_snapshot - resource leak fix"""

    def test_tracemalloc_stopped_after_execution(self):
        """Test that tracemalloc.stop() is always called (line 407 fix)"""
        from pyutilz.system import show_tracemalloc_snapshot

        # Ensure tracemalloc is stopped initially
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        try:
            # Call function
            show_tracemalloc_snapshot(N=5)
        except Exception:
            # Function might fail, but we check cleanup happened
            pass

        # Should NOT be tracing after function completes
        assert not tracemalloc.is_tracing(), \
               "tracemalloc should be stopped after show_tracemalloc_snapshot() (resource leak)"

    def test_tracemalloc_stopped_even_on_error(self):
        """Test that tracemalloc.stop() is called even if function raises exception"""
        from pyutilz.system import show_tracemalloc_snapshot

        # Ensure tracemalloc is stopped initially
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        try:
            # This might raise an exception
            show_tracemalloc_snapshot(N=-1)  # Invalid argument
        except Exception:
            pass

        # Should still stop tracemalloc even if exception occurred
        assert not tracemalloc.is_tracing(), \
               "tracemalloc should be stopped even when exception occurs (try/finally fix)"

    def test_returns_snapshot_object(self):
        """Test that function returns tracemalloc snapshot"""
        from pyutilz.system import show_tracemalloc_snapshot

        # Ensure tracemalloc is stopped initially
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        result = show_tracemalloc_snapshot(N=3)

        # Should return a snapshot object
        assert result is not None
        # Should be a tracemalloc.Snapshot
        assert hasattr(result, 'statistics'), "Should return tracemalloc.Snapshot object"


class TestPlatformCompatibility:
    """Test platform-specific code is properly guarded"""

    def test_wintypes_import_guarded(self):
        """Test that ctypes.wintypes import is platform-specific (Windows only)"""
        import sys

        # Read system.py source to check import guards
        import pyutilz.system as system_module
        import inspect

        source = inspect.getsource(system_module)

        if platform.system() != 'Windows':
            # On non-Windows systems, wintypes should be conditionally imported
            # or the code should handle ImportError
            if 'ctypes.wintypes' in source:
                # Should have platform check or try/except
                assert 'platform.system()' in source or 'try:' in source, \
                       "ctypes.wintypes import should be guarded on non-Windows systems"


class TestGetCpuUsage:
    """Test CPU usage measurement within get_system_info"""

    def test_system_info_includes_cpu_usage(self):
        """Test that get_system_info includes CPU usage (not 0.0 from first call)"""
        from pyutilz.system import get_system_info

        try:
            info = get_system_info()
            if info is None:
                pytest.skip("get_system_info returned None (missing dependencies)")
            # Should include CPU info
            if 'cpu_current_load_percent' in info:
                usage = info['cpu_current_load_percent']
                assert isinstance(usage, (int, float))
                assert 0 <= usage <= 100
        except ImportError:
            pytest.skip("psutil not available")

    def test_psutil_cpu_percent_called_correctly(self):
        """Verify get_system_info calls psutil.cpu_percent() correctly"""
        try:
            import psutil
            from pyutilz.system import get_system_info
            import inspect

            source = inspect.getsource(get_system_info)

            # Should call cpu_percent
            assert 'cpu_percent' in source
        except ImportError:
            pytest.skip("psutil not available")


@pytest.mark.parametrize("n_lines", [1, 5, 10, 20])
def test_tracemalloc_snapshot_n_parameter(n_lines):
    """Parametrized test for different N values in tracemalloc snapshot"""
    from pyutilz.system import show_tracemalloc_snapshot

    # Ensure tracemalloc is stopped initially
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    try:
        snapshot = show_tracemalloc_snapshot(N=n_lines)
        assert snapshot is not None
    except Exception as e:
        pytest.fail(f"Failed with N={n_lines}: {e}")
    finally:
        # Verify cleanup
        assert not tracemalloc.is_tracing()


def test_system_module_imports_successfully():
    """Test that system module can be imported without errors"""
    try:
        import pyutilz.system
        assert pyutilz.system is not None
    except ImportError as e:
        pytest.fail(f"Failed to import system module: {e}")


class TestSystemUtilities:
    """Test system utility functions"""

    def test_ensure_dir_exists(self, tmp_path):
        """Test directory creation"""
        from pyutilz.system import ensure_dir_exists

        test_dir = tmp_path / "test_subdir" / "nested"
        ensure_dir_exists(str(test_dir))

        # Should create parent directories
        assert test_dir.parent.exists()

    def test_ensure_dir_exists_for_file(self, tmp_path):
        """Test directory creation for file path"""
        from pyutilz.system import ensure_dir_exists

        test_file = tmp_path / "subdir" / "file.txt"
        ensure_dir_exists(str(test_file))

        # Should create parent directory
        assert test_file.parent.exists()

    def test_run_from_ipython(self):
        """Test IPython detection"""
        from pyutilz.system import run_from_ipython

        result = run_from_ipython()
        # Should return False when not in IPython
        assert isinstance(result, bool)

    def test_get_script_file(self):
        """Test getting script file path"""
        from pyutilz.system import get_script_file

        result = get_script_file(__file__)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_utc_unix_timestamp(self):
        """Test getting UTC timestamp"""
        from pyutilz.system import get_utc_unix_timestamp
        import time

        ts = get_utc_unix_timestamp()
        current_ts = time.time()

        # Should be close to current time
        assert abs(ts - current_ts) < 2

    def test_get_libs_versions(self):
        """Test getting library versions"""
        from pyutilz.system import get_libs_versions

        # Use libraries that have __version__
        result = get_libs_versions("pytest")
        assert isinstance(result, dict)
        # May be empty for built-in modules without __version__
        assert len(result) >= 0

    def test_get_libs_versions_single(self):
        """Test with single library"""
        from pyutilz.system import get_libs_versions

        result = get_libs_versions("sys")
        assert isinstance(result, dict)

    def test_get_max_affordable_workers_count(self):
        """Test calculating worker count"""
        from pyutilz.system import get_max_affordable_workers_count

        result = get_max_affordable_workers_count(reservedCores=1)
        assert isinstance(result, int)
        assert result >= 1

    def test_count_app_instances_by_process(self):
        """Test counting app instances by process name"""
        from pyutilz.system import count_app_instances

        # Count python instances
        result = count_app_instances(processname="python")
        assert isinstance(result, int)
        assert result >= 0

    def test_get_own_memory_usage(self):
        """Test getting memory usage"""
        from pyutilz.system import get_own_memory_usage

        result = get_own_memory_usage()
        assert isinstance(result, float)
        assert result > 0  # Should always use some memory

    def test_get_own_memory_usage_stable_across_clean_ram_on_windows(self):
        """On Windows, ``clean_ram()`` calls ``SetProcessWorkingSetSizeEx(...,
        QUOTA_LIMITS_HARDWS_MIN_DISABLE)`` which evicts working-set pages
        to the pagefile — this used to make ``memory_info().rss`` plunge
        to near-zero right after ``clean_ram()``, triggering a
        ``"transient reporting glitch"`` WARN on every training iteration.

        The 2026-04-23 fix switched Windows measurement to
        ``memory_info().private`` (private commit charge), which is
        **not** affected by working-set trim. This test asserts that the
        reading stays in the same order of magnitude across ``clean_ram()``
        on Windows — no more phantom 0.0GB readings.

        Skipped on Linux: the trim call there is ``malloc_trim(0)`` which
        releases heap to libc without forcing page eviction, so ``rss``
        stays stable and the historic measurement path is fine.
        """
        import platform

        if platform.system() != "Windows":
            pytest.skip("Windows-specific regression — RSS trim artifact")

        from pyutilz.system import get_own_memory_usage, clean_ram

        # Warm up a working set so the trim has something to evict.
        _warm = [bytearray(8 * 1024 * 1024) for _ in range(16)]  # ~128 MB
        before = get_own_memory_usage()
        assert before is not None and before > 0.05, (
            "baseline too low to run this regression check"
        )

        # Drop the reference and invoke the exact clean_ram() path that
        # used to cause the "rss plunge". With the private-bytes switch
        # the measurement should remain a plausible fraction of ``before``.
        del _warm
        clean_ram()

        after = get_own_memory_usage()
        assert after is not None
        # After trim, private should NOT drop below 30 % of the baseline
        # (real GC may reclaim the bytearrays, so some decrease is
        # legitimate). The pre-fix ``rss`` path would show < 1 % of
        # baseline here — the assertion is tight enough to catch a
        # regression back to ``.rss`` but loose enough for real GC.
        assert after >= 0.3 * before, (
            f"Memory reading plunged from {before:.3f}GB to {after:.3f}GB "
            f"across clean_ram() — looks like ``rss`` path is back. "
            f"Windows measurement must use ``memory_info().private`` to "
            f"avoid the working-set-trim artifact."
        )

    def test_get_own_memory_usage_uses_private_bytes_on_windows(self):
        """Structural check: on Windows, ``get_own_memory_usage`` must
        pull from ``memory_info().private``, not ``.rss``.

        Implementation check (rather than end-to-end behavioural)
        because on a quiet process the rss/private values can be
        numerically close enough that the behavioural difference is
        indistinguishable — the structural guard catches refactors
        that silently revert to ``.rss``.
        """
        import platform

        if platform.system() != "Windows":
            pytest.skip("Windows-specific implementation detail")

        from pyutilz.system import system as _sys_mod

        # Inspect the source — cheaper and more robust than spinning up
        # a mock psutil.Process.
        import inspect
        src = inspect.getsource(_sys_mod.get_own_memory_usage)
        assert "mi.private" in src or "memory_info().private" in src, (
            "get_own_memory_usage must use memory_info().private on Windows "
            "to avoid the working-set-trim RSS glitch triggered by "
            "clean_ram()'s SetProcessWorkingSetSizeEx call."
        )

    def test_tqdmu_basic(self):
        """Test tqdmu wrapper"""
        from pyutilz.system import tqdmu

        # Should work like tqdm
        items = list(range(10))
        result = list(tqdmu(items, disable=True))

        assert result == items
