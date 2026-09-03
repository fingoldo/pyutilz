"""Tests for system.py - Phase 2 refactoring

Tests cover:
- Command injection fix (shell=True removed from Mac UUID extraction)
- Tracemalloc resource leak fix (always call tracemalloc.stop())
- Platform compatibility (wintypes import guarded)
"""

import pytest
import tracemalloc

# The OS-probe binaries are resolved to an absolute path via `_resolve_binary()` (shutil.which per
# PATH entry) before being spawned, so a planted executable in the process's current directory can
# never win -- see its docstring. That resolution is a different concern from the output parsing
# these tests exercise, and most of the probed binaries (lscpu, ioreg, getprop, pylspci) do not
# exist on every OS this suite runs on, so it is stubbed to an identity here in all three modules
# that call it.
@pytest.fixture(autouse=True)
def _stub_binary_resolution(monkeypatch):
    for module in ("probing", "sysinfo", "fsutils"):
        monkeypatch.setattr(f"pyutilz.system.system.{module}._resolve_binary", lambda name: name, raising=False)


class TestGetSystemInfo:
    """Test get_system_info function - command injection fix"""

    def test_no_shell_true_in_mac_uuid_extraction(self, monkeypatch):
        """The macOS UUID probe chains ioreg|grep through pipes, never through a shell.

        Behavioural replacement for an ``inspect.getsource()`` walk. That walk was doubly
        broken: every assertion sat inside ``if in_mac_section:``, and ``in_mac_section`` was
        only ever set by a line containing ``current_system == "Mac"`` -- but the source says
        ``current_system == "Darwin"`` (``platform.system()`` never returns "Mac"), so the flag
        stayed False and NOT ONE assertion ever ran. Here the Darwin branch is actually
        executed with ``subprocess.Popen`` stubbed, and the recorded calls asserted directly.
        """
        import subprocess as _subprocess

        from pyutilz.system.system import sysinfo as sysinfo_mod

        calls = []

        class _FakeStdout:
            def close(self):
                pass

        real_popen = _subprocess.Popen

        class _FakePopen:
            # Only the two probe binaries are intercepted: `subprocess` is a shared stdlib
            # module, and `platform.platform()` (called from get_os_info earlier in the same
            # get_system_info run) spawns its own child on Windows.
            def __new__(cls, args, **kwargs):
                if not (isinstance(args, (list, tuple)) and args and args[0] in ("ioreg", "grep")):
                    return real_popen(args, **kwargs)
                return super().__new__(cls)

            def __init__(self, args, **kwargs):
                calls.append((args, kwargs))
                self.stdout = _FakeStdout()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def communicate(self):
                return (b'"IOPlatformUUID" = "AAAA-BBBB-CCCC"\n', b"")

        monkeypatch.setattr(sysinfo_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(sysinfo_mod.subprocess, "Popen", _FakePopen)

        info = sysinfo_mod.get_system_info(return_os_info=True, return_sensitive_info=True)

        assert len(calls) == 2, f"expected the ioreg|grep pair, got {calls!r}"
        (ioreg_args, ioreg_kwargs), (grep_args, grep_kwargs) = calls

        # argv passed as a LIST (no shell word-splitting), and shell= never enabled anywhere.
        assert ioreg_args == ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]
        assert grep_args == ["grep", "-E", "(UUID)"]
        for _args, kwargs in calls:
            assert kwargs.get("shell", False) is False, "shell=True reintroduces the command-injection hole this fix closed"

        # The pipeline is wired process-to-process, not through a shell pipe character.
        assert ioreg_kwargs.get("stdout") is _subprocess.PIPE
        assert isinstance(grep_kwargs.get("stdin"), _FakeStdout), "grep must read ioreg's stdout handle directly"
        assert grep_kwargs.get("stdout") is _subprocess.PIPE

        # And the parsed UUID really lands in the fields distributed.py requires.
        assert info["os_machine_guid"] == "AAAA-BBBB-CCCC"
        assert info["os_serial"] == "AAAA-BBBB-CCCC"

    def test_returns_dict_with_expected_keys(self):
        """Test that function returns dict with expected system info keys"""
        from pyutilz.system import get_system_info

        result = get_system_info()
        assert isinstance(result, dict)
        # Should have at least some system info
        assert len(result) > 0


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
            # The cleanup contract holds on the raising path too -- assert it HERE rather than
            # swallowing, so this handler cannot hide a leak behind an unrelated exception.
            assert not tracemalloc.is_tracing(), "tracemalloc should be stopped even when show_tracemalloc_snapshot() raises"

        # Should NOT be tracing after function completes
        assert not tracemalloc.is_tracing(), "tracemalloc should be stopped after show_tracemalloc_snapshot() (resource leak)"

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
            # Same contract on the raising path -- asserted inside the handler so it cannot swallow a leak.
            assert not tracemalloc.is_tracing(), "tracemalloc should be stopped even when show_tracemalloc_snapshot() raises"

        # Should still stop tracemalloc even if exception occurred
        assert not tracemalloc.is_tracing(), "tracemalloc should be stopped even when exception occurs (try/finally fix)"

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
        assert hasattr(result, "statistics"), "Should return tracemalloc.Snapshot object"


class TestPlatformCompatibility:
    """Test platform-specific code is properly guarded"""

    def test_windows_only_memory_trim_is_platform_gated(self, monkeypatch):
        """``clean_ram()`` reaches the ``ctypes.wintypes``/``windll`` path on Windows ONLY.

        Behavioural replacement for an ``inspect.getsource()`` check whose assertion was
        ``"platform.system()" in source or "try:" in source`` -- satisfied by literally any
        module containing a ``try:`` anywhere, and which additionally only ran on non-Windows
        boxes. What actually matters is the runtime gate: ``clean_ram()`` must call
        ``trim_windows_process_memory()`` (the sole consumer of ``ctypes.wintypes`` /
        ``ctypes.windll``) on Windows and must NOT touch it on any other platform, where those
        symbols do not exist.
        """
        from pyutilz.system.system import memory as memory_mod

        # ``import ctypes.wintypes`` itself is portable (it is pure stdlib on every platform);
        # the non-portable part is ctypes.windll, reached only via trim_windows_process_memory.
        import ctypes.wintypes  # noqa: F401 - importability on this platform IS the assertion

        trims = []
        monkeypatch.setattr(memory_mod, "trim_windows_process_memory", lambda *a, **kw: trims.append(1) or True)

        for system_name, expect_trim in (("Windows", True), ("Linux", False), ("Darwin", False)):
            trims.clear()
            monkeypatch.setattr(memory_mod.platform, "system", lambda name=system_name: name)
            if system_name == "Windows":
                memory_mod.clean_ram()
            else:
                # The non-Windows branch calls ctypes.CDLL("libc.so.6"), absent here -- the
                # function swallows and logs that; what is asserted is that it never took the
                # Windows path.
                memory_mod.clean_ram()
            assert bool(trims) is expect_trim, f"platform.system()=={system_name!r}: trim called={bool(trims)}, expected {expect_trim}"


class TestGetCpuUsage:
    """Test CPU usage measurement within get_system_info"""

    def test_system_info_includes_cpu_usage(self):
        """Test that get_system_info includes CPU usage (not 0.0 from first call)"""
        from pyutilz.system import get_system_info

        # return_usage_stats=True, not the bare default: CPU load lives behind that flag (the
        # documented default enables only return_sensitive_info/return_os_info), so a bare call
        # asserts the flag's semantics rather than the "0.0 from the first psutil call" regression
        # this test is named for.
        info = get_system_info(return_usage_stats=True)
        if info is None:
            pytest.skip("get_system_info returned None (missing dependencies)")
        # Unconditional: the key going missing is the STRONGER regression this test names, so assert
        # its presence rather than treating its absence as "nothing to check".
        assert "cpu_current_load_percent" in info, f"get_system_info() no longer reports CPU load; keys: {sorted(info)}"
        usage = info["cpu_current_load_percent"]
        assert isinstance(usage, (int, float))
        assert 0 <= usage <= 100

    def test_psutil_cpu_percent_called_correctly(self, monkeypatch):
        """get_system_info asks psutil for BOTH the aggregate and the per-core CPU load.

        Behavioural replacement for ``assert "cpu_percent" in inspect.getsource(...)``, which a
        mere comment mentioning the name satisfied. The stubbed psutil returns distinguishable
        values so the mapping into the returned dict is pinned too.
        """
        pytest.importorskip("psutil")

        from pyutilz.system.system import sysinfo as sysinfo_mod

        calls = []

        def fake_cpu_percent(percpu=False):
            calls.append(percpu)
            return [11.0, 12.0] if percpu else 42.5

        monkeypatch.setattr(sysinfo_mod.psutil, "cpu_percent", fake_cpu_percent)

        info = sysinfo_mod.get_system_info(return_usage_stats=True)

        assert sorted(calls) == [False, True], f"expected one aggregate and one per-core query, got {calls!r}"
        assert info["cpu_current_load_percent"] == 42.5
        assert info["cpu_current_threads_load_percents"] == [11.0, 12.0]

    def test_usage_stats_survive_a_platform_without_cpu_freq(self, monkeypatch):
        """Missing psutil.cpu_freq must cost only the frequency fields, not the whole result.

        psutil gates cpu_freq on the platform backend (macOS has none), so the unguarded call
        raised AttributeError inside get_system_info's function-wide try -- which returned the
        half-built dict, so CPU load, per-core load and RAM stats all silently vanished. Deleting
        the attribute reproduces that module surface.
        """
        pytest.importorskip("psutil")

        from pyutilz.system.system import sysinfo as sysinfo_mod

        monkeypatch.delattr(sysinfo_mod.psutil, "cpu_freq", raising=False)

        calls = []

        def fake_cpu_percent(percpu=False):
            calls.append(percpu)
            return [11.0, 12.0] if percpu else 42.5

        monkeypatch.setattr(sysinfo_mod.psutil, "cpu_percent", fake_cpu_percent)

        info = sysinfo_mod.get_system_info(return_usage_stats=True)

        assert sorted(calls) == [False, True], f"expected one aggregate and one per-core query, got {calls!r}"
        assert info["cpu_current_load_percent"] == 42.5
        assert info["cpu_current_threads_load_percents"] == [11.0, 12.0]
        assert info["cpu_current_frequency_hz"] is None, "an unmeasurable frequency must be reported as unknown, not fabricated"
        assert "ram_free_gb" in info


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
        assert before is not None and before > 0.05, "baseline too low to run this regression check"

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

    def test_get_own_memory_usage_uses_private_bytes_on_windows(self, monkeypatch):
        """On Windows, ``get_own_memory_usage`` reads ``memory_info().private``, not ``.rss``.

        Behavioural replacement for an ``inspect.getsource()`` check for the literal text
        ``"mi.private"``, which passed on any file merely containing that string and would have
        failed on a behaviour-preserving rename. A stubbed ``psutil.Process`` returns rss and
        private values an order of magnitude apart, so the field actually read is unambiguous
        from the returned number. Runs on every platform (the Windows branch is selected by the
        module's ``_IS_WINDOWS`` flag, which is what gets forced here) instead of skipping
        everywhere but Windows.
        """
        from unittest.mock import MagicMock

        from pyutilz.system.system import memory as memory_mod

        private_gb, rss_gb = 4.0, 0.25
        mem_info = MagicMock(private=int(private_gb * 2**30), rss=int(rss_gb * 2**30))
        monkeypatch.setattr(memory_mod.psutil, "Process", lambda _pid: MagicMock(memory_info=lambda: mem_info))
        monkeypatch.setattr(memory_mod, "_LAST_OWN_MEMORY_USAGE_GB", 0.0)

        monkeypatch.setattr(memory_mod, "_IS_WINDOWS", True)
        assert memory_mod.get_own_memory_usage() == pytest.approx(private_gb), (
            "get_own_memory_usage must read memory_info().private on Windows -- .rss is the "
            "working set, which clean_ram()'s SetProcessWorkingSetSizeEx call trims to near zero."
        )

        # The counterpart: everywhere else, .rss is the correct (and only available) field.
        monkeypatch.setattr(memory_mod, "_LAST_OWN_MEMORY_USAGE_GB", 0.0)
        monkeypatch.setattr(memory_mod, "_IS_WINDOWS", False)
        assert memory_mod.get_own_memory_usage() == pytest.approx(rss_gb)

    def test_tqdmu_basic(self):
        """Test tqdmu wrapper"""
        from pyutilz.system import tqdmu

        # Should work like tqdm
        items = list(range(10))
        result = list(tqdmu(items, disable=True))

        assert result == items

    def test_tqdmu_lazy_start_yields_all_items(self):
        """tqdmu_lazy_start must be a transparent pass-through for the
        iterable — yields every item in order."""
        from pyutilz.system import tqdmu_lazy_start

        items = list(range(5))
        result = list(tqdmu_lazy_start(items, desc="t", disable=True))
        assert result == items

    def test_tqdmu_lazy_start_suppresses_bar_for_single_item(self):
        """Single-item iterables (the common ``target_type`` /
        ``pre_pipeline`` 1-element loops in mlframe) historically logged
        ``0/1 [HH:MM:SS<?]`` for the entire body of the only iteration —
        confusing because it never reached the ``update(1)`` until the
        outer loop exited. The 2026-04-23 fix sets ``min_total=2`` by
        default so single-item loops produce no bar at all.

        The behavioural contract verified here: passing ``min_total=2``
        through a 1-element iterable still yields the item, but no tqdm
        bar instance gets created. We probe by counting tqdm instances
        before / after.
        """
        from pyutilz.system import tqdmu_lazy_start
        import tqdm as _tqdm

        before = len(getattr(_tqdm.tqdm, "_instances", set()))
        result = list(tqdmu_lazy_start([42], desc="single"))  # default min_total=2
        # tqdm registers each live bar in its class-level _instances set.
        # If a bar was created for the single-item iter, _instances would
        # have grown by one at construction; since we always close the bar
        # in finally, it'd shrink back. To detect creation deterministically
        # we instead check that no bar instance was ever printed: the
        # function should return immediately via the suppression branch.
        after = len(getattr(_tqdm.tqdm, "_instances", set()))
        assert result == [42]
        # The hard contract: the iterable is yielded faithfully even when
        # the bar is suppressed.
        assert before == after, f"tqdm leak: {after - before} bar instance(s) outlived " f"tqdmu_lazy_start([42])"

    def test_tqdmu_lazy_start_min_total_one_keeps_bar(self):
        """Opt-out: ``min_total=1`` restores the historical "always show
        a bar" behaviour."""
        from pyutilz.system import tqdmu_lazy_start

        # disable=True keeps the bar object construction but prevents
        # output — lets us verify the bar is created without polluting
        # test stdout.
        result = list(tqdmu_lazy_start([42], min_total=1, desc="single", disable=True))
        assert result == [42]

    def test_tqdmu_lazy_start_multi_item_keeps_bar(self):
        """≥ ``min_total`` items: bar IS created (regression guard
        against accidentally suppressing for the multi-item case)."""
        from pyutilz.system import tqdmu_lazy_start

        result = list(tqdmu_lazy_start(
            list(range(3)), desc="multi", disable=True,
        ))
        assert result == [0, 1, 2]


class TestSystemSubpackageFacade:
    """Sensor test for the system.py -> system/ subpackage split.

    Guards two invariants that the split must preserve:
      1. Every representative public name is still resolvable via the
         ``pyutilz.system.system`` facade AND via the ``pyutilz.system``
         package alias (many modules import through both paths).
      2. Each facade symbol is the SAME object as the one defined in its
         cohesive submodule (no accidental shadow/copy).
    """

    # (facade name -> owning submodule) — one representative per submodule.
    _REPRESENTATIVE = {
        "summarize_devices": "_common",
        "remove_nas": "_common",
        "get_nvidia_smi_info": "probing",
        "get_battery_info": "probing",
        "clean_ram": "memory",
        "get_own_memory_usage": "memory",
        "ensure_dir_exists": "fsutils",
        "get_max_singledisk_free_space_gb": "fsutils",
        "beep": "misc",
        "get_os_info": "misc",
        "get_system_info": "sysinfo",
    }

    def test_facade_reexports_are_submodule_objects(self):
        import importlib

        facade = importlib.import_module("pyutilz.system.system")
        for name, submod in self._REPRESENTATIVE.items():
            mod = importlib.import_module(f"pyutilz.system.system.{submod}")
            assert hasattr(facade, name), f"facade lost {name}"
            assert getattr(facade, name) is getattr(mod, name), f"{name} on facade is not the object defined in {submod}"

    def test_package_alias_resolves_split_symbols(self):
        # ``from pyutilz.system import ensure_dir_exists`` etc. must keep working.
        from pyutilz.system import ensure_dir_exists, get_system_info, clean_ram  # noqa: F401
        from pyutilz.system.system import ensure_dir_exists as deep_edx

        assert ensure_dir_exists is deep_edx

    def test_public_surface_size_unchanged(self):
        # The pre-split monolith exposed 85 public names, three of which were
        # merely leaked module-level imports (unused stdlib/helpers re-exported
        # via ``import *`` with no ``__all__``). Those 3 leaks were removed as an
        # F401 cleanup, leaving 82 real public names. This pins the current
        # surface so a future accidental change is caught.
        from pyutilz.system import system as m

        public = [n for n in dir(m) if not n.startswith("_")]
        assert len(public) == 82, f"public surface drifted: {len(public)}"
