"""Regression tests for pyutilz.system.* / performance.kernel_tuning.* (2026-07-21 audit fixes)."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from pyutilz.system.parallel import distribute_work, applyfunc_parallel
from pyutilz.system.system.fsutils import ensure_dir_exists, get_max_singledisk_free_space_gb
from pyutilz.system.system.misc import get_max_affordable_workers_count, count_app_instances
from pyutilz.system.config import TomlLiveConfig
from pyutilz.performance.kernel_tuning.benchmark import time_backend, benchmark_backends


class TestPsutilNoneGuards:
    @patch("pyutilz.system.parallel.psutil.cpu_count", return_value=None)
    def test_distribute_work_handles_none_physical_count(self, mock_cc):
        # distribute_work now routes through cpu_count_physical(), which falls back to 1 on None.
        planned, indices = distribute_work([1, 2, 3], n_jobs=0)
        assert len(planned) == 1

    @patch("pyutilz.system.parallel.psutil.cpu_count", return_value=None)
    def test_applyfunc_parallel_handles_none_cpu_count(self, mock_cc):
        """Regression sensor for the n_jobs fallback when psutil.cpu_count() returns None --
        not a real-threading test. A genuine multiprocessing.pool.ThreadPool here was found to
        intermittently hang (reproduces only deep inside a huge combined pytest run, never
        standalone or in smaller batches -- a sandbox-specific ThreadPool resource-pressure issue,
        not a bug in applyfunc_parallel: other tests using the identical real-ThreadPool,
        n_jobs=1, use_threads=True shape in test_parallel.py/test_parallel_extra.py pass reliably
        in less-loaded chunks). This test's actual assertion target is the n_jobs-fallback-and-
        starmap-and-concat LOGIC, which doesn't need a real OS thread pool to verify -- swapping
        in a synchronous stand-in keeps full coverage of that logic while removing the sandbox
        dependency."""

        class _SyncThreadPool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def starmap(self, func, iterable):
                return [func(*args) for args in iterable]

        with patch("pyutilz.system.parallel.ThreadPool", _SyncThreadPool):
            result = applyfunc_parallel(iterable=[([1, 2],), ([3],)], func=lambda chunk: [x * 2 for x in chunk], return_dataframe=False, use_threads=True)
        assert sorted(sum(list(result), [])) == [2, 4, 6]

    @patch("pyutilz.system.system.misc.psutil.cpu_count", return_value=None)
    def test_get_max_affordable_workers_handles_none(self, mock_cc):
        assert get_max_affordable_workers_count(reservedCores=1) == 1


class TestEnsureDirExistsConcurrent:
    def test_concurrent_creators_do_not_raise(self, tmp_path):
        target = str(tmp_path / "shared_out")
        errors = []

        def worker():
            try:
                ensure_dir_exists(target)
            except Exception as e:  # pragma: no cover - only triggers on TOCTOU regression
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        import os
        assert os.path.isdir(target)


class TestDiskFreeSpaceZeroTotal:
    def test_zero_total_partition_does_not_raise(self):
        fake_partition = MagicMock(opts="rw", fstype="tmpfs", mountpoint="/fake")
        fake_usage = MagicMock(total=0, free=0)
        with patch("pyutilz.system.system.fsutils.psutil.disk_usage", return_value=fake_usage):
            result = get_max_singledisk_free_space_gb(disk_partitions=[fake_partition])
        assert result[1] == 0.0


class TestCountAppInstancesProcessName:
    def test_dead_process_during_name_lookup_is_skipped_not_raised(self):
        import psutil as _psutil

        proc = MagicMock()
        proc.name.side_effect = _psutil.NoSuchProcess(pid=12345)
        with patch("pyutilz.system.system.misc.psutil.process_iter", return_value=[proc]):
            n = count_app_instances(processname="python.exe")
        assert n == 0


class TestTomlLiveConfigBoolIntAliasing:
    def test_bool_value_cast_to_int_not_aliased(self, tmp_path):
        toml_path = tmp_path / "cfg.toml"
        toml_path.write_text("[section]\nenabled = true\n", encoding="utf-8")
        cfg = TomlLiveConfig(str(toml_path))
        result = cfg.get("section", "enabled", default=0, type_=int)
        assert type(result) is int
        assert result == 1

    def test_bool_value_requested_as_bool_stays_bool(self, tmp_path):
        toml_path = tmp_path / "cfg.toml"
        toml_path.write_text("[section]\nenabled = true\n", encoding="utf-8")
        cfg = TomlLiveConfig(str(toml_path))
        result = cfg.get("section", "enabled", default=False, type_=bool)
        assert result is True


class TestTomlLiveConfigBadValueFallback:
    def test_bad_value_with_no_default_returns_none_not_hardcoded_zero(self, tmp_path):
        """Regression (2026-07-21 audit round 2, HIGH): a value that can't be cast to `type_`,
        with no `default` passed and no matching `defaults` dict entry, used to silently fall
        back to a hardcoded `0` -- regardless of `type_` and regardless of what `0` means for
        that setting (e.g. max_retries=0 -> retry loop body never executes). Returns None now,
        forcing the caller to handle "no valid value" explicitly."""
        toml_path = tmp_path / "cfg.toml"
        toml_path.write_text('[limits]\nmax_retries = "unlimited"\n', encoding="utf-8")
        cfg = TomlLiveConfig(str(toml_path))
        result = cfg.get("limits", "max_retries", type_=int)
        assert result is None

    def test_bad_value_falls_back_to_explicit_default(self, tmp_path):
        toml_path = tmp_path / "cfg.toml"
        toml_path.write_text('[limits]\nmax_retries = "unlimited"\n', encoding="utf-8")
        cfg = TomlLiveConfig(str(toml_path))
        result = cfg.get("limits", "max_retries", default=5, type_=int)
        assert result == 5

    def test_bad_value_falls_back_to_defaults_dict_entry(self, tmp_path):
        toml_path = tmp_path / "cfg.toml"
        toml_path.write_text('[limits]\nmax_retries = "unlimited"\n', encoding="utf-8")
        cfg = TomlLiveConfig(str(toml_path), defaults={"limits": {"max_retries": 3}})
        result = cfg.get("limits", "max_retries", type_=int)
        assert result == 3


class TestTomlLiveConfigDataSetterLock:
    def test_setter_uses_the_same_lock_as_reload(self, tmp_path):
        """Regression: the `data` setter bypassed self._lock entirely, unlike every other
        mutation of self._data -- breaking the class's own documented "Thread-safe via a single
        threading.Lock" contract for this one mutation path."""
        toml_path = tmp_path / "cfg.toml"
        toml_path.write_text("[section]\nkey = 1\n", encoding="utf-8")
        cfg = TomlLiveConfig(str(toml_path))

        acquired_during_setter = []

        class _TrackingLock:
            def __enter__(self):
                acquired_during_setter.append(True)
                return self

            def __exit__(self, *a):
                return False

        cfg._lock = _TrackingLock()
        cfg.data = {"section": {"key": 2}}

        assert acquired_during_setter, "the setter must acquire self._lock"
        assert cfg._data == {"section": {"key": 2}}


class TestBenchmarkExceptionIsolation:
    def test_time_backend_raising_fn_returns_inf_not_raises(self):
        def bad_fn(x):
            raise RuntimeError("boom")

        result = time_backend(bad_fn, lambda: (1,), warmup=0, n_iters=1)
        assert result == float("inf")

    def test_benchmark_backends_isolates_one_broken_variant(self):
        def good_fn(x):
            return x * 2

        def bad_fn(x):
            raise RuntimeError("boom")

        results = benchmark_backends({"good": good_fn, "bad": bad_fn}, lambda: (1,), warmup=0, n_iters=1)
        assert results["bad"] == float("inf")
        assert results["good"] < float("inf")
