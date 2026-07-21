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
        planned, indices = distribute_work([1, 2, 3], nworkers=0)
        assert len(planned) == 1

    @patch("pyutilz.system.parallel.psutil.cpu_count", return_value=None)
    def test_applyfunc_parallel_handles_none_cpu_count(self, mock_cc):
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
