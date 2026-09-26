"""Regression tests for audits/2026-09-26/50_general.md (GEN-1 .. GEN-15), one class per finding.

Each test reproduces the audited failure and fails on the pre-fix code; see the disposition file for the mapping.
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import logging
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------------------------------------------------
# GEN-1: full-content array hash
# ---------------------------------------------------------------------------------------------------------------------

import pyutilz.core.disk_cache as dc


class TestGen1FullContentHash:
    def test_middle_swap_1d_changes_the_key(self):
        a = np.arange(1000.0)
        b = a.copy()
        b[[500, 501]] = b[[501, 500]]
        assert dc.hash_array_summary(a) != dc.hash_array_summary(b)

    def test_middle_row_swap_2d_changes_the_key(self):
        a = np.random.default_rng(0).random((1000, 3))
        b = a.copy()
        b[[400, 401]] = b[[401, 400]]
        assert dc.hash_array_summary(a) != dc.hash_array_summary(b)

    def test_change_in_a_column_containing_nan_changes_the_key(self):
        a = np.random.default_rng(1).random((1000, 3))
        a[0, 0] = np.nan
        b = a.copy()
        b[500, 0] = 123.0
        assert dc.hash_array_summary(a) != dc.hash_array_summary(b)

    def test_change_in_a_1d_array_containing_nan_changes_the_key(self):
        a = np.random.default_rng(2).random(1000)
        a[0] = np.nan
        b = a.copy()
        b[500] = 7.0
        assert dc.hash_array_summary(a) != dc.hash_array_summary(b)

    def test_version_was_bumped(self):
        assert dc._HASH_VERSION == 4

    def test_multi_leaf_path_covers_the_last_leaf_and_matches_for_strided_input(self, monkeypatch):
        """Small leaves force the parallel leaf-tree path; a change in the LAST leaf must change the key, and a strided
        view (hashed leaf by leaf via ``arr.flat``) must give the digest of its contiguous copy."""
        monkeypatch.setattr(dc, "_LEAF_BYTES", 256)
        base = np.random.default_rng(3).random((400, 8))
        view = base[:, ::2]
        assert not view.flags["C_CONTIGUOUS"]
        assert dc.hash_array_summary(view) == dc.hash_array_summary(np.ascontiguousarray(view))
        changed = np.ascontiguousarray(view).copy()
        changed[-1, -1] += 1.0
        assert dc.hash_array_summary(changed) != dc.hash_array_summary(np.ascontiguousarray(view))

    def test_multi_leaf_digest_is_independent_of_thread_count(self, monkeypatch):
        monkeypatch.setattr(dc, "_LEAF_BYTES", 256)
        a = np.random.default_rng(4).random(5000)
        monkeypatch.setattr(dc, "_MAX_HASH_THREADS", 1)
        serial = dc.hash_array_summary(a)
        monkeypatch.setattr(dc, "_MAX_HASH_THREADS", 8)
        assert dc.hash_array_summary(a) == serial

    def test_n_summary_rows_is_accepted_and_ignored(self):
        a = np.arange(10_000.0)
        assert dc.hash_array_summary(a, n_summary_rows=1) == dc.hash_array_summary(a)


# ---------------------------------------------------------------------------------------------------------------------
# GEN-2 / GEN-10: kernel directory naming + legacy compat
# ---------------------------------------------------------------------------------------------------------------------

from pyutilz.performance.kernel_tuning import cache as ktc
from pyutilz.performance.kernel_tuning.cache import cache_base as ktc_base


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    ktc.hw_fingerprint.cache_clear()
    yield str(tmp_path)
    ktc.hw_fingerprint.cache_clear()


class TestGen2KernelDirs:
    @pytest.mark.parametrize(
        "names",
        [("hist.cpu", "hist.gpu", "hist cpu", "hist"), ("a@b", "a@c"), ("Mm", "mm"), ("x(1)", "x1"), ("k" * 90 + "a", "k" * 90 + "b")],
    )
    def test_names_that_shared_a_slug_get_distinct_dirs(self, names):
        dirs = {os.path.normcase(ktc._kernel_dir("h", n)) for n in names}
        assert len(dirs) == len(names)

    def test_two_colliding_kernels_both_survive_a_reload(self, tmp_cache_dir):
        c1 = ktc.KernelTuningCache()
        c1.update("hist.cpu", axes=["n"], regions=[{"n_max": None, "variant": "cpu"}])
        c1.update("hist.gpu", axes=["n"], regions=[{"n_max": None, "variant": "gpu"}])
        c2 = ktc.KernelTuningCache()
        assert c2.lookup("hist.cpu", n=1) == {"variant": "cpu"}
        assert c2.lookup("hist.gpu", n=1) == {"variant": "gpu"}

    def _write_legacy(self, cache: Any, name: str, variant: str, tuned: str) -> str:
        legacy = ktc_base._legacy_kernel_dir(cache._path, name)
        os.makedirs(legacy, exist_ok=True)
        rec = {
            "schema_version": ktc.SCHEMA_VERSION,
            "hw_fingerprint": ktc.hw_fingerprint(),
            "kernel_name": name,
            "provenance": ktc._build_provenance(),
            "entry": {"axes": ["n"], "regions": [{"n_max": None, "variant": variant}], "tuned_utc": tuned},
        }
        path = os.path.join(legacy, f"nocv.0.1.{abs(hash((name, variant)))}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        return path

    def test_legacy_shared_dir_yields_every_kernel_it_holds(self, tmp_cache_dir):
        c = ktc.KernelTuningCache()
        c._ensure_loaded()
        self._write_legacy(c, "a@b", "B", "2026-01-01T00:00:00+00:00")
        self._write_legacy(c, "a@c", "C", "2026-01-02T00:00:00+00:00")
        fresh = ktc.KernelTuningCache()
        assert fresh.lookup("a@b", n=1) == {"variant": "B"}
        assert fresh.lookup("a@c", n=1) == {"variant": "C"}

    def test_newer_record_in_new_dir_beats_legacy_one(self, tmp_cache_dir):
        c = ktc.KernelTuningCache()
        c._ensure_loaded()
        self._write_legacy(c, "a@b", "old", "2020-01-01T00:00:00+00:00")
        c.update("a@b", axes=["n"], regions=[{"n_max": None, "variant": "new"}])
        assert ktc.KernelTuningCache().lookup("a@b", n=1) == {"variant": "new"}

    def test_evict_removes_only_this_kernels_legacy_records(self, tmp_cache_dir):
        c = ktc.KernelTuningCache()
        c._ensure_loaded()
        self._write_legacy(c, "a@b", "B", "2026-01-01T00:00:00+00:00")
        self._write_legacy(c, "a@c", "C", "2026-01-02T00:00:00+00:00")
        fresh = ktc.KernelTuningCache()
        assert fresh.evict("a@b") is True
        again = ktc.KernelTuningCache()
        assert again.lookup("a@b", n=1) is None
        assert again.lookup("a@c", n=1) == {"variant": "C"}


class TestGen10NoDuplicateReader:
    def test_unused_duplicate_reader_is_gone(self):
        assert not hasattr(ktc.KernelTuningCache, "_read_kernel_newest")

    def test_gc_uses_the_shared_sort_key(self, tmp_cache_dir):
        c = ktc.KernelTuningCache()
        for i in range(6):
            c.update("k", axes=["n"], regions=[{"n_max": None, "variant": str(i)}], tuned_utc=f"2026-01-0{i + 1}T00:00:00+00:00")
        with mock.patch.object(type(c), "_record_sort_key", wraps=c._record_sort_key) as spy:
            c._gc_kernel_dir(ktc._kernel_dir(c._path, "k"), keep=2)
        assert spy.call_count >= 3
        assert ktc.KernelTuningCache().lookup("k", n=1) == {"variant": "5"}


# ---------------------------------------------------------------------------------------------------------------------
# GEN-3: region matching
# ---------------------------------------------------------------------------------------------------------------------

from pyutilz.performance.kernel_tuning.cache import cache_class as ktc_class
from pyutilz.performance.kernel_tuning.cache.region_matching import _region_match_reason, _region_matches


class TestGen3RegionMatching:
    def test_nan_dim_does_not_match_a_bounded_region(self):
        assert _region_matches({"n_max": 10}, {"n": float("nan")}) is False
        assert _region_matches({"n_min": 10}, {"n": float("nan")}) is False
        assert _region_match_reason({"n_max": 10}, {"n": float("nan")})[1] == "n is NaN"

    def test_tuple_dim_matches_a_list_eq_after_json_reload(self):
        assert _region_matches({"shape_eq": [2, 3]}, {"shape": (2, 3)}) is True
        assert _region_matches({"shape_eq": [2, 3]}, {"shape": (3, 2)}) is False

    def test_lookup_hot_path_agrees(self):
        c = ktc.KernelTuningCache(in_memory=True)
        c.update(
            "k",
            axes=["n", "shape"],
            regions=[{"n_max": 10, "variant": "small"}, {"shape_eq": [2, 3], "variant": "tuple"}, {"n_max": None, "variant": "all"}],
        )
        assert c.lookup("k", n=float("nan")) == {"variant": "tuple"} or c.lookup("k", n=float("nan")) == {"variant": "all"}
        assert c.lookup("k", n=float("nan"))["variant"] != "small"
        assert c.lookup("k", n=100, shape=(2, 3)) == {"variant": "tuple"}

    def test_misspelled_dim_warns_once(self, caplog):
        ktc_class._UNCONSTRAINED_DIM_WARNED.clear()
        c = ktc.KernelTuningCache(in_memory=True)
        c.update("kw", axes=["n"], regions=[{"n_max": 10, "variant": "a"}, {"n_max": None, "variant": "b"}])
        with caplog.at_level(logging.WARNING):
            c.lookup("kw", rows=10**9)
            c.lookup("kw", rows=10**9)
            c.lookup("kw", n=5)
        msgs = [r.getMessage() for r in caplog.records if "no region of that kernel constrains" in r.getMessage()]
        assert len(msgs) == 1 and "'rows'" in msgs[0]


# ---------------------------------------------------------------------------------------------------------------------
# GEN-4: device-wide GPU sync
# ---------------------------------------------------------------------------------------------------------------------

import pyutilz.dev.benchmarking as bm


def _fake_cupy(ctx: int) -> mock.MagicMock:
    cp = mock.MagicMock(name="cupy")
    cp.cuda.runtime.getDeviceCount.return_value = 1
    cp.cuda.driver.ctxGetCurrent.return_value = ctx
    return cp


class TestGen4DeviceWideSync:
    def test_cupy_sync_is_device_wide_not_null_stream(self):
        cp = _fake_cupy(ctx=1)
        with mock.patch.dict(sys.modules, {"cupy": cp}), mock.patch.object(bm, "_numba_cuda_initialised", return_value=False):
            bm.synchronize_gpu_if_available()
        cp.cuda.runtime.deviceSynchronize.assert_called_once_with()
        cp.cuda.Stream.null.synchronize.assert_not_called()

    def test_no_context_means_no_sync_and_no_context_creation(self):
        cp = _fake_cupy(ctx=0)
        with mock.patch.dict(sys.modules, {"cupy": cp}), mock.patch.object(bm, "_numba_cuda_initialised", return_value=False):
            bm.synchronize_gpu_if_available()
        cp.cuda.runtime.deviceSynchronize.assert_not_called()

    def test_driver_is_initialised_before_the_context_probe(self):
        """``cuCtxGetCurrent`` before driver init segfaults; ``getDeviceCount`` must come first."""
        cp = _fake_cupy(ctx=1)
        order: list = []
        cp.cuda.runtime.getDeviceCount.side_effect = lambda: order.append("count") or 1
        cp.cuda.driver.ctxGetCurrent.side_effect = lambda: order.append("ctx") or 1
        assert bm._cupy_has_context(cp) is True
        assert order == ["count", "ctx"]

    def test_numba_cuda_is_synchronized_when_initialised(self):
        nc = types.SimpleNamespace(synchronize=mock.MagicMock())
        with mock.patch.dict(sys.modules, {"cupy": None, "numba.cuda": nc}), mock.patch.object(bm, "_numba_cuda_initialised", return_value=True):
            bm.synchronize_gpu_if_available()
        nc.synchronize.assert_called_once_with()

    def test_numba_cuda_not_touched_when_not_initialised(self):
        nc = types.SimpleNamespace(synchronize=mock.MagicMock())
        with mock.patch.dict(sys.modules, {"cupy": None, "numba.cuda": nc}), mock.patch.object(bm, "_numba_cuda_initialised", return_value=False):
            bm.synchronize_gpu_if_available()
        nc.synchronize.assert_not_called()


# ---------------------------------------------------------------------------------------------------------------------
# GEN-5: NaN / tuple outputs in the backend sweeps
# ---------------------------------------------------------------------------------------------------------------------


class TestGen5SweepEquivalence:
    def test_scale_ignores_nan(self):
        assert bm._output_scale(np.array([np.nan, -3.0, 2.0])) == 3.0
        assert bm._output_scale(np.array([np.nan])) == 1.0
        assert bm._output_scale(np.zeros(3)) == 1.0

    def test_shared_nan_is_equal_and_a_moved_nan_is_not(self):
        a = np.array([np.nan, 1.0, 2.0])
        assert bm._max_abs_diff(a, a.copy()) == 0.0
        assert bm._max_abs_diff(a, np.array([1.0, np.nan, 2.0])) == float("inf")
        assert bm._max_abs_diff(np.array([np.inf, 1.0]), np.array([-np.inf, 1.0])) == float("inf")
        assert bm._max_abs_diff(np.array([1.0, 2.0]), np.array([1.0, 2.5])) == 0.5

    def test_ragged_tuple_and_dict_outputs_are_compared(self):
        ref = (np.ones(2), np.zeros(3))
        assert bm._max_abs_diff(ref, (np.ones(2), np.zeros(3))) == 0.0
        assert bm._max_abs_diff(ref, (np.ones(2), np.full(3, 0.25))) == 0.25
        assert bm._max_abs_diff({"b": 1.0, "a": 2.0}, {"a": 2.0, "b": 1.5}) == 0.5

    def test_to_host_leaves_dicts_alone(self):
        assert bm._to_host({"a": 1}) == {"a": 1}

    def test_crossover_candidate_survives_a_nan_reference(self):
        """A faster candidate matching a NaN-bearing reference must win; before the fix the NaN scale gated it out."""
        import time

        def ref(x):
            time.sleep(0.02)
            out = x.copy()
            out[0] = np.nan
            return out

        def cand(x):
            out = x.copy()
            out[0] = np.nan
            return out

        regions = bm.sweep_backend_crossover({"ref": ref, "cand": cand}, [10], lambda n: (np.arange(float(n)),), "n", repeats=2)
        assert regions[0]["backend_choice"] == "cand"

    def test_crossover_does_not_crash_on_ragged_tuple_outputs(self):
        import time

        def slow(x):
            time.sleep(0.02)
            return (x, x[:2])

        def fast(x):
            return (x, x[:2])

        regions = bm.sweep_backend_crossover({"slow": slow, "fast": fast}, [10], lambda n: (np.arange(float(n)),), "n", repeats=2)
        assert regions[0]["backend_choice"] == "fast"

    def test_grid_reference_returning_a_dict_is_compared_not_skipped(self):
        import time

        def slow(x):
            time.sleep(0.02)
            return {"a": x, "b": x[:1]}

        def fast(x):
            return {"a": x, "b": x[:1]}

        regions = bm.sweep_backend_grid({"slow": slow, "fast": fast}, {"n": [10]}, lambda d: (np.arange(float(d["n"])),), repeats=2)
        assert regions and regions[0]["backend_choice"] == "fast"


# ---------------------------------------------------------------------------------------------------------------------
# GEN-6 / GEN-7: gpu_dispatch
# ---------------------------------------------------------------------------------------------------------------------

import pyutilz.system.gpu_dispatch as gd


class TestGen6MemoryGuard:
    def _run(self, driver_free: int, pooled: int, required: int, **kw: Any) -> mock.MagicMock:
        cp = mock.MagicMock(name="cupy")
        with mock.patch.object(gd, "is_cuda_available", return_value=True), mock.patch.object(
            gd, "_free_bytes_via_cupy", return_value=driver_free
        ), mock.patch.object(gd, "_pool_free_bytes", return_value=pooled), mock.patch.dict(sys.modules, {"cupy": cp}):
            with gd.cuda_memory_guard(required, headroom_factor=1.0, **kw):
                pass
        return cp

    def test_pooled_memory_counts_as_available(self):
        # 100 free on the driver alone is short of 500; with the 1000 pooled bytes it is not, so no MemoryError, and
        # the guard flushes the pool to hand those bytes back.
        cp = self._run(driver_free=100, pooled=1000, required=500)
        cp.get_default_memory_pool.return_value.free_all_blocks.assert_called_once_with()

    def test_still_raises_when_driver_plus_pool_is_short(self):
        with pytest.raises(MemoryError):
            self._run(driver_free=100, pooled=100, required=500)

    def test_pool_not_flushed_when_memory_was_not_tight(self):
        cp = self._run(driver_free=10_000, pooled=0, required=500)
        cp.get_default_memory_pool.return_value.free_all_blocks.assert_not_called()

    def test_pool_flushed_when_memory_was_tight(self):
        cp = self._run(driver_free=100, pooled=1000, required=500)
        cp.get_default_memory_pool.return_value.free_all_blocks.assert_called_once_with()

    def test_release_pool_overrides(self):
        cp = self._run(driver_free=10_000, pooled=0, required=500, release_pool=True)
        cp.get_default_memory_pool.return_value.free_all_blocks.assert_called_once_with()
        cp = self._run(driver_free=100, pooled=1000, required=500, release_pool=False)
        cp.get_default_memory_pool.return_value.free_all_blocks.assert_not_called()


class TestGen7CudaOrdinals:
    def test_uuid_formats(self):
        raw = bytes.fromhex("31e313d0e5f5826bd399a9cc83ade48a") + b"\x13\x14\x01"
        assert gd._format_uuid(raw) == "GPU-31e313d0-e5f5-826b-d399-a9cc83ade48a"
        assert gd._format_uuid("GPU-ABC") == "gpu-abc"

    def test_smi_ids_are_remapped_and_invisible_gpus_dropped(self):
        gpus = [{"id": 0, "uuid": "GPU-aaa", "memoryFree": 1.0}, {"id": 1, "uuid": "GPU-bbb", "memoryFree": 9.0}]
        with mock.patch.object(gd, "_cuda_ordinals_by_uuid", return_value={"gpu-bbb": 0}):
            out = gd._gputil_to_cuda_ids(gpus)
        assert out == [{"id": 0, "smi_id": 1, "uuid": "GPU-bbb", "memoryFree": 9.0}]

    def test_select_best_gpu_returns_the_cuda_ordinal(self):
        gd.reset_cache()
        gpus = [
            {"id": 0, "uuid": "GPU-aaa", "memoryFree": 1.0, "memoryTotal": 2.0, "load": 0.0, "name": "a"},
            {"id": 1, "uuid": "GPU-bbb", "memoryFree": 9.0, "memoryTotal": 10.0, "load": 0.0, "name": "b"},
        ]
        try:
            with mock.patch.object(gd, "is_cuda_available", return_value=True), mock.patch.object(
                gd, "get_gpuutil_gpu_info", return_value=gpus
            ), mock.patch.object(gd, "_cuda_ordinals_by_uuid", return_value={"gpu-aaa": 1, "gpu-bbb": 0}):
                assert gd.select_best_gpu("vram") == 0
        finally:
            gd.reset_cache()


# ---------------------------------------------------------------------------------------------------------------------
# GEN-8 / GEN-12: safe_pickle
# ---------------------------------------------------------------------------------------------------------------------

import pyutilz.core.safe_pickle as sp


class TestGen8SafePickle:
    def test_load_verifies_the_bytes_it_unpickles_without_rereading(self, tmp_path):
        p = str(tmp_path / "x.pkl")
        sp.safe_dump({"a": 1}, p)
        with mock.patch.object(sp, "_sha256_of_file", side_effect=AssertionError("file re-read")):
            assert sp.safe_load(p) == {"a": 1}

    def test_load_still_rejects_tampered_payload(self, tmp_path):
        p = str(tmp_path / "x.pkl")
        sp.safe_dump({"a": 1}, p)
        import pickle

        with open(p, "wb") as f:
            pickle.dump({"a": 2}, f)
        with pytest.raises(sp.PickleVerificationError):
            sp.safe_load(p)

    def test_dump_sidecar_matches_the_file_without_reading_it_back(self, tmp_path):
        p = str(tmp_path / "x.pkl")
        with mock.patch.object(sp, "_sha256_of_file", side_effect=AssertionError("read back")):
            sp.safe_dump(list(range(100)), p)
        with open(p, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        with open(p + ".sha256", encoding="utf-8") as f:
            assert f.read().split()[0] == digest

    def test_sidecar_is_replaced_atomically(self, tmp_path):
        p = str(tmp_path / "x.pkl")
        with open(p, "wb") as f:
            f.write(b"payload")
        with mock.patch.object(sp, "_replace_with_retry", wraps=sp._replace_with_retry) as spy:
            sp.write_sidecar(p)
        assert [c.args[1] for c in spy.call_args_list] == [p + ".sha256"]
        assert not glob.glob(p + ".sha256.tmp*")


class TestGen12PathLockCase:
    def test_lock_key_is_normcased(self, tmp_path):
        p = str(tmp_path / "Some.PKL")
        with sp._get_path_lock(p):
            assert os.path.normcase(os.path.abspath(p)) in sp._path_locks
            if os.name == "nt":
                assert os.path.normcase(os.path.abspath(p.upper())) in sp._path_locks


# ---------------------------------------------------------------------------------------------------------------------
# GEN-9: unique temp name for the hw fingerprint file
# ---------------------------------------------------------------------------------------------------------------------


class TestGen9FingerprintTempName:
    def test_each_writer_uses_its_own_temp_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ktc_base, "cache_dir", lambda: str(tmp_path))
        seen: list = []
        real_replace = os.replace

        def _spy(src: str, dst: str) -> None:
            seen.append(src)
            real_replace(src, dst)

        monkeypatch.setattr(ktc_base.os, "replace", _spy)
        threads = [threading.Thread(target=ktc_base._write_hw_fingerprint_to_disk, args=("fp",)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        final = os.path.join(str(tmp_path), ktc_base._HW_FP_DISK_FILENAME)
        assert len(seen) == 4 and len(set(seen)) == 4
        assert final + ".tmp" not in seen
        assert not glob.glob(final + ".*.tmp")


# ---------------------------------------------------------------------------------------------------------------------
# GEN-11: SingleFlightCache
# ---------------------------------------------------------------------------------------------------------------------

from pyutilz.system.single_flight_cache import SingleFlightCache


class TestGen11SingleFlight:
    def test_recheck_hit_counts_as_hit(self):
        async def main() -> tuple:
            sfc = SingleFlightCache()
            cache: dict = {}
            lock = sfc._get_inflight_lock()
            await lock.acquire()

            async def fetcher() -> str:
                raise AssertionError("must not fetch")

            task = asyncio.ensure_future(sfc.get_or_fetch(cache, "k", fetcher))
            await asyncio.sleep(0)
            cache["k"] = "v"
            lock.release()
            return await task, sfc.hits, sfc.misses

        assert asyncio.run(main()) == ("v", 1, 0)

    def test_cancelled_fetcher_wakes_waiters_even_while_the_lock_is_held(self):
        async def main() -> Any:
            sfc = SingleFlightCache()
            cache: dict = {}
            started = asyncio.Event()

            async def slow() -> str:
                started.set()
                await asyncio.sleep(10)
                return "x"

            fetch_task = asyncio.ensure_future(sfc.get_or_fetch(cache, "k", slow))
            await started.wait()
            waiter = asyncio.ensure_future(sfc.get_or_fetch(cache, "k", slow, default="dflt"))
            await asyncio.sleep(0)
            lock = sfc._get_inflight_lock()
            await lock.acquire()  # the old finally awaited this lock, so it could not reach evt.set()
            fetch_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await fetch_task
            try:
                return await asyncio.wait_for(waiter, timeout=2.0)
            finally:
                lock.release()

        assert asyncio.run(main()) == "dflt"


# ---------------------------------------------------------------------------------------------------------------------
# GEN-13 / GEN-14: parallel helpers
# ---------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib.objects import batch
from pyutilz.system.parallel import distribute_work, split_array, split_list_into_chunks, split_list_into_chunks_indices


class TestGen13Lpt:
    def test_largest_first_balances_better_than_input_order(self):
        work, idx = distribute_work([1, 1, 1, 1, 4], n_jobs=2)
        assert max(sum(w) for w in work) == 4
        assert sorted(sum(idx, [])) == [0, 1, 2, 3, 4]
        for w_items, w_idx in zip(work, idx):
            assert w_idx == sorted(w_idx)
            assert w_items == [[1, 1, 1, 1, 4][i] for i in w_idx]


class TestGen14Chunking:
    def test_helpers_agree(self):
        data = list(range(10))
        assert [data[a:b] for a, b in split_list_into_chunks_indices(data, 3)] == list(split_list_into_chunks(data, 3))
        assert split_array(np.zeros(10), 3) == list(split_list_into_chunks_indices(data, 3))
        assert list(batch(data, 3)) == list(split_list_into_chunks(data, 3))

    @pytest.mark.parametrize("n", [0, -1])
    def test_all_reject_non_positive_sizes(self, n):
        with pytest.raises(ValueError):
            list(batch([1, 2, 3], n))
        with pytest.raises(ValueError):
            split_array([1, 2, 3], n)
        with pytest.raises(ValueError):
            list(split_list_into_chunks([1, 2, 3], n))


# ---------------------------------------------------------------------------------------------------------------------
# GEN-15: the one atomic_write_staging finding
# ---------------------------------------------------------------------------------------------------------------------

import gzip

import pyutilz.data.git_checkpoint_cache as gcc


class TestGen15CheckpointRestoreIsAtomic:
    def test_restore_goes_through_the_atomic_writer(self, tmp_path: Path):
        ckpt = tmp_path / "c.json.gz"
        ckpt.write_bytes(gzip.compress(json.dumps({"a": 1}).encode("utf-8")))
        cache_path = tmp_path / "sub" / "c.json"
        with mock.patch.object(gcc, "_atomic_write_bytes", wraps=gcc._atomic_write_bytes) as spy:
            assert gcc.read_json_with_checkpoint_fallback(cache_path, ckpt) == {"a": 1}
        assert spy.call_args.args[0] == cache_path
        assert json.loads(cache_path.read_text(encoding="utf-8")) == {"a": 1}
