"""Smoke tests for pyutilz.performance.kernel_tuning.cache."""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from pyutilz.performance.kernel_tuning import cache as ktc


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    # Reset the lru_cache on hw_fingerprint so the env var is honoured
    ktc.hw_fingerprint.cache_clear()
    yield str(tmp_path)
    ktc.hw_fingerprint.cache_clear()


class TestCacheRoundTrip:
    def test_empty_cache_returns_none_on_lookup(self, tmp_cache_dir):
        cache = ktc.KernelTuningCache()
        assert cache.has("anything") is False
        assert cache.get_regions("anything") is None
        assert cache.lookup("anything", n_samples=1000) is None

    def test_update_then_lookup(self, tmp_cache_dir):
        cache = ktc.KernelTuningCache()
        regions = [
            {"n_samples_max": 1000, "joint_size_max": 25,
             "variant": "shared", "block_size": 256},
            {"n_samples_max": None, "joint_size_max": None,
             "variant": "shared", "block_size": 512},
        ]
        cache.update("joint_hist_batched", axes=["n_samples", "joint_size"], regions=regions)
        # First region matches
        r1 = cache.lookup("joint_hist_batched", n_samples=500, joint_size=10)
        assert r1 == {"variant": "shared", "block_size": 256}
        # Above cap -> falls through to catch-all
        r2 = cache.lookup("joint_hist_batched", n_samples=10_000_000, joint_size=10_000)
        assert r2 == {"variant": "shared", "block_size": 512}

    def test_persistence_across_instances(self, tmp_cache_dir):
        c1 = ktc.KernelTuningCache()
        c1.update(
            "k1", axes=["n_samples"],
            regions=[{"n_samples_max": None, "variant": "v1", "block_size": 64}],
        )
        # New instance reads from disk
        c2 = ktc.KernelTuningCache()
        assert c2.has("k1") is True
        assert c2.lookup("k1", n_samples=42) == {"variant": "v1", "block_size": 64}

    def test_multiple_kernels_preserved(self, tmp_cache_dir):
        cache = ktc.KernelTuningCache()
        cache.update("kernel_a", axes=["n"],
                     regions=[{"n_max": None, "variant": "a"}])
        cache.update("kernel_b", axes=["n"],
                     regions=[{"n_max": None, "variant": "b"}])
        # Both must be present
        assert cache.has("kernel_a")
        assert cache.has("kernel_b")
        assert cache.lookup("kernel_a", n=1) == {"variant": "a"}
        assert cache.lookup("kernel_b", n=1) == {"variant": "b"}

    def test_schema_mismatch_treated_as_miss(self, tmp_cache_dir):
        # Write a v999 file directly
        path = os.path.join(tmp_cache_dir, ktc.hw_fingerprint() + ".json")
        with open(path, "w") as f:
            json.dump({"schema_version": 999, "hw_fingerprint": ktc.hw_fingerprint(),
                       "kernels": {"k": {"axes": ["n"], "regions": [{"variant": "x"}]}}}, f)
        cache = ktc.KernelTuningCache()
        # Mismatch -> miss
        assert cache.has("k") is False

    def test_fingerprint_mismatch_treated_as_miss(self, tmp_cache_dir):
        path = os.path.join(tmp_cache_dir, ktc.hw_fingerprint() + ".json")
        with open(path, "w") as f:
            json.dump({"schema_version": ktc.SCHEMA_VERSION,
                       "hw_fingerprint": "totally-different-host",
                       "kernels": {"k": {"axes": ["n"], "regions": [{"variant": "x"}]}}}, f)
        cache = ktc.KernelTuningCache()
        assert cache.has("k") is False


class TestRegionMatching:
    def test_unconstrained_axis_matches_any(self, tmp_cache_dir):
        cache = ktc.KernelTuningCache()
        cache.update("k", axes=["a", "b"], regions=[
            {"a_max": 100, "variant": "small"},   # b is unconstrained
            {"a_max": None, "b_max": None, "variant": "catch-all"},
        ])
        # a within cap, b ignored
        assert cache.lookup("k", a=50, b=999) == {"variant": "small"}
        # a above cap -> catch-all
        assert cache.lookup("k", a=200, b=999) == {"variant": "catch-all"}

    def test_first_match_wins(self, tmp_cache_dir):
        cache = ktc.KernelTuningCache()
        cache.update("k", axes=["a"], regions=[
            {"a_max": 100, "variant": "first"},
            {"a_max": 200, "variant": "second"},  # never reached at a=50
        ])
        assert cache.lookup("k", a=50) == {"variant": "first"}
        assert cache.lookup("k", a=150) == {"variant": "second"}


class TestHwFingerprint:
    def test_returns_non_empty_string(self, tmp_cache_dir):
        # tmp_cache_dir overrides PYUTILZ_KERNEL_CACHE_DIR so the disk
        # cache file lands in an empty per-test directory; without it
        # this test would read a stale fingerprint from any prior run.
        fp = ktc.hw_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) > 0
        assert fp.startswith("cpu_")

    def test_includes_no_gpu_marker_when_cuda_absent(self, tmp_cache_dir):
        # ``tmp_cache_dir`` gives us an empty disk cache so the mocked
        # ``gpu_capability_summary`` actually runs (the disk cache layer
        # would otherwise short-circuit before any GPU probe).
        try:
            with mock.patch.object(ktc, "gpu_capability_summary", return_value=None):
                ktc._gpu_summary_cached.cache_clear()  # GPU probe is per-device-cached now
                fp = ktc.hw_fingerprint()
            assert "no-gpu" in fp
        finally:
            # _gpu_summary_cached is an lru on the REAL function; calling it under
            # the mock caches the mocked ``None`` for the process lifetime, which
            # then makes _build_provenance() drop its ``gpu_summary`` block in
            # EVERY later test -> a real-GPU host's persisted tunings are wrongly
            # judged provenance-stale and dropped (observed: this leaked into
            # test_different_kernels_no_lost_update_D1, which then lost k_0). Clear
            # the lru on the way out so the next real probe repopulates it.
            ktc._gpu_summary_cached.cache_clear()

    def test_disk_cache_persists_across_cache_clear(self, tmp_cache_dir):
        # First call: compute + write to disk.
        fp1 = ktc.hw_fingerprint()
        ktc.hw_fingerprint.cache_clear()
        # Second call: the lru_cache is gone but the disk file isn't, so
        # the result must match without re-running the slow probes.
        # Patch both ``_cpu_model_slug`` and ``_gpu_slug_and_cc`` to
        # return sentinels; if the disk cache works, neither runs.
        with mock.patch.object(ktc, "_cpu_model_slug", return_value="UNCALLED-CPU"), \
             mock.patch.object(ktc, "_gpu_slug_and_cc", return_value=("UNCALLED-GPU", "9.9")):
            fp2 = ktc.hw_fingerprint()
        assert fp1 == fp2
        assert "UNCALLED" not in fp2

    def test_disk_cache_refresh_env_forces_recompute(self, tmp_cache_dir, monkeypatch):
        # Seed the on-disk cache with a sentinel value, then set
        # ``PYUTILZ_HW_FP_REFRESH=1`` and confirm the next call ignores it.
        import json as _json
        disk_file = os.path.join(tmp_cache_dir, ".hw_fingerprint.json")
        with open(disk_file, "w", encoding="utf-8") as f:
            _json.dump({"schema_version": 1, "fingerprint": "STALE-SENTINEL"}, f)
        ktc.hw_fingerprint.cache_clear()
        monkeypatch.setenv("PYUTILZ_HW_FP_REFRESH", "1")
        fp = ktc.hw_fingerprint()
        assert fp != "STALE-SENTINEL"
        assert fp.startswith("cpu_")

    def test_disk_cache_schema_mismatch_triggers_recompute(self, tmp_cache_dir):
        # An old (or hand-edited) cache file with a future schema_version
        # must be treated as a miss, NOT returned as the fingerprint.
        import json as _json
        disk_file = os.path.join(tmp_cache_dir, ".hw_fingerprint.json")
        with open(disk_file, "w", encoding="utf-8") as f:
            _json.dump({"schema_version": 999, "fingerprint": "STALE-SENTINEL"}, f)
        ktc.hw_fingerprint.cache_clear()
        fp = ktc.hw_fingerprint()
        assert fp != "STALE-SENTINEL"
        assert fp.startswith("cpu_")


class TestEnvOverride:
    def test_pyutilz_kernel_cache_dir_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
        d = ktc.cache_dir()
        assert d == str(tmp_path)
        p = ktc.cache_path()
        assert p.startswith(str(tmp_path))


class TestProvenance:
    """Cover ``_build_provenance`` + ``provenance_changed`` + the
    ``_load`` staleness short-circuit."""

    def test_build_provenance_returns_dict_with_versions(self):
        prov = ktc._build_provenance()
        # Always-present keys (None values OK for missing modules):
        assert "python_version" in prov
        assert "numpy_version" in prov

    def test_provenance_unchanged_returns_false(self):
        prov = {"cuda_driver_version": 12000, "cupy_version": "13.0.0",
                "gpu_summary": {"cc_major": 6, "cc_minor": 1, "name": "GTX"}}
        assert ktc.provenance_changed(prov, dict(prov)) is False

    def test_cuda_driver_bump_detected(self):
        old = {"cuda_driver_version": 12000, "cupy_version": "13.0.0"}
        new = {"cuda_driver_version": 12300, "cupy_version": "13.0.0"}
        assert ktc.provenance_changed(old, new) is True

    def test_cupy_version_bump_detected(self):
        old = {"cupy_version": "13.0.0"}
        new = {"cupy_version": "13.1.0"}
        assert ktc.provenance_changed(old, new) is True

    def test_gpu_cc_change_detected(self):
        old = {"gpu_summary": {"cc_major": 6, "cc_minor": 1, "name": "GTX 1050 Ti"}}
        new = {"gpu_summary": {"cc_major": 8, "cc_minor": 6, "name": "RTX 3070"}}
        assert ktc.provenance_changed(old, new) is True

    def test_python_minor_bump_detected(self):
        # v2: python MAJOR.MINOR is material (numba/cupy codegen can differ
        # across interpreter minors) -> a minor change re-tunes deterministically.
        old = {"python_version": "3.11", "cupy_version": "13.0.0"}
        new = {"python_version": "3.12", "cupy_version": "13.0.0"}
        assert ktc.provenance_changed(old, new) is True

    def test_python_patch_does_not_invalidate(self):
        # ...but a patch bump does NOT: _build_provenance records ONLY
        # major.minor, so 3.11.5 and 3.11.6 both become "3.11" and compare equal.
        import re
        pv = ktc._build_provenance()["python_version"]
        assert re.fullmatch(r"\d+\.\d+", pv), pv

    def test_numpy_major_bump_detected(self):
        # NumPy 1.x -> 2.x has ABI-level changes that can affect cupy interop;
        # treated as material so the cache invalidates.
        old = {"numpy_version": "1.26.4"}
        new = {"numpy_version": "2.0.0"}
        assert ktc.provenance_changed(old, new) is True

    def test_none_payload_treated_as_no_change(self):
        # Be conservative on missing data -- don't invalidate the cache.
        assert ktc.provenance_changed(None, {"x": 1}) is False
        assert ktc.provenance_changed({"x": 1}, None) is False

    def _read_kernel_record(self, cache, kernel_name):
        """Read the single immutable per-kernel record file (v3 storage)."""
        import glob as _glob
        import json
        kdir = ktc._kernel_dir(cache._path, kernel_name)
        files = _glob.glob(os.path.join(kdir, "*.json"))
        assert files, f"no immutable file written for {kernel_name}"
        # Newest wins; for these single-write tests there is exactly one.
        with open(sorted(files)[-1], "r", encoding="utf-8") as f:
            return json.load(f)

    def test_provenance_written_on_save(self, tmp_cache_dir):
        cache = ktc.KernelTuningCache()
        cache.update("k", axes=["n"], regions=[{"n_max": None, "variant": "x"}])
        # The immutable per-kernel record on disk must carry a "provenance" block.
        rec = self._read_kernel_record(cache, "k")
        assert "provenance" in rec
        assert isinstance(rec["provenance"], dict)
        assert rec["entry"]["regions"] == [{"n_max": None, "variant": "x"}]

    def test_stale_provenance_treated_as_miss(self, tmp_cache_dir):
        # Write a tuning with a deliberately-stale CUDA driver version into its
        # immutable record, then load with a mocked-current driver -> miss.
        import json
        from unittest import mock
        import glob as _glob
        cache = ktc.KernelTuningCache()
        cache.update("k", axes=["n"], regions=[{"n_max": None, "variant": "x"}])
        kdir = ktc._kernel_dir(cache._path, "k")
        rec_path = sorted(_glob.glob(os.path.join(kdir, "*.json")))[-1]
        with open(rec_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        rec.setdefault("provenance", {})["cuda_driver_version"] = 1000  # ancient
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        # Fresh instance (no in-memory state) + a mocked NEW driver version.
        fresh = ktc.KernelTuningCache()
        with mock.patch.object(
            ktc, "_build_provenance",
            return_value={"cuda_driver_version": 12300},
        ):
            assert fresh._load() is None  # stale -> miss (the only kernel is filtered out)


# ============================================================================
# v2: categorical/range axes, get_or_tune, code_version, equiv_tol, metadata
# (in-memory cache -> no disk / no hw_fingerprint / no cpuinfo)
# ============================================================================

class TestV2Matcher:
    def test_eq_min_max(self):
        c = ktc.KernelTuningCache(in_memory=True)
        c.update("k", axes=["n", "dtype", "ndim"], regions=[
            {"ndim_eq": 3, "backend": "numpy3d"},
            {"n_max": 1000, "n_min": 100, "dtype_eq": "float64", "backend": "small64"},
            {"backend": "catch"},
        ])
        assert c.lookup("k", n=50, dtype="float64", ndim=3) == {"backend": "numpy3d"}
        assert c.lookup("k", n=500, dtype="float64", ndim=2) == {"backend": "small64"}
        assert c.lookup("k", n=50, dtype="float64", ndim=2) == {"backend": "catch"}   # below n_min
        assert c.lookup("k", n=500, dtype="float32", ndim=2) == {"backend": "catch"}  # dtype mismatch

    def test_lookup_explain(self):
        c = ktc.KernelTuningCache(in_memory=True)
        c.update("k", axes=["n"], regions=[{"n_max": 100, "backend": "x"}])
        assert c.lookup_explain("k", n=50)["region_index"] == 0
        miss = c.lookup_explain("k", n=500)
        assert not miss["matched"] and "n=500" in miss["reason"]


class TestV2GetOrTune:
    def _fresh(self):
        ktc._TUNED_THIS_PROCESS.clear()
        return ktc.KernelTuningCache(in_memory=True)

    def test_miss_then_hit_one_sweep(self):
        c = self._fresh(); calls = {"n": 0}
        def tuner():
            calls["n"] += 1; return [{"backend": "numba"}]
        r1 = c.get_or_tune("g", dims={"n": 100}, tuner=tuner, axes=["n"], fallback={"backend": "FB"})
        r2 = c.get_or_tune("g", dims={"n": 100}, tuner=tuner, axes=["n"], fallback={"backend": "FB"})
        assert r1 == {"backend": "numba"} and r2 == {"backend": "numba"} and calls["n"] == 1

    def test_env_override(self, monkeypatch):
        c = self._fresh(); monkeypatch.setenv("MY_BK", "forced")
        assert c.get_or_tune("g", dims={"n": 1}, tuner=lambda: [{"x": 1}],
                             axes=["n"], fallback="FB", env_key="MY_BK") == "forced"

    def test_fallback_on_empty_and_failing_tuner(self):
        c = self._fresh()
        assert c.get_or_tune("a", dims={"n": 1}, tuner=lambda: [], axes=["n"], fallback="FB") == "FB"
        c2 = self._fresh()
        def boom():
            raise RuntimeError("x")
        assert c2.get_or_tune("b", dims={"n": 1}, tuner=boom, axes=["n"], fallback="FB") == "FB"

    def test_once_per_process_guard(self):
        c = self._fresh(); calls = {"n": 0}
        def boom():
            calls["n"] += 1; raise RuntimeError("x")
        c.get_or_tune("g", dims={"n": 1}, tuner=boom, axes=["n"], fallback="FB")
        c.get_or_tune("g", dims={"n": 2}, tuner=boom, axes=["n"], fallback="FB")
        assert calls["n"] == 1

    def test_code_version_invalidation(self):
        c = self._fresh()
        c.get_or_tune("g", dims={"n": 1}, tuner=lambda: [{"backend": "v1"}],
                      axes=["n"], fallback="FB", code_version="A")
        ktc._TUNED_THIS_PROCESS.clear()
        got = c.get_or_tune("g", dims={"n": 1}, tuner=lambda: [{"backend": "v2"}],
                            axes=["n"], fallback="FB", code_version="B")
        assert got == {"backend": "v2"}


class TestV2EquivGate:
    def test_divergent_region_rejected_not_masked(self):
        c = ktc.KernelTuningCache(in_memory=True)
        c.update("k", axes=["n"], regions=[
            {"n_max": 100, "backend": "good", "max_abs_diff": 1e-14},
            {"backend": "bad", "max_abs_diff": 1e-2},
        ], equiv_tol=1e-9)
        assert (c.lookup("k", n=50) or {}).get("backend") == "good"
        assert c.lookup("k", n=5000) is None  # divergent catch-all dropped, not substituted


class TestV2Metadata:
    def test_metadata_and_evict_no_autoevict(self):
        c = ktc.KernelTuningCache(in_memory=True)
        c.update("k", axes=["n"], regions=[{"backend": "x"}], code_version="CV", salt=3)
        m = c.get_metadata("k")
        assert m["code_version"] == "CV" and m["salt"] == 3 and m["n_regions"] == 1
        assert m["age_seconds"] is not None
        assert c.evict("k") is True and c.get_metadata("k") is None and c.evict("k") is False


class TestV2Disk:
    """v2 disk round-trip (cpuinfo mocked so hw_fingerprint can't hang on WMI)."""

    @pytest.fixture
    def fast_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(ktc, "_cpu_model_slug", lambda: "testcpu")
        monkeypatch.setattr(ktc, "_gpu_slug_and_cc", lambda: ("no-gpu", ""))
        ktc.hw_fingerprint.cache_clear()
        ktc._TUNED_THIS_PROCESS.clear()
        yield str(tmp_path)
        ktc.hw_fingerprint.cache_clear()

    def test_code_version_persists_and_invalidates_across_instances(self, fast_cache_dir):
        ktc.KernelTuningCache().update("k", axes=["n"],
                                       regions=[{"n_max": None, "backend": "v1"}], code_version="A")
        c2 = ktc.KernelTuningCache()
        assert c2.get_metadata("k")["code_version"] == "A"
        assert c2.lookup("k", n=1) == {"backend": "v1"}
        ktc._TUNED_THIS_PROCESS.clear()
        c3 = ktc.KernelTuningCache()
        got = c3.get_or_tune("k", dims={"n": 1}, tuner=lambda: [{"n_max": None, "backend": "v2"}],
                             axes=["n"], fallback="FB", code_version="B")
        assert got == {"backend": "v2"}

    def test_categorical_region_survives_disk_roundtrip(self, fast_cache_dir):
        ktc.KernelTuningCache().update("k", axes=["dtype"], regions=[
            {"dtype_eq": "float64", "backend": "f64"}, {"backend": "other"}])
        c2 = ktc.KernelTuningCache()
        assert c2.lookup("k", dtype="float64") == {"backend": "f64"}
        assert c2.lookup("k", dtype="int32") == {"backend": "other"}


class TestLoadOrCreate:
    """load_or_create() revives the 8 mlframe dispatch sites that consulted a
    non-existent classmethod (AttributeError was swallowed -> cache unused)."""

    def setup_method(self):
        # Reset the singleton between tests so each starts clean.
        ktc._DEFAULT_INSTANCE = None

    def teardown_method(self):
        ktc._DEFAULT_INSTANCE = None

    def test_returns_instance(self, tmp_cache_dir):
        c = ktc.KernelTuningCache.load_or_create()
        assert isinstance(c, ktc.KernelTuningCache)

    def test_is_singleton(self, tmp_cache_dir):
        a = ktc.KernelTuningCache.load_or_create()
        b = ktc.KernelTuningCache.load_or_create()
        assert a is b

    def test_empty_cache_lookup_returns_none(self, tmp_cache_dir):
        # The revived-site contract: empty cache -> lookup None -> caller falls
        # back to its hand-tuned default (no error).
        c = ktc.KernelTuningCache.load_or_create()
        assert c.lookup("rmse_partial_sum", n_samples=50000, n_cols=10) is None

    def test_singleton_thread_safe_init(self, tmp_cache_dir):
        import threading

        results = []
        barrier = threading.Barrier(8)

        def grab():
            barrier.wait()
            results.append(ktc.KernelTuningCache.load_or_create())

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All threads see the same singleton.
        assert len({id(r) for r in results}) == 1


class TestAsyncSweepHwBusyGate:
    """The async (fit-time) sweep must DEFER while the CPU OR GPU is busy: a contended
    sweep both taxes the caller's fit (~18% wall on a 100k MRMR fit) and records
    contended timings as this host's optimum. Verifies the CPU+GPU-backed gate."""

    def _reset(self):
        from pyutilz.performance.kernel_tuning import benchmark as bm
        bm._HW_BUSY_CACHE = None

    def _patch(self, monkeypatch, *, gpu_load=None, cpu_pct=None):
        """Patch GPUtil + psutil into the lazy imports the gate uses."""
        monkeypatch.delenv("PYUTILZ_KERNEL_SWEEP_HW_BUSY", raising=False)
        import builtins
        real_import = builtins.__import__
        gpu_mod = mock.MagicMock()
        gpu_mod.getGPUs.return_value = ([mock.Mock(load=gpu_load)] if gpu_load is not None else [])
        ps_mod = mock.MagicMock()
        ps_mod.cpu_percent.return_value = (cpu_pct if cpu_pct is not None else 0.0)

        def _imp(name, *a, **k):
            if name == "GPUtil":
                if gpu_load is None:
                    raise ImportError("no GPUtil")
                return gpu_mod
            if name == "psutil":
                if cpu_pct is None:
                    raise ImportError("no psutil")
                return ps_mod
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _imp)
        return gpu_mod, ps_mod

    def test_busy_gpu_defers(self, monkeypatch):
        self._reset()
        self._patch(monkeypatch, gpu_load=0.95, cpu_pct=1.0)
        assert ktc._async_sweep_hw_busy() is True

    def test_busy_cpu_defers(self, monkeypatch):
        self._reset()
        self._patch(monkeypatch, gpu_load=0.02, cpu_pct=80.0)
        assert ktc._async_sweep_hw_busy() is True

    def test_idle_hw_proceeds(self, monkeypatch):
        self._reset()
        self._patch(monkeypatch, gpu_load=0.02, cpu_pct=5.0)
        assert ktc._async_sweep_hw_busy() is False

    def test_no_deps_does_not_defer(self, monkeypatch):
        self._reset()
        self._patch(monkeypatch, gpu_load=None, cpu_pct=None)  # both imports raise
        assert ktc._async_sweep_hw_busy() is False

    def test_threshold_above_one_disables_gate(self, monkeypatch):
        self._reset()
        self._patch(monkeypatch, gpu_load=0.99, cpu_pct=99.0)
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_HW_BUSY", "2.0")
        assert ktc._async_sweep_hw_busy() is False

    def test_verdict_is_ttl_cached(self, monkeypatch):
        self._reset()
        gpu_mod, _ = self._patch(monkeypatch, gpu_load=0.95, cpu_pct=1.0)
        assert ktc._async_sweep_hw_busy() is True
        # Second call within the TTL must NOT re-poll GPUtil.
        assert ktc._async_sweep_hw_busy() is True
        assert gpu_mod.getGPUs.call_count == 1

    def test_start_delay_env(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_START_DELAY", "10")
        assert ktc._async_sweep_start_delay() == 10.0
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_START_DELAY", "0")
        assert ktc._async_sweep_start_delay() == 0.0
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_START_DELAY", "garbage")
        assert ktc._async_sweep_start_delay() == 10.0


class TestHardwareBusyHelpers:
    """The shared CPU/GPU busy helpers used by both the async-sweep deferral and
    the per-iteration sweep gate (``time_backend(hw_idle_gate=True)``)."""

    def _reset(self):
        from pyutilz.performance.kernel_tuning import benchmark as bm
        bm._HW_BUSY_CACHE = None

    def test_wait_for_idle_returns_true_when_idle(self, monkeypatch):
        from pyutilz.performance.kernel_tuning import benchmark as bm
        self._reset()
        monkeypatch.setattr(bm, "hardware_busy", lambda *a, **k: False)
        slept = []
        assert bm.wait_for_idle_hardware(sleep=slept.append) is True
        assert slept == []  # never had to wait

    def test_wait_for_idle_times_out_when_busy(self, monkeypatch):
        from pyutilz.performance.kernel_tuning import benchmark as bm
        self._reset()
        monkeypatch.setattr(bm, "hardware_busy", lambda *a, **k: True)
        clock = {"t": 0.0}

        def _timer():
            return clock["t"]

        def _sleep(s):
            clock["t"] += s

        # Always busy -> must time out (return False) and not loop forever.
        assert bm.wait_for_idle_hardware(poll=0.5, max_wait=2.0, sleep=_sleep, timer=_timer) is False
