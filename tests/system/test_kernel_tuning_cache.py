"""Smoke tests for pyutilz.system.kernel_tuning_cache."""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from pyutilz.system import kernel_tuning_cache as ktc


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
        with mock.patch.object(ktc, "gpu_capability_summary", return_value=None):
            fp = ktc.hw_fingerprint()
        assert "no-gpu" in fp

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

    def test_python_patch_does_not_invalidate(self):
        # python_version is NOT in the material-keys list -> ignored.
        old = {"python_version": "3.11.5", "cupy_version": "13.0.0"}
        new = {"python_version": "3.11.6", "cupy_version": "13.0.0"}
        assert ktc.provenance_changed(old, new) is False

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

    def test_provenance_written_on_save(self, tmp_cache_dir):
        cache = ktc.KernelTuningCache()
        cache.update("k", axes=["n"], regions=[{"n_max": None, "variant": "x"}])
        # File on disk must contain a "provenance" block now.
        import json
        with open(cache._path, "r") as f:
            data = json.load(f)
        assert "provenance" in data
        assert isinstance(data["provenance"], dict)

    def test_stale_provenance_treated_as_miss(self, tmp_cache_dir):
        # Write a cache with deliberately-stale CUDA driver version, then
        # load with mocked-current driver -> should treat as miss.
        import json
        from unittest import mock
        cache = ktc.KernelTuningCache()
        cache.update("k", axes=["n"], regions=[{"n_max": None, "variant": "x"}])
        # Hand-edit the saved provenance to look stale.
        with open(cache._path, "r") as f:
            data = json.load(f)
        data["provenance"]["cuda_driver_version"] = 1000  # ancient
        with open(cache._path, "w") as f:
            json.dump(data, f)
        # Force a fresh KernelTuningCache instance (no in-memory state).
        fresh = ktc.KernelTuningCache()
        # Mock _build_provenance to claim a NEW driver version.
        with mock.patch.object(
            ktc, "_build_provenance",
            return_value={"cuda_driver_version": 12300},
        ):
            assert fresh._load() is None  # stale -> miss
