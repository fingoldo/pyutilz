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
    def test_returns_non_empty_string(self):
        ktc.hw_fingerprint.cache_clear()
        fp = ktc.hw_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) > 0
        assert fp.startswith("cpu_")

    def test_includes_no_gpu_marker_when_cuda_absent(self):
        ktc.hw_fingerprint.cache_clear()
        with mock.patch.object(ktc, "gpu_capability_summary", return_value=None):
            fp = ktc.hw_fingerprint()
        assert "no-gpu" in fp
        ktc.hw_fingerprint.cache_clear()


class TestEnvOverride:
    def test_pyutilz_kernel_cache_dir_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
        d = ktc.cache_dir()
        assert d == str(tmp_path)
        p = ktc.cache_path()
        assert p.startswith(str(tmp_path))
