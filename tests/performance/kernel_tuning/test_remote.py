"""Tests for the pluggable remote kernel-tuning backend (read/write-through)."""
from __future__ import annotations

import pytest

from pyutilz.system import kernel_tuning_cache as ktc
from pyutilz.system import kernel_tuning_remote as ktr


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    ktc.hw_fingerprint.cache_clear()
    yield str(tmp_path)
    ktc.hw_fingerprint.cache_clear()


class FakeRemote(ktr.RemoteBackend):
    """In-memory backend: {fingerprint: payload}, with call counters."""

    def __init__(self, seed=None):
        self.store = dict(seed or {})
        self.reads = 0
        self.writes = 0

    def read(self, fingerprint):
        self.reads += 1
        return self.store.get(fingerprint)

    def write(self, fingerprint, payload):
        self.writes += 1
        self.store[fingerprint] = payload
        return True


class RaisingRemote(ktr.RemoteBackend):
    def read(self, fingerprint):
        raise RuntimeError("network down")

    def write(self, fingerprint, payload):
        raise RuntimeError("network down")


def _make_cache_with_remote(remote):
    cache = ktc.KernelTuningCache()
    cache._remote = remote  # inject (constructor reads env; tests inject directly)
    return cache


class TestGetRemoteBackend:
    def test_no_spec_returns_none(self, monkeypatch):
        monkeypatch.delenv("PYUTILZ_KERNEL_REMOTE", raising=False)
        assert ktr.get_remote_backend() is None

    def test_s3_spec_builds_backend_without_importing_boto3(self, monkeypatch):
        # Building the backend must NOT import boto3 (lazy until first read/write).
        import sys

        monkeypatch.setitem(sys.modules, "boto3", None)  # poison: import would fail
        b = ktr.get_remote_backend("s3://my-bucket/tunings")
        assert isinstance(b, ktr.S3Backend)
        assert b.bucket == "my-bucket"
        assert b.prefix == "tunings"

    def test_unknown_scheme_returns_none(self):
        assert ktr.get_remote_backend("ftp://nope") is None

    def test_s3_no_bucket_returns_none(self):
        assert ktr.get_remote_backend("s3://") is None


class TestReadThrough:
    def test_read_through_populates_local(self, tmp_cache_dir):
        # Seed the remote with a valid payload for this host's fingerprint.
        fp = ktc.hw_fingerprint()
        payload = {
            "schema_version": ktc.SCHEMA_VERSION,
            "hw_fingerprint": fp,
            "kernels": {"k": {"axes": ["dtype"], "regions": [{"backend": "remote_win"}]}},
        }
        remote = FakeRemote(seed={fp: payload})
        cache = _make_cache_with_remote(remote)

        # Local file absent -> read-through pulls from remote + caches locally.
        choice = cache.lookup("k", dtype="float64")
        assert choice == {"backend": "remote_win"}
        assert remote.reads >= 1

        # A fresh cache instance now reads the locally-cached copy (no 2nd remote read).
        ktc.hw_fingerprint.cache_clear()
        import os
        assert os.path.isfile(cache._path), "remote payload should be cached locally"

    def test_remote_miss_returns_none(self, tmp_cache_dir):
        remote = FakeRemote(seed={})  # empty remote
        cache = _make_cache_with_remote(remote)
        assert cache.lookup("k", dtype="float64") is None


class TestWriteThrough:
    def test_update_writes_through(self, tmp_cache_dir):
        remote = FakeRemote()
        cache = _make_cache_with_remote(remote)
        cache.update("k", axes=["dtype"], regions=[{"backend": "b"}])
        assert remote.writes >= 1
        fp = ktc.hw_fingerprint()
        assert fp in remote.store
        assert "k" in remote.store[fp]["kernels"]


class TestDegradesOnError:
    def test_raising_remote_does_not_break_lookup_or_update(self, tmp_cache_dir):
        cache = _make_cache_with_remote(RaisingRemote())
        # read raising -> lookup degrades to local miss (None), no exception.
        with pytest.raises(RuntimeError):
            # NOTE: our cache calls remote.read inside _load; a raising backend
            # would propagate unless the cache guards it. Verify the guard:
            cache._remote.read("x")  # direct call raises (sanity)
        # Through the cache path, _load must NOT propagate the remote error:
        cache2 = _make_cache_with_remote(RaisingRemote())
        assert cache2.lookup("k", dtype="float64") is None  # degrades cleanly
        cache2.update("k", axes=["dtype"], regions=[{"backend": "b"}])  # no raise
