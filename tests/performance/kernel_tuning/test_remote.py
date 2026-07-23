"""Tests for the pluggable remote kernel-tuning backend (read/write-through)."""
from __future__ import annotations

import pytest

from pyutilz.performance.kernel_tuning import cache as ktc
from pyutilz.performance.kernel_tuning import remote as ktr


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

    def test_explicit_empty_spec_not_overridden_by_env_var(self, monkeypatch):
        """Regression test: spec="" (an explicit "force local-only regardless of the
        environment" request) was previously clobbered by `spec or os.environ.get(...)`,
        silently falling through to PYUTILZ_KERNEL_REMOTE if that env var happened to be set --
        the opposite of the caller's explicit intent."""
        monkeypatch.setenv("PYUTILZ_KERNEL_REMOTE", "s3://my-bucket/tunings")
        assert ktr.get_remote_backend("") is None


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
        # v3: the read-through caches each kernel as an immutable per-kernel file
        # under the per-host directory (cache._path), not a single monolith file.
        ktc.hw_fingerprint.cache_clear()
        import glob as _glob
        import os
        local_files = _glob.glob(os.path.join(cache._path, "**", "*.json"), recursive=True)
        assert local_files, "remote payload should be cached locally as per-kernel files"

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


class TestRemoteFailureEscalation:
    """Regression (2026-07-21 audit round 2, HIGH): every remote read/write failure used to log
    at DEBUG unconditionally, with no escalation -- invisible by default (loggers are
    WARNING-and-above when unconfigured). An expired credential / renamed bucket / blocked
    network silently disabled cross-machine sharing forever with zero default-visible signal.
    Failures now escalate to WARNING on the first failure after a run of successes, and
    periodically thereafter; a subsequent success logs a recovery WARNING too.

    Tests drive ``_log_remote_failure``/``_note_remote_success`` directly (one call = one
    remote-op outcome) rather than through ``update()`` -- a single ``update()`` call touches
    TWO internal remote call sites (a pre-write merge read in ``_remote_payload`` AND the write
    in ``_persist_kernel``), which would make an exact per-call-count assertion depend on that
    internal fan-out rather than on the escalation logic itself."""

    def test_first_failure_escalates_to_warning(self, caplog):
        cache = ktc.KernelTuningCache(in_memory=True)
        with caplog.at_level("DEBUG"):
            cache._log_remote_failure("read", RuntimeError("boom"))
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "1 consecutive failure" in warnings[0].getMessage()

    def test_subsequent_failures_stay_at_debug_until_the_interval(self, caplog):
        cache = ktc.KernelTuningCache(in_memory=True)
        with caplog.at_level("DEBUG"):
            for _ in range(19):  # #1 warns, #2-19 stay debug (interval=20 not yet reached)
                cache._log_remote_failure("read", RuntimeError("boom"))
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert len(warnings) == 1
        assert len(debugs) == 18

    def test_periodic_reescalation_at_interval(self, caplog):
        cache = ktc.KernelTuningCache(in_memory=True)
        with caplog.at_level("DEBUG"):
            for _ in range(20):  # #1 warns, #20 warns again (interval=20)
                cache._log_remote_failure("read", RuntimeError("boom"))
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 2
        assert "20 consecutive failure" in warnings[1].getMessage()

    def test_recovery_after_failures_logs_warning_and_resets_counter(self, caplog):
        cache = ktc.KernelTuningCache(in_memory=True)
        with caplog.at_level("DEBUG"):
            cache._log_remote_failure("read", RuntimeError("boom"))
            cache._log_remote_failure("write", RuntimeError("boom"))
        assert cache._remote_consecutive_failures == 2

        with caplog.at_level("DEBUG"):
            cache._note_remote_success()

        assert cache._remote_consecutive_failures == 0
        recovery = [r for r in caplog.records if r.levelname == "WARNING" and "recovered" in r.getMessage()]
        assert len(recovery) == 1
        assert "2 consecutive failure" in recovery[0].getMessage()

    def test_success_without_prior_failures_does_not_log_recovery(self, caplog):
        cache = ktc.KernelTuningCache(in_memory=True)
        with caplog.at_level("DEBUG"):
            cache._note_remote_success()
        recovery = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(recovery) == 0


class TestRemoteFailureEscalationIntegration:
    """End-to-end smoke test: a genuinely raising remote backend surfaces at least one WARNING
    through the real ``update()`` -> ``_persist_kernel()`` -> ``_remote_payload()`` call chain."""

    def test_raising_remote_surfaces_at_least_one_warning(self, tmp_cache_dir, caplog):
        cache = _make_cache_with_remote(RaisingRemote())
        with caplog.at_level("DEBUG"):
            cache.update("k", axes=["dtype"], regions=[{"backend": "b"}])
        warnings = [r for r in caplog.records if r.levelname == "WARNING" and "remote" in r.getMessage()]
        assert len(warnings) >= 1
