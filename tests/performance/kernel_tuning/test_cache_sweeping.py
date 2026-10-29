"""Behaviour of ``pyutilz.performance.kernel_tuning.cache.cache_sweeping``: the sweep claim is a single winner per
``(kernel, code_version)``, the marker is published with its payload already in place, a live owner inside the budget
keeps its claim, and a dead owner, an over-budget owner, or a stale incomplete marker is stolen. Markers live under
``tmp_path``; the owning pid is this process or one that is certainly dead."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from pyutilz.core import jsonlib
from pyutilz.performance.kernel_tuning.cache import cache_base, cache_sweeping
from pyutilz.performance.kernel_tuning.cache.cache_sweeping import _CacheSweepClaimMixin

_DEAD_PID = 2**31 - 2


class _Recorder:
    def __init__(self):
        self.detected = []

    def concurrent_sweep_detected(self, kernel):
        self.detected.append(kernel)


class _Cache(_CacheSweepClaimMixin):
    def __init__(self, path, in_memory=False):
        self._path = path
        self._in_memory = in_memory


@pytest.fixture
def cache(tmp_path):
    return _Cache(str(tmp_path / "cache.json"))


def _write_marker(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(jsonlib.dumps(payload))


class TestMarkerPath:
    def test_the_marker_is_named_after_the_code_version_inside_the_kernel_dir(self, cache):
        marker = cache._marker_path("matmul", "v1")
        assert marker.endswith("v1.INPROGRESS")
        assert cache_base._kernel_dir(cache._path, "matmul") == os.path.dirname(marker)

    def test_a_missing_code_version_uses_the_sentinel(self, cache):
        assert os.path.basename(cache._marker_path("matmul", None)) == f"{cache_base._NO_CODE_VERSION}.INPROGRESS"


class TestClaimSweep:
    def test_an_in_memory_cache_always_owns_and_writes_no_marker(self, tmp_path):
        cache = _Cache(None, in_memory=True)
        with cache._claim_sweep("matmul", "v1", _Recorder()) as owns:
            assert owns is True
        assert not any(tmp_path.iterdir())

    def test_the_first_claim_owns_and_removes_the_marker_on_exit(self, cache):
        marker = cache._marker_path("matmul", "v1")
        with cache._claim_sweep("matmul", "v1", _Recorder()) as owns:
            assert owns is True
            assert os.path.exists(marker)
        assert not os.path.exists(marker)

    def test_a_second_claim_while_the_first_is_held_gives_up(self, cache):
        recorder = _Recorder()
        with cache._claim_sweep("matmul", "v1", recorder) as first:
            with cache._claim_sweep("matmul", "v1", recorder) as second:
                assert first is True and second is False
        assert recorder.detected == []

    def test_a_marker_is_published_with_this_process_pid_and_start_time(self, cache):
        before = time.time()
        with cache._claim_sweep("matmul", "v1", _Recorder()):
            with open(cache._marker_path("matmul", "v1"), encoding="utf-8") as f:
                info = jsonlib.loads(f.read())
        assert info["pid"] == os.getpid()
        assert before <= info["start_ts"] <= time.time()


class TestStealing:
    def test_a_marker_of_a_dead_owner_is_stolen_and_reported(self, cache):
        marker = cache._marker_path("matmul", "v1")
        _write_marker(marker, {"pid": _DEAD_PID, "start_ts": time.time()})
        recorder = _Recorder()
        with cache._claim_sweep("matmul", "v1", recorder) as owns:
            assert owns is True
        assert recorder.detected == ["matmul"]

    def test_a_live_owner_inside_the_budget_is_not_stolen(self, cache):
        marker = cache._marker_path("matmul", "v1")
        _write_marker(marker, {"pid": os.getpid(), "start_ts": time.time(), "host": cache_base.hw_fingerprint()})
        with cache._claim_sweep("matmul", "v1", _Recorder()) as owns:
            assert owns is False

    def test_a_live_owner_past_the_budget_is_stolen(self, cache, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "60")
        marker = cache._marker_path("matmul", "v1")
        _write_marker(marker, {"pid": os.getpid(), "start_ts": time.time() - 3600, "host": cache_base.hw_fingerprint()})
        with cache._claim_sweep("matmul", "v1", _Recorder()) as owns:
            assert owns is True

    def test_a_fresh_incomplete_marker_is_left_alone_for_its_writer(self, cache):
        marker = cache._marker_path("matmul", "v1")
        _write_marker(marker, {})
        with cache._claim_sweep("matmul", "v1", _Recorder()) as owns:
            assert owns is False

    def test_an_old_incomplete_marker_is_a_crash_and_is_stolen(self, cache, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "60")
        marker = cache._marker_path("matmul", "v1")
        _write_marker(marker, {})
        old = time.time() - 3600
        os.utime(marker, (old, old))
        with cache._claim_sweep("matmul", "v1", _Recorder()) as owns:
            assert owns is True


class TestDecideSteal:
    def test_a_marker_from_another_host_is_not_judged_dead_by_a_local_pid_probe(self, cache, tmp_path):
        marker = tmp_path / "other.INPROGRESS"
        _write_marker(str(marker), {"pid": _DEAD_PID, "start_ts": time.time(), "host": "some-other-machine"})
        should_steal, _pid, _age, owner_dead, _budget = cache._decide_steal(str(marker))
        assert owner_dead is False
        assert should_steal is False

    def test_a_zero_pid_with_a_fresh_file_is_a_peer_still_writing(self, cache, tmp_path):
        marker = tmp_path / "half.INPROGRESS"
        _write_marker(str(marker), {"pid": 0, "start_ts": time.time() - 3600})
        should_steal, pid, _age, _dead, _budget = cache._decide_steal(str(marker))
        assert pid == 0
        assert should_steal is False

    def test_a_missing_marker_is_stealable_because_its_age_is_unknown(self, cache, tmp_path):
        should_steal, pid, age, owner_dead, _budget = cache._decide_steal(str(tmp_path / "absent.INPROGRESS"))
        assert should_steal is True
        assert pid == 0
        assert age == float("inf")
        assert owner_dead is True


class TestIncompleteMarkerBoundaries:
    def test_a_tiny_positive_start_time_is_a_complete_marker(self, cache, tmp_path):
        marker = tmp_path / "tiny.INPROGRESS"
        _write_marker(str(marker), {"pid": os.getpid(), "start_ts": 0.05, "host": cache_base.hw_fingerprint()})
        should_steal, _pid, _age, _dead, _budget = cache._decide_steal(str(marker))
        assert should_steal is True

    def test_pid_one_is_a_complete_marker(self, cache, tmp_path):
        marker = tmp_path / "pid1.INPROGRESS"
        _write_marker(str(marker), {"pid": 1, "start_ts": time.time() - 3600, "host": cache_base.hw_fingerprint()})
        should_steal, pid, _age, _dead, _budget = cache._decide_steal(str(marker))
        assert pid == 1
        assert should_steal is True

    def test_a_zero_start_time_with_a_fresh_file_is_incomplete_not_stale(self, cache, tmp_path):
        marker = tmp_path / "zero.INPROGRESS"
        _write_marker(str(marker), {"pid": os.getpid(), "start_ts": 0.0, "host": cache_base.hw_fingerprint()})
        should_steal, _pid, _age, _dead, _budget = cache._decide_steal(str(marker))
        assert should_steal is False


class TestDecideStealExactBoundaries:
    def test_a_marker_without_a_start_time_is_incomplete_not_ancient(self, cache, tmp_path):
        marker = tmp_path / "nostart.INPROGRESS"
        _write_marker(str(marker), {"pid": os.getpid()})
        should_steal, _pid, _age, owner_dead, _budget = cache._decide_steal(str(marker))
        assert should_steal is False
        assert owner_dead is False

    def test_an_incomplete_marker_reports_its_owner_as_not_dead(self, cache, tmp_path):
        marker = tmp_path / "incomplete.INPROGRESS"
        _write_marker(str(marker), {})
        _steal, _pid, _age, owner_dead, _budget = cache._decide_steal(str(marker))
        assert owner_dead is False

    def test_an_age_exactly_at_the_budget_is_not_yet_stealable(self, cache, tmp_path, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "60")
        now = 1_000_000_000.0
        monkeypatch.setattr(cache_sweeping, "time", SimpleNamespace(time=lambda: now))
        marker = tmp_path / "edge.INPROGRESS"
        _write_marker(str(marker), {"pid": os.getpid(), "start_ts": now - 60.0, "host": cache_base.hw_fingerprint()})
        should_steal, _pid, age, _dead, _budget = cache._decide_steal(str(marker))
        assert age == pytest.approx(60.0)
        assert should_steal is False

    def test_a_just_past_budget_live_owner_is_stealable(self, cache, tmp_path, monkeypatch):
        monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "60")
        now = 1_000_000_000.0
        monkeypatch.setattr(cache_sweeping, "time", SimpleNamespace(time=lambda: now))
        marker = tmp_path / "past.INPROGRESS"
        _write_marker(str(marker), {"pid": os.getpid(), "start_ts": now - 60.5, "host": cache_base.hw_fingerprint()})
        should_steal, _pid, _age, _dead, _budget = cache._decide_steal(str(marker))
        assert should_steal is True
