"""Concurrency tests for the v3 immutable-file kernel-tuning cache redesign.

These cover the defects the redesign was built to dissolve:

* D1 lost-update: N processes tuning DIFFERENT kernels concurrently -- every
  kernel's tuning must survive (the regression that the monolithic-JSON +
  snapshot-merge model could silently drop).
* singleton-without-blocking: N processes tuning the SAME kernel -- exactly one
  sweep runs and all readers see the same persisted result.
* stale-marker steal: a sweeper killed mid-INPROGRESS leaves a marker whose pid
  is dead / start_ts is old; a later sweeper STEALS it and completes (no wedge).
* lookup never blocks while a sweep is INPROGRESS.

The multi-process tests launch a minimal standalone worker via ``subprocess``
(NOT multiprocessing-spawn of pytest) so each worker imports only the cache
module -- cheap startup, no import-deadlock under disk contention.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from unittest import mock

import pytest

from pyutilz.performance.kernel_tuning import cache as ktc

_WORKER = os.path.join(os.path.dirname(__file__), "_concurrency_worker.py")
_PY = sys.executable


@pytest.fixture
def host_dir(tmp_path, monkeypatch):
    """A tmp cache dir + a pinned fake fingerprint (no HW probe stalls)."""
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ktc, "_cpu_model_slug", lambda: "testcpu")
    monkeypatch.setattr(ktc, "_gpu_slug_and_cc", lambda: ("no-gpu", ""))
    ktc.hw_fingerprint.cache_clear()
    ktc._TUNED_THIS_PROCESS.clear()
    yield str(tmp_path)
    ktc.hw_fingerprint.cache_clear()


def _launch(mode, cache_dir, out_json, kernel_name="shared_k", sleep_ms=150):
    env = dict(os.environ)
    env["PYUTILZ_KERNEL_CACHE_DIR"] = cache_dir
    return subprocess.Popen(
        [_PY, _WORKER, mode, cache_dir, out_json, kernel_name, str(sleep_ms)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _run_workers(procs, timeout=300):
    # 300s (was 120): each worker subprocess imports the cache module, which triggers the cpuinfo WMI probe + the
    # GPU-capability probe + numpy -- off a postgres-saturated disk that cold import alone can take a minute-plus.
    # The window we are testing (the O_EXCL sweep marker) opens in milliseconds; the timeout only guards a hung
    # worker, so it must comfortably exceed a worst-case cold import, not the concurrency window.
    deadline = time.time() + timeout
    for p in procs:
        remaining = max(1, deadline - time.time())
        try:
            out, err = p.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            raise AssertionError(f"worker timed out; stderr={err!r}")
        if p.returncode != 0:
            raise AssertionError(f"worker exited {p.returncode}; stderr={err!r}")


# ---------------------------------------------------------------------------
# Multi-process tests (the real cross-process claim)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(480)
def test_same_kernel_exactly_one_sweep_no_lost_update(host_dir, tmp_path):
    """(a) N processes tuning the SAME kernel -> exactly ONE sweep runs and
    every process reads the SAME persisted result (single-winner claim)."""
    n = 4
    outs = [str(tmp_path / f"same_{i}.json") for i in range(n)]
    procs = [_launch("same_kernel", host_dir, outs[i], sleep_ms=400) for i in range(n)]
    _run_workers(procs)

    results = [json.load(open(o, encoding="utf-8")) for o in outs]
    swept = [r for r in results if r["ran_sweep"]]
    assert len(swept) == 1, f"expected exactly one sweep, got {len(swept)}: {[r['pid'] for r in swept]}"
    # All workers must end up with the SAME persisted winner (the one sweep).
    winners = {json.dumps(r["persisted"], sort_keys=True) for r in results if r["persisted"]}
    assert len(winners) == 1, f"workers disagree on persisted winner: {winners}"
    assert json.loads(next(iter(winners)))["backend"] == "measured"


@pytest.mark.timeout(480)
def test_different_kernels_no_lost_update_D1(host_dir, tmp_path):
    """(b) D1 REGRESSION: N processes tuning DIFFERENT kernels concurrently --
    NO kernel's entry may be lost. The monolithic-JSON snapshot-merge model
    could silently drop a concurrently-written kernel; the immutable per-kernel
    files make that impossible."""
    n = 5
    kernels = [f"k_{i}" for i in range(n)]
    outs = [str(tmp_path / f"own_{i}.json") for i in range(n)]
    procs = [_launch("own_kernel", host_dir, outs[i], kernel_name=kernels[i], sleep_ms=300)
             for i in range(n)]
    _run_workers(procs)

    # Each worker swept its own kernel + persisted it.
    for i in range(n):
        r = json.load(open(outs[i], encoding="utf-8"))
        assert r["ran_sweep"], f"{kernels[i]} did not sweep"
        assert r["persisted"] and r["persisted"]["backend"] == "measured"

    # A FRESH cache in THIS process must now see ALL n kernels (none lost).
    ktc.hw_fingerprint.cache_clear()
    fresh = ktc.KernelTuningCache()
    for k in kernels:
        assert fresh.has(k), f"D1 lost update: kernel {k} missing from cache"
        assert fresh.lookup(k, n=1)["backend"] == "measured"


# ---------------------------------------------------------------------------
# In-process tests for the claim / steal / non-blocking primitives
# ---------------------------------------------------------------------------

def test_stale_marker_is_stolen_when_owner_pid_dead(host_dir):
    """(c) A sweeper killed mid-INPROGRESS leaves a marker with a DEAD pid; the
    next sweeper STEALS it and completes (no permanent wedge)."""
    cache = ktc.KernelTuningCache()
    # Plant a marker owned by a guaranteed-dead pid (recent start_ts so only the
    # dead-pid branch -- not the budget branch -- can trigger the steal).
    marker = cache._marker_path("wedged", "cv1")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"pid": 999999999, "start_ts": time.time(), "host": ktc.hw_fingerprint()}, f)

    swept = {"v": False}

    def tuner():
        swept["v"] = True
        return [{"backend": "stolen_and_done"}]

    got = cache.get_or_tune("wedged", dims={"n": 1}, tuner=tuner, axes=["n"],
                            fallback={"backend": "FB"}, code_version="cv1",
                            once_per_process=False, async_sweep=False)
    assert swept["v"], "stale-marker steal failed: sweep never ran"
    assert got["backend"] == "stolen_and_done"
    assert not os.path.exists(marker), "marker should be removed after a completed sweep"


def test_stale_marker_is_stolen_when_start_ts_expired(host_dir, monkeypatch):
    """(c') The owner pid may be alive (we don't know) but the start_ts is older
    than the max-sweep budget -> the marker is steal-able (self-healing)."""
    monkeypatch.setenv("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "1")
    cache = ktc.KernelTuningCache()
    marker = cache._marker_path("expired", "cv1")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    # Our OWN live pid, but a start_ts well past the 1s budget -> stealable.
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "start_ts": time.time() - 100, "host": ktc.hw_fingerprint()}, f)

    swept = {"v": False}

    def tuner():
        swept["v"] = True
        return [{"backend": "budget_steal"}]

    got = cache.get_or_tune("expired", dims={"n": 1}, tuner=tuner, axes=["n"],
                            fallback={"backend": "FB"}, code_version="cv1",
                            once_per_process=False, async_sweep=False)
    assert swept["v"]
    assert got["backend"] == "budget_steal"


def test_live_in_budget_marker_is_NOT_stolen(host_dir):
    """A marker owned by a LIVE in-budget process must NOT be stolen -- the
    would-be second sweeper gives up (returns the fallback) without sweeping."""
    cache = ktc.KernelTuningCache()
    marker = cache._marker_path("busy", "cv1")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    # Our own (alive) pid + a fresh start_ts -> a legitimate in-flight sweep.
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "start_ts": time.time(), "host": ktc.hw_fingerprint()}, f)

    swept = {"v": False}

    def tuner():
        swept["v"] = True
        return [{"backend": "should_not_run"}]

    got = cache.get_or_tune("busy", dims={"n": 1}, tuner=tuner, axes=["n"],
                            fallback={"backend": "FB"}, code_version="cv1",
                            once_per_process=False, async_sweep=False)
    assert not swept["v"], "must not duplicate a live, in-budget sweep"
    assert got["backend"] == "FB"
    assert os.path.exists(marker), "we must not remove a marker we don't own"


def test_lookup_never_blocks_during_inprogress_sweep(host_dir):
    """(d) ``lookup`` is a pure read; it must NOT block while a sweep marker is
    held (the redesign removes the read-side lock entirely)."""
    cache = ktc.KernelTuningCache()
    cache.update("kx", axes=["n"], regions=[{"n_max": None, "backend": "v"}], code_version="cv1")
    # Hold an INPROGRESS marker for the kernel as if a sweep were running.
    marker = cache._marker_path("kx", "cv1")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "start_ts": time.time(), "host": ktc.hw_fingerprint()}, f)
    try:
        ktc.hw_fingerprint.cache_clear()
        fresh = ktc.KernelTuningCache()
        t0 = time.perf_counter()
        for _ in range(50):
            assert fresh.lookup("kx", n=1) == {"backend": "v"}
        elapsed = time.perf_counter() - t0
        # 50 lookups while a marker is held must be near-instant (no lock wait).
        assert elapsed < 5.0, f"lookup appears to block during sweep: {elapsed:.2f}s"
    finally:
        os.remove(marker)


def test_concurrent_claim_threads_single_owner(host_dir):
    """Many threads racing the O_EXCL claim for one (kernel, code_version):
    exactly ONE wins ownership; the rest give up (in-process analog of the
    cross-process single-sweep guarantee)."""
    cache = ktc.KernelTuningCache()
    owners = []
    barrier = threading.Barrier(8)

    def race():
        barrier.wait()
        with cache._claim_sweep("racek", "cv1", ktc._DEFAULT_HOOKS) as owns:
            if owns:
                owners.append(threading.get_ident())
                time.sleep(0.05)  # hold the claim so others see EEXIST

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(owners) == 1, f"exactly one thread must own the claim, got {len(owners)}"
