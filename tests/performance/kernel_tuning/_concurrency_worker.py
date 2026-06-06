"""Standalone worker for the kernel-tuning concurrency tests.

Run as a subprocess (NOT via multiprocessing-spawn-of-pytest) so it imports only
the cache module -- no pytest re-import -- which keeps startup cheap and avoids
the import-deadlock seen under disk saturation when many short python processes
re-import the heavy test stack.

Usage::

    python _concurrency_worker.py <mode> <cache_dir> <out_json> [kernel_name] [sleep_ms]

Modes:
    same_kernel   -- tune the SAME kernel ("shared_k"); records whether THIS
                     worker actually ran the sweep + the region it reads back.
    own_kernel    -- tune a worker-UNIQUE kernel (kernel_name arg); the D1
                     lost-update regression test (every worker's kernel must
                     survive).

The tuner sleeps ``sleep_ms`` to widen the concurrency window, and stamps the
region with the worker pid so we can tell which sweep won.
"""
import json
import os
import sys
import time

# The cache module sets up its own hw_fingerprint via env; we force a stable
# fake fingerprint so every worker agrees on the host dir without probing HW.
os.environ.setdefault("PYUTILZ_HW_FP_REFRESH", "0")


def _load_cache_module():
    from pyutilz.performance.kernel_tuning import cache as ktc
    return ktc


def main():
    mode = sys.argv[1]
    cache_dir = sys.argv[2]
    out_json = sys.argv[3]
    kernel_name = sys.argv[4] if len(sys.argv) > 4 else "shared_k"
    sleep_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 150

    os.environ["PYUTILZ_KERNEL_CACHE_DIR"] = cache_dir
    ktc = _load_cache_module()
    # Pin a deterministic fingerprint so all workers share one host directory
    # without each running the (slow, possibly-stalling) cpuinfo/GPU probe.
    ktc.hw_fingerprint.cache_clear()
    import unittest.mock as mock
    patcher_cpu = mock.patch.object(ktc, "_cpu_model_slug", lambda: "testcpu")
    patcher_gpu = mock.patch.object(ktc, "_gpu_slug_and_cc", lambda: ("no-gpu", ""))
    patcher_cpu.start()
    patcher_gpu.start()

    ran_sweep = {"v": False}

    def tuner():
        ran_sweep["v"] = True
        time.sleep(sleep_ms / 1000.0)
        return [{"backend": "measured", "winner_pid": os.getpid()}]

    if mode == "same_kernel":
        k = "shared_k"
    else:  # own_kernel
        k = kernel_name

    cache = ktc.KernelTuningCache()
    result = cache.get_or_tune(
        k, dims={"n": 1}, tuner=tuner, axes=["n"],
        fallback={"backend": "FB"}, code_version="cv1",
        once_per_process=False, async_sweep=False,
    )
    # Re-read from a fresh instance to confirm the persisted winner.
    fresh = ktc.KernelTuningCache()
    persisted = fresh.lookup(k, n=1)

    out = {
        "pid": os.getpid(),
        "mode": mode,
        "kernel": k,
        "ran_sweep": ran_sweep["v"],
        "result": result,
        "persisted": persisted,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
