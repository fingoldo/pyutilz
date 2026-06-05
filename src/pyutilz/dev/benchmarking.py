"""Live benchmarking."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any
import numpy as np
from timeit import default_timer as timer

from pyutilz.system.system import tqdmu

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def synchronize_gpu_if_available() -> None:
    """Block until pending GPU (cupy / numba.cuda) work on the default stream
    completes.

    GPU kernels launch ASYNCHRONOUSLY: a cupy/cuda call returns a device handle
    immediately and the kernel runs later. Timing such a call with a wall-clock
    timer therefore measures the LAUNCH, not the compute, and a fast-looking
    "win" can be pure launch overhead. (Real incident 2026-06-05: an unsynced
    measurement showed a cupy kernel beating numba 16x; with a per-call
    synchronize it actually LOST.) Call this immediately before stopping the
    timer so the measured interval includes the device compute.

    No-op (and cheap -- an empty-stream sync is microseconds) when cupy is not
    installed or no device work is pending. Forcing a host copy of the result
    (``cp.asnumpy``) is an equivalent barrier, which is why DRAM-round-trip
    timings are already honest.
    """
    try:
        import cupy as _cp

        _cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def benchmark_algos_by_runtime(
    implementations: list, algo_name: str = "", n_reps: int = 2, verbose: int = 0, synchronize_gpu: bool = True, **algo_kwargs
) -> tuple:
    """For a list of implementations on the same algorithm (accepting same inputs),
    repeats them n_reps times and sorts them by min runtime.
    n_reps>=2 means prewarm is possible (good for jit-compiled functions).

    ``synchronize_gpu`` (default True) forces a GPU sync after each call before
    the timer stops, so asynchronous cupy/cuda kernels are timed at COMPLETION,
    not launch -- see ``synchronize_gpu_if_available``. Leave on whenever any
    implementation may touch the GPU; the sync is a cheap no-op for CPU-only
    code and without cupy. Also benchmark DRAM-resident and VRAM-resident inputs
    separately when comparing a GPU backend: the optimal choice is
    residency-AND-HW-dependent (transfer cost vs on-device compute)."""

    durations = []
    if verbose > 1:
        mes_parts = []
        logger.info("Benchmarking %s implementations...", algo_name)

    for algo in tqdmu(implementations, desc=f"Comparing {algo_name} implementations", leave=False):
        min_duration = 1e20
        for _rep_id in range(n_reps):
            algo_start = timer()
            _ = algo(**algo_kwargs)
            if synchronize_gpu:
                # Block on any async GPU kernels so the timer captures compute,
                # not just launch (else a cupy kernel can post a phantom win).
                synchronize_gpu_if_available()
            duration = timer() - algo_start
            if duration < min_duration:
                min_duration = duration
        durations.append(min_duration)
        if verbose > 1:
            mes_parts.append(f"{algo.__name__}: {duration:.3f} sec.")

    if verbose > 1:
        logger.info("Benchmark timings: %s", ", ".join(mes_parts))

    durations = np.array(durations)
    indices = np.argsort(durations)

    sorted_implementations = [implementations[idx] for idx in indices]
    sorted_durations = durations[indices].tolist()

    return sorted_implementations, sorted_durations
