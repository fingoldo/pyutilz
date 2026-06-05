"""Live benchmarking."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Callable, Optional
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


def _max_abs_diff(a, b) -> float:
    """Max abs elementwise difference between two array-likes (host or device).
    cupy arrays are pulled to host for the comparison; shape-mismatch -> inf."""
    try:
        a = np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)
        b = np.asarray(b.get() if hasattr(b, "get") else b, dtype=np.float64)
        if a.shape != b.shape:
            return float("inf")
        if a.size == 0:
            return 0.0
        return float(np.abs(a - b).max())
    except Exception:
        return float("inf")


def sweep_backend_crossover(
    variants: "dict[str, Callable]",
    sizes: "list[int]",
    make_inputs: "Callable[[int], tuple]",
    primary_axis: str,
    *,
    reference: "Optional[str]" = None,
    extra_region_keys: "Optional[dict]" = None,
    repeats: int = 15,
    equiv_atol: float = 1e-6,
    equiv_rtol: float = 1e-6,
    synchronize_gpu: bool = True,
    decision_key: str = "backend_choice",
    verbose: int = 0,
) -> list:
    """Benchmark backend ``variants`` across a primary-size grid and return
    ``kernel_tuning_cache`` regions (fastest EQUIVALENT backend per size band).

    This is the generic form of the per_member / recursion sweeps, for kernels
    whose dispatch turns on a single size axis (n_samples / n_cells / arr_size).

    Args:
        variants: ``{"numpy": fn, "numba": fn, "cupy": fn, ...}``. Each is called
            as ``fn(*make_inputs(size))``. Order is preserved for tie-breaking
            (earlier = preferred on equal time).
        sizes: ascending primary-axis values to benchmark.
        make_inputs: ``size -> args tuple`` fed to every variant (same inputs,
            so timings + outputs are comparable).
        primary_axis: region key; emitted as ``"<primary_axis>_max"`` bands.
        reference: variant whose output defines correctness; others must match
            within tol or they are disqualified at that size (SURFACES a
            divergent-but-faster variant instead of silently picking it).
            Defaults to the first variant key.
        extra_region_keys: fixed keys merged into every region (e.g. other dims
            held constant for this sweep).
        repeats: timed reps per variant per size (a warmup pass precedes timing).
        synchronize_gpu: sync the GPU before stopping each timer so async
            cupy/cuda kernels are timed at completion, not launch.

    Returns:
        Region dicts ``[{"<axis>_max": int|None, decision_key: name,
        "max_abs_diff": float, "wall_ms": {name: ms}}, ...]`` ascending, with a
        catch-all (``"<axis>_max": None``) carrying the largest size's winner.
        Consecutive equal-winner sizes are collapsed into one band.
    """
    if not variants:
        return []
    names = list(variants)
    ref = reference or names[0]
    extra = dict(extra_region_keys or {})

    per_size_winner: list[tuple] = []  # (size, winner_name, max_diff_of_winner)
    for size in tqdmu(sizes, desc=f"sweep {primary_axis}", leave=False):
        args = make_inputs(size)
        try:
            ref_out = variants[ref](*args)
            synchronize_gpu_if_available() if synchronize_gpu else None
        except Exception:
            ref_out = None
        best_name, best_ms, best_diff = None, float("inf"), 0.0
        for name in names:
            fn = variants[name]
            try:
                fn(*args)  # warmup (jit / cudagraph / alloc)
                if synchronize_gpu:
                    synchronize_gpu_if_available()
                out = fn(*args)
                if synchronize_gpu:
                    synchronize_gpu_if_available()
                diff = 0.0 if (name == ref or ref_out is None) else _max_abs_diff(ref_out, out)
                # equivalence gate: a faster-but-divergent variant is a bug, not a winner
                if name != ref and not (diff <= equiv_atol + equiv_rtol * float(np.abs(np.asarray(
                        ref_out.get() if hasattr(ref_out, "get") else ref_out, dtype=np.float64)).max() or 1.0)):
                    if verbose:
                        logger.info("sweep %s=%d: %s DIVERGES (maxdiff=%.2e) -> skip", primary_axis, size, name, diff)
                    continue
                t0 = timer()
                for _ in range(repeats):
                    fn(*args)
                if synchronize_gpu:
                    synchronize_gpu_if_available()
                ms = (timer() - t0) / repeats * 1e3
            except Exception as e:
                if verbose:
                    logger.info("sweep %s=%d: %s failed (%s) -> skip", primary_axis, size, name, e)
                continue
            if ms < best_ms:
                best_name, best_ms, best_diff = name, ms, diff
        if best_name is None:
            best_name, best_diff = ref, 0.0
        per_size_winner.append((size, best_name, best_diff))
        if verbose:
            logger.info("sweep %s=%d -> %s (%.3f ms)", primary_axis, size, best_name, best_ms)

    # Collapse consecutive equal-winner sizes into <axis>_max bands.
    regions: list = []
    i = 0
    while i < len(per_size_winner):
        j = i
        while j + 1 < len(per_size_winner) and per_size_winner[j + 1][1] == per_size_winner[i][1]:
            j += 1
        band_max = per_size_winner[j][0]
        is_last = j == len(per_size_winner) - 1
        worst_diff = max(d for _, _, d in per_size_winner[i:j + 1])
        region = {f"{primary_axis}_max": None if is_last else int(band_max), decision_key: per_size_winner[i][1]}
        if np.isfinite(worst_diff):
            region["max_abs_diff"] = float(worst_diff)
        region.update(extra)
        regions.append(region)
        i = j + 1
    return regions
