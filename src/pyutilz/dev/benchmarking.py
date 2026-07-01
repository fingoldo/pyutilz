"""Live benchmarking."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Callable, Optional
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
    ranking: str = "robust",
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
        ranking: ``"robust"`` (default) interleaves candidates per rep + takes the
            per-candidate MIN over reps (contention-robust); ``"mean"`` is the legacy
            sequential per-candidate mean. See :func:`_rank_candidates`.

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
        ref_scale = float(np.abs(np.asarray(
            ref_out.get() if hasattr(ref_out, "get") else ref_out, dtype=np.float64)).max() or 1.0) if ref_out is not None else 1.0
        # Pass 1: warm up + equivalence-gate; survivors go into the timed rank.
        survivors: dict = {}
        diffs: dict = {}
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
                if name != ref and not (diff <= equiv_atol + equiv_rtol * (ref_scale or 1.0)):
                    if verbose:
                        logger.info("sweep %s=%d: %s DIVERGES (maxdiff=%.2e) -> skip", primary_axis, size, name, diff)
                    continue
            except Exception as e:
                if verbose:
                    logger.info("sweep %s=%d: %s failed warmup (%s) -> skip", primary_axis, size, name, e)
                continue
            diffs[name] = diff
            survivors[name] = (lambda _fn=fn, _a=args: _fn(*_a))
        # Pass 2: rank survivors (robust=interleaved min over reps; mean=legacy).
        timings = _rank_candidates(survivors, repeats=repeats, synchronize_gpu=synchronize_gpu, ranking=ranking)
        for name in names:  # declared order -> ties prefer the earlier (reference) variant
            if name not in timings:
                continue
            ms = timings[name]
            if ms < best_ms:
                best_name, best_ms, best_diff = name, ms, diffs[name]
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


def _to_host(x):
    """Pull a cupy array to host numpy; pass numpy / scalars through."""
    return x.get() if hasattr(x, "get") else x


def _rank_candidates(
    candidates: "dict[str, Callable]",
    *,
    repeats: int,
    synchronize_gpu: bool,
    ranking: str,
) -> "dict[str, float]":
    """Time each already-equivalence-vetted candidate and return {name: ms}.

    Two ranking modes:

    * ``"mean"`` (legacy) -- for each candidate run all ``repeats`` calls
      back-to-back and report the MEAN per-call ms. SEQUENTIAL per candidate: one
      candidate finishes all its reps before the next starts. This is fine on a
      QUIET device, but on a CONTENDED GPU (a concurrent process competing for the
      device) a candidate that happens to be measured during a contention SPIKE
      loses to one measured in a LULL -- the absolute timings interleave with the
      other process's kernels, so the sweep mis-ranks and can pin a SLOW config as
      "fastest". (Real incident: an MI-gate hist kernel proved 3.05x faster at 1024
      vs 128 threads in an isolated CUDA-event A/B, yet a contended mean-sweep
      picked 128 and the win never materialised -- ~1.28s left on the table.)

    * ``"robust"`` (default, contention-robust) -- for each rep, time ALL
      candidates back-to-back (INTERLEAVED), so within one rep every candidate sees
      the SAME contention weather. Then take, per candidate, the MIN over reps. min
      is the right estimator under contention because noise only ADDS time: the
      FASTEST observed call for a candidate approaches its true uncontended cost, so
      the candidate that is genuinely faster wins even while another process churns
      the GPU. On a quiet device min-of-interleaved-reps converges to the same pick
      as the legacy mean, so the ranking is unchanged where it was already correct.

    A per-candidate failure (exception) is recorded as ``inf`` (disqualified), never
    crashing the whole rank. Returns an empty dict if ``candidates`` is empty.
    """
    names = list(candidates)
    if not names:
        return {}
    reps = max(1, int(repeats))

    if ranking == "mean":
        out: dict[str, float] = {}
        for name in names:
            fn = candidates[name]
            try:
                t0 = timer()
                for _ in range(reps):
                    fn()
                if synchronize_gpu:
                    synchronize_gpu_if_available()
                out[name] = (timer() - t0) / reps * 1e3
            except Exception:
                out[name] = float("inf")
        return out

    # robust: interleave candidates within each rep, take per-candidate min over reps.
    best: dict[str, float] = {name: float("inf") for name in names}
    for _rep in range(reps):
        for name in names:
            fn = candidates[name]
            try:
                t0 = timer()
                fn()
                if synchronize_gpu:
                    synchronize_gpu_if_available()
                ms = (timer() - t0) * 1e3
            except Exception:
                ms = float("inf")
            if ms < best[name]:
                best[name] = ms
    return best


def sweep_backend_grid(
    variants: "dict[str, Callable]",
    axes: "dict[str, list]",
    make_inputs: "Callable[[dict], tuple]",
    *,
    reference: "Optional[str]" = None,
    residencies: "tuple" = ("host",),
    to_device: "Optional[Callable]" = None,
    repeats: int = 15,
    equiv_atol: float = 1e-6,
    equiv_rtol: float = 1e-6,
    synchronize_gpu: bool = True,
    decision_key: str = "backend_choice",
    ranking: str = "robust",
    verbose: int = 0,
) -> list:
    """Full-grid, residency-aware backend sweep -> kernel_tuning_cache regions.

    Benchmarks every Cartesian combination of ``axes`` (not a 1-D crossover --
    every cell is measured) and, when ``residencies=("host","device")``, both
    DRAM-resident and VRAM-resident inputs. Emits ONE region per
    ``(grid cell × residency)`` keyed on ``<dim>_max`` for every dim (plus
    ``location_eq`` when measuring residency) carrying the fastest EQUIVALENT
    backend. The multi-axis matcher then selects the smallest region bounding the
    live dims at dispatch, so the full N-D surface is honoured exactly.

    Residency model (real transfer cost, not assumed):
      * ``variants`` must accept inputs in EITHER memory and pay the matching
        transfer themselves -- a GPU variant ``cp.asarray``-es host input (H2D),
        a CPU variant ``cp.asnumpy``-es device input (D2H) -- so the timed
        interval includes the residency-dependent transfer.
      * For ``residency="device"`` the host inputs from ``make_inputs`` are moved
        to VRAM via ``to_device`` OUTSIDE the timed region (default: ``cp.asarray``
        each ndarray), so only compute + any per-variant transfer is timed.

    Args:
        variants: ``{name: fn}``; ``fn(*make_inputs(dims))``. First key (or
            ``reference``) defines correctness; divergent variants are dropped.
        axes: ``{dim: [values]}`` -- the full Cartesian grid to sweep.
        make_inputs: ``dims_dict -> args tuple`` of HOST (numpy) arrays.
        residencies: subset of ``("host", "device")``.
        to_device: ``args -> args`` mover to VRAM (default ``cp.asarray`` ndarrays).
        ranking: how the per-cell timings are aggregated to pick the winner.
            ``"robust"`` (default) interleaves candidates within each rep and takes
            the per-candidate MIN over reps -- contention-robust, so a concurrent GPU
            process can't mis-rank the sweep (min approaches the uncontended cost
            since noise only adds time). ``"mean"`` is the legacy sequential
            per-candidate mean (kept for A/B; correct only on a quiet device). See
            :func:`_rank_candidates`.

    Returns:
        Region dicts ``[{"<dim>_max": int, ..., "location_eq": "host"|"device",
        decision_key: name, "max_abs_diff": float}, ...]`` -- one per cell ×
        residency.
    """
    import itertools

    if not variants or not axes:
        return []
    names = list(variants)
    ref = reference or names[0]
    dim_names = list(axes)
    # Sort each axis ascending so (a) the matcher's first-match scan hits the
    # tightest enclosing band first and (b) the all-max corner is well-defined
    # for the catch-all below.
    axes = {d: sorted(axes[d]) for d in dim_names}
    _max_combo = tuple(axes[d][-1] for d in dim_names)

    def _default_to_device(args):
        import cupy as cp

        return tuple(cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args)

    mover = to_device or _default_to_device
    regions: list = []
    # Decision at the all-max corner per residency -> emitted as a catch-all
    # (all caps None) so inputs LARGER than the swept grid still resolve to the
    # largest-measured winner instead of falling through to None (the heuristic).
    catchall: dict = {}

    for combo in tqdmu(list(itertools.product(*(axes[d] for d in dim_names))), desc="grid sweep", leave=False):
        dims = dict(zip(dim_names, combo))
        host_args = make_inputs(dims)
        for res in residencies:
            try:
                args = mover(host_args) if res == "device" else host_args
            except Exception as e:
                if verbose:
                    logger.info("grid %s res=%s: to_device failed (%s) -> skip residency", dims, res, e)
                continue
            try:
                ref_out = _to_host(variants[ref](*args))
                if synchronize_gpu:
                    synchronize_gpu_if_available()
            except Exception:
                ref_out = None
            best_name, best_ms, best_diff = None, float("inf"), 0.0
            ref_scale = float(np.abs(np.asarray(ref_out, dtype=np.float64)).max()) if ref_out is not None else 1.0
            # Pass 1: warm up + equivalence-gate every variant. Survivors (those whose
            # output matches the reference within tol) go into the timed rank; a
            # divergent-but-faster variant is a bug, never a winner, so it is dropped
            # here BEFORE timing. Warmup absorbs jit / cudagraph / alloc / transfer-plan.
            survivors: dict = {}
            diffs: dict = {}
            for name in names:
                fn = variants[name]
                try:
                    fn(*args)  # warmup
                    if synchronize_gpu:
                        synchronize_gpu_if_available()
                    diff = 0.0 if (name == ref or ref_out is None) else _max_abs_diff(ref_out, fn(*args))
                    if name != ref and diff > equiv_atol + equiv_rtol * (ref_scale or 1.0):
                        if verbose:
                            logger.info("grid %s res=%s: %s DIVERGES (%.2e) -> skip", dims, res, name, diff)
                        continue
                except Exception as e:
                    if verbose:
                        logger.info("grid %s res=%s: %s failed warmup (%s) -> skip", dims, res, name, e)
                    continue
                diffs[name] = diff
                survivors[name] = (lambda _fn=fn, _a=args: _fn(*_a))
            # Pass 2: rank survivors under the chosen metric (robust=interleaved min over reps,
            # which is contention-robust; mean=legacy sequential per-candidate mean).
            timings = _rank_candidates(survivors, repeats=repeats, synchronize_gpu=synchronize_gpu, ranking=ranking)
            for name in names:  # iterate in declared order so ties prefer the earlier (e.g. reference) variant
                if name not in timings:
                    continue
                ms = timings[name]
                if ms < best_ms:
                    best_name, best_ms, best_diff = name, ms, diffs[name]
            region = {f"{d}_max": int(dims[d]) for d in dim_names}
            if len(residencies) > 1:
                region["location_eq"] = res
            region[decision_key] = best_name or ref
            if np.isfinite(best_diff):
                region["max_abs_diff"] = float(best_diff)
            regions.append(region)
            if combo == _max_combo:
                catchall[res] = region[decision_key]
            if verbose:
                logger.info("grid %s res=%s -> %s (%.3f ms)", dims, res, best_name, best_ms)

    # Catch-all per residency: all caps None -> matches any input beyond the
    # grid, carrying the largest-cell winner. Appended LAST so the specific
    # per-cell bands match first (the matcher returns the first match in order).
    for res in residencies:
        if res not in catchall:
            continue
        region = {f"{d}_max": None for d in dim_names}
        if len(residencies) > 1:
            region["location_eq"] = res
        region[decision_key] = catchall[res]
        regions.append(region)
    return regions
