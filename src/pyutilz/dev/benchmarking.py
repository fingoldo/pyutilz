"""Live benchmarking."""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import sys
from typing import Any, Callable, Optional
import numpy as np
from timeit import default_timer as timer

from pyutilz.system.system import tqdmu

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def _cupy_has_context(cp: Any) -> bool:
    """True when this thread already has a current CUDA context (so a sync cannot be what creates one). May raise when
    there is no driver; the caller treats that as nothing to wait for."""
    # ``cudaGetDeviceCount`` initialises the driver WITHOUT creating a context; ``cuCtxGetCurrent`` before driver
    # initialisation segfaults (reproduced with cupy 13 on Windows), so the order matters.
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        return False
    return bool(cp.cuda.driver.ctxGetCurrent())


def _numba_cuda_initialised() -> bool:
    """True when numba.cuda's driver has already been initialised in this process (never initialises it). May raise;
    the caller treats that as nothing to wait for."""
    from numba.cuda.cudadrv import driver as _drv  # only reached when numba.cuda is already imported

    return bool(getattr(_drv.driver, "is_initialized", False))


def synchronize_gpu_if_available() -> None:
    """Block until ALL pending GPU work of this process completes: every cupy stream and every numba.cuda stream.

    GPU kernels launch ASYNCHRONOUSLY: a cupy/cuda call returns a device handle
    immediately and the kernel runs later. Timing such a call with a wall-clock
    timer therefore measures the LAUNCH, not the compute, and a fast-looking
    "win" can be pure launch overhead. (Real incident 2026-06-05: an unsynced
    measurement showed a cupy kernel beating numba 16x; with a per-call
    synchronize it actually LOST.) Call this immediately before stopping the
    timer so the measured interval includes the device compute.

    Device-wide, not a null-stream wait: ``cupy.cuda.Stream.null.synchronize()`` (the previous body) did not wait for
    work on other cupy streams or under ``CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1``, and nothing waited for numba.cuda
    at all, so those backends were timed at launch and the kernel-tuning cache stored launch times. Now:

    * cupy (only if already imported and this thread has a CUDA context): ``cupy.cuda.runtime.deviceSynchronize()``;
    * numba.cuda (only if already imported and its driver initialised): ``numba.cuda.synchronize()``.

    Neither library is imported and no CUDA context is created here, so on a CPU-only run this stays a cheap no-op.
    Forcing a host copy of the result (``cp.asnumpy``) is an equivalent barrier.
    """
    synced = False
    cp = sys.modules.get("cupy")
    if cp is not None:
        try:
            if _cupy_has_context(cp):
                cp.cuda.runtime.deviceSynchronize()
                synced = True
        except Exception as e:
            # Legitimate best-effort: no driver / no device means nothing to wait for, and a sync helper must never
            # fail the benchmark it is called from.
            logger.debug("synchronize_gpu_if_available: cupy sync skipped (%s)", e)  # nosec B110
    numba_cuda = sys.modules.get("numba.cuda")
    if numba_cuda is not None:
        try:
            if _numba_cuda_initialised():
                numba_cuda.synchronize()
                synced = True
        except Exception as e:
            logger.debug("synchronize_gpu_if_available: numba.cuda sync skipped (%s)", e)  # nosec B110
    if not synced:
        logger.debug("synchronize_gpu_if_available: no-op (no initialised cupy / numba.cuda context)")


def _preserve_axis_value(value):
    """Return an axis value as ``int`` when it is integral, otherwise unchanged.

    ``axes`` is documented as ``{dim: [values]}`` with no integrality requirement, so a blanket
    ``int()`` silently truncated fractional axes (density 0.25 and 0.5 both became 0) and made the
    emitted ``<dim>_max`` bounds indistinguishable to the kernel_tuning_cache matcher.
    """
    try:
        if float(value) == int(value):
            return int(value)
    except (TypeError, ValueError, OverflowError):
        return value
    logger.warning("benchmarking: axis value %r is not integral; keeping it as-is in the emitted region bound.", value)
    return value


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

    if n_reps < 1:
        # With no repetition the min-tracking sentinel below was returned AS A TIMING: every backend
        # came back at 1e20 s and a dispatcher persisted that with an arbitrary argsort order.
        raise ValueError(f"benchmark_algos_by_runtime: n_reps must be >= 1, got {n_reps}")

    durations: Any = []
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
            # Report the same min-over-reps estimator that is sorted and returned; the loop
            # variable `duration` holds only the LAST repetition and is unbound when n_reps == 0.
            mes_parts.append(f"{algo.__name__}: {min_duration:.3f} sec.")

    if verbose > 1:
        logger.info("Benchmark timings: %s", ", ".join(mes_parts))

    durations = np.array(durations)
    indices = np.argsort(durations)

    sorted_implementations = [implementations[idx] for idx in indices]
    sorted_durations = durations[indices].tolist()

    return sorted_implementations, sorted_durations


def _output_vector(x: Any) -> np.ndarray:
    """One flat float64 vector for a backend output: cupy arrays are pulled to host, and tuple / list / dict outputs
    (possibly ragged, e.g. a tuple of arrays with different shapes) are flattened part by part (dicts in sorted key
    order) and concatenated. Raises when a part cannot be read as numbers."""
    if isinstance(x, dict):
        parts = [x[k] for k in sorted(x, key=repr)]
        return np.concatenate([_output_vector(v) for v in parts]) if parts else np.zeros(0, dtype=np.float64)
    if isinstance(x, (tuple, list)):
        try:
            return np.asarray(x, dtype=np.float64).ravel()
        except (TypeError, ValueError):  # swallow-ok: ragged / nested -> flatten each part below
            pass
        return np.concatenate([_output_vector(v) for v in x]) if x else np.zeros(0, dtype=np.float64)
    if hasattr(x, "get") and callable(x.get) and not isinstance(x, np.ndarray):
        x = x.get()
    return np.asarray(x, dtype=np.float64).ravel()


def _output_scale(x: Any) -> float:
    """Largest finite ``|value|`` of an output (the rtol scale), 1.0 when there is none or it is 0 (all-zero or
    all-NaN reference). ``np.abs(...).max() or 1.0`` stayed NaN for a reference containing a NaN (NaN is truthy), which
    made every candidate fail the tolerance check."""
    v = _output_vector(x)
    finite = np.abs(v[np.isfinite(v)])
    if finite.size == 0:
        return 1.0
    m = float(finite.max())
    return m if m > 0.0 else 1.0


def _max_abs_diff(a: Any, b: Any) -> float:
    """Max abs elementwise difference between two outputs (host or device arrays, or tuple / list / dict of them).

    Non-finite values must sit at the same positions with the same value (NaN matches NaN, +inf matches +inf);
    a mismatch there, or a size mismatch, returns inf. The difference is taken over the finite positions only, so a NaN
    shared by both outputs no longer turns the result into NaN (which failed every tolerance check)."""
    try:
        va = _output_vector(a)
        vb = _output_vector(b)
        if va.shape != vb.shape:
            return float("inf")
        if va.size == 0:
            return 0.0
        fa = np.isfinite(va)
        fb = np.isfinite(vb)
        if not np.array_equal(fa, fb):
            return float("inf")
        nonfinite = ~fa
        if nonfinite.any():
            na, nb = va[nonfinite], vb[nonfinite]
            if not (np.array_equal(np.isnan(na), np.isnan(nb)) and np.array_equal(na[~np.isnan(na)], nb[~np.isnan(nb)])):
                return float("inf")
        if not fa.any():
            return 0.0
        return float(np.abs(va[fa] - vb[fa]).max())
    except Exception as e:
        logger.debug("_max_abs_diff: comparison failed (%s), treating as maximally different", e)
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
        equiv_atol: absolute tolerance for the reference-output equivalence check (see
            ``reference`` above).
        equiv_rtol: relative tolerance for the reference-output equivalence check.
        synchronize_gpu: sync the GPU before stopping each timer so async
            cupy/cuda kernels are timed at completion, not launch.
        decision_key: the dict key under which the winning variant's name is stored in
            each returned region (default ``"backend_choice"``).
        ranking: ``"robust"`` (default) interleaves candidates per rep + takes the
            per-candidate MIN over reps (contention-robust); ``"mean"`` is the legacy
            sequential per-candidate mean. See :func:`_rank_candidates`.
        verbose: 0 = silent; >0 increases logging detail of the sweep's progress.

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
            # Largest finite |ref| (1.0 for an all-zero / all-NaN reference). Inside this try: a reference output that
            # cannot be read as numbers skips the size instead of aborting the whole sweep.
            ref_scale = _output_scale(ref_out) if ref_out is not None else 1.0
        except Exception as exc:
            # Without a (numeric) reference output there is nothing to gate candidates against, so every
            # candidate would record a fabricated max_abs_diff of 0.0 and a divergent-but-faster
            # backend could be persisted as a tuned decision. Skip the size instead.
            logger.warning("sweep %s=%s: reference variant %r raised (%s) -> size skipped (no equivalence gate possible)", primary_axis, size, ref, exc)
            continue
        best_name, best_ms, best_diff = None, float("inf"), 0.0
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
                if name != ref and not (diff <= equiv_atol + equiv_rtol * ref_scale):
                    if verbose:
                        logger.info("sweep %s=%d: %s DIVERGES (maxdiff=%.2e) -> skip", primary_axis, size, name, diff)
                    continue
            except Exception as e:
                if verbose:
                    logger.info("sweep %s=%d: %s failed warmup (%s) -> skip", primary_axis, size, name, e)
                continue
            diffs[name] = diff
            survivors[name] = lambda _fn=fn, _a=args: _fn(*_a)
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
        worst_diff = max(d for _, _, d in per_size_winner[i : j + 1])
        # Axis values keep their own type: int() truncated a non-integer axis (density 0.25/0.5 both
        # became 0), so the kernel_tuning_cache matcher could not tell the measured cells apart and every
        # real value fell through to the catch-all entry.
        region = {f"{primary_axis}_max": None if is_last else _preserve_axis_value(band_max), decision_key: per_size_winner[i][1]}
        if np.isfinite(worst_diff):
            region["max_abs_diff"] = float(worst_diff)
        region.update(extra)
        regions.append(region)
        i = j + 1
    return regions


def _to_host(x: Any) -> Any:
    """Pull cupy arrays to host numpy (also inside a tuple / list / dict output); pass numpy / scalars through.
    ``dict.get`` must not be mistaken for cupy's ``.get()``."""
    if isinstance(x, dict):
        return {k: _to_host(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(_to_host(v) for v in x)
    return x.get() if hasattr(x, "get") and not isinstance(x, np.ndarray) else x


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
        reference: name of the variant treated as correctness ground truth; defaults to the
            first key in ``variants`` when omitted.
        residencies: subset of ``("host", "device")``.
        to_device: ``args -> args`` mover to VRAM (default ``cp.asarray`` ndarrays).
        repeats: timed reps per variant per cell (a warmup pass precedes timing).
        equiv_atol: absolute tolerance for the reference-output equivalence check (see
            ``reference`` above).
        equiv_rtol: relative tolerance for the reference-output equivalence check.
        synchronize_gpu: sync the GPU before stopping each timer so async
            cupy/cuda kernels are timed at completion, not launch.
        decision_key: the dict key under which the winning variant's name is stored in
            each returned region (default ``"backend_choice"``).
        ranking: how the per-cell timings are aggregated to pick the winner.
            ``"robust"`` (default) interleaves candidates within each rep and takes
            the per-candidate MIN over reps -- contention-robust, so a concurrent GPU
            process can't mis-rank the sweep (min approaches the uncontended cost
            since noise only adds time). ``"mean"`` is the legacy sequential
            per-candidate mean (kept for A/B; correct only on a quiet device). See
            :func:`_rank_candidates`.
        verbose: 0 = silent; >0 increases logging detail of the sweep's progress.

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
        """Move any numpy-array arguments to a CuPy device array, leaving other argument types unchanged."""
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
                ref_scale = _output_scale(ref_out) if ref_out is not None else 1.0  # see sweep_backend_crossover
            except Exception as exc:
                # Same reasoning as sweep_backend_crossover: no reference output means no
                # equivalence gate, and an ungated winner must never be emitted as tuned.
                logger.warning("grid %s res=%s: reference variant %r raised (%s) -> combination skipped", dims, res, ref, exc)
                continue
            best_name, best_ms, best_diff = None, float("inf"), 0.0
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
                    if name != ref and not (diff <= equiv_atol + equiv_rtol * ref_scale):
                        if verbose:
                            logger.info("grid %s res=%s: %s DIVERGES (%.2e) -> skip", dims, res, name, diff)
                        continue
                except Exception as e:
                    if verbose:
                        logger.info("grid %s res=%s: %s failed warmup (%s) -> skip", dims, res, name, e)
                    continue
                diffs[name] = diff
                survivors[name] = lambda _fn=fn, _a=args: _fn(*_a)
            # Pass 2: rank survivors under the chosen metric (robust=interleaved min over reps,
            # which is contention-robust; mean=legacy sequential per-candidate mean).
            timings = _rank_candidates(survivors, repeats=repeats, synchronize_gpu=synchronize_gpu, ranking=ranking)
            for name in names:  # iterate in declared order so ties prefer the earlier (e.g. reference) variant
                if name not in timings:
                    continue
                ms = timings[name]
                if ms < best_ms:
                    best_name, best_ms, best_diff = name, ms, diffs[name]
            region: dict = {f"{d}_max": _preserve_axis_value(dims[d]) for d in dim_names}
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
