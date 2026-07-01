"""Realistic backend timing for kernel-tuning sweeps.

The naive ``for _ in range(n): t0=...; fn(X); dt=...`` loop most sweeps hand-roll systematically
MIS-RANKS GPU backends, because it grants two luxuries a production caller never gets:

1. **Warm-buffer reuse.** Reusing the SAME input array across iterations lets the cupy memory pool
   serve a hot, already-resident block -- hiding the per-call ``cudaMalloc`` + H2D upload a real
   caller pays for every FRESH input (e.g. a newly-engineered feature column). Measured gap on a
   GTX 1050 Ti: a batched plug-in-MI GPU kernel timed ~36ms under reuse but cost ~746ms/call in the
   real feature-engineering pipeline at n=100k -- a 20x under-count that flipped the dispatch to the
   wrong (GPU) backend.

2. **Solo execution.** Timing one call at a time gives the kernel the whole device, hiding the
   serialization a real thread/worker pool inflicts when many callers hit ONE GPU at once.

:func:`time_backend` measures under BOTH realistic conditions -- fresh inputs per call (via a
``make_inputs`` factory) and a configurable ``concurrency`` -- with an untimed warmup to absorb
JIT / cupy compile. It returns the median per-call wall time in milliseconds so a sweep can pick the
backend that actually wins in production, not the one that wins a warm solo microbenchmark.

Note the residual limit, by construction: an isolated per-kernel benchmark still cannot reproduce
*whole-pipeline* effects (other kernels interleaved on the same device, an outer thread pool's
scheduling, allocator fragmentation across a long run). When the end-to-end measurement disagrees
with even a realistic micro-measurement, trust the end-to-end one.
"""
from __future__ import annotations

import os
import statistics
import threading
import time
from typing import Callable, Optional, Sequence


# ---------------------------------------------------------------------------
# Hardware-busy gate (shared by the async-sweep deferral and the per-iteration
# sweep gate). A sweep that runs while the CPU *or* GPU it needs is loaded both
# contends with the real workload AND records contended timings as this host's
# "optimum" -- so we only ever benchmark on genuinely idle hardware.
# ---------------------------------------------------------------------------

# Default load fraction (0..1) above which a device counts as busy. A kernel
# sweep is short and bursty, so 0.40 lets light background activity through
# while still deferring under a real concurrent fit. Env-overridable; > 1 disables.
DEFAULT_HW_BUSY_THRESHOLD = 0.40

_HW_BUSY_CACHE: "Optional[tuple[float, float, bool]]" = None  # (sampled_at, threshold, busy)
_HW_BUSY_LOCK = threading.Lock()
_HW_BUSY_TTL = 1.0


def _busy_threshold(threshold: Optional[float]) -> float:
    if threshold is not None:
        return float(threshold)
    try:
        return float(os.environ.get("PYUTILZ_KERNEL_SWEEP_HW_BUSY", DEFAULT_HW_BUSY_THRESHOLD))
    except ValueError:
        return DEFAULT_HW_BUSY_THRESHOLD


def hardware_busy(threshold: Optional[float] = None, *, check_cpu: bool = True,
                  check_gpu: bool = True, timer: Callable[[], float] = time.perf_counter) -> bool:
    """True iff the CPU OR any GPU is loaded above ``threshold`` (fraction 0..1).

    CPU load via ``psutil``, GPU load via ``GPUtil``. A missing dependency or an
    absent device for a kind simply does not trip that kind -- we never block on
    hardware we cannot measure (so a no-psutil / no-GPU host behaves as before).
    ``threshold`` defaults to ``PYUTILZ_KERNEL_SWEEP_HW_BUSY`` (else 0.40); a
    value > 1 disables the gate. The verdict is TTL-cached process-wide (1s) so a
    burst of checks (many kernels spawning sweeps at once) shares one poll."""
    thr = _busy_threshold(threshold)
    if thr > 1.0:
        return False  # gate disabled
    global _HW_BUSY_CACHE
    now = timer()
    cached = _HW_BUSY_CACHE
    if cached is not None and cached[1] == thr and (now - cached[0]) < _HW_BUSY_TTL:
        return cached[2]
    with _HW_BUSY_LOCK:
        cached = _HW_BUSY_CACHE
        if cached is not None and cached[1] == thr and (timer() - cached[0]) < _HW_BUSY_TTL:
            return cached[2]
        busy = False
        if check_gpu:
            try:
                import GPUtil  # optional dep -> lazy

                busy = any(g.load > thr for g in GPUtil.getGPUs())
            except (ImportError, OSError, RuntimeError, ValueError):
                pass  # no GPUtil / no GPU / poll failed -> cannot tell, don't trip
        if not busy and check_cpu:
            try:
                import psutil  # optional dep -> lazy

                # interval=0.1 -> a real sample (cpu_percent(None) returns 0.0 on first call);
                # off the hot path (sweep thread), so a 100ms poll is fine.
                busy = (psutil.cpu_percent(interval=0.1) / 100.0) > thr
            except (ImportError, OSError, RuntimeError, ValueError):
                pass  # no psutil / poll failed -> cannot tell, don't trip
        _HW_BUSY_CACHE = (timer(), thr, busy)
        return busy


def wait_for_idle_hardware(threshold: Optional[float] = None, *, poll: float = 0.5,
                           max_wait: float = 30.0, check_cpu: bool = True,
                           check_gpu: bool = True, sleep: Callable[[float], None] = time.sleep,
                           timer: Callable[[], float] = time.perf_counter) -> bool:
    """Block until the CPU/GPU drop below ``threshold`` or ``max_wait`` seconds elapse.

    Returns True if the hardware became idle, False if it timed out still busy.
    Used to hold off the NEXT sweep iteration while the device is in use."""
    deadline = timer() + max(0.0, float(max_wait))
    while hardware_busy(threshold, check_cpu=check_cpu, check_gpu=check_gpu, timer=timer):
        if timer() >= deadline:
            return False
        sleep(float(poll))
    return True


def time_backend(
    fn: Callable,
    make_inputs: Callable[[], Sequence],
    *,
    concurrency: int = 1,
    n_iters: int = 2,
    warmup: int = 2,
    fresh_inputs_per_call: bool = True,
    hw_idle_gate: bool = False,
    hw_idle_threshold: Optional[float] = None,
    hw_idle_poll: float = 0.5,
    hw_idle_max_wait: float = 30.0,
    timer: Callable[[], float] = time.perf_counter,
) -> float:
    """Median per-call wall time (ms) of ``fn(*make_inputs())`` under realistic conditions.

    Parameters
    ----------
    fn
        The backend callable under test; invoked as ``fn(*args)`` where ``args`` is the tuple
        returned by ``make_inputs()``.
    make_inputs
        Zero-arg factory returning the argument tuple for ``fn``. Called afresh per timed call when
        ``fresh_inputs_per_call`` (the default) so each call allocates NEW inputs -- this is what
        exposes the per-call alloc / H2D-transfer cost a warm reused buffer would hide. Inputs are
        pre-built OUTSIDE the timed region, so the factory's own (host-side) allocation is not timed;
        only ``fn`` is.
    concurrency
        Number of threads that hammer ``fn`` simultaneously, each with its OWN fresh inputs -- models
        a worker pool contending on one shared device. ``1`` (default) is the legacy solo path.
    n_iters
        Timed calls per thread (>=1). Default 2 (was 5): the median of 2 is enough to pick a backend in a
        one-time per-host tuning sweep, and halving the timed calls roughly halves the (multi-minute) grid
        sweep wall -- callers that want a more stable median can pass a higher value explicitly.
    warmup
        Untimed calls (on throwaway fresh inputs) before timing, to absorb JIT / cupy-compile cost.
    fresh_inputs_per_call
        When ``True`` (default) every timed call gets new inputs from ``make_inputs``; when ``False``
        a single input set is built once and reused (the legacy warm-buffer behaviour -- kept only
        for explicit A/B of the reuse bias).
    timer
        Monotonic timer; overridable for tests.

    Returns
    -------
    float
        Median per-call time across all threads x iters, in milliseconds. ``inf`` if nothing ran.
    """
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")
    concurrency = max(1, int(concurrency))

    def _prebuild(count: int) -> list:
        if fresh_inputs_per_call:
            return [tuple(make_inputs()) for _ in range(count)]
        shared = tuple(make_inputs())
        return [shared] * count

    for _ in range(max(0, int(warmup))):
        fn(*make_inputs())

    def _run(inputs_list: list, out: list) -> None:
        local = []
        for args in inputs_list:
            # Don't START the next iteration while the device we need is busy: a contended
            # sample is a wrong measurement. Wait (bounded) for the hardware to free up first.
            if hw_idle_gate:
                wait_for_idle_hardware(hw_idle_threshold, poll=hw_idle_poll,
                                       max_wait=hw_idle_max_wait, timer=timer)
            t0 = timer()
            fn(*args)
            local.append(timer() - t0)
        out.extend(local)

    if concurrency == 1:
        samples: list = []
        _run(_prebuild(n_iters), samples)
        return statistics.median(samples) * 1000.0 if samples else float("inf")

    # Pre-build every thread's inputs (host-side) before starting, so the timed region is purely
    # ``fn`` -- including, for a GPU backend, its own H2D upload of the fresh buffer.
    per_thread_inputs = [_prebuild(n_iters) for _ in range(concurrency)]
    results: list[list] = [[] for _ in range(concurrency)]
    threads = [
        threading.Thread(target=_run, args=(per_thread_inputs[i], results[i]))
        for i in range(concurrency)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flat = [s for r in results for s in r]
    return statistics.median(flat) * 1000.0 if flat else float("inf")


def benchmark_backends(
    backends: dict,
    make_inputs: Callable[[], Sequence],
    **kwargs,
) -> dict:
    """Time several competing backends under the same realistic conditions.

    ``backends`` maps name -> callable. Returns ``{name: median_ms}`` (see :func:`time_backend` for
    the timing semantics and kwargs). The caller picks ``min(result, key=result.get)`` -- the
    backend that wins under fresh-input, contended timing, not a warm solo loop.
    """
    return {name: time_backend(fn, make_inputs, **kwargs) for name, fn in backends.items()}
