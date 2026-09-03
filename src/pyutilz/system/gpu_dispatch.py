"""GPU dispatch helpers: device selection, shared-memory budgeting, and
CPU/GPU routing built on top of the existing pyutilz GPU primitives.

This module is **safe to import on CPU-only hosts**: every function either
returns a sensible default or ``None``. The optional dependencies
(``numba.cuda``, ``cupy``, ``GPUtil``) are imported lazily inside the call
sites that need them and absences are degraded gracefully.

The intent is to give downstream callers a single place to ask:

* "Which GPU should I use?"      -> :func:`select_best_gpu`
* "How big can my smem array be?" -> :func:`get_shared_mem_budget_per_block`
* "How wide should my block be?"  -> :func:`optimal_threads_per_block`
* "Do I have headroom?"           -> :func:`cuda_memory_guard`
* "Tell me everything"            -> :func:`gpu_capability_summary`
* "CPU or GPU for this workload?" -> :func:`dispatch_cpu_vs_gpu`

The static-shared-memory budget table is sourced from the NVIDIA CUDA C
Programming Guide, Table 21 ("Technical Specifications per Compute
Capability"). The opt-in column is the higher dynamic-smem ceiling
reachable via ``cudaFuncSetAttribute(...,
cudaFuncAttributeMaxDynamicSharedMemorySize, ...)``; callers must actually
set that attribute on their kernel to use the larger budget.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Optional

from pyutilz.core.pythonlib import is_cuda_available
from pyutilz.system.system import (
    get_gpu_cuda_capabilities,
    get_gpuutil_gpu_info,
)

logger = logging.getLogger(__name__)

WARP_SIZE = 32

# ---------------------------------------------------------------------------
# Per-CC static shared-memory budget table (bytes). Per-BLOCK (not per-SM).
# Sorted by (major, minor). For each entry: (default, opt_in_max).
#
# "default" is the static shared-memory ceiling every kernel gets without
# any opt-in. On all hardware shipped to date this is 48 KB (49152 bytes).
#
# "opt_in_max" is the larger per-block ceiling reachable via
# ``cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
# requested_bytes)``. The opt-in path only ships from Volta (cc 7.0) onward;
# earlier architectures had only the static 48 KB regardless of per-SM
# capacity. On cc >= 8.0 the runtime reserves 1 KB per block so the
# per-block opt-in is (per-SM - 1024), e.g. A100 is 164 KB per-SM but
# 163 KB = 166912 bytes per-block.
#
# Source: NVIDIA CUDA C Programming Guide, Appendix H "Compute Capabilities",
# Table 21 "Maximum amount of shared memory per thread block" column.
# ---------------------------------------------------------------------------
CC_SHARED_MEM_BUDGET: dict[tuple[int, int], tuple[int, int]] = {
    # cc          default     opt_in_max
    (3, 0): (49152, 49152),  # Kepler: 48 KB per block, no opt-in
    (3, 2): (49152, 49152),
    (3, 5): (49152, 49152),
    (3, 7): (49152, 49152),  # 112 KB is per-SM, NOT per-block
    (5, 0): (49152, 49152),  # Maxwell: 48 KB per block, no opt-in
    (5, 2): (49152, 49152),  # 96 KB per-SM, per-block stays at 48 KB
    (5, 3): (49152, 49152),
    (6, 0): (49152, 49152),  # Pascal: 48 KB per block, no opt-in
    (6, 1): (49152, 49152),  # 96 KB per-SM, per-block stays at 48 KB
    (6, 2): (49152, 49152),
    (7, 0): (49152, 98304),  # Volta V100: 96 KB per-block opt-in
    (7, 2): (49152, 98304),  # Xavier
    (7, 5): (49152, 65536),  # Turing: 64 KB per-block opt-in
    (8, 0): (49152, 166912),  # A100: 163 KB per-block (164 KB per-SM - 1 KB)
    (8, 6): (49152, 101376),  # Ampere consumer: 99 KB per-block (100 - 1)
    (8, 7): (49152, 166912),  # Orin: 163 KB per-block (same as A100)
    (8, 9): (49152, 101376),  # Ada Lovelace: 99 KB per-block
    (9, 0): (49152, 232448),  # Hopper: 227 KB per-block (228 - 1)
}

_SAFE_DEFAULT_SMEM = 49152  # 48 KB fallback for unknown / forward-compat


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _select_best_gpu_cached(strategy: str, pid: int) -> Optional[int]:
    """Inner cached worker. ``pid`` is part of the key so a fork resets it."""
    if not is_cuda_available():
        return None

    gpus = get_gpuutil_gpu_info(attrs="id,memoryFree,memoryTotal,load,name,uuid")
    if not gpus:
        logger.debug("select_best_gpu: GPUtil returned no devices")
        return None

    def _cc_tuple(dev_id: int) -> tuple[int, int]:
        """Returns the (major, minor) CUDA compute capability of `dev_id`, or (0, 0) if unavailable."""
        caps = get_gpu_cuda_capabilities(device_id=dev_id) or {}
        return (
            int(caps.get("COMPUTE_CAPABILITY_MAJOR", 0)),
            int(caps.get("COMPUTE_CAPABILITY_MINOR", 0)),
        )

    if strategy == "vram":
        best = max(gpus, key=lambda g: g.get("memoryFree", 0.0))
    elif strategy == "idle":
        best = min(gpus, key=lambda g: g.get("load", 100.0))
    elif strategy == "compute":
        best = max(gpus, key=lambda g: _cc_tuple(int(g["id"])))
    elif strategy == "auto":
        def _score(g: dict) -> float:
            """Scores a GPU info dict as free VRAM (bytes) times compute capability (major.minor)."""
            cc_major, cc_minor = _cc_tuple(int(g["id"]))
            cc = cc_major + cc_minor / 10.0
            return float(g.get("memoryFree", 0.0)) * cc
        best = max(gpus, key=_score)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}. " "Expected one of: 'auto', 'vram', 'compute', 'idle'.")

    return int(best["id"])


def select_best_gpu(strategy: str = "auto") -> Optional[int]:
    """Return device id of the best available GPU, or ``None`` on CPU-only hosts.

    Args:
        strategy: One of:

            * ``"auto"``    -- score = ``memoryFree * compute_capability``
              (balance VRAM headroom against raw FLOPS).
            * ``"vram"``    -- highest free VRAM.
            * ``"compute"`` -- highest ``(major, minor)`` compute capability.
            * ``"idle"``    -- lowest current load%.

    Returns:
        Integer device id, or ``None`` if no CUDA-capable device is visible.

    The result is cached per ``(strategy, pid)`` to avoid repeated nvidia-smi
    shell-outs. Call :func:`reset_cache` to force a re-probe.
    """
    return _select_best_gpu_cached(strategy, os.getpid())


def reset_cache() -> None:
    """Clear every memoized hardware probe in this module.

    Covers :func:`select_best_gpu`, the static per-device capability cache behind
    :func:`gpu_capability_summary` / :func:`occupancy_aware_block_size`, and
    ``pyutilz.core.pythonlib.is_cuda_available``. This is the single reset seam tests use when they
    mock CUDA availability or the GPUtil / driver probes."""
    _select_best_gpu_cached.cache_clear()
    _static_gpu_caps.cache_clear()
    is_cuda_available.cache_clear()


# ---------------------------------------------------------------------------
# Shared-memory budget
# ---------------------------------------------------------------------------

def get_shared_mem_budget_per_block(
    cc_major: int,
    cc_minor: int,
    allow_opt_in: bool = False,
) -> int:
    """Return safe per-block static shared-memory budget in bytes.

    Args:
        cc_major: CUDA compute capability major (e.g. 7 for Volta).
        cc_minor: CUDA compute capability minor (e.g. 5 for Turing).
        allow_opt_in: If True, return the higher dynamic-smem value reachable
            via ``cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)``.
            The caller is responsible for actually setting that attribute.

    Returns:
        Budget in bytes. Falls back to 48 KB for unknown CCs.
    """
    key = (int(cc_major), int(cc_minor))
    entry = CC_SHARED_MEM_BUDGET.get(key)
    if entry is None:
        # Forward-compat: match by major only (highest minor we know).
        same_major = [k for k in CC_SHARED_MEM_BUDGET if k[0] == key[0]]
        if same_major:
            entry = CC_SHARED_MEM_BUDGET[max(same_major)]
        else:
            logger.debug(
                "get_shared_mem_budget_per_block: unknown cc %s, "
                "falling back to %d bytes", key, _SAFE_DEFAULT_SMEM,
            )
            return _SAFE_DEFAULT_SMEM

    default_b, opt_in_b = entry
    return opt_in_b if allow_opt_in else default_b


# ---------------------------------------------------------------------------
# Threads-per-block heuristic
# ---------------------------------------------------------------------------

def optimal_threads_per_block(
    cc_major: int,
    cc_minor: int,
    *,
    max_required: Optional[int] = None,
    multiple_of_warp: bool = True,
) -> int:
    """Heuristic ``threads_per_block`` for a generic kernel.

    Defaults:
        * cc <= 6.x -> 128 (broadly-good occupancy on Pascal and earlier)
        * cc >= 7.x -> 256 (Volta+ have more registers/SM and prefer wider blocks)

    Args:
        cc_major: CUDA compute capability major.
        cc_minor: CUDA compute capability minor (accepted for API symmetry; unused).
        max_required: Optional clamp; if the kernel only has N items of work
            per block, picking more threads than that just wastes occupancy.
        multiple_of_warp: Round the result up to a multiple of WARP_SIZE (32).

    Returns:
        Positive integer threads-per-block.
    """
    del cc_minor  # accepted for symmetry; unused in current heuristic
    base = 256 if int(cc_major) >= 7 else 128

    if max_required is not None and max_required > 0:
        base = min(base, int(max_required))

    if multiple_of_warp:
        # Round up to nearest warp, but never zero.
        base = max(WARP_SIZE, ((base + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE)

    return int(base)


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

def _free_bytes_via_cupy(device_id: Optional[int]) -> Optional[int]:
    """Returns free GPU memory in bytes for `device_id` (or the current device if None) via cupy, or None if cupy is unavailable or the query fails transiently.

    Re-raises on an invalid-device CUDA error (status 101).
    """
    try:
        import cupy as cp
        from cupy.cuda.runtime import CUDARuntimeError
    except ImportError:
        return None
    try:
        if device_id is not None:
            with cp.cuda.Device(int(device_id)):
                free, _total = cp.cuda.runtime.memGetInfo()
        else:
            free, _total = cp.cuda.runtime.memGetInfo()
        return int(free)
    except CUDARuntimeError as e:
        # Loud-fail on invalid device id; silently degrade only on transient
        # / lookup errors. ``cudaErrorInvalidDevice == 101``.
        if getattr(e, "status", None) == 101:
            raise
        logger.debug("cupy memGetInfo runtime error: %s", e)
        return None
    except Exception as e:
        logger.debug("cupy memGetInfo failed: %s", e)
        return None


def _free_bytes_via_gputil(device_id: Optional[int]) -> Optional[int]:
    """Returns free GPU memory in bytes for `device_id` (or the first listed device if None) via GPUtil, or None if unavailable."""
    gpus = get_gpuutil_gpu_info(attrs="id,memoryFree,memoryTotal")
    if not gpus:
        return None
    if device_id is None:
        gpu = gpus[0]
    else:
        gpu = None
        for g in gpus:
            if int(g["id"]) == int(device_id):
                gpu = g
                break
        if gpu is None:
            return None
    # get_gpuutil_gpu_info divides by 1024 -> the value is in GB.
    free_gb = float(gpu.get("memoryFree", 0.0))
    return int(free_gb * 1024**3)


@contextmanager
def cuda_memory_guard(
    required_bytes: int,
    device_id: Optional[int] = None,
    headroom_factor: float = 1.2,
) -> Iterator[None]:
    """Context manager that asserts enough free VRAM before a GPU allocation.

    Args:
        required_bytes: Expected peak allocation in bytes.
        device_id: Target device id (``None`` -> current/default device).
        headroom_factor: Multiplier applied to ``required_bytes`` to leave a
            safety margin (default 1.2 = 20% headroom).

    Raises:
        MemoryError: If we can probe free VRAM and the threshold is not met.
        RuntimeError: If no CUDA backend is available at all.

    On exit, if CuPy is importable, calls
    ``cupy.get_default_memory_pool().free_all_blocks()`` so the pool returns
    memory to the driver rather than holding it for the next pool allocation.
    """
    if required_bytes < 0:
        raise ValueError(f"required_bytes must be >= 0, got {required_bytes}")

    if not is_cuda_available():
        raise RuntimeError("cuda_memory_guard called but no CUDA device is available")

    threshold = int(required_bytes * float(headroom_factor))
    free = _free_bytes_via_cupy(device_id)
    source = "cupy"
    if free is None:
        free = _free_bytes_via_gputil(device_id)
        source = "GPUtil"
    if free is None:
        logger.warning("cuda_memory_guard: cannot determine free VRAM (neither cupy nor " "GPUtil available); proceeding without check")
    else:
        if free < threshold:
            raise MemoryError(
                f"Not enough free VRAM on device {device_id}: "
                f"{free / 1024**2:.1f} MiB free, "
                f"{threshold / 1024**2:.1f} MiB required "
                f"({required_bytes / 1024**2:.1f} MiB * {headroom_factor}x headroom) "
                f"[probe={source}]"
            )

    try:
        yield
    finally:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass
        except Exception as e:
            logger.debug("cuda_memory_guard: cupy free_all_blocks failed: %s", e)


# ---------------------------------------------------------------------------
# Capability summary
# ---------------------------------------------------------------------------

# Driver attribute codes for limits numba's own `cudadrv.enums` does not define. It has no
# `CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`, so `Device.__getattr__` raises for it - but the
# driver entry point numba already binds answers perfectly well when called with the code directly
# (`query_cuda_device_attribute` below). Codes are from the CUDA driver API's own `CUdevice_attribute` enum.
CU_DEVICE_ATTRIBUTE_EXTRA: dict[str, int] = {
    "MAX_BLOCKS_PER_MULTIPROCESSOR": 106,
    "MAX_PERSISTING_L2_CACHE_SIZE": 108,
    "MAX_ACCESS_POLICY_WINDOW_SIZE": 109,
    "RESERVED_SHARED_MEMORY_PER_BLOCK": 111,
    "MEMORY_POOLS_SUPPORTED": 115,
    "GPU_DIRECT_RDMA_SUPPORTED": 116,
}


def query_cuda_device_attribute(name_or_code: "str | int", device_id: int = 0) -> Optional[int]:
    """One CUDA device attribute straight from the driver, by `CUdevice_attribute` name or numeric code.

    Exists because numba's `cudadrv.enums` covers only part of that enum, and the attributes it omits
    include `MAX_BLOCKS_PER_MULTIPROCESSOR` - an occupancy limit with no substitute. The usual workaround is
    to pull in cupy for it or hardcode a per-architecture table; neither is necessary, since numba already
    binds `cuDeviceGetAttribute` and it takes the raw code.

    Handles both of numba's driver backends: the newer `cuda.bindings` path (where the call returns the
    value) and the classic ctypes path (where it writes through a pointer).

    Args:
        name_or_code: A `CUdevice_attribute` suffix (e.g. ``"MAX_BLOCKS_PER_MULTIPROCESSOR"``, with or
            without the ``CU_DEVICE_ATTRIBUTE_`` prefix), or the numeric code itself.
        device_id: CUDA device id.

    Returns:
        The attribute value, or ``None`` on a CPU-only host or when the driver refuses the query.
    """
    try:
        from numba import cuda
        from numba.cuda.cudadrv import driver as _drv
    except ImportError:
        return None

    code: "int | None"
    if isinstance(name_or_code, int):
        code = name_or_code
    else:
        key = str(name_or_code).replace("CU_DEVICE_ATTRIBUTE_", "")
        code = CU_DEVICE_ATTRIBUTE_EXTRA.get(key)
        if code is None:
            try:
                from numba.cuda.cudadrv import enums

                code = int(getattr(enums, f"CU_DEVICE_ATTRIBUTE_{key}"))
            except (ImportError, AttributeError, TypeError, ValueError):
                return None
    try:
        cuda.select_device(device_id)
        device = cuda.get_current_device()
        # Read the real module flag rather than `getattr(..., False)`, which would silently take the ctypes
        # branch on a numba that renamed it - and the ctypes branch on the nv-binding backend passes a
        # pointer where a code belongs. A numba too old to define it at all predates the flag entirely,
        # which is exactly the ctypes case, so that is what the fallback means here.
        try:
            use_nv_binding = bool(_drv.USE_NV_BINDING)
        except AttributeError:
            use_nv_binding = False
        if use_nv_binding:
            # No `cuda.bindings` import needed: the code is passed as a plain int and numba's own binding
            # layer converts it. Importing it here would add a transitive dependency for nothing.
            return int(_drv.driver.cuDeviceGetAttribute(code, device.id))
        from ctypes import byref, c_int

        result = c_int()
        _drv.driver.cuDeviceGetAttribute(byref(result), code, device.id)
        return int(result.value)
    except Exception:
        logger.debug("query_cuda_device_attribute: driver refused attribute %s", name_or_code, exc_info=True)
        return None


# Maximum RESIDENT BLOCKS per SM, by compute capability - a FALLBACK only, for a driver that refuses the
# live query above. Values are from the CUDA C Programming Guide's own occupancy table.
CC_MAX_BLOCKS_PER_SM: dict[tuple[int, int], int] = {
    (3, 0): 16, (3, 2): 16, (3, 5): 16, (3, 7): 16,
    (5, 0): 32, (5, 2): 32, (5, 3): 32,
    (6, 0): 32, (6, 1): 32, (6, 2): 32,
    (7, 0): 32, (7, 2): 32, (7, 5): 16,
    (8, 0): 32, (8, 6): 16, (8, 7): 16, (8, 9): 24,
    (9, 0): 32, (10, 0): 32, (12, 0): 32,
}


def occupancy_aware_block_size(
    bytes_per_thread: int,
    caps: Optional[dict] = None,
    device_id: int = 0,
    min_threads: int = WARP_SIZE,
    max_threads: Optional[int] = None,
    power_of_two: bool = True,
) -> tuple[int, int]:
    """Largest warp-multiple block size that maximises RESIDENT THREADS per SM, and its shared-memory bytes.

    For a kernel whose shared memory scales with its block width - one accumulator slot per thread, the
    common reduction shape - block size and occupancy trade against each other, and the trade depends
    entirely on the device: how much shared memory an SM has, how many threads it can hold, and how many
    blocks. Picking a block width therefore cannot be done with a constant. The usual stand-in, "aim for N
    resident blocks", encodes one device's answer as if it were every device's.

    This computes it. For each candidate width it derives how many blocks the SM can actually hold - the
    strictest of the shared-memory limit, the threads-per-SM limit and the hardware blocks-per-SM cap - and
    keeps the width giving the most resident threads, breaking ties toward the WIDER block (fewer blocks
    doing the same work means fewer block-level reductions and less launch bookkeeping).

    Args:
        bytes_per_thread: Shared memory the kernel needs per thread. 0 means shared memory does not bind.
        caps: `gpu_capability_summary` output, if already queried; fetched here when omitted.
        device_id: CUDA device id, used only when ``caps`` is not supplied.
        min_threads: Floor on the returned width (default one warp).
        max_threads: Ceiling; defaults to the device's own `max_threads_per_block`.
        power_of_two: Restrict the answer to powers of two (the DEFAULT, because the classic tree
            reduction - ``for (s = blockDim.x / 2; s > 0; s >>= 1)`` - silently drops elements at any other
            width, and that is the overwhelmingly common shape for a kernel whose shared memory scales per
            thread, i.e. exactly the kernels that call this. Pass False only for a kernel whose reduction
            handles an odd tail, which buys the warp multiples in between.

    Returns:
        ``(threads_per_block, shared_bytes_per_block)``. Falls back to ``(min_threads, ...)`` on a CPU-only
        host or when the device reports no usable limits, so a caller never has to special-case that.
    """
    if caps is None:
        # Static caps ONLY: a block size is a function of the hardware's limits, never of how much
        # VRAM happens to be free right now. Going through ``gpu_capability_summary`` here used to
        # spawn an ``nvidia-smi`` subprocess (measured 63.5 ms) per sizing call, on a helper whose
        # whole job is to be cheap enough to call per kernel launch.
        summary = _static_gpu_caps(device_id=device_id)
        caps = dict(summary) if summary is not None else {}

    def _limit(key: str, fallback: int) -> int:
        """A stated limit, treating only MISSING/None as absent - a stated 0 means the device reports none.

        `caps.get(key) or fallback` would be shorter and wrong: it also replaces a genuine 0, which is the
        value a device or a synthetic test dict uses to say "this limit is unknown to me", and silently
        substituting a plausible number there is how an occupancy calculation starts reporting a width the
        hardware cannot hold.
        """
        value = caps.get(key)
        return fallback if value is None else int(value)

    warp = _limit("warp_size", WARP_SIZE)
    if warp <= 0:  # a device reporting no warp size cannot size a block; the architectural constant is safe
        warp = WARP_SIZE
    shared_per_sm = _limit("max_shared_mem_per_sm", 0)
    shared_per_block = _limit("max_shared_mem_per_block", 0)
    threads_per_sm = _limit("max_threads_per_sm", 0)
    blocks_per_sm = _limit("max_blocks_per_sm", 0)
    reserved_per_block = _limit("reserved_shared_mem_per_block", 0)
    if max_threads is not None:
        ceiling = int(max_threads)
    else:
        stated_ceiling = _limit("max_threads_per_block", 0)
        ceiling = stated_ceiling if stated_ceiling > 0 else min_threads
    # shared_per_sm is only REQUIRED when shared memory actually binds: with bytes_per_thread == 0
    # (this code's own definition of "does not bind") a device reporting max_shared_mem_per_sm: 0
    # used to bail to a single warp, silently crippling occupancy the other limits fully determine.
    if (shared_per_sm <= 0 and int(bytes_per_thread) > 0) or not (threads_per_sm and blocks_per_sm) or ceiling < min_threads:
        return max(min_threads, warp), max(0, int(bytes_per_thread)) * max(min_threads, warp)

    best = (0, 0, 0)  # (resident threads, width, shared bytes) - the width breaks ties, wider wins
    start = max(min_threads, warp)
    if power_of_two and start & (start - 1):
        start = 1 << (start - 1).bit_length()
    width = start
    while width <= ceiling:
        shared = int(bytes_per_thread) * width
        if shared_per_block and shared > shared_per_block:
            break  # every wider block needs more, so nothing past here fits either
        # The driver's own per-block reservation counts against the SM's shared memory just as the kernel's
        # own bytes do, so a block occupies `shared + reserved`. Ignoring it overcounts resident blocks.
        occupied = shared + reserved_per_block if shared else 0
        by_shared = shared_per_sm // occupied if occupied else blocks_per_sm
        resident_blocks = min(blocks_per_sm, threads_per_sm // width, by_shared)
        if resident_blocks >= 1:
            resident = resident_blocks * width
            if resident >= best[0]:  # >=, not >: a tie prefers the wider block reached later
                best = (resident, width, shared)
        width = width * 2 if power_of_two else width + warp
    if best[1] == 0:
        floor_width = max(min_threads, warp)
        return floor_width, int(bytes_per_thread) * floor_width
    return best[1], best[2]


def _blocks_per_sm(cc_major: int, cc_minor: int, device_id: int) -> int:
    """Resident-block cap per SM: the driver's own answer, falling back to the per-CC table."""
    live = query_cuda_device_attribute("MAX_BLOCKS_PER_MULTIPROCESSOR", device_id)
    if live is not None and live > 0:
        return live
    return CC_MAX_BLOCKS_PER_SM.get((int(cc_major), int(cc_minor)), 16)


@lru_cache(maxsize=16)
def _static_gpu_caps(device_id: int = 0) -> Optional[dict]:
    """HARDWARE-INVARIANT half of :func:`gpu_capability_summary`, memoized per device id.

    Everything here (compute capability, SM count, the per-block / per-SM limits, warp size, the
    device name and its TOTAL VRAM) is a property of the installed part and cannot change while the
    process runs -- yet assembling it costs an ``nvidia-smi`` subprocess through GPUtil (measured
    64-66 ms) plus a driver query (4.6 ms). Keyed on ``device_id`` so a multi-GPU box gets a
    distinct entry per device; cleared by :func:`reset_cache`.

    Returns ``None`` on a CPU-only host. The returned dict is the CACHED object -- callers must copy
    before mutating (:func:`gpu_capability_summary` does).
    """
    if not is_cuda_available():
        return None

    caps = get_gpu_cuda_capabilities(device_id=device_id)
    if not caps:
        return None

    cc_major = int(caps.get("COMPUTE_CAPABILITY_MAJOR", 0))
    cc_minor = int(caps.get("COMPUTE_CAPABILITY_MINOR", 0))

    summary: dict = {
        "cc_major": cc_major,
        "cc_minor": cc_minor,
        "sm_count": int(caps.get("MULTIPROCESSOR_COUNT", 0)),
        "total_cuda_cores": int(caps.get("TOTAL_CUDA_CORES", 0)),
        "max_threads_per_block": int(caps.get("MAX_THREADS_PER_BLOCK", 0)),
        "max_shared_mem_per_block": int(caps.get("MAX_SHARED_MEMORY_PER_BLOCK", 0)),
        "max_shared_mem_per_block_optin": int(caps.get("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", 0)),
        # PER-SM limits. These are what an occupancy calculation actually needs, and they were being
        # dropped even though `get_gpu_cuda_capabilities` already returns them - a caller wanting a block
        # size had nothing to compute one FROM and had to hardcode a target instead.
        "max_threads_per_sm": int(caps.get("MAX_THREADS_PER_MULTI_PROCESSOR", 0)),
        "max_shared_mem_per_sm": int(caps.get("MAX_SHARED_MEMORY_PER_MULTIPROCESSOR", 0)),
        "max_registers_per_sm": int(caps.get("MAX_REGISTERS_PER_MULTIPROCESSOR", 0)),
        # LIVE from the driver, with the per-CC table only as a fallback: a table cannot know about a part
        # released after it was written, and this attribute has no substitute in an occupancy calculation.
        "max_blocks_per_sm": _blocks_per_sm(cc_major, cc_minor, device_id),
        # The driver reserves shared memory per block on top of what the kernel asks for, so an occupancy
        # calculation that ignores it overcounts resident blocks - measured on an Ada part, 23 blocks
        # against the 17 the hardware really holds.
        "reserved_shared_mem_per_block": query_cuda_device_attribute("RESERVED_SHARED_MEMORY_PER_BLOCK", device_id) or 0,
        "warp_size": int(caps.get("WARP_SIZE", WARP_SIZE)),
        "free_vram_gb": None,
        "total_vram_gb": None,
        "name": None,
    }

    gpus = get_gpuutil_gpu_info(attrs="id,name,memoryFree,memoryTotal")
    for g in gpus or ():
        try:
            if int(g["id"]) == int(device_id):
                # Recorded here only as the STARTING value; ``gpu_capability_summary`` overwrites it
                # with a live reading, since free VRAM is the one field that genuinely moves.
                summary["free_vram_gb"] = float(g.get("memoryFree", 0.0))
                summary["total_vram_gb"] = float(g.get("memoryTotal", 0.0))
                summary["name"] = g.get("name")
                break
        except (KeyError, TypeError, ValueError):
            continue

    return summary


def free_vram_gb(device_id: int = 0) -> Optional[float]:
    """Free VRAM on ``device_id`` in GB, right now, or ``None`` when it cannot be read.

    Sourced from cupy's ``memGetInfo`` (an in-process driver call) rather than GPUtil, which shells
    out to ``nvidia-smi`` -- 64 ms and, under GPU contention, unbounded. This is the only genuinely
    LIVE field of :func:`gpu_capability_summary`, which is why it is a separate helper.
    """
    free = _free_bytes_via_cupy(device_id)
    if free is not None:
        return float(free) / (1024.0**3)
    for g in get_gpuutil_gpu_info(attrs="id,memoryFree") or ():
        try:
            if int(g["id"]) == int(device_id):
                return float(g.get("memoryFree", 0.0))
        except (KeyError, TypeError, ValueError):  # noqa: PERF203 -- a malformed GPUtil row must be skipped, not abort the scan
            continue
    return None


def gpu_capability_summary(device_id: int = 0) -> Optional[dict]:
    """Flat summary of a GPU device's CUDA capabilities and live VRAM.

    Args:
        device_id: CUDA device id (default 0).

    Returns:
        Dict with keys: ``cc_major``, ``cc_minor``, ``sm_count``,
        ``total_cuda_cores``, ``max_threads_per_block``,
        ``max_shared_mem_per_block``, ``warp_size``, ``free_vram_gb``,
        ``total_vram_gb``, ``name``. Returns ``None`` on CPU-only hosts.

    The hardware-invariant fields come from the per-device :func:`_static_gpu_caps` cache (one
    ``nvidia-smi`` subprocess per device per process instead of one per call); ``free_vram_gb`` is
    re-read live on every call via :func:`free_vram_gb`. Call :func:`reset_cache` to force a
    re-probe of the static half.
    """
    static = _static_gpu_caps(device_id=device_id)
    if static is None:
        return None
    summary = dict(static)  # never hand out the cached object -- a caller mutating it would poison every later call
    live = free_vram_gb(device_id)
    if live is not None:
        summary["free_vram_gb"] = live
    return summary


# ---------------------------------------------------------------------------
# CPU vs GPU dispatcher
# ---------------------------------------------------------------------------

def dispatch_cpu_vs_gpu(
    n_work: int,
    gpu_min_work: int = 100_000,
    prefer_gpu: bool = True,
) -> str:
    """Pick ``"gpu"`` or ``"cpu"`` based on workload size and CUDA availability.

    Args:
        n_work: Number of work items (rows, pairs, samples, etc.).
        gpu_min_work: Workload size below which CPU wins regardless of
            availability (the kernel-launch + H2D-copy overhead dominates).
        prefer_gpu: If False, always return ``"cpu"``.

    Returns:
        ``"gpu"`` if all of ``prefer_gpu``, ``is_cuda_available()``, and
        ``n_work >= gpu_min_work`` hold; else ``"cpu"``.
    """
    if not prefer_gpu:
        return "cpu"
    if n_work < gpu_min_work:
        return "cpu"
    if not is_cuda_available():
        return "cpu"
    return "gpu"


__all__ = [
    "CC_SHARED_MEM_BUDGET",
    "WARP_SIZE",
    "cuda_memory_guard",
    "dispatch_cpu_vs_gpu",
    "get_shared_mem_budget_per_block",
    "free_vram_gb",
    "gpu_capability_summary",
    "optimal_threads_per_block",
    "reset_cache",
    "select_best_gpu",
]
