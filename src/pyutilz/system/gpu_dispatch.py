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
    """Clear the :func:`select_best_gpu` memoization cache."""
    _select_best_gpu_cached.cache_clear()


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

# Maximum RESIDENT BLOCKS per SM, by compute capability. The one occupancy limit numba's device attributes
# do not expose (cupy does, as `MaxBlocksPerMultiprocessor`, but requiring cupy for a constant would make
# this module depend on it). Values are from the CUDA C Programming Guide's own occupancy table.
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

    Returns:
        ``(threads_per_block, shared_bytes_per_block)``. Falls back to ``(min_threads, ...)`` on a CPU-only
        host or when the device reports no usable limits, so a caller never has to special-case that.
    """
    if caps is None:
        summary = gpu_capability_summary(device_id=device_id)
        caps = summary if summary is not None else {}

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
    if max_threads is not None:
        ceiling = int(max_threads)
    else:
        stated_ceiling = _limit("max_threads_per_block", 0)
        ceiling = stated_ceiling if stated_ceiling > 0 else min_threads
    if not (shared_per_sm and threads_per_sm and blocks_per_sm) or ceiling < min_threads:
        return max(min_threads, warp), max(0, int(bytes_per_thread)) * max(min_threads, warp)

    best = (0, 0, 0)  # (resident threads, width, shared bytes) - the width breaks ties, wider wins
    width = max(min_threads, warp)
    while width <= ceiling:
        shared = int(bytes_per_thread) * width
        if shared_per_block and shared > shared_per_block:
            break  # every wider block needs more, so nothing past here fits either
        by_shared = shared_per_sm // shared if shared else blocks_per_sm
        resident_blocks = min(blocks_per_sm, threads_per_sm // width, by_shared)
        if resident_blocks >= 1:
            resident = resident_blocks * width
            if resident >= best[0]:  # >=, not >: a tie prefers the wider block reached later
                best = (resident, width, shared)
        width += warp
    if best[1] == 0:
        floor_width = max(min_threads, warp)
        return floor_width, int(bytes_per_thread) * floor_width
    return best[1], best[2]


def gpu_capability_summary(device_id: int = 0) -> Optional[dict]:
    """Flat summary of a GPU device's CUDA capabilities and live VRAM.

    Args:
        device_id: CUDA device id (default 0).

    Returns:
        Dict with keys: ``cc_major``, ``cc_minor``, ``sm_count``,
        ``total_cuda_cores``, ``max_threads_per_block``,
        ``max_shared_mem_per_block``, ``warp_size``, ``free_vram_gb``,
        ``total_vram_gb``, ``name``. Returns ``None`` on CPU-only hosts.
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
        "max_blocks_per_sm": CC_MAX_BLOCKS_PER_SM.get((cc_major, cc_minor), 16),
        "warp_size": int(caps.get("WARP_SIZE", WARP_SIZE)),
        "free_vram_gb": None,
        "total_vram_gb": None,
        "name": None,
    }

    gpus = get_gpuutil_gpu_info(attrs="id,name,memoryFree,memoryTotal")
    for g in gpus or ():
        try:
            if int(g["id"]) == int(device_id):
                summary["free_vram_gb"] = float(g.get("memoryFree", 0.0))
                summary["total_vram_gb"] = float(g.get("memoryTotal", 0.0))
                summary["name"] = g.get("name")
                break
        except (KeyError, TypeError, ValueError):
            continue

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
    "gpu_capability_summary",
    "optimal_threads_per_block",
    "reset_cache",
    "select_best_gpu",
]
