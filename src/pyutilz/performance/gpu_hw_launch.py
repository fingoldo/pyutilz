"""Hardware-derived launch geometry for CUDA kernels: threads per block, grid, from the device the kernel runs on.

A block size written as a literal (``kernel((n // 256 + 1,), (256,), args)``) is a number measured on one card and wrong on every other: a 6-SM consumer card and a 100-SM data-centre
card want different geometry, and so does a kernel that uses more registers or shared memory per thread. This module queries the device properties ONCE per device (cached) and derives the
launch parameters from them with the standard CUDA occupancy formula:

  * ``occupancy_block_candidates``: the warp-multiple block sizes that reach a target number of active blocks per SM for a kernel's measured register and shared-memory use. The candidate set
    is hardware-valid on any card, and a tuning-cache sweep can range over it.
  * ``block_size_for(kernel)``: one call that reads ``kernel.num_regs`` / ``kernel.shared_size_bytes`` and returns the block size that keeps the most threads resident.
  * ``fill_grid_1d`` / ``launch_1d``: the grid for a flat 1-D kernel, enough blocks to cover the work and to fill every SM.
  * ``fixed_shared_block_size``: the width for a kernel whose shared memory is fixed per block (a histogram or bitset that does not grow with the width).

Why an analytic occupancy and not the CUDA occupancy API: ``cupy.cuda.runtime`` does not export ``occupancyMaxActiveBlocksPerMultiprocessor`` and ``RawKernel`` has no occupancy method, but it
does expose ``num_regs`` / ``shared_size_bytes`` / ``max_threads_per_block``, and the device attributes give the per-SM limits, so max-active-blocks/SM follows from the formula (the minimum
over the warp-, register-, shared- and block-count limits). Without a device (no cupy, no CUDA) every function degrades to "no hardware information" (an empty dict, an empty list, 0), and the
caller keeps its default.

The derived geometry never changes a kernel's RESULT as long as the kernel is invariant to the launch shape (a grid-stride loop, a sum or order-statistic reduction); a kernel whose shared
arrays are sized by the block at compile time has to pass ``max_width`` (or be written with dynamic shared memory) to stay inside what it was written for.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# Device-property cache: the properties of ONE device, the one whose id is ``_DEV_PROPS_ID`` (None = the id was not
# recorded, e.g. a test that sets ``_DEV_PROPS`` directly). A different current device re-queries, so a host with
# two different cards never hands the first card's limits to the second.
_DEV_PROPS: "dict | None" = None
_DEV_PROPS_ID: "int | None" = None
_DEV_PROPS_LOCK = threading.Lock()


def _current_device_id() -> "int | None":
    """Id of the current cupy device, or ``None`` when cupy / CUDA is unavailable."""
    try:
        import cupy as cp

        return int(cp.cuda.Device().id)
    except Exception as _silent_exc:
        logger.debug("gpu_hw_launch: suppressed %r", _silent_exc)
        return None


def launch_device_id() -> int:
    """Id of the current CUDA device (the one the next launch runs on), 0 when it cannot be read: what a capability query must be asked about instead of a hard-coded device 0."""
    dev = _current_device_id()
    return 0 if dev is None else int(dev)


def device_props() -> dict:
    """Launch-relevant properties of the CURRENT device (cached per device id). Empty dict if cupy/CUDA is unavailable - callers
    treat an empty dict as 'no HW info' and fall back to their historical default / KTC sweep."""
    global _DEV_PROPS
    cur = _current_device_id()
    if _DEV_PROPS is not None and (_DEV_PROPS_ID is None or cur is None or _DEV_PROPS_ID == cur):  # fast path, no lock
        return _DEV_PROPS
    with _DEV_PROPS_LOCK:
        if _DEV_PROPS is not None and (_DEV_PROPS_ID is None or cur is None or _DEV_PROPS_ID == cur):  # re-check under the lock
            return _DEV_PROPS
        return _device_props_uncached()


def _device_props_uncached() -> dict:
    """Populate and cache ``_DEV_PROPS`` under the caller's lock. Query the current CUDA device once."""
    global _DEV_PROPS, _DEV_PROPS_ID
    props: dict = {}
    dev_id: "int | None" = None
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        dev_id = int(dev.id)
        a = dev.attributes
        props = {
            "sm_count": int(a.get("MultiProcessorCount", 0)) or 0,
            "max_threads_per_block": int(a.get("MaxThreadsPerBlock", 1024)) or 1024,
            "max_threads_per_sm": int(a.get("MaxThreadsPerMultiProcessor", 2048)) or 2048,
            "max_blocks_per_sm": int(a.get("MaxBlocksPerMultiprocessor", 16)) or 16,
            "warp": int(a.get("WarpSize", 32)) or 32,
            "shared_per_block": int(a.get("MaxSharedMemoryPerBlock", 48 * 1024)) or 48 * 1024,
            "regs_per_block": int(a.get("MaxRegistersPerBlock", 65536)) or 65536,
            # shared memory the driver reserves per block on top of the kernel's own (1 KB from cc 8.x on, 0 on Pascal) and the L2 size
            "reserved_shared_per_block": int(a.get("ReservedSharedMemoryPerBlock", 0)) or 0,
            "l2_bytes": int(a.get("L2CacheSize", 0)) or 0,
            # registers are handed out per warp in units of 256 on every architecture from Kepler to Hopper
            "reg_alloc_unit": 256,
        }
        # Per-SM register / shared budgets (occupancy is limited per-SM, not per-block). getDeviceProperties
        # exposes them; fall back to the per-block budget when absent (a safe lower bound for occupancy).
        try:
            p = cp.cuda.runtime.getDeviceProperties(dev.id)
            props["regs_per_sm"] = int(p.get("regsPerMultiprocessor", props["regs_per_block"])) or props["regs_per_block"]
            props["shared_per_sm"] = int(p.get("sharedMemPerMultiprocessor", props["shared_per_block"])) or props["shared_per_block"]
        except Exception as e:
            logger.warning("querying regs/shared-per-multiprocessor failed, using per-block values: %s", e)
            props["regs_per_sm"] = props["regs_per_block"]
            props["shared_per_sm"] = props["shared_per_block"]
    except Exception as e:
        logger.warning("device-property probe failed: %s", e)
        props = {}
    _DEV_PROPS = props
    _DEV_PROPS_ID = dev_id
    return props


def _max_active_blocks_per_sm(block: int, regs_per_thread: int, static_smem: int, dyn_smem: int, props: dict) -> int:
    """Analytic max active blocks/SM for a candidate ``block`` given the kernel's per-thread register use and
    its static+dynamic shared bytes - min over the warp-, register-, shared-, and block-count SM limits
    (the standard CUDA occupancy calculation). Returns 0 if ``block`` is itself infeasible (over the
    per-block thread or shared cap)."""
    warp = props["warp"]
    if block <= 0 or block > props["max_threads_per_block"]:
        return 0
    smem_per_block = static_smem + dyn_smem
    if smem_per_block > props["shared_per_block"]:
        return 0
    # warp / thread limit
    warps_per_block = (block + warp - 1) // warp
    max_warps_per_sm = props["max_threads_per_sm"] // warp
    by_threads = max_warps_per_sm // warps_per_block if warps_per_block > 0 else 0
    # register limit: registers are allocated per warp, rounded up to ``reg_alloc_unit`` (53 registers cost 56)
    unit = int(props.get("reg_alloc_unit", 1)) or 1
    regs_per_warp = ((max(regs_per_thread, 1) * warp + unit - 1) // unit) * unit
    regs_per_block_used = warps_per_block * regs_per_warp
    by_regs = props["regs_per_sm"] // regs_per_block_used if regs_per_block_used > 0 else props["max_blocks_per_sm"]
    # shared limit: the driver adds its per-block reserve to the kernel's request
    smem_charged = smem_per_block + int(props.get("reserved_shared_per_block", 0))
    by_smem = (props["shared_per_sm"] // smem_charged) if smem_charged > 0 else props["max_blocks_per_sm"]
    # hard block-count cap
    by_blocks = props["max_blocks_per_sm"]
    return int(max(0, min(by_threads, by_regs, by_smem, by_blocks)))


def occupancy_block_candidates(
    *,
    regs_per_thread: int = 32,
    static_smem: int = 0,
    dyn_smem: int = 0,
    min_active_blocks: int = 2,
    block_cap: "int | None" = None,
) -> list:
    """HW-VALID, occupancy-reasonable threads/block candidates for the current device: every WARP-MULTIPLE
    block size (warp .. min(MaxThreadsPerBlock, block_cap)) whose analytic max-active-blocks/SM is >=
    ``min_active_blocks`` (so the SM is not starved by one fat block). These seed/bound the KTC sweep so the
    empirical winner is always HW-sane and the candidate set ports to other cards. Empty list -> no HW info
    (caller keeps its historical sweep / default).

    ``regs_per_thread`` / ``static_smem`` come from the compiled kernel (RawKernel.num_regs /
    .shared_size_bytes) so the occupancy bound reflects THIS kernel; ``dyn_smem`` is the per-launch dynamic
    shared (e.g. the radix histogram R*256*4 or the hist nb_k*K_y*4). When a kernel's dynamic shared scales
    with the data, pass a representative/worst-case value."""
    props = device_props()
    if not props:
        return []
    warp = props["warp"]
    cap = props["max_threads_per_block"]
    if block_cap is not None:
        cap = min(cap, int(block_cap))
    out = []
    b = warp
    while b <= cap:
        if _max_active_blocks_per_sm(b, regs_per_thread, static_smem, dyn_smem, props) >= min_active_blocks:
            out.append(b)
        b += warp
    # If occupancy is so register/shared-bound that NO block hits min_active_blocks, fall back to the
    # warp-multiples that are merely feasible (>=1 active block) so the sweep still has HW-valid options.
    if not out:
        b = warp
        while b <= cap:
            if _max_active_blocks_per_sm(b, regs_per_thread, static_smem, dyn_smem, props) >= 1:
                out.append(b)
            b += warp
    return out


def fill_grid_1d(n_work: int, block: int, *, regs_per_thread: int = 32, static_smem: int = 0, dyn_smem: int = 0) -> int:
    """Grid (number of blocks) for a flat 1-D kernel over ``n_work`` elements: enough to cover the work AND
    to fill every SM (>= MultiProcessorCount * max_active_blocks_per_SM) so a big card is never left with a
    tiny grid. For a grid-stride kernel the extra blocks are harmless (they just take a stride each); for a
    one-element-per-thread kernel ceil(n_work/block) already covers it and dominates on large n."""
    if block <= 0:
        return 1
    cover = (int(n_work) + block - 1) // block
    props = device_props()
    if not props:
        return max(1, cover)
    mab = _max_active_blocks_per_sm(block, regs_per_thread, static_smem, dyn_smem, props)
    fill = props["sm_count"] * max(mab, 1)
    return int(max(1, cover, fill))


def fixed_shared_block_size(
    shared_bytes: int,
    *,
    regs_per_thread: int = 32,
    static_smem: int = 0,
    pow2: bool = True,
    min_width: "int | None" = None,
    max_width: "int | None" = None,
    work_items: "int | None" = None,
    tie_prefers_wider: bool = False,
) -> int:
    """Threads per block for a kernel whose shared memory is FIXED per block (a histogram or bitset that does not grow with the width).

    Wider blocks share the same table among more threads, so the right width is the one that keeps the most threads resident on an SM:
    for every candidate width (powers of two when ``pow2``, else warp multiples) the analytic active blocks per SM is taken from the
    thread, register (per-warp allocation), shared (plus the driver reserve) and block-count limits, and the width with the most resident
    threads wins; at equal resident threads the NARROWER block wins by default (more blocks per SM hide the barriers of a phased kernel: measured 147 ms for 512 x 2 blocks against 170 ms for 1024 x 1 on the honest-null accept kernel), the wider one with ``tie_prefers_wider`` (fewer block-level reductions). ``work_items`` caps the width at the work one block can use.
    Returns 0 without device information, so the caller keeps its historical constant.
    """
    props = device_props()
    if not props:
        return 0
    warp = props["warp"]
    lo = max(warp, int(min_width) if min_width is not None else warp)
    hi = min(props["max_threads_per_block"], int(max_width) if max_width is not None else props["max_threads_per_block"])
    if work_items is not None and int(work_items) > 0:
        hi = min(hi, ((int(work_items) + warp - 1) // warp) * warp)
    if hi < lo:
        return 0
    best_w, best_resident = 0, -1
    w = 1 << (lo - 1).bit_length() if pow2 else lo
    while w <= hi:
        blocks = _max_active_blocks_per_sm(w, regs_per_thread, static_smem, int(shared_bytes), props)
        resident = blocks * w
        if blocks > 0 and (resident > best_resident or (tie_prefers_wider and resident == best_resident)):
            best_w, best_resident = w, resident
        w = w * 2 if pow2 else w + warp
    return int(best_w)


def block_size_for(
    kernel: object,
    *,
    dyn_smem: int = 0,
    min_active_blocks: int = 2,
    max_width: "int | None" = None,
    work_items: "int | None" = None,
    pow2: bool = False,
    preferred: int = 256,
    default: int = 256,
) -> int:
    """Threads per block for ``kernel`` (a compiled cupy ``RawKernel`` / module function) on the CURRENT device: the warp multiple that keeps the most threads resident on an SM.

    The kernel's own ``num_regs`` and ``shared_size_bytes`` and the device limits go through the occupancy formula; ``dyn_smem`` is the dynamic shared memory the launch asks for.
    Among the widths that reach ``min_active_blocks`` per SM the one with the most resident threads wins. Several widths usually reach the same resident count (a card with 2048 threads per SM
    is full with 64 x 32, 256 x 8 or 1024 x 2 blocks); the one closest to ``preferred`` wins then, the narrower of two equally close. ``max_width`` bounds the width for a kernel written for at
    most that many threads; ``work_items`` stops it growing past the work (one thread per item cannot use more).
    Returns ``default`` when there is no device information, so the CPU / no-CUDA path and the old behaviour are unchanged.
    """
    regs = int(getattr(kernel, "num_regs", 0) or 0) or 32
    static = int(getattr(kernel, "shared_size_bytes", 0) or 0)
    candidates = occupancy_block_candidates(regs_per_thread=regs, static_smem=static, dyn_smem=int(dyn_smem), min_active_blocks=min_active_blocks, block_cap=max_width)
    if pow2:
        candidates = [b for b in candidates if b & (b - 1) == 0]
    if not candidates:
        return int(default)
    props = device_props()
    if work_items is not None and int(work_items) > 0:
        warp = props["warp"]
        limit = ((int(work_items) + warp - 1) // warp) * warp
        capped = [b for b in candidates if b <= limit]
        candidates = capped or candidates[:1]
    best, best_resident = candidates[0], -1
    for width in candidates:
        resident = _max_active_blocks_per_sm(width, regs, static, int(dyn_smem), props) * width
        if resident > best_resident or (resident == best_resident and abs(width - int(preferred)) < abs(best - int(preferred))):
            best, best_resident = width, resident
    return int(best)


def launch_1d(kernel: object, n_work: int, *, dyn_smem: int = 0, max_width: "int | None" = None, pow2: bool = False, default_block: int = 256) -> "tuple[tuple[int], tuple[int]]":
    """``((grid,), (block,))`` for a flat 1-D launch of ``kernel`` over ``n_work`` elements: the block from :func:`block_size_for`, the grid from :func:`fill_grid_1d`."""
    block = block_size_for(kernel, dyn_smem=dyn_smem, max_width=max_width, work_items=n_work, pow2=pow2, default=default_block)
    regs = int(getattr(kernel, "num_regs", 0) or 0) or 32
    static = int(getattr(kernel, "shared_size_bytes", 0) or 0)
    return (fill_grid_1d(n_work, block, regs_per_thread=regs, static_smem=static, dyn_smem=int(dyn_smem)),), (block,)
