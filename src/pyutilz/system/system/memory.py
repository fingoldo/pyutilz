"""Process/session memory-usage measurement and cleanup helpers (RAM reporting, working-set trim, tracemalloc)."""

#!$ pip install pympler psutil gpu-info pylspci gputil py-cpuinfo
#!$ pip install pycuda

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------


# ensure_installed("pympler psutil numba tqdm gpu-info")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any as _Any, Optional

import pandas as pd
import psutil
import platform, sys
import os

import gc
import ctypes
import tracemalloc
import ctypes.wintypes

# ----------------------------------------------------------------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------------------------------------------------------------


_LAST_OWN_MEMORY_USAGE_GB: float = 0.0
_IS_WINDOWS: bool = platform.system() == "Windows"


def get_own_memory_usage() -> "Optional[float]":
    """Return RAM usage of our own Python process in gigabytes, or
    ``None`` if psutil raises AND we have no prior successful reading
    to fall back on.

    Windows-specific measurement (2026-04-23 fix): on Windows we use
    ``memory_info().private`` (private commit charge) instead of ``.rss``.
    Rationale: ``rss`` on Windows is the *working set* (pages currently
    resident in physical RAM). ``clean_ram()`` calls
    ``SetProcessWorkingSetSizeEx(..., QUOTA_LIMITS_HARDWS_MIN_DISABLE)``
    to release freed pandas / libc pages back to the OS, which
    deliberately evicts working-set pages to the pagefile — so ``rss``
    plunges to near-zero right after ``clean_ram()`` even though the
    process still has the same committed memory. The old mitigation
    logged ``"psutil reported rss=0.003GB after previous 58.6GB; likely
    transient reporting glitch"`` on every training run, obscuring real
    RAM usage in the log. ``private`` (a.k.a. Private Bytes) reflects
    committed memory, unaffected by working-set trim; verified
    empirically 2026-04-23 (rss: 27MB→1MB after trim, private:
    331MB→331MB unchanged).

    Linux keeps the historical ``rss`` semantics — there's no
    SetProcessWorkingSetSizeEx analogue, and ``malloc_trim`` only
    releases heap back to libc without forcing page eviction.

    Contract for exceptional paths:
      * If ``psutil.Process(...)`` raises AND there is NO prior cached
        reading (module-level cache still at its init value ``0.0``),
        return ``None``. Preserves the original sentinel for "measurement
        never succeeded".
      * If there IS a prior cached reading, return it — a transient
        failure shouldn't discard a known-good reading and show 0 GB.
    """
    global _LAST_OWN_MEMORY_USAGE_GB
    try:
        pid = os.getpid()
        py = psutil.Process(pid)
        mi = py.memory_info()
        if _IS_WINDOWS and hasattr(mi, "private"):
            memory_usage = mi.private / 2.0**30  # committed memory, stable across trim
        else:
            memory_usage = mi.rss / 2.0**30
    except Exception as e:
        logger.exception(e)
        if _LAST_OWN_MEMORY_USAGE_GB > 0.0:
            return _LAST_OWN_MEMORY_USAGE_GB
        return None

    # Safety net retained for Linux — on rare occasions psutil still
    # reports implausibly low rss after large Arrow/Polars frees there.
    # On Windows this branch should now never trigger since ``private``
    # doesn't exhibit the working-set-trim plunge.
    if _LAST_OWN_MEMORY_USAGE_GB > 1.0 and memory_usage < 0.1:
        logger.warning(
            "psutil reported memory=%.3fGB after previous %.1fGB; likely transient reporting glitch, returning previous value to keep the RAM log honest.",
            memory_usage,
            _LAST_OWN_MEMORY_USAGE_GB,
        )
        return _LAST_OWN_MEMORY_USAGE_GB

    _LAST_OWN_MEMORY_USAGE_GB = memory_usage
    return memory_usage  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def trim_windows_process_memory(pid: Optional[int] = None) -> bool:
    """Causes effect similar to malloc_trim on -nix."""

    # Define SIZE_T based on the platform (32-bit or 64-bit)
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        SIZE_T: _Any = ctypes.c_uint32
    else:
        SIZE_T = ctypes.c_uint64

    # Get a handle to the current process
    if not pid:
        pid = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]

    # Define argument and return types for SetProcessWorkingSetSizeEx
    ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.argtypes = [  # type: ignore[attr-defined]
        ctypes.wintypes.HANDLE,  # Process handle
        SIZE_T,  # Minimum working set size
        SIZE_T,  # Maximum working set size
        ctypes.wintypes.DWORD,  # Flags
    ]
    ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.restype = ctypes.wintypes.BOOL  # type: ignore[attr-defined]

    # Define constants for SetProcessWorkingSetSizeEx
    QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002

    # Attempt to set the working set size
    result = ctypes.windll.kernel32.SetProcessWorkingSetSizeEx(pid, SIZE_T(-1), SIZE_T(-1), QUOTA_LIMITS_HARDWS_MIN_DISABLE)  # type: ignore[attr-defined]

    if result == 0:
        # Retrieve the error code
        error_code = ctypes.windll.kernel32.GetLastError()  # type: ignore[attr-defined]
        logger.error("SetProcessWorkingSetSizeEx failed with error code: %s", error_code)
        return False
    else:
        return True


def clean_ram() -> None:
    """Forces python garbage collection.
    Most importantly, calls malloc_trim/SetProcessWorkingSetSizeEx, which fixes pandas/libc (?) memory leak.

    Windows RSS-reporting side-effect (important for RAM-monitoring callers):
        ``trim_windows_process_memory()`` (called below) invokes
        ``SetProcessWorkingSetSizeEx(pid, (SIZE_T)-1, (SIZE_T)-1,
        QUOTA_LIMITS_HARDWS_MIN_DISABLE)``. The `-1, -1` pair sets both min
        and max working set to the "no limit" sentinel, and
        ``QUOTA_LIMITS_HARDWS_MIN_DISABLE`` removes the hard minimum so
        Windows is free to shrink the working set aggressively. The
        immediate practical effect is that the OS evicts currently-resident
        pages from the process working set to the pagefile.

        Because ``psutil.Process.memory_info().rss`` on Windows is the
        *working set size* (resident pages in physical RAM), ``rss``
        **plunges to near-zero right after this call** — even though the
        process still holds the same committed memory and will page it
        back in on next access. ``memory_info().private`` (Private Bytes,
        committed memory) is **unaffected** by the trim.

        Empirical confirmation (2026-04-23):
            BEFORE trim:  rss=0.027 GB, private=0.331 GB
            AFTER  trim:  rss=0.001 GB, private=0.331 GB

        Downstream monitoring code (e.g. ``get_own_memory_usage``) that
        wants a stable reading across ``clean_ram()`` on Windows must
        read ``private``, not ``rss``. ``get_own_memory_usage`` in this
        module does this automatically; if you call ``memory_info()``
        directly, be aware of the trade-off.
    """

    gc.collect()
    if platform.system() == "Windows":
        trim_windows_process_memory()
    else:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            logger.error("malloc_trim attempt failed")


def show_biggest_session_objects(session: dict, N: int = 5, min_size_bytes: int = 1) -> pd.DataFrame:
    """

    Then reports own process RAM usage & the mnost RAM consuming objects.
    """

    clean_ram()

    logger.info("Own process RAM usage: %.2f GB", get_own_memory_usage())

    # Start tracing memory allocations
    # tracemalloc.start()

    # Retrieve all objects from the current Python session
    res = []
    for obj in session.values():
        try:
            size = sys.getsizeof(obj)
        except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional (skip this object, keep scanning the rest)
            logger.warning("stumbled on object of type %s", type(obj))
            pass
        else:
            if size >= min_size_bytes:
                res.append(dict(type=type(obj), size_gb=size / 1024**3))
    if res:
        res = pd.DataFrame(res).sort_values("size_gb", ascending=False).head(N)
    return res


def show_tracemalloc_snapshot(N: int = 10):
    """Take a snapshot of memory usage and display top N lines.

    Note: This function properly starts and stops tracemalloc to avoid resource leaks.
    """
    tracemalloc.start()
    try:
        # Take a snapshot of memory usage
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        logger.info("Top %s memory-consuming lines:", N)
        for stat in top_stats[:N]:
            logger.info("%s", stat)

        return snapshot
    finally:
        # Always stop tracemalloc to prevent resource leak
        tracemalloc.stop()
