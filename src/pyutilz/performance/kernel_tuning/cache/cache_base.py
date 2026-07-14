"""Core on-disk kernel-tuning cache: hardware fingerprinting, cache-dir resolution, and provenance tracking."""
from __future__ import annotations

import datetime as _dt
import errno
import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from typing import Optional

from pyutilz.system.gpu_dispatch import gpu_capability_summary

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3  # v3: immutable per-(host,kernel,code_version) files + O_EXCL sweep markers (was v2: monolithic JSON)

# How long an INPROGRESS sweep marker is trusted before a would-be sweeper may
# STEAL it (owner crashed / hung). The expensive GPU/CPU sweeps run hundreds of
# seconds; this budget must exceed a legitimate sweep but bound the wedge from a
# killed-mid-sweep process. Override via $PYUTILZ_KERNEL_SWEEP_BUDGET_SEC.
_DEFAULT_SWEEP_BUDGET_SECONDS = 1800.0

# Sentinel used in a filename when a tuning carries no code_version (update()
# may be called without one). A real code_version is a SHA-256 hex string.
_NO_CODE_VERSION = "nocv"

def _slug(s: str, maxlen: int = 40) -> str:
    """Filename-safe lowercase + truncated form of an arbitrary string."""
    s = re.sub(r"\(R\)|\(TM\)|\bCPU\b|\bGPU\b|@.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s.strip("-._").lower()[:maxlen] or "unknown"


@lru_cache(maxsize=1)
def _cpu_model_slug() -> str:
    """Filename-safe slug of the CPU's brand string (e.g. "unknown" on probe failure), cached for the process lifetime."""
    # cpuinfo.get_cpu_info() runs a ~2s WMI/CPUID probe on Windows (and can
    # stall under load); the CPU model is invariant, so cache it for the process
    # lifetime. This lru survives hw_fingerprint.cache_clear() (which tests call
    # every test), so cpuinfo runs at most ONCE per process regardless of how
    # often the fingerprint lru or the cache-dir is reset.
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return _slug(info.get("brand_raw", "unknown"))
    except Exception:
        return "unknown"


def _current_device_id() -> int:
    """Return the live CUDA device id (whatever the caller is using).
    Falls back to 0 on probe failure. Lets the cache key reflect e.g.
    ``device=1`` on a 2-GPU box where the user routed mlframe to a
    non-default device."""
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDevice())
    except Exception:
        return 0


@lru_cache(maxsize=16)
def _gpu_summary_cached(device_id: int):
    """Per-device GPU capability probe (nvidia-smi / gputil / cupy query --
    ~0.1-2s and can stall under load), cached for the process. Keyed BY device
    id so a multi-GPU box gets a distinct cached summary per device (the
    per-device fingerprint + the multi-GPU sweep rely on this). CPU is cached
    globally (_cpu_model_slug -- one CPU); GPUs must be per-device. Tests that
    mock gpu_capability_summary must call _gpu_summary_cached.cache_clear()."""
    # Resolve ``gpu_capability_summary`` through the FACADE package so a
    # ``mock.patch.object(cache, "gpu_capability_summary", ...)`` on the public
    # package (as the HW-fingerprint tests do) is honored; falls back to the real
    # import when unpatched.
    import sys as _sys
    _facade = _sys.modules.get("pyutilz.performance.kernel_tuning.cache")
    _probe = getattr(_facade, "gpu_capability_summary", gpu_capability_summary)
    return _probe(device_id)


def _gpu_slug_and_cc() -> tuple[str, str]:
    """Returns (gpu_name_slug, cc_str). On CPU-only host: ("no-gpu", "").

    Uses the LIVE current CUDA device id, not always 0, so a 2-GPU box
    where the user routes to device 1 gets a distinct fingerprint
    (different GPU model + cc may apply).
    """
    try:
        dev_id = _current_device_id()
        summary = _gpu_summary_cached(dev_id)
        if summary is None:
            return ("no-gpu", "")
        name = summary.get("name") or "unknown"
        cc = f"{int(summary.get('cc_major', 0))}.{int(summary.get('cc_minor', 0))}"
        return (_slug(name), cc)
    except Exception as e:
        logger.debug("gpu_capability_summary failed: %s", e)
        return ("no-gpu", "")


_HW_FP_DISK_FILENAME = ".hw_fingerprint.json"
_HW_FP_SCHEMA_VERSION = 1
_HW_FP_FRESHNESS_SECONDS = 7 * 24 * 3600  # 7 days


def _read_hw_fingerprint_from_disk() -> Optional[str]:
    """Return the fingerprint from the on-disk cache if present, schema-
    compatible, and recent enough; ``None`` otherwise.

    Uses file mtime rather than an embedded timestamp so a stale file is
    invalidated even if the JSON parses fine. The default freshness
    window is 7 days; ``PYUTILZ_HW_FP_REFRESH=1`` forces a recompute
    even on a fresh file (for users who just swapped GPU / upgraded
    drivers and don't want to delete the file by hand).
    """
    if os.environ.get("PYUTILZ_HW_FP_REFRESH", "").strip() == "1":
        return None
    try:
        # ``cache_dir()`` makedirs on first call -- safe to invoke here
        # before any kernel-tuning JSON exists.
        path = os.path.join(cache_dir(), _HW_FP_DISK_FILENAME)
    except Exception:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    age = max(0.0, time.time() - st.st_mtime)
    if age > _HW_FP_FRESHNESS_SECONDS:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema_version") != _HW_FP_SCHEMA_VERSION:
        return None
    fp = data.get("fingerprint")
    return fp if isinstance(fp, str) and fp else None


def _write_hw_fingerprint_to_disk(fingerprint: str) -> None:
    """Persist the freshly-computed fingerprint. Best-effort: silently
    swallows write errors (read-only homedir, permissions, etc.) so the
    in-memory lru_cache still works."""
    try:
        path = os.path.join(cache_dir(), _HW_FP_DISK_FILENAME)
        tmp = path + ".tmp"
        payload = {
            "schema_version": _HW_FP_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception as e:
        logger.debug("hw_fingerprint: failed to persist to disk: %s", e)


@lru_cache(maxsize=1)
def hw_fingerprint() -> str:
    """Stable per-host key. Format::

        cpu_<cpu_slug>_gpu_<gpu_slug>_cc<major>.<minor>
        cpu_<cpu_slug>_no-gpu                    (CPU-only host)

    Cached two ways:
      * ``lru_cache(maxsize=1)`` for the process lifetime.
      * On-disk JSON at ``<cache_dir>/.hw_fingerprint.json`` shared
        across processes (7-day freshness window via file mtime;
        invalidate manually by deleting the file or setting
        ``PYUTILZ_HW_FP_REFRESH=1``).

    The cross-process cache exists because ``_cpu_model_slug()`` calls
    ``cpuinfo.get_cpu_info()`` which on Windows queries WMI / runs
    CPUID probes (~1.9s cold per process), and ``_gpu_slug_and_cc()``
    queries nvidia-smi via gputil (~100ms-2s cold). For short-lived
    CLI tools / tests / per-target training scripts that pay this
    ~2.7s first-call cost on every invocation, the disk cache drops
    subsequent processes to ~1ms (file read + JSON parse + mtime
    check). HW doesn't change between processes on the same host;
    the 7-day staleness gate covers driver / GPU swaps without
    manual maintenance.
    """
    disk = _read_hw_fingerprint_from_disk()
    if disk is not None:
        return disk
    # Resolve the two HW probes through the FACADE package
    # (``pyutilz.performance.kernel_tuning.cache``) rather than this submodule so a
    # ``monkeypatch.setattr(cache, "_cpu_model_slug", ...)`` on the public package --
    # as every kernel-tuning test does -- is honored here. The facade re-exports the
    # real functions from this module by default; late attribute lookup means a patch
    # applied to the package is seen, while an unpatched run calls the originals.
    import sys as _sys
    _facade = _sys.modules.get("pyutilz.performance.kernel_tuning.cache")
    _cpu_probe = getattr(_facade, "_cpu_model_slug", _cpu_model_slug)
    _gpu_probe = getattr(_facade, "_gpu_slug_and_cc", _gpu_slug_and_cc)
    cpu = _cpu_probe()
    gpu, cc = _gpu_probe()
    if gpu == "no-gpu":
        fp = f"cpu_{cpu}_no-gpu"
    else:
        fp = f"cpu_{cpu}_gpu_{gpu}_cc{cc}"
    _write_hw_fingerprint_to_disk(fp)
    return fp


def cache_dir() -> str:
    """Resolve the on-disk cache directory.

    Order:
        1. ``$PYUTILZ_KERNEL_CACHE_DIR`` env var, if set.
        2. ``~/.pyutilz/kernel_tuning/`` default.

    Creates the directory on first call.
    """
    override = os.environ.get("PYUTILZ_KERNEL_CACHE_DIR", "").strip()
    if override:
        path = override
    else:
        path = os.path.join(os.path.expanduser("~"), ".pyutilz", "kernel_tuning")
    os.makedirs(path, exist_ok=True)
    return path


def cache_path() -> str:
    """Path to the LEGACY monolithic per-host JSON file.

    Kept for backward compatibility (migration source + a stable public path).
    The v3 storage no longer writes this file; tunings live as immutable
    per-kernel files under :func:`host_cache_dir`. ``cache_path`` still resolves
    to the v1/v2 location so a pre-existing monolith is found + migrated.
    """
    return os.path.join(cache_dir(), f"{hw_fingerprint()}.json")


def host_cache_dir() -> str:
    """Per-host directory holding the immutable per-kernel tuning files (v3).

    Layout: ``<cache_dir>/<hw_fingerprint>/<kernel_slug>/<...>.json``. Created on
    first call.
    """
    path = os.path.join(cache_dir(), hw_fingerprint())
    os.makedirs(path, exist_ok=True)
    return path


def _kernel_dir(host_dir: str, kernel_name: str) -> str:
    """Directory for one kernel's immutable tuning files."""
    return os.path.join(host_dir, _slug(kernel_name, maxlen=80))


def _sweep_budget_seconds() -> float:
    """Max-sweep budget (seconds) after which an INPROGRESS marker is steal-able."""
    raw = os.environ.get("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return _DEFAULT_SWEEP_BUDGET_SECONDS


def _pid_alive(pid: int) -> bool:
    """Best-effort: is ``pid`` a live process on this host? Conservative -- on any
    probe uncertainty returns True (don't steal a marker we can't prove is dead)."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            # No os.kill(pid, 0) signal semantics on Windows; query the OS task list.
            # Must build the kernel32 handle with use_last_error=True: the shared
            # ``ctypes.windll.kernel32`` does NOT capture the Win32 thread-local last
            # error, so ``ctypes.get_last_error()`` would read 0 (uninitialized) after
            # a failed OpenProcess -> a dead pid was reported ALIVE, so a stale sweep
            # marker owned by a crashed process was never stolen (the sweep wedged).
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                # ERROR_INVALID_PARAMETER (87) => no such pid (dead). Any other failure
                # (e.g. ERROR_ACCESS_DENIED 5 => alive but not ours) => assume alive.
                return ctypes.get_last_error() not in (87,)  # type: ignore[attr-defined]
            try:
                exit_code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    STILL_ACTIVE = 259
                    return exit_code.value == STILL_ACTIVE
                return True
            finally:
                kernel32.CloseHandle(handle)
        else:
            os.kill(pid, 0)
            return True
    except OSError as e:
        # ESRCH => dead; EPERM => alive but not ours.
        return getattr(e, "errno", None) != errno.ESRCH
    except Exception:
        return True  # never steal on an unexpected probe failure


# ---------------------------------------------------------------------------
# Provenance (recorded on every save; readable for staleness checks)
# ---------------------------------------------------------------------------

def _safe_version(import_name: str, attr: str = "__version__") -> Optional[str]:
    """Return module's version string or None if module / attr is missing."""
    try:
        mod = __import__(import_name)
        return str(getattr(mod, attr, None))
    except Exception:
        return None


def _build_provenance() -> dict:
    """Snapshot of the env that produced this tuning. Recorded on save.
    Readers can compare this dict to the live env and invalidate if
    something material changed (CUDA driver bump, cupy upgrade, etc.).
    """
    prov: dict[str, object] = {
        "python_version": "%d.%d" % __import__("sys").version_info[:2],
        "numpy_version": _safe_version("numpy"),
        "numba_version": _safe_version("numba"),
        "cupy_version": _safe_version("cupy"),
    }
    # CUDA runtime + driver (if cupy is importable).
    try:
        import cupy as cp  # type: ignore
        try:
            prov["cuda_runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
        except Exception as e:  # nosec B110 - best-effort provenance field; a missing/failing CUDA runtime query must not break cache save, provenance dict just omits the field
            logger.debug("Could not read cuda_runtime_version for provenance: %s", e)
            pass
        try:
            prov["cuda_driver_version"] = int(cp.cuda.runtime.driverGetVersion())
        except Exception as e:  # nosec B110 - best-effort provenance field; a missing/failing CUDA driver query must not break cache save, provenance dict just omits the field
            logger.debug("Could not read cuda_driver_version for provenance: %s", e)
            pass
    except ImportError:
        pass
    # Live GPU capability summary (cc, vram, name) at save time.
    try:
        summary = _gpu_summary_cached(_current_device_id())
        if summary is not None:
            prov["gpu_summary"] = {
                "cc_major": summary.get("cc_major"),
                "cc_minor": summary.get("cc_minor"),
                "name": summary.get("name"),
                "total_vram_gb": summary.get("total_vram_gb"),
                "sm_count": summary.get("sm_count"),
            }
    except Exception as e:  # nosec B110 - best-effort provenance enrichment (GPU cc/vram/name summary); failure must not block cache save, provenance dict just omits gpu_summary
        logger.debug("Could not build gpu_summary for provenance: %s", e)
        pass
    return prov


def provenance_changed(old: Optional[dict], new: Optional[dict]) -> bool:
    """True iff a MATERIAL provenance field differs (cuda driver/runtime,
    cupy/numba/numpy versions, GPU cc/name, or python MAJOR.MINOR). python
    major.minor IS material: numba/cupy codegen can differ across interpreter
    minors, so a Python upgrade should re-tune deterministically (cheap +
    correct) rather than let codegen drift invalidate unpredictably. Patch
    bumps are NOT material (python_version stores only major.minor)."""
    if old is None or new is None:
        return False  # be conservative: no data -> no invalidation
    keys = ("cuda_driver_version", "cuda_runtime_version", "cupy_version", "numba_version", "numpy_version", "python_version")
    for k in keys:
        if old.get(k) != new.get(k):
            return True
    old_gpu = old.get("gpu_summary") or {}
    new_gpu = new.get("gpu_summary") or {}
    for k in ("cc_major", "cc_minor", "name"):
        if old_gpu.get(k) != new_gpu.get(k):
            return True
    return False


# ---------------------------------------------------------------------------
# GPU-busy gate for async sweeps
# ---------------------------------------------------------------------------

# An async (fit-time) sweep that runs while the GPU is busy is doubly wrong: it
# (a) contends with the caller's own fit (measured ~18% wall tax on a 100k MRMR
# fit -- 151s vs 124s) and (b) records CONTENDED kernel timings as this host's
# "optimum", which then mis-route every future dispatch. Auto-tuning is only
# valid on an idle GPU, so we defer the sweep when the GPU is loaded and let the
# offline CLI (or a later idle process) tune instead. The shipped default cache
# covers correctness in the meantime. GPUtil.getGPUs() costs ~0.1-2s (nvidia-smi),
# so cache the verdict process-wide for a short TTL -- a burst of sweep spawns at
# fit start then shares ONE poll instead of one per kernel.
# Seconds to wait after deciding a sweep is needed before actually starting it: lets the triggering
# fit get past its bursty start (kernel launches, H2D) so the busy-check below sees the real load, and
# avoids stealing the device the instant the caller needs it. Env-overridable; 0 disables the delay.
def _async_sweep_start_delay() -> float:
    """Seconds to wait after deciding an async sweep is needed before starting it (env ``PYUTILZ_KERNEL_SWEEP_START_DELAY``, default 10s)."""
    try:
        return max(0.0, float(os.environ.get("PYUTILZ_KERNEL_SWEEP_START_DELAY", "10.0")))
    except ValueError:
        return 10.0


def _async_sweep_idle_max_wait() -> float:
    """Max seconds the async sweep waits for the hardware to go idle before sweeping ANYWAY (rather than
    abandoning and leaving the per-host cache empty forever). Env PYUTILZ_KERNEL_SWEEP_IDLE_MAX_WAIT,
    default 120s. 0 -> proceed immediately (no idle wait)."""
    try:
        return max(0.0, float(os.environ.get("PYUTILZ_KERNEL_SWEEP_IDLE_MAX_WAIT", "120.0")))
    except ValueError:
        return 120.0


def _async_sweep_hw_busy() -> bool:
    """True iff the CPU or GPU is busy enough that an async sweep should defer (so we only ever
    benchmark on idle hardware -- a contended sweep both taxes the caller and records contended
    timings as this host's optimum). Delegates to the shared ``benchmark.hardware_busy`` (CPU via
    psutil, GPU via GPUtil, threshold ``PYUTILZ_KERNEL_SWEEP_HW_BUSY`` default 0.40). Never trips on
    hardware it cannot measure (no psutil / no GPU)."""
    try:
        from ..benchmark import hardware_busy
        return hardware_busy()
    except Exception:
        return False

# Process-scoped "tuned this run" guard, keyed on (kernel_name, cache_path), so
# get_or_tune sweeps at most once per kernel per process (tests that switch
# PYUTILZ_KERNEL_CACHE_DIR get a different path -> a fresh re-tune).
_TUNED_THIS_PROCESS: set = set()

# Guards the check-then-add on _TUNED_THIS_PROCESS in get_or_tune(): without it, two threads
# racing for the same (kernel, cache-path) can both observe "not yet tuned" and both spawn a
# redundant async sweep thread (or both proceed into the synchronous sweep-claim path).
_tuned_guard_lock = threading.Lock()

# Process-scoped "already logged the invalidation banner for this kernel" guard, keyed
# exactly like _TUNED_THIS_PROCESS on (kernel_name, cache_path). get_or_tune re-evaluates
# code_version staleness on EVERY call (so a fresh entry an async sweep lands mid-process is
# picked up immediately -- once_per_process only gates the SWEEP), but while a kernel stays
# stale (e.g. a no-op tuner that never persists a fresh entry) that would otherwise fire the
# INFO "invalidated...will re-tune" log on every single call. Log it at most once per kernel
# per process instead; the staleness re-check itself stays unconditional (a cheap dict lookup).
_INVALIDATION_LOGGED_THIS_PROCESS: set = set()
