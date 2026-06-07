"""Persistent per-host kernel-tuning cache for GPU/CPU dispatchers.

When a project ships multiple implementations of a hot numerical kernel
(global-atomic vs shared-mem CUDA RawKernels, numba.cuda vs cupy, plain
numpy vs njit-prange), the "best" choice depends on the live hardware.
Hardcoded thresholds in source code stop being correct as soon as the
package runs on a different GPU or CPU.

This module gives projects a persistent, per-host JSON cache of the
empirically-measured best ``(kernel_variant, block_size)`` (or any other
discrete decision) per ``(input_size_axis_1, input_size_axis_2, ...)``.

Generic API
-----------

* :func:`hw_fingerprint` -- stable per-host key (CPU model + GPU name + cc).
* :class:`KernelTuningCache` -- read / write / lookup; multiple kernels per
  host file, schema-versioned JSON.
* :func:`cache_path` / :func:`cache_dir` -- file-system layout.

Project-specific pieces (auto-tune sweep, kernel registration) stay in
the project. This module provides only the storage + lookup primitives.

Concrete consumer: ``mlframe.feature_selection._benchmarks.kernel_tuning_cache``
houses the sweep for ``joint_hist_batched`` (CUDA RawKernels in
``mlframe.feature_selection.filters.gpu``).

Schema-v1 JSON layout (one file per ``hw_fingerprint``)::

    {
      "schema_version": 1,
      "hw_fingerprint": "cpu_<...>_gpu_<...>_cc<M.m>",
      "timestamp_utc": "2026-05-19T18:14:00Z",
      "kernels": {
        "<kernel_name>": {
          "axes": ["axis_1_name", "axis_2_name", ...],
          "regions": [
            {"axis_1_max": 200000, "axis_2_max": 25,
             "variant": "shared", "block_size": 256, "wall_ms": 0.21},
            ...,
            {"axis_1_max": null, "axis_2_max": null,
             "variant": "shared", "block_size": 512}   # catch-all
          ]
        }
      }
    }

The fields inside each region are free-form (the cache stores whatever
dict the project's tuner emits) -- only ``axis_N_max`` keys are
interpreted by the lookup. Catch-all = all ``..._max`` keys absent or
None.

Honours ``$PYUTILZ_KERNEL_CACHE_DIR`` (env override) > default
``~/.pyutilz/kernel_tuning/``. Atomic writes; corrupt / version-
mismatched files trigger a re-tune (safe degradation).

Storage model (v3, immutable per-(host,kernel,code_version) files)
------------------------------------------------------------------

The legacy v1/v2 model stored ALL kernels for a host in one mutable JSON
(``<fp>.json``) and serialised every writer with a ``filelock`` around a
read-modify-write. That single mutable document was the root cause of the
lost-update / stale-lock / 900s-wedge defects (see the locking critique). v3
dissolves them by making each tuning an **immutable** file::

    <cache_dir>/<hw_fingerprint>/<kernel_slug>/<code_version>.<salt>.<pid>.<ts>.json

* **WRITE** -- a tuner writes a brand-new uniquely-named file (tempfile +
  ``os.replace``); no file is ever modified in place, so there is no
  read-modify-write and no lost update, hence NO write lock.
* **READ** -- ``lookup`` resolves a kernel by globbing its directory and
  picking the newest file by ``tuned_utc`` (falling back to mtime); pure read,
  NO lock. The parse is cached in ``self._loaded`` exactly as before so the hot
  path stays in-memory after the first resolution.
* **SINGLETON-WITHOUT-BLOCKING** -- to start a sweep, a process atomically
  creates ``<kernel_dir>/<code_version>.INPROGRESS`` via
  ``os.open(O_CREAT|O_EXCL)``. Win = own the sweep; ``EEXIST`` = someone else
  is sweeping -> give up immediately (no filelock, no timeout). The marker
  embeds ``pid`` + ``start_ts``; a would-be sweeper STEALS a stale marker
  (dead owner pid OR start_ts older than the max-sweep budget), so a crashed
  sweeper never wedges anyone (self-healing).
* **MIGRATION** -- on first access the legacy monolithic ``<fp>.json`` is split
  into per-kernel immutable files once (under an ``O_EXCL`` claim) and then
  renamed aside; existing caches keep working.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import errno
import glob as _glob
import json
import logging
import os
import random
import re
import threading
import time
from functools import lru_cache
from typing import Callable, Optional, Protocol

from pyutilz.core.pythonlib import is_cuda_available
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


# ---------------------------------------------------------------------------
# HW fingerprint
# ---------------------------------------------------------------------------

def _slug(s: str, maxlen: int = 40) -> str:
    """Filename-safe lowercase + truncated form of an arbitrary string."""
    s = re.sub(r"\(R\)|\(TM\)|\bCPU\b|\bGPU\b|@.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s.strip("-._").lower()[:maxlen] or "unknown"


@lru_cache(maxsize=1)
def _cpu_model_slug() -> str:
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
    return gpu_capability_summary(device_id)


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
        with open(path, "r", encoding="utf-8") as f:
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
    cpu = _cpu_model_slug()
    gpu, cc = _gpu_slug_and_cc()
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
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                # ERROR_INVALID_PARAMETER (87) => no such pid (dead). Any other failure
                # (e.g. ERROR_ACCESS_DENIED 5 => alive but not ours) => assume alive.
                return ctypes.get_last_error() not in (87,)
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
        except Exception:
            pass
        try:
            prov["cuda_driver_version"] = int(cp.cuda.runtime.driverGetVersion())
        except Exception:
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
    except Exception:
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
    keys = ("cuda_driver_version", "cuda_runtime_version", "cupy_version",
            "numba_version", "numpy_version", "python_version")
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
# Cache class
# ---------------------------------------------------------------------------

# Process-wide singleton for load_or_create() -- hot-path dispatch callers
# (GPU/numba kernels) consult the cache on every invocation; one shared
# instance avoids re-reading the per-host JSON each call.
_DEFAULT_INSTANCE = None
_DEFAULT_INSTANCE_LOCK = threading.Lock()

# Project-shipped, anonymized (hw-agnostic) DEFAULT tunings: a read-only cache consulted on a local per-host MISS,
# BEFORE the hand-specified fallback. Measurement-derived (averaged to abstract cpu/gpu, committed to the project
# repo), so a fresh host gets near-optimal dispatch immediately while its own async sweep runs. The LOCAL measured
# cache ALWAYS wins (it is checked first in get_or_tune); this is only the fallback layer. None until a project
# registers its defaults file via register_default_cache().
_DEFAULT_CACHE = None


def register_default_cache(path: str) -> bool:
    """Register a project's anonymized default-tuning JSON (committed to the project repo). On a local per-host
    cache miss, ``get_or_tune`` consults it before the hand-specified fallback (and still kicks the async sweep to
    measure THIS host). Returns whether it loaded. A missing/unreadable file degrades silently to no defaults."""
    global _DEFAULT_CACHE
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        c = KernelTuningCache(in_memory=True)
        with c._lock:
            c._loaded = {"schema_version": payload.get("schema_version", SCHEMA_VERSION),
                         "kernels": payload.get("kernels", {})}
        _DEFAULT_CACHE = c
        logger.debug("kernel_tuning_cache: registered %d default kernels from %s", len(payload.get("kernels", {})), path)
        return True
    except FileNotFoundError:
        logger.debug("kernel_tuning_cache: no default cache at %s", path)
        _DEFAULT_CACHE = None
        return False
    except Exception as e:
        logger.warning("kernel_tuning_cache: could not load default cache %s: %s", path, e)
        _DEFAULT_CACHE = None
        return False


class KernelTuningCache:
    """Per-host kernel-tuning cache. Single instance per process is enough;
    the class is thread-safe under a single ``threading.Lock``.

    Typical usage::

        cache = KernelTuningCache()
        choice = cache.lookup("joint_hist_batched", n_samples=1_000_000, joint_size=25)
        if choice is None:
            regions = run_my_project_sweep()
            cache.update("joint_hist_batched",
                         axes=["n_samples", "joint_size"], regions=regions)
            choice = cache.lookup("joint_hist_batched", n_samples=1_000_000, joint_size=25)

    On cache miss ``lookup`` returns ``None`` -- caller decides whether to
    trigger an auto-tune sweep or fall back to a hand-tuned default.
    """

    def __init__(self, path: Optional[str] = None, *, in_memory: bool = False):
        # in_memory=True skips all disk / provenance -- a seedable RAM cache for
        # fast unit tests (KernelTuningCache(in_memory=True)).
        self._in_memory = in_memory
        # ``self._path`` is the per-host DIRECTORY holding immutable per-kernel
        # files (v3), NOT a single JSON file (v1/v2). ``path`` (legacy positional
        # arg) is interpreted as that directory if a caller passes one. The
        # once-per-process guard + sweep markers key off this path, so it stays a
        # stable per-cache identity. None for in-memory caches.
        self._path = None if in_memory else (path or host_cache_dir())
        # The kernels whose legacy monolith has already been migrated this
        # process (one-time split per host dir).
        self._migrated = False
        # Reentrant so ``update`` can call ``_ensure_loaded`` under its own
        # lock without deadlocking (regression caught by
        # tests/system/test_kernel_tuning_cache.py during the first
        # implementation pass).
        self._lock = threading.RLock()
        self._loaded: Optional[dict] = None  # None until first load() call
        # Optional shared remote store (PYUTILZ_KERNEL_REMOTE=s3://...). None
        # for in-memory caches or when no remote is configured -> local-only.
        if in_memory:
            self._remote = None
        else:
            from .remote import get_remote_backend
            self._remote = get_remote_backend()

    @classmethod
    def load_or_create(cls) -> "KernelTuningCache":
        """Return the process-wide singleton cache (load-or-create once).

        Hot-path dispatch callers (GPU/numba kernels) consult the cache on
        every kernel invocation; constructing a fresh instance per call would
        re-read the per-host JSON each time. The first call builds and caches
        the singleton (the file is lazy-loaded on first ``lookup``); later
        calls return the same object. Thread-safe (double-checked lock).

        On an un-tuned host the cache is empty, so ``lookup`` returns ``None``
        and callers fall back to their hand-tuned default -- identical to the
        prior behavior, but now via a real miss instead of a swallowed
        ``AttributeError`` (the method did not previously exist).
        """
        global _DEFAULT_INSTANCE
        if _DEFAULT_INSTANCE is None:
            with _DEFAULT_INSTANCE_LOCK:
                if _DEFAULT_INSTANCE is None:
                    _DEFAULT_INSTANCE = cls()
        return _DEFAULT_INSTANCE

    # ----- I/O (v3 immutable per-kernel files) -----

    @staticmethod
    def _atomic_write_json(final_path: str, payload: dict, *, retries: int = 3, backoff: float = 0.05) -> bool:
        """Write ``payload`` to ``final_path`` via tempfile + ``os.replace``,
        with a bounded retry around the rename (D3: Windows AV / share-delete
        without FILE_SHARE_DELETE can make ``os.replace`` transiently raise
        ``OSError`` errno 5/13). Returns True on success, False on persistent
        failure (degrades silently -- a failed cache write must never break
        dispatch). Each attempt writes a FRESH uniquely-named temp file so a
        concurrent writer never collides on the temp name."""
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        last_err: Optional[Exception] = None
        for attempt in range(max(1, retries)):
            tmp = f"{final_path}.{os.getpid()}.{random.randrange(1 << 30):x}.tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, sort_keys=True)
                os.replace(tmp, final_path)
                return True
            except OSError as e:
                last_err = e
                with contextlib.suppress(OSError):
                    if os.path.exists(tmp):
                        os.remove(tmp)
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
        logger.debug("kernel_tuning_cache: atomic write of %s failed after %d tries: %s",
                     final_path, retries, last_err)
        return False

    def _migrate_legacy(self) -> None:
        """One-time split of a legacy monolithic ``<fp>.json`` into per-kernel
        immutable files, under an ``O_EXCL`` claim so only one process migrates.
        After a successful split the monolith is renamed aside (``.migrated``) so
        it is found-once but never re-read. Backward compatible: existing caches
        keep working transparently. Idempotent + crash-safe (a partial migration
        just re-runs; immutable writes can't corrupt)."""
        if self._in_memory or self._path is None or self._migrated:
            return
        self._migrated = True  # at most one attempt per process
        legacy = cache_path()
        if not os.path.isfile(legacy):
            return
        claim = os.path.join(self._path, ".migrate.INPROGRESS")
        try:
            os.makedirs(self._path, exist_ok=True)
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return  # another process is migrating; its result will be read on next _load
        except OSError as e:
            logger.debug("kernel_tuning_cache: migration claim failed: %s", e)
            return
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Only import kernels from a SCHEMA- and FINGERPRINT-compatible
            # monolith (v2 or v3 shape, matching host). A schema-999 / foreign /
            # corrupt monolith is treated exactly as the old _load did -- as a
            # miss -- so it is renamed aside WITHOUT importing anything (no
            # accidental resurrection of an invalid cache).
            compatible = (data.get("schema_version") in (SCHEMA_VERSION, 2)
                          and data.get("hw_fingerprint") == hw_fingerprint())
            kernels = (data.get("kernels", {}) or {}) if compatible else {}
            prov = data.get("provenance")
            for name, entry in kernels.items():
                if not isinstance(entry, dict):
                    continue
                # Carry provenance into each split file so the staleness check
                # behaves exactly as it did against the monolith.
                self._persist_kernel(name, dict(entry), provenance=prov, remote=False)
            os.replace(legacy, legacy + ".migrated")
            logger.info("kernel_tuning_cache: migrated %d kernels from legacy %s (compatible=%s)",
                        len(kernels), legacy, compatible)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("kernel_tuning_cache: legacy migration failed: %s", e)
        finally:
            with contextlib.suppress(OSError):
                os.remove(claim)

    def _read_kernel_newest(self, kernel_name: str) -> Optional[dict]:
        """Resolve one kernel by globbing its directory and picking the NEWEST
        immutable file (by embedded ``tuned_utc``, mtime as tiebreaker). Pure
        read, NO lock. Returns the kernel ENTRY dict (axes/regions/code_version/
        salt/tuned_utc) with a per-file provenance staleness check applied, or
        None on miss / stale."""
        kdir = _kernel_dir(self._path, kernel_name)
        files = [p for p in _glob.glob(os.path.join(kdir, "*.json"))]
        if not files:
            return None
        live_prov = _build_provenance()
        candidates: list[tuple] = []  # (tuned_ts, mtime, entry)
        for p in files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue  # os.replace is atomic; a parse failure is a foreign/partial file -> skip
            if rec.get("schema_version") != SCHEMA_VERSION:
                continue
            if rec.get("hw_fingerprint") != hw_fingerprint():
                continue
            saved_prov = rec.get("provenance")
            if saved_prov and provenance_changed(saved_prov, live_prov):
                continue  # driver/cupy/numba bump since this tuning -> ignore (structural staleness)
            entry = rec.get("entry")
            if not isinstance(entry, dict):
                continue
            ts = entry.get("tuned_utc") or ""
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = 0.0
            candidates.append((ts, mtime, entry))
        if not candidates:
            return None
        # Newest by tuned_utc (ISO-8601 strings sort chronologically), then mtime.
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[-1][2]

    def _load(self) -> Optional[dict]:
        """Build the in-memory ``{schema_version, hw_fingerprint, kernels}`` view
        by resolving every kernel directory to its newest immutable file. Pure
        read, NO lock. On an empty local store, reads through to the shared remote
        (if any) and caches the pulled kernels locally as immutable files before
        resolving. Returns None when nothing is found (so ``_ensure_loaded``
        installs an empty stub)."""
        if self._in_memory:
            return None
        self._migrate_legacy()
        host_dir = self._path
        kernels: dict = {}
        if os.path.isdir(host_dir):
            try:
                kernel_dirs = [d for d in os.scandir(host_dir) if d.is_dir()]
            except OSError:
                kernel_dirs = []
            for d in kernel_dirs:
                # Recover the kernel name from any file in the dir (the slug is
                # lossy); the entry itself carries the canonical kernel_name.
                entry = self._read_kernel_dir_by_path(d.path)
                if entry is not None:
                    kernels[entry[0]] = entry[1]
        if not kernels and self._remote is not None:
            # Read-through: pull this host's payload from the shared store + cache
            # each kernel as an immutable local file, then they resolve normally.
            try:
                remote_data = self._remote.read(hw_fingerprint())
            except Exception as e:
                logger.debug("kernel_tuning_cache: remote read failed: %s", e)
                remote_data = None
            if remote_data and remote_data.get("schema_version") in (SCHEMA_VERSION, 2):
                prov = remote_data.get("provenance")
                live_prov = _build_provenance()
                if not (prov and provenance_changed(prov, live_prov)):
                    for name, entry in (remote_data.get("kernels", {}) or {}).items():
                        if isinstance(entry, dict):
                            self._persist_kernel(name, dict(entry), provenance=prov, remote=False)
                            kernels[name] = entry
        if not kernels:
            return None
        return {
            "schema_version": SCHEMA_VERSION,
            "hw_fingerprint": hw_fingerprint(),
            "kernels": kernels,
        }

    def _read_kernel_dir_by_path(self, kdir: str) -> Optional[tuple]:
        """Like ``_read_kernel_newest`` but takes a directory path + returns
        ``(kernel_name, entry)`` (the name is read from the winning record).
        Used by ``_load`` which scans directories without knowing kernel names."""
        files = _glob.glob(os.path.join(kdir, "*.json"))
        if not files:
            return None
        live_prov = _build_provenance()
        candidates: list[tuple] = []
        for p in files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get("schema_version") != SCHEMA_VERSION:
                continue
            if rec.get("hw_fingerprint") != hw_fingerprint():
                continue
            saved_prov = rec.get("provenance")
            if saved_prov and provenance_changed(saved_prov, live_prov):
                continue
            entry = rec.get("entry")
            name = rec.get("kernel_name")
            if not isinstance(entry, dict) or not name:
                continue
            ts = entry.get("tuned_utc") or ""
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = 0.0
            candidates.append((ts, mtime, name, entry))
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c[0], c[1]))
        winner = candidates[-1]
        return (winner[2], winner[3])

    def _persist_kernel(self, kernel_name: str, entry: dict, *,
                        provenance: Optional[dict] = None, remote: bool = True) -> None:
        """Write ONE immutable per-kernel tuning file (no read-modify-write, no
        lock). Filename: ``<code_version>.<salt>.<pid>.<ts>.<rand>.json`` so every
        write is unique and prior tunings are never overwritten (a reader picks
        the newest). Best-effort remote write-through happens AFTER the local
        write and OUTSIDE any lock (D9), so a hung S3 never stalls the local save.
        """
        if self._in_memory or self._path is None:
            return
        cv = entry.get("code_version") or _NO_CODE_VERSION
        salt = entry.get("salt", 0)
        ts = time.time()
        fname = f"{_slug(str(cv), maxlen=70)}.{int(salt)}.{os.getpid()}.{int(ts * 1000)}.{random.randrange(1 << 24):x}.json"
        kdir = _kernel_dir(self._path, kernel_name)
        final_path = os.path.join(kdir, fname)
        record = {
            "schema_version": SCHEMA_VERSION,
            "hw_fingerprint": hw_fingerprint(),
            "kernel_name": kernel_name,
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "provenance": provenance if provenance is not None else _build_provenance(),
            "entry": entry,
        }
        if self._atomic_write_json(final_path, record):
            logger.info("kernel_tuning_cache: saved %s", final_path)
            self._gc_kernel_dir(kdir)
        # Remote write-through OUTSIDE any lock (D9): one immutable object per
        # (fp, kernel) -- last writer wins remotely, race-free, fire-and-forget.
        if remote and self._remote is not None:
            try:
                self._remote.write(hw_fingerprint(), self._remote_payload())
            except Exception as e:
                logger.debug("kernel_tuning_cache: remote write failed: %s", e)

    def _remote_payload(self) -> dict:
        """Assemble the legacy-shaped monolithic payload (all kernels) for the
        remote store, so the remote object stays one-per-fingerprint and a peer's
        read-through repopulates every kernel."""
        kernels = (self._loaded or {}).get("kernels", {}) if self._loaded else {}
        return {
            "schema_version": SCHEMA_VERSION,
            "hw_fingerprint": hw_fingerprint(),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "provenance": _build_provenance(),
            "kernels": dict(kernels),
        }

    def _gc_kernel_dir(self, kdir: str, keep: int = 4) -> None:
        """Lazily garbage-collect a kernel directory, keeping the newest ``keep``
        immutable files (by mtime). Negligible space + never blocks; best-effort
        (a failed unlink is harmless -- the reader always picks the newest)."""
        try:
            files = [p for p in _glob.glob(os.path.join(kdir, "*.json"))]
        except OSError:
            return
        if len(files) <= keep:
            return
        try:
            files.sort(key=lambda p: os.path.getmtime(p))
        except OSError:
            return
        for p in files[:-keep]:
            with contextlib.suppress(OSError):
                os.remove(p)

    def _ensure_loaded(self) -> dict:
        """Return the live cache payload (loaded once per process). On
        miss returns an empty stub so subsequent updates can populate it."""
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            self._loaded = self._load() or {
                "schema_version": SCHEMA_VERSION,
                "hw_fingerprint": "in_memory" if self._in_memory else hw_fingerprint(),
                "kernels": {},
            }
            return self._loaded

    # ----- public API -----

    def update(self, kernel_name: str, *, axes: list[str], regions: list[dict],
               code_version: Optional[str] = None, salt: int = 0,
               equiv_tol: Optional[float] = None, tuned_utc: Optional[str] = None,
               hooks: "Optional[TuningHooks]" = None) -> None:
        """Replace ``kernels[kernel_name]`` with the given axes + regions and
        persist (other kernels preserved). Records ``code_version``/``salt``/
        ``tuned_utc`` metadata. If ``equiv_tol`` is given, any region whose
        recorded ``max_abs_diff`` exceeds it is REJECTED (dropped) with a loud
        warning -- a faster-but-numerically-divergent variant is a bug, never a
        winner; regions without a recorded ``max_abs_diff`` are unvalidated and
        kept as-is."""
        regions = list(regions)
        if equiv_tol is not None:
            regions = self._apply_equiv_gate(kernel_name, regions, equiv_tol, hooks)
        entry: dict = {
            "axes": list(axes),
            "regions": regions,
            "tuned_utc": tuned_utc or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
        if code_version is not None:
            entry["code_version"] = code_version
        if salt:
            entry["salt"] = int(salt)
        with self._lock:
            self._ensure_loaded()
            # Update the in-memory snapshot, then persist ONLY this one kernel as
            # a new immutable file. No read-modify-write of a shared document ->
            # no lost update (D1 dissolved): a concurrent writer of a DIFFERENT
            # kernel writes a different directory, and a concurrent writer of the
            # SAME kernel writes a distinct file (newest wins), never clobbering.
            self._loaded["kernels"][kernel_name] = entry
            if hooks is not None:
                hooks.persist(kernel_name, self._path, len(regions))
            self._persist_kernel(kernel_name, entry)

    def has(self, kernel_name: str) -> bool:
        """True iff a tuning for ``kernel_name`` is present on disk."""
        data = self._ensure_loaded()
        return bool(data.get("kernels", {}).get(kernel_name, {}).get("regions"))

    def get_regions(self, kernel_name: str) -> Optional[list[dict]]:
        """Raw region list for ``kernel_name`` or None on miss."""
        data = self._ensure_loaded()
        entry = data.get("kernels", {}).get(kernel_name)
        if entry and entry.get("regions"):
            return list(entry["regions"])
        return None

    def lookup(self, kernel_name: str, **dims) -> Optional[dict]:
        """First region whose axis constraints all match the requested ``dims``.
        Constraints: ``<axis>_max`` (dim <= max), ``<axis>_min`` (dim >= min),
        ``<axis>_eq`` (dim == value, categorical -- e.g. dtype/ndim/location);
        absent/None = unconstrained. Returns the region dict minus all constraint
        keys. None on cache miss.

        Example::

            cache.lookup("joint_hist_batched", n_samples=1_000_000, joint_size=25)
            # -> {"variant": "shared", "block_size": 512, "wall_ms": 0.78}
        """
        entry = self._ensure_loaded().get("kernels", {}).get(kernel_name)
        if not entry:
            return None
        regions = entry.get("regions") or []
        # Strip ONLY the declared axes' constraint keys (<axis>_max/_min/_eq), not
        # arbitrary payload fields that happen to end in such a suffix (e.g. a
        # ``block_size_max`` decision field) -- B4: suffix-based strip is a
        # namespace collision; drive it from the entry's declared ``axes``.
        constraint_keys = {f"{ax}{suf}" for ax in (entry.get("axes") or []) for suf in _AXIS_SUFFIXES}
        for region in regions:
            if _region_matches(region, dims):
                return {k: v for k, v in region.items() if k not in constraint_keys}
        return None

    def reset(self) -> None:
        """Drop the in-memory cache; next read reloads from disk. For tests
        + on driver-update invalidation hooks."""
        with self._lock:
            self._loaded = None

    # ----- metadata / eviction (NO auto-evict on read) -----

    def get_metadata(self, kernel_name: str) -> Optional[dict]:
        """``{code_version, salt, tuned_utc, age_seconds, n_regions}`` for a
        tuned kernel, else None. ``age_seconds`` = wall-age since the tuning."""
        data = self._ensure_loaded()
        entry = data.get("kernels", {}).get(kernel_name)
        if not entry:
            return None
        meta = {
            "code_version": entry.get("code_version"),
            "salt": entry.get("salt", 0),
            "tuned_utc": entry.get("tuned_utc"),
            "n_regions": len(entry.get("regions", [])),
            "age_seconds": None,
        }
        ts = entry.get("tuned_utc")
        if ts:
            try:
                t = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                meta["age_seconds"] = max(0.0, time.time() - t.timestamp())
            except (ValueError, AttributeError):
                pass
        return meta

    def evict(self, kernel_name: str) -> bool:
        """Drop one kernel's tuning + persist. Returns whether it was present.
        Deliberately NO auto-evict-on-read: a tuning is permanent for a given
        hw_fingerprint + provenance + code_version."""
        with self._lock:
            self._ensure_loaded()
            # Clear the once-per-process sweep guard so a subsequent get_or_tune
            # can actually re-tune this kernel (B11) instead of short-circuiting
            # to the fallback because "we already swept it this process".
            _TUNED_THIS_PROCESS.discard((kernel_name, self._path or id(self)))
            present = kernel_name in self._loaded.get("kernels", {})
            if present:
                del self._loaded["kernels"][kernel_name]
            self._delete_kernel_files(kernel_name)
            if present:
                # Refresh the remote object so the eviction propagates (best-effort).
                if not self._in_memory and self._remote is not None:
                    try:
                        self._remote.write(hw_fingerprint(), self._remote_payload())
                    except Exception as e:
                        logger.debug("kernel_tuning_cache: remote write failed: %s", e)
                return True
            return False

    def _delete_kernel_files(self, kernel_name: str) -> None:
        """Remove all immutable files for a kernel on disk (used by evict). No-op
        for in-memory caches. Best-effort; a failed unlink degrades to a stale
        file the newest-wins reader still resolves correctly after a re-tune."""
        if self._in_memory or self._path is None:
            return
        kdir = _kernel_dir(self._path, kernel_name)
        for p in _glob.glob(os.path.join(kdir, "*.json")):
            with contextlib.suppress(OSError):
                os.remove(p)

    # ----- introspection -----

    def lookup_explain(self, kernel_name: str, **dims) -> dict:
        """Like ``lookup`` but explains the decision -- returns
        ``{matched, region_index, region, reason}``. For tests + debugging which
        region (and why) a dispatch resolved to."""
        entry = self._ensure_loaded().get("kernels", {}).get(kernel_name)
        regions = (entry or {}).get("regions") or []
        if not regions:
            return {"matched": False, "region_index": None, "region": None,
                    "reason": f"no regions for kernel {kernel_name!r}"}
        constraint_keys = {f"{ax}{suf}" for ax in (entry.get("axes") or []) for suf in _AXIS_SUFFIXES}
        first_reason = None
        for i, region in enumerate(regions):
            ok, why = _region_match_reason(region, dims)
            if ok:
                payload = {k: v for k, v in region.items() if k not in constraint_keys}
                return {"matched": True, "region_index": i, "region": payload,
                        "reason": f"region {i} matched"}
            if first_reason is None:
                first_reason = f"region 0 rejected: {why}"
        return {"matched": False, "region_index": None, "region": None,
                "reason": first_reason or f"no region matched dims {dims}"}

    # ----- equiv-tol gate (used by update) -----

    def _apply_equiv_gate(self, kernel_name, regions, equiv_tol, hooks=None):
        """Drop (with a loud warning) any region whose recorded ``max_abs_diff``
        exceeds ``equiv_tol`` -- SURFACE divergence, never silently substitute a
        reference. Regions without ``max_abs_diff`` are unvalidated + kept."""
        kept = []
        for r in regions:
            d = r.get("max_abs_diff")
            if d is not None and d > equiv_tol:
                caps = {k: v for k, v in r.items() if k.endswith(_AXIS_SUFFIXES)}
                logger.warning(
                    "kernel_tuning_cache: %s region %s REJECTED -- max_abs_diff=%.3e > "
                    "equiv_tol=%.3e. A faster-but-divergent variant is a bug, not a winner; "
                    "not persisting it.", kernel_name, caps, d, equiv_tol)
                if hooks is not None:
                    hooks.winner_chosen(kernel_name, r, f"rejected: diverges {d:.3e}>{equiv_tol:.3e}")
                continue
            kept.append(r)
        return kept

    # ----- orchestration: env -> code_version-checked lookup -> locked sweep -> fallback -----

    def get_or_tune(self, kernel_name: str, *, dims: dict, tuner: Callable,
                    axes: list[str], fallback, env_key: Optional[str] = None,
                    code_version: Optional[str] = None, salt: int = 0,
                    equiv_tol: Optional[float] = None,
                    hooks: "Optional[TuningHooks]" = None,
                    once_per_process: bool = True, lock_timeout: float = 900.0,
                    async_sweep: bool = False):
        """Unified dispatch collapsing the env -> lookup -> on-miss sweep ->
        persist -> re-lookup -> fallback flow.

        ``tuner``: zero-arg callable returning a region list (the project sweep).
        ``fallback``: a value or zero-arg callable (called lazily).
        ``env_key``: if set and the env var is a non-empty string, short-circuit
        to that raw string. ``code_version``: when both stored + passed are
        non-None and differ, the stale entry is bypassed. The once-per-process
        guard is keyed on (kernel, cache-path) so tests switching
        PYUTILZ_KERNEL_CACHE_DIR re-tune. Sweeps serialize cross-process."""
        hk = hooks if hooks is not None else _DEFAULT_HOOKS

        def _fb():
            # DEFAULT-cache layer: a project-shipped anonymized (hw-agnostic) tuning, consulted on a local per-host
            # MISS BEFORE the hand-specified fallback. Measurement-derived, so better than the heuristic; the async
            # sweep still runs to replace it with THIS host's measured optimum. The local measured cache already
            # took precedence (checked above), so this never overrides a real local result.
            dc = _DEFAULT_CACHE
            if dc is not None and dc is not self:
                try:
                    if not dc._code_version_stale(kernel_name, code_version):
                        d = dc.lookup(kernel_name, **dims)
                        if d is not None:
                            return d
                except Exception:
                    pass
            return fallback() if callable(fallback) else fallback

        if env_key:
            forced = os.environ.get(env_key, "").strip()
            if forced:
                hk.env_override(kernel_name, forced)
                return forced
        if self._code_version_stale(kernel_name, code_version):
            hk.invalidation(kernel_name, "code_version changed")
        else:
            hit = self.lookup(kernel_name, **dims)
            if hit is not None:
                hk.cache_hit(kernel_name, dims, hit)
                return hit
        hk.cache_miss(kernel_name, dims)
        guard_key = (kernel_name, self._path or id(self))
        if once_per_process and guard_key in _TUNED_THIS_PROCESS:
            return _fb()

        _sweep_disabled = os.environ.get("PYUTILZ_KERNEL_DISABLE_SWEEP", "").strip() not in ("", "0", "false", "False")

        # ASYNC sweep (opt-in via async_sweep=True; the FIT-TIME path -- TunerSpec.choose + hot-path dispatchers pass
        # it). The CPU-vs-GPU sweep is EXPENSIVE (hundreds of seconds) and must NEVER run synchronously inside a
        # caller's fit -- doing so blocks the fit, contaminates its timing, and (if the
        # process is killed mid-sweep) can wedge every fresh fit on the cross-process tuning lock. Instead: mark the
        # kernel tuned-this-process, kick the sweep into a background daemon thread, and return the measurement-backed
        # FALLBACK immediately. The sweep measures on THIS host and writes the cache, so SUBSEQUENT get_or_tune calls
        # (this process once the thread finishes, and all future processes) return the per-host-OPTIMAL backend -- the
        # measurement is preserved, just moved off the hot path. The background thread acquires the lock with a SHORT
        # timeout and GIVES UP if another process is already sweeping (no wedge, no duplicate). One-shot scripts get the
        # fallback (which they don't benefit from tuning anyway). ``mlframe-tune-kernels`` / ``retune_all`` pass
        # ``async_sweep=False`` for synchronous, wait-for-result offline tuning.
        if async_sweep and not _sweep_disabled:
            _TUNED_THIS_PROCESS.add(guard_key)
            self._spawn_async_sweep(kernel_name, dims=dims, tuner=tuner, axes=axes,
                                    code_version=code_version, salt=salt, equiv_tol=equiv_tol, hooks=hk)
            hk.winner_chosen(kernel_name, None, "fallback (async sweep dispatched)")
            return _fb()

        # SYNCHRONOUS path: explicit offline tuning (async_sweep=False) or the disable-sweep escape hatch.
        # Claim the sweep via an O_EXCL INPROGRESS marker (no filelock, no blocking, no 900s wedge). Win = own the
        # sweep; lose = another process is already sweeping this (kernel, code_version) -> we DON'T duplicate it, we
        # re-check for a freshly-landed result and otherwise return the fallback (lock_timeout retained for signature
        # stability only -- there is no blocking wait to time out anymore).
        with self._claim_sweep(kernel_name, code_version, hk) as owns:
            # another process may have tuned it while we were resolving the claim
            if not self._code_version_stale(kernel_name, code_version):
                self.reset()
                hit = self.lookup(kernel_name, **dims)
                if hit is not None:
                    return hit
            if not owns:
                hk.winner_chosen(kernel_name, None, "fallback (another process is sweeping)")
                return _fb()
            _TUNED_THIS_PROCESS.add(guard_key)
            regions = None if _sweep_disabled else self._run_tuner(kernel_name, tuner, axes, hk)
            if regions:
                self.update(kernel_name, axes=axes, regions=regions, code_version=code_version,
                            salt=salt, equiv_tol=equiv_tol, hooks=hk)
                hk.sweep_end(kernel_name, len(regions))
                hit = self.lookup(kernel_name, **dims)
                if hit is not None:
                    hk.winner_chosen(kernel_name, hit, "from sweep")
                    return hit
        hk.winner_chosen(kernel_name, None, "fallback")
        return _fb()

    def _run_tuner(self, kernel_name: str, tuner: Callable, axes: list, hooks):
        """Run a project sweep, firing the start hook and swallowing any failure (a sweep error must never
        break dispatch). Returns the region list or None."""
        hooks.sweep_start(kernel_name, axes)
        try:
            return tuner()
        except Exception as e:
            logger.debug("kernel_tuning_cache: tuner for %s failed: %s", kernel_name, e)
            return None

    def _spawn_async_sweep(self, kernel_name: str, *, dims, tuner, axes, code_version, salt, equiv_tol, hooks):
        """Run the sweep in a background daemon thread: measure on this host + write the cache for SUBSEQUENT
        calls, without ever blocking the caller's fit. Claims the sweep via an O_EXCL INPROGRESS marker and gives up
        if another process is already sweeping this kernel -> a crashed/killed sweep can never wedge a fresh fit
        (the stale marker is steal-able), and there is at most one sweep per (kernel, code_version) across processes."""
        def _run():
            try:
                with self._claim_sweep(kernel_name, code_version, hooks) as owns:
                    if not owns:
                        return  # another process is already tuning this kernel; let it
                    self.reset()  # pick up any result a peer landed since we were spawned
                    if not self._code_version_stale(kernel_name, code_version) and self.lookup(kernel_name, **dims) is not None:
                        return  # tuned while we claimed
                    regions = self._run_tuner(kernel_name, tuner, axes, hooks)
                    if regions:
                        self.update(kernel_name, axes=axes, regions=list(regions), code_version=code_version,
                                    salt=salt, equiv_tol=equiv_tol, hooks=hooks)
                        hooks.sweep_end(kernel_name, len(regions))
            except Exception as e:  # a background sweep must never surface
                logger.debug("kernel_tuning_cache: async sweep for %s crashed: %s", kernel_name, e)
        threading.Thread(target=_run, name="ktc-sweep-" + _slug(kernel_name), daemon=True).start()

    def _code_version_stale(self, kernel_name: str, code_version: Optional[str]) -> bool:
        """True iff a stored code_version exists and differs from the live one."""
        if code_version is None:
            return False
        data = self._ensure_loaded()
        stored = data.get("kernels", {}).get(kernel_name, {}).get("code_version")
        return stored is not None and stored != code_version

    def _marker_path(self, kernel_name: str, code_version: Optional[str]) -> str:
        """Path to the per-(kernel, code_version) INPROGRESS sweep marker."""
        cv = _slug(str(code_version or _NO_CODE_VERSION), maxlen=70)
        return os.path.join(_kernel_dir(self._path, kernel_name), f"{cv}.INPROGRESS")

    @contextlib.contextmanager
    def _claim_sweep(self, kernel_name: str, code_version: Optional[str], hooks):
        """Singleton-without-blocking: yield True iff THIS process owns the sweep
        for ``(kernel, code_version)``, False if another live process already does.

        Atomically create the INPROGRESS marker via ``os.open(O_CREAT|O_EXCL)``.
        Win -> own it (marker removed on exit). ``EEXIST`` -> read the marker's
        ``pid`` + ``start_ts``: if the owner pid is dead OR the start_ts is older
        than the max-sweep budget, STEAL it (unlink + recreate) and own; else give
        up (yield False) immediately -- no filelock, no timeout, no 900s wedge. A
        crashed sweeper self-heals after at most one budget window. No-op (yields
        True) for in-memory caches."""
        if self._in_memory or self._path is None:
            yield True
            return
        marker = self._marker_path(kernel_name, code_version)
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        owns = self._try_create_marker(marker)
        if not owns:
            owns = self._maybe_steal_marker(marker, kernel_name, hooks)
        try:
            yield owns
        finally:
            if owns:
                with contextlib.suppress(OSError):
                    os.remove(marker)

    def _try_create_marker(self, marker: str) -> bool:
        """Atomically create the marker (O_EXCL), stamping pid + start_ts.
        Returns True on success, False if it already exists."""
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError as e:
            logger.debug("kernel_tuning_cache: marker create failed (%s); sweeping without claim", e)
            return True  # can't claim -> behave as owner (degrade to no-singleton, never wedge)
        try:
            payload = json.dumps({"pid": os.getpid(), "start_ts": time.time(),
                                  "host": hw_fingerprint()}).encode("utf-8")
            os.write(fd, payload)
        except OSError:
            pass
        finally:
            os.close(fd)
        return True

    def _maybe_steal_marker(self, marker: str, kernel_name: str, hooks) -> bool:
        """An existing marker was found. Steal it (return True) iff the owning pid
        is dead OR start_ts is older than the max-sweep budget; else give up
        (False). Stealing is itself racy-safe: we remove the stale marker and
        re-create via O_EXCL; if a third process beats us to the recreate, we lose
        the claim (return False) -- correct, exactly one sweeper wins."""
        try:
            with open(marker, "r", encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            info = {}
        pid = int(info.get("pid", 0) or 0)
        start_ts = float(info.get("start_ts", 0.0) or 0.0)
        age = time.time() - start_ts if start_ts else float("inf")
        budget = _sweep_budget_seconds()
        same_host = info.get("host") in (None, hw_fingerprint())
        # Only trust the pid-liveness probe for a marker written on THIS host.
        owner_dead = same_host and not _pid_alive(pid)
        if not (owner_dead or age > budget):
            return False  # a live, in-budget sweeper owns it -> give up
        logger.info("kernel_tuning_cache: stealing stale sweep marker for %s "
                    "(pid=%s alive=%s age=%.0fs budget=%.0fs)",
                    kernel_name, pid, not owner_dead, age, budget)
        hooks.concurrent_sweep_detected(kernel_name)
        with contextlib.suppress(OSError):
            os.remove(marker)
        return self._try_create_marker(marker)


# Region keys ending with one of these suffixes are interpreted as axis
# CONSTRAINTS by the matcher; everything else in a region dict is opaque
# decision payload. ``lookup`` strips exactly these suffixes from its return.
_AXIS_SUFFIXES = ("_max", "_min", "_eq")


def _region_matches(region: dict, dims: dict) -> bool:
    """A region matches a dims dict iff, for every requested dim, the region's
    constraints on that axis hold:
      * ``<axis>_max``: dim <= max   (numeric upper cap)
      * ``<axis>_min``: dim >= min   (numeric lower cap)
      * ``<axis>_eq`` : dim == value (categorical / exact -- dtype, ndim, location)
    A constraint key absent or None is unconstrained; a dim with no constraint
    key in the region is ignored (the region applies to any value of it)."""
    for axis_name, axis_value in dims.items():
        cap = region.get(f"{axis_name}_max")
        if cap is not None and axis_value > cap:
            return False
        lo = region.get(f"{axis_name}_min")
        if lo is not None and axis_value < lo:
            return False
        eq = region.get(f"{axis_name}_eq")
        if eq is not None and axis_value != eq:
            return False
    return True


def _region_match_reason(region: dict, dims: dict) -> tuple:
    """Like ``_region_matches`` but returns ``(ok, reason)`` -- the first failing
    constraint -- for ``lookup_explain``."""
    for axis_name, axis_value in dims.items():
        cap = region.get(f"{axis_name}_max")
        if cap is not None and axis_value > cap:
            return False, f"{axis_name}={axis_value} > {axis_name}_max={cap}"
        lo = region.get(f"{axis_name}_min")
        if lo is not None and axis_value < lo:
            return False, f"{axis_name}={axis_value} < {axis_name}_min={lo}"
        eq = region.get(f"{axis_name}_eq")
        if eq is not None and axis_value != eq:
            return False, f"{axis_name}={axis_value!r} != {axis_name}_eq={eq!r}"
    return True, "all constraints satisfied"


# Process-scoped "tuned this run" guard, keyed on (kernel_name, cache_path), so
# get_or_tune sweeps at most once per kernel per process (tests that switch
# PYUTILZ_KERNEL_CACHE_DIR get a different path -> a fresh re-tune).
_TUNED_THIS_PROCESS: set = set()


class TuningHooks(Protocol):
    """Optional instrumentation fired by ``get_or_tune`` / sweeps. Supply a
    custom implementation for structured logging / dashboards; the default
    ``LoggerHooks`` logs at sensible levels."""

    def env_override(self, kernel: str, choice: str) -> None: ...
    def cache_hit(self, kernel: str, dims: dict, region: dict) -> None: ...
    def cache_miss(self, kernel: str, dims: dict) -> None: ...
    def sweep_start(self, kernel: str, axes: list) -> None: ...
    def sweep_end(self, kernel: str, n_regions: int) -> None: ...
    def winner_chosen(self, kernel: str, region: Optional[dict], reason: str) -> None: ...
    def persist(self, kernel: str, path: Optional[str], n_regions: int) -> None: ...
    def invalidation(self, kernel: str, reason: str) -> None: ...
    def concurrent_sweep_detected(self, kernel: str) -> None: ...


class LoggerHooks:
    """Default ``TuningHooks`` -- logs at the levels the cache used historically
    (info for sweep start/done + invalidation, debug for hit/miss/persist)."""

    def env_override(self, kernel, choice):
        logger.debug("kernel_tuning_cache: %s env override -> %r", kernel, choice)

    def cache_hit(self, kernel, dims, region):
        logger.debug("kernel_tuning_cache: %s cache hit %s", kernel, dims)

    def cache_miss(self, kernel, dims):
        logger.debug("kernel_tuning_cache: %s cache miss %s", kernel, dims)

    def sweep_start(self, kernel, axes):
        logger.info("kernel_tuning_cache: %s sweep starting (axes=%s)", kernel, axes)

    def sweep_end(self, kernel, n_regions):
        logger.info("kernel_tuning_cache: %s sweep done (%d regions)", kernel, n_regions)

    def winner_chosen(self, kernel, region, reason):
        logger.debug("kernel_tuning_cache: %s winner (%s): %s", kernel, reason, region)

    def persist(self, kernel, path, n_regions):
        logger.debug("kernel_tuning_cache: %s persisted %d regions -> %s", kernel, n_regions, path)

    def invalidation(self, kernel, reason):
        logger.info("kernel_tuning_cache: %s invalidated (%s); will re-tune", kernel, reason)

    def concurrent_sweep_detected(self, kernel):
        logger.warning("kernel_tuning_cache: %s concurrent sweep / lock timeout", kernel)


_DEFAULT_HOOKS = LoggerHooks()


__all__ = [
    "SCHEMA_VERSION",
    "KernelTuningCache",
    "TuningHooks",
    "LoggerHooks",
    "cache_dir",
    "cache_path",
    "host_cache_dir",
    "register_default_cache",
    "hw_fingerprint",
]


# ---------------------------------------------------------------------------
# Note for downstream callers
# ---------------------------------------------------------------------------

# Why ``is_cuda_available`` is imported but unused here: keeps the symbol
# in scope for ``__init__`` re-exports that downstream packages may
# expect; remove if the import-time cost becomes a concern.
_ = is_cuda_available  # silence "unused import"
