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
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from typing import Any, Optional

from pyutilz.core.pythonlib import is_cuda_available
from pyutilz.system.gpu_dispatch import gpu_capability_summary

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# HW fingerprint
# ---------------------------------------------------------------------------

def _slug(s: str, maxlen: int = 40) -> str:
    """Filename-safe lowercase + truncated form of an arbitrary string."""
    s = re.sub(r"\(R\)|\(TM\)|\bCPU\b|\bGPU\b|@.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s.strip("-._").lower()[:maxlen] or "unknown"


def _cpu_model_slug() -> str:
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


def _gpu_slug_and_cc() -> tuple[str, str]:
    """Returns (gpu_name_slug, cc_str). On CPU-only host: ("no-gpu", "").

    Uses the LIVE current CUDA device id, not always 0, so a 2-GPU box
    where the user routes to device 1 gets a distinct fingerprint
    (different GPU model + cc may apply).
    """
    try:
        dev_id = _current_device_id()
        summary = gpu_capability_summary(dev_id)
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
    """Full path to the JSON file for the live host."""
    return os.path.join(cache_dir(), f"{hw_fingerprint()}.json")


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
        "python_version": _safe_version("sys", "version"),
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
        summary = gpu_capability_summary(0)
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
    cupy/numba versions, GPU cc/name). Timestamps + python version are
    NOT considered material -- a Python patch bump shouldn't invalidate
    a kernel tuning."""
    if old is None or new is None:
        return False  # be conservative: no data -> no invalidation
    keys = ("cuda_driver_version", "cuda_runtime_version",
            "cupy_version", "numba_version", "numpy_version")
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

    def __init__(self, path: Optional[str] = None):
        self._path = path or cache_path()
        # Reentrant so ``update`` can call ``_ensure_loaded`` under its own
        # lock without deadlocking (regression caught by
        # tests/system/test_kernel_tuning_cache.py during the first
        # implementation pass).
        self._lock = threading.RLock()
        self._loaded: Optional[dict] = None  # None until first load() call

    # ----- I/O -----

    def _load(self) -> Optional[dict]:
        """Lazy file-read. Returns the validated payload, or None on
        absent / unreadable / schema-version / fingerprint mismatch."""
        if not os.path.isfile(self._path):
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("kernel_tuning_cache: failed to read %s: %s", self._path, e)
            return None
        if data.get("schema_version") != SCHEMA_VERSION:
            logger.info(
                "kernel_tuning_cache: schema mismatch at %s (got %r, expected %d); "
                "treating as miss",
                self._path, data.get("schema_version"), SCHEMA_VERSION,
            )
            return None
        if data.get("hw_fingerprint") != hw_fingerprint():
            logger.info(
                "kernel_tuning_cache: hw_fingerprint mismatch at %s "
                "(got %r, expected %r); treating as miss",
                self._path, data.get("hw_fingerprint"), hw_fingerprint(),
            )
            return None
        # Provenance staleness check: a CUDA driver / cupy / numba bump
        # since the tuning was saved invalidates the cache (a new sweep
        # may find different optima under updated kernels / drivers).
        saved_prov = data.get("provenance")
        live_prov = _build_provenance()
        if saved_prov and provenance_changed(saved_prov, live_prov):
            logger.info(
                "kernel_tuning_cache: provenance changed since save "
                "(CUDA/cupy/numba/GPU bump); treating as miss to trigger re-tune"
            )
            return None
        return data

    def _save(self, kernels: dict) -> None:
        """Atomic write of the full payload, safe across PROCESSES.

        Two joblib worker processes both calling ``update(...)`` on the
        same on-disk cache file race: each loaded a snapshot, mutates
        its own ``kernels`` dict, saves -- last writer wins, silently
        dropping the other worker's kernel update. Fix: take an OS-level
        advisory file lock around the read-modify-write so concurrent
        saves serialise and merge correctly. Best-effort: if the
        ``filelock`` lib is missing, fall back to plain atomic rename
        and accept the race (in-process callers always go through
        ``_ensure_loaded`` under the ``RLock`` so single-process is
        safe regardless).
        """
        lock_path = self._path + ".lock"
        try:
            from filelock import FileLock
            _file_lock = FileLock(lock_path, timeout=10)
        except ImportError:
            _file_lock = None

        def _do_save() -> None:
            # Re-read existing on-disk state INSIDE the lock + merge the
            # caller's ``kernels`` dict so concurrent writers preserve
            # each other's entries instead of clobbering.
            existing_kernels: dict = {}
            if os.path.isfile(self._path):
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    if (existing_data.get("schema_version") == SCHEMA_VERSION
                            and existing_data.get("hw_fingerprint") == hw_fingerprint()):
                        existing_kernels = existing_data.get("kernels", {}) or {}
                except (OSError, json.JSONDecodeError):
                    pass
            # Merge: caller's kernels win for matching names; other names
            # preserved.
            merged = dict(existing_kernels)
            merged.update(kernels)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "hw_fingerprint": hw_fingerprint(),
                "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "provenance": _build_provenance(),
                "kernels": merged,
            }
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, self._path)
            logger.info("kernel_tuning_cache: saved %s", self._path)

        if _file_lock is not None:
            with _file_lock:
                _do_save()
        else:
            _do_save()

    def _ensure_loaded(self) -> dict:
        """Return the live cache payload (loaded once per process). On
        miss returns an empty stub so subsequent updates can populate it."""
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            self._loaded = self._load() or {
                "schema_version": SCHEMA_VERSION,
                "hw_fingerprint": hw_fingerprint(),
                "kernels": {},
            }
            return self._loaded

    # ----- public API -----

    def update(self, kernel_name: str, *, axes: list[str], regions: list[dict]) -> None:
        """Replace ``kernels[kernel_name]`` with the given axes + regions and
        persist. Other kernels' tunings are preserved."""
        with self._lock:
            self._ensure_loaded()
            self._loaded["kernels"][kernel_name] = {"axes": list(axes), "regions": list(regions)}
            self._save(self._loaded["kernels"])

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

    def lookup(self, kernel_name: str, **dims: int) -> Optional[dict]:
        """First region whose ``..._max`` caps are all >= the requested
        ``dims`` (or None / absent = match any). Returns the region dict
        verbatim, minus the ``..._max`` keys. None on cache miss.

        Example::

            cache.lookup("joint_hist_batched", n_samples=1_000_000, joint_size=25)
            # -> {"variant": "shared", "block_size": 512, "wall_ms": 0.78}
        """
        regions = self.get_regions(kernel_name)
        if not regions:
            return None
        for region in regions:
            if _region_matches(region, dims):
                return {k: v for k, v in region.items() if not k.endswith("_max")}
        return None

    def reset(self) -> None:
        """Drop the in-memory cache; next read reloads from disk. For tests
        + on driver-update invalidation hooks."""
        with self._lock:
            self._loaded = None


def _region_matches(region: dict, dims: dict[str, int]) -> bool:
    """A region matches a dims dict iff every ``axis_max`` key in the
    region is None / absent, OR the request's matching dim is <= max."""
    for axis_name, axis_value in dims.items():
        cap_key = f"{axis_name}_max"
        cap_value = region.get(cap_key)
        if cap_value is None:
            continue  # this axis is unconstrained for this region
        if axis_value > cap_value:
            return False
    return True


__all__ = [
    "SCHEMA_VERSION",
    "KernelTuningCache",
    "cache_dir",
    "cache_path",
    "hw_fingerprint",
]


# ---------------------------------------------------------------------------
# Note for downstream callers
# ---------------------------------------------------------------------------

# Why ``is_cuda_available`` is imported but unused here: keeps the symbol
# in scope for ``__init__`` re-exports that downstream packages may
# expect; remove if the import-time cost becomes a concern.
_ = is_cuda_available  # silence "unused import"
