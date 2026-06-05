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

import contextlib
import datetime as _dt
import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from typing import Callable, Optional, Protocol

from pyutilz.core.pythonlib import is_cuda_available
from pyutilz.system.gpu_dispatch import gpu_capability_summary

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2  # v2: code_version + categorical (_eq) / range (_min) axes + python_version material


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
        # in_memory=True skips all disk / filelock / provenance -- a seedable
        # RAM cache for fast unit tests (KernelTuningCache(in_memory=True)).
        self._in_memory = in_memory
        self._path = None if in_memory else (path or cache_path())
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
            from .kernel_tuning_remote import get_remote_backend
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

    # ----- I/O -----

    def _write_local_copy(self, data: dict) -> None:
        """Atomically cache a remote-fetched payload to the local file so the
        normal local read path validates + serves it on subsequent calls."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.debug("kernel_tuning_cache: failed to cache remote payload: %s", e)

    def _load(self) -> Optional[dict]:
        """Lazy file-read. Returns the validated payload, or None on
        absent / unreadable / schema-version / fingerprint mismatch.

        On a local miss, reads through to the shared remote store (if any) for
        this fingerprint and caches it locally before validating."""
        if self._in_memory:
            return None
        if not os.path.isfile(self._path):
            # Read-through: pull this host's fingerprint payload from the
            # shared remote store, cache locally, then validate via the normal
            # local path below. Remote miss / no remote -> stays absent.
            if self._remote is not None:
                try:
                    remote_data = self._remote.read(hw_fingerprint())
                except Exception as e:  # any backend error -> local-only
                    logger.debug("kernel_tuning_cache: remote read failed: %s", e)
                    remote_data = None
                if remote_data is not None:
                    self._write_local_copy(remote_data)
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
        if not saved_prov:
            # _save always writes provenance, so a payload without it is
            # hand-crafted or from an old/foreign writer -- we can't validate it
            # against a driver/cupy/numba bump, so surface it (B8). Conservative:
            # we still serve it (the hw_fingerprint already matched).
            logger.warning("kernel_tuning_cache: %s has no provenance -- cannot "
                           "validate against a driver/cupy bump; serving as-is", self._path)
        elif provenance_changed(saved_prov, live_prov):
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
        if self._in_memory:
            return  # RAM-only: nothing to persist
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
            # Write-through to the shared remote store (best-effort; a flaky
            # network or raising backend never breaks the local save).
            if self._remote is not None:
                try:
                    self._remote.write(hw_fingerprint(), payload)
                except Exception as e:
                    logger.debug("kernel_tuning_cache: remote write failed: %s", e)

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
            self._loaded["kernels"][kernel_name] = entry
            if hooks is not None:
                hooks.persist(kernel_name, self._path, len(regions))
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
            if kernel_name in self._loaded.get("kernels", {}):
                del self._loaded["kernels"][kernel_name]
                self._save(self._loaded["kernels"])
                return True
            return False

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
                    once_per_process: bool = True, lock_timeout: float = 900.0):
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
            return fallback() if callable(fallback) else fallback
        with self._tuning_lock(kernel_name, lock_timeout, hk):
            # another process may have tuned it while we waited for the lock
            if not self._code_version_stale(kernel_name, code_version):
                hit = self.lookup(kernel_name, **dims)
                if hit is not None:
                    return hit
            _TUNED_THIS_PROCESS.add(guard_key)
            hk.sweep_start(kernel_name, axes)
            try:
                regions = tuner()
            except Exception as e:  # a sweep failure must never break dispatch
                logger.debug("kernel_tuning_cache: tuner for %s failed: %s", kernel_name, e)
                regions = None
            if regions:
                self.update(kernel_name, axes=axes, regions=regions, code_version=code_version,
                            salt=salt, equiv_tol=equiv_tol, hooks=hk)
                hk.sweep_end(kernel_name, len(regions))
                hit = self.lookup(kernel_name, **dims)
                if hit is not None:
                    hk.winner_chosen(kernel_name, hit, "from sweep")
                    return hit
        hk.winner_chosen(kernel_name, None, "fallback")
        return fallback() if callable(fallback) else fallback

    def _code_version_stale(self, kernel_name: str, code_version: Optional[str]) -> bool:
        """True iff a stored code_version exists and differs from the live one."""
        if code_version is None:
            return False
        data = self._ensure_loaded()
        stored = data.get("kernels", {}).get(kernel_name, {}).get("code_version")
        return stored is not None and stored != code_version

    @contextlib.contextmanager
    def _tuning_lock(self, kernel_name: str, timeout: float, hooks):
        """Cross-process advisory lock so two cold processes don't both run the
        (expensive) sweep. No-op if ``filelock`` is absent or in_memory. On
        timeout (a crashed / over-long tuner) fires a hook and proceeds WITHOUT
        the lock rather than deadlocking dispatch (accept a rare duplicate
        sweep, which ``_save`` merges, over a lost update)."""
        if self._in_memory or self._path is None:
            yield
            return
        try:
            from filelock import FileLock, Timeout
        except ImportError:
            yield
            return
        lock = FileLock(self._path + "." + _slug(kernel_name) + ".tune.lock", timeout=timeout)
        try:
            lock.acquire()
        except Timeout:
            hooks.concurrent_sweep_detected(kernel_name)
            logger.warning("kernel_tuning_cache: tuning lock timeout for %s after %.0fs; "
                           "proceeding without lock (possible duplicate sweep)", kernel_name, timeout)
            yield
            return
        try:
            yield
        finally:
            try:
                lock.release()
            except Exception:
                pass


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
    "hw_fingerprint",
]


# ---------------------------------------------------------------------------
# Note for downstream callers
# ---------------------------------------------------------------------------

# Why ``is_cuda_available`` is imported but unused here: keeps the symbol
# in scope for ``__init__`` re-exports that downstream packages may
# expect; remove if the import-time cost becomes a concern.
_ = is_cuda_available  # silence "unused import"
