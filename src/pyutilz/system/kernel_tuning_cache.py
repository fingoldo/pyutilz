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


def _gpu_slug_and_cc() -> tuple[str, str]:
    """Returns (gpu_name_slug, cc_str). On CPU-only host: ("no-gpu", "")."""
    try:
        summary = gpu_capability_summary(0)
        if summary is None:
            return ("no-gpu", "")
        name = summary.get("name") or "unknown"
        cc = f"{int(summary.get('cc_major', 0))}.{int(summary.get('cc_minor', 0))}"
        return (_slug(name), cc)
    except Exception as e:
        logger.debug("gpu_capability_summary failed: %s", e)
        return ("no-gpu", "")


@lru_cache(maxsize=1)
def hw_fingerprint() -> str:
    """Stable per-host key. Format::

        cpu_<cpu_slug>_gpu_<gpu_slug>_cc<major>.<minor>
        cpu_<cpu_slug>_no-gpu                    (CPU-only host)

    Cached for the process lifetime (host doesn't change mid-run).
    """
    cpu = _cpu_model_slug()
    gpu, cc = _gpu_slug_and_cc()
    if gpu == "no-gpu":
        return f"cpu_{cpu}_no-gpu"
    return f"cpu_{cpu}_gpu_{gpu}_cc{cc}"


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
        return data

    def _save(self, kernels: dict) -> None:
        """Atomic write of the full payload."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "hw_fingerprint": hw_fingerprint(),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "kernels": kernels,
        }
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)
        logger.info("kernel_tuning_cache: saved %s", self._path)

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
