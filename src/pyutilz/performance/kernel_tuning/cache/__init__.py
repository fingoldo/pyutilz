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

from pyutilz.core.pythonlib import is_cuda_available

from .cache_base import (
    SCHEMA_VERSION,
    _NO_CODE_VERSION,
    _TUNED_THIS_PROCESS,
    _async_sweep_hw_busy,
    _async_sweep_idle_max_wait,
    _async_sweep_start_delay,
    _build_provenance,
    _cpu_model_slug,
    _current_device_id,
    _gpu_slug_and_cc,
    _gpu_summary_cached,
    _kernel_dir,
    _pid_alive,
    _read_hw_fingerprint_from_disk,
    _safe_version,
    _slug,
    _sweep_budget_seconds,
    _write_hw_fingerprint_to_disk,
    cache_dir,
    cache_path,
    gpu_capability_summary,
    host_cache_dir,
    hw_fingerprint,
    provenance_changed,
)
import threading  # re-exported: a test references ``cache.threading.Event()``

from .cache_class import KernelTuningCache, register_default_cache
from .cache_hooks import LoggerHooks, TuningHooks, _DEFAULT_HOOKS
from .region_matching import _AXIS_SUFFIXES, _region_match_reason, _region_matches

# Process-wide singletons owned by THIS facade module (not the ``cache_class``
# submodule): ``KernelTuningCache.load_or_create`` / ``register_default_cache`` /
# ``get_or_tune`` route through ``sys.modules[<facade>]`` to read/write them, so a
# test that rebinds ``cache._DEFAULT_INSTANCE = None`` or calls
# ``register_default_cache`` is observed everywhere -- exactly as when the whole
# module was monolithic.
_DEFAULT_INSTANCE = None
_DEFAULT_INSTANCE_LOCK = threading.Lock()
_DEFAULT_CACHE = None


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

