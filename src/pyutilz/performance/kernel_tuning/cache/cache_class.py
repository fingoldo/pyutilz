"""``KernelTuningCache`` -- per-host read/write/lookup + get_or_tune orchestration."""
from __future__ import annotations

# The names below that this module no longer USES itself (the persistence / sweep-claim / tuning
# blocks moved to the sibling mixins) are kept imported ON PURPOSE: ``cache_class`` is a direct
# import target for callers and tests, so every attribute it exposed before the split must keep
# resolving as ``cache_class.<name>``. Hence the per-name ``noqa: F401``.
import contextlib  # noqa: F401
import datetime as _dt
import glob as _glob  # noqa: F401
import json
import logging  # noqa: F401
import os  # noqa: F401
import random  # noqa: F401
import sys  # noqa: F401
import threading
import time
from typing import Any, Callable, Optional, cast  # noqa: F401

from .cache_base import (
    SCHEMA_VERSION,
    _NO_CODE_VERSION,  # noqa: F401
    _INVALIDATION_LOGGED_THIS_PROCESS,
    _TUNED_THIS_PROCESS,
    _async_sweep_idle_max_wait,  # noqa: F401
    _async_sweep_start_delay,  # noqa: F401
    _build_provenance,  # noqa: F401
    _kernel_dir,  # noqa: F401
    _pid_alive,  # noqa: F401
    _slug,  # noqa: F401
    _sweep_budget_seconds,  # noqa: F401
    _tuned_guard_lock,  # noqa: F401
    cache_path,  # noqa: F401
    host_cache_dir,
    hw_fingerprint,
    provenance_changed,  # noqa: F401
)
from .cache_hooks import LoggerHooks, TuningHooks, _DEFAULT_HOOKS  # noqa: F401 (TuningHooks used in annotations)
from .region_matching import _AXIS_SUFFIXES, _OP_EQ, _OP_MAX, _OP_MIN, _SUFFIX_OPS, _region_match_reason, _region_matches  # noqa: F401 - _OP_EQ/_region_matches are re-exported for the historical cache_class.<name> surface
from ._common import _CacheState, _FACADE_NAME, _facade, logger  # noqa: F401 (re-exported: historical ``cache_class`` attribute surface)
from .cache_persistence import _CachePersistenceMixin
from .cache_sweeping import _CacheSweepClaimMixin
from .cache_tuning import _CacheTuningMixin

# Process-wide singleton for load_or_create() -- hot-path dispatch callers
# (GPU/numba kernels) consult the cache on every invocation; one shared
# instance avoids re-reading the per-host JSON each call. The singletons
# themselves live on the FACADE package and are reached through ``_facade()``
# (imported above from ``._common``, where the rationale is documented).


def register_default_cache(path: str) -> bool:
    """Register a project's anonymized default-tuning JSON (committed to the project repo). On a local per-host
    cache miss, ``get_or_tune`` consults it before the hand-specified fallback (and still kicks the async sweep to
    measure THIS host). Returns whether it loaded. A missing/unreadable file degrades silently to no defaults."""
    fac = _facade()
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        c = KernelTuningCache(in_memory=True)
        with c._lock:
            c._loaded = {"schema_version": payload.get("schema_version", SCHEMA_VERSION), "kernels": payload.get("kernels", {})}
            c._invalidate_plan()
        fac._DEFAULT_CACHE = c
        logger.debug("kernel_tuning_cache: registered %d default kernels from %s", len(payload.get("kernels", {})), path)
        return True
    except FileNotFoundError:
        logger.debug("kernel_tuning_cache: no default cache at %s", path)
        fac._DEFAULT_CACHE = None
        return False
    except Exception as e:
        logger.warning("kernel_tuning_cache: could not load default cache %s: %s", path, e)
        fac._DEFAULT_CACHE = None
        return False


class KernelTuningCache(_CachePersistenceMixin, _CacheSweepClaimMixin, _CacheTuningMixin):
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
        # Annotated explicitly: the two assignments below otherwise join to bare ``object``, which
        # hides every attribute of the duck-typed remote backend from the type checker.
        self._remote: Any
        if in_memory:
            self._remote = None
        else:
            from ..remote import get_remote_backend
            self._remote = get_remote_backend()
        # Regression fix (2026-07-21 audit round 2, HIGH): remote read/write failures used to
        # log at DEBUG unconditionally, with no escalation -- invisible by default (loggers are
        # WARNING-and-above when unconfigured), so an expired credential / renamed bucket /
        # blocked network silently disabled cross-machine sharing forever with zero signal. See
        # ``_log_remote_failure``/``_note_remote_success``.
        self._remote_consecutive_failures = 0
        # Per-kernel PRECOMPILED lookup plans (see ``_lookup_plan``). Purely derived from
        # ``self._loaded``; rebuilt on demand, so it is dropped by ``__getstate__`` rather than
        # pickled (a runtime cache in pickled state is how stale plans outlive their entry).
        self._plan_cache: dict = {}

    def __getstate__(self) -> dict:
        """Drop the unpicklable ``threading.RLock`` (a fresh one is created in ``__setstate__``) --
        flagged by ``pyutilz.dev.code_audit.unpicklable_resource_state``; this cache isn't on any
        current pickling path, but the guard is cheap and matches the repo-wide convention.

        Also drops the derived ``_plan_cache``: it holds compiled lookup plans keyed by entry
        IDENTITY, which does not survive a pickle round-trip, so carrying it across would be both
        dead weight and a correctness hazard."""
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_plan_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and re-create the ``threading.RLock`` dropped by ``__getstate__``, plus an
        empty ``_plan_cache`` (also for instances pickled before that field existed)."""
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._plan_cache = {}

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
        fac = _facade()
        if fac._DEFAULT_INSTANCE is None:
            with fac._DEFAULT_INSTANCE_LOCK:
                if fac._DEFAULT_INSTANCE is None:
                    fac._DEFAULT_INSTANCE = cls()
        return fac._DEFAULT_INSTANCE  # type: ignore[no-any-return]  # _facade() is a runtime module lookup, so its _DEFAULT_INSTANCE global is untyped

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
            loaded = self._ensure_loaded()
            # Update the in-memory snapshot, then persist ONLY this one kernel as
            # a new immutable file. No read-modify-write of a shared document ->
            # no lost update (D1 dissolved): a concurrent writer of a DIFFERENT
            # kernel writes a different directory, and a concurrent writer of the
            # SAME kernel writes a distinct file (newest wins), never clobbering.
            loaded["kernels"][kernel_name] = entry
            self._invalidate_plan(kernel_name)
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
        return self._lookup_in_entry(kernel_name, entry, dims)

    def _lookup_in_entry(self, kernel_name: str, entry: Optional[dict], dims: dict) -> Optional[dict]:
        """``lookup`` against an ALREADY-RESOLVED entry, so a caller holding one (``get_or_tune``)
        does not re-enter ``_ensure_loaded`` -- a second lock acquire plus dict walk -- to fetch it
        again."""
        if not entry:
            return None
        for constraints, payload in self._lookup_plan(kernel_name, entry):
            for axis_name, op, bound in constraints:
                if axis_name not in dims:
                    continue  # a region constraint on an axis the caller didn't ask about is unconstrained
                axis_value = dims[axis_name]
                if op == _OP_MAX:
                    if axis_value > bound:
                        break
                elif op == _OP_MIN:
                    if axis_value < bound:
                        break
                elif axis_value != bound:  # _OP_EQ
                    break
            else:
                return cast(Optional[dict], payload)
        return None

    def _lookup_plan(self, kernel_name: str, entry: dict) -> list:
        """Compiled ``[(constraints, payload)]`` plan for one kernel entry, memoized per instance.

        ``lookup`` is the per-launch dispatch decision the whole kernel-tuning subsystem exists to
        make cheap, and its old shape rebuilt the SAME derived data on every single call: a set
        comprehension over ``axes x _AXIS_SUFFIXES``, then three f-string keys per axis PER REGION
        inside the matcher, then a dict comprehension to strip the winner. None of it depends on
        ``dims``. Compiling it once per entry turns a hit into a tuple walk plus a dict return
        (measured 4.7-6.3 us -> 0.9-2.0 us, 3-5x).

        ``constraints`` is a tuple of ``(axis_name, op, bound)`` with ``op`` one of the ``_OP_*``
        codes, built from EVERY region key ending in an axis suffix (matching ``_region_matches``,
        which keys off the requested dims rather than the declared axes). ``payload`` is the region
        already stripped of the DECLARED axes' constraint keys only -- the B4 distinction that keeps
        a decision field like ``block_size_max`` in the returned payload.

        Keyed by kernel name and validated by entry IDENTITY, so replacing an entry (``update``,
        ``evict``, ``reset``, a reload) invalidates the plan automatically.
        """
        cached = self._plan_cache.get(kernel_name)
        if cached is not None and cached[0] is entry:
            return cached[1]  # type: ignore[no-any-return]
        constraint_keys = {f"{ax}{suf}" for ax in (entry.get("axes") or []) for suf in _AXIS_SUFFIXES}
        plan = []
        for region in entry.get("regions") or []:
            constraints = []
            for key, bound in region.items():
                if bound is None:
                    continue  # an absent/None constraint is unconstrained
                for suffix, op in _SUFFIX_OPS:
                    if key.endswith(suffix):
                        constraints.append((key[: -len(suffix)], op, bound))
                        break
            payload = {k: v for k, v in region.items() if k not in constraint_keys}
            plan.append((tuple(constraints), payload))
        self._plan_cache[kernel_name] = (entry, plan)
        return plan

    def _invalidate_plan(self, kernel_name: Optional[str] = None) -> None:
        """Drop the compiled lookup plan for one kernel (or all of them when ``kernel_name`` is
        None). Cheap insurance on top of the identity check in ``_lookup_plan``: a mutation made
        IN PLACE on an existing entry keeps its identity, so the writers call this explicitly."""
        if kernel_name is None:
            self._plan_cache.clear()
        else:
            self._plan_cache.pop(kernel_name, None)

    def reset(self) -> None:
        """Drop the in-memory cache; next read reloads from disk. For tests
        + on driver-update invalidation hooks."""
        with self._lock:
            self._loaded = None
            self._invalidate_plan()

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
            loaded = self._ensure_loaded()
            # Clear the once-per-process sweep guard so a subsequent get_or_tune
            # can actually re-tune this kernel (B11) instead of short-circuiting
            # to the fallback because "we already swept it this process".
            _TUNED_THIS_PROCESS.discard((kernel_name, self._path or id(self)))
            # Same for the once-per-process invalidation-log guard: an explicit evict is a fresh start,
            # so a future staleness detection for this kernel should log again.
            _INVALIDATION_LOGGED_THIS_PROCESS.discard((kernel_name, self._path or id(self)))
            present = kernel_name in loaded.get("kernels", {})
            self._invalidate_plan(kernel_name)
            if present:
                del loaded["kernels"][kernel_name]
            self._delete_kernel_files(kernel_name)
            if present:
                # Refresh the remote object so the eviction propagates (best-effort).
                if not self._in_memory and self._remote is not None:
                    try:
                        self._remote.write(hw_fingerprint(), self._remote_payload())
                        self._note_remote_success()
                    except Exception as e:
                        self._log_remote_failure("write", e)
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
            return {"matched": False, "region_index": None, "region": None, "reason": f"no regions for kernel {kernel_name!r}"}
        constraint_keys = {f"{ax}{suf}" for ax in (entry.get("axes") or []) for suf in _AXIS_SUFFIXES}
        first_reason = None
        for i, region in enumerate(regions):
            ok, why = _region_match_reason(region, dims)
            if ok:
                payload = {k: v for k, v in region.items() if k not in constraint_keys}
                return {"matched": True, "region_index": i, "region": payload, "reason": f"region {i} matched"}
            if first_reason is None:
                first_reason = f"region 0 rejected: {why}"
        return {"matched": False, "region_index": None, "region": None, "reason": first_reason or f"no region matched dims {dims}"}
