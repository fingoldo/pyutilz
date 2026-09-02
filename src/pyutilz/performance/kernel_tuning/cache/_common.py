"""Shared state declarations + facade resolution for the ``cache_class`` implementation modules.

``cache_class.py`` is split into cohesive siblings (``cache_persistence`` / ``cache_sweeping`` /
``cache_tuning``); this hub holds what all of them need: the logger channel, the facade lookup, and
the attribute declarations of the cache instance they are mixed into.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

# Pinned to the historical module path so records logged by the relocated methods keep the channel
# they had before the split.
logger = logging.getLogger("pyutilz.performance.kernel_tuning.cache.cache_class")

# The three module singletons (``_DEFAULT_INSTANCE``, ``_DEFAULT_INSTANCE_LOCK``,
# ``_DEFAULT_CACHE``) LIVE ON THE FACADE package (``...kernel_tuning.cache``), not
# in these submodules: tests reset them via ``cache._DEFAULT_INSTANCE = None`` /
# ``register_default_cache`` and the reads here must observe those rebinds. We route
# every access through ``_facade()`` so a reassignment on the package (whether by a
# test or by ``register_default_cache``) is seen by ``load_or_create`` / ``_fb``,
# exactly as it was when this all lived in one module. The lock is created lazily on
# first access so import order (facade imports this module) can't deadlock.
_FACADE_NAME = "pyutilz.performance.kernel_tuning.cache"


def _facade():
    """Return the facade package module, so reads observe any test/registration rebinds of its singletons."""
    return sys.modules[_FACADE_NAME]


class _CacheState:
    """Declaration-only base of the ``KernelTuningCache`` instance state shared by the mixins.

    Carries no behaviour and no class attributes -- only annotations (plus, for type-checkers, the
    signatures of the methods that cross module boundaries between the mixins), so each sibling
    module type-checks on its own without importing the others.
    """

    _in_memory: bool
    _path: Optional[str]
    _migrated: bool
    _lock: "threading.RLock"
    _loaded: Optional[dict]
    _remote: Any  # a remote backend instance (see ..remote.get_remote_backend) or None; duck-typed read/write
    _remote_consecutive_failures: int
    _plan_cache: dict

    if TYPE_CHECKING:  # implemented in the sibling mixins; declared here so cross-module calls type-check
        def _ensure_loaded(self) -> dict:
            """Returns the live cache payload, loaded from disk once per process (empty stub on miss)."""
            ...

        def _persist_kernel(self, kernel_name: str, entry: dict, *, provenance: Optional[dict] = ..., remote: bool = ...) -> None:
            """Writes one immutable per-kernel tuning file, then optionally write-through to the remote store."""
            ...

        def _delete_kernel_files(self, kernel_name: str) -> None:
            """Removes every on-disk tuning file of one kernel (best-effort; no-op for in-memory caches)."""
            ...

        def _remote_payload(self) -> dict:
            """Assembles the monolithic all-kernels payload for the remote store, merging in peer-only kernels."""
            ...

        def _log_remote_failure(self, op: str, exc: Exception) -> None:
            """Records a remote-backend failure, escalating to WARNING on the first one and periodically after."""
            ...

        def _note_remote_success(self) -> None:
            """Clears the consecutive-failure counter, logging the recovery when failures preceded it."""
            ...

        def _try_create_marker(self, marker: str) -> bool:
            """Atomically publishes the marker file with its pid/start_ts content; False if it already exists."""
            ...

        def _maybe_steal_migration_claim(self, claim: str) -> bool:
            """Takes over the process-global legacy-migration claim when its owner is dead or over budget."""
            ...

        def _claim_sweep(self, kernel_name: str, code_version: Optional[str], hooks) -> "contextlib.AbstractContextManager[bool]":
            """Context manager yielding True iff this process owns the sweep for the kernel/code_version pair."""
            ...

        def _apply_equiv_gate(self, kernel_name, regions, equiv_tol, hooks=...):
            """Returns the regions minus any whose recorded ``max_abs_diff`` exceeds ``equiv_tol``, warning loudly."""
            ...

        def _code_version_stale(self, kernel_name: str, code_version: Optional[str]) -> bool:
            """True iff the kernel has a stored code_version that differs from the live one."""
            ...

        def _run_tuner(self, kernel_name: str, tuner: Callable, axes: list, hooks):
            """Runs a project sweep behind the start hook, returning its regions or None when it raised."""
            ...

        def _spawn_async_sweep(self, kernel_name: str, *, dims, tuner, axes, code_version, salt, equiv_tol, hooks) -> None:
            """Starts the sweep on a background daemon thread so the caller's fit is never blocked by tuning."""
            ...

        def _invalidate_plan(self, kernel_name: Optional[str] = ...) -> None:
            """Drops the compiled lookup plan of one kernel, or of all of them when ``kernel_name`` is None."""
            ...

        def _lookup_in_entry(self, kernel_name: str, entry: Optional[dict], dims: dict) -> Optional[dict]:
            """Performs ``lookup`` against an already-resolved entry, skipping the repeated cache load."""
            ...

        def lookup(self, kernel_name: str, **dims) -> Optional[dict]:
            """Returns the first tuned region whose axis constraints all match ``dims``, else None."""
            ...

        def update(self, kernel_name: str, *, axes: list[str], regions: list[dict], code_version: Optional[str] = ...,
                   salt: int = ..., equiv_tol: Optional[float] = ..., tuned_utc: Optional[str] = ..., hooks=...) -> None:
            """Replaces one kernel's axes/regions plus tuning metadata and persists it, gating on ``equiv_tol``."""
            ...

        def reset(self) -> None:
            """Drops the in-memory cache and its lookup plans, so the next read reloads from disk."""
            ...
