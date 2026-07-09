"""``KernelTuningCache`` -- per-host read/write/lookup + get_or_tune orchestration."""
from __future__ import annotations

import contextlib
import datetime as _dt
import glob as _glob
import json
import logging
import os
import random
import sys
import threading
import time
from typing import Callable, Optional

from .cache_base import (
    SCHEMA_VERSION,
    _NO_CODE_VERSION,
    _TUNED_THIS_PROCESS,
    _async_sweep_idle_max_wait,
    _async_sweep_start_delay,
    _build_provenance,
    _kernel_dir,
    _pid_alive,
    _slug,
    _sweep_budget_seconds,
    _tuned_guard_lock,
    cache_path,
    host_cache_dir,
    hw_fingerprint,
    provenance_changed,
)
from .cache_hooks import LoggerHooks, TuningHooks, _DEFAULT_HOOKS  # noqa: F401 (TuningHooks used in annotations)
from .region_matching import _AXIS_SUFFIXES, _region_match_reason, _region_matches

logger = logging.getLogger(__name__)


# Process-wide singleton for load_or_create() -- hot-path dispatch callers
# (GPU/numba kernels) consult the cache on every invocation; one shared
# instance avoids re-reading the per-host JSON each call.
#
# The three module singletons (``_DEFAULT_INSTANCE``, ``_DEFAULT_INSTANCE_LOCK``,
# ``_DEFAULT_CACHE``) LIVE ON THE FACADE package (``...kernel_tuning.cache``), not
# in this submodule: tests reset them via ``cache._DEFAULT_INSTANCE = None`` /
# ``register_default_cache`` and the reads here must observe those rebinds. We route
# every access through ``_facade()`` so a reassignment on the package (whether by a
# test or by ``register_default_cache``) is seen by ``load_or_create`` / ``_fb``,
# exactly as it was when this all lived in one module. The lock is created lazily on
# first access so import order (facade imports this module) can't deadlock.
_FACADE_NAME = "pyutilz.performance.kernel_tuning.cache"


def _facade():
    """Return the facade package module, so reads observe any test/registration rebinds of its singletons."""
    return sys.modules[_FACADE_NAME]


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
            from ..remote import get_remote_backend
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
        fac = _facade()
        if fac._DEFAULT_INSTANCE is None:
            with fac._DEFAULT_INSTANCE_LOCK:
                if fac._DEFAULT_INSTANCE is None:
                    fac._DEFAULT_INSTANCE = cls()
        return fac._DEFAULT_INSTANCE  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

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
            tmp = f"{final_path}.{os.getpid()}.{random.randrange(1 << 30):x}.tmp"  # nosec B311 - unique-ish temp filename suffix to avoid concurrent-writer collisions, not security/cryptographic use
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
        logger.debug("kernel_tuning_cache: atomic write of %s failed after %d tries: %s", final_path, retries, last_err)
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
            with open(legacy, encoding="utf-8") as f:
                data = json.load(f)
            # Only import kernels from a SCHEMA- and FINGERPRINT-compatible
            # monolith (v2 or v3 shape, matching host). A schema-999 / foreign /
            # corrupt monolith is treated exactly as the old _load did -- as a
            # miss -- so it is renamed aside WITHOUT importing anything (no
            # accidental resurrection of an invalid cache).
            compatible = data.get("schema_version") in (SCHEMA_VERSION, 2) and data.get("hw_fingerprint") == hw_fingerprint()
            kernels = (data.get("kernels", {}) or {}) if compatible else {}
            prov = data.get("provenance")
            for name, entry in kernels.items():
                if not isinstance(entry, dict):
                    continue
                # Carry provenance into each split file so the staleness check
                # behaves exactly as it did against the monolith.
                self._persist_kernel(name, dict(entry), provenance=prov, remote=False)
            os.replace(legacy, legacy + ".migrated")
            logger.info("kernel_tuning_cache: migrated %d kernels from legacy %s (compatible=%s)", len(kernels), legacy, compatible)
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
        if self._path is None:
            return None
        kdir = _kernel_dir(self._path, kernel_name)
        files = [p for p in _glob.glob(os.path.join(kdir, "*.json"))]
        if not files:
            return None
        live_prov = _build_provenance()
        candidates: list[tuple] = []  # (tuned_ts, mtime, entry)
        for p in files:
            try:
                with open(p, encoding="utf-8") as f:
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
        return candidates[-1][2]  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

    def _load(self) -> Optional[dict]:
        """Build the in-memory ``{schema_version, hw_fingerprint, kernels}`` view
        by resolving every kernel directory to its newest immutable file. Pure
        read, NO lock. On an empty local store, reads through to the shared remote
        (if any) and caches the pulled kernels locally as immutable files before
        resolving. Returns None when nothing is found (so ``_ensure_loaded``
        installs an empty stub)."""
        if self._in_memory or self._path is None:
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
                with open(p, encoding="utf-8") as f:
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

    def _persist_kernel(self, kernel_name: str, entry: dict, *, provenance: Optional[dict] = None, remote: bool = True) -> None:
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
        fname = f"{_slug(str(cv), maxlen=70)}.{int(salt)}.{os.getpid()}.{int(ts * 1000)}.{random.randrange(1 << 24):x}.json"  # nosec B311 - unique cache-record filename suffix to avoid collisions between concurrent writers, not security/cryptographic use
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
            loaded = self._ensure_loaded()
            # Update the in-memory snapshot, then persist ONLY this one kernel as
            # a new immutable file. No read-modify-write of a shared document ->
            # no lost update (D1 dissolved): a concurrent writer of a DIFFERENT
            # kernel writes a different directory, and a concurrent writer of the
            # SAME kernel writes a distinct file (newest wins), never clobbering.
            loaded["kernels"][kernel_name] = entry
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
            loaded = self._ensure_loaded()
            # Clear the once-per-process sweep guard so a subsequent get_or_tune
            # can actually re-tune this kernel (B11) instead of short-circuiting
            # to the fallback because "we already swept it this process".
            _TUNED_THIS_PROCESS.discard((kernel_name, self._path or id(self)))
            present = kernel_name in loaded.get("kernels", {})
            if present:
                del loaded["kernels"][kernel_name]
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
            """Consult the project-shipped DEFAULT-cache layer before falling back to ``fallback``."""
            # DEFAULT-cache layer: a project-shipped anonymized (hw-agnostic) tuning, consulted on a local per-host
            # MISS BEFORE the hand-specified fallback. Measurement-derived, so better than the heuristic; the async
            # sweep still runs to replace it with THIS host's measured optimum. The local measured cache already
            # took precedence (checked above), so this never overrides a real local result.
            dc = _facade()._DEFAULT_CACHE
            if dc is not None and dc is not self:
                try:
                    if not dc._code_version_stale(kernel_name, code_version):
                        d = dc.lookup(kernel_name, **dims)
                        if d is not None:
                            return d
                except Exception as e:  # nosec B110 - best-effort consult of the optional DEFAULT-cache layer on a local miss; any failure here must fall through to the caller-supplied fallback, not raise
                    logger.debug("DEFAULT-cache consult failed for kernel %s: %s", kernel_name, e)
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
        with _tuned_guard_lock:
            if once_per_process and guard_key in _TUNED_THIS_PROCESS:
                return _fb()
            if once_per_process:
                # Claim the guard immediately, atomically with the check above, so a second
                # thread racing for the same kernel can never observe "not yet tuned" and
                # spawn a duplicate async sweep / duplicate synchronous sweep-claim attempt.
                _TUNED_THIS_PROCESS.add(guard_key)

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
            self._spawn_async_sweep(kernel_name, dims=dims, tuner=tuner, axes=axes, code_version=code_version, salt=salt, equiv_tol=equiv_tol, hooks=hk)
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
                self.update(kernel_name, axes=axes, regions=regions, code_version=code_version, salt=salt, equiv_tol=equiv_tol, hooks=hk)
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
            """Background-thread body: debounce, wait for idle hardware, run the sweep, and persist its result."""
            try:
                # Debounce: wait before starting so the triggering fit gets past its bursty start, then the
                # busy-check sees the real device load (and we never grab the device the instant it's needed).
                delay = _async_sweep_start_delay()
                if delay:
                    time.sleep(delay)
                # WAIT for the hardware to go idle before benchmarking -- an async sweep that contends with
                # the caller's fit both taxes it (~18% wall) AND records contended timings as the optimum.
                # But do NOT abandon: the sweep is TRIGGERED by a fit, and a fit keeps the device busy, so a
                # defer-and-return would mean the cache NEVER populates on any host that actually fits (the
                # once-per-process guard is already set, so this process won't retry). Instead wait for an
                # idle gap up to a bounded budget; if the host stays busy that long (back-to-back fits),
                # proceed ANYWAY -- one mildly-contended sweep that populates the per-host cache beats never
                # tuning. The offline CLI remains the clean path; this is the best-effort fit-time fallback.
                try:
                    from ..benchmark import wait_for_idle_hardware
                    idle = wait_for_idle_hardware(max_wait=_async_sweep_idle_max_wait(), poll=2.0)
                except Exception:
                    idle = True
                if not idle:
                    logger.debug("kernel_tuning_cache: hardware still busy after wait; sweeping %s anyway "
                                 "(populate cache rather than starve tuning)", kernel_name)
                with self._claim_sweep(kernel_name, code_version, hooks) as owns:
                    if not owns:
                        return  # another process is already tuning this kernel; let it
                    self.reset()  # pick up any result a peer landed since we were spawned
                    if not self._code_version_stale(kernel_name, code_version) and self.lookup(kernel_name, **dims) is not None:
                        return  # tuned while we claimed
                    regions = self._run_tuner(kernel_name, tuner, axes, hooks)
                    if regions:
                        self.update(kernel_name, axes=axes, regions=list(regions), code_version=code_version, salt=salt, equiv_tol=equiv_tol, hooks=hooks)
                        hooks.sweep_end(kernel_name, len(regions))
            except Exception as e:  # a background sweep must never surface
                logger.debug("kernel_tuning_cache: async sweep for %s crashed: %s", kernel_name, e)
        threading.Thread(target=_run, name="ktc-sweep-" + _slug(kernel_name), daemon=True).start()

    def code_version_stale(self, kernel_name: str, code_version: Optional[str]) -> bool:
        """Public: True iff a stored code_version exists and differs from the live one.

        Stable entry point for callers (e.g. the tuner registry) that must decide
        whether a cached tuning is still valid without reaching into private state."""
        return self._code_version_stale(kernel_name, code_version)

    def _code_version_stale(self, kernel_name: str, code_version: Optional[str]) -> bool:
        """True iff a stored code_version exists and differs from the live one."""
        if code_version is None:
            return False
        data = self._ensure_loaded()
        stored = data.get("kernels", {}).get(kernel_name, {}).get("code_version")
        return stored is not None and stored != code_version

    def _marker_path(self, kernel_name: str, code_version: Optional[str]) -> str:
        """Path to the per-(kernel, code_version) INPROGRESS sweep marker. Callers must have
        already guarded on ``self._path is not None`` (in-memory caches have no marker path)."""
        assert self._path is not None, "_marker_path() requires a non-in-memory cache"
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
        """Atomically PUBLISH the marker WITH its content, stamping pid + start_ts.
        Returns True on success, False if it already exists.

        The marker must never be observable in an EMPTY state: the old code did
        ``os.open(O_CREAT|O_EXCL)`` then a SEPARATE ``os.write`` of the payload, so
        between those two syscalls a concurrent loser could read a zero-byte marker,
        parse ``{}`` -> ``pid=0`` (``_pid_alive(0)`` is False) + ``start_ts=0``
        (``age=inf > budget``), judge it STALE, and STEAL it -> two sweepers run. The
        window is sub-microsecond on an idle host but the OS scheduler widens it on a
        contended runner (observed only on the 2-core CI box: "expected one sweep,
        got 2"). Fix: write the payload to a per-attempt temp file, then ``os.link``
        it into place -- an atomic, exclusive publish (fails if the marker exists), so
        the marker is only ever visible fully-formed."""
        payload = json.dumps({"pid": os.getpid(), "start_ts": time.time(), "host": hw_fingerprint()}).encode("utf-8")
        # Staging path must be UNIQUE per concurrent claimer: same-process THREADS share os.getpid(), and
        # time.time_ns() can collide on a coarse-resolution clock (Windows), so (pid, tid, ns) -- tid disambiguates
        # concurrent threads, ns disambiguates a thread's sequential retries -- guarantees no two live claimers
        # pick the same tmp (a collision would make the loser's O_EXCL fail and the degrade-path wrongly own).
        tmp = f"{marker}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError as e:
            logger.debug("kernel_tuning_cache: marker tmp create failed (%s); sweeping without claim", e)
            return True  # can't stage -> behave as owner (degrade to no-singleton, never wedge)
        try:
            os.write(fd, payload)
        except OSError:
            pass
        finally:
            os.close(fd)
        try:
            # Atomic exclusive publish: link() fails with FileExistsError if the marker already exists,
            # giving the same single-winner guarantee as O_EXCL but with the content already in place.
            os.link(tmp, marker)
            return True
        except FileExistsError:
            return False
        except OSError as e:
            # Hardlinks unsupported on this fs (rare): fall back to the legacy O_EXCL-then-write path,
            # which still serialises winners; the empty-marker steal window reappears only there.
            logger.debug("kernel_tuning_cache: marker link unsupported (%s); O_EXCL fallback", e)
            try:
                fd2 = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
            except OSError:
                return True
            try:
                os.write(fd2, payload)
            except OSError:
                pass
            finally:
                os.close(fd2)
            return True
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)

    def _maybe_steal_marker(self, marker: str, kernel_name: str, hooks) -> bool:
        """An existing marker was found. Steal it (return True) iff the owning pid
        is dead OR start_ts is older than the max-sweep budget; else give up
        (False). Stealing is itself racy-safe: we remove the stale marker and
        re-create via O_EXCL; if a third process beats us to the recreate, we lose
        the claim (return False) -- correct, exactly one sweeper wins."""
        try:
            with open(marker, encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            info = {}
        pid = int(info.get("pid", 0) or 0)
        start_ts = float(info.get("start_ts", 0.0) or 0.0)
        budget = _sweep_budget_seconds()
        # INCOMPLETE-MARKER GUARD: a marker missing pid/start_ts is either a peer caught mid-creation (the
        # legacy O_EXCL-then-write fallback's empty-file window) or a process that crashed between create and
        # write. Do NOT steal it on the empty-payload heuristic alone (pid=0 -> _pid_alive False; start_ts=0 ->
        # age=inf > budget) -- that is exactly the double-sweep race. Fall back to the file mtime as the age:
        # a FRESH incomplete marker (within budget) means a live peer is publishing -> give up; only an mtime-
        # stale one is a genuine crash to steal.
        if pid <= 0 or start_ts <= 0.0:
            try:
                age = time.time() - os.path.getmtime(marker)
            except OSError:
                age = float("inf")
            if age <= budget:
                return False  # a peer is mid-creation -> let it finish
        else:
            age = time.time() - start_ts
        same_host = info.get("host") in (None, hw_fingerprint())
        # Only trust the pid-liveness probe for a marker written on THIS host.
        owner_dead = same_host and not _pid_alive(pid)
        if not (owner_dead or age > budget):
            return False  # a live, in-budget sweeper owns it -> give up
        logger.info(
            "kernel_tuning_cache: stealing stale sweep marker for %s " "(pid=%s alive=%s age=%.0fs budget=%.0fs)", kernel_name, pid, not owner_dead, age, budget
        )
        hooks.concurrent_sweep_detected(kernel_name)
        with contextlib.suppress(OSError):
            os.remove(marker)
        return self._try_create_marker(marker)
