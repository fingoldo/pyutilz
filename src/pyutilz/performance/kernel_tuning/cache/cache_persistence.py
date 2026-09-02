"""Storage layer of ``KernelTuningCache``: immutable per-kernel file IO, legacy-monolith migration,
remote read-through / write-through, and per-kernel directory garbage collection.

Split out of ``cache_class.py``; mixed into the class there, so every method keeps its original
``self.<attr>`` semantics and stays reachable as ``KernelTuningCache._<name>``.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import glob as _glob
import json
import os
import random
import time
from typing import Optional

from ._common import _CacheState, logger
from .cache_base import (
    SCHEMA_VERSION,
    _NO_CODE_VERSION,
    _build_provenance,
    _kernel_dir,
    _slug,
    cache_path,
    hw_fingerprint,
    provenance_changed,
)


class _CachePersistenceMixin(_CacheState):
    """Disk + remote persistence for ``KernelTuningCache`` (v3 immutable per-kernel files)."""

    # Re-announce at WARNING every this-many consecutive failures (not just the first) so a
    # long-lived process doesn't go silent again after the initial warning.
    _REMOTE_FAILURE_WARN_INTERVAL = 20

    def _log_remote_failure(self, op: str, exc: Exception) -> None:
        """Log a remote-backend failure, escalating to WARNING on the first failure after a run
        of successes and periodically thereafter (every ``_REMOTE_FAILURE_WARN_INTERVAL``
        consecutive failures); every other occurrence stays at DEBUG so a backend failing on
        EVERY call doesn't flood the log."""
        self._remote_consecutive_failures += 1
        n = self._remote_consecutive_failures
        if n == 1 or n % self._REMOTE_FAILURE_WARN_INTERVAL == 0:
            logger.warning("kernel_tuning_cache: remote %s failed (%d consecutive failure(s)): %s", op, n, exc)
        else:
            logger.debug("kernel_tuning_cache: remote %s failed: %s", op, exc)

    def _note_remote_success(self) -> None:
        """Reset the consecutive-failure counter; if recovering from a run of failures, log the
        recovery at WARNING too -- otherwise an operator who saw the earlier warning has no
        signal that the remote backend is working again."""
        if self._remote_consecutive_failures > 0:
            logger.warning("kernel_tuning_cache: remote operations recovered after %d consecutive failure(s)", self._remote_consecutive_failures)
        self._remote_consecutive_failures = 0

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
        immutable files, under a claim marker so only one process migrates.
        After a successful split the monolith is renamed aside (``.migrated``) so
        it is found-once but never re-read. Backward compatible: existing caches
        keep working transparently. Idempotent + crash-safe (a partial migration
        just re-runs; immutable writes can't corrupt).

        The claim marker uses the SAME pid+start_ts+steal pattern as the sweep
        INPROGRESS markers (``_try_create_marker``/pid-liveness + budget staleness
        check below) rather than a bare O_EXCL empty file -- a bare claim has no
        way to tell "a live peer is migrating" from "a process crashed after
        claiming and before removing it," so an orphaned claim from a killed
        process (OOM-killer, kill -9, container eviction) would otherwise disable
        migration on this host FOREVER (every subsequent process's O_EXCL fails
        and just returns, with no retry and no steal logic)."""
        if self._in_memory or self._path is None or self._migrated:
            return
        self._migrated = True  # at most one attempt per process
        legacy = cache_path()
        if not os.path.isfile(legacy):
            return
        claim = os.path.join(self._path, ".migrate.INPROGRESS")
        os.makedirs(self._path, exist_ok=True)
        owns = self._try_create_marker(claim)
        if not owns:
            owns = self._maybe_steal_migration_claim(claim)
        if not owns:
            return  # a live peer is migrating; its result will be read on next _load
        try:
            with open(legacy, encoding="utf-8") as f:
                data = json.load(f)
            # Only import kernels from a SCHEMA- and FINGERPRINT-compatible
            # monolith (CURRENT schema only, matching host). Pre-v4 tunings are deliberately NOT
            # accepted any more: they were recorded by an unsynchronized ``time_backend`` and so
            # carry launch-only GPU wall_ms (see SCHEMA_VERSION's note). A schema-999 / foreign /
            # corrupt monolith is treated exactly as the old _load did -- as a
            # miss -- so it is renamed aside WITHOUT importing anything (no
            # accidental resurrection of an invalid cache).
            compatible = data.get("schema_version") == SCHEMA_VERSION and data.get("hw_fingerprint") == hw_fingerprint()
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
        if self._remote is not None:
            # Read-through: pull this host's payload from the shared store + cache each kernel
            # as an immutable local file, then they resolve normally. Gated PER-KERNEL (any
            # remote kernel not already in the local view), not on the local store being
            # entirely empty -- a host that already has ANY locally-tuned kernel would otherwise
            # never learn about kernels tuned only by a peer sharing the same hw_fingerprint,
            # and its next write-through (which serializes only its own local view) would then
            # permanently overwrite/delete those peer-only kernels from the shared object.
            try:
                remote_data = self._remote.read(hw_fingerprint())
                self._note_remote_success()
            except Exception as e:
                self._log_remote_failure("read", e)
                remote_data = None
            if remote_data and remote_data.get("schema_version") == SCHEMA_VERSION:
                prov = remote_data.get("provenance")
                live_prov = _build_provenance()
                if not (prov and provenance_changed(prov, live_prov)):
                    for name, entry in (remote_data.get("kernels", {}) or {}).items():
                        if name in kernels:
                            continue  # local copy already resolved (possibly newer); don't shadow it
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
                self._note_remote_success()
            except Exception as e:
                self._log_remote_failure("write", e)

    def _remote_payload(self) -> dict:
        """Assemble the legacy-shaped monolithic payload (all kernels) for the
        remote store, so the remote object stays one-per-fingerprint and a peer's
        read-through repopulates every kernel.

        Merges in any kernel present on the CURRENT remote object but absent from this
        process's local view before writing -- a plain "serialize only what I have locally"
        write would otherwise overwrite/delete kernels tuned only by a peer host sharing the
        same hw_fingerprint that this process never loaded (e.g. tuned by that peer AFTER this
        process's own _load() ran). Local entries always win over remote ones for the same name.
        """
        kernels = dict((self._loaded or {}).get("kernels", {}) if self._loaded else {})
        try:
            remote_data = self._remote.read(hw_fingerprint()) if self._remote is not None else None
            if self._remote is not None:
                self._note_remote_success()
        except Exception as e:
            self._log_remote_failure("read (pre-write merge)", e)
            remote_data = None
        if remote_data and remote_data.get("schema_version") == SCHEMA_VERSION:
            for name, entry in (remote_data.get("kernels", {}) or {}).items():
                if name not in kernels and isinstance(entry, dict):
                    kernels[name] = entry
        return {
            "schema_version": SCHEMA_VERSION,
            "hw_fingerprint": hw_fingerprint(),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "provenance": _build_provenance(),
            "kernels": kernels,
        }

    def _gc_kernel_dir(self, kdir: str, keep: int = 4) -> None:
        """Lazily garbage-collect a kernel directory, keeping the newest ``keep`` immutable
        files. "Newest" uses the SAME ``(tuned_utc, mtime)`` key the readers
        (``_read_kernel_newest`` / ``_read_kernel_dir_by_path``) use to pick the current tuning
        -- NOT raw mtime alone.

        Regression fix (2026-07-21 audit round 2, MEDIUM): GC previously sorted purely by mtime
        (write order). ``update()`` accepts an explicit ``tuned_utc`` override; whenever that's
        set non-monotonically w.r.t. wall-clock write order (importing/replaying
        historically-timestamped tunings, or merging tunings recorded on a different host/
        clock), the entry the reader would pick as logically newest could have an OLDER mtime
        than several others and get silently, permanently deleted here before ever being read.
        Unreadable/corrupt files sort as oldest (empty ``tuned_utc``) and are evicted first --
        consistent with the readers, which skip them entirely anyway.

        Negligible space + never blocks; best-effort (a failed unlink is harmless -- the reader
        always picks the newest of whatever remains).
        """
        try:
            files = [p for p in _glob.glob(os.path.join(kdir, "*.json"))]
        except OSError:
            return
        if len(files) <= keep:
            return
        candidates: list[tuple] = []  # (tuned_ts, mtime, path)
        for p in files:
            ts = ""
            try:
                with open(p, encoding="utf-8") as f:
                    rec = json.load(f)
                entry = rec.get("entry")
                if isinstance(entry, dict):
                    ts = entry.get("tuned_utc") or ""
            except (OSError, json.JSONDecodeError):
                pass  # unreadable -- sorts as oldest (ts=""), evicted first
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = 0.0
            candidates.append((ts, mtime, p))
        candidates.sort(key=lambda c: (c[0], c[1]))
        for _ts, _mtime, p in candidates[:-keep]:
            with contextlib.suppress(OSError):
                os.remove(p)

    def _ensure_loaded(self) -> dict:
        """Return the live cache payload (loaded once per process). On
        miss returns an empty stub so subsequent updates can populate it."""
        # Lock-free fast path. Once loaded, ``self._loaded`` is only ever REBOUND (never torn), and
        # reading one attribute is atomic under the GIL, so taking the RLock to hand back a
        # already-present dict is pure overhead on the hottest path in the module (measured ~0.7-0.9
        # us of a ~2.2 us lookup). The locked slow path below still does the double-check.
        loaded = self._loaded
        if loaded is not None:
            return loaded
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            self._loaded = self._load() or {
                "schema_version": SCHEMA_VERSION,
                "hw_fingerprint": "in_memory" if self._in_memory else hw_fingerprint(),
                "kernels": {},
            }
            return self._loaded

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
