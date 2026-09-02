"""Cross-process sweep-claim protocol for ``KernelTuningCache``: INPROGRESS marker publication,
stale-claim stealing, and the ``_claim_sweep`` context manager.

Split out of ``cache_class.py``; mixed into the class there, so every method keeps its original
``self.<attr>`` semantics and stays reachable as ``KernelTuningCache._<name>``.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import Optional

from ._common import _CacheState, logger
from .cache_base import _NO_CODE_VERSION, _kernel_dir, _pid_alive, _slug, _sweep_budget_seconds, hw_fingerprint


class _CacheSweepClaimMixin(_CacheState):
    """Singleton-without-blocking sweep ownership (O_EXCL markers + stale-owner steal)."""

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

    def _decide_steal(self, path: str) -> tuple[bool, int, float, bool, float]:
        """Read a claim/marker JSON file at ``path`` and decide whether it's stealable.

        2026-08-02 near-duplicate-function-body finding: _maybe_steal_migration_claim and
        _maybe_steal_marker independently duplicated this whole read-and-decide step; extracted
        so the two callers differ only in what they do with a "yes, steal it" verdict (marker
        removal, ``hooks`` notification, log message wording).

        Returns ``(should_steal, pid, age, owner_dead, budget)``. Steal (True) iff the owning
        pid is dead OR the claim/marker is older than the max-sweep budget; else give up
        (False). INCOMPLETE-MARKER GUARD: a marker/claim missing pid/start_ts is either a peer
        caught mid-creation (the legacy O_EXCL-then-write fallback's empty-file window) or a
        process that crashed between create and write. Do NOT steal it on the empty-payload
        heuristic alone (pid=0 -> _pid_alive False; start_ts=0 -> age=inf > budget) -- that is
        exactly the double-sweep race. Fall back to the file mtime as the age: a FRESH
        incomplete marker (within budget) means a live peer is publishing -> give up; only an
        mtime-stale one is a genuine crash to steal.
        """
        try:
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            info = {}
        pid = int(info.get("pid", 0) or 0)
        start_ts = float(info.get("start_ts", 0.0) or 0.0)
        budget = _sweep_budget_seconds()
        if pid <= 0 or start_ts <= 0.0:
            try:
                age = time.time() - os.path.getmtime(path)
            except OSError:
                age = float("inf")
            if age <= budget:
                return False, pid, age, False, budget  # a peer is mid-creation -> let it finish
        else:
            age = time.time() - start_ts
        same_host = info.get("host") in (None, hw_fingerprint())
        # Only trust the pid-liveness probe for a marker/claim written on THIS host.
        owner_dead = same_host and not _pid_alive(pid)
        if not (owner_dead or age > budget):
            return False, pid, age, owner_dead, budget  # a live, in-budget owner -> give up
        return True, pid, age, owner_dead, budget

    def _maybe_steal_migration_claim(self, claim: str) -> bool:
        """Like ``_maybe_steal_marker`` but for the legacy-migration claim (no ``hooks``
        callback, no kernel_name -- the migration claim is process-global, not per-kernel)."""
        should_steal, pid, age, owner_dead, budget = self._decide_steal(claim)
        if not should_steal:
            return False
        logger.info("kernel_tuning_cache: stealing stale migration claim (pid=%s alive=%s age=%.0fs budget=%.0fs)", pid, not owner_dead, age, budget)
        with contextlib.suppress(OSError):
            os.remove(claim)
        return self._try_create_marker(claim)

    def _maybe_steal_marker(self, marker: str, kernel_name: str, hooks) -> bool:
        """An existing marker was found. Steal it (return True) iff the owning pid
        is dead OR start_ts is older than the max-sweep budget; else give up
        (False). Stealing is itself racy-safe: we remove the stale marker and
        re-create via O_EXCL; if a third process beats us to the recreate, we lose
        the claim (return False) -- correct, exactly one sweeper wins."""
        should_steal, pid, age, owner_dead, budget = self._decide_steal(marker)
        if not should_steal:
            return False
        logger.info(
            "kernel_tuning_cache: stealing stale sweep marker for %s " "(pid=%s alive=%s age=%.0fs budget=%.0fs)", kernel_name, pid, not owner_dead, age, budget
        )
        hooks.concurrent_sweep_detected(kernel_name)
        with contextlib.suppress(OSError):
            os.remove(marker)
        return self._try_create_marker(marker)
