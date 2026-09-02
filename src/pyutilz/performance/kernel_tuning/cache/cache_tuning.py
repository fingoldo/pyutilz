"""Tuning orchestration for ``KernelTuningCache``: the equivalence gate, ``get_or_tune``
(env -> lookup -> claimed sweep -> fallback) and the background async sweep.

Split out of ``cache_class.py``; mixed into the class there, so every method keeps its original
``self.<attr>`` semantics and stays reachable as ``KernelTuningCache._<name>``.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from ._common import _CacheState, _facade, logger
from .cache_base import (
    _INVALIDATION_LOGGED_THIS_PROCESS,
    _TUNED_THIS_PROCESS,
    _async_sweep_idle_max_wait,
    _async_sweep_start_delay,
    _slug,
    _tuned_guard_lock,
)
from .cache_hooks import TuningHooks, _DEFAULT_HOOKS  # TuningHooks is used in annotations only
from .region_matching import _AXIS_SUFFIXES


class _CacheTuningMixin(_CacheState):
    """The ``get_or_tune`` orchestration path and the sweep helpers it drives."""

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

        if env_key:
            forced = os.environ.get(env_key, "").strip()
            if forced:
                hk.env_override(kernel_name, forced)
                return forced
        # guard_key computed up-front: reused both for the once-per-process sweep guard below AND the
        # once-per-process invalidation-log guard, so a kernel stuck stale (e.g. a no-op tuner that never
        # persists a fresh entry) logs the INFO invalidation banner at most once per process, not on every call.
        guard_key = (kernel_name, self._path or id(self))
        # Resolve the entry ONCE and hand it to both the staleness check and the lookup: each used to
        # re-enter ``_ensure_loaded`` (a lock acquire plus dict walk) for the very same dict.
        entry = self._ensure_loaded().get("kernels", {}).get(kernel_name)
        if self._entry_code_version_stale(entry, code_version):
            if guard_key not in _INVALIDATION_LOGGED_THIS_PROCESS:
                _INVALIDATION_LOGGED_THIS_PROCESS.add(guard_key)
                hk.invalidation(kernel_name, "code_version changed")
        else:
            hit = self._lookup_in_entry(kernel_name, entry, dims)
            if hit is not None:
                hk.cache_hit(kernel_name, dims, hit)
                return hit
        hk.cache_miss(kernel_name, dims)

        # Defined only past the hit path: on a cache HIT this closure was built (and its cells
        # populated) on every call for a branch that was never taken.
        def _fb():
            """Consult the project-shipped DEFAULT-cache layer before falling back to ``fallback``."""
            # DEFAULT-cache layer: a project-shipped anonymized (hw-agnostic) tuning, consulted on a local per-host
            # MISS BEFORE the hand-specified fallback. Measurement-derived, so better than the heuristic; the async
            # sweep still runs to replace it with THIS host's measured optimum. The local measured cache already
            # took precedence (checked above), so this never overrides a real local result.
            #
            # Diagnostic branch-tagging (not just a swallowed exception): a CI-only occurrence of
            # test_register_default_cache_loads_and_local_miss_returns_default returned the hand
            # fallback instead of the DEFAULT cache's value, with no local repro despite exhaustive
            # static review of every function this branch touches (code_version determinism,
            # _ensure_loaded, lookup, _region_matches -- none raise or mismatch under same-process
            # execution as far as static reading can show). Logging exactly which branch was taken
            # (rather than only the previous DEBUG-level swallow) turns the next occurrence into a
            # one-line diagnosis instead of another blind investigation.
            dc = _facade()._DEFAULT_CACHE
            if dc is None:
                logger.warning("DEFAULT-cache consult for kernel %s: _DEFAULT_CACHE is None, falling back", kernel_name)
            elif dc is self:
                logger.warning("DEFAULT-cache consult for kernel %s: _DEFAULT_CACHE is this same instance, falling back", kernel_name)
            else:
                try:
                    if dc._code_version_stale(kernel_name, code_version):
                        logger.warning(
                            "DEFAULT-cache consult for kernel %s: code_version stale (requested=%r, stored=%r), falling back",
                            kernel_name, code_version, dc._ensure_loaded().get("kernels", {}).get(kernel_name, {}).get("code_version"),
                        )
                    else:
                        d = dc.lookup(kernel_name, **dims)
                        if d is not None:
                            return d
                        logger.warning("DEFAULT-cache consult for kernel %s: lookup(**%r) returned no matching region, falling back", kernel_name, dims)
                except Exception as e:  # nosec B110 - best-effort consult of the optional DEFAULT-cache layer on a local miss; any failure here must fall through to the caller-supplied fallback, not raise
                    logger.warning("DEFAULT-cache consult for kernel %s raised %s: %s, falling back", kernel_name, type(e).__name__, e, exc_info=True)
            return fallback() if callable(fallback) else fallback

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
        return self._entry_code_version_stale(self._ensure_loaded().get("kernels", {}).get(kernel_name), code_version)

    @staticmethod
    def _entry_code_version_stale(entry: Optional[dict], code_version: Optional[str]) -> bool:
        """``_code_version_stale`` against an ALREADY-RESOLVED entry, for callers (``get_or_tune``)
        that have one in hand and must not pay a second ``_ensure_loaded`` to re-fetch it."""
        if code_version is None:
            return False
        stored = (entry or {}).get("code_version")
        return bool(stored is not None and stored != code_version)
