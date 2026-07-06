"""Kernel tuner registry: TunerSpec + kernel_tuner() registration + discovery.

A tuner spec defines a hot kernel's variants, how to measure them, what hardware
it targets, and how to gate the decision. A consumer registers its spec with a
single ``kernel_tuner(...)`` call at module top. Discovery walks the mlframe
package (only when explicitly invoked, never at `import mlframe`) and collects
all specs for batch tuning via retune_all() or the CLI.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .cache import KernelTuningCache
from .code_versioning import compute_code_version

logger = logging.getLogger(__name__)

__all__ = [
    "TunerSpec",
    "kernel_tuner",
    "discover_tuners",
    "retune_all",
    "tune_spec",
    "get_registry",
]

# Global registry: kernel_name -> TunerSpec (kernel names are globally unique)
_REGISTRY: Dict[Any, Any] = {}


@dataclass
class TunerSpec:
    """Specification for one kernel variant, its tuning, and dispatch decision."""

    kernel_name: str
    """Globally-unique kernel name (e.g. 'dtw_dispatch', 'batch_pair_mi') -- the
    registry key and the kernel_tuning_cache key."""

    variant_fns: tuple
    """The reference (and any always-defined) kernel implementations, hashed by
    compute_code_version() so a kernel edit re-tunes. GPU variants that are only
    conditionally defined are covered by ``salt`` instead."""

    tuner: Callable[[], list]
    """Zero-arg callable returning a region list (the project's benchmark sweep).
    Called by get_or_tune() on a cache miss."""

    axes: dict[str, list[Any]]
    """Categorical/range axes for the region matcher. Keys map to axes in the
    cache regions (e.g. {'ndim_eq': [2, 3], 'n_max': [100, 1000, 10000]})."""

    fallback: Any
    """Fallback decision if tuning fails or is skipped (e.g. the numpy variant,
    or a constant threshold)."""

    extra_fns: tuple = ()
    """Env-toggle kernels, shared helpers, or module constants that affect
    the variant but have no Python source (e.g. a function that switches kernel
    at runtime based on an env var). Passed to compute_code_version() alongside
    variant_fns."""

    salt: int = 0
    """Integer salt for semantic dependencies the source can't see (module
    constants, env toggles, refactored-but-equivalent logic). Bumped by the
    developer when the fallback behavior changes in a way that invalidates old
    tunings."""

    env_key: Optional[str] = None
    """Environment variable key for a one-shot override (e.g. 'MLFRAME_FE_BACKEND').
    The value selects the variant/decision directly, bypassing the cache."""

    max_dims: Optional[dict[str, int]] = None
    """Upper-bound dims for practical tuning. Sweep stops if any axis exceeds
    its bound (e.g. {'n_max': 1_000_000}), sparing huge benchmarks. Use to
    cap grid-search explosion."""

    equiv_tol: Optional[dict[str, float]] = None
    """Tolerance for numerical equivalence when comparing variants. Keys are
    metrics (e.g. 'max_abs_diff', 'rel_error'). If set, conflicting regions
    are rejected + warned, never silently substituted."""

    gpu_capable: bool = False
    """Whether this kernel has GPU variants and residency matters. If True,
    the tuning sweep should discover and model H2D/D2H transfer costs via the
    'location' axis (host/device). CPU-only kernels leave this False."""

    cli_label: Optional[str] = None
    """Human-friendly name for the CLI (defaults to kernel_name). Used in
    ``mlframe-tune-kernels show <label>`` commands."""

    _choice_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    """Per-spec memo of choose() results, keyed by sorted dims (the dispatch is hot)."""

    def code_version(self) -> Optional[str]:
        """code_version over variant_fns + extra_fns + salt (memoized in
        compute_code_version). None if code-versioning is unavailable."""
        from .code_versioning import compute_code_version

        try:
            return compute_code_version(*self.variant_fns, extra_fns=self.extra_fns, salt=self.salt)
        except Exception as e:
            logger.debug("%s code_version() unavailable: %s", self.kernel_name, e)
            return None

    def _fallback_choice(self, dims: dict) -> str:
        """Resolve the fallback to a backend_choice string. ``fallback`` may be a
        callable (dims -> str/dict, the dynamic heuristic), a {'backend_choice': X}
        dict, or a bare string."""
        fb = self.fallback(**dims) if callable(self.fallback) else self.fallback
        if isinstance(fb, dict):
            return str(fb.get("backend_choice", ""))
        return str(fb)

    def choose(self, **dims) -> str:
        """Per-host backend decision for these dims -- the one-call dispatch that
        replaces a consumer's hand-written _backend_choice + _code_version. Routes
        env -> code-version-checked cache -> on-miss (locked, once-per-process)
        sweep -> fallback via get_or_tune, returns the backend_choice string,
        memoized per dims. The caller MUST still gate a GPU choice on live CUDA +
        per-op compatibility before routing to device."""
        key = tuple(sorted(dims.items()))
        if key in self._choice_cache:
            return self._choice_cache[key]
        fb = self._fallback_choice(dims)
        bc = fb
        tuned = False
        try:
            from .cache import KernelTuningCache

            cache = KernelTuningCache.load_or_create()
            result = cache.get_or_tune(
                self.kernel_name, dims=dims, tuner=self.tuner, axes=list(self.axes.keys()),
                fallback={"backend_choice": fb}, code_version=self.code_version(),
                env_key=self.env_key,
                equiv_tol=(self.equiv_tol or {}).get("max_abs_diff") if self.equiv_tol else None,
                async_sweep=True,  # FIT-TIME dispatch: never block the fit on the sweep; measure in the background
            )
            cand = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
            if cand:
                bc = cand
            # The kernel is "tuned" once regions exist (the background sweep wrote them). Until then get_or_tune
            # returns the fallback; we must NOT memoize that, or every fit in this process would be pinned to the
            # fallback even after the async sweep lands.
            tuned = cache.has(self.kernel_name) and not cache.code_version_stale(self.kernel_name, self.code_version())
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            logger.debug("%s choose() failed: %s", self.kernel_name, e)
        except Exception as e:
            logger.warning("%s choose() failed with unexpected %s: %s", self.kernel_name, type(e).__name__, e)
        # Memoize only a settled (tuned) decision. While the async sweep is pending, re-resolve each call (a cheap
        # in-memory lookup) so the measured backend is picked up the moment the background sweep finishes.
        if tuned:
            self._choice_cache[key] = bc
        return bc


def kernel_tuner(**kwargs) -> "TunerSpec":
    """Build + register a TunerSpec. Call it directly at a consumer module's
    top level (pyutilz is a hard dependency, so no defensive try/except is
    needed)::

        from pyutilz.performance.kernel_tuning.registry import kernel_tuner

        kernel_tuner(
            kernel_name="dtw_dispatch",
            variant_fns=(dtw_cpu,),          # reference body for code_version
            tuner=_run_dtw_sweep,
            axes={"n_cells": [10_000, 160_000, 2_560_000]},
            fallback={"backend_choice": "cpu"},
            gpu_capable=True,
            salt=1,
        )

    Registered under its (globally-unique) ``kernel_name``. Idempotent on
    re-registration. Returns the spec (handy for tests / introspection)."""
    spec = TunerSpec(**kwargs)
    if spec.kernel_name in _REGISTRY:
        # Re-registration is legitimate: a module re-import (importlib.reload, a test that drops + re-imports the
        # consumer subgraph, or two import paths reaching the same module) re-runs the decorator with an equivalent
        # spec. Overwrite -- it is the same kernel. Raising here would break every consumer imported after the
        # re-registration (e.g. the whole MRMR/RFECV path once the feature_selection.filters subgraph is re-imported).
        logger.debug("kernel_tuner: re-registering %s (module re-import); overwriting prior spec", spec.kernel_name)
    _REGISTRY[spec.kernel_name] = spec
    return spec


def get_registry() -> dict:
    """Return the global tuner registry: {kernel_name: TunerSpec}."""
    return dict(_REGISTRY)


def discover_tuners(package: str = "mlframe", warn_on_import_fail: bool = True) -> dict:
    """Import every module under ``package`` so each module-level
    ``kernel_tuner(...)`` call fires, then return the accumulated registry.

    Registration happens at module import. We do NOT clear the registry first:
    Python caches imported modules, so re-importing an already-imported module
    does NOT re-run its top-level ``kernel_tuner(...)`` -- clearing would
    therefore wipe specs we can never re-register in the same process, leaving
    ``retune_all`` with an empty/partial registry. Accumulation is safe: each
    module registers exactly once per process (import cache), and the
    duplicate-name guard in ``kernel_tuner`` surfaces genuine collisions.

    Import failures are logged at WARNING (not swallowed) and skipped so one
    broken module never stops the walk. Returns ``{kernel_name: TunerSpec}``.
    """
    # Resolve the package module. On failure the existing registry still stands
    # (we never cleared it), so return that rather than a misleading empty dict.
    try:
        pkg = importlib.import_module(package)
    except ImportError as e:
        logger.error("Failed to import package %r: %s", package, e)
        return dict(_REGISTRY)

    pkg_path = pkg.__path__ if hasattr(pkg, "__path__") else []
    collected = 0

    for importer, modname, ispkg in pkgutil.walk_packages(path=pkg_path, prefix=f"{package}.", onerror=lambda _: None):
        try:
            importlib.import_module(modname)
            collected += 1
        except Exception as e:
            if warn_on_import_fail:
                logger.warning("Failed to import %r: %s", modname, e)
            # Continue; don't let one broken module stop the walk.

    logger.debug("Discovered %s tuner specs from %s modules", len(_REGISTRY), collected)
    return dict(_REGISTRY)


def retune_all(
    package: str = "mlframe",
    force: bool = False,
    idle_wait_tries: int = 5,
    idle_wait_sec: float = 0.5,
    hooks: Optional[Any] = None,
    skip_existing: bool = True,
) -> dict:
    """Orchestrate tuning of all discovered specs: multi-GPU grouping, retry.

    Algorithm:
    1. discover_tuners() to populate the registry.
    2. Group GPU devices by unique model (name + compute capability).
       Identical GPUs on the same machine share one fingerprint file.
    3. For each (TunerSpec, GPU-model group):
       - Find the least-loaded device in that group (via GPUtil.load).
       - Idle-wait: sleep(idle_wait_sec) × idle_wait_tries if GPU busy.
       - Call get_or_tune(..., once_per_process=False, force=force, hooks=hooks)
         with the device selected, and compute_code_version() from the spec's
         variant_fns + extra_fns + salt.
       - Persist under that model's hw_fingerprint.
    4. CPU-only specs (gpu_capable=False) run once (no multi-GPU grouping).

    Args:
        package: Package to discover specs from (default: "mlframe").
        force: If True, re-tune even if cache hits. If False, respect cache.
        idle_wait_tries: Retry count if GPU load > 80%.
        idle_wait_sec: Sleep duration (seconds) between retries.
        hooks: Optional TuningHooks for progress/events.
        skip_existing: If True (default), SKIP sweeping any (kernel, hardware)
            whose CURRENT code_version already has a cached tuning -- re-running
            the benchmark does not redo finished work. False forces a full
            re-sweep of every spec (combine with ``force`` to also discard the
            existing tuning before measuring).

    Returns:
        {kernel_name: n_regions_tuned, ...}

    Side effects:
        - Calls discover_tuners (clears + repopulates registry).
        - Updates cache files under the host's hw_fingerprint.
    """
    from tqdm import tqdm

    specs = discover_tuners(package=package)

    if not specs:
        logger.info("No tuner specs discovered in %r", package)
        return {}

    # Group GPU devices by unique model (name + compute capability).
    # _group_gpus_by_model imports GPUtil; if unavailable, it raises ImportError
    # which we catch here to skip GPU specs (CPU specs still run).
    try:
        gpu_groups = _group_gpus_by_model()
    except ImportError:
        logger.warning("GPUtil not available; skipping multi-GPU grouping. " "Install with: pip install gputil")
        gpu_groups = {}

    results = {}  # {(model, kernel_name): n_regions} -- keyed per (model, kernel), NOT clobbered

    # CPU-only specs: run once.
    cpu_specs = [(k, s) for k, s in specs.items() if not s.gpu_capable]
    for kernel_name, spec in tqdm(cpu_specs, desc="Tuning CPU specs", leave=False):
        code_version = compute_code_version(*spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt)
        results[("cpu", kernel_name)] = _run_spec_tuning(
            KernelTuningCache(), spec, code_version, device_id=None, force=force, hooks=hooks,
            skip_existing=skip_existing,
        )

    # GPU-capable specs: tune once per unique GPU model, on its least-loaded device.
    # NOTE (known limitation): the cache persists under cache_path()'s hw_fingerprint,
    # which is computed for the LIVE default device. On a host with DIFFERENT GPU
    # models this writes every model's tuning under device-0's fingerprint -- correct
    # for single-GPU and identical-multi-GPU hosts (one fingerprint), but a
    # device-aware hw_fingerprint() is needed for mixed-model hosts (untestable here).
    gpu_specs = [(k, s) for k, s in specs.items() if s.gpu_capable]
    for kernel_name, spec in tqdm(gpu_specs, desc="Tuning GPU specs", leave=False):
        for model_name, device_ids in gpu_groups.items():
            device_id = _pick_least_loaded_device(device_ids, idle_wait_tries, idle_wait_sec)
            if device_id is None:
                logger.warning("No available device for %s; skipping %s", model_name, kernel_name)
                continue
            code_version = compute_code_version(*spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt)
            results[(model_name, kernel_name)] = _run_spec_tuning(
                KernelTuningCache(), spec, code_version, device_id=device_id, force=force, hooks=hooks,
                skip_existing=skip_existing,
            )

    return results


def _group_gpus_by_model() -> dict[str, list[int]]:
    """Group GPUs by unique model (name + compute capability).

    Returns: {model_name: [device_ids], ...}
    """
    import GPUtil

    gpus = GPUtil.getGPUs()
    groups: Dict[Any, Any] = {}
    for gpu in gpus:
        # Model name: abbreviated GPU name + compute capability.
        model = f"{gpu.name.split()[0]}_{gpu.compute_capability[0]}{gpu.compute_capability[1]}"
        groups.setdefault(model, []).append(gpu.id)
    return groups


def _pick_least_loaded_device(device_ids: list[int], idle_wait_tries: int, idle_wait_sec: float) -> Optional[int]:
    """Pick the least-loaded GPU from a list, with idle-wait retry.

    If all devices are busy (load > 80%) after idle_wait_tries, return None.
    """
    import GPUtil  # optional dep -> lazy

    for attempt in range(idle_wait_tries):
        gpus = {g.id: g.load for g in GPUtil.getGPUs() if g.id in device_ids}
        available = [d for d, load in gpus.items() if load <= 0.8]
        if available:
            return min(available, key=lambda d: gpus[d])  # Least loaded.
        if attempt < idle_wait_tries - 1:
            time.sleep(idle_wait_sec)
    return None


def tune_spec(spec: TunerSpec, *, force: bool = True, device_id: Optional[int] = None, hooks: Optional[Any] = None, skip_existing: bool = True) -> int:
    """Tune a single registered spec (compute its code_version, run the sweep,
    persist) and return the number of regions now cached. The public single-spec
    entry point -- e.g. ``mlframe-tune-kernels refresh <kernel>``; ``retune_all``
    batches this across every spec + GPU model.

    ``skip_existing`` (default True): if the kernel's CURRENT code_version is
    already cached, skip the sweep and return the existing region count -- so a
    re-run of an offline benchmark doesn't redo finished work. ``force=True``
    overrides skip_existing (it evicts then re-sweeps unconditionally)."""
    code_version = compute_code_version(*spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt)
    return _run_spec_tuning(KernelTuningCache(), spec, code_version, device_id=device_id, force=force, hooks=hooks, skip_existing=skip_existing)


def _run_spec_tuning(cache, spec: TunerSpec, code_version: str, device_id: Optional[int], force: bool, hooks: Optional[Any], skip_existing: bool = True) -> int:
    """Tune one spec via get_or_tune; return the number of regions now cached.

    ``force`` evicts first so the sweep re-runs even on a cache hit. A GPU spec
    runs inside ``cp.cuda.Device(device_id)`` so the sweep measures on the chosen
    device; if cupy is unavailable the spec is SKIPPED (running a GPU sweep on the
    default/CPU path would measure garbage -- B10), not silently mis-measured.
    CPU specs (device_id None) run as-is. Idle-wait is handled upstream by
    ``_pick_least_loaded_device`` -- not a concern here.

    ``skip_existing`` (default True): if the kernel's CURRENT code_version is
    already tuned, skip the (expensive) sweep entirely and just return the
    cached region count. ``force`` takes precedence (always re-sweeps)."""
    # equiv_tol on the spec is a {metric: tol} dict; the cache gate takes the
    # max-abs-diff float (surfaces + rejects divergent regions, never masks).
    tol = spec.equiv_tol.get("max_abs_diff") if spec.equiv_tol else None

    if skip_existing and not force:
        # Already tuned at the live code_version on this hardware -> nothing to do.
        if cache.has(spec.kernel_name) and not cache.code_version_stale(spec.kernel_name, code_version):
            logger.debug("skip_existing: %s already tuned at code_version %s; skipping sweep", spec.kernel_name, code_version)
            return len(cache.get_regions(spec.kernel_name) or [])

    def _tune():
        if force:
            cache.evict(spec.kernel_name)
        # dims={} just drives the tuner + persist; the return is ignored (we
        # count persisted regions). equiv_tol is threaded so a divergent region
        # is rejected at update even on this forced sweep.
        cache.get_or_tune(
            spec.kernel_name, dims={}, tuner=spec.tuner, axes=list(spec.axes.keys()),
            fallback=spec.fallback, env_key=spec.env_key, code_version=code_version,
            salt=spec.salt, equiv_tol=tol, hooks=hooks, once_per_process=False,
            async_sweep=False,  # offline tuning must run the sweep SYNCHRONOUSLY and wait for/persist the result
        )

    if device_id is not None:
        try:
            import cupy as cp
        except ImportError:
            logger.warning("cupy unavailable -> skipping GPU spec %s (won't mis-measure on CPU)", spec.kernel_name)
            return 0
        with cp.cuda.Device(device_id):
            _tune()
    else:
        _tune()

    return len(cache.get_regions(spec.kernel_name) or [])
