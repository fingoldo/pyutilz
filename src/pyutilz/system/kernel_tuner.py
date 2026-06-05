"""Kernel tuner registry: TunerSpec + kernel_tuner() registration + discovery.

A tuner spec defines a hot kernel's variants, how to measure them, what hardware
it targets, and how to gate the decision. A consumer registers its spec with a
single ``kernel_tuner(...)`` call at module top. Discovery walks the mlframe
package (only when explicitly invoked, never at `import mlframe`) and collects
all specs for batch tuning via retune_all() or the CLI.
"""
import importlib
import logging
import pkgutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..dev.code_versioning import compute_code_version
from .kernel_tuning_cache import KernelTuningCache

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
_REGISTRY = {}


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


def kernel_tuner(**kwargs) -> "TunerSpec":
    """Build + register a TunerSpec. Call it directly at a consumer module's
    top level (pyutilz is a hard dependency, so no defensive try/except is
    needed)::

        from pyutilz.system.kernel_tuner import kernel_tuner

        kernel_tuner(
            kernel_name="dtw_dispatch",
            variant_fns=(dtw_cpu,),          # reference body for code_version
            tuner=_run_dtw_sweep,
            axes={"n_cells": [10_000, 160_000, 2_560_000]},
            fallback={"backend_choice": "cpu"},
            gpu_capable=True,
            salt=1,
        )

    Registered under its (globally-unique) ``kernel_name``. Raises on a
    duplicate name. Returns the spec (handy for tests / introspection)."""
    spec = TunerSpec(**kwargs)
    if spec.kernel_name in _REGISTRY:
        raise ValueError(f"Duplicate kernel_tuner registration: {spec.kernel_name}")
    _REGISTRY[spec.kernel_name] = spec
    return spec


def get_registry() -> dict:
    """Return the global tuner registry: {kernel_name: TunerSpec}."""
    return dict(_REGISTRY)


def discover_tuners(
    package: str = "mlframe", warn_on_import_fail: bool = True
) -> dict:
    """Walk a package, import all modules, and collect @kernel_tuner specs.

    Algorithm: use pkgutil.walk_packages to enumerate all modules under
    `package`. For each module, attempt import. If the import fails, log a
    WARNING (not silently swallowed) and skip that module — the remaining
    modules' specs are still collected.

    Returns:
        {kernel_name: TunerSpec, ...}

    Side effects:
        - Clears the global registry and repopulates it (so subsequent
          discover_tuners() calls give the freshest state).
        - Logs WARNINGs for any import failures.
    """
    _REGISTRY.clear()

    # Resolve the package module.
    try:
        pkg = importlib.import_module(package)
    except ImportError as e:
        logger.error("Failed to import package %r: %s", package, e)
        return {}

    pkg_path = pkg.__path__ if hasattr(pkg, "__path__") else []
    collected = 0

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=pkg_path, prefix=f"{package}.", onerror=lambda _: None
    ):
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
        logger.warning(
            "GPUtil not available; skipping multi-GPU grouping. "
            "Install with: pip install gputil"
        )
        gpu_groups = {}

    results = {}

    # CPU-only specs: run once.
    cpu_specs = [(k, s) for k, s in specs.items() if not s.gpu_capable]
    for kernel_name, spec in tqdm(cpu_specs, desc="Tuning CPU specs", leave=False):
        code_version = compute_code_version(*spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt)
        cache = KernelTuningCache()
        results[kernel_name] = _run_spec_tuning(
            cache, spec, code_version, device_id=None, force=force,
            idle_wait_tries=idle_wait_tries, idle_wait_sec=idle_wait_sec, hooks=hooks,
        )

    # GPU-capable specs: group by device model, run on least-loaded.
    gpu_specs = [(k, s) for k, s in specs.items() if s.gpu_capable]
    for kernel_name, spec in tqdm(gpu_specs, desc="Tuning GPU specs", leave=False):
        for model_name, device_ids in gpu_groups.items():
            # Pick least-loaded device in this group.
            device_id = _pick_least_loaded_device(device_ids, idle_wait_tries, idle_wait_sec)
            if device_id is None:
                logger.warning("No available device for %s; skipping %s", model_name, kernel_name)
                continue

            code_version = compute_code_version(*spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt)
            cache = KernelTuningCache()
            results[kernel_name] = _run_spec_tuning(
                cache, spec, code_version, device_id=device_id, force=force,
                idle_wait_tries=idle_wait_tries, idle_wait_sec=idle_wait_sec, hooks=hooks,
            )

    return results


def _group_gpus_by_model() -> dict[str, list[int]]:
    """Group GPUs by unique model (name + compute capability).

    Returns: {model_name: [device_ids], ...}
    """
    import GPUtil

    gpus = GPUtil.getGPUs()
    groups = {}
    for gpu in gpus:
        # Model name: abbreviated GPU name + compute capability.
        model = f"{gpu.name.split()[0]}_{gpu.compute_capability[0]}{gpu.compute_capability[1]}"
        groups.setdefault(model, []).append(gpu.id)
    return groups


def _pick_least_loaded_device(
    device_ids: list[int], idle_wait_tries: int, idle_wait_sec: float
) -> Optional[int]:
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


def tune_spec(
    spec: TunerSpec,
    *,
    force: bool = True,
    device_id: Optional[int] = None,
    idle_wait_tries: int = 5,
    idle_wait_sec: float = 0.5,
    hooks: Optional[Any] = None,
) -> int:
    """Tune a single registered spec (compute its code_version, run the sweep,
    persist) and return the number of regions now cached. The public single-spec
    entry point -- e.g. ``mlframe-tune-kernels refresh <kernel>``; ``retune_all``
    batches this across every spec + GPU model."""
    code_version = compute_code_version(*spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt)
    return _run_spec_tuning(
        KernelTuningCache(), spec, code_version, device_id=device_id, force=force,
        idle_wait_tries=idle_wait_tries, idle_wait_sec=idle_wait_sec, hooks=hooks,
    )


def _run_spec_tuning(
    cache,
    spec: TunerSpec,
    code_version: str,
    device_id: Optional[int],
    force: bool,
    idle_wait_tries: int,
    idle_wait_sec: float,
    hooks: Optional[Any],
) -> int:
    """Tune one spec via get_or_tune (forcing a fresh sweep when ``force``);
    return the number of regions now in the cache for this kernel.

    GPU specs run inside ``cp.cuda.Device(device_id)`` so the sweep measures on
    the chosen device; CPU specs (device_id None) run as-is. ``once_per_process``
    is False so each spec re-tunes even if a normal dispatch already ran it."""
    # equiv_tol on the spec is a {metric: tol} dict; the cache gate takes the
    # max-abs-diff float (surfaces + rejects divergent regions, never masks).
    tol = None
    if spec.equiv_tol:
        tol = spec.equiv_tol.get("max_abs_diff")

    def _tune():
        if force:
            cache.evict(spec.kernel_name)
        # dims={} -> no specific lookup point; this just drives the tuner +
        # persist. The return value is ignored; we count persisted regions.
        cache.get_or_tune(
            spec.kernel_name,
            dims={},
            tuner=spec.tuner,
            axes=list(spec.axes.keys()),
            fallback=spec.fallback,
            env_key=spec.env_key,
            code_version=code_version,
            salt=spec.salt,
            equiv_tol=tol,
            hooks=hooks,
            once_per_process=False,
        )

    if device_id is not None:
        try:
            import cupy as cp

            with cp.cuda.Device(device_id):
                _tune()
        except ImportError:
            _tune()
    else:
        _tune()

    regions = cache.get_regions(spec.kernel_name)
    return len(regions or [])
