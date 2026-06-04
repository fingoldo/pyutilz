"""Kernel tuner registry: @kernel_tuner decorator + TunerSpec + discovery.

A tuner spec defines a hot kernel variant, how to measure it, what hardware it
targets, and how to gate the decision. Each spec registers at import time (cheap;
no heavy deps). Discovery walks the mlframe package (only when explicitly
invoked, never at `import mlframe`) and collects all specs for batch tuning via
retune_all() or the CLI.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "TunerSpec",
    "kernel_tuner",
    "discover_tuners",
    "retune_all",
    "get_registry",
]

# Global registry: (module_name, kernel_name) -> TunerSpec
_REGISTRY = {}


@dataclass
class TunerSpec:
    """Specification for one kernel variant, its tuning, and dispatch decision."""

    kernel_name: str
    """Name of this variant (e.g. 'joint_hist_2d', 'mi_classif_gpu'). Unique
    within its module."""

    variant_fns: tuple
    """One or more callables (the kernel implementations to benchmark). Passed
    to compute_code_version() for version-stable hashing."""

    tuner: Callable[..., dict[str, Any]]
    """Function that runs a sweep. Signature: tuner(dims_dict, axes_dict,
    fallback) -> {region_key: decision_dict}. Called by get_or_tune() on cache
    miss."""

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


def kernel_tuner(**kwargs) -> Callable:
    """Decorator registering a TunerSpec at the module's import time.

    Usage:
        @kernel_tuner(
            kernel_name='joint_hist_2d',
            variant_fns=(_joint_hist_numpy, _joint_hist_numba),
            tuner=tune_joint_hist,
            axes={'ndim_eq': [2], 'n_max': [100, 1000, 10000]},
            fallback=_joint_hist_numpy,
            gpu_capable=True,
        )
        def _joint_hist_gpu():
            return _joint_hist_cupy

    The decorated function is NOT called; it's metadata only. Registration
    happens immediately at the module's import, before __init__.py finishes.
    """

    spec = TunerSpec(**kwargs)

    def decorator(fn):
        # Record the spec in the global registry, keyed by module + kernel name.
        # Module path is inferred from fn.__module__.
        module_name = fn.__module__
        key = (module_name, spec.kernel_name)
        if key in _REGISTRY:
            raise ValueError(
                f"Duplicate kernel_tuner registration: {module_name}:{spec.kernel_name}"
            )
        _REGISTRY[key] = spec
        # Return the original function unmodified (not called by the decorator).
        return fn

    return decorator


def get_registry() -> dict:
    """Return the global tuner registry: {(module, kernel_name): TunerSpec}."""
    return dict(_REGISTRY)


def discover_tuners(
    package: str = "mlframe", warn_on_import_fail: bool = True
) -> dict[tuple[str, str], TunerSpec]:
    """Walk a package, import all modules, and collect @kernel_tuner specs.

    Algorithm: use pkgutil.walk_packages to enumerate all modules under
    `package`. For each module, attempt import. If the import fails, log a
    WARNING (not silently swallowed) and skip that module — the remaining
    modules' specs are still collected.

    Returns:
        {(module_name, kernel_name): TunerSpec, ...}

    Side effects:
        - Clears the global registry and repopulates it (so subsequent
          discover_tuners() calls give the freshest state).
        - Logs WARNINGs for any import failures.
    """
    import importlib
    import pkgutil
    import logging

    logger = logging.getLogger(__name__)
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
) -> dict[tuple[str, str], int]:
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
        {(module_name, kernel_name): n_regions_tuned, ...}

    Side effects:
        - Calls discover_tuners (clears + repopulates registry).
        - Updates cache files under the host's hw_fingerprint.
    """
    import logging
    from tqdm import tqdm

    logger = logging.getLogger(__name__)
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
    for (mod_name, kernel_name), spec in tqdm(
        cpu_specs, desc="Tuning CPU specs", leave=False
    ):
        from .kernel_tuning_cache import KernelTuningCache
        from .code_versioning import compute_code_version

        code_version = compute_code_version(
            *spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt
        )
        cache = KernelTuningCache()
        n_regions = _run_spec_tuning(
            cache,
            spec,
            code_version,
            device_id=None,
            force=force,
            idle_wait_tries=idle_wait_tries,
            idle_wait_sec=idle_wait_sec,
            hooks=hooks,
        )
        results[(mod_name, kernel_name)] = n_regions

    # GPU-capable specs: group by device model, run on least-loaded.
    gpu_specs = [(k, s) for k, s in specs.items() if s.gpu_capable]
    for (mod_name, kernel_name), spec in tqdm(
        gpu_specs, desc="Tuning GPU specs", leave=False
    ):
        for model_name, device_ids in gpu_groups.items():
            # Pick least-loaded device in this group.
            device_id = _pick_least_loaded_device(device_ids, idle_wait_tries, idle_wait_sec)
            if device_id is None:
                logger.warning(
                    "No available device for %s; skipping %s:%s",
                    model_name, mod_name, kernel_name,
                )
                continue

            from .kernel_tuning_cache import KernelTuningCache
            from .code_versioning import compute_code_version

            code_version = compute_code_version(
                *spec.variant_fns, extra_fns=spec.extra_fns, salt=spec.salt
            )
            cache = KernelTuningCache()
            n_regions = _run_spec_tuning(
                cache,
                spec,
                code_version,
                device_id=device_id,
                force=force,
                idle_wait_tries=idle_wait_tries,
                idle_wait_sec=idle_wait_sec,
                hooks=hooks,
            )
            results[(mod_name, kernel_name)] = n_regions

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
    import time

    import GPUtil

    for attempt in range(idle_wait_tries):
        gpus = {g.id: g.load for g in GPUtil.getGPUs() if g.id in device_ids}
        available = [d for d, load in gpus.items() if load <= 0.8]
        if available:
            return min(available, key=lambda d: gpus[d])  # Least loaded.
        if attempt < idle_wait_tries - 1:
            time.sleep(idle_wait_sec)
    return None


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
    """Run one spec via get_or_tune, return n_regions."""
    # This is a placeholder orchestration. In practice, you'd call:
    #   cache.get_or_tune(
    #       kernel=spec.kernel_name,
    #       dims={},
    #       tuner=spec.tuner,
    #       axes=spec.axes,
    #       fallback=spec.fallback,
    #       code_version=code_version,
    #       env_key=spec.env_key,
    #       hooks=hooks,
    #       once_per_process=False,
    #   )
    # And handle device selection via CUDA_VISIBLE_DEVICES or similar.
    # For now, return a placeholder.
    return 0
