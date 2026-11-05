"""Performance tooling: per-host kernel auto-tuning, hardware-derived CUDA launch geometry (``gpu_hw_launch``) and benchmarking helpers."""

__all__ = [
    "block_size_for",
    "device_props",
    "fill_grid_1d",
    "fixed_shared_block_size",
    "gpu_hw_launch",
    "kernel_tuning",
    "launch_1d",
    "launch_device_id",
    "occupancy_block_candidates",
]

_GPU_HW_LAUNCH_NAMES = frozenset(
    {"block_size_for", "device_props", "fill_grid_1d", "fixed_shared_block_size", "launch_1d", "launch_device_id", "occupancy_block_candidates"}
)


def __getattr__(name):
    """Bind the subpackage's submodules, and the launch-geometry functions of ``gpu_hw_launch``, on attribute access (PEP 562), matching every sibling
    subpackage. Lazy rather than eager so that importing this package costs nothing until a tuner or a launch helper is actually touched.
    """
    import importlib

    if name in _GPU_HW_LAUNCH_NAMES:
        value = getattr(importlib.import_module(".gpu_hw_launch", __name__), name)
        globals()[name] = value
        return value
    if name in __all__:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
