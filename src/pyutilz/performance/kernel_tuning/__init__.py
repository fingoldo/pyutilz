"""Per-host kernel auto-tuning.

A code-versioned, residency- and multi-GPU-aware JSON cache that picks the
fastest implementation of a hot kernel (numpy / numba / cupy / CUDA, block size,
backend) per input-size region, keyed by a hardware fingerprint; the
``@kernel_tuner`` registry + discovery + multi-GPU ``retune_all``; and a
pluggable remote backend for sharing tunings across hosts.

Relocated from ``pyutilz.system`` -- kernel tuning is a *performance* concern.
Import the public API from here; the implementation lives in the ``cache`` /
``registry`` / ``remote`` / ``code_versioning`` submodules.
"""
from .benchmark import benchmark_backends, time_backend
from .cache import KernelTuningCache, cache_path, register_default_cache
from .code_versioning import compute_code_version
from .registry import TunerSpec, discover_tuners, get_registry, kernel_tuner, retune_all, tune_spec
from .remote import RemoteBackend, S3Backend


def array_location(x) -> str:
    """Residency of an array-like: ``"device"`` if GPU-resident (a cupy array or
    anything exposing ``__cuda_array_interface__`` -- incl. numba device arrays),
    else ``"host"``. The single detection point for residency-aware dispatch:
    pass the result as the ``location`` dim of a residency-tuned kernel so the
    cache picks the backend measured for where the data actually lives."""
    return "device" if hasattr(x, "__cuda_array_interface__") else "host"


__all__ = [
    "KernelTuningCache",
    "cache_path",
    "register_default_cache",
    "compute_code_version",
    "TunerSpec",
    "kernel_tuner",
    "discover_tuners",
    "retune_all",
    "tune_spec",
    "get_registry",
    "RemoteBackend",
    "S3Backend",
    "array_location",
    "time_backend",
    "benchmark_backends",
]
