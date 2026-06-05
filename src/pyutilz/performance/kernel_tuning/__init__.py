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
from .cache import KernelTuningCache, cache_path
from .code_versioning import compute_code_version
from .registry import TunerSpec, discover_tuners, get_registry, kernel_tuner, retune_all, tune_spec
from .remote import RemoteBackend, S3Backend

__all__ = [
    "KernelTuningCache",
    "cache_path",
    "compute_code_version",
    "TunerSpec",
    "kernel_tuner",
    "discover_tuners",
    "retune_all",
    "tune_spec",
    "get_registry",
    "RemoteBackend",
    "S3Backend",
]
