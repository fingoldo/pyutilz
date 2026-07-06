# Kernel tuning cache

## Why this exists

When a project ships multiple CUDA / numba / cupy variants of the same hot numerical kernel, the fastest choice depends on the *live* GPU: a `cc6.1` GTX 1050 Ti and a datacenter `cc8.9` H100 have different memory bandwidths, SM counts, and shared-memory sizes, so a threshold hardcoded for one is wrong on the other. `pyutilz.performance.kernel_tuning` stores empirically-measured `(variant, block_size, ...)` decisions per hardware fingerprint and dispatches at runtime — no re-tuning by hand every time the code runs on different hardware.

This is genuinely novel infra, not a thin wrapper: mlframe's MRMR feature selection uses it for joint-histogram CUDA `RawKernel` dispatch (shared-mem vs global-atomic vs `numba.cuda`), measured at a 2.6x cumulative speedup at N=1M, p=30 on its reference hardware.

## Core pieces

- **`hw_fingerprint()`** — a stable string identifying the current host's compute environment, e.g. `"cpu_intel-i7-9700k_gpu_gtx-1050-ti_cc6.1"`. This is the cache's partition key: tunings measured on one machine never leak onto another.
- **`KernelTuningCache`** — loads or creates the per-host cache. Backed by immutable per-`(host, kernel, code_version)` JSON files under `~/.pyutilz/kernel_tuning/` (override via `$PYUTILZ_KERNEL_CACHE_DIR`). No `filelock`, no read-modify-write cycle — concurrent writers can never revert each other's fresher entry, because each write targets its own immutable file.
- **`compute_code_version()`** (`code_versioning.py`) — stamps provenance (CUDA driver/runtime version, cupy/numba/numpy versions, GPU summary) into the cache key, so a library upgrade that could change kernel behaviour is detected via `provenance_changed()` rather than silently serving a stale tuning.
- **`@kernel_tuner`** registry (`registry.py`) — `discover_tuners()` / `retune_all()` support re-tuning every registered kernel across multiple GPUs in one pass.
- **`RemoteBackend` / `S3Backend`** (`remote.py`) — a pluggable backend for sharing tunings across a fleet of hosts instead of re-measuring per-machine.

## Quick example

```python
from pyutilz.performance.kernel_tuning import KernelTuningCache, hw_fingerprint

cache = KernelTuningCache.load_or_create()
print(hw_fingerprint())                # "cpu_intel-i7-9700k_gpu_gtx-1050-ti_cc6.1"

# Project-side tuner emits per-region winners; pyutilz only stores them.
cache.update("joint_hist_batched", axes=["n_samples", "joint_size"], regions=[
    {"n_samples_max": 200_000, "joint_size_max": 25, "variant": "shared", "block_size": 256},
    {"n_samples_max": None,    "joint_size_max": None, "variant": "shared", "block_size": 512},  # catch-all
])

# Runtime dispatch:
region = cache.lookup("joint_hist_batched", n_samples=1_000_000, joint_size=100)
launch_kernel(variant=region["variant"], block=region["block_size"])
```

## Design principle: measure, don't guess

The project's convention (also documented in mlframe's own contributor notes) is: never hardcode a CUDA threshold or block size, and never assume "GPU is fast" from a single measurement. A dispatcher built on this cache benches all applicable backends across a size sweep first, saves the sweep as a runnable script, and only then writes the dispatch logic with thresholds derived from the actual measurements on that host — not from the dev machine's numbers copy-pasted into a constant.

## Residency awareness

`array_location(x)` reports `"device"` if `x` is GPU-resident (a cupy array or anything exposing `__cuda_array_interface__`, including numba device arrays) and `"host"` otherwise. Passing this as a dimension into a residency-tuned kernel lookup lets the cache pick the backend actually measured for where the data lives — a kernel that wins when its inputs are already on the GPU can lose once host-to-device transfer is added back in, so residency is tracked as its own axis rather than folded into `n_samples`.
