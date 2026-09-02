"""Meta-test: no wall-clock measurement in pyutilz times GPU work without synchronizing the device.

A CUDA launch is asynchronous, so a timer stopped straight after it measures the launch, not the
compute. Measured on this repo's box with a cupy 4000x4000 float32 matmul: 0.0366 ms unsynchronized
vs 69.42 ms synchronized, a 1894x under-count. ``time_backend`` shipped exactly that defect, and its
output is written into the on-disk kernel-tuning cache -- which is read back in later sessions and
by mlframe, so one unsynchronized timer locks the wrong backend in permanently and across projects.
Nothing about the number looks wrong once it is in the cache, which is why this has to be caught at
the source rather than by inspecting results.

The rule lives in ``py_ci_shared.gpu_timing_sync`` (unit-tested there) because mlframe carries the
same exposure; this file only points it at pyutilz's source tree. Blocking, no baseline: the tree is
at zero, and a deliberate launch-latency measurement declares itself with an in-source
``gpu-timing-async-intentional`` comment instead of a grandfathered entry.

Static only -- no CUDA device, no cupy, no GPU of any kind is needed to run it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# py-ci-shared requires python>=3.9 (dev-dependency marker), so it's absent on the 3.8 CI leg --
# skip cleanly there instead of erroring at collection.
gpu_timing_sync = pytest.importorskip("py_ci_shared.gpu_timing_sync", reason="py-ci-shared is a dev-only git dependency (requirements-dev.txt)")

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
SRC_ROOT = PYUTILZ_DIR.parent

# Keys are "<path relative to src/>::<function>". Empty by design: prefer the in-source
# ``gpu-timing-async-intentional`` marker, which keeps the justification beside the measurement.
ALLOWLIST: frozenset = frozenset()


def test_no_unsynchronized_gpu_timings_in_pyutilz() -> None:
    gpu_timing_sync.assert_no_unsynchronized_gpu_timings(sorted(PYUTILZ_DIR.rglob("*.py")), root=SRC_ROOT, allowlist=ALLOWLIST)


def test_checker_flags_the_kernel_tuning_regression_shape(tmp_path: Path) -> None:
    """The gate must be able to fail, on the exact shape ``time_backend`` shipped.

    Without this the meta-test above is indistinguishable from a check that scans nothing -- the
    recurring failure mode across this repo's audit waves has been a gate that passes because it
    never actually looked.
    """
    regression_shape = '''
import time
from typing import Callable


def time_backend(fn, make_inputs, *, n_iters=2, timer: Callable[[], float] = time.perf_counter) -> float:
    """Median per-call wall time of a CPU or GPU backend, contended (see GPUtil idle gate)."""

    def _run(inputs_list, out):
        for args in inputs_list:
            t0 = timer()
            fn(*args)
            out.append(timer() - t0)

    samples: list = []
    _run([make_inputs() for _ in range(n_iters)], samples)
    return min(samples)
'''
    path = tmp_path / "benchmark.py"
    path.write_text(regression_shape.lstrip("\n"), encoding="utf-8")
    findings = gpu_timing_sync.find_unsynchronized_gpu_timings([path])
    assert [(f.function, f.shape) for f in findings] == [("_run", "injected-callable-in-gpu-module")]
