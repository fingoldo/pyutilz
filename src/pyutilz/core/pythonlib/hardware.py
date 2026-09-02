"""Runtime environment probing: Jupyter detection, CUDA availability, CPU instruction-set flags.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

from ._common import logger, lru_cache, os


def is_jupyter_notebook():
    """Detects whether the code is currently running inside a Jupyter notebook/JupyterLab kernel."""
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False
        if "IPKernelApp" in ipython.config:  # Jupyter notebook or JupyterLab
            return True
        return False
    except (ImportError, NameError):
        return False


def _ensure_cuda_home_from_pip() -> None:
    """Point CUDA_HOME/CUDA_PATH at the pip ``nvidia-cuda-nvcc`` package when they are unset.

    pip-installed ``cupy-cuda12x`` + ``nvidia-cuda-nvcc-cu12`` ship nvvm/ptxas/libdevice/headers under
    ``site-packages/nvidia/cuda_nvcc/`` but set NO env var. cupy locates them via cuda-pathfinder, but numba.cuda
    relies on CUDA_HOME/CUDA_PATH -- so without a system CUDA toolkit AND without these vars, ``numba.cuda.is_available()``
    silently returns False and every GPU/kernel-tuning path is disabled. Set the vars from the pip package so numba
    finds the same CUDA cupy already uses. Must run before the first ``cuda.is_available()`` probe (numba caches it).

    Deliberately NOT ``lru_cache``d, even though it is idempotent: its only production caller is
    ``is_cuda_available``, which IS cached, so memoizing here buys nothing measurable (the 6-7us of
    the early-return path is paid once per process) while it WOULD freeze the env-var read, which is
    exactly what its tests vary."""
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    try:
        import nvidia
        import pathlib as _pathlib

        cand = _pathlib.Path(nvidia.__file__).parent / "cuda_nvcc"
        if (cand / "nvvm").exists():
            os.environ["CUDA_HOME"] = str(cand)
            os.environ["CUDA_PATH"] = str(cand)
    except Exception as e:  # nosec B110 - best-effort optional CUDA env-var setup; failing here (e.g. nvidia package absent) just leaves CUDA_HOME unset, which is the pre-existing safe default
        logger.debug("Failed to set CUDA_HOME/CUDA_PATH from pip nvidia package: %s", e)


@lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """Check if CUDA is available via numba.

    Memoized for the process lifetime: CUDA availability cannot change inside a running process
    (numba caches its own probe too), yet this is called PER DISPATCH DECISION by
    ``gpu_dispatch.dispatch_cpu_vs_gpu``, where the uncached probe measured 16.5-31.5 us against
    0.105 us cached -- for a small work item the decision cost more than the work. Tests that mock
    the probe must call ``is_cuda_available.cache_clear()`` (``gpu_dispatch.reset_cache()`` does it
    for them).

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        _ensure_cuda_home_from_pip()
        from numba import cuda
        return cuda.is_available()  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
    except (ImportError, Exception) as e:  # nosec B110 - best-effort optional CUDA probe; failing here (e.g. numba absent or no driver) just reports no CUDA support
        logger.debug("Failed to probe CUDA availability: %s", e)
        return False


def check_cpu_flag(flag: str = "avx2") -> bool:
    """Check if CPU supports a specific instruction set flag.

    Args:
        flag: CPU flag to check (e.g., "avx2", "sse4_2", "avx512f")

    Returns:
        bool: True if flag is supported, False otherwise
    """
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
    except Exception as e:  # nosec B110 - best-effort optional CPU-flag probe; py-cpuinfo absent or raising just reports flag unsupported
        logger.debug("Failed to probe CPU flag %r: %s", flag, e)
        return False
    flags = info.get("flags") if isinstance(info, dict) else None
    if flags is None:
        # py-cpuinfo returns {} on several platforms and under some virtualization, and its schema
        # has changed across releases. That is NOT the same as "the CPU lacks this flag", and
        # silently reporting it as such disables every SIMD fast path on a capable machine.
        logger.warning("check_cpu_flag(%r): py-cpuinfo reported no 'flags' key (info keys: %s); assuming unsupported", flag, sorted(info) if isinstance(info, dict) else type(info).__name__)
        return False
    return flag in flags
