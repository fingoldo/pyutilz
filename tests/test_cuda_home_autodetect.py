"""``_ensure_cuda_home_from_pip`` points CUDA_HOME at the pip nvidia-cuda-nvcc package when unset.

pip ``cupy-cuda12x`` + ``nvidia-cuda-nvcc-cu12`` ship CUDA under site-packages but set no env var; numba.cuda needs
CUDA_HOME, so without it the whole GPU / kernel-tuning stack silently disables. The helper bridges that gap.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from pyutilz.core.pythonlib import _ensure_cuda_home_from_pip


def _save():
    return {k: os.environ.get(k) for k in ("CUDA_HOME", "CUDA_PATH")}


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_does_not_override_existing_cuda_home():
    saved = _save()
    try:
        os.environ["CUDA_HOME"] = "/already/set/cuda"
        os.environ.pop("CUDA_PATH", None)
        _ensure_cuda_home_from_pip()
        assert os.environ["CUDA_HOME"] == "/already/set/cuda"  # must not clobber a caller's CUDA_HOME
    finally:
        _restore(saved)


def test_sets_cuda_home_from_pip_when_unset():
    try:
        import nvidia
        cand = pathlib.Path(nvidia.__file__).parent / "cuda_nvcc"
    except Exception:
        cand = None
    if cand is None or not (cand / "nvvm").exists():
        pytest.skip("pip nvidia-cuda-nvcc not installed")
    saved = _save()
    try:
        os.environ.pop("CUDA_HOME", None)
        os.environ.pop("CUDA_PATH", None)
        _ensure_cuda_home_from_pip()
        assert os.environ.get("CUDA_HOME") == str(cand)
        assert os.environ.get("CUDA_PATH") == str(cand)
    finally:
        _restore(saved)


def test_graceful_no_raise_when_unset():
    saved = _save()
    try:
        os.environ.pop("CUDA_HOME", None)
        os.environ.pop("CUDA_PATH", None)
        _ensure_cuda_home_from_pip()  # must never raise, whatever the environment
    finally:
        _restore(saved)
