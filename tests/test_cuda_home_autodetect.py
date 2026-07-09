"""``_ensure_cuda_home_from_pip`` points CUDA_HOME at the pip nvidia-cuda-nvcc package when unset.

pip ``cupy-cuda12x`` + ``nvidia-cuda-nvcc-cu12`` ship CUDA under site-packages but set no env var; numba.cuda needs
CUDA_HOME, so without it the whole GPU / kernel-tuning stack silently disables. The helper bridges that gap.
"""
from __future__ import annotations

import os
import pathlib

import pytest

import logging

from pyutilz.core.pythonlib import _ensure_cuda_home_from_pip, is_cuda_available, check_cpu_flag


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


def test_is_cuda_available_returns_false_and_logs_on_probe_failure(monkeypatch, caplog):
    """A driver/runtime error inside numba.cuda.is_available() must be swallowed, not raised, and logged."""
    import numba.cuda as numba_cuda

    def _raise(*args, **kwargs):
        raise AttributeError("simulated driver failure")

    monkeypatch.setattr(numba_cuda, "is_available", _raise)
    with caplog.at_level(logging.DEBUG, logger="pyutilz.core.pythonlib"):
        result = is_cuda_available()
    assert result is False
    assert any("CUDA" in record.message for record in caplog.records)


def test_check_cpu_flag_returns_false_and_logs_on_missing_flags_key(monkeypatch, caplog):
    """A cpuinfo payload missing the 'flags' key must be swallowed, not raised, and logged."""
    import cpuinfo

    monkeypatch.setattr(cpuinfo, "get_cpu_info", lambda: {"brand_raw": "fake cpu"})
    with caplog.at_level(logging.DEBUG, logger="pyutilz.core.pythonlib"):
        result = check_cpu_flag("avx2")
    assert result is False
    assert any("avx2" in record.message for record in caplog.records)
