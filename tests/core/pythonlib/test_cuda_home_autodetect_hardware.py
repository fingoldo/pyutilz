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


def test_does_not_override_existing_cuda_home(monkeypatch):
    monkeypatch.setenv("CUDA_HOME", "/already/set/cuda")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    _ensure_cuda_home_from_pip()
    assert os.environ["CUDA_HOME"] == "/already/set/cuda"  # must not clobber a caller's CUDA_HOME


def test_sets_cuda_home_from_pip_when_unset(monkeypatch):
    # `except ImportError`, not a broad `except Exception`: only a MISSING package may skip this
    # test; anything else (a broken nvidia package, a namespace pkg with no __file__) must fail.
    try:
        import nvidia
    except ImportError:
        pytest.skip("pip nvidia-cuda-nvcc not installed")
    cand = pathlib.Path(nvidia.__file__).parent / "cuda_nvcc"
    if not (cand / "nvvm").exists():
        pytest.skip("pip nvidia-cuda-nvcc not installed")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    _ensure_cuda_home_from_pip()
    home_is_cand = os.environ.get("CUDA_HOME") == str(cand)
    path_is_cand = os.environ.get("CUDA_PATH") == str(cand)
    assert home_is_cand, "CUDA_HOME was not pointed at the pip nvcc directory"
    assert path_is_cand, "CUDA_PATH was not pointed at the pip nvcc directory"


def test_graceful_no_raise_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    _ensure_cuda_home_from_pip()  # must never raise, whatever the environment
    # Either it found nothing (both stay unset), or it set BOTH to one really-existing dir:
    # a half-set or bogus CUDA_HOME is worse than none, since numba then fails deeper in.
    home, path = os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")
    consistent = (home is None) == (path is None)
    assert consistent, "CUDA_HOME and CUDA_PATH were set inconsistently"
    if home is not None:
        same = home == path
        assert same, "CUDA_HOME and CUDA_PATH differ"
        exists = pathlib.Path(home).is_dir()
        assert exists, "CUDA_HOME points at a non-existent directory"


def test_is_cuda_available_returns_false_and_logs_on_probe_failure(monkeypatch, caplog):
    """A driver/runtime error inside numba.cuda.is_available() must be swallowed, not raised, and logged."""
    import numba.cuda as numba_cuda

    def _raise(*args, **kwargs):
        raise AttributeError("simulated driver failure")

    monkeypatch.setattr(numba_cuda, "is_available", _raise)
    # ``is_cuda_available`` is memoized for the process (the answer cannot change inside one), so a
    # test that mocks the underlying probe must drop the cached verdict first -- and again after, so
    # it does not hand the mocked answer to whatever runs next.
    is_cuda_available.cache_clear()
    try:
        with caplog.at_level(logging.DEBUG, logger="pyutilz.core.pythonlib"):
            result = is_cuda_available()
    finally:
        is_cuda_available.cache_clear()
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
