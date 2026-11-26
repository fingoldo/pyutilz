"""Mutation-survivor pins for core/array_summary.py: the numba-missing fallback, the kernel's compile flags, and the
fast path's reshape (array_summary.py 36, 41, 105)."""

import builtins
import importlib.util
import logging

import numpy as np
import pytest

from pyutilz.core import array_summary as asum


def _fresh_module_without_numba(monkeypatch):
    real_import = builtins.__import__

    def no_numba(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_numba)
    spec = importlib.util.spec_from_file_location("array_summary_no_numba", asum.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNumbaMissingFallbackSurvivors:
    """Without numba the module records that it is missing and every reduction takes the numpy path (array_summary.py 36)."""

    def test_the_module_records_that_numba_is_missing(self, monkeypatch):
        module = _fresh_module_without_numba(monkeypatch)
        assert module._HAVE_NUMBA is False

    def test_reductions_still_answer_through_numpy(self, monkeypatch):
        module = _fresh_module_without_numba(monkeypatch)
        arr = np.arange(12, dtype=np.float64).reshape(4, 3)
        s, lo, hi = module.column_sum_min_max(arr)
        assert np.array_equal(s, arr.sum(axis=0))
        assert np.array_equal(lo, arr.min(axis=0))
        assert np.array_equal(hi, arr.max(axis=0))


class TestKernelCompileFlagsSurvivors:
    """The fused kernel is compiled with parallel and nogil on (array_summary.py 41)."""

    @pytest.mark.skipif(not asum._HAVE_NUMBA, reason="numba not installed")
    def test_the_kernel_is_compiled_parallel_and_without_the_gil(self):
        options = asum._fused_col_reductions.targetoptions
        assert options.get("parallel") is True
        assert options.get("nogil") is True


class TestFastPathReshapeSurvivors:
    """A valid float64 array takes the fused kernel without ever reaching the fallback log (array_summary.py 105)."""

    @pytest.mark.skipif(not asum._HAVE_NUMBA, reason="numba not installed")
    def test_a_valid_array_does_not_fall_back_to_numpy(self, caplog):
        arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        with caplog.at_level(logging.DEBUG, logger=asum.logger.name):
            asum.column_sum_min_max(arr)
        assert "falling back to numpy" not in caplog.text
