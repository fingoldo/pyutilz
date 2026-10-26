"""Behaviour of ``pyutilz.core.array_summary.column_sum_min_max``: the fused kernel and the numpy fallback
must agree element for element, NaN must propagate per column, and bad shapes must raise."""

from __future__ import annotations

import numpy as np
import pytest

from pyutilz.core.array_summary import NUMBA_SUPPORTED_KINDS, column_sum_min_max


def _reference(arr: np.ndarray):
    """Three independent numpy reductions over every axis but the last -- the contract being tested."""
    axis = tuple(range(arr.ndim - 1))
    return (
        arr.sum(axis=axis, dtype=np.float64).ravel(),
        arr.min(axis=axis).astype(np.float64).ravel(),
        arr.max(axis=axis).astype(np.float64).ravel(),
    )


class TestMatchesNumpy:
    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8, np.float32, np.float64])
    def test_supported_dtypes_match_three_pass_numpy(self, dtype):
        rng = np.random.default_rng(0)
        arr = (rng.standard_normal((50, 4)) * 100).astype(dtype)
        got = column_sum_min_max(arr)
        want = _reference(arr)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=1e-12)

    def test_three_dimensional_array_reduces_every_axis_but_the_last(self):
        arr = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
        got = column_sum_min_max(arr)
        want = _reference(arr)
        for g, w in zip(got, want):
            assert g.shape == (4,)
            np.testing.assert_array_equal(g, w)

    def test_outputs_are_float64(self):
        arr = np.ones((5, 3), dtype=np.int32)
        for out in column_sum_min_max(arr):
            assert out.dtype == np.float64


class TestNaNPropagation:
    def test_a_nan_makes_its_own_column_nan_in_all_three_outputs(self):
        arr = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        sums, mins, maxs = column_sum_min_max(arr)
        assert np.isnan(sums[0]) and np.isnan(mins[0]) and np.isnan(maxs[0])
        assert not np.isnan(sums[1])
        assert mins[1] == 2.0
        assert maxs[1] == 5.0

    def test_a_nan_in_the_first_row_is_still_seen(self):
        arr = np.array([[np.nan], [1.0], [2.0]])
        sums, mins, maxs = column_sum_min_max(arr)
        assert np.isnan(mins[0]) and np.isnan(maxs[0])


class TestFallback:
    @pytest.mark.parametrize("dtype", [np.complex128, np.float16, np.bool_])
    def test_dtypes_outside_the_kernel_take_the_numpy_path(self, dtype):
        assert np.dtype(dtype).kind not in NUMBA_SUPPORTED_KINDS or dtype == np.float16
        arr = np.ones((4, 2), dtype=dtype)
        sums, mins, maxs = column_sum_min_max(arr)
        np.testing.assert_allclose(sums, _reference(arr)[0])
        np.testing.assert_allclose(mins, _reference(arr)[1])
        np.testing.assert_allclose(maxs, _reference(arr)[2])


class TestRejectedInputs:
    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="ndim >= 2"):
            column_sum_min_max(np.arange(5.0))

    def test_empty_input_raises_on_both_paths(self):
        with pytest.raises(ValueError, match="empty array"):
            column_sum_min_max(np.empty((0, 3), dtype=np.float64))
        with pytest.raises(ValueError, match="empty array"):
            column_sum_min_max(np.empty((0, 3), dtype=np.complex128))
