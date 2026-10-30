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


class TestKernelSelection:
    """Which path runs is observable only by watching the kernel, so each test replaces it with a recorder."""

    @pytest.fixture
    def recorder(self, monkeypatch):
        from pyutilz.core import array_summary

        calls = []

        def _fake(a2):
            calls.append(a2.shape)
            return _reference(a2)

        monkeypatch.setattr(array_summary, "_fused_col_reductions", _fake)
        monkeypatch.setattr(array_summary, "_HAVE_NUMBA", True)
        return calls

    @pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.float32, np.float64])
    def test_supported_kinds_go_through_the_kernel(self, recorder, dtype):
        column_sum_min_max(np.ones((4, 2), dtype=dtype))
        assert len(recorder) == 1

    @pytest.mark.parametrize("dtype", [np.complex128, np.float16, np.bool_])
    def test_unsupported_kinds_never_reach_the_kernel(self, recorder, dtype):
        column_sum_min_max(np.ones((4, 2), dtype=dtype))
        assert recorder == []

    def test_no_numba_means_the_kernel_is_never_called(self, recorder, monkeypatch):
        from pyutilz.core import array_summary

        monkeypatch.setattr(array_summary, "_HAVE_NUMBA", False)
        column_sum_min_max(np.ones((4, 2), dtype=np.float64))
        assert recorder == []

    def test_the_kernel_receives_the_last_axis_as_columns_for_a_three_dimensional_input(self, recorder):
        column_sum_min_max(np.zeros((2, 3, 4), dtype=np.float64))
        assert recorder == [(6, 4)]

    def test_the_supported_kinds_are_exactly_signed_unsigned_and_real_float(self):
        assert NUMBA_SUPPORTED_KINDS == frozenset("iuf")


class TestFallbackLogging:
    def test_a_kernel_failure_falls_back_to_numpy_and_logs_the_reason(self, monkeypatch, caplog):
        from pyutilz.core import array_summary

        def _boom(a2):
            raise ValueError("no such layout")

        monkeypatch.setattr(array_summary, "_fused_col_reductions", _boom)
        monkeypatch.setattr(array_summary, "_HAVE_NUMBA", True)
        arr = np.array([[1.0, 4.0], [3.0, 2.0]])
        with caplog.at_level("DEBUG", logger="pyutilz.core.array_summary"):
            got = column_sum_min_max(arr)
        for g, w in zip(got, _reference(arr)):
            np.testing.assert_array_equal(g, w)
        messages = [rec.getMessage() for rec in caplog.records if rec.name == "pyutilz.core.array_summary"]
        assert messages == ["Fused column reduction unavailable (no such layout), falling back to numpy"]

    def test_a_successful_kernel_logs_nothing(self, monkeypatch, caplog):
        from pyutilz.core import array_summary

        monkeypatch.setattr(array_summary, "_fused_col_reductions", lambda a2: _reference(a2))
        monkeypatch.setattr(array_summary, "_HAVE_NUMBA", True)
        with caplog.at_level("DEBUG", logger="pyutilz.core.array_summary"):
            column_sum_min_max(np.ones((3, 2)))
        assert [rec for rec in caplog.records if rec.name == "pyutilz.core.array_summary"] == []


class TestEmptyInputMessage:
    def test_the_message_names_the_shape_that_was_empty(self):
        with pytest.raises(ValueError, match=r"empty array \(shape \(0, 3\)\)"):
            column_sum_min_max(np.empty((0, 3), dtype=np.float64))


class TestNumpyFallbackOrdersMinAndMax:
    def test_the_fallback_keeps_min_and_max_distinct_for_non_kernel_dtypes(self):
        """A constant array hides a swapped min and max; distinct values per column do not."""
        arr = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.bool_)
        got = column_sum_min_max(arr)
        for g, w in zip(got, _reference(arr)):
            np.testing.assert_array_equal(g, w)
        assert list(got[1]) == [0.0, 0.0] and list(got[2]) == [1.0, 1.0]

    def test_the_fallback_matches_numpy_on_distinct_complex_values(self):
        arr = np.array([[1 + 0j, 5 + 0j], [3 + 0j, 2 + 0j], [0 + 0j, 9 + 0j]], dtype=np.complex128)
        got = column_sum_min_max(arr)
        want = _reference(arr)
        np.testing.assert_array_equal(got[1], want[1])
        np.testing.assert_array_equal(got[2], want[2])


class TestSingleRow:
    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    def test_a_one_row_array_summarises_to_the_row_itself_on_the_real_kernel(self, dtype):
        """The kernel seeds each column from its first row, so a one-row input has nothing else to read."""
        arr = np.array([[3, -7, 0]], dtype=dtype)
        sums, mins, maxs = column_sum_min_max(arr)
        np.testing.assert_array_equal(sums, np.array([3.0, -7.0, 0.0]))
        np.testing.assert_array_equal(mins, sums)
        np.testing.assert_array_equal(maxs, sums)
