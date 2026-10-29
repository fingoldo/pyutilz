"""Behaviour of ``pyutilz.data.polarslib.binning``: the Shannon entropy of binned values is measured in nats, nulls count as their
own bin unless asked to be dropped, a bin dtype is widened only when it cannot hold ``num_bins - 1``, and outlier clipping
follows the feature and target flags. Frames are small polars frames built in memory."""

from __future__ import annotations

import math

import polars as pl
import pytest

from pyutilz.data.polarslib import binning


class TestEntropy:
    def test_a_uniform_binary_column_has_one_bit_of_entropy_in_nats(self):
        df = pl.DataFrame({"x": [0, 0, 1, 1]})
        assert binning.entropy_for_column(df, "x") == pytest.approx(math.log(2))

    def test_a_constant_column_has_no_entropy(self):
        df = pl.DataFrame({"x": [7, 7, 7]})
        assert binning.entropy_for_column(df, "x") == pytest.approx(0.0)

    def test_a_null_counts_as_its_own_bin_by_default(self):
        df = pl.DataFrame({"x": pl.Series([0, 1, None, None], dtype=pl.Int32)})
        assert binning.entropy_for_column(df, "x") == pytest.approx(1.5 * math.log(2))

    def test_dropping_nulls_measures_the_observed_values_only(self):
        df = pl.DataFrame({"x": pl.Series([0, 1, None, None], dtype=pl.Int32)})
        assert binning.entropy_for_column(df, "x", drop_nulls=True) == pytest.approx(math.log(2))


class TestBinDtypeWidening:
    def test_a_dtype_that_holds_the_largest_bin_is_kept(self):
        assert binning._resolve_bin_dtype(pl.Int8, 128, False) == pl.Int8

    def test_a_dtype_too_small_for_the_largest_bin_is_widened(self):
        assert binning._resolve_bin_dtype(pl.Int8, 300, False) == pl.Int16

    def test_a_dtype_outside_the_table_is_returned_unchanged(self):
        assert binning._resolve_bin_dtype(None, 300, False) is None


class TestClipScope:
    def test_a_target_is_left_alone_when_targets_are_not_cleaned(self):
        assert binning._should_clip_column("y", True, False, ["y"]) is False

    def test_a_feature_is_left_alone_when_only_targets_are_cleaned(self):
        assert binning._should_clip_column("z", False, True, ["y"]) is False

    def test_a_target_is_clipped_when_only_targets_are_cleaned(self):
        assert binning._should_clip_column("y", False, True, ["y"]) is True

    def test_everything_is_in_scope_when_both_are_cleaned(self):
        assert binning._should_clip_column("z", True, True, ["y"]) is True


class TestBinDtypeBoundaries:
    def test_an_unsigned_dtype_that_holds_the_largest_bin_is_kept(self):
        assert binning._resolve_bin_dtype(pl.UInt16, 65536, False) == pl.UInt16

    def test_an_unsigned_dtype_one_bin_too_small_widens_to_int32(self):
        assert binning._resolve_bin_dtype(pl.UInt16, 65537, False) == pl.Int32

    def test_a_signed_dtype_holds_exactly_its_largest_bin(self):
        assert binning._resolve_bin_dtype(pl.Int16, 32768, False) == pl.Int16

    def test_a_signed_dtype_one_bin_too_small_widens(self):
        assert binning._resolve_bin_dtype(pl.Int16, 32769, False) == pl.Int32

    def test_an_int8_frame_widens_to_int16_past_127(self):
        assert binning._resolve_bin_dtype(pl.Int8, 129, False) == pl.Int16


class TestFrequenciesAndMutualInformation:
    def test_an_empty_frame_has_no_frequencies(self):
        assert binning._group_freqs(pl.DataFrame({"a": []}, schema={"a": pl.Int32}), "a").size == 0

    def test_each_of_four_equally_likely_pairs_has_a_quarter(self):
        df = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
        assert list(binning._group_freqs(df, ["a", "b"])) == [0.25, 0.25, 0.25, 0.25]

    def test_an_identical_pair_shares_all_its_information(self):
        df = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]})
        entropies = {"a": math.log(2), "b": math.log(2)}
        assert binning.mi_for_column(df, entropies, "a", "b") == pytest.approx(math.log(2))

    def test_dropping_nulls_ignores_the_precomputed_entropies(self):
        df = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]})
        assert binning.mi_for_column(df, {}, "a", "b", drop_nulls=True) == pytest.approx(math.log(2))
