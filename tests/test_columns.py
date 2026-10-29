"""Behaviour of ``pyutilz.data.polarslib.columns``: the NaN and infinite column finders report only the columns that hold
such values, ``clean_numeric`` replaces non-finite floats while keeping nulls unless asked to fill them, the aggregate
guard cleans only the stats that go non-finite on near-constant data, and the f64 to f32 cast leaves other columns
alone. Polars is used directly on small in-memory frames."""

from __future__ import annotations

import polars as pl

from pyutilz.data.polarslib import columns


class TestFinders:
    def test_only_the_column_with_a_nan_is_reported(self):
        df = pl.DataFrame({"a": [1.0, float("nan")], "b": [1.0, 2.0]})
        assert columns.find_nan_cols(df).columns == ["a"]

    def test_only_the_column_with_an_infinity_is_reported(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [1.0, float("inf")]})
        assert columns.find_infinite_cols(df).columns == ["b"]

    def test_a_clean_frame_has_no_nan_or_infinite_columns(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        assert columns.find_nan_cols(df).columns == []
        assert columns.find_infinite_cols(df).columns == []


class TestCleanNumeric:
    def test_infinities_and_nans_are_replaced_with_the_filler(self):
        df = pl.DataFrame({"x": [1.0, float("inf"), float("-inf"), float("nan")]})
        out = df.select(columns.clean_numeric(pl.col("x"), nans_filler=-1.0)).to_series().to_list()
        assert out == [1.0, -1.0, -1.0, -1.0]

    def test_a_null_is_kept_as_null_by_default(self):
        df = pl.DataFrame({"x": [1.0, None]})
        assert df.select(columns.clean_numeric(pl.col("x"))).to_series().to_list() == [1.0, None]

    def test_fill_nulls_turns_a_null_into_the_filler_as_well(self):
        df = pl.DataFrame({"x": [1.0, None]})
        out = df.select(columns.clean_numeric(pl.col("x"), nans_filler=0.5, fill_nulls=True)).to_series().to_list()
        assert out == [1.0, 0.5]


class TestAggregateGuard:
    def test_skew_on_constant_data_is_cleaned_to_the_filler(self):
        df = pl.DataFrame({"k": [2.0, 2.0, 2.0]})
        raw = df.select(pl.col("k").skew()).to_series().to_list()[0]
        assert raw != raw  # NaN: the raw stat is non-finite
        assert df.select(columns.apply_agg_func_safe(pl.col("k").skew(), "skew")).to_series().to_list() == [0.0]

    def test_a_stat_not_prone_to_nan_passes_through_unchanged(self):
        df = pl.DataFrame({"k": [2.0, 2.0, 2.0]})
        assert df.select(columns.apply_agg_func_safe(pl.col("k").mean(), "mean")).to_series().to_list() == [2.0]


class TestCastAndDtypes:
    def test_float64_columns_become_float32(self):
        out = columns.cast_f64_to_f32(pl.DataFrame({"x": [1.5]}))
        assert out.schema["x"] == pl.Float32

    def test_integer_columns_are_cast_to_float32_too(self):
        out = columns.cast_f64_to_f32(pl.DataFrame({"x": [1.5], "y": [2]}))
        assert out.schema["y"] == pl.Float32

    def test_non_numeric_columns_keep_their_dtype(self):
        out = columns.cast_f64_to_f32(pl.DataFrame({"x": [1.5], "s": ["a"]}))
        assert out.schema["s"] == pl.Utf8

    def test_an_integer_beyond_the_exact_float32_range_is_logged(self, caplog):
        with caplog.at_level("WARNING"):
            columns.cast_f64_to_f32(pl.DataFrame({"big": [2**25]}))
        assert any("beyond 2**24" in rec.getMessage() for rec in caplog.records)

    def test_the_castable_integer_dtypes_include_the_common_ones(self):
        castable = columns.polars_castable_int_dtypes()
        assert pl.Int64 in castable and pl.UInt64 in castable
        assert pl.Float64 not in castable


class TestColumnBoundaries:
    def test_a_column_with_a_null_is_reported_by_the_nan_finder(self):
        df = pl.DataFrame({"a": [1.0, None], "b": [1.0, 2.0]})
        assert columns.find_nan_cols(df).columns == ["a"]

    def test_clean_numeric_defaults_to_a_zero_filler(self):
        df = pl.DataFrame({"x": [float("inf")]})
        assert df.select(columns.clean_numeric(pl.col("x"))).to_series().to_list() == [0.0]

    def test_the_castable_integer_dtypes_include_int128_when_polars_has_it(self):
        if hasattr(pl, "Int128"):
            assert pl.Int128 in columns.polars_castable_int_dtypes()

    def test_an_integer_exactly_at_the_float32_limit_is_not_flagged(self, caplog):
        with caplog.at_level("WARNING"):
            columns.cast_f64_to_f32(pl.DataFrame({"x": [2**24, -(2**24)]}))
        assert not any("beyond 2**24" in rec.getMessage() for rec in caplog.records)

    def test_an_integer_one_past_the_float32_limit_is_flagged_by_name(self, caplog):
        with caplog.at_level("WARNING"):
            columns.cast_f64_to_f32(pl.DataFrame({"x": [2**24 + 1]}))
        assert any("['x']" in rec.getMessage() for rec in caplog.records)

    def test_a_negative_integer_one_past_the_float32_limit_is_flagged(self, caplog):
        with caplog.at_level("WARNING"):
            columns.cast_f64_to_f32(pl.DataFrame({"x": [-(2**24) - 1]}))
        assert any("beyond 2**24" in rec.getMessage() for rec in caplog.records)

    def test_kurtosis_on_constant_data_is_cleaned_to_the_filler(self):
        df = pl.DataFrame({"k": [2.0, 2.0, 2.0]})
        assert df.select(columns.apply_agg_func_safe(pl.col("k").kurtosis(), "kurtosis")).to_series().to_list() == [0.0]
