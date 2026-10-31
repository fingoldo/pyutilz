"""Behaviour of ``pyutilz.data.polarslib.frames``: the pandas-style ``polars_df_info`` summary text, and
``drop_constant_columns`` (with its alias): which numeric columns are dropped, how the caller's frame is left
alone, the warning text with its thousands separator, and the frames it passes through unchanged."""

import logging
import re

import polars as pl

from pyutilz.data.polarslib import frames


class TestPolarsDfInfo:
    def test_lists_type_rows_columns_and_dtypes_in_that_order(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        lines = frames.polars_df_info(df).split("\n")
        assert lines[0] == f"{type(df)}"
        assert lines[1] == "Rows: 3"
        assert lines[2] == "Columns: 2 entries, a to b"
        assert lines[3] == f"dtypes: {', '.join(sorted([f'{pl.Int64}(1)', f'{pl.String}(1)']))}"
        assert len(lines) == 5

    def test_repeated_dtypes_are_counted(self):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert f"dtypes: {pl.Int64}(3)" in frames.polars_df_info(df).split("\n")

    def test_an_empty_frame_has_zero_columns_and_no_dtypes(self):
        lines = frames.polars_df_info(pl.DataFrame()).split("\n")
        assert lines[1] == "Rows: 0"
        assert lines[2] == "Columns: 0 entries"
        assert lines[3] == "dtypes: "

    def test_a_small_frame_is_reported_in_bytes(self):
        assert re.search(r"memory usage: \d+\.0\+ bytes$", frames.polars_df_info(pl.DataFrame({"a": [1]})))

    def test_a_large_frame_is_reported_in_megabytes(self):
        df = pl.DataFrame({"a": pl.arange(0, 3_000_000, eager=True)})
        assert re.search(r"memory usage: \d+\.\d\+ MB$", frames.polars_df_info(df))


class TestDropConstantColumns:
    def test_a_column_with_equal_min_and_max_is_dropped(self):
        df = pl.DataFrame({"a": [1, 1], "b": [1, 2], "s": ["x", "y"]})
        assert frames.drop_constant_columns(df).columns == ["b", "s"]

    def test_the_callers_frame_is_not_changed(self):
        df = pl.DataFrame({"a": [1, 1], "b": [1, 2]})
        frames.drop_constant_columns(df)
        assert df.columns == ["a", "b"]

    def test_a_column_of_nulls_has_no_min_and_is_dropped(self):
        df = pl.DataFrame({"n": [None, None], "b": [1, 2]}, schema={"n": pl.Float64, "b": pl.Int64})
        assert frames.drop_constant_columns(df).columns == ["b"]

    def test_a_frame_without_numeric_columns_is_returned_as_is(self):
        df = pl.DataFrame({"s": ["x", "y"]})
        assert frames.drop_constant_columns(df) is df

    def test_the_alias_is_the_same_function(self):
        assert frames.remove_constant_columns is frames.drop_constant_columns

    def test_verbose_logs_the_count_with_an_underscore_separator_and_the_names(self, caplog):
        df = pl.DataFrame({f"c{i}": [1, 1] for i in range(1000)} | {"keep": [1, 2]})
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.drop_constant_columns(df, verbose=True)
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert messages[0].startswith("Dropping 1_000 columns with no change: c0, c1, c2")

    def test_a_quiet_call_logs_nothing(self, caplog):
        df = pl.DataFrame({"a": [1, 1]})
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.drop_constant_columns(df)
        assert caplog.records == []


class _Sized:
    """Stands in for a frame of a chosen size: only the attributes ``polars_df_info`` reads."""

    height = 1
    width = 1
    columns = ["a"]
    dtypes = [pl.Int64]

    def __init__(self, size):
        self._size = size

    def estimated_size(self, unit="b"):
        return self._size


class TestUnitsAndSingleColumn:
    def test_a_single_column_frame_names_it_as_both_first_and_last(self):
        assert "Columns: 1 entries, a to a" in frames.polars_df_info(pl.DataFrame({"a": [1]})).split("\n")

    def test_the_unit_is_bytes_below_one_kilobyte(self):
        assert frames.polars_df_info(_Sized(1023.9)).endswith("memory usage: 1023.9+ bytes")

    def test_one_kilobyte_exactly_is_reported_in_kilobytes(self):
        assert frames.polars_df_info(_Sized(1024)).endswith("memory usage: 1.0+ KB")

    def test_a_little_over_one_kilobyte_is_reported_in_kilobytes(self):
        assert frames.polars_df_info(_Sized(1024.05)).endswith("memory usage: 1.0+ KB")

    def test_a_size_in_kilobytes_is_divided_by_1024(self):
        assert frames.polars_df_info(_Sized(2048)).endswith("memory usage: 2.0+ KB")

    def test_a_size_in_megabytes_is_reported_in_megabytes(self):
        assert frames.polars_df_info(_Sized(5 * 1024 * 1024)).endswith("memory usage: 5.0+ MB")

    def test_a_size_in_gigabytes_is_reported_in_gigabytes(self):
        assert frames.polars_df_info(_Sized(3 * 1024**3)).endswith("memory usage: 3.0+ GB")


class TestLogTextWidth:
    def test_a_name_longer_than_the_default_width_is_shortened_in_the_warning(self, caplog):
        import logging

        df = pl.DataFrame({"c" * 301: [1, 1], "k": [1, 2]})
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.drop_constant_columns(df, verbose=True)
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert "[...]" in messages[0]


class TestDivisorExactness:
    def test_gigabytes_are_not_divided_again_once_reached(self):
        assert frames.polars_df_info(_Sized(5000 * 1024**3)).endswith("memory usage: 5000.0+ GB")

    def test_the_megabyte_value_is_rounded_from_an_exact_1024_divisor(self):
        assert frames.polars_df_info(_Sized(5_295_420)).endswith("memory usage: 5.1+ MB")
