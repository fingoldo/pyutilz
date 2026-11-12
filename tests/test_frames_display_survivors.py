"""Exact value-count display output for data.pandaslib.frames.showcase_df_columns (mutation survivors)."""

from __future__ import annotations

import pandas as pd
import polars as pl

from pyutilz.data.pandaslib import frames


def _showcase(df, **kwargs):
    return frames.showcase_df_columns(df, use_markdown=False, **kwargs)


class TestPolarsDisplay:
    def test_a_zero_cap_prints_only_the_header_and_a_blank_line(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "b", "a"]}), max_vars=0)
        assert capsys.readouterr().out == "X String\n\n"

    def test_a_cap_of_one_prints_only_the_most_common_value(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "b", "a"]}), max_vars=1)
        assert capsys.readouterr().out == "X String\nx\na    2\nName: count, dtype: int64\n"

    def test_no_cap_prints_every_value(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "b", "a"]}), max_vars=None)
        assert capsys.readouterr().out == "X String\nx\na    2\nb    1\nName: count, dtype: int64\n"

    def test_a_single_value_column_is_printed(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "a", "a"]}))
        assert capsys.readouterr().out == "X String\nx\na    3\nName: count, dtype: int64\n"

    def test_an_empty_column_prints_an_empty_count_series(self, capsys):
        _showcase(pl.DataFrame({"e": pl.Series([], dtype=pl.Utf8)}))
        assert capsys.readouterr().out == "E String\nSeries([], Name: count, dtype: int64)\n"


class TestPandasDisplay:
    def test_a_zero_cap_prints_only_the_header_and_a_blank_line(self, capsys):
        _showcase(pd.DataFrame({"x": pd.Series(["a", "b", "a"], dtype=object)}), max_vars=0)
        assert capsys.readouterr().out == "X object\n\n"

    def test_a_cap_of_one_prints_only_the_most_common_value(self, capsys):
        _showcase(pd.DataFrame({"x": pd.Series(["a", "b", "a"], dtype=object)}), max_vars=1)
        assert capsys.readouterr().out == "X object\nx\na    2\nName: count, dtype: int64\n"

    def test_an_empty_column_prints_an_empty_count_series(self, capsys):
        _showcase(pd.DataFrame({"e": pd.Series([], dtype=object)}))
        assert capsys.readouterr().out == "E object\nSeries([], Name: count, dtype: int64)\n"


class TestRareClassification:
    def test_an_empty_pandas_column_is_not_reported_as_rare(self):
        df = pd.DataFrame({"e": pd.Series([], dtype=object)})
        rare, _ = frames.showcase_df_columns(df, use_markdown=False, use_print=False)
        assert "e" not in rare


class TestDisplayWithoutRareCategories:
    def test_a_single_value_column_is_printed_when_no_rare_category_threshold_applies(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "a", "a"]}), max_vars=None, max_cat_uniq_qty=0)
        assert capsys.readouterr().out == "X String\nx\na    3\nName: count, dtype: int64\n"

    def test_a_single_value_column_with_a_cap_of_one_is_printed_without_rare_threshold(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "a", "a"]}), max_vars=1, max_cat_uniq_qty=0)
        assert capsys.readouterr().out == "X String\nx\na    3\nName: count, dtype: int64\n"

    def test_two_values_are_all_printed_without_rare_threshold(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "b", "a"]}), max_vars=None, max_cat_uniq_qty=0)
        assert capsys.readouterr().out == "X String\nx\na    2\nb    1\nName: count, dtype: int64\n"
