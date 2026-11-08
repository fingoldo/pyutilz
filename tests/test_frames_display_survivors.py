"""Value-count display and rare-column branches of data.pandaslib.frames.showcase_df_columns (mutation survivors)."""

from __future__ import annotations

import polars as pl
import pandas as pd

from pyutilz.data.pandaslib import frames


def _showcase(df, **kwargs):
    return frames.showcase_df_columns(df, use_markdown=False, **kwargs)


class TestPolarsValueCountPrinting:
    def test_a_zero_row_cap_prints_a_blank_line(self, capsys):
        frames._print_polars_value_counts("v", 3, ["a", "b", "c"], [2, 1, 1], 0)
        assert capsys.readouterr().out == "\n"

    def test_a_cap_of_one_shows_only_the_first_value(self, capsys):
        frames._print_polars_value_counts("v", 3, ["a", "b", "c"], [2, 1, 1], 1)
        out = capsys.readouterr().out
        assert "a" in out and "b" not in out

    def test_no_cap_shows_every_value(self, capsys):
        frames._print_polars_value_counts("v", 3, ["a", "b", "c"], [2, 1, 1], None)
        out = capsys.readouterr().out
        assert "b" in out and "c" in out

    def test_an_empty_column_prints_an_empty_count_series(self, capsys):
        frames._print_polars_value_counts("v", 0, [], [], None)
        assert "Series([]" in capsys.readouterr().out


class TestShowcaseDisplayGates:
    def test_a_zero_row_cap_prints_no_value_rows_for_a_populated_column(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "b", "a"]}), max_vars=0)
        out = capsys.readouterr().out
        assert "count" not in out and "a" not in out

    def test_a_single_value_column_is_displayed(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "a", "a"]}), max_vars=None)
        assert capsys.readouterr().out.strip() != ""

    def test_a_cap_of_one_is_a_real_cap_and_still_displays(self, capsys):
        _showcase(pl.DataFrame({"x": ["a", "b", "a"]}), max_vars=1)
        assert capsys.readouterr().out.strip() != ""

    def test_an_empty_polars_column_shows_an_empty_count_series(self, capsys):
        _showcase(pl.DataFrame({"e": pl.Series([], dtype=pl.Utf8)}))
        assert "Series([]" in capsys.readouterr().out

    def test_an_empty_pandas_column_is_not_reported_as_rare(self):
        df = pd.DataFrame({"e": pd.Series([], dtype=object)})
        result = frames.showcase_df_columns(df, use_markdown=False, use_print=False)
        assert "e" not in list(result[0])
