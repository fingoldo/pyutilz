"""Behaviour of ``pyutilz.data.pandaslib.frames``: standard responses are nulled only when they repeat more than the
threshold (optionally counting distinct persons instead of rows), the verbose mode prints the values it nulls, and the
column prefixer returns a (frame, rename mapping) pair whether it renames in place or on a copy, honouring special
prefixes and exclusions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pyutilz.data.pandaslib import frames


class TestNullifyStandardValues:
    def test_a_value_repeating_more_than_the_threshold_is_nulled(self):
        df = pd.DataFrame({"v": ["std"] * 3 + ["x", "y"]})
        frames.nullify_standard_values(df, "v", min_records=2)
        assert all(pd.isna(x) for x in df["v"].tolist()[:3])
        assert df["v"].tolist()[3:] == ["x", "y"]

    def test_a_value_at_exactly_the_threshold_is_kept(self):
        df = pd.DataFrame({"v": ["std"] * 2 + ["x"]})
        frames.nullify_standard_values(df, "v", min_records=2)
        assert df["v"].tolist() == ["std", "std", "x"]

    def test_a_custom_placeholder_replaces_the_nulled_values(self):
        df = pd.DataFrame({"v": ["std"] * 3})
        frames.nullify_standard_values(df, "v", min_records=1, placeholder="NONE")
        assert df["v"].tolist() == ["NONE", "NONE", "NONE"]

    def test_with_a_persons_field_only_values_used_by_enough_people_are_nulled(self):
        df = pd.DataFrame({"v": ["std"] * 3 + ["x"], "p": [1, 2, 3, 4]})
        frames.nullify_standard_values(df, "v", min_records=2, persons_field="p", min_persons=2)
        assert all(pd.isna(x) for x in df["v"].tolist()[:3])
        assert df["v"].tolist()[3] == "x"

    def test_verbose_mode_reports_each_value_it_treats_as_standard(self, capsys):
        df = pd.DataFrame({"v": ["std"] * 3, "p": [1, 2, 3]})
        frames.nullify_standard_values(df, "v", min_records=1, persons_field="p", min_persons=2, verbose=True)
        out = capsys.readouterr().out
        assert "value std is not custom, as used by 3 persons" in out


class TestPrefixizeColumns:
    def test_every_column_is_renamed_in_place_by_default(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        out, columns = frames.prefixize_columns(df, "src")
        assert out is df
        assert list(df.columns) == ["src_a", "src_b"]
        assert columns == {"a": "src_a", "b": "src_b"}

    def test_inplace_false_leaves_the_original_untouched(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        out, columns = frames.prefixize_columns(df, "src", inplace=False)
        assert list(df.columns) == ["a", "b"]
        assert list(out.columns) == ["src_a", "src_b"]
        assert columns == {"a": "src_a", "b": "src_b"}

    def test_a_special_prefix_overrides_the_default_for_that_column(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        _, columns = frames.prefixize_columns(df, "src", special_prefixes={"b": "special"})
        assert columns == {"a": "src_a", "b": "special_b"}

    def test_excluded_columns_keep_their_names(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        _, columns = frames.prefixize_columns(df, "src", exclusions=["a"])
        assert columns == {"a": "a", "b": "src_b"}

    def test_the_separator_is_configurable(self):
        df = pd.DataFrame({"a": [1]})
        _, columns = frames.prefixize_columns(df, "src", sep="-")
        assert columns == {"a": "src-a"}


class TestNullifyBoundaries:
    def test_a_value_repeated_past_the_default_threshold_is_nulled(self):
        df = pd.DataFrame({"v": ["std"] * 301})
        frames.nullify_standard_values(df, "v")
        assert df["v"].isna().all()

    def test_a_value_at_exactly_the_default_threshold_is_kept(self):
        df = pd.DataFrame({"v": ["std"] * 300})
        frames.nullify_standard_values(df, "v")
        assert (df["v"] == "std").all()

    def test_a_value_with_only_missing_persons_is_not_nulled_by_default(self):
        df = pd.DataFrame({"v": ["std"] * 3, "p": [None, None, None]})
        frames.nullify_standard_values(df, "v", min_records=1, persons_field="p")
        assert (df["v"] == "std").all()

    def test_a_value_used_by_exactly_min_persons_is_kept(self):
        df = pd.DataFrame({"v": ["std"] * 3, "p": [1, 1, 2]})
        frames.nullify_standard_values(df, "v", min_records=1, persons_field="p", min_persons=2)
        assert (df["v"] == "std").all()


class TestNullifyVerbose:
    def test_verbose_is_off_by_default(self, capsys):
        df = pd.DataFrame({"v": ["std"] * 3, "p": [1, 2, 3]})
        frames.nullify_standard_values(df, "v", min_records=1, persons_field="p")
        assert capsys.readouterr().out == ""

    def test_verbose_prints_nothing_when_no_value_qualifies(self, capsys):
        df = pd.DataFrame({"v": ["a", "b"], "p": [1, 2]})
        frames.nullify_standard_values(df, "v", min_records=10, persons_field="p", min_persons=0, verbose=True)
        assert capsys.readouterr().out == ""

    def test_verbose_names_the_field_before_its_values(self, capsys):
        df = pd.DataFrame({"v": ["std"] * 3, "p": [1, 2, 3]})
        frames.nullify_standard_values(df, "v", min_records=1, persons_field="p", min_persons=2, verbose=True)
        assert capsys.readouterr().out.startswith("Field v")


class TestRareFromLists:
    def test_a_value_at_exactly_the_threshold_is_rare(self):
        rare, _ = frames._rare_and_uninformative_from_lists(["a", "b"], [5, 5], 0.5)
        assert rare == ["a", "b"]

    def test_the_fraction_uses_the_column_total_not_a_frame_height(self):
        rare, uninformative = frames._rare_and_uninformative_from_lists(["a", "b"], [90, 1], 0.05)
        assert rare == ["b"]
        assert uninformative == pytest.approx(1 / 91)

    def test_two_common_values_leave_no_uninformative_fraction(self):
        rare, uninformative = frames._rare_and_uninformative_from_lists(["a", "b", "c"], [50, 49, 1], 0.02)
        assert rare == ["c"]
        assert uninformative is None

    def test_a_single_dominant_value_reports_the_share_it_does_not_hold(self):
        rare, uninformative = frames._rare_and_uninformative_from_lists(["a", "b"], [98, 2], 0.05)
        assert rare == ["b"]
        assert uninformative == pytest.approx(0.02)

    def test_no_rare_value_reports_no_fraction(self):
        assert frames._rare_and_uninformative_from_lists(["a", "b"], [50, 50], 0.1) == ([], None)


class TestRareFromValueCounts:
    def test_a_value_at_exactly_the_threshold_is_rare(self):
        stats = pd.Series([5, 5], index=["a", "b"])
        rare, _ = frames._rare_and_uninformative_from_value_counts(stats, 0.5)
        assert rare == ["a", "b"]

    def test_a_single_dominant_value_reports_the_share_it_does_not_hold(self):
        stats = pd.Series([98, 2], index=["a", "b"])
        rare, uninformative = frames._rare_and_uninformative_from_value_counts(stats, 0.05)
        assert rare == ["b"]
        assert uninformative == pytest.approx(0.02)

    def test_two_common_values_leave_no_uninformative_fraction(self):
        stats = pd.Series([50, 49, 1], index=["a", "b", "c"])
        rare, uninformative = frames._rare_and_uninformative_from_value_counts(stats, 0.02)
        assert rare == ["c"]
        assert uninformative is None

    def test_no_rare_value_reports_no_fraction(self):
        stats = pd.Series([50, 50], index=["a", "b"])
        assert frames._rare_and_uninformative_from_value_counts(stats, 0.1) == ([], None)


class TestRareBoundaries:
    def test_values_exactly_at_the_threshold_are_rare_and_none_is_non_rare(self):
        rare, uninformative = frames._rare_and_uninformative_from_lists(["a", "b"], [5, 5], 0.5)
        assert rare == ["a", "b"]
        assert uninformative == pytest.approx(1.0)

    def test_a_zero_total_column_reports_no_uninformative_fraction(self):
        assert frames._rare_and_uninformative_from_lists(["a"], [0], 0.1) == (["a"], 0.0)

    def test_a_single_rare_value_with_a_total_of_one_is_fully_uninformative(self):
        assert frames._rare_and_uninformative_from_lists(["a"], [1], 1.0) == (["a"], 1.0)

    def test_value_counts_exactly_at_the_threshold_are_rare_and_none_is_non_rare(self):
        rare, uninformative = frames._rare_and_uninformative_from_value_counts(pd.Series([5, 5], index=["a", "b"]), 0.5)
        assert rare == ["a", "b"]
        assert uninformative == pytest.approx(1.0)

    def test_a_zero_total_value_counts_reports_no_uninformative_fraction(self):
        assert frames._rare_and_uninformative_from_value_counts(pd.Series([0], index=["a"]), 0.1) == (["a"], 0.0)

    def test_a_single_rare_value_with_a_total_of_one_is_fully_uninformative_in_value_counts(self):
        assert frames._rare_and_uninformative_from_value_counts(pd.Series([1], index=["a"]), 1.0) == (["a"], 1.0)


class TestPolarsValueCountsPrinting:
    def test_a_zero_cap_prints_only_a_blank_line(self, capsys):
        frames._print_polars_value_counts("col", 3, ["a"], [3], 0)
        assert capsys.readouterr().out == "\n"

    def test_an_empty_column_prints_an_empty_named_series(self, capsys):
        frames._print_polars_value_counts("col", 0, [], [], None)
        out = capsys.readouterr().out
        assert out.startswith("Series([]")
        assert "Name: count" in out

    def test_no_cap_prints_every_value(self, capsys):
        frames._print_polars_value_counts("col", 6, ["alpha", "beta", "gamma"], [3, 2, 1], None)
        out = capsys.readouterr().out
        assert "alpha" in out and "beta" in out and "gamma" in out

    def test_a_positive_cap_truncates_to_the_first_rows(self, capsys):
        frames._print_polars_value_counts("col", 6, ["alpha", "beta", "gamma"], [3, 2, 1], 2)
        out = capsys.readouterr().out
        assert "alpha" in out and "beta" in out
        assert "gamma" not in out

    def test_a_non_positive_cap_other_than_zero_prints_every_value(self, capsys):
        frames._print_polars_value_counts("col", 6, ["alpha", "beta", "gamma"], [3, 2, 1], -1)
        assert "gamma" in capsys.readouterr().out


class TestPandasValueCountsPrinting:
    def test_nothing_is_printed_when_not_asked_to(self, capsys):
        frames._print_value_counts(pd.Series([3, 1], index=["alpha", "beta"]), False, None)
        assert capsys.readouterr().out == ""

    def test_no_cap_prints_every_value(self, capsys):
        frames._print_value_counts(pd.Series([3, 2, 1], index=["alpha", "beta", "gamma"]), True, None)
        assert "gamma" in capsys.readouterr().out

    def test_a_positive_cap_truncates_to_the_first_rows(self, capsys):
        frames._print_value_counts(pd.Series([3, 2, 1], index=["alpha", "beta", "gamma"]), True, 2)
        out = capsys.readouterr().out
        assert "alpha" in out and "beta" in out
        assert "gamma" not in out

    def test_a_zero_cap_prints_only_a_blank_line(self, capsys):
        frames._print_value_counts(pd.Series([3], index=["alpha"]), True, 0)
        assert capsys.readouterr().out == "\n"


class TestRecordRareAndUninformative:
    def test_rare_values_are_recorded_without_a_fraction_when_two_values_survive(self):
        rare, uninformative = {}, {}
        frames._record_rare_and_uninformative("x", ["a", "b", "c"], [50, 49, 1], 0.02, rare, uninformative)
        assert rare == {"x": ["c"]}
        assert uninformative == {}

    def test_rare_values_and_the_fraction_are_recorded_when_one_value_survives(self):
        rare, uninformative = {}, {}
        frames._record_rare_and_uninformative("x", ["a", "b"], [98, 2], 0.05, rare, uninformative)
        assert rare == {"x": ["b"]}
        assert uninformative == {"x": pytest.approx(0.02)}

    def test_a_column_with_no_rare_value_records_nothing(self):
        rare, uninformative = {}, {}
        frames._record_rare_and_uninformative("x", ["a", "b"], [50, 50], 0.1, rare, uninformative)
        assert rare == {}
        assert uninformative == {}


class TestPrintBoundaries:
    def test_a_single_row_polars_column_is_printed_and_not_treated_as_empty(self, capsys):
        frames._print_polars_value_counts("col", 1, ["alpha"], [1], None)
        assert "alpha" in capsys.readouterr().out

    def test_a_non_empty_polars_series_is_named_count(self, capsys):
        frames._print_polars_value_counts("col", 1, ["alpha"], [1], None)
        assert "Name: count" in capsys.readouterr().out

    def test_a_cap_of_one_prints_only_the_first_polars_row(self, capsys):
        frames._print_polars_value_counts("col", 2, ["alpha", "beta"], [2, 1], 1)
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" not in out

    def test_a_cap_of_one_prints_only_the_first_pandas_row(self, capsys):
        frames._print_value_counts(pd.Series([2, 1], index=["alpha", "beta"]), True, 1)
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" not in out


class TestShowcaseDispatch:
    def test_excluded_columns_are_not_analysed(self):
        df = pd.DataFrame({"a": ["p"] * 98 + ["q"] * 2, "b": ["p"] * 98 + ["q"] * 2})
        rare, _ = frames.showcase_df_columns(df, excluded_cols=["a"], use_print=False, use_markdown=False, max_unique_percent=0.05)
        assert set(rare) == {"b"}

    def test_every_column_excluded_returns_empty_pair(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert frames.showcase_df_columns(df, excluded_cols=["a"], use_print=False) == ({}, {})

    def test_no_cols_argument_analyses_every_column(self):
        df = pd.DataFrame({"a": ["p"] * 98 + ["q"] * 2})
        rare, _ = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05)
        assert rare == {"a": ["q"]}

    def test_use_print_false_writes_nothing_to_stdout_for_pandas(self, capsys):
        df = pd.DataFrame({"a": ["p"] * 98 + ["q"] * 2})
        frames.showcase_df_columns(df, use_print=False, use_markdown=False)
        assert capsys.readouterr().out == ""

    def test_use_print_false_writes_nothing_to_stdout_for_polars(self, capsys):
        df = pl.DataFrame({"a": ["p"] * 98 + ["q"] * 2})
        frames.showcase_df_columns(df, use_print=False, use_markdown=False)
        assert capsys.readouterr().out == ""

    def test_a_pandas_header_names_the_column_in_upper_case_with_its_dtype(self, capsys):
        df = pd.DataFrame({"a": [1, 2, 3]})
        frames.showcase_df_columns(df, use_markdown=False)
        assert "A int64" in capsys.readouterr().out

    def test_a_polars_header_names_the_column_in_upper_case_with_its_dtype(self, capsys):
        df = pl.DataFrame({"a": [1, 2, 3]})
        frames.showcase_df_columns(df, use_markdown=False)
        assert "A Int64" in capsys.readouterr().out

    def test_a_negative_max_vars_is_refused_for_pandas(self):
        with pytest.raises(AssertionError, match="max_vars must be >= 0"):
            frames.showcase_df_columns(pd.DataFrame({"a": [1, 2]}), max_vars=-1, use_print=False, use_markdown=False)

    def test_a_non_dataframe_is_refused(self):
        with pytest.raises(AssertionError, match="expected pandas DataFrame, got dict"):
            frames.showcase_df_columns({"a": [1, 2]}, cols=["a"], use_print=False)

    def test_pandas_nulls_are_a_category_unless_dropna(self):
        df = pd.DataFrame({"a": ["p"] * 98 + [None] * 2})
        kept, _ = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05, dropna=False)
        dropped, _ = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05, dropna=True)
        assert kept == {"a": [None]}
        assert dropped == {}

    def test_polars_nulls_are_a_category_unless_dropna(self):
        df = pl.DataFrame({"a": ["p"] * 98 + [None] * 2})
        kept, _ = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05, dropna=False)
        dropped, _ = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05, dropna=True)
        assert kept == {"a": [None]}
        assert dropped == {}

    def test_a_column_at_exactly_the_distinct_cap_is_analysed_and_one_over_is_not(self):
        at_cap = pd.DataFrame({"a": ["p"] * 98 + ["q"] * 2})
        over_cap = pd.DataFrame({"a": ["p"] * 97 + ["q"] * 2 + ["r"]})
        analysed, _ = frames.showcase_df_columns(at_cap, use_print=False, use_markdown=False, max_unique_percent=0.05, max_cat_uniq_qty=2)
        skipped, _ = frames.showcase_df_columns(over_cap, use_print=False, use_markdown=False, max_unique_percent=0.05, max_cat_uniq_qty=2)
        assert analysed == {"a": ["q"]}
        assert skipped == {}

    def test_a_polars_column_at_exactly_the_distinct_cap_is_analysed_and_one_over_is_not(self):
        at_cap = pl.DataFrame({"a": ["p"] * 98 + ["q"] * 2})
        over_cap = pl.DataFrame({"a": ["p"] * 97 + ["q"] * 2 + ["r"]})
        analysed, _ = frames.showcase_df_columns(at_cap, use_print=False, use_markdown=False, max_unique_percent=0.05, max_cat_uniq_qty=2)
        skipped, _ = frames.showcase_df_columns(over_cap, use_print=False, use_markdown=False, max_unique_percent=0.05, max_cat_uniq_qty=2)
        assert analysed == {"a": ["q"]}
        assert skipped == {}

    def test_a_dominant_value_reports_its_uninformative_fraction_for_pandas(self):
        df = pd.DataFrame({"a": ["p"] * 98 + ["q"] * 2})
        _, uninformative = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05)
        assert uninformative == {"a": pytest.approx(0.02)}

    def test_a_dominant_value_reports_its_uninformative_fraction_for_polars(self):
        df = pl.DataFrame({"a": ["p"] * 98 + ["q"] * 2})
        _, uninformative = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05)
        assert uninformative == {"a": pytest.approx(0.02)}


@pytest.fixture
def ipython_recorder(monkeypatch):
    shown = []
    monkeypatch.setattr(frames._facade, "HAS_IPYTHON", True, raising=False)
    monkeypatch.setattr(frames._facade, "Markdown", lambda text: text, raising=False)
    monkeypatch.setattr(frames._facade, "display", lambda text: shown.append(text), raising=False)
    return shown


class TestShowcaseMarkdownAndValues:
    def test_a_pandas_header_is_rendered_as_bold_markdown_when_ipython_is_present(self, ipython_recorder):
        frames.showcase_df_columns(pd.DataFrame({"a": [1, 2]}), use_print=False, use_markdown=True)
        assert ipython_recorder == ["**a** int64"]

    def test_a_polars_header_is_rendered_as_bold_markdown_when_ipython_is_present(self, ipython_recorder):
        frames.showcase_df_columns(pl.DataFrame({"a": [1, 2]}), use_print=False, use_markdown=True)
        assert ipython_recorder == ["**a** Int64"]

    def test_pandas_value_counts_are_printed_with_their_values(self, capsys):
        frames.showcase_df_columns(pd.DataFrame({"a": ["alpha", "beta"]}), use_markdown=False)
        out = capsys.readouterr().out
        assert "alpha" in out and "beta" in out

    def test_polars_value_counts_are_printed_with_their_values(self, capsys):
        frames.showcase_df_columns(pl.DataFrame({"a": ["alpha", "beta"]}), use_markdown=False)
        out = capsys.readouterr().out
        assert "alpha" in out and "beta" in out

    def test_a_zero_cap_on_pandas_hides_the_values_and_is_not_refused(self, capsys):
        frames.showcase_df_columns(pd.DataFrame({"a": ["alpha"]}), max_vars=0, use_markdown=False)
        assert "alpha" not in capsys.readouterr().out


class TestPandasSingleValueBoundary:
    def test_a_single_value_column_that_is_wholly_rare_is_fully_uninformative(self):
        rare, uninformative = frames.showcase_df_columns(pd.DataFrame({"a": ["a", "a", "a"]}), use_print=False, use_markdown=False, max_unique_percent=1.0)
        assert rare == {"a": ["a"]}
        assert uninformative == {"a": pytest.approx(1.0)}


class TestShowcaseDefaults:
    def test_nulls_are_counted_by_default(self):
        df = pd.DataFrame({"a": ["p"] * 98 + [None] * 2})
        rare, _ = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05)
        assert rare == {"a": [None]}

    def test_polars_nulls_are_counted_by_default(self):
        df = pl.DataFrame({"a": ["p"] * 98 + [None] * 2})
        rare, _ = frames.showcase_df_columns(df, use_print=False, use_markdown=False, max_unique_percent=0.05)
        assert rare == {"a": [None]}

    def test_markdown_is_on_by_default(self, ipython_recorder):
        frames.showcase_df_columns(pd.DataFrame({"a": [1, 2]}), use_print=False)
        assert ipython_recorder == ["**a** int64"]

    def test_printing_is_on_by_default(self, capsys):
        frames.showcase_df_columns(pd.DataFrame({"a": [1, 2]}), use_markdown=False)
        assert capsys.readouterr().out != ""

    def test_fifty_distinct_values_are_analysed_by_default_and_fifty_one_are_not(self):
        base = {f"v{i}": 20 for i in range(50)}
        base["rare"] = 1
        values = [v for v, n in base.items() for _ in range(n)]
        analysed = frames.showcase_df_columns(pd.DataFrame({"a": values}), use_print=False, use_markdown=False, max_cat_uniq_qty=51)[0]
        skipped = frames.showcase_df_columns(pd.DataFrame({"a": values}), use_print=False, use_markdown=False)[0]
        assert analysed == {"a": ["rare"]}
        assert skipped == {}

    def test_the_default_unique_percent_is_a_tenth_of_a_percent(self):
        base = {f"v{i}": 21 for i in range(49)}
        base["rare"] = 1
        values = [v for v, n in base.items() for _ in range(n)]
        rare, _ = frames.showcase_df_columns(pd.DataFrame({"a": values}), use_print=False, use_markdown=False)
        assert rare == {"a": ["rare"]}


class TestPolarsPrintedRows:
    def test_polars_values_are_listed_most_frequent_first(self, capsys):
        frames.showcase_df_columns(pl.DataFrame({"a": ["beta"] + ["alpha"] * 3}), use_markdown=False)
        out = capsys.readouterr().out
        assert out.index("alpha") < out.index("beta")

    def test_the_polars_series_is_indexed_by_the_column_name(self, capsys):
        frames.showcase_df_columns(pl.DataFrame({"a": ["alpha", "alpha", "beta"]}), use_markdown=False)
        lines = capsys.readouterr().out.splitlines()
        assert lines[1] == "a"

    def test_each_polars_value_sits_beside_its_count(self, capsys):
        import re

        frames.showcase_df_columns(pl.DataFrame({"a": ["alpha"] * 3 + ["beta"]}), use_markdown=False)
        out = capsys.readouterr().out
        assert re.search(r"alpha\s+3", out)
        assert re.search(r"beta\s+1", out)

    def test_polars_values_are_printed_even_when_the_rare_analysis_is_skipped(self, capsys):
        frames.showcase_df_columns(pl.DataFrame({"a": ["alpha", "beta"]}), use_markdown=False, max_cat_uniq_qty=1)
        out = capsys.readouterr().out
        assert "alpha" in out and "beta" in out


class TestPolarsSingleValueColumns:
    def test_a_single_distinct_polars_value_is_printed(self, capsys):
        frames.showcase_df_columns(pl.DataFrame({"a": ["alpha"] * 3}), use_markdown=False)
        assert "alpha" in capsys.readouterr().out

    def test_polars_values_are_printed_under_a_positive_cap(self, capsys):
        frames.showcase_df_columns(pl.DataFrame({"a": ["alpha", "beta"]}), use_markdown=False, max_vars=5)
        out = capsys.readouterr().out
        assert "alpha" in out and "beta" in out

    def test_a_single_wholly_rare_polars_value_is_fully_uninformative(self):
        rare, uninformative = frames.showcase_df_columns(pl.DataFrame({"a": ["x"] * 3}), use_print=False, use_markdown=False, max_unique_percent=1.0)
        assert rare == {"a": ["x"]}
        assert uninformative == {"a": pytest.approx(1.0)}


class TestPolarsDisplayWithoutRareAnalysis:
    def test_polars_values_are_printed_under_a_positive_cap_when_the_rare_analysis_is_skipped(self, capsys):
        frames.showcase_df_columns(pl.DataFrame({"a": ["alpha", "beta"]}), use_markdown=False, max_vars=5, max_cat_uniq_qty=1)
        out = capsys.readouterr().out
        assert "alpha" in out and "beta" in out


class TestFeatureNamer:
    def test_names_get_consecutive_ids_and_repeats_keep_theirs(self):
        fname = frames.FeatureNamer()
        assert (fname("abc"), fname("abcd"), fname("abc")) == (0, 1, 0)
        assert fname.rev(0) == "abc"

    def test_an_unknown_id_has_no_name(self):
        assert frames.FeatureNamer(["x"]).rev(99) is None

    def test_initial_values_are_numbered_in_order(self):
        fname = frames.FeatureNamer(["x", "y"])
        assert (fname("x"), fname("y")) == (0, 1)
        assert fname.fnames_index == 2


class TestShareDataframe:
    def test_values_and_dtypes_survive_the_round_trip(self):
        df = pd.DataFrame({"f": [1.5, 2.5], "i": [1, 2], "b": [True, False]})
        shared = frames.share_dataframe(df)
        assert shared.equals(df)
        assert list(shared.dtypes) == list(df.dtypes)

    def test_a_large_int64_is_not_corrupted_by_a_float_route(self):
        big = 2**60 + 1
        shared = frames.share_dataframe(pd.DataFrame({"i": [big, 3]}))
        assert int(shared["i"].iloc[0]) == big

    def test_a_large_uint64_is_not_corrupted_by_a_float_route(self):
        big = 2**63 + 1
        shared = frames.share_dataframe(pd.DataFrame({"u": pd.Series([big, 3], dtype="uint64")}))
        assert int(shared["u"].iloc[0]) == big

    def test_the_original_index_is_restored(self):
        df = pd.DataFrame({"f": [1.5, 2.5]}, index=["x", "y"])
        shared = frames.share_dataframe(df)
        assert list(shared.index) == ["x", "y"]
        assert shared.loc["y", "f"] == 2.5

    def test_the_original_column_order_is_restored(self):
        df = pd.DataFrame({"b": [1, 2], "a": [1.5, 2.5], "c": [True, False]})
        assert list(frames.share_dataframe(df).columns) == ["b", "a", "c"]

    def test_mixed_integer_widths_keep_their_own_dtypes(self):
        df = pd.DataFrame({"i32": pd.Series([1, 2], dtype="int32"), "i64": pd.Series([3, 4], dtype="int64")})
        shared = frames.share_dataframe(df)
        assert shared["i32"].dtype == "int32"
        assert shared["i64"].dtype == "int64"
        assert shared.equals(df)


class TestNonStaleColumns:
    def test_a_frame_of_zero_or_one_rows_keeps_every_column(self):
        assert frames.get_non_stale_columns(pd.DataFrame({"a": [], "b": []})) == ["a", "b"]
        assert frames.get_non_stale_columns(pd.DataFrame({"a": [1], "b": [2]})) == ["a", "b"]

    def test_constant_and_all_nan_columns_are_stale_and_the_rest_are_kept_in_order(self, caplog):
        import logging

        df = pd.DataFrame({"a": [1, 2, 3], "c": [5, 5, 5], "b": [np.nan, np.nan, np.nan], "d": [4, 5, 6]})
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            kept = frames.get_non_stale_columns(df)
        assert kept == ["a", "d"]
        assert any("Found 2 stale columns: c,b" in rec.getMessage() for rec in caplog.records)

    def test_the_caller_frame_is_not_mutated(self):
        df = pd.DataFrame({"a": [1, 2], "c": [5, 5]})
        frames.get_non_stale_columns(df)
        assert list(df.columns) == ["a", "c"]

    def test_the_deprecated_alias_warns_and_returns_the_same_list(self):
        df = pd.DataFrame({"a": [1, 2], "c": [5, 5]})
        with pytest.warns(DeprecationWarning, match="remove_stale_columns is deprecated"):
            kept = frames.remove_stale_columns(df)
        assert kept == frames.get_non_stale_columns(df) == ["a"]


class TestSuspiciouslyConstantColumns:
    def test_a_column_with_one_value_and_a_null_is_not_constant(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "c": [5, 5]})
        assert frames.get_suspiciously_constant_columns(df) == ["c"]

    def test_all_nan_is_constant(self):
        df = pd.DataFrame({"n": [np.nan, np.nan], "v": [1, 2]})
        assert frames.get_suspiciously_constant_columns(df) == ["n"]

    def test_unhashable_values_are_skipped_with_a_warning_naming_the_column(self, caplog):
        import logging

        df = pd.DataFrame({"lst": [[1], [2]], "c": [5, 5]})
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            constant = frames.get_suspiciously_constant_columns(df)
        assert constant == ["c"]
        assert any("could not check 1 column(s) (unhashable values)" in rec.getMessage() and "['lst']" in rec.getMessage() for rec in caplog.records)


class TestRemoveConstantColumns:
    def test_constant_columns_are_dropped_in_place(self):
        df = pd.DataFrame({"a": [1, 2], "c": [5, 5]})
        frames.remove_constant_columns(df)
        assert list(df.columns) == ["a"]

    def test_verbose_names_the_few_constant_columns_it_removes(self, caplog):
        import logging

        df = pd.DataFrame({"a": [1, 2], "c": [5, 5]})
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.remove_constant_columns(df, verbose=True)
        assert any(rec.getMessage() == "Removing constant columns ['c']" for rec in caplog.records)

    def test_verbose_counts_when_there_are_many_constant_columns(self, caplog):
        import logging

        data = {f"c{i}": [1, 1] for i in range(21)}
        data["a"] = [1, 2]
        df = pd.DataFrame(data)
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.remove_constant_columns(df, verbose=True)
        assert any(rec.getMessage() == "Removing 21 constant columns" for rec in caplog.records)
        assert list(df.columns) == ["a"]

    def test_a_column_constant_only_in_the_head_sample_is_kept(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3, 4], "b": [7, 7, 7, 7, 7]})
        frames.remove_constant_columns(df, prewarm_size=2)
        assert list(df.columns) == ["a"]


class TestFeatureNamerInitialOrder:
    def test_initial_values_are_registered_before_any_later_name(self):
        fname = frames.FeatureNamer(["x", "y"])
        assert fname("y") == 1
        assert fname.rev(0) == "x"


class TestStaleAndConstantEdges:
    def test_a_null_does_not_make_a_stale_column_look_constant(self):
        df = pd.DataFrame({"a": [5.0, np.nan, 5.0], "b": [1, 2, 3]})
        assert "a" in frames.get_non_stale_columns(df)

    def test_no_warning_is_logged_when_nothing_is_stale(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.get_non_stale_columns(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        assert not any("stale columns" in rec.getMessage() for rec in caplog.records)

    def test_the_deprecation_warning_points_at_the_caller(self):
        import warnings

        df = pd.DataFrame({"a": [1, 2]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frames.remove_stale_columns(df)
        assert any(w.filename == __file__ for w in caught if issubclass(w.category, DeprecationWarning))

    def test_a_null_column_in_the_fallback_loop_is_not_constant(self):
        df = pd.DataFrame({"lst": [[1], [2]], "a": [1.0, np.nan]})
        assert frames.get_suspiciously_constant_columns(df) == []

    def test_two_distinct_values_in_the_fallback_loop_are_not_constant(self):
        df = pd.DataFrame({"lst": [[1], [2], [3]], "a": [1, 1, 2]})
        assert frames.get_suspiciously_constant_columns(df) == []

    def test_default_remove_constant_columns_is_quiet(self, caplog):
        import logging

        df = pd.DataFrame({"a": [1, 2], "c": [5, 5]})
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.remove_constant_columns(df)
        assert not any("Removing" in rec.getMessage() for rec in caplog.records)

    def test_a_column_that_varies_beyond_the_head_sample_with_a_null_is_kept(self):
        df = pd.DataFrame({"a": [1, 1, 1, 1, np.nan], "b": [7, 7, 7, 7, 7]})
        frames.remove_constant_columns(df, prewarm_size=2)
        assert list(df.columns) == ["a"]

    def test_two_values_beyond_the_head_sample_keep_the_column(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2, 2], "b": [7, 7, 7, 7, 7]})
        frames.remove_constant_columns(df, prewarm_size=2)
        assert list(df.columns) == ["a"]

    def test_twenty_constant_columns_are_listed_by_name_in_verbose_mode(self, caplog):
        import logging

        data = {f"c{i}": [1, 1] for i in range(20)}
        data["a"] = [1, 2]
        df = pd.DataFrame(data)
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.remove_constant_columns(df, verbose=True)
        assert any(rec.getMessage().startswith("Removing constant columns [") for rec in caplog.records)

    def test_a_thousand_constant_columns_are_counted_with_underscores(self, caplog):
        import logging

        data = {f"c{i}": [1, 1] for i in range(1000)}
        data["a"] = [1, 2]
        df = pd.DataFrame(data)
        with caplog.at_level(logging.WARNING, logger=frames.logger.name):
            frames.remove_constant_columns(df, verbose=True)
        assert any(rec.getMessage() == "Removing 1_000 constant columns" for rec in caplog.records)
