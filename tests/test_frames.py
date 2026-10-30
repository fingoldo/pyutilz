"""Behaviour of ``pyutilz.data.pandaslib.frames``: standard responses are nulled only when they repeat more than the
threshold (optionally counting distinct persons instead of rows), the verbose mode prints the values it nulls, and the
column prefixer returns a (frame, rename mapping) pair whether it renames in place or on a copy, honouring special
prefixes and exclusions."""

from __future__ import annotations

import pandas as pd
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
