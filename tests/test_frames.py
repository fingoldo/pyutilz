"""Behaviour of ``pyutilz.data.pandaslib.frames``: standard responses are nulled only when they repeat more than the
threshold (optionally counting distinct persons instead of rows), the verbose mode prints the values it nulls, and the
column prefixer returns a (frame, rename mapping) pair whether it renames in place or on a copy, honouring special
prefixes and exclusions."""

from __future__ import annotations

import pandas as pd

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
