"""Behaviour of ``pyutilz.data.polarslib.aggregations``: a missing field group is filled from the frame's columns of the matching
dtype, a given group is kept as it is, a subgroup drops its own filter field from the categorical list, exclusions remove the
named fields from every group, and a default is used only for a None value. Frames are small polars frames built in memory."""

from __future__ import annotations

import sys

import polars as pl
import pytest

from pyutilz.data.polarslib import aggregations


def _frame():
    return pl.DataFrame(
        {
            "flag": [True, False],
            "amount": [1.5, 2.5],
            "count": [1, 2],
            "city": ["a", "b"],
            "when": [None, None],
        },
        schema_overrides={"when": pl.Datetime},
    )


class TestOrDefault:
    def test_none_takes_the_default(self):
        assert aggregations._or_default(None, 3) == 3

    def test_any_other_value_is_kept_even_when_falsy(self):
        assert aggregations._or_default(0, 3) == 0
        assert aggregations._or_default("", "x") == ""


class TestExpandDefaultFields:
    def test_missing_groups_are_filled_from_the_frame_by_dtype(self):
        booleans, timestamps, numerics, categoricals = aggregations._expand_default_fields(_frame(), None, None, None, None)
        assert booleans == ["flag"]
        assert timestamps == ["when"]
        assert set(numerics) == {"amount", "count"}
        assert categoricals == ["city"]

    def test_a_given_group_is_kept_as_it_is(self):
        booleans, _, numerics, _ = aggregations._expand_default_fields(_frame(), ["mine"], None, ["amount"], None)
        assert booleans == ["mine"]
        assert numerics == ["amount"]


class TestSubgroupCategoricals:
    def test_the_filter_field_is_removed_from_a_copy(self):
        original = ["city", "shop"]
        assert aggregations._subgroup_categoricals(original, "shop") == ["city"]
        assert original == ["city", "shop"]

    def test_a_filter_field_not_in_the_list_leaves_it_unchanged(self):
        assert aggregations._subgroup_categoricals(["city"], "shop") == ["city"]


class TestWithout:
    def test_the_excluded_names_are_removed(self):
        assert sorted(aggregations._without(["a", "b", "c"], ["b"])) == ["a", "c"]

    def test_an_empty_or_missing_list_is_returned_unchanged(self):
        assert aggregations._without(None, ["a"]) is None
        assert aggregations._without([], ["a"]) == []


class TestApplyExclusions:
    def test_no_exclusions_return_every_group_unchanged(self):
        groups = (["f"], ["t"], ["n"], ["c"])
        assert aggregations._apply_exclusions(*groups, None) == groups

    def test_the_excluded_field_is_dropped_from_every_group(self):
        out = aggregations._apply_exclusions(["x", "y"], ["x"], ["x", "z"], ["x"], ["x"])
        assert [sorted(group) for group in out] == [["y"], [], ["z"], []]


class TestEngineChecks:
    def test_polars_ds_is_imported_when_the_cpu_has_avx2(self, monkeypatch):
        seen = []
        monkeypatch.setattr(aggregations._facade, "check_cpu_flag", lambda flag: seen.append(flag) or True)
        monkeypatch.setitem(sys.modules, "polars_ds", None)
        with pytest.raises(ImportError):
            aggregations._import_polars_ds_for_avx2()
        assert seen == ["avx2"]

    def test_polars_ds_is_not_imported_without_avx2(self, monkeypatch):
        seen = []
        monkeypatch.setattr(aggregations._facade, "check_cpu_flag", lambda flag: seen.append(flag) or False)
        monkeypatch.setitem(sys.modules, "polars_ds", None)
        aggregations._import_polars_ds_for_avx2()
        assert seen == ["avx2"]

    def test_a_gpu_engine_without_cuda_warns(self, monkeypatch, caplog):
        monkeypatch.setattr(aggregations._facade, "is_cuda_available", lambda: False)
        with caplog.at_level("WARNING"):
            aggregations._warn_if_cuda_unavailable("gpu")
        assert any("CUDA seems to be unavailable" in rec.getMessage() for rec in caplog.records)

    def test_a_gpu_engine_with_cuda_does_not_warn(self, monkeypatch, caplog):
        monkeypatch.setattr(aggregations._facade, "is_cuda_available", lambda: True)
        with caplog.at_level("WARNING"):
            aggregations._warn_if_cuda_unavailable("gpu")
        assert caplog.records == []

    def test_a_cpu_engine_never_warns_about_cuda(self, monkeypatch, caplog):
        monkeypatch.setattr(aggregations._facade, "is_cuda_available", lambda: False)
        with caplog.at_level("WARNING"):
            aggregations._warn_if_cuda_unavailable("cpu")
        assert caplog.records == []
