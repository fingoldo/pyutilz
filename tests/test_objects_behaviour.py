"""Behaviour of ``pyutilz.core.pythonlib.objects``: attribute listing, dict flattening to lists and sets, the verbose
logs of opaque objects, the attribute getter's fallbacks, the change detector used for alerts, batching and the
sortedness checks. Each expected value is worked out from the function's contract, not read back from its output."""

import logging
from types import SimpleNamespace

import pytest

from pyutilz.core.pythonlib import objects as ob


class _Attrs:
    Public = 1
    lower = 2
    _mangled = 3
    a__b = 4

    def __init__(self):
        self.Instance = 5


class _Opaque:
    def __str__(self):
        return "OPAQUE_REPR_LONG"


class TestShowMethods:
    def test_lists_every_non_dunder_name_including_private_ones(self):
        names = set(ob.show_methods(_Attrs()))
        assert {"Public", "lower", "_mangled", "a__b", "Instance"} <= names
        assert not [n for n in names if n.startswith("__") and n.endswith("__")]

    def test_uppercased_keeps_only_names_starting_with_a_capital(self):
        assert set(ob.show_methods(_Attrs(), uppercased=True)) >= {"Public", "Instance"}
        assert "lower" not in ob.show_methods(_Attrs(), uppercased=True)


class TestPrefixAndPopulate:
    def test_prefix_is_prepended_to_every_key(self):
        assert ob.prefix_dict_elems({"a": 1, "b": 2}, "p_") == {"p_a": 1, "p_b": 2}

    def test_populate_sets_each_key_as_an_attribute(self):
        obj = SimpleNamespace()
        ob.populate_object_from_dict(obj, {"x": 1, "y": "two"})
        assert (obj.x, obj.y) == (1, "two")


class TestFlattenKeysToDict:
    def test_list_items_are_merged_into_one_dict(self):
        assert ob.flatten_keys_to_dict([{"a": 1}, {"b": 2}]) == {"a": 1, "b": 2}

    def test_a_nested_dict_value_is_flattened_and_its_key_dropped(self):
        assert ob.flatten_keys_to_dict({"a": 1, "b": {"c": 2}}) == {"a": 1, "c": 2}

    def test_a_nested_list_under_a_key_is_flattened(self):
        assert ob.flatten_keys_to_dict({"x": [{"y": 3}]}) == {"y": 3}

    def test_a_tuple_value_is_kept_as_is(self):
        assert ob.flatten_keys_to_dict({"a": (1, 2)}) == {"a": (1, 2)}

    def test_a_non_container_gives_nothing(self):
        assert ob.flatten_keys_to_dict(5) == {}


class TestFlattenKeysToSet:
    def test_a_mapping_becomes_key_colon_value_pairs(self):
        assert ob.flatten_keys_to_set({"a": "b"}) == {"a:b"}

    def test_the_merge_symbol_is_configurable(self):
        assert ob.flatten_keys_to_set({"a": {"b": 1}}, dict_merge_symbol="/") == {"b/1"}

    def test_an_empty_merge_symbol_keeps_key_and_value_apart(self):
        assert ob.flatten_keys_to_set({"a": "b"}, dict_merge_symbol="") == {"a", "b"}

    def test_a_string_value_is_one_pair_and_not_recursed_into(self):
        assert ob.flatten_keys_to_set({"a": "xy"}) == {"a:xy"}

    def test_a_nested_mapping_value_is_flattened(self):
        assert ob.flatten_keys_to_set({"a": {"b": 1}}) == {"b:1"}

    def test_a_string_is_one_element(self):
        assert ob.flatten_keys_to_set("abc") == {"abc"}

    def test_a_number_is_kept_unless_stringified(self):
        assert ob.flatten_keys_to_set(5) == {5}
        assert ob.flatten_keys_to_set(5, stringify=True) == {"5"}

    def test_an_iterable_contributes_each_of_its_items(self):
        assert ob.flatten_keys_to_set([1, "a", {"k": 1}]) == {1, "a", "k:1"}

    def test_an_opaque_object_is_dropped_by_default(self):
        assert ob.flatten_keys_to_set(_Opaque()) == set()

    def test_an_opaque_object_is_its_str_when_stringified(self):
        assert ob.flatten_keys_to_set(_Opaque(), stringify=True) == {"OPAQUE_REPR_LONG"}

    def test_verbose_logs_the_size_of_a_stringified_opaque_object(self, caplog):
        with caplog.at_level(logging.INFO):
            ob.flatten_keys_to_set(_Opaque(), stringify=True, verbose=True)
        messages = [r.getMessage() for r in caplog.records if "Processing object" in r.getMessage()]
        assert messages == [f"Processing object of type {_Opaque}, size 16 as a string."]

    def test_verbose_logs_a_truncated_preview_of_a_skipped_opaque_object(self, caplog):
        with caplog.at_level(logging.INFO):
            ob.flatten_keys_to_set(_Opaque(), verbose=True, max_chars=10)
        messages = [r.getMessage() for r in caplog.records if "Skipping object" in r.getMessage()]
        assert messages == [f"Skipping object of type {_Opaque}, size 16: OPAQUE_REP ."]


class TestEnsureDictElem:
    def test_a_missing_key_is_set(self):
        d = {}
        ob.ensure_dict_elem(d, "k", 1)
        assert d == {"k": 1}

    def test_an_existing_key_is_left_alone(self):
        d = {"k": 0}
        ob.ensure_dict_elem(d, "k", 1)
        assert d == {"k": 0}


class TestGetAttr:
    def test_a_present_key_is_returned(self):
        assert ob.get_attr({"a": 1}, "a") == 1

    def test_a_missing_key_gives_an_empty_list_by_default(self):
        assert ob.get_attr({}, "a") == []

    def test_an_explicit_default_is_returned_for_a_missing_key(self):
        assert ob.get_attr({}, "a", default_value=5) == 5

    def test_an_explicit_none_default_is_honoured(self):
        assert ob.get_attr({}, "a", default_value=None) is None

    def test_a_none_value_is_replaced_when_none_is_the_unwanted_value(self):
        assert ob.get_attr({"a": None}, "a", default_value=7) == 7

    def test_a_non_none_unwanted_value_is_replaced(self):
        assert ob.get_attr({"a": "x"}, "a", default_value=7, unwanted_value="x") == 7

    def test_a_none_object_gives_the_default(self):
        assert ob.get_attr(None, "a", default_value=5) == 5

    def test_a_none_object_with_a_non_none_unwanted_value_gives_the_default(self):
        assert ob.get_attr(None, "a", default_value=5, unwanted_value="x") == 5


class TestKeysChangedEnough:
    def test_a_change_at_exactly_the_threshold_counts(self):
        assert ob.keys_changed_enough({"a": 110.0}, {"a": 100.0}, min_change_percent=10.0) is True

    def test_a_change_just_below_the_threshold_does_not(self):
        assert ob.keys_changed_enough({"a": 109.99}, {"a": 100.0}, min_change_percent=10.0) is False

    def test_a_decrease_is_measured_by_its_absolute_size(self):
        assert ob.keys_changed_enough({"a": 90.0}, {"a": 100.0}, min_change_percent=10.0) is True

    def test_the_change_is_relative_to_the_previous_value(self):
        assert ob.keys_changed_enough({"a": 150.0}, {"a": 100.0}, min_change_percent=10.0) is True
        assert ob.keys_changed_enough({"a": 105.0}, {"a": 100.0}, min_change_percent=10.0) is False

    def test_only_keys_containing_the_given_text_are_compared(self):
        assert ob.keys_changed_enough({"a": 200.0, "b": 100.0}, {"a": 100.0, "b": 100.0}, key_contains="b") is False

    def test_a_key_that_vanished_counts_as_a_change_and_is_logged(self, caplog):
        with caplog.at_level(logging.DEBUG):
            assert ob.keys_changed_enough({}, {"a": 1.0}) is True
        assert any("present in prev_obj but absent from obj" in r.getMessage() for r in caplog.records)

    def test_a_non_numeric_previous_value_is_ignored(self):
        assert ob.keys_changed_enough({}, {"s": "x"}) is False

    def test_a_non_numeric_new_value_is_not_a_change(self):
        assert ob.keys_changed_enough({"a": "x"}, {"a": 1.0}) is False

    def test_a_zero_previous_value_changes_on_any_nonzero_new_value(self):
        assert ob.keys_changed_enough({"a": 0.05}, {"a": 0.0}) is True
        assert ob.keys_changed_enough({"a": 0.0}, {"a": 0.0}) is False


class TestSequences:
    def test_unpack_counter_keeps_the_tokens(self):
        assert ob.unpack_counter([("a", 2), ("b", 1)]) == ["a", "b"]

    def test_a_list_is_returned_unchanged(self):
        lst = [1]
        assert ob.ensure_list_set_tuple(lst) is lst

    @pytest.mark.parametrize("value", [{1}, frozenset({1}), (1,)])
    def test_sets_and_tuples_are_returned_unchanged(self, value):
        assert ob.ensure_list_set_tuple(value) is value

    def test_a_scalar_is_wrapped_in_a_list(self):
        assert ob.ensure_list_set_tuple(5) == [5]

    def test_a_dict_is_wrapped_not_unpacked(self):
        assert ob.ensure_list_set_tuple({"a": 1}) == [{"a": 1}]

    def test_a_substring_match_of_any_element_is_true(self):
        assert ob.anyof_elements_in_string(["x", "y"], "axb") is True

    def test_no_element_in_the_target_is_false(self):
        assert ob.anyof_elements_in_string(["x", "y"], "abc") is False

    def test_no_elements_is_false(self):
        assert ob.anyof_elements_in_string([], "abc") is False

    def test_a_dict_is_filtered_by_value_type(self):
        assert ob.filter_elements_by_type({"a": "t", "b": 3}, allowed_types=(str,)) == {"a": "t"}

    def test_a_sequence_is_filtered_by_type_with_numbers_and_strings_by_default(self):
        assert ob.filter_elements_by_type([1, "a", None]) == [1, "a"]

    def test_a_set_is_filtered_to_a_list(self):
        assert sorted(ob.filter_elements_by_type({"a", 1, "test"}, allowed_types=(str,))) == ["a", "test"]


class TestBatch:
    def test_consecutive_slices_of_size_n_with_a_short_last_one(self):
        assert list(ob.batch(list(range(10)), 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_the_default_size_is_one(self):
        assert list(ob.batch([1, 2])) == [[1], [2]]

    def test_an_empty_sequence_gives_no_batches(self):
        assert list(ob.batch([], 3)) == []

    def test_a_size_below_one_is_refused_with_its_value(self):
        with pytest.raises(ValueError) as info:
            list(ob.batch([1, 2], 0))
        assert str(info.value) == "batch: n must be >= 1, got 0"

    def test_a_negative_size_is_refused_too(self):
        with pytest.raises(ValueError, match="got -2"):
            list(ob.batch([1], -2))


class TestSortedness:
    def test_a_non_increasing_list_with_repeats_is_sorted(self):
        assert ob.list_is_non_increasing([3, 2, 2, 1]) is True

    def test_an_increasing_pair_is_not_non_increasing(self):
        assert ob.list_is_non_increasing([1, 2]) is False

    def test_an_empty_or_single_list_is_both_orders(self):
        assert ob.list_is_non_increasing([]) is True
        assert ob.list_is_non_decreasing([5]) is True

    def test_a_non_decreasing_list_with_repeats_is_sorted(self):
        assert ob.list_is_non_decreasing([1, 2, 2, 3]) is True

    def test_a_decreasing_pair_is_not_non_decreasing(self):
        assert ob.list_is_non_decreasing([2, 1]) is False


class TestSortDict:
    def test_sorted_by_value_ascending(self):
        assert list(ob.sort_dict_by_value({"a": 2, "b": 1})) == ["b", "a"]

    def test_sorted_by_value_descending(self):
        assert list(ob.sort_dict_by_value({"a": 2, "b": 1}, reverse=True)) == ["a", "b"]

    def test_sorted_by_key_ascending(self):
        assert list(ob.sort_dict_by_key({"b": 1, "a": 2})) == ["a", "b"]

    def test_sorted_by_key_descending(self):
        assert list(ob.sort_dict_by_key({"b": 1, "a": 2}, reverse=True)) == ["b", "a"]


class TestVerboseDefaultsAndRecursionArguments:
    def test_verbose_is_off_by_default(self, caplog):
        with caplog.at_level(logging.INFO):
            ob.flatten_keys_to_set(_Opaque())
        assert not [r for r in caplog.records if "Skipping object" in r.getMessage()]

    def test_the_default_preview_is_ten_characters(self, caplog):
        with caplog.at_level(logging.INFO):
            ob.flatten_keys_to_set(_Opaque(), verbose=True)
        messages = [r.getMessage() for r in caplog.records if "Skipping object" in r.getMessage()]
        assert messages == [f"Skipping object of type {_Opaque}, size 16: OPAQUE_REP ."]

    def test_stringify_reaches_an_opaque_object_nested_in_a_list_under_a_key(self):
        assert ob.flatten_keys_to_set({"a": [_Opaque()]}, stringify=True) == {"OPAQUE_REPR_LONG"}

    def test_max_chars_reaches_the_preview_inside_a_mapping(self, caplog):
        with caplog.at_level(logging.INFO):
            ob.flatten_keys_to_set({"a": [_Opaque()]}, verbose=True, max_chars=4)
        messages = [r.getMessage() for r in caplog.records if "Skipping object" in r.getMessage()]
        assert messages == [f"Skipping object of type {_Opaque}, size 16: OPAQ ."]


class TestThresholdBoundary:
    def test_the_default_threshold_is_ten_percent(self):
        assert ob.keys_changed_enough({"a": 110.05}, {"a": 100.0}) is True
        assert ob.keys_changed_enough({"a": 100.5}, {"a": 100.0}) is False


class TestGetAttrNonNoneGuard:
    def test_an_object_equal_to_the_unwanted_value_gives_the_default(self):
        assert ob.get_attr("x", "a", default_value=7, unwanted_value="x") == 7


class TestShowMethodsDunderEdges:
    def test_a_name_with_only_a_leading_double_underscore_is_kept(self):
        obj = _Attrs()
        setattr(obj, "__priv", 1)
        assert "__priv" in ob.show_methods(obj)

    def test_a_name_with_only_a_trailing_double_underscore_is_kept(self):
        obj = _Attrs()
        obj.trailing__ = 2
        assert "trailing__" in ob.show_methods(obj)


class _AmbiguousEq:
    """Stands in for a numpy array: ``==`` returns an object whose truth value is refused."""

    def __eq__(self, other):
        return self

    def __bool__(self):
        raise ValueError("truth value is ambiguous")


class TestGetAttrNoneIdentity:
    def test_a_value_whose_equality_is_ambiguous_is_returned_untouched(self):
        value = _AmbiguousEq()
        assert ob.get_attr({"a": value}, "a") is value

    def test_a_none_value_is_replaced_by_the_default(self):
        assert ob.get_attr({"a": None}, "a", default_value=7) == 7
