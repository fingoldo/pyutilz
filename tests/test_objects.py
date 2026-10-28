"""Behaviour of ``pyutilz.core.pythonlib.objects``: attribute and dict-key lookups with a default and an unwanted value,
dict flattening to leaf keys or ``key:value`` strings, the change test used to decide when a cache is stale, the
type filter that keeps only numbers and strings, batching, monotonic checks and the dict sorters."""

from __future__ import annotations

from pyutilz.core.pythonlib import objects


class _Holder:
    pass


class TestLookups:
    def test_a_present_key_is_returned(self):
        assert objects.get_attr({"x": 1}, "x") == 1

    def test_a_missing_key_gives_the_default(self):
        assert objects.get_attr({"x": 1}, "y", "dflt") == "dflt"

    def test_a_missing_object_gives_the_default(self):
        assert objects.get_attr(None, "x", "d") == "d"

    def test_a_value_equal_to_the_unwanted_one_gives_the_default(self):
        assert objects.get_attr({"x": 1}, "x", 0, unwanted_value=1) == 0

    def test_ensure_dict_elem_sets_the_key(self):
        dct: dict = {}
        objects.ensure_dict_elem(dct, "k", 5)
        assert dct == {"k": 5}

    def test_prefix_dict_elems_renames_every_key(self):
        assert objects.prefix_dict_elems({"a": 1, "b": 2}, "p_") == {"p_a": 1, "p_b": 2}

    def test_populate_object_sets_each_attribute(self):
        holder = _Holder()
        objects.populate_object_from_dict(holder, {"u": 3})
        assert holder.u == 3


class TestFlattening:
    def test_nested_keys_are_flattened_to_their_leaf_names(self):
        assert objects.flatten_keys_to_dict({"a": {"b": 1}, "c": 2}) == {"b": 1, "c": 2}

    def test_flattening_to_a_set_gives_key_colon_value_strings(self):
        assert objects.flatten_keys_to_set({"a": {"b": 1}, "c": 2}) == {"b:1", "c:2"}


class TestChangeTest:
    def test_a_changed_value_is_enough_change(self):
        assert objects.keys_changed_enough({"a": 1, "b": 2}, {"a": 1, "b": 1}) is True

    def test_an_unchanged_mapping_is_not_enough_change(self):
        assert objects.keys_changed_enough({"a": 1}, {"a": 1}) is False


class TestCollections:
    def test_a_counter_result_is_reduced_to_its_tokens(self):
        assert objects.unpack_counter([("surgery", 252), ("theatre", 251)]) == ["surgery", "theatre"]

    def test_a_list_or_tuple_is_returned_as_is(self):
        assert objects.ensure_list_set_tuple((1, 2)) == (1, 2)

    def test_a_scalar_is_wrapped_in_a_list(self):
        assert objects.ensure_list_set_tuple(5) == [5]

    def test_any_element_found_in_the_string_is_enough(self):
        assert objects.anyof_elements_in_string(["ab", "xy"], "zabz") is True

    def test_no_element_found_in_the_string_is_not_enough(self):
        assert objects.anyof_elements_in_string(["q"], "zabz") is False

    def test_a_sequence_keeps_only_numbers_and_strings(self):
        assert objects.filter_elements_by_type([1, "a", None, 2.5]) == [1, "a", 2.5]

    def test_a_dict_keeps_only_the_entries_of_allowed_types(self):
        assert objects.filter_elements_by_type({"a": 1, "b": None}) == {"a": 1}

    def test_batch_yields_fixed_size_chunks_with_a_short_tail(self):
        assert list(objects.batch([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


class TestMonotonic:
    def test_a_non_increasing_sequence_allows_equal_neighbours(self):
        assert objects.list_is_non_increasing([3, 2, 2]) is True

    def test_a_rise_breaks_non_decreasing(self):
        assert objects.list_is_non_decreasing([1, 3, 2]) is False


class TestSorters:
    def test_sort_by_value_orders_ascending(self):
        assert list(objects.sort_dict_by_value({"a": 2, "b": 1})) == ["b", "a"]

    def test_sort_by_key_honours_reverse(self):
        assert list(objects.sort_dict_by_key({"b": 1, "a": 2}, reverse=True)) == ["b", "a"]
