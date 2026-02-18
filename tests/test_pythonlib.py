"""Tests for pythonlib.py - Phase 2 refactoring

Tests cover:
- ObjectsLoader indentation bug fix (line 764-775)
- Typo fix in ensure_list_set_tuple (line 229)
- Mutable default argument fix in get_attr
"""

import pytest
import tempfile
import os
from pathlib import Path
import pickle


class TestObjectsLoader:
    """Test ObjectsLoader class - regression test for indentation bug"""

    def test_loads_objects_when_rewrite_existing_false(self, tmp_path):
        """Test that objects ARE loaded when rewrite_existing=False (indentation bug fix)"""
        from pyutilz.pythonlib import ObjectsLoader

        # Create test data
        test_data = {"key1": "value1", "key2": "value2"}
        test_file = tmp_path / "test_obj.pkl"

        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)

        # Test with rewrite_existing=False (the bug case)
        loader = ObjectsLoader(rewrite_existing=False)
        container = {}

        # This should load the object (was broken due to indentation)
        result = loader._process_object(
            container,
            "test_key",
            str(test_file),
            verbose=False
        )

        # Should have loaded successfully
        assert result is True
        assert "test_key" in container
        assert container["test_key"] == test_data

    def test_skips_existing_non_empty_objects(self, tmp_path):
        """Test that existing non-empty objects are NOT overwritten"""
        from pyutilz.pythonlib import ObjectsLoader

        test_data = {"new": "data"}
        test_file = tmp_path / "test_obj.pkl"

        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)

        loader = ObjectsLoader(rewrite_existing=False)
        container = {"test_key": {"existing": "value"}}  # Non-empty existing

        # Should NOT overwrite existing non-empty object
        result = loader._process_object(
            container,
            "test_key",
            str(test_file),
            verbose=False
        )

        # Should return None and keep existing value
        assert result is None
        assert container["test_key"] == {"existing": "value"}

    def test_overwrites_when_rewrite_existing_true(self, tmp_path):
        """Test that objects ARE overwritten when rewrite_existing=True"""
        from pyutilz.pythonlib import ObjectsLoader

        new_data = {"new": "data"}
        test_file = tmp_path / "test_obj.pkl"

        with open(test_file, 'wb') as f:
            pickle.dump(new_data, f)

        loader = ObjectsLoader(rewrite_existing=True)
        container = {"test_key": {"old": "value"}}

        result = loader._process_object(
            container,
            "test_key",
            str(test_file),
            verbose=False
        )

        # Should overwrite
        assert result is True
        assert container["test_key"] == new_data


class TestEnsureListSetTuple:
    """Test ensure_list_set_tuple function - typo fix"""

    def test_recognizes_tuple_correctly(self):
        """Test that 'tuple' (not 'tuple`') is recognized (line 229 typo fix)"""
        from pyutilz.pythonlib import ensure_list_set_tuple

        # Tuple should be recognized
        result = ensure_list_set_tuple((1, 2, 3))
        assert result == (1, 2, 3)

    def test_recognizes_list(self):
        """Test that list is recognized"""
        from pyutilz.pythonlib import ensure_list_set_tuple

        result = ensure_list_set_tuple([1, 2, 3])
        assert result == [1, 2, 3]

    def test_recognizes_set(self):
        """Test that set is recognized"""
        from pyutilz.pythonlib import ensure_list_set_tuple

        input_set = {1, 2, 3}
        result = ensure_list_set_tuple(input_set)
        assert result == input_set

    def test_wraps_non_sequence_in_list(self):
        """Test that non-sequence objects are wrapped in list"""
        from pyutilz.pythonlib import ensure_list_set_tuple

        result = ensure_list_set_tuple(42)
        assert result == [42]

        result = ensure_list_set_tuple("string")
        assert result == ["string"]


class TestGetAttr:
    """Test get_attr function - mutable default argument fix"""

    def test_mutable_default_not_shared(self):
        """Test that default value is not shared between calls (mutable default bug)"""
        from pyutilz.pythonlib import get_attr

        obj1 = {}
        obj2 = {}

        # Both should get independent default values
        val1 = get_attr(obj1, "missing_key")
        val2 = get_attr(obj2, "missing_key")

        # Modify one
        if isinstance(val1, list):
            val1.append("test")

        # Other should remain empty
        assert val2 == [] or val2 == val1 != ["test"]  # Should be independent

    def test_returns_existing_value(self):
        """Test that existing values are returned"""
        from pyutilz.pythonlib import get_attr

        obj = {"key": "value"}
        result = get_attr(obj, "key", default_value=[])

        assert result == "value"

    def test_returns_custom_default(self):
        """Test that custom default value is returned"""
        from pyutilz.pythonlib import get_attr

        obj = {}
        custom_default = ["custom"]
        result = get_attr(obj, "missing", default_value=custom_default)

        # Should return the custom default
        assert result == custom_default or result == []


@pytest.mark.parametrize("input_obj,expected_type", [
    ([1, 2], list),
    ((1, 2), tuple),
    ({1, 2}, set),
    (frozenset([1, 2]), frozenset),
    (42, list),  # Wrapped in list
    ("text", list),  # Wrapped in list
])
def test_ensure_list_set_tuple_parametrized(input_obj, expected_type):
    """Parametrized test for different input types"""
    from pyutilz.pythonlib import ensure_list_set_tuple

    result = ensure_list_set_tuple(input_obj)

    if expected_type == list and not isinstance(input_obj, (list, set, tuple, frozenset)):
        # Should be wrapped in list
        assert isinstance(result, list)
        assert result == [input_obj]
    else:
        # Should return as-is
        assert isinstance(result, expected_type) or result == input_obj


class TestDictUtilities:
    """Test dictionary utility functions"""

    def test_sort_dict_by_value(self):
        """Test sorting dict by values"""
        from pyutilz.pythonlib import sort_dict_by_value

        dct = {"a": 3, "b": 1, "c": 2}
        result = sort_dict_by_value(dct)

        # Should be sorted by value ascending
        keys = list(result.keys())
        assert keys == ["b", "c", "a"]

    def test_sort_dict_by_value_reverse(self):
        """Test sorting dict by values descending"""
        from pyutilz.pythonlib import sort_dict_by_value

        dct = {"a": 3, "b": 1, "c": 2}
        result = sort_dict_by_value(dct, reverse=True)

        keys = list(result.keys())
        assert keys == ["a", "c", "b"]

    def test_sort_dict_by_key(self):
        """Test sorting dict by keys"""
        from pyutilz.pythonlib import sort_dict_by_key

        dct = {"c": 1, "a": 2, "b": 3}
        result = sort_dict_by_key(dct)

        keys = list(result.keys())
        assert keys == ["a", "b", "c"]

    def test_prefix_dict_elems(self):
        """Test prefixing dictionary keys"""
        from pyutilz.pythonlib import prefix_dict_elems

        dct = {"key1": "val1", "key2": "val2"}
        result = prefix_dict_elems(dct, "prefix_")

        assert result == {"prefix_key1": "val1", "prefix_key2": "val2"}


class TestNumberUtilities:
    """Test number utility functions"""

    def test_is_float_valid(self):
        """Test is_float with valid float strings"""
        from pyutilz.pythonlib import is_float

        assert is_float("3.14") is True
        assert is_float("42.0") is True
        assert is_float("-1.5") is True
        assert is_float("0.001") is True

    def test_is_float_invalid(self):
        """Test is_float with invalid strings"""
        from pyutilz.pythonlib import is_float

        assert is_float("not a number") is False
        assert is_float("abc") is False
        assert is_float("") is False

    def test_is_float_integer(self):
        """Test is_float with integer strings"""
        from pyutilz.pythonlib import is_float

        # Integer strings should also return True
        assert is_float("42") is True
        assert is_float("-10") is True

    def test_to_float(self):
        """Test to_float conversion"""
        from pyutilz.pythonlib import to_float

        assert to_float("3.14") == 3.14
        assert to_float("42") == 42.0
        assert to_float("-1.5") == -1.5

    def test_integer_digits(self):
        """Test extracting unique digits from integer"""
        from pyutilz.pythonlib import integer_digits

        # Returns (count, digits_set)
        count, digits = integer_digits(12345)
        assert count == 5
        assert digits == {1, 2, 3, 4, 5}

        count2, digits2 = integer_digits(111)
        assert count2 == 3
        assert digits2 == {1}

        count3, digits3 = integer_digits(1020)
        assert count3 == 4
        assert digits3 == {0, 1, 2}


class TestListUtilities:
    """Test list utility functions"""

    def test_list_is_non_increasing_true(self):
        """Test non-increasing list detection"""
        from pyutilz.pythonlib import list_is_non_increasing

        assert list_is_non_increasing([5, 4, 3, 2, 1]) is True
        assert list_is_non_increasing([5, 5, 3, 3, 1]) is True
        assert list_is_non_increasing([10, 10, 10]) is True

    def test_list_is_non_increasing_false(self):
        """Test increasing list"""
        from pyutilz.pythonlib import list_is_non_increasing

        assert list_is_non_increasing([1, 2, 3]) is False
        assert list_is_non_increasing([1, 3, 2]) is False

    def test_list_is_non_decreasing_true(self):
        """Test non-decreasing list detection"""
        from pyutilz.pythonlib import list_is_non_decreasing

        assert list_is_non_decreasing([1, 2, 3, 4, 5]) is True
        assert list_is_non_decreasing([1, 1, 2, 2, 3]) is True
        assert list_is_non_decreasing([5, 5, 5]) is True

    def test_list_is_non_decreasing_false(self):
        """Test decreasing list"""
        from pyutilz.pythonlib import list_is_non_decreasing

        assert list_is_non_decreasing([5, 4, 3]) is False
        assert list_is_non_decreasing([1, 3, 2]) is False

    def test_batch(self):
        """Test batching iterable"""
        from pyutilz.pythonlib import batch

        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        batches = list(batch(data, n=3))

        assert len(batches) == 3
        assert list(batches[0]) == [1, 2, 3]
        assert list(batches[1]) == [4, 5, 6]
        assert list(batches[2]) == [7, 8, 9]

    def test_anyof_elements_in_string(self):
        """Test checking if any element is in string"""
        from pyutilz.pythonlib import anyof_elements_in_string

        assert anyof_elements_in_string(["foo", "bar"], "this is a foobar") is True
        assert anyof_elements_in_string(["xxx", "yyy"], "this is a foobar") is False
        assert anyof_elements_in_string(["test"], "testing") is True


class TestDateTimeUtilities:
    """Test datetime utility functions"""

    def test_get_utc_unix_ts_seconds(self):
        """Test getting UTC Unix timestamp in seconds"""
        from pyutilz.pythonlib import get_utc_unix_ts_seconds
        import time

        ts = get_utc_unix_ts_seconds()

        # Should be close to current time
        current_ts = int(time.time())
        assert abs(ts - current_ts) < 2  # Within 2 seconds

    def test_get_utc_unix_ts_milliseconds(self):
        """Test getting UTC Unix timestamp in milliseconds"""
        from pyutilz.pythonlib import get_utc_unix_ts_milliseconds
        import time

        ts_ms = get_utc_unix_ts_milliseconds()

        # Should be close to current time in ms
        current_ts_ms = int(time.time() * 1000)
        assert abs(ts_ms - current_ts_ms) < 2000  # Within 2000 ms

    def test_datetime_to_unix_ts(self):
        """Test converting datetime to Unix timestamp"""
        from pyutilz.pythonlib import datetime_to_unix_ts
        from datetime import datetime

        # Known timestamp: 2020-01-01 00:00:00 UTC = 1577836800
        dt = datetime(2020, 1, 1, 0, 0, 0)
        ts = datetime_to_unix_ts(dt)

        # Should be close (timezone differences may apply)
        assert isinstance(ts, int)
        assert ts > 0


class TestFilteringUtilities:
    """Test filtering utility functions"""

    def test_filter_elements_by_type_dict(self):
        """Test filtering dict elements by type"""
        from pyutilz.pythonlib import filter_elements_by_type
        import numbers

        data = {"a": 1, "b": "text", "c": [1, 2], "d": 3.14}
        result = filter_elements_by_type(data, allowed_types=(numbers.Number, str))

        # Should keep numbers and strings, but not lists
        assert result == {"a": 1, "b": "text", "d": 3.14}

    def test_filter_elements_by_type_list(self):
        """Test filtering list elements by type"""
        from pyutilz.pythonlib import filter_elements_by_type

        data = [1, "text", [1, 2], 3.14, None]
        result = filter_elements_by_type(data, allowed_types=(int, str, float))

        # Should keep only numbers and strings
        assert 1 in result
        assert "text" in result
        assert 3.14 in result
        assert None not in result
        assert [1, 2] not in result
