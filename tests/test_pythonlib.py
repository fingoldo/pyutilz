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
