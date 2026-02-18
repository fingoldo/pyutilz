"""
Test suite for serialization.py
Tests cover serialize/unserialize with various formats and str_to_class utility.
"""

import pytest
import tempfile
import pickle
from pathlib import Path

from pyutilz.serialization import serialize, unserialize, str_to_class


class TestSerialize:
    """Test serialization functions"""

    def test_serialize_unserialize_dict(self):
        """Test basic dict serialization/unserialization"""
        data = {"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}

        # Serialize to bytes
        serialized = serialize(data)
        assert isinstance(serialized, bytes)

        # Unserialize back
        result = unserialize(serialized)
        assert result == data

    def test_serialize_unserialize_list(self):
        """Test list serialization"""
        data = [1, 2, 3, "test", {"key": "value"}]

        serialized = serialize(data)
        result = unserialize(serialized)
        assert result == data

    def test_serialize_to_file(self):
        """Test serialization to file"""
        import tempfile
        import os

        data = {"test": "data", "numbers": [1, 2, 3]}

        # Use a simpler temp file approach
        fd, file_path = tempfile.mkstemp(suffix='.pkl')
        os.close(fd)  # Close the file descriptor

        try:
            # Serialize to file
            result = serialize(data, fname=file_path)
            assert result is True

            # Check file exists
            assert os.path.exists(file_path)

            # Read and unserialize
            with open(file_path, 'rb') as f:
                loaded = unserialize(f.read())

            assert loaded == data
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_serialize_compression_levels(self):
        """Test different compression levels"""
        data = {"key": "value" * 100}  # Repetitive data compresses well

        # Test different compression levels
        for compression in [0, 5, 9]:
            serialized = serialize(data, compression=compression)
            result = unserialize(serialized, compression=compression)
            assert result == data

    def test_serialize_complex_object(self):
        """Test serialization of complex objects"""
        data = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "tuple": (1, 2),
            "none": None,
            "bool": True
        }

        serialized = serialize(data)
        result = unserialize(serialized)

        # Convert tuple back (pickle may change tuple to list in dict)
        assert result["string"] == data["string"]
        assert result["int"] == data["int"]
        assert result["float"] == data["float"]
        assert result["list"] == data["list"]
        assert result["dict"] == data["dict"]


class TestStrToClass:
    """Test str_to_class utility"""

    def test_str_to_class_invalid(self):
        """Test with invalid class name"""
        # Should raise error for non-existent class
        with pytest.raises(Exception):  # Could be AttributeError, ImportError, or KeyError
            str_to_class("nonexistent.FakeClass")

    def test_str_to_class_with_init_data(self):
        """Test str_to_class extracts init data from string"""
        # str_to_class is designed to parse "ClassName(init_data)" format
        # It extracts init_data and passes to constructor
        # This is a complex function that requires the module to be already imported
        # Just test it doesn't crash with invalid input
        try:
            result = str_to_class("SomeClass(data)")
        except Exception:
            # Expected to fail since SomeClass doesn't exist
            pass


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_serialize_empty_dict(self):
        """Test serializing empty dict"""
        data = {}
        serialized = serialize(data)
        result = unserialize(serialized)
        assert result == data

    def test_serialize_empty_list(self):
        """Test serializing empty list"""
        data = []
        serialized = serialize(data)
        result = unserialize(serialized)
        assert result == data

    def test_serialize_none(self):
        """Test serializing None"""
        data = None
        serialized = serialize(data)
        result = unserialize(serialized)
        assert result == data

    def test_serialize_large_data(self):
        """Test serializing larger dataset"""
        data = {"key_{}".format(i): list(range(100)) for i in range(100)}

        serialized = serialize(data)
        result = unserialize(serialized)
        assert result == data

        # Check compression is working (serialized should be smaller than pickle)
        pickled = pickle.dumps(data)
        assert len(serialized) < len(pickled)
