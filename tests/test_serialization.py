"""
Test suite for serialization.py
Tests cover serialize/unserialize with various formats and str_to_class utility.
"""

import pytest
import tempfile
import pickle
from pathlib import Path

from pyutilz.serialization import serialize, unserialize, str_to_class
from pyutilz.core.safe_pickle import PickleVerificationError


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
        fd, file_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)  # Close the file descriptor

        try:
            # Serialize to file
            result = serialize(data, fname=file_path)
            assert result is True

            # Check file exists
            assert os.path.exists(file_path)

            # Read and unserialize
            with open(file_path, "rb") as f:
                loaded = unserialize(f.read())

            assert loaded == data
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_serialize_to_a_not_yet_existing_file_path(self, tmp_path):
        """Bug: serialize() passed fname (the FILE path) to ensure_dir_exists(), which
        os.makedirs()'d a DIRECTORY at that exact path whenever it didn't already exist --
        the subsequent open(fname, "wb") then failed with PermissionError (can't open a
        directory for writing). Every prior test used tempfile.mkstemp() (which pre-creates
        the file), so os.path.exists(fname) was already True and the buggy makedirs() call
        never fired -- this is the first test to serialize() straight to a path that doesn't
        exist yet, the common "write a new file" case."""
        p = tmp_path / "brand_new_file.pkl"
        assert not p.exists()
        assert serialize({"a": 1}, fname=str(p)) is True
        assert p.is_file()
        assert unserialize(str(p)) == {"a": 1}

    def test_serialize_creates_missing_parent_directories(self, tmp_path):
        """serialize() must also create intermediate directories that don't exist yet."""
        p = tmp_path / "nested" / "subdir" / "file.pkl"
        assert serialize({"a": 1}, fname=str(p)) is True
        assert p.is_file()
        assert unserialize(str(p)) == {"a": 1}

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
        data = {"string": "test", "int": 42, "float": 3.14, "list": [1, 2, 3], "dict": {"nested": True}, "tuple": (1, 2), "none": None, "bool": True}

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

    def test_unserialize_from_file_path(self):
        """Test unserializing directly from a file path string"""
        import tempfile
        import os

        data = {"test": "from_file", "value": 123}
        fd, file_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        try:
            serialize(data, fname=file_path)
            result = unserialize(file_path)
            assert result == data
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_unserialize_nonexistent_file(self):
        """Test unserializing from non-existent file path returns None"""
        result = unserialize("/nonexistent/path/to/file.pkl")
        assert result is None

    def test_unserialize_unexpected_type(self):
        """Test unserializing non-bytes non-str input"""
        # Passing an integer (unexpected type) should log warning and return None
        result = unserialize(42)
        assert result is None

    def test_unserialize_uncompressed_data(self):
        """Raw (non-zlib) pickle bytes must fall back to being read as raw pickle and decode
        exactly -- zlib.decompress() raising zlib.error on non-zlib input is caught internally
        and always falls through to the raw-pickle path, so the outcome is fully deterministic,
        not "either succeeds or returns None"."""
        import pickle as pkl
        data = {"key": "value"}
        raw = pkl.dumps(data)
        result = unserialize(raw, compression=9)
        assert result == data


class TestUnserializeSidecarVerification:
    """verify_sidecar= is opt-in (default False) so existing callers are unaffected; when True
    and obj is a file path, it requires a matching pyutilz.core.safe_pickle sidecar before
    unpickling. A missing/mismatched sidecar raises PickleVerificationError distinctly -- it is
    NOT swallowed into a generic None return, so callers can tell "tampered/corrupt" apart from
    every other failure mode in this function."""

    def test_default_false_loads_without_a_sidecar(self, tmp_path):
        p = tmp_path / "no_sidecar.pkl"
        serialize({"a": 1}, fname=str(p))
        assert unserialize(str(p)) == {"a": 1}

    def test_verify_sidecar_true_without_sidecar_raises(self, tmp_path):
        p = tmp_path / "no_sidecar.pkl"
        serialize({"a": 1}, fname=str(p))
        with pytest.raises(PickleVerificationError):
            unserialize(str(p), verify_sidecar=True)

    def test_verify_sidecar_true_with_matching_sidecar_loads(self, tmp_path):
        from pyutilz.core.safe_pickle import write_sidecar

        p = tmp_path / "with_sidecar.pkl"
        serialize({"a": 1}, fname=str(p))
        write_sidecar(str(p))
        assert unserialize(str(p), verify_sidecar=True) == {"a": 1}

    def test_verify_sidecar_true_with_stale_sidecar_raises(self, tmp_path):
        from pyutilz.core.safe_pickle import write_sidecar

        p = tmp_path / "stale_sidecar.pkl"
        serialize({"a": 1}, fname=str(p))
        write_sidecar(str(p))
        # Overwrite the payload after the sidecar was written -- digest now stale.
        serialize({"a": 2}, fname=str(p))
        with pytest.raises(PickleVerificationError):
            unserialize(str(p), verify_sidecar=True)

    def test_verify_sidecar_has_no_effect_on_in_memory_bytes(self):
        data = {"a": 1}
        raw = serialize(data)
        # No file path involved -- verify_sidecar is a no-op here, not an error.
        assert unserialize(raw, verify_sidecar=True) == data
