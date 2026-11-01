"""
Test suite for serialization.py
Tests cover serialize/unserialize with various formats and str_to_class utility.
"""

import os

import pytest
import pickle

from pyutilz.core.serialization import serialize, unserialize, str_to_class
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

    def test_serialize_failure_logs_fname_and_type(self, caplog):
        """Regression (2026-07-21 audit round 2, MEDIUM): a serialize() failure used to log a
        bare traceback with no fname/obj info -- identifying WHICH of a batch job's many
        objects/paths failed required re-running under a debugger. The exception log now
        includes both.

        The failure is also RE-RAISED now (2026-09-03 audit, F115): returning a bare None let a
        caller store the None as if it were the payload, and the round-trip partner
        ``unserialize`` already raises. The log assertions below are unchanged."""

        class _Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle this")

        import logging

        with caplog.at_level(logging.ERROR), pytest.raises(TypeError, match="cannot pickle this"):
            serialize(_Unpicklable(), fname="some_target_file.bin")
        assert "some_target_file.bin" in caplog.text
        assert "_Unpicklable" in caplog.text


class TestStrToClass:
    """Test str_to_class utility"""

    def test_str_to_class_invalid(self):
        """Test with invalid class name"""
        # Should raise error for non-existent class
        # Enumerated, not a blind `Exception`: a bare catch-all passes on ANY future failure
        # (e.g. a TypeError from a signature change), so it could not detect the regression it
        # exists to catch.
        with pytest.raises((AttributeError, ImportError, KeyError, ModuleNotFoundError, ValueError)):
            str_to_class("nonexistent.FakeClass")

    def test_str_to_class_with_init_data(self):
        """Test str_to_class extracts init data from string"""
        # str_to_class is designed to parse "ClassName(init_data)" format
        # It extracts init_data and passes to constructor
        # This is a complex function that requires the module to be already imported
        # Just test it doesn't crash with invalid input
        with pytest.raises(AttributeError):
            str_to_class("SomeClass(data)")


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
        data = {f"key_{i}": list(range(100)) for i in range(100)}

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
        """A missing path raises: None is reserved for a successfully-unpickled None."""
        with pytest.raises(FileNotFoundError):
            unserialize("/nonexistent/path/to/file.pkl")

    def test_unserialize_unexpected_type(self):
        """An unsupported input type raises TypeError rather than returning None."""
        with pytest.raises(TypeError):
            unserialize(42)

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

    def test_default_round_trips_a_file_serialize_wrote(self, tmp_path):
        p = tmp_path / "round_trip.pkl"
        serialize({"a": 1}, fname=str(p))
        assert unserialize(str(p)) == {"a": 1}

    def test_default_refuses_a_file_without_a_sidecar(self, tmp_path):
        p = tmp_path / "no_sidecar.pkl"
        serialize({"a": 1}, fname=str(p))
        os.remove(str(p) + ".sha256")
        with pytest.raises(PickleVerificationError):
            unserialize(str(p))

    def test_verify_sidecar_false_loads_a_legacy_file_without_a_sidecar(self, tmp_path):
        p = tmp_path / "legacy.pkl"
        serialize({"a": 1}, fname=str(p))
        os.remove(str(p) + ".sha256")
        assert unserialize(str(p), verify_sidecar=False) == {"a": 1}

    def test_verify_sidecar_true_with_matching_sidecar_loads(self, tmp_path):
        from pyutilz.core.safe_pickle import write_sidecar

        p = tmp_path / "with_sidecar.pkl"
        serialize({"a": 1}, fname=str(p))
        write_sidecar(str(p))
        assert unserialize(str(p), verify_sidecar=True) == {"a": 1}

    def test_verify_sidecar_true_with_stale_sidecar_raises(self, tmp_path):
        p = tmp_path / "stale_sidecar.pkl"
        serialize({"a": 1}, fname=str(p))
        # Overwrite the payload behind serialize()'s back, so the sidecar it wrote no longer matches the bytes.
        p.write_bytes(b"not the bytes the sidecar was written for")
        with pytest.raises(PickleVerificationError):
            unserialize(str(p), verify_sidecar=True)

    def test_verify_sidecar_has_no_effect_on_in_memory_bytes(self):
        data = {"a": 1}
        raw = serialize(data)
        # No file path involved -- verify_sidecar is a no-op here, not an error.
        assert unserialize(raw, verify_sidecar=True) == data


import io as _io
import logging as _logging
import sys as _sys
import zlib as _zlib
from unittest.mock import patch as _patch

from pyutilz.core import serialization as ser_mod
from pyutilz.core.serialization import atomic_write_bytes


class TestSecondRoundSerialization:
    def test_str_to_class_passes_the_parenthesised_argument_through(self):
        assert str_to_class("builtins.str(hi)") == "(hi)"

    def test_str_to_class_without_arguments_builds_with_an_empty_one(self):
        assert str_to_class("builtins.str") == ""

    def test_str_to_class_without_a_module_looks_in_this_module(self):
        assert isinstance(str_to_class("PickleVerificationError"), PickleVerificationError)

    def test_serialize_refuses_a_non_path_non_writable_target(self):
        with pytest.raises(TypeError, match=r"^Unsupported fname type for serialize: <class 'int'>$"):
            serialize({"a": 1}, fname=3)

    def test_serialize_refuses_a_non_integer_compression(self):
        with pytest.raises(TypeError, match=r"^compression must be an int, got float$"):
            serialize({"a": 1}, compression=1.5)

    @pytest.mark.parametrize("level", [-1, 0, 9])
    def test_the_edge_compression_levels_are_accepted(self, level):
        assert unserialize(serialize({"a": 1}, compression=level), compression=level) == {"a": 1}

    @pytest.mark.parametrize("level", [-2, 10])
    def test_compression_outside_zlib_levels_is_refused_with_the_range(self, level):
        with pytest.raises(ValueError, match=r"^compression must be in -1\.\.9 \(zlib's accepted levels\), got " + str(level) + "$"):
            serialize({"a": 1}, compression=level)

    def test_the_default_output_is_zlib_level_nine_of_the_pickle(self):
        data = {"a": 1}
        assert serialize(data) == _zlib.compress(pickle.dumps(data), 9)

    def test_no_compression_stores_the_raw_pickle(self):
        data = {"a": 1}
        assert serialize(data, compression=None) == pickle.dumps(data)

    def test_writing_to_a_path_creates_its_directories_and_a_sidecar(self, tmp_path):
        target = tmp_path / "a" / "b" / "obj.pkl"
        assert serialize({"a": 1}, str(target)) is True
        assert target.exists()
        assert (tmp_path / "a" / "b" / "obj.pkl.sha256").exists()
        assert unserialize(str(target)) == {"a": 1}

    def test_writing_to_a_file_object_writes_the_compressed_pickle(self):
        buf = _io.BytesIO()
        assert serialize({"a": 2}, buf) is True
        assert buf.getvalue() == _zlib.compress(pickle.dumps({"a": 2}), 9)

    def test_a_failed_serialisation_is_logged_and_reraised(self, caplog):
        with caplog.at_level(_logging.ERROR, logger=ser_mod.logger.name), pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            serialize(lambda: None)
        assert any(r.getMessage().startswith("serialize failed (fname=None, type=function): ") for r in caplog.records)

    def test_a_missing_file_raises_with_its_errno_and_is_logged(self, tmp_path, caplog):
        import errno

        missing = str(tmp_path / "nope.pkl")
        with caplog.at_level(_logging.ERROR, logger=ser_mod.logger.name), pytest.raises(FileNotFoundError) as info:
            unserialize(missing)
        assert info.value.errno == errno.ENOENT
        assert info.value.filename == missing
        assert f"File {missing} not found" in [r.getMessage() for r in caplog.records]

    def test_a_file_without_its_sidecar_is_refused_with_the_fixes_named(self, tmp_path):
        path = tmp_path / "bare.pkl"
        path.write_bytes(_zlib.compress(pickle.dumps({"a": 1}), 9))
        with pytest.raises(PickleVerificationError) as info:
            unserialize(str(path))
        message = str(info.value)
        assert message.startswith("unserialize: refusing to unpickle ")
        assert "sha256 sidecar missing or mismatch." in message
        assert "write_sidecar(path)" in message
        assert "verify_sidecar=False" in message

    def test_a_file_without_its_sidecar_loads_when_verification_is_off(self, tmp_path):
        path = tmp_path / "legacy.pkl"
        path.write_bytes(_zlib.compress(pickle.dumps({"a": 1}), 9))
        assert unserialize(str(path), verify_sidecar=False) == {"a": 1}

    def test_a_non_path_non_bytes_input_is_refused_with_its_type(self):
        with pytest.raises(TypeError, match=r"^unserialize: unexpected input data type <class 'int'>; expected str path, bytes or a file-like object$"):
            unserialize(7)

    def test_a_file_like_input_is_read(self):
        raw = serialize({"a": 3})
        assert unserialize(_io.BytesIO(raw)) == {"a": 3}

    def test_a_zlib_bomb_is_refused_past_the_cap_with_the_bound_named(self):
        bomb = _zlib.compress(b"\0" * (1024 * 1024), 9)
        with pytest.raises(ValueError) as info:
            unserialize(bomb, compression=9, max_output_bytes=1000)
        assert str(info.value) == (
            "unserialize: decompressed payload exceeds max_output_bytes=1000; "
            "refusing to continue (pass max_output_bytes=None to accept an unbounded expansion)"
        )

    def test_a_payload_within_the_cap_is_returned_exactly(self):
        data = {"a": "x" * 500}
        assert unserialize(serialize(data), max_output_bytes=10_000) == data

    def test_no_cap_accepts_any_expansion(self):
        bomb = _zlib.compress(b"\0" * (1024 * 1024), 9)
        assert len(_zlib.decompress(bomb)) == 1024 * 1024
        assert unserialize(serialize({"a": 1}), max_output_bytes=None) == {"a": 1}

    def test_the_default_cap_is_four_gibibytes_and_the_chunk_is_sixteen_mebibytes(self):
        assert ser_mod.DEFAULT_MAX_DECOMPRESSED_BYTES == 4 * 1024**3
        assert ser_mod._DECOMPRESS_CHUNK_BYTES == 16 * 1024 * 1024

    def test_uncompressed_pickle_is_read_with_a_warning(self, caplog):
        raw = pickle.dumps({"a": 4})
        with caplog.at_level(_logging.WARNING, logger=ser_mod.logger.name):
            assert unserialize(raw, compression=9) == {"a": 4}
        assert "Data seems to be not compressed; reading as raw pickle" in [r.getMessage() for r in caplog.records]

    def test_garbage_bytes_raise_and_are_logged(self, caplog):
        with caplog.at_level(_logging.ERROR, logger=ser_mod.logger.name), pytest.raises(pickle.UnpicklingError):
            unserialize(b"not a pickle at all", compression=None)
        assert caplog.records

    @pytest.mark.parametrize("level", [-2, 10])
    def test_unserialize_refuses_compression_outside_the_zlib_levels(self, level):
        with pytest.raises(ValueError, match=r"^compression must be in -1\.\.9"):
            unserialize(b"", compression=level)

    def test_unserialize_refuses_a_non_integer_compression(self):
        with pytest.raises(TypeError, match=r"^compression must be an int, got str$"):
            unserialize(b"", compression="9")


class TestAtomicWrite:
    def test_the_writer_gets_the_file_and_the_bytes_land_at_the_target(self, tmp_path):
        target = tmp_path / "out.bin"
        atomic_write_bytes(str(target), lambda f: f.write(b"payload"))
        assert target.read_bytes() == b"payload"

    def test_a_relative_target_in_the_working_directory_is_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        atomic_write_bytes("rel.bin", lambda f: f.write(b"x"))
        assert (tmp_path / "rel.bin").read_bytes() == b"x"

    def test_an_existing_target_is_replaced_whole(self, tmp_path):
        target = tmp_path / "out.bin"
        target.write_bytes(b"old contents that are longer")
        atomic_write_bytes(str(target), lambda f: f.write(b"new"))
        assert target.read_bytes() == b"new"

    @pytest.mark.skipif(_sys.platform == "win32", reason="POSIX permission bits")
    def test_the_target_is_private_to_its_owner(self, tmp_path):
        target = tmp_path / "private.bin"
        atomic_write_bytes(str(target), lambda f: f.write(b"x"))
        assert target.stat().st_mode & 0o777 == 0o600

    def test_a_failed_writer_leaves_no_target_and_no_temp_file(self, tmp_path):
        target = tmp_path / "out.bin"

        def boom(f):
            f.write(b"partial")
            raise RuntimeError("writer failed")

        with pytest.raises(RuntimeError, match="writer failed"):
            atomic_write_bytes(str(target), boom)
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_fsync_is_requested_only_when_asked(self, tmp_path):
        with _patch("pyutilz.core.serialization.os.fsync") as fsync:
            atomic_write_bytes(str(tmp_path / "a.bin"), lambda f: f.write(b"x"))
            fsync.assert_not_called()
            atomic_write_bytes(str(tmp_path / "b.bin"), lambda f: f.write(b"x"), fsync=True)
            fsync.assert_called_once()

    def test_a_replace_failure_removes_the_temp_file(self, tmp_path):
        target = tmp_path / "out.bin"
        with _patch("pyutilz.core.serialization.os.replace", side_effect=OSError("no rename")), pytest.raises(OSError, match="no rename"):
            atomic_write_bytes(str(target), lambda f: f.write(b"x"))
        assert list(tmp_path.iterdir()) == []

    def test_the_counter_gives_each_write_its_own_temp_name(self):
        assert ser_mod._atomic_write_counter() != ser_mod._atomic_write_counter()

    def test_a_failed_descriptor_adoption_closes_the_raw_descriptor(self, tmp_path):
        with _patch("pyutilz.core.serialization.os.fdopen", side_effect=RuntimeError("no fdopen")), _patch(
            "pyutilz.core.serialization.os.close"
        ) as close, pytest.raises(RuntimeError, match="no fdopen"):
            atomic_write_bytes(str(tmp_path / "x.bin"), lambda f: None)
        close.assert_called_once()


class TestSerializationBoundaries:
    def test_a_leading_parenthesis_is_not_taken_as_an_argument_marker(self):
        with pytest.raises(AttributeError, match="has no attribute ''"):
            str_to_class("(7)")

    def test_text_after_the_closing_parenthesis_is_not_passed_to_the_constructor(self):
        assert str_to_class("builtins.str(hi)xyz") == "(hi)"

    def test_a_dotted_module_path_is_joined_with_dots(self):
        import types

        module = types.ModuleType("fakepkg.sub")
        module.Thing = lambda arg: ("made", arg)
        _sys.modules["fakepkg.sub"] = module
        try:
            assert str_to_class("fakepkg.sub.Thing") == ("made", "")
        finally:
            _sys.modules.pop("fakepkg.sub", None)

    def test_a_decompressed_total_is_counted_across_chunks(self):
        bomb = _zlib.compress(b"\0" * (40 * 1024 * 1024), 9)
        with pytest.raises(ValueError, match="exceeds max_output_bytes=" + str(20 * 1024 * 1024)):
            unserialize(bomb, compression=9, max_output_bytes=20 * 1024 * 1024)

    def test_exactly_the_cap_is_accepted(self):
        payload = b"\0" * 1000
        assert unserialize(_zlib.compress(pickle.dumps(payload), 9), compression=9, max_output_bytes=len(pickle.dumps(payload))) == payload
