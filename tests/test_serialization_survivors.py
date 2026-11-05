"""Atomic-write and capped-decompression pins for core.serialization (mutation survivors)."""

from __future__ import annotations

import os
import stat
import zlib

import pytest

from pyutilz.core import serialization as ser


class TestAtomicWriteSignature:
    def test_target_and_writer_are_accepted_by_keyword(self, tmp_path):
        target = str(tmp_path / "out.bin")
        ser.atomic_write_bytes(target_path=target, writer_fn=lambda f: f.write(b"abc"))
        assert open(target, "rb").read() == b"abc"

    def test_fsync_is_keyword_only(self, tmp_path):
        target = str(tmp_path / "out.bin")
        with pytest.raises(TypeError):
            ser.atomic_write_bytes(target, lambda f: f.write(b"abc"), True)


class TestAtomicWriteTempLocation:
    def test_the_temporary_file_is_created_beside_the_target(self, tmp_path):
        target_dir = tmp_path / "nested"
        target_dir.mkdir()
        target = str(target_dir / "out.bin")
        seen: list = []

        def writer(f):
            seen.extend(name for name in os.listdir(target_dir) if ".tmp." in name)
            f.write(b"abc")

        ser.atomic_write_bytes(target, writer)
        assert len(seen) == 1


@pytest.mark.skipif(os.name != "posix", reason="file mode bits are only enforced on POSIX")
def test_the_written_file_is_readable_only_by_its_owner(tmp_path):
    target = str(tmp_path / "out.bin")
    ser.atomic_write_bytes(target, lambda f: f.write(b"abc"))
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


class TestCappedDecompression:
    def test_an_output_larger_than_one_chunk_is_returned_in_full(self):
        payload = b"a" * (20 * 1024 * 1024)
        assert ser._decompress_capped(zlib.compress(payload), 64 * 1024 * 1024) == payload

    @pytest.mark.timeout(30)
    def test_a_truncated_stream_returns_what_it_can_instead_of_looping(self):
        full = zlib.compress(b"x" * (2 * 1024 * 1024))
        out = ser._decompress_capped(full[: len(full) // 2], 64 * 1024 * 1024)
        assert isinstance(out, bytes)
        assert len(out) < 2 * 1024 * 1024
