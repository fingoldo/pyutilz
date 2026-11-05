"""Capped-decompression invariant for core.serialization (mutation survivors 158:23 and 158:48)."""

from __future__ import annotations

import os
import zlib

import pytest

from pyutilz.core import serialization as ser


@pytest.mark.parametrize("chunk", [1, 3, 7, 64])
def test_small_output_blocks_round_trip_the_whole_payload(monkeypatch, chunk):
    monkeypatch.setattr(ser, "_DECOMPRESS_CHUNK_BYTES", chunk)
    payload = b"hello world " * 300 + os.urandom(200)
    assert ser._decompress_capped(zlib.compress(payload), 64 * 1024 * 1024) == payload
