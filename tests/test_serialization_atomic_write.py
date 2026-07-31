"""Tests for pyutilz.core.serialization.atomic_write_bytes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyutilz.core.serialization import atomic_write_bytes


def test_writes_content(tmp_path: Path):
    target = tmp_path / "out.bin"
    atomic_write_bytes(str(target), lambda f: f.write(b"hello"))
    assert target.read_bytes() == b"hello"


def test_overwrites_existing_file(tmp_path: Path):
    target = tmp_path / "out.bin"
    target.write_bytes(b"old")
    atomic_write_bytes(str(target), lambda f: f.write(b"new"))
    assert target.read_bytes() == b"new"


def test_no_tmp_file_left_on_success(tmp_path: Path):
    target = tmp_path / "out.bin"
    atomic_write_bytes(str(target), lambda f: f.write(b"x"))
    leftovers = [p for p in tmp_path.iterdir() if p.name != "out.bin"]
    assert leftovers == []


def test_cleans_up_tmp_file_on_writer_exception(tmp_path: Path):
    target = tmp_path / "out.bin"

    def _boom(f):
        f.write(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        atomic_write_bytes(str(target), _boom)

    assert not target.exists()
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [], f"tmp file(s) leaked: {leftovers}"


def test_pre_existing_file_untouched_on_writer_exception(tmp_path: Path):
    """A crash mid-write must never truncate the pre-existing target file."""
    target = tmp_path / "out.bin"
    target.write_bytes(b"original")

    def _boom(f):
        f.write(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError):
        atomic_write_bytes(str(target), _boom)

    assert target.read_bytes() == b"original"


def test_fsync_true_still_writes_correctly(tmp_path: Path):
    target = tmp_path / "out.bin"
    atomic_write_bytes(str(target), lambda f: f.write(b"durable"), fsync=True)
    assert target.read_bytes() == b"durable"


def test_concurrent_writes_never_produce_truncated_file(tmp_path: Path):
    """Concurrent writers race (last writer wins, and a losing os.replace() on Windows may raise
    PermissionError under heavy contention -- that's an acceptable racing-writer outcome), but a
    reader must never see a torn/partial file: whatever ends up on disk is one full payload."""
    import threading

    target = tmp_path / "shared.bin"
    payloads = [b"A" * 10_000, b"B" * 10_000, b"C" * 10_000, b"D" * 10_000]
    errors = []

    def worker(payload):
        for _ in range(5):
            try:
                atomic_write_bytes(str(target), lambda f, p=payload: f.write(p))
            except OSError:
                pass  # a losing writer under contention; not a correctness violation
            except Exception as exc:  # pragma: no cover - test fails if hit
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(p,)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    content = target.read_bytes()
    assert len(content) == 10_000, "file must be exactly one full payload, never a torn mix"
    assert content == content[:1] * 10_000, "content must be a single uniform payload, not interleaved"
