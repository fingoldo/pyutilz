"""Tests for pyutilz.data.git_checkpoint_cache."""

from __future__ import annotations

import gzip
import json

from pyutilz.data.git_checkpoint_cache import (
    read_bytes_with_checkpoint_fallback,
    read_json_with_checkpoint_fallback,
    write_bytes_checkpoint,
    write_json_checkpoint,
)


def test_json_checkpoint_round_trips(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json.gz"
    obj = {"a": 1, "b": {"c": [1, 2, 3]}}
    write_json_checkpoint(obj, checkpoint_path)

    restored = json.loads(gzip.decompress(checkpoint_path.read_bytes()).decode("utf-8"))
    assert restored == obj


def test_json_checkpoint_is_deterministic_sort_keys():
    # A dict written in two different key orders must produce byte-identical checkpoints - callers
    # rely on this for `git diff` to show only real content changes, never key-order churn.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        p1 = Path(d) / "a.json.gz"
        p2 = Path(d) / "b.json.gz"
        write_json_checkpoint({"z": 1, "a": 2}, p1)
        write_json_checkpoint({"a": 2, "z": 1}, p2)
        assert p1.read_bytes() == p2.read_bytes()


def test_read_json_prefers_live_cache_when_present(tmp_path):
    cache_path = tmp_path / "cache.json"
    checkpoint_path = tmp_path / "checkpoint.json.gz"
    cache_path.write_text(json.dumps({"live": True}), encoding="utf-8")
    write_json_checkpoint({"live": False}, checkpoint_path)

    assert read_json_with_checkpoint_fallback(cache_path, checkpoint_path) == {"live": True}


def test_read_json_restores_from_checkpoint_when_cache_missing(tmp_path):
    cache_path = tmp_path / "missing" / "cache.json"
    checkpoint_path = tmp_path / "checkpoint.json.gz"
    write_json_checkpoint({"restored": True}, checkpoint_path)

    result = read_json_with_checkpoint_fallback(cache_path, checkpoint_path)
    assert result == {"restored": True}
    # Restore also writes the live cache back out, so a second read doesn't need the checkpoint.
    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"restored": True}


def test_read_json_restores_from_checkpoint_when_cache_is_empty(tmp_path):
    cache_path = tmp_path / "cache.json"
    checkpoint_path = tmp_path / "checkpoint.json.gz"
    cache_path.write_text("", encoding="utf-8")
    write_json_checkpoint({"restored": True}, checkpoint_path)

    assert read_json_with_checkpoint_fallback(cache_path, checkpoint_path) == {"restored": True}


def test_read_json_returns_none_when_neither_exists(tmp_path):
    cache_path = tmp_path / "cache.json"
    checkpoint_path = tmp_path / "checkpoint.json.gz"
    assert read_json_with_checkpoint_fallback(cache_path, checkpoint_path) is None


def test_bytes_checkpoint_round_trips(tmp_path):
    cache_path = tmp_path / "ledger.jsonl"
    checkpoint_path = tmp_path / "ledger_checkpoint.jsonl.gz"
    cache_path.write_bytes(b'{"a": 1}\n{"b": 2}\n')

    write_bytes_checkpoint(cache_path, checkpoint_path)
    assert gzip.decompress(checkpoint_path.read_bytes()) == b'{"a": 1}\n{"b": 2}\n'


def test_bytes_checkpoint_is_a_noop_when_cache_path_does_not_exist(tmp_path):
    cache_path = tmp_path / "missing.jsonl"
    checkpoint_path = tmp_path / "checkpoint.jsonl.gz"
    write_bytes_checkpoint(cache_path, checkpoint_path)
    assert not checkpoint_path.exists()


def test_read_bytes_restores_from_checkpoint_when_cache_missing(tmp_path):
    cache_path = tmp_path / "missing" / "ledger.jsonl"
    checkpoint_path = tmp_path / "checkpoint.jsonl.gz"
    _write_gz(checkpoint_path, b'{"row": 1}\n')

    restored = read_bytes_with_checkpoint_fallback(cache_path, checkpoint_path)
    assert restored is True
    assert cache_path.read_bytes() == b'{"row": 1}\n'


def test_read_bytes_does_nothing_when_live_cache_already_has_content(tmp_path):
    cache_path = tmp_path / "ledger.jsonl"
    checkpoint_path = tmp_path / "checkpoint.jsonl.gz"
    cache_path.write_bytes(b'{"row": "live"}\n')
    _write_gz(checkpoint_path, b'{"row": "stale"}\n')

    restored = read_bytes_with_checkpoint_fallback(cache_path, checkpoint_path)
    assert restored is False
    assert cache_path.read_bytes() == b'{"row": "live"}\n'


def test_read_bytes_returns_false_when_neither_exists(tmp_path):
    cache_path = tmp_path / "missing.jsonl"
    checkpoint_path = tmp_path / "checkpoint.jsonl.gz"
    assert read_bytes_with_checkpoint_fallback(cache_path, checkpoint_path) is False


def _write_gz(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload))
