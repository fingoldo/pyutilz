"""Behaviour of ``pyutilz.performance.kernel_tuning.cache.cache_persistence``: an atomic JSON write leaves a whole file
or none and reports a persistent failure without raising, the recency key orders records by their tuning time and breaks
ties by mtime, the readers skip records of another schema or hardware, and garbage collection keeps the logically newest
records even when their files are older. Everything runs under ``tmp_path``."""

from __future__ import annotations

import os

import pytest

from pyutilz.core import jsonlib
from pyutilz.performance.kernel_tuning.cache import cache_base
from pyutilz.performance.kernel_tuning.cache import cache_persistence as cache_persistence_module
from pyutilz.performance.kernel_tuning.cache.cache_persistence import _CachePersistenceMixin


class _Cache(_CachePersistenceMixin):
    def __init__(self):
        self._in_memory = True
        self._path = None


def _record(kernel, tuned_utc, *, schema=cache_base.SCHEMA_VERSION, hw=None):
    return {
        "schema_version": schema,
        "hw_fingerprint": hw if hw is not None else cache_base.hw_fingerprint(),
        "kernel_name": kernel,
        "entry": {"tuned_utc": tuned_utc, "regions": []},
    }


def _write(path, payload, mtime=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(jsonlib.dumps(payload))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return str(path)


class TestAtomicWrite:
    def test_the_payload_is_written_whole_and_no_temp_file_is_left(self, tmp_path):
        target = tmp_path / "sub" / "k.json"
        assert _CachePersistenceMixin._atomic_write_json(str(target), {"b": 2, "a": 1}) is True
        assert jsonlib.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
        assert [p.name for p in target.parent.iterdir()] == ["k.json"]

    def test_a_persistent_failure_returns_false_and_leaves_no_temp_file(self, tmp_path, monkeypatch):
        def _always_fails(src, dst):
            raise OSError(5, "share violation")

        monkeypatch.setattr(os, "replace", _always_fails)
        target = tmp_path / "k.json"
        assert _CachePersistenceMixin._atomic_write_json(str(target), {"a": 1}, retries=2, backoff=0.0) is False
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []


class TestRecencyKey:
    def test_the_tuning_time_decides_before_the_mtime(self, tmp_path):
        older_written_later = _write(tmp_path / "a.json", {}, mtime=2_000_000)
        newer_written_earlier = _write(tmp_path / "b.json", {}, mtime=1_000_000)
        cache = _Cache()
        assert cache._record_sort_key({"tuned_utc": "2026-01-02"}, newer_written_earlier) > cache._record_sort_key({"tuned_utc": "2026-01-01"}, older_written_later)

    def test_the_mtime_breaks_a_tie_in_tuning_time(self, tmp_path):
        early = _write(tmp_path / "a.json", {}, mtime=1_000_000)
        late = _write(tmp_path / "b.json", {}, mtime=2_000_000)
        cache = _Cache()
        assert cache._record_sort_key({"tuned_utc": "2026-01-01"}, late) > cache._record_sort_key({"tuned_utc": "2026-01-01"}, early)

    def test_an_unreadable_record_sorts_oldest(self, tmp_path):
        cache = _Cache()
        good = _write(tmp_path / "good.json", {}, mtime=1)
        assert cache._record_sort_key(None, str(tmp_path / "missing.json"))[0] == ""
        assert cache._record_sort_key({"tuned_utc": "2026-01-01"}, good)[0] > ""


class TestReadKernelDir:
    def test_the_newest_record_per_kernel_name_is_returned(self, tmp_path):
        _write(tmp_path / "old.json", _record("matmul", "2026-01-01"))
        _write(tmp_path / "new.json", _record("matmul", "2026-02-01"))
        found = _Cache()._read_kernel_dir_all(str(tmp_path))
        assert found["matmul"][1]["tuned_utc"] == "2026-02-01"

    def test_records_of_another_schema_or_hardware_are_skipped(self, tmp_path):
        _write(tmp_path / "schema.json", _record("matmul", "2026-03-01", schema=cache_base.SCHEMA_VERSION - 1))
        _write(tmp_path / "hw.json", _record("matmul", "2026-03-01", hw="some-other-machine"))
        assert _Cache()._read_kernel_dir_all(str(tmp_path)) == {}

    def test_a_corrupt_file_is_skipped_without_raising(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        _write(tmp_path / "good.json", _record("matmul", "2026-01-01"))
        assert set(_Cache()._read_kernel_dir_all(str(tmp_path))) == {"matmul"}

    def test_an_empty_directory_gives_nothing(self, tmp_path):
        assert _Cache()._read_kernel_dir_all(str(tmp_path)) == {}


class TestGarbageCollection:
    def test_the_newest_tuning_time_is_kept_even_when_its_file_is_older(self, tmp_path):
        newest_tuned_oldest_file = _write(tmp_path / "a.json", _record("matmul", "2026-03-01"), mtime=1_000_000)
        middle = _write(tmp_path / "b.json", _record("matmul", "2026-02-01"), mtime=2_000_000)
        oldest_tuned_newest_file = _write(tmp_path / "c.json", _record("matmul", "2026-01-01"), mtime=3_000_000)
        _Cache()._gc_kernel_dir(str(tmp_path), keep=2)
        assert os.path.exists(newest_tuned_oldest_file)
        assert os.path.exists(middle)
        assert not os.path.exists(oldest_tuned_newest_file)

    def test_nothing_is_removed_while_within_the_keep_count(self, tmp_path):
        paths = [_write(tmp_path / f"{i}.json", _record("matmul", f"2026-0{i}-01")) for i in range(1, 4)]
        _Cache()._gc_kernel_dir(str(tmp_path), keep=3)
        assert paths and all(os.path.exists(p) for p in paths)

    @pytest.mark.parametrize("keep", [1, 2])
    def test_exactly_the_keep_count_survives(self, tmp_path, keep):
        for i in range(1, 5):
            _write(tmp_path / f"{i}.json", _record("matmul", f"2026-0{i}-01"))
        _Cache()._gc_kernel_dir(str(tmp_path), keep=keep)
        assert len(list(tmp_path.glob("*.json"))) == keep


class TestGarbageCollectionDefaults:
    def test_the_default_keep_count_is_four(self, tmp_path):
        for i in range(1, 7):
            _write(tmp_path / f"{i}.json", _record("matmul", f"2026-0{i}-01"))
        _Cache()._gc_kernel_dir(str(tmp_path))
        assert len(list(tmp_path.glob("*.json"))) == 4

    def test_an_unlistable_directory_is_logged_and_skipped(self, tmp_path, monkeypatch, caplog):
        def _fails(pattern):
            raise PermissionError(13, "denied")

        monkeypatch.setattr(cache_persistence_module._glob, "glob", _fails)
        with caplog.at_level("DEBUG"):
            _Cache()._gc_kernel_dir(str(tmp_path), keep=1)
        assert any(rec.getMessage().endswith("GC skipped: PermissionError: [Errno 13] denied") for rec in caplog.records)
