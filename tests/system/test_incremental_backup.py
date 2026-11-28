"""Incremental backup: only changes are stored, nothing is overwritten or deleted, and a restore is byte-identical or refused."""

from __future__ import annotations

import gzip
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pyutilz.system import incremental_backup as ib


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """A directory with two small files."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.jsonl").write_bytes(b"one\n")
    (root / "sub").mkdir()
    (root / "sub" / "b.jsonl").write_bytes(b"two\n")
    return root


def _run(targets: List[ib.BackupTarget], dest: Path, tmp_path: Path, archive_bytes: int = ib.ARCHIVE_BYTES) -> int:
    """One folder backup run the way a CLI wires it: set aside one-file records, find changes, archive them."""
    manifest_path = dest / ib.FOLDER_MANIFEST
    manifest = ib.set_aside_one_file_objects(dest, ib.load_manifest(manifest_path))
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    changes = ib.find_changes(targets, manifest, staging)
    return ib.copy_to_folder(dest, changes, manifest, manifest_path, ib.utc_day(), archive_bytes=archive_bytes)


def _archives(dest: Path) -> List[Path]:
    """The archives under the destination, sorted."""
    return sorted((dest / ib.FOLDER_ARCHIVE_DIR).glob("*.tar.gz"))


def test_first_run_archives_everything_second_nothing_a_change_one_new_archive(data: Path, tmp_path: Path) -> None:
    targets = [ib.BackupTarget("d", data)]
    dest = tmp_path / "share"
    assert _run(targets, dest, tmp_path) == 2
    first = _archives(dest)
    assert len(first) == 1
    assert _run(targets, dest, tmp_path) == 0
    assert _archives(dest) == first
    (data / "a.jsonl").write_bytes(b"one\nmore\n")
    assert _run(targets, dest, tmp_path) == 1
    after = _archives(dest)
    assert len(after) == 2 and set(first) <= set(after)
    assert not list(dest.rglob("*.partial"))
    out = tmp_path / "restored"
    assert ib.restore_from_folder(dest, out) == 2
    assert (out / "d" / "a.jsonl").read_bytes() == b"one\nmore\n"
    assert (out / "d" / "sub" / "b.jsonl").read_bytes() == b"two\n"


def test_archives_split_by_raw_size(data: Path, tmp_path: Path) -> None:
    dest = tmp_path / "share"
    assert _run([ib.BackupTarget("d", data)], dest, tmp_path, archive_bytes=1) == 2
    assert len(_archives(dest)) == 2


def test_restore_refuses_bytes_that_do_not_match_the_manifest(data: Path, tmp_path: Path) -> None:
    dest = tmp_path / "share"
    _run([ib.BackupTarget("d", data)], dest, tmp_path)
    manifest = ib.load_manifest(dest / ib.FOLDER_MANIFEST)
    manifest["d/a.jsonl"]["sha256"] = "0" * 64
    ib.save_manifest(dest / ib.FOLDER_MANIFEST, manifest)
    with pytest.raises(ValueError, match=r"d/a.jsonl"):
        ib.restore_from_folder(dest, tmp_path / "restored")


def test_restore_refuses_an_archive_missing_a_named_member(data: Path, tmp_path: Path) -> None:
    dest = tmp_path / "share"
    _run([ib.BackupTarget("d", data)], dest, tmp_path)
    manifest = ib.load_manifest(dest / ib.FOLDER_MANIFEST)
    manifest["d/ghost.jsonl"] = dict(manifest["d/a.jsonl"], member="d/ghost.jsonl")
    ib.save_manifest(dest / ib.FOLDER_MANIFEST, manifest)
    with pytest.raises(ValueError, match="lacks 1 member"):
        ib.restore_from_folder(dest, tmp_path / "restored")


def test_one_file_objects_restore_and_are_archived_again_with_their_records_kept(data: Path, tmp_path: Path) -> None:
    dest = tmp_path / "share"
    key = "prefix/d/a.jsonl/20261127-0123456789abcdef"
    (dest / key).parent.mkdir(parents=True)
    with gzip.open(dest / f"{key}.gz", "wb") as out:
        out.write(b"one\n")
    old = {"sha256": ib.file_sha256(data / "a.jsonl"), "size": 4, "object_key": key}
    ib.save_manifest(dest / ib.FOLDER_MANIFEST, {"d/a.jsonl": old})
    assert ib.restore_from_folder(dest, tmp_path / "legacy") == 1
    assert (tmp_path / "legacy" / "d" / "a.jsonl").read_bytes() == b"one\n"
    assert _run([ib.BackupTarget("d", data)], dest, tmp_path) == 2
    manifest = ib.load_manifest(dest / ib.FOLDER_MANIFEST)
    assert sorted(manifest) == ["d/a.jsonl", "d/sub/b.jsonl"]
    assert all("archive" in e for e in manifest.values())
    assert ib.load_manifest(dest / ib.FOLDER_LEGACY_MANIFEST) == {"d/a.jsonl": old}
    assert (dest / f"{key}.gz").exists()
    assert ib.archived_entries({"x": old, **manifest}) == manifest


def test_manifest_keeps_every_archived_file_when_a_later_archive_fails(data: Path, tmp_path: Path) -> None:
    good = ib.Candidate("d/a.jsonl", data / "a.jsonl", ib.file_sha256(data / "a.jsonl"), 4)
    missing = ib.Candidate("d/gone.jsonl", data / "gone.jsonl", "0" * 64, 1)
    dest = tmp_path / "share"
    manifest: ib.Manifest = {}
    with pytest.raises(FileNotFoundError):
        ib.copy_to_folder(dest, [good, missing], manifest, dest / ib.FOLDER_MANIFEST, "20261127", archive_bytes=1)
    saved = ib.load_manifest(dest / ib.FOLDER_MANIFEST)
    assert list(saved) == ["d/a.jsonl"]
    assert (dest / saved["d/a.jsonl"]["archive"]).exists()


def test_find_changes_snapshots_sqlite_honours_a_file_selector_and_skips_unreadable(data: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    db = data / "store.sqlite3"
    con = sqlite3.connect(str(db))
    con.execute("create table t (x)")
    con.execute("insert into t values (1)")
    con.commit()
    staging = tmp_path / "staging"
    staging.mkdir()
    only_db = ib.BackupTarget("d", data, files=lambda root: [root / "store.sqlite3"])
    try:
        changes = ib.find_changes([only_db], {}, staging)
    finally:
        con.close()
    assert [c.rel for c in changes] == ["d/store.sqlite3"]
    assert changes[0].source.parent == staging
    assert ib.find_changes([ib.BackupTarget("x", tmp_path / "absent")], {}, staging) == []
    broken = ib.BackupTarget("d", data, files=lambda root: [root / "nope.jsonl", root / "gone.jsonl"])
    with caplog.at_level("WARNING", logger=ib.__name__):
        assert ib.find_changes([broken], {}, staging) == []
    assert [r.getMessage().split(",")[0] for r in caplog.records] == ["skipped 2 unreadable file(s) this run"]


def test_replace_retries_a_briefly_locked_destination_then_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src, dst = tmp_path / "a.tmp", tmp_path / "b.json"
    src.write_bytes(b"new")
    dst.write_bytes(b"old")
    real_replace = ib.os.replace
    denials = iter([True, True, False])
    waits: List[float] = []

    def flaky(a: Any, b: Any) -> None:
        if next(denials):
            raise PermissionError(5, "Access is denied")
        real_replace(a, b)

    monkeypatch.setattr(ib.os, "replace", flaky)
    monkeypatch.setattr(ib.time, "sleep", waits.append)
    ib.replace_with_retry(src, dst)
    assert dst.read_bytes() == b"new"
    assert waits == [0.25, 0.5]


def test_replace_gives_up_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def denied(_a: Any, _b: Any) -> None:
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(ib.os, "replace", denied)
    monkeypatch.setattr(ib.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        ib.replace_with_retry(tmp_path / "a", tmp_path / "b", attempts=3)


class FakeS3:
    """Records every call; any method other than `upload_file` would be an error."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def upload_file(self, path: str, bucket: str, key: str, ExtraArgs: Dict[str, str]) -> None:
        """Record one upload."""
        self.calls.append({"path": path, "bucket": bucket, "key": key, "extra": ExtraArgs})


def test_s3_upload_writes_new_keys_only_and_records_them(data: Path, tmp_path: Path) -> None:
    client = FakeS3()
    manifest_path = tmp_path / "m.json"
    manifest: ib.Manifest = {}
    changes = ib.find_changes([ib.BackupTarget("d", data)], manifest, tmp_path)
    assert ib.upload_to_s3(client, "bkt", changes, manifest, manifest_path, "20261127", "pre") == 2
    digest = hashlib.sha256(b"one\n").hexdigest()
    assert client.calls[0]["key"] == f"pre/d/a.jsonl/20261127-{digest[:16]}"
    assert client.calls[0]["extra"] == {"StorageClass": ib.DEEP_ARCHIVE}
    assert ib.load_manifest(manifest_path)["d/a.jsonl"]["object_key"] == client.calls[0]["key"]
    assert ib.find_changes([ib.BackupTarget("d", data)], manifest, tmp_path) == []


def test_snapshot_tree_copies_into_a_new_stamped_folder_and_refuses_one_inside(data: Path, tmp_path: Path) -> None:
    made = ib.snapshot_tree(data, tmp_path / "backups", stamp="s1")
    assert (made / "sub" / "b.jsonl").read_bytes() == b"two\n"
    with pytest.raises(FileExistsError):
        ib.snapshot_tree(data, tmp_path / "backups", stamp="s1")
    with pytest.raises(ValueError, match="inside the data directory"):
        ib.snapshot_tree(data, data / "backups")


def test_task_script_and_daily_command(tmp_path: Path) -> None:
    body = ib.task_script(tmp_path / "repo dir", ["-m", "bench.backup", "--dest-dir", "//host/share"], tmp_path / "t.log", {"DATA": "C:/x y"}, python="py.exe")
    assert body.endswith("\r\n")
    assert f'cd /d "{tmp_path / "repo dir"}"' in body
    assert 'set "DATA=C:/x y"' in body
    assert f'"py.exe" -m bench.backup --dest-dir "//host/share" >> "{tmp_path / "t.log"}" 2>&1' in body
    cmd = ib.daily_task_command("T", '"C:/t.cmd"')
    assert cmd[-2:] == ["/TR", '"C:/t.cmd"'] and "/F" not in cmd
    with pytest.raises(ValueError, match="261"):
        ib.daily_task_command("T", "x" * 262)


def test_require_env_names_the_missing_variable_and_not_a_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IB_PRESENT", "secret-value")
    monkeypatch.delenv("IB_ABSENT", raising=False)
    with pytest.raises(ib.CredentialError) as info:
        ib.require_env(["IB_PRESENT", "IB_ABSENT"])
    assert "IB_ABSENT" in str(info.value) and "secret-value" not in str(info.value)
    assert ib.summary([ib.Candidate("a", Path("a"), "h", 2_000_000_000)]) == "1 file(s), 2000000000 bytes (2.000 GB)"
