"""Incremental, append-only backup of directories that must never be lost: to a folder (a network share) or to S3.

The pieces a project wires into its own CLI:

- `BackupTarget` names a directory and, optionally, which of its files count; `find_changes` hashes every file (SQLite files through the
  online backup API, so a live database is read consistently) and returns those whose sha256 differs from the manifest.
- `copy_to_folder` packs the changes into tar.gz archives built locally and copies each into `<dest>/archives/` in one write; a network
  share costs about half a second per file operation, so one object per file took a day for 57k small files and an archive takes minutes.
  `restore_from_folder` unpacks the latest copy of every file and refuses any whose bytes differ from the manifest.
- `upload_to_s3` uploads each change as a new object under a key carrying the date and hash (Glacier Deep Archive by default).
- `snapshot_tree` copies a directory into a new timestamped folder.
- `task_script` / `daily_task_command` register a daily Windows task through a batch file (`schtasks /TR` takes at most 261 characters).

Nothing here deletes or overwrites a stored object: archive and object names carry a date, a hash or a random suffix, and the manifest is
replaced atomically, retrying while a share holds the file just written.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SQLITE_SUFFIXES = (".sqlite3", ".sqlite", ".db")
FOLDER_MANIFEST = "_manifest.json"
FOLDER_ARCHIVE_DIR = "archives"
FOLDER_LEGACY_MANIFEST = "_manifest_one_file_objects.json"  # records of one-file gzip objects written before archives; the objects stay
ARCHIVE_BYTES = 256 * 1024 * 1024  # raw bytes per archive
REPLACE_ATTEMPTS = 8  # a network share can hold a just-written file locked for a moment; the rename is retried with doubling waits
DEEP_ARCHIVE = "DEEP_ARCHIVE"

Manifest = Dict[str, Dict[str, Any]]


class CredentialError(RuntimeError):
    """A required environment variable is missing; the message names it and never carries a value."""


@dataclass(frozen=True)
class BackupTarget:
    """One directory to back up: `label` is its stable prefix in the manifest, `path` where it lives, `why` what it holds.

    `files` picks the files that count (a callable over `path`); by default every regular file under `path`, recursively."""

    label: str
    path: Path
    why: str = ""
    files: Optional[Callable[[Path], Iterable[Path]]] = None


@dataclass(frozen=True)
class Candidate:
    """One file that is new or changed since the manifest: `rel` is its manifest key, `source` what is read, `size` its bytes."""

    rel: str
    source: Path
    sha256: str
    size: int


def require_env(names: Sequence[str]) -> None:
    """Raise `CredentialError` naming every variable in `names` that is unset or empty."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise CredentialError(f"missing environment variable(s): {', '.join(missing)}; set them in the environment or in .env")


def file_sha256(path: Path) -> str:
    """Hex sha256 of the file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def target_files(target: BackupTarget) -> Iterator[Tuple[str, Path]]:
    """`(manifest key, path)` for every file of the target, keyed `<label>/<path relative to the target>`."""
    if target.files is not None:
        for path in target.files(target.path):
            yield f"{target.label}/{path.relative_to(target.path).as_posix()}", path
        return
    if not target.path.exists():
        return
    for path in sorted(target.path.rglob("*")):  # walk-ok: a backup target is a data directory, every file of which must be kept
        if path.is_file():
            yield f"{target.label}/{path.relative_to(target.path).as_posix()}", path


def snapshot_sqlite(path: Path, out: Path) -> Path:
    """A consistent copy of a live SQLite file at `out`, taken through the online backup API; returns `out`."""
    src = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(str(out))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return out


def load_manifest(path: Path) -> Manifest:
    """The manifest `{key: {sha256, size, ...}}`; empty when none exists yet."""
    if not path.exists():
        return {}
    loaded = json.loads(path.read_bytes())
    return dict(loaded) if isinstance(loaded, dict) else {}


def replace_with_retry(src: Path, dst: Path, attempts: int = REPLACE_ATTEMPTS, first_wait: float = 0.25) -> None:
    """`os.replace`, retried when the destination is briefly locked (an SMB share right after a write denies the rename now and then)."""
    wait = first_wait
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:  # noqa: PERF203 -- per-attempt retry loop; the try/except IS the retry mechanism
            if attempt == attempts:
                raise
            logger.warning("replace %s -> %s denied (attempt %d of %d), retrying in %.2fs: %s", src, dst, attempt, attempts, wait, exc)
            time.sleep(wait)
            wait *= 2


def save_manifest(path: Path, manifest: Mapping[str, Mapping[str, Any]]) -> None:
    """Write the manifest atomically (sorted keys, temp file, then a retried replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    replace_with_retry(tmp, path)


def find_changes(targets: Sequence[BackupTarget], manifest: Mapping[str, Mapping[str, Any]], staging: Path) -> List[Candidate]:
    """Files whose sha256 differs from the manifest or that the manifest has not seen; SQLite files are snapshotted under `staging` first.

    A file locked or removed mid-run is logged and skipped; the next run picks it up."""
    found: List[Candidate] = []
    n = 0
    for target in targets:
        for rel, path in target_files(target):
            n += 1
            try:
                source = snapshot_sqlite(path, staging / f"{n}.sqlite") if path.suffix in SQLITE_SUFFIXES else path
                digest = file_sha256(source)
                size = source.stat().st_size
            except (OSError, sqlite3.Error) as exc:
                logger.warning("skipping %s this run: %s", path, exc)
                continue
            if manifest.get(rel, {}).get("sha256") != digest:
                found.append(Candidate(rel, source, digest, size))
    return found


def utc_day() -> str:
    """Today's UTC date as `YYYYMMDD`, the date part of every stored name."""
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def object_key(prefix: str, rel: str, digest: str, day: str) -> str:
    """Stable, never-reused object key: prefix, manifest key, date and the first 16 hash characters."""
    return f"{prefix}/{rel}/{day}-{digest[:16]}"


def upload_to_s3(
    client: Any,
    bucket: str,
    changes: Sequence[Candidate],
    manifest: Manifest,
    manifest_path: Path,
    day: str,
    key_prefix: str,
    storage_class: str = DEEP_ARCHIVE,
) -> int:
    """Upload each change as a new object, saving the manifest after every file; returns the number uploaded. There is no delete call."""
    done = 0
    for item in changes:
        key = object_key(key_prefix, item.rel, item.sha256, day)
        client.upload_file(str(item.source), bucket, key, ExtraArgs={"StorageClass": storage_class})
        manifest[item.rel] = {"sha256": item.sha256, "size": item.size, "object_key": key}
        save_manifest(manifest_path, manifest)
        done += 1
    return done


class _HashingReader:
    """Reads a file for `tarfile` while hashing exactly the bytes it hands over, so the manifest describes what the archive holds."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        """The next `size` bytes, added to the running hash."""
        chunk = self._handle.read(size)
        self.digest.update(chunk)
        return chunk


def _chunks(changes: Sequence[Candidate], limit: int) -> Iterator[List[Candidate]]:
    """Consecutive groups of changes, each at most `limit` raw bytes unless one file alone is larger."""
    group: List[Candidate] = []
    size = 0
    for item in changes:
        if group and size + item.size > limit:
            yield group
            group, size = [], 0
        group.append(item)
        size += item.size
    if group:
        yield group


def copy_to_folder(dest: Path, changes: Sequence[Candidate], manifest: Manifest, manifest_path: Path, day: str, archive_bytes: int = ARCHIVE_BYTES) -> int:
    """Pack the changes into tar.gz archives built locally and copied into `dest/archives/` one file each; returns the number of files.

    An existing archive is never rewritten. The manifest is saved after every archive and names only archives already in place, so an
    interrupted run is resumed by the next one. Each entry records the sha256 and size of the bytes actually archived."""
    stamp = f"{datetime.now(timezone.utc).strftime('%H%M%S%f')}-{uuid.uuid4().hex[:8]}"
    done = 0
    with tempfile.TemporaryDirectory() as tmp:
        for index, group in enumerate(_chunks(changes, archive_bytes)):
            name = f"{FOLDER_ARCHIVE_DIR}/{day}-{stamp}-{index:03d}.tar.gz"
            target = dest / name
            if target.exists():
                raise FileExistsError(f"{target} already exists; an archive is never rewritten")
            local = Path(tmp) / f"{index:03d}.tar.gz"
            entries: Manifest = {}
            with tarfile.open(local, "w:gz", compresslevel=6) as tar:
                for item in group:
                    info = tarfile.TarInfo(item.rel)
                    with item.source.open("rb") as handle:
                        info.size = os.fstat(handle.fileno()).st_size
                        reader = _HashingReader(handle)
                        tar.addfile(info, reader)  # type: ignore[arg-type]
                    entries[item.rel] = {"sha256": reader.digest.hexdigest(), "size": info.size, "archive": name, "member": item.rel}
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(target.name + ".partial")
            shutil.copyfile(local, staging)
            replace_with_retry(staging, target)
            manifest.update(entries)
            save_manifest(manifest_path, manifest)
            done += len(group)
    return done


def _write_checked(out_dir: Path, rel: str, data: bytes, entry: Mapping[str, Any]) -> None:
    """Write one restored file and refuse it when its bytes differ from the recorded sha256."""
    target = out_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if file_sha256(target) != entry["sha256"]:
        raise ValueError(f"restored {rel} does not match its recorded sha256")


def restore_from_folder(dest: Path, out_dir: Path) -> int:
    """Unpack the latest copy of every manifest entry under `out_dir/<manifest key>`, checking each against its sha256; returns the count.

    Each archive is copied off the share once and read front to back: pulling members out of a tar.gz in another order re-decompresses
    it from the start for every backward step, which over a share took two hours for a fifth of the data before the read failed.
    Entries without an archive are one-file gzip objects named `<object_key>.gz`."""
    manifest = load_manifest(dest / FOLDER_MANIFEST)
    by_archive: Dict[str, Dict[str, Tuple[str, Dict[str, Any]]]] = {}
    one_file: List[Tuple[str, Dict[str, Any]]] = []
    for rel, entry in manifest.items():
        if "archive" in entry:
            by_archive.setdefault(entry["archive"], {})[entry["member"]] = (rel, entry)
        else:
            one_file.append((rel, entry))
    restored = 0
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "archive.tar.gz"
        for name, wanted in sorted(by_archive.items()):
            shutil.copyfile(dest / name, local)
            with tarfile.open(local, "r|gz") as tar:
                for member in tar:
                    hit = wanted.pop(member.name, None)
                    if hit is None or not member.isfile():
                        continue
                    handle = tar.extractfile(member)
                    if handle is None:
                        raise ValueError(f"{name} member {member.name} cannot be read")
                    _write_checked(out_dir, hit[0], handle.read(), hit[1])
                    restored += 1
            if wanted:
                raise ValueError(f"{name} lacks {len(wanted)} member(s) the manifest names, e.g. {sorted(wanted)[:3]}")
    for rel, entry in sorted(one_file):
        with gzip.open(dest / f"{entry['object_key']}.gz", "rb") as src:
            _write_checked(out_dir, rel, src.read(), entry)
        restored += 1
    return restored


def set_aside_one_file_objects(dest: Path, manifest: Manifest) -> Manifest:
    """Move the records of one-file gzip objects into `FOLDER_LEGACY_MANIFEST` and return the rest, so those files are archived again.

    The objects themselves stay on the share; restoring thousands of them one share operation at a time costs hours, an archive seconds."""
    legacy = {rel: entry for rel, entry in manifest.items() if "archive" not in entry}
    if not legacy:
        return manifest
    path = dest / FOLDER_LEGACY_MANIFEST
    kept = load_manifest(path)
    for rel, entry in legacy.items():
        kept.setdefault(rel, entry)
    save_manifest(path, kept)
    return {rel: entry for rel, entry in manifest.items() if "archive" in entry}


def archived_entries(manifest: Mapping[str, Dict[str, Any]]) -> Manifest:
    """Only the entries stored in archives: what a dry run compares against, without moving the one-file records aside."""
    return {rel: entry for rel, entry in manifest.items() if "archive" in entry}


def snapshot_tree(source: Path, dest_root: Path, stamp: Optional[str] = None) -> Path:
    """Copy every file under `source` into a new `dest_root/<stamp>` folder and return it; refuses a destination inside `source`.

    SQLite files go through the online backup API so a copy taken while a server writes is consistent; others are copied byte for byte."""
    source, dest_root = source.resolve(), dest_root.resolve()
    if dest_root == source or source in dest_root.parents:
        raise ValueError(f"backup destination {dest_root} is inside the data directory {source}")
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") if stamp is None else stamp
    target = dest_root / name
    target.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):  # walk-ok: a snapshot copies the whole data directory
        if not path.is_file():
            continue
        out = target / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in SQLITE_SUFFIXES:
            snapshot_sqlite(path, out)
        else:
            shutil.copy2(path, out)
    return target


def summary(changes: Sequence[Candidate]) -> str:
    """`N file(s), B bytes (G GB)` for a list of changes."""
    total = sum(c.size for c in changes)
    return f"{len(changes)} file(s), {total} bytes ({total / 1e9:.3f} GB)"


def task_script(cwd: Path, argv: Sequence[str], log_path: Path, env: Optional[Mapping[str, str]] = None, python: str = sys.executable) -> str:
    """A Windows batch file that enters `cwd`, sets `env`, runs `python <argv>` and appends its output to `log_path`."""
    lines = ["@echo off", f'cd /d "{cwd}"']
    for name, value in sorted((env or {}).items()):
        lines.append(f'set "{name}={value}"')
    args = " ".join(f'"{a}"' if (" " in a or "/" in a or "\\" in a) else a for a in argv)
    lines.append(f'"{python}" {args} >> "{log_path}" 2>&1')
    return "\r\n".join(lines) + "\r\n"


def daily_task_command(task_name: str, run: str, time_of_day: str = "03:00") -> List[str]:
    """The `schtasks` argument list for a daily task; without `/F` it never overwrites an existing task. `run` must fit in 261 characters."""
    if len(run) > 261:
        raise ValueError(f"schtasks /TR takes at most 261 characters, got {len(run)}; run a batch file from task_script instead")
    return ["schtasks", "/Create", "/SC", "DAILY", "/ST", time_of_day, "/TN", task_name, "/TR", run]
