"""A git-tracked gzip backup for a machine-local cache, with auto-restore when the local copy is
missing or empty.

Ported from a downstream project's own LLM-cache durability fix (`autopsia`, 2026-08-13): a
multi-hour paid LLM classification run's verdict cache lived only under ``~/.cache`` with zero
backup anywhere, and survived a close call (a seed-directory wipe) purely by luck. The pattern this
module generalizes -- refresh a git-tracked gzip snapshot on every cache write, auto-restore from
it when the local cache is missing/empty -- was then found duplicated near-verbatim three times in
that project (a JSON-object cache, a second near-identical JSON-object cache, and a JSONL ledger)
before being pulled out here so a new cache in ANY project can opt in with two function calls
instead of re-copying the pattern a fourth time.

Two independent shapes are supported, matching the two the downstream project actually had:

- :func:`write_json_checkpoint` / :func:`read_json_with_checkpoint_fallback` for a cache that is a
  single JSON-serializable object (typically ``dict[str, Any]``) rewritten wholesale on every
  write -- the common shape for a ``{id: verdict}``-style cache.
- :func:`write_bytes_checkpoint` / :func:`read_bytes_with_checkpoint_fallback` for a cache that is
  itself a flat file on disk (a JSONL append-only ledger, a SQLite database dump) -- the checkpoint
  is a byte-for-byte gzip of whatever the live file currently holds, no re-serialization.

Both write paths use the same atomic-rename discipline as the rest of this package's file-writing
helpers (a per-process-unique temp file, then ``Path.replace``): a crash mid-write can never leave
a torn/corrupt checkpoint that silently reads back as valid-but-wrong data.

Never a substitute for a real backup strategy on genuinely irreplaceable data -- this exists for
the specific, narrower case of a LOCALLY REGENERATED cache (an LLM call result, a scraped API
response) whose loss costs re-computation time/money but not correctness, where "committed to git"
is itself the durability guarantee a project already relies on for everything else in the repo.
"""

from __future__ import annotations

import gzip
import json
import os
import uuid
from pathlib import Path
from typing import Any


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a per-process-unique temp file then ``Path.replace`` --
    atomic on both POSIX and Windows (same volume), so a crash mid-write never leaves ``path``
    holding a torn/partial file that a reader could mistake for valid data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}{path.suffix}.tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(path)


def write_json_checkpoint(obj: Any, checkpoint_path: Path) -> None:
    """Gzip a JSON serialization of ``obj`` to ``checkpoint_path``, atomically. Call this after
    every write to the live cache so the checkpoint is never more than one write stale -- a human
    still has to ``git add``/commit the resulting file, but its on-disk content is always current,
    never dependent on a separately-remembered backup script."""
    payload = json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(checkpoint_path, gzip.compress(payload, mtime=0))


def read_json_with_checkpoint_fallback(cache_path: Path, checkpoint_path: Path) -> Any:
    """Read and return the JSON object at ``cache_path``. If ``cache_path`` is missing or empty
    (a fresh checkout, a wiped local cache directory) and ``checkpoint_path`` exists, decompresses
    and parses the checkpoint instead, writes it back out to ``cache_path`` (so the next call
    reads it directly, and any live-cache-writing code path stays oblivious to where the data
    originally came from), and returns it. Returns ``None`` if neither exists -- the same honest-
    absence contract as a cache that was simply never populated."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if not checkpoint_path.exists():
        return None
    obj = json.loads(gzip.decompress(checkpoint_path.read_bytes()).decode("utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return obj


def write_bytes_checkpoint(cache_path: Path, checkpoint_path: Path) -> None:
    """Gzip a byte-for-byte snapshot of the file at ``cache_path`` to ``checkpoint_path``,
    atomically. For a cache that IS a file on disk (a JSONL ledger, a SQLite database) rather than
    an in-memory object serialized fresh each time -- no re-serialization, just a compressed copy
    of whatever bytes are currently there. A no-op if ``cache_path`` does not exist yet."""
    if not cache_path.exists():
        return
    _atomic_write_bytes(checkpoint_path, gzip.compress(cache_path.read_bytes(), mtime=0))


def read_bytes_with_checkpoint_fallback(cache_path: Path, checkpoint_path: Path) -> bool:
    """Restore ``cache_path`` from ``checkpoint_path`` if ``cache_path`` is missing or empty and
    the checkpoint exists. Returns ``True`` if a restore happened, ``False`` otherwise (including
    when ``cache_path`` already had real content, or neither file exists) -- a caller that needs
    the actual bytes reads ``cache_path`` itself afterward, since restoring in place (rather than
    returning the bytes directly) is what lets an existing file-based reader (a SQLite connection
    open, a line-by-line JSONL scan) pick it up with no further change."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return False
    if not checkpoint_path.exists():
        return False
    _atomic_write_bytes(cache_path, gzip.decompress(checkpoint_path.read_bytes()))
    return True
