"""On-disk memo of full code-audit runs, shared by every gate in every process of a test session.

Several gates scan the same production tree, and under pytest-xdist each worker used to repeat the whole scan. The key covers
everything a result depends on: the scanned root and its check selection, the excluded directories, every file under the root
(scanners read ``.sql``, tests and docs too, not only ``.py``), and the code_audit package itself. A hit is therefore a result
the current tree and scanner code would reproduce. The cache is an optimisation: any read or write failure falls back to a
fresh scan.

The directory lives under the user's home, not the shared temp directory, so another local user cannot pre-create or plant an
entry there (the predictable-temp-path class of defect).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ._base import Finding

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent


def _signature(root: Path, exclude_dirs: frozenset[str]) -> Tuple[Tuple[str, int, int], ...]:
    """(relative path, mtime_ns, size) of every regular file under *root* outside *exclude_dirs*, sorted."""
    entries: list[tuple[str, int, int]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for name in filenames:
            full = Path(dirpath) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            entries.append((full.relative_to(root).as_posix(), stat.st_mtime_ns, stat.st_size))
    entries.sort()
    return tuple(entries)


def cache_key(root: Path, selected: Iterable[str], exclude_dirs: frozenset[str], consumer_roots: Iterable[Path] = ()) -> Tuple[str, Tuple]:
    """A hex digest for the run, and the signature it was computed from (stored alongside, compared on read).

    ``consumer_roots`` are extra trees a scanner may read (see ``scan_possibly_dead_import``): their file
    stats join the signature, so editing a test that consumes a re-export invalidates the cached run.
    """
    consumers = tuple(sorted(str(c.resolve()) for c in consumer_roots))
    signature = (
        _signature(root, exclude_dirs),
        _signature(_PACKAGE_DIR, frozenset({"__pycache__"})),
        tuple((c, _signature(Path(c), exclude_dirs)) for c in consumers),
    )
    description = (str(root.resolve()), tuple(sorted(selected)), tuple(sorted(exclude_dirs)), consumers)
    digest = hashlib.sha256(repr(description).encode("utf-8")).hexdigest()
    return digest, signature


def _cache_dir() -> Path:
    """The per-user memo directory, created on first use and private to its owner on POSIX."""
    path = Path.home() / ".cache" / "pyutilz-code-audit"
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)
    return path


def load(digest: str, signature: Tuple) -> Optional[Tuple[List[Finding], Dict[str, str]]]:
    """The stored ``(findings, failed)`` for *digest* when its signature still matches, else None."""
    try:
        with open(_cache_dir() / f"{digest}.json", "r", encoding="utf-8") as handle:
            document = json.load(handle)
        if document["signature"] != _as_json(signature):
            return None
        findings = [Finding(**entry) for entry in document["findings"]]
        failed = dict(document["failed"])
    except (OSError, ValueError, KeyError, TypeError):  # best-effort: an unreadable or foreign entry only costs a fresh scan
        return None
    return findings, failed


def store(digest: str, signature: Tuple, findings: List[Finding], failed: Dict[str, str]) -> None:
    """Write the result for *digest* as JSON; a failure to write only costs the next run its scan."""
    document = {
        "signature": _as_json(signature),
        "findings": [{name: getattr(finding, name) for name in _FINDING_FIELDS} for finding in findings],
        "failed": dict(failed),
    }
    try:
        target = _cache_dir() / f"{digest}.json"
        tmp = target.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        os.replace(tmp, target)
    except OSError as exc:
        logger.warning("code-audit scan cache not written (%s); the next run scans again", exc)


_FINDING_FIELDS = ("check", "severity", "file", "line", "snippet", "detail")


def _as_json(value: object) -> object:
    """Tuples become lists, so a signature read back from JSON compares equal to the one computed in memory."""
    if isinstance(value, (tuple, list)):
        return [_as_json(item) for item in value]
    return value
