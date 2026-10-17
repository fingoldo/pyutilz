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
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Tuple

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


def cache_key(root: Path, selected: Iterable[str], exclude_dirs: frozenset[str]) -> Tuple[str, Tuple]:
    """A hex digest for the run, and the signature it was computed from (stored alongside, compared on read)."""
    signature = (_signature(root, exclude_dirs), _signature(_PACKAGE_DIR, frozenset({"__pycache__"})))
    description = (str(root.resolve()), tuple(sorted(selected)), tuple(sorted(exclude_dirs)))
    digest = hashlib.sha256(repr(description).encode("utf-8")).hexdigest()
    return digest, signature


def _cache_dir() -> Path:
    path = Path.home() / ".cache" / "pyutilz-code-audit"
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)
    return path


def load(digest: str, signature: Tuple) -> Any:
    """The stored result for *digest* when its signature still matches, else None.

    Typed ``Any``: the caller stored exactly one shape under this digest, so it is the one that comes back.
    """
    try:
        with open(_cache_dir() / f"{digest}.pkl", "rb") as handle:
            stored_signature, result = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, AttributeError, ImportError):  # best-effort: an unreadable entry only costs a fresh scan
        return None
    return result if stored_signature == signature else None


def store(digest: str, signature: Tuple, result: object) -> None:
    """Write *result* for *digest*; a failure to write only costs the next run its scan."""
    try:
        target = _cache_dir() / f"{digest}.pkl"
        tmp = target.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump((signature, result), handle)
        os.replace(tmp, target)
    except OSError as exc:
        logger.warning("code-audit scan cache not written (%s); the next run scans again", exc)
