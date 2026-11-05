"""One pruned directory walk for the dev tools: never enters ``.claude`` (Claude Code's agent worktrees).

Each worktree under ``.claude`` is a COMPLETE copy of the repository. ``root.rglob(...)`` and an unpruned ``os.walk(root)`` enter all of
them and a tool handed the repository root then reads every copy (an audit test took 19 to 30 minutes with about 20 of them). Pruning
happens before a directory is entered, so the copies are never listed. ``tests/test_meta/test_no_unpruned_tree_walk.py`` fails a new
``rglob``/``os.walk`` in this package that neither uses this helper nor carries a ``# walk-ok: <reason>`` marker.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple, Union

#: Directories no whole-tree scan enters. ``.claude`` is the one that matters: it holds the agent worktrees.
DEFAULT_PRUNED_DIRS = frozenset({".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules", ".venv", ".claude"})


def iter_files_pruned(root: Union[str, Path], suffixes: Tuple[str, ...] = (), exclude_dirs: Iterable[str] = DEFAULT_PRUNED_DIRS) -> List[Path]:
    """Sorted files under *root* whose name ends with one of *suffixes* (every file when empty), never entering *exclude_dirs*.

    Only directories BELOW *root* are judged, so where *root* itself lives never matters. A missing *root* yields an empty list.
    """
    skip = frozenset(exclude_dirs)
    base = Path(root)
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in skip]
        out.extend(Path(dirpath) / name for name in filenames if not suffixes or name.endswith(suffixes))
    return sorted(out, key=lambda p: p.relative_to(base).as_posix())
