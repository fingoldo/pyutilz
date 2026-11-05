"""The dev tools' tree walks must not enter ``.claude`` (Claude Code's agent worktrees, each a COMPLETE copy of the repository).

Each test hands a toy repo ROOT to a tool, with a real file and a decoy under ``.claude/worktrees/agent-x/``, and checks both the
result and that no directory below ``.claude`` was listed (``os.scandir`` recorded).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyutilz.dev._pruned_walk import iter_files_pruned
from pyutilz.dev.persistence_sweep import written_files

DECOY_DIR = ".claude/worktrees/agent-x"


def write(root: Path, rel: str, text: str = "x = 1\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def listed(monkeypatch):
    seen = []
    real = os.scandir

    def recording(path="."):
        seen.append(os.fsdecode(path))
        return real(path)

    monkeypatch.setattr(os, "scandir", recording)
    return seen


def assert_never_entered(seen):
    assert seen, "the tool listed no directory at all, so this test proved nothing"
    assert [p for p in seen if ".claude" in Path(p).parts] == []


def test_iter_files_pruned_skips_claude_filters_suffix_and_sorts(tmp_path, listed):
    write(tmp_path, "pkg/b.py")
    write(tmp_path, "pkg/a.py")
    write(tmp_path, "pkg/c.txt")
    write(tmp_path, f"{DECOY_DIR}/d.py")
    assert iter_files_pruned(tmp_path, (".py",)) == [tmp_path / "pkg" / "a.py", tmp_path / "pkg" / "b.py"]
    assert_never_entered(listed)


def test_iter_files_pruned_honours_exclude_dirs_and_a_missing_root(tmp_path):
    write(tmp_path, "a/x.py")
    write(tmp_path, "b/y.txt")
    assert iter_files_pruned(tmp_path, exclude_dirs={"a"}) == [tmp_path / "b" / "y.txt"]
    assert iter_files_pruned(tmp_path / "missing", (".py",)) == []


def test_iter_files_pruned_ignores_where_the_root_itself_lives(tmp_path):
    root = tmp_path / ".claude" / "worktrees" / "agent-y"
    write(root, "pkg/a.py")
    assert iter_files_pruned(root, (".py",)) == [root / "pkg" / "a.py"]


def test_written_files(tmp_path, listed):
    real = write(tmp_path, "store/a.jsonl", "{}\n")
    write(tmp_path, f"{DECOY_DIR}/store/b.jsonl", "{}\n")
    assert written_files([tmp_path]) == [real]
    assert_never_entered(listed)
