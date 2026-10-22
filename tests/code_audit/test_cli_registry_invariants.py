"""The CLI test split out of test_registry_invariants.py: the CLI stays in pyutilz, the rest of that file moved to py-ci-shared."""

from __future__ import annotations

from pathlib import Path


from py_ci_shared.code_audit import (
    Finding,
)


def _oom_outside_parent(root, exclude_dirs=None, *, parent_pid: int):
    """Runs out of memory in any process but ``parent_pid`` -- a scanner that fits alone but not beside its pool siblings."""
    import os

    if os.getpid() != parent_pid:
        raise MemoryError
    return [Finding(check="oom_probe", severity="Low", file="ok.py", line=1, snippet="x = 1", detail="ran in the parent")]


def test_cli_renders_and_gates_on_an_unknown_severity(tmp_path: Path, capsys, monkeypatch):
    """F200 regression: ``sev_order.get(f.severity, 99) <= cutoff`` deleted an unrecognised severity
    at EVERY --min-severity setting and the exit code ignored it too."""
    from py_ci_shared.code_audit import cli as _cli
    from py_ci_shared.code_audit.cli import main as cli_main

    stray = Finding.__new__(Finding)
    object.__setattr__(stray, "check", "stray")
    object.__setattr__(stray, "severity", "Medium")
    object.__setattr__(stray, "file", "x.py")
    object.__setattr__(stray, "line", 1)
    object.__setattr__(stray, "snippet", "x = 1")
    object.__setattr__(stray, "detail", "d")
    monkeypatch.setattr(_cli, "run_all", lambda *a, **k: [stray])

    assert cli_main([str(tmp_path), "--min-severity", "P0"]) == 1
    assert "stray" in capsys.readouterr().out
