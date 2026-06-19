#!/usr/bin/env python
"""Warn-only formatting / lint check for the pre-commit hook.

Runs ``ruff format --check --diff`` and ``ruff check`` (lint) over the staged
Python files in CHECK mode. It NEVER rewrites files and ALWAYS exits 0, so it
surfaces formatting / lint differences as WARNINGS without auto-reformatting and
without blocking the commit.

Auto-reformat is intentionally disabled (project policy): the formatters churn
large diffs that collide with concurrent work. To actually apply changes, run
``ruff format`` / ``ruff check --fix`` manually.
"""
import subprocess
import sys

_files = [a for a in sys.argv[1:] if a.endswith(".py")]
if not _files:
    sys.exit(0)

_warned = False
for _cmd in (
    # taste only: formatting + pycodestyle/naming/pyupgrade/import-order. Real
    # problems (pyflakes F + bugbear B) are a SEPARATE blocking hook, not here.
    [sys.executable, "-m", "ruff", "format", "--check", "--diff", *_files],
    # ``--ignore`` mirrors the pyproject [tool.ruff.lint] py38-parity ignores: an explicit ``--select UP``
    # would otherwise re-enable them on the CLI and spam UP006/UP007/UP037/UP045 (Optional/Union/List
    # modernization) warnings the project deliberately keeps for 3.8/3.9 compatibility. Keep this in sync
    # with pyproject's ignore list so the warn-only hook does not nag about deliberately-kept idioms.
    [sys.executable, "-m", "ruff", "check", "--select", "E,W,N,UP,I",
     "--ignore", "UP006,UP007,UP037,UP045,UP031", *_files],
):
    try:
        if subprocess.run(_cmd).returncode != 0:
            _warned = True
    except Exception as _e:  # ruff missing / any error -> warn, never block
        print(f"[format-warn] skipped {' '.join(_cmd[2:4])}: {_e}", file=sys.stderr)

if _warned:
    print(
        "\n[format-warn] The diffs / lint issues above are WARNINGS ONLY -- nothing "
        "was rewritten and the commit is NOT blocked. Run 'ruff format' / "
        "'ruff check --fix' manually to apply.",
        file=sys.stderr,
    )
sys.exit(0)
