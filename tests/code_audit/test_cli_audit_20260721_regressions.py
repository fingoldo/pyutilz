"""The CLI test split out of test_audit_20260721_regressions.py: the CLI stays in pyutilz, the rest of that file moved to py-ci-shared."""

from __future__ import annotations

from pathlib import Path


from ._helpers import _write


def test_cli_min_severity_does_not_weaken_exit_code(tmp_path: Path):
    """Regression: --min-severity previously filtered `findings` BEFORE the exit-code check,
    so a real P1 finding silently exited 0 once filtered out of the display."""
    from pyutilz.dev.code_audit import main as cli_main

    _write(tmp_path, "bad.py", """
async def process(item):
    await item.save()

def caller(item):
    process(item)
""")
    assert cli_main([str(tmp_path), "--min-severity", "Low"]) == 1
    assert cli_main([str(tmp_path), "--min-severity", "P0"]) == 1
