#!/usr/bin/env python
"""Warn-only dead-code check for the pre-commit hook.

Runs ``vulture`` (min-confidence 80, matching the CI step) over the staged Python
files under ``src/pyutilz`` and ALWAYS exits 0, so findings surface locally as
WARNINGS at commit time instead of only showing up later in the CI artifact.

Vulture has real false positives on dynamic-dispatch code (lazy-import proxies,
pydantic fields, pytest fixtures, tuple-unpacking placeholders) -- that's why this
is warn-only rather than a blocking gate, same posture as bandit here.
"""
import subprocess
import sys

_files = [a for a in sys.argv[1:] if a.endswith(".py") and "src/pyutilz" in a.replace("\\", "/")]
if not _files:
    sys.exit(0)

_cmd = [sys.executable, "-m", "vulture", "--min-confidence", "80", *_files]
try:
    _warned = subprocess.run(_cmd).returncode != 0
except Exception as _e:  # vulture missing / any error -> warn, never block
    print(f"[vulture-warn] skipped: {_e}", file=sys.stderr)
    sys.exit(0)

if _warned:
    print(
        "\n[vulture-warn] The findings above are WARNINGS ONLY -- the commit is NOT "
        "blocked. Some are false positives on dynamic-dispatch code (lazy-import "
        "proxies, pydantic fields, pytest fixtures) -- use judgement before deleting.",
        file=sys.stderr,
    )
sys.exit(0)
