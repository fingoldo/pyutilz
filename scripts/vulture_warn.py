#!/usr/bin/env python
"""Warn-only dead-code check for the pre-commit hook.

Runs ``vulture`` (min-confidence 80, matching the CI step) over the staged Python
files under ``src/pyutilz`` and ALWAYS exits 0, so findings surface locally as
WARNINGS at commit time instead of only showing up later in the CI artifact.

Vulture has real false positives on dynamic-dispatch code (lazy-import proxies,
pydantic fields, pytest fixtures, tuple-unpacking placeholders) -- that's why this
is warn-only rather than a blocking gate, same posture as bandit here. Confirmed
false positives / documented no-ops are tracked once in ``scripts/vulture_whitelist.py``
instead of being re-reviewed on every run.
"""
import os
import subprocess
import sys

_files = [a for a in sys.argv[1:] if a.endswith(".py") and "src/pyutilz" in a.replace("\\", "/")]
if not _files:
    sys.exit(0)

_whitelist = os.path.join(os.path.dirname(__file__), "vulture_whitelist.py")
_cmd = [sys.executable, "-m", "vulture", "--min-confidence", "80", *_files, _whitelist]
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
