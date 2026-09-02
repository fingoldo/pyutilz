"""Complexity ratchet: C901 findings may go DOWN, never up.

``C901`` (mccabe, ``max-complexity = 20``) is selected by the project's ruff config but dropped
from both blocking entry points -- CI's ``ruff-blocking`` job passes ``ignore: "C901"`` and the
``ruff-real-bugs`` pre-commit hook runs ``--ignore C901`` -- so it surfaces only in warn-only
advisory runs. The threshold was measured from this codebase's own complexity distribution, but
with nothing enforcing it both the COUNT and the per-function maxima could only grow.

This test freezes the current set as a baseline: an existing over-threshold function may shrink or
disappear, but a NEW one, or an existing one getting MORE complex, fails here. The 23 functions in
the baseline are left to deliberate, separately-reviewed refactors rather than being made blocking
today (making them blocking now would just mean 23 blanket noqas).

Refresh with ``pytest tests/test_meta/test_complexity_ratchet.py --refresh-complexity-baseline``
-- and only ever to record a REDUCTION.
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404 - runs the pinned dev-extra `ruff` module via sys.executable, no user input reaches argv
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_BASELINE_PATH = _HERE / "_complexity_baseline.json"
_SCAN_PATHS = ["src", "tests", "scripts"]

_MESSAGE_RE = re.compile(r"^`(?P<name>[^`]+)` is too complex \((?P<score>\d+) > \d+\)$")


def _collect() -> dict[str, int]:
    """Run ruff's C901 check over the scanned paths, returning {"<rel path>::<function>": complexity}."""
    proc = subprocess.run(  # nosec B603 - fixed argv, sys.executable + the ruff module
        [sys.executable, "-m", "ruff", "check", *_SCAN_PATHS, "--select", "C901", "--output-format", "json"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        pytest.skip(f"ruff C901 scan unavailable in this environment (rc={proc.returncode}): {proc.stderr.strip()[:400]}")
    try:
        findings = json.loads(proc.stdout)
    except json.JSONDecodeError:  # pragma: no cover - defensive
        pytest.skip("ruff produced non-JSON output; C901 ratchet cannot run here")
    out: dict[str, int] = {}
    for item in findings:
        match = _MESSAGE_RE.match(item["message"])
        if match is None:  # pragma: no cover - defensive
            continue
        rel = Path(item["filename"]).resolve().relative_to(_REPO_ROOT).as_posix()
        out[f"{rel}::{match.group('name')}"] = int(match.group("score"))
    return out


def test_c901_complexity_does_not_regress(request: pytest.FixtureRequest) -> None:
    """Every C901 finding must already be in the baseline, at no worse a complexity than recorded."""
    if os.environ.get("PY_CI_SHARED_DIR") is None:
        pytest.skip("PY_CI_SHARED_DIR unset; pyproject.toml's ruff `extend` cannot resolve, so the rule set would be wrong")
    current = _collect()

    if request.config.getoption("--refresh-complexity-baseline"):
        _BASELINE_PATH.write_text(json.dumps(dict(sorted(current.items())), indent=2) + "\n", encoding="utf-8", newline="\n")
        pytest.skip(f"Refreshed complexity baseline: {len(current)} entries")

    baseline: dict[str, int] = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

    new_offenders = sorted(k for k in current if k not in baseline)
    assert not new_offenders, (
        "New function(s) over the C901 complexity threshold. Split them, or -- if the complexity is genuinely "
        f"irreducible -- refresh the baseline deliberately as part of that review: {new_offenders}"
    )

    worsened = sorted(f"{k}: {baseline[k]} -> {v}" for k, v in current.items() if v > baseline[k])
    assert not worsened, f"Already-complex function(s) got MORE complex; the ratchet only turns down: {worsened}"
