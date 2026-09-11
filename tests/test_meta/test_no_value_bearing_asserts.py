"""No new guard that checks a VALUE may ride on an ``assert`` in ``src/pyutilz``.

``python -O`` deletes every ``assert``. A guard that only narrows a type for mypy loses nothing when it goes; a guard
that checks a bound, a sum or a membership (``compression >= -1 and compression <= 9``) loses the only thing
enforcing it. ``py_ci_shared.value_bearing_asserts`` keeps ``assert x is not None`` and a bare name legal and flags
the rest, ``isinstance`` included.

The existing ones are recorded in ``_value_bearing_asserts_baseline.json`` as debt keyed ``file::expression``. The
baseline only shrinks: converting one to an explicit ``raise`` makes its entry stale, and a stale entry fails.
"""

from __future__ import annotations

import json
from pathlib import Path

from py_ci_shared.value_bearing_asserts import assert_no_value_bearing_asserts, find_value_bearing_asserts

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "pyutilz"
_BASELINE_PATH = Path(__file__).resolve().parent / "_value_bearing_asserts_baseline.json"


def regenerate_baseline(path: Path = _BASELINE_PATH) -> None:
    """Rewrite the debt baseline from today's value-bearing asserts."""
    offenders, _seen = find_value_bearing_asserts(PACKAGE_ROOT)
    keys = sorted({f"{entry.split('  ', 1)[0].rsplit(':', 1)[0]}::{entry.split('  ', 1)[1]}" for entry in offenders})
    path.write_text(json.dumps(keys, indent=2) + "\n", encoding="utf-8")


def test_no_new_value_bearing_asserts():
    """No production assert checks a value beyond the recorded debt, and the debt holds no stale entry."""
    # 49 asserts in src/pyutilz today; a walk that saw fewer than 20 has stopped reaching the package.
    assert_no_value_bearing_asserts(PACKAGE_ROOT, min_asserts_seen=20, baseline_path=_BASELINE_PATH)
