"""Drift tracker — surfaces growth in ``_USER_DEFERRED_*`` whitelists.

Every meta-test in the suite has a per-file ``_USER_DEFERRED_*`` (or
``_GRANDFATHERED``) set/dict that holds findings the maintainer chose to
defer. Without a tracker, those collections quietly grow — a deferred
item costs nothing to add but a real cleanup costs effort, so the
asymmetry favours accumulation.

This test:

  1. Walks every meta-test file in ``tests/test_meta/`` and counts the
     entries in every ``_USER_DEFERRED_*`` / ``_GRANDFATHERED`` set/dict
     (via AST so the count survives reformatting).
  2. Compares to a baseline stored in ``tests/test_meta/_debt_baseline.json``.
  3. Fails if any whitelist GREW since the baseline; ignores shrinkage.
  4. On the very first run (no baseline), captures the snapshot and skips.

Refresh after intentionally accepting more debt::

    pytest tests/test_meta/test_deferred_drift.py --refresh-debt-baseline

Then commit the baseline. Reviewers see the diff and can push back if
the growth is unexplained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pyutilz.dev.meta_test_utils import count_user_deferred_entries

TEST_META_DIR = Path(__file__).resolve().parent
_BASELINE_PATH = TEST_META_DIR / "_debt_baseline.json"


def _refresh_requested() -> bool:
    return "--refresh-debt-baseline" in sys.argv


def test_user_deferred_lists_havent_grown():
    current = count_user_deferred_entries(TEST_META_DIR)

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        total = sum(current.values())
        pytest.skip(
            f"debt baseline refreshed at {_BASELINE_PATH.name} "
            f"({len(current)} whitelist(s), {total} total entry(ies))"
        )

    baseline = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    grown: list[str] = []
    new_keys: list[str] = []

    for key, count in current.items():
        old = baseline.get(key)
        if old is None:
            new_keys.append(f"{key}: NEW (now {count})")
        elif count > old:
            grown.append(f"{key}: {old} → {count} (+{count - old})")

    if grown or new_keys:
        diff_total = sum(current.values()) - sum(baseline.values())
        sign = "+" if diff_total > 0 else ""
        msg_parts = [
            f"Tech-debt whitelist(s) GREW since baseline "
            f"(net {sign}{diff_total} entries):"
        ]
        if grown:
            msg_parts.append("  GROWN:\n    " + "\n    ".join(grown))
        if new_keys:
            msg_parts.append("  NEW:\n    " + "\n    ".join(new_keys))
        msg_parts.append(
            "  Either drain entries from the listed whitelists, OR "
            "refresh the baseline if the growth is intentional:\n    "
            "pytest tests/test_meta/test_deferred_drift.py "
            "--refresh-debt-baseline"
        )
        pytest.fail("\n".join(msg_parts))

    # Informational only — print a summary so the dashboard stays visible.
    shrunk = [
        f"{k}: {baseline[k]} → {current[k]} (-{baseline[k] - current[k]})"
        for k in current
        if k in baseline and current[k] < baseline[k]
    ]
    if shrunk:
        sys.stderr.write(
            f"\n[test_user_deferred_lists_havent_grown] {len(shrunk)} "
            f"whitelist(s) SHRANK — drained:\n  " + "\n  ".join(shrunk)
            + "\n  Refresh baseline to lock in: pytest "
            f"tests/test_meta/test_deferred_drift.py --refresh-debt-baseline\n"
        )
