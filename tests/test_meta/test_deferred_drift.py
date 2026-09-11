"""Drift tracker: the ``_USER_DEFERRED_*`` / ``_GRANDFATHERED`` whitelists in this directory may not grow.

Every meta-test that tolerates known debt keeps it in a named collection. A deferred item costs nothing to add and
a real cleanup costs effort, so left alone the collections only grow. ``py_ci_shared.deferred_drift`` counts their
entries with ``pyutilz.dev.meta_test_utils.count_user_deferred_entries`` and compares with ``_debt_baseline.json``:
a new or grown list fails, and so does a list that shrank or vanished until the baseline follows it down.

Refresh after an intentional change::

    pytest tests/test_meta/test_deferred_drift.py --refresh-debt-baseline
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.deferred_drift import assert_deferred_lists_not_grown

TEST_META_DIR = Path(__file__).resolve().parent
_BASELINE_PATH = TEST_META_DIR / "_debt_baseline.json"


def test_user_deferred_lists_havent_grown():
    """No deferred-debt whitelist grew, appeared, shrank or vanished without the baseline following it."""
    assert_deferred_lists_not_grown(TEST_META_DIR, _BASELINE_PATH)
