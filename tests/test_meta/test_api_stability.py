"""PT-6 — public-API stability snapshot for pyutilz.

Captures the public surface of the package — top-level ``__all__`` plus
the alias map plus, for each alias, the public names of the underlying
target module — into ``tests/test_meta/_api_snapshot.json``. Renames
and removals fail the test until the snapshot is explicitly refreshed;
additions are silent (the API can grow without breaking downstream).

Refresh after an intentional rename:

  pytest tests/test_meta/test_api_stability.py --refresh-api-snapshot

Then commit the snapshot. Reviewers see the diff and approve / push back.

This is the highest-value defensive test for a backward-compat-oriented
library — pyutilz's whole point is that downstream code has been
pinning against ``pyutilz.<alias>`` for years; a silent rename breaks
people we don't control.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

import pyutilz
from pyutilz.dev.meta_test_utils import (
    capture_module_surface,
    capture_signature,
)

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "_api_snapshot.json"


def _refresh_requested() -> bool:
    return "--refresh-api-snapshot" in sys.argv


# Signature / surface capture moved to ``pyutilz.dev.meta_test_utils``;
# imported as ``capture_signature`` / ``capture_module_surface`` above.


def _build_snapshot() -> dict:
    snapshot: dict = {
        "package_all": sorted(getattr(pyutilz, "__all__", [])),
        "module_aliases": dict(sorted(pyutilz._MODULE_ALIASES.items())),
        "alias_surfaces": {},
    }
    # For each alias, capture the public surface of the target.
    for alias, real_path in sorted(pyutilz._MODULE_ALIASES.items()):
        try:
            mod = importlib.import_module(real_path)
        except ImportError:
            snapshot["alias_surfaces"][alias] = {
                "_import_error": "module fails to import"
            }
            continue
        snapshot["alias_surfaces"][alias] = capture_module_surface(mod)
    return snapshot


def test_public_api_matches_snapshot():
    current = _build_snapshot()
    if _refresh_requested() or not _SNAPSHOT_PATH.exists():
        _SNAPSHOT_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        n_aliases = len(current["module_aliases"])
        n_symbols = sum(len(v) for v in current["alias_surfaces"].values())
        pytest.skip(
            f"snapshot refreshed at {_SNAPSHOT_PATH.name} "
            f"({n_aliases} aliases, {n_symbols} symbols)"
        )

    expected = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    diffs: list[str] = []

    # Top-level __all__ and alias map — flat dict comparison.
    for key in ("package_all", "module_aliases"):
        if expected.get(key) != current.get(key):
            old = expected.get(key)
            new = current.get(key)
            diffs.append(f"CHANGED [{key}]:\n      was: {old}\n      now: {new}")

    # Per-alias surfaces — only flag REMOVALS / RENAMES (additions silent).
    additions: list[str] = []
    for alias, exp_surface in expected.get("alias_surfaces", {}).items():
        cur_surface = current.get("alias_surfaces", {}).get(alias)
        if cur_surface is None:
            diffs.append(f"REMOVED alias: {alias}")
            continue
        for name, exp_kind in exp_surface.items():
            cur_kind = cur_surface.get(name)
            if cur_kind is None:
                diffs.append(f"REMOVED: pyutilz.{alias}.{name}")
                continue
            if cur_kind != exp_kind:
                diffs.append(
                    f"CHANGED: pyutilz.{alias}.{name}\n      was: {exp_kind}"
                    f"\n      now: {cur_kind}"
                )
        for name in cur_surface:
            if name not in exp_surface:
                additions.append(f"pyutilz.{alias}.{name}")

    if additions:
        sys.stderr.write(
            f"\n[test_public_api_matches_snapshot] {len(additions)} new "
            f"public symbol(s): {', '.join(additions[:15])}"
            + (" ..." if len(additions) > 15 else "")
            + "\n  (additions are non-breaking; refresh the snapshot when "
            f"ready: pytest "
            f"{Path(__file__).relative_to(Path.cwd()).as_posix()} "
            f"--refresh-api-snapshot)\n"
        )

    if diffs:
        pytest.fail(
            f"{len(diffs)} public-API change(s) detected against snapshot. "
            f"If intentional, refresh with ``pytest "
            f"tests/test_meta/test_api_stability.py "
            f"--refresh-api-snapshot`` and commit:\n  "
            + "\n  ".join(diffs[:20])
            + (f"\n  ... and {len(diffs) - 20} more" if len(diffs) > 20 else "")
        )
