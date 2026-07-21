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
import importlib.util
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


def _fresh_module_copy(mod):
    """Execute a fully separate copy of ``mod`` from its own import spec, without touching
    ``sys.modules`` or the shared, already-imported module object the rest of the test session
    depends on.

    Regression (2026-07-21 audit round 2, verification pass): this used to be
    ``importlib.reload(mod)``, which re-executes a module's top-level code IN PLACE, into the
    SAME ``__dict__`` other already-imported code still references. Several pyutilz modules use a
    module-level sentinel object for the "was this parameter passed at all?" idiom (e.g.
    ``pythonlib._GET_ATTR_UNSET = object()``) -- reload silently rebinds that name to a brand new
    object, while any function imported (``from pyutilz.pythonlib import get_attr``) by an
    EARLIER-collected test file keeps its OLD sentinel as its own frozen parameter default (bound
    once at def time). The two objects then disagree under ``is``, breaking that function for
    every subsequent test in the same session -- reproduced concretely: running this file before
    ``test_pythonlib_extra2.py`` made ``get_attr({"a": 1}, "b")`` return the raw sentinel instead
    of ``[]``, purely because of THIS test's reload, regardless of test order otherwise. Building
    an unregistered, throwaway module object from the same spec gets the same "clean state,
    uncontaminated by other tests' monkeypatching" snapshot goal without mutating anything shared.
    """
    spec = importlib.util.find_spec(mod.__name__)
    if spec is None or spec.loader is None:
        return mod
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    return fresh


def _build_snapshot() -> dict:
    snapshot: dict = {
        "package_all": sorted(getattr(pyutilz, "__all__", [])),
        "module_aliases": dict(sorted(pyutilz._MODULE_ALIASES.items())),
        "alias_surfaces": {},
    }
    # For each alias, capture the public surface of the target.
    # A fresh, unregistered module copy (see ``_fresh_module_copy``) ensures a CLEAN module state
    # — earlier tests in the same session may have mutated module-level globals (e.g.
    # ``test_graphql.py::test_sets_global_client`` patches ``pyutilz.web.graphql.client``;
    # ``init_logging`` reassigns ``pyutilz.dev.logginglib.logger`` to a real Logger). Without this
    # the snapshot captures the polluted state and CI surfaces phantom diffs vs the clean-import
    # snapshot committed to git. Deliberately NOT ``importlib.reload`` -- see
    # ``_fresh_module_copy``'s docstring for the identity-corruption bug that caused.
    for alias, real_path in sorted(pyutilz._MODULE_ALIASES.items()):
        try:
            mod = importlib.import_module(real_path)
        except ImportError:
            snapshot["alias_surfaces"][alias] = {"_import_error": "module fails to import"}
            continue
        # ``_fresh_module_copy`` never touches ``mod``'s own ``__dict__`` -- unlike an in-place
        # ``importlib.reload(mod)``, which independently caused TWO distinct identity-trap
        # symptoms caught live in this codebase: pythonlib._GET_ATTR_UNSET rebinding broke
        # get_attr() for any earlier-imported caller (see _fresh_module_copy's docstring), and
        # tests/test_polarslib_extra.py's ``remove_constant_columns is drop_constant_columns``
        # alias check flipped to False whenever this test ran first in the same session (a name
        # some OTHER test module imported BEFORE this one ran kept pointing to the OLD object
        # while a later fresh import of the same name saw reload's NEW one). Building a separate,
        # unregistered module object sidesteps both failure modes at the root, with nothing to
        # snapshot or restore afterward.
        try:
            mod = _fresh_module_copy(mod)
        except Exception:
            # If the fresh exec fails (rare — modules with side-effecting top
            # level), fall back to the live module state.
            pass
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
        pytest.skip(f"snapshot refreshed at {_SNAPSHOT_PATH.name} " f"({n_aliases} aliases, {n_symbols} symbols)")

    expected = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    diffs: list[str] = []

    # Top-level __all__ and alias map — flat dict comparison.
    for key in ("package_all", "module_aliases"):
        if expected.get(key) != current.get(key):
            old = expected.get(key)
            new = current.get(key)
            diffs.append(f"CHANGED [{key}]:\n      was: {old}\n      now: {new}")

    # Per-alias surfaces — only flag REMOVALS / RENAMES (additions silent).
    # If EITHER side captured ``_import_error`` (e.g. CI lacks an optional
    # third-party dep like jellyfish/filelock/flask/IPython that the snapshot
    # author had installed) skip the alias entirely — that's an env mismatch,
    # not API drift. ``test_module_alias_integrity`` already polices real
    # pyutilz module presence.
    additions: list[str] = []
    for alias, exp_surface in expected.get("alias_surfaces", {}).items():
        cur_surface = current.get("alias_surfaces", {}).get(alias)
        if cur_surface is None:
            diffs.append(f"REMOVED alias: {alias}")
            continue
        if "_import_error" in exp_surface or "_import_error" in cur_surface:
            continue
        for name, exp_kind in exp_surface.items():
            cur_kind = cur_surface.get(name)
            if cur_kind is None:
                diffs.append(f"REMOVED: pyutilz.{alias}.{name}")
                continue
            if cur_kind != exp_kind:
                diffs.append(f"CHANGED: pyutilz.{alias}.{name}\n      was: {exp_kind}" f"\n      now: {cur_kind}")
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
            f"--refresh-api-snapshot`` and commit:\n  " + "\n  ".join(diffs[:20]) + (f"\n  ... and {len(diffs) - 20} more" if len(diffs) > 20 else "")
        )
