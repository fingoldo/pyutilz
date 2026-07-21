"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: a module whose public
top-level functions are predominantly ``snake_case`` should not also expose a PascalCase/camelCase
one -- undiscoverable via the IDE-autocomplete convention the rest of the module trains a caller
to expect.

Class this catches: ``database.db.__init__``'s ``EnsurePgTableExists``/``ReadTableIntoDic``/
``ReadTableIntoDicReversed``/``GetIdByKeyFieldAndInsertIfNeeded`` mixed in with ~28 modern
snake_case functions (the original HIGH-severity audit finding, fixed with snake_case primaries +
deprecated PascalCase aliases). Running this check surfaced one more real instance beyond the
original audit -- ``database.db.sql_helpers.MakeSetExcludedClause`` (Hungarian-notation params
``sFields``/``bAddUpdatedAtTimestamp``, same TypeError-from-mismatched-keyword risk as the
original finding) -- now fixed the same way (renamed to ``make_set_excluded_clause``, old name
kept as a deprecated, warning wrapper).

Mechanism: per module, classify each public top-level function name against
``^[a-z_][a-z0-9_]*$``. If a module has BOTH matching and non-matching names, flag the
non-matching ones.

Baseline-JSON snapshot (same idiom as ``test_no_bare_except.py``), not an unconditional fail: the
5 non-conforming names currently in this codebase are all DELIBERATE, reviewed exceptions --
- ``EnsurePgTableExists``/``ReadTableIntoDic``/``ReadTableIntoDicReversed``/
  ``GetIdByKeyFieldAndInsertIfNeeded``/``MakeSetExcludedClause``: intentionally-kept, thin,
  ``DeprecationWarning``-emitting backward-compat aliases for their snake_case primaries -- they
  will ALWAYS match this check's shape (a PascalCase top-level def next to snake_case siblings) by
  design, since removing them is the actual backward-compat break this test elsewhere protects
  against (see ``test_api_stability.py``).
- ``web.browser.LoginAndGetCookies``: reviewed and deliberately NOT renamed -- unlike the cases
  above, its parameters are already fully modern/snake_case (``default_headers``,
  ``seconds_to_sleep_on_error``, ...), so there's no Hungarian-notation TypeError-from-keyword
  risk, only the function name's capitalization. A safe rename would additionally require
  threading a new name through a SELF-RECURSIVE call site (its own retry logic calls itself) and
  updating two test files' ``monkeypatch.setattr(browser_mod, "LoginAndGetCookies", ...)`` targets
  -- a disproportionate amount of churn/risk for a purely cosmetic inconsistency with no
  functional footgun, unlike the Hungarian-notation cases this check exists to catch.
A NEW module hitting this shape needs the SAME review (is this a deliberate, low-risk backward-
compat shim, or a genuine oversight?) before being grandfathered in.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_naming_convention_baseline.json"

_SNAKE_CASE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _refresh_requested() -> bool:
    return "--refresh-naming-convention-baseline" in sys.argv


def _build_findings() -> set[str]:
    out: set[str] = set()
    for py in _SRC_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rel = py.relative_to(_REPO_ROOT).as_posix()
        names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")]
        if not names:
            continue
        matching = [n for n in names if _SNAKE_CASE_RE.match(n)]
        nonmatching = [n for n in names if not _SNAKE_CASE_RE.match(n)]
        if matching and nonmatching:
            out.update(f"{rel} {n}" for n in nonmatching)
    return out


def test_no_new_naming_convention_inconsistency():
    current = _build_findings()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(json.dumps(sorted(current), indent=2), encoding="utf-8")
        pytest.skip(f"naming-convention baseline refreshed at {_BASELINE_PATH.name} ({len(current)} finding(s))")

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_naming_convention_inconsistency] {len(fixed)} site(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-naming-convention-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} non-snake_case public function(s) mixed into an otherwise snake_case "
            f"module. Either rename to snake_case (keeping the old name as a deprecated, "
            f"warning-emitting alias if it's an established public API -- see "
            f"database.db.sql_helpers.make_set_excluded_clause), OR if it's a reviewed, "
            f"low-risk exception, refresh the baseline after documenting why:\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
