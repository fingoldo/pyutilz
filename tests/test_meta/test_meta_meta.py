"""F1 + F2 + F3 — meta-tests on the meta-test suite itself.

When the suite grows past 10 files, the suite ITSELF becomes a piece of
production code worth policing. These tests catch:

  F1. Failure messages without actionable detail.
      ``pytest.fail("broken")`` is useless; ``pytest.fail(f"{n} fields
      have no consumer:\n  {names}")`` lets the reviewer act. Audit
      every ``pytest.fail`` call in the meta-test directory; each must
      contain at least one of: a colon, a path separator, an angle-
      bracket placeholder, or a fix-prompt verb (``Add``, ``Either``,
      ``Refresh``, ``Whitelist``).

  F2. Meta-tests reaching into private internals of the code they
      police. The whole point of a meta-test is to cover the public
      contract — if the test imports ``_foo`` from a production module,
      it's testing implementation, not behaviour. Whitelist via
      ``_PERMITTED_PRIVATE_IMPORTS`` for legitimate cases (e.g. the
      lazy-proxy meta-test must touch ``_create_lazy_module`` because
      that IS the surface under test).

  F3. Per-test wall-clock budget. Meta-tests are designed to run in
      seconds — anything > 30 s is a yellow flag (likely accidentally
      doing work that should live in an integration test). Currently
      a soft warning emitted to stderr; PT-8/PT-9 sub-process tests
      and PT-2 alias resolution exceed this and are whitelisted.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

_TEST_META_DIR = Path(__file__).resolve().parent

# Words / characters that indicate an actionable failure message.
# Match a colon (file:line, key: value), a slash (path), an angle (template
# placeholder), or any of the fix-prompt verbs.
_ACTIONABLE_RE = re.compile(
    r"[:/<>]|\b(Add|Either|Refresh|Whitelist|Fix|Run|Update|Remove|Document|"
    r"Check|See|Catches|Replace|OR)\b",
    re.IGNORECASE,
)

# Imports of a production private symbol from a meta-test that are
# legitimate. Each entry is "test_meta_filename::imported_dotted_name".
_PERMITTED_PRIVATE_IMPORTS: set[str] = {
    "test_lazy_import_safety::pyutilz._create_lazy_module",
    "test_lazy_import_safety::pyutilz._MODULE_ALIASES",
    "test_module_alias_integrity::pyutilz._MODULE_ALIASES",
    "test_provider_registration::pyutilz.llm.factory._PROVIDER_MODULES",
    "test_provider_registration::pyutilz.llm.factory._ALIASES",
    "test_provider_cache_concurrency::pyutilz.llm.factory._provider_cache",
    "test_provider_cache_concurrency::pyutilz.llm.factory._provider_lock",
    "test_provider_cache_concurrency::pyutilz.llm.factory._PROVIDER_MODULES",
    "test_provider_cache_concurrency::pyutilz.llm.factory._ALIASES",
}


def _meta_test_files() -> list[Path]:
    out: list[Path] = []
    for py in _TEST_META_DIR.glob("test_*.py"):
        if py.name == Path(__file__).name:
            continue
        out.append(py)
    return sorted(out)


def _pytest_fail_strings(tree: ast.AST) -> list[tuple[int, str, bool]]:
    """Yield ``(lineno, joined_static_text, has_dynamic)`` for every
    ``pytest.fail(...)`` call.

    ``joined_static_text`` concatenates every ``ast.Constant`` string
    chunk of the first arg, and ``has_dynamic`` is True when the message
    is built from an f-string / ``%``-format / ``+`` concat with a
    non-constant operand — the rich detail then comes from the
    dynamic substitution and the joined statics alone won't reflect
    that.
    """
    out: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "fail"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "pytest"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        chunks: list[str] = []
        has_dynamic = False
        # ``pytest.fail(msg)`` where ``msg`` is a bare Name / Attribute /
        # Subscript / Call: the message is built entirely outside this
        # expression — definitively dynamic.
        if not isinstance(first, ast.Constant):
            has_dynamic = True
        for sub in ast.walk(first):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                chunks.append(sub.value)
            elif isinstance(sub, ast.FormattedValue):
                has_dynamic = True
            elif isinstance(sub, (ast.Name, ast.Attribute, ast.Subscript,
                                  ast.Call)) and sub is not first:
                # A non-string-constant inside the message expression.
                has_dynamic = True
        out.append((node.lineno, " ".join(chunks), has_dynamic))
    return out


def _imports(tree: ast.AST) -> list[str]:
    """Yield fully-qualified imported names from ``import X`` and
    ``from X import Y`` (where Y joins the dotted base)."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for alias in node.names:
                if base:
                    out.append(f"{base}.{alias.name}")
                else:
                    out.append(alias.name)
    return out


# ---------------------------------------------------------------------------
# F1 — actionable failure messages
# ---------------------------------------------------------------------------


def test_every_pytest_fail_call_has_actionable_text():
    bad: list[str] = []
    audited = 0
    for py in _meta_test_files():
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for lineno, text, has_dynamic in _pytest_fail_strings(tree):
            audited += 1
            # A dynamic message (f-string, "...".join(parts), etc.) is
            # presumed actionable — the rich content lives in the
            # runtime substitutions which the static walker can't see.
            if has_dynamic:
                continue
            if not text:
                bad.append(f"{py.name}:{lineno} (empty message)")
                continue
            if not _ACTIONABLE_RE.search(text):
                bad.append(
                    f"{py.name}:{lineno} → {text[:80]!r}"
                )

    if audited == 0:
        pytest.skip("no pytest.fail calls found in meta-test directory")
    if bad:
        pytest.fail(
            f"{len(bad)} pytest.fail message(s) lack actionable detail "
            f"(file paths, fix verbs, or template placeholders). The "
            f"reviewer will need to read the test source to figure out "
            f"what to do — improve the message:\n  "
            + "\n  ".join(bad[:20])
        )


# ---------------------------------------------------------------------------
# F2 — no private internals reached into without justification
# ---------------------------------------------------------------------------


def test_meta_tests_dont_reach_private_internals():
    bad: list[str] = []
    for py in _meta_test_files():
        stem = py.stem
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for imp in _imports(tree):
            # Only audit our own package imports.
            if not (imp.startswith("pyutilz") or imp.startswith("mlframe")):
                continue
            # Last segment with a single leading underscore is "private".
            last = imp.rsplit(".", 1)[-1]
            if not last.startswith("_") or last.startswith("__"):
                continue
            entry = f"{stem}::{imp}"
            if entry in _PERMITTED_PRIVATE_IMPORTS:
                continue
            bad.append(entry)
    if bad:
        pytest.fail(
            f"{len(bad)} meta-test(s) import a private symbol without "
            f"justification. Either use the public API instead, OR "
            f"whitelist via _PERMITTED_PRIVATE_IMPORTS with reasoning:\n  "
            + "\n  ".join(sorted(set(bad)))
        )


# ---------------------------------------------------------------------------
# F3 — per-meta-test wall-clock budget (advisory)
# ---------------------------------------------------------------------------

# Tests permitted above the soft wall-clock budget (in seconds).
_PERF_BUDGET_OVERRIDES: dict[str, float] = {
    # Sub-process based tests — necessarily slower.
    "test_optional_deps_isolation": 30.0,
    "test_no_top_level_side_effects": 60.0,
    # Walks every alias target's module surface — touches optional deps.
    "test_module_alias_integrity": 30.0,
    # API stability captures full surface across all alias targets.
    "test_api_stability": 30.0,
}
_DEFAULT_PERF_BUDGET_S = 10.0


def test_perf_budget_overrides_are_documented():
    """Static check: any test in ``_PERF_BUDGET_OVERRIDES`` corresponds
    to an actual file in the meta-test directory. Catches a stale
    override after a rename.
    """
    test_stems = {p.stem for p in _meta_test_files()}
    stale = [k for k in _PERF_BUDGET_OVERRIDES if k not in test_stems]
    if stale:
        pytest.fail(
            f"_PERF_BUDGET_OVERRIDES has entries for {stale} which no "
            f"longer exist in the meta-test dir — clean up after rename"
        )
