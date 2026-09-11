"""F1 + F2 + F3 — meta-tests on the meta-test suite itself.

When the suite grows past 10 files, the suite ITSELF becomes a piece of
production code worth policing. These tests catch:

  F1. Failure messages without actionable detail, via ``py_ci_shared.fail_message_quality``.
      ``pytest.fail("broken")`` is useless; a message must name a fix verb (``Add``, ``Either``, ``Refresh``,
      ``Remove``, ...) or carry a ``<placeholder>``. A colon or a path no longer counts: every static message
      contains one, so the looser rule passed all of them without reading a word.

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
from pathlib import Path

import pytest
from py_ci_shared.fail_message_quality import assert_fail_messages_actionable

_TEST_META_DIR = Path(__file__).resolve().parent

# Imports of a production private symbol from a meta-test that are
# legitimate. Each entry is "test_meta_filename::imported_dotted_name".
_PERMITTED_PRIVATE_IMPORTS: set[str] = {
    "test_lazy_import_safety::pyutilz._create_lazy_module",
    "test_lazy_import_safety::pyutilz._MODULE_ALIASES",
    "test_module_alias_integrity::pyutilz._MODULE_ALIASES",
    "test_provider_registration::pyutilz.llm.factory._PROVIDER_MODULES",
    # The facade-integrity check asks whether the FACADE agrees with the registry, so the private
    # registry is the fact under test -- reading its public mirror instead would compare the facade to itself.
    "test_facade_and_exception_root_integrity::pyutilz.llm.factory._PROVIDER_MODULES",
    # The README/docs claim "across N providers"; the registry is what N counts, so the
    # check must read the registry itself rather than a public mirror that could drift with it.
    "test_prose_numeric_claims::pyutilz.llm.factory._PROVIDER_MODULES",
    "test_provider_registration::pyutilz.llm.factory._ALIASES",
    "test_provider_cache_concurrency::pyutilz.llm.factory._provider_cache",
    "test_provider_cache_concurrency::pyutilz.llm.factory._provider_lock",
    "test_provider_cache_concurrency::pyutilz.llm.factory._PROVIDER_MODULES",
    "test_provider_cache_concurrency::pyutilz.llm.factory._ALIASES",
    "test_retry_predicate_matches_sdk_hierarchy::pyutilz.llm.gemini_provider._is_retryable_genai_error",
}


def _meta_test_files() -> list[Path]:
    out: list[Path] = []
    for py in _TEST_META_DIR.glob("test_*.py"):
        if py.name == Path(__file__).name:
            continue
        out.append(py)
    return sorted(out)


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
    """F1: every static ``pytest.fail(...)`` message in the meta-test directory names a fix verb or a placeholder."""
    assert_fail_messages_actionable(_TEST_META_DIR, exclude=(Path(__file__).name,), min_audited=10)


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
            f"whitelist via _PERMITTED_PRIVATE_IMPORTS with reasoning:\n  " + "\n  ".join(sorted(set(bad)))
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
        pytest.fail(f"_PERF_BUDGET_OVERRIDES has entries for {stale} which no " f"longer exist in the meta-test dir — clean up after rename")
