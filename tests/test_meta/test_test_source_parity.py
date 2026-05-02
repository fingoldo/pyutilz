"""PT-3 — meta-test: every non-trivial production module under
``src/pyutilz/`` should have at least one corresponding test file in
``tests/``.

Catches the failure mode where a new module is added to the package
without a test file. The audit is intentionally NAME-based — matching
``pyutilz/<sub>/<module>.py`` against any of:

  - ``tests/test_<module>.py``                (the convention pyutilz uses)
  - ``tests/test_<module>_extra.py``          (split-out coverage tests)
  - ``tests/test_<module>_extra2.py``         (further splits)
  - ``tests/<sub>/test_<module>.py``          (sub-package tests, future-proof)
  - ``tests/test_<sub>_<module>.py``          (collapsed-prefix variant)
  - ``tests/test_<alias>.py``                 (where alias is in
                                               ``_MODULE_ALIASES``)

Modules under ``__init__.py`` / ``version.py`` / ``_<name>.py`` (private
helpers) are exempt — those don't need standalone tests.

The reverse direction (test_X.py with no corresponding source) is also
surfaced as a separate test, with a smaller alarm bell — sometimes
tests are intentionally cross-cutting (``test_serialization.py`` covers
``core/serialization.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pyutilz
from pyutilz.dev.meta_test_utils import enumerate_test_files

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
REPO_ROOT = PYUTILZ_DIR.parent.parent  # src/pyutilz/ → repo root
TESTS_DIR = REPO_ROOT / "tests"

# Modules that don't need a dedicated test file. Cite reason.
_TEST_EXEMPT_MODULES: dict[str, str] = {
    "version": "single-line constant; no behaviour to test",
    "config": "module-level constants/literals only — no functions",
    # Sub-package boundary modules are pure namespaces; behaviour lives
    # in the modules under them.
    "cloud": "namespace stub with one helper covered indirectly",
    "system": "namespace; behaviour split across distributed/parallel/monitoring",
    "web": "namespace; behaviour in browser/proxy/web sub-modules",
    # LLM provider modules are covered jointly by tests/test_llm_providers.py
    # which exercises every provider through a shared parametrize matrix.
    "anthropic_provider": "covered by test_llm_providers.py + test_llm_base.py",
    "claude_code_provider": "covered by test_llm_providers.py",
    "deepseek_provider": "covered by test_llm_deepseek.py + test_llm_providers.py",
    "gemini_provider": "covered by test_llm_providers.py",
    "openai_provider": "covered by test_llm_providers.py",
    "openrouter_provider": "covered by test_llm_providers.py + test_llm_account_credits.py",
    "xai_provider": "covered by test_llm_xai.py + test_llm_providers.py",
    "exceptions": "trivial exception class definitions; no logic to test",
    "openai_compat": "covered by test_llm_openai_compat.py",
    "meta_test_utils": "tooling for the meta-test suite — exercised by every meta-test that imports it",
    # Modules covered jointly by ``tests/test_smoke_untested_modules.py``
    # (parametrized smoke suite). Each will graduate to a dedicated
    # test file when behavioural coverage is added.
    "openai": "covered by test_smoke_untested_modules.py (smoke parametrize)",
    "filemaker": "covered by test_smoke_untested_modules.py",
    "db": "covered by test_smoke_untested_modules.py",
    "deltalakes": "covered by test_smoke_untested_modules.py",
    "redislib": "covered by test_smoke_untested_modules.py",
    "dashlib": "covered by test_smoke_untested_modules.py",
    "notebook_init": "covered by test_smoke_untested_modules.py",
    "tokenizers": "covered by test_smoke_untested_modules.py",
}

# Test files (by stem — no .py) that don't have a 1:1 source counterpart
# but cover a real concern. Reverse-direction whitelist.
_TEST_FILES_WITHOUT_SOURCE: dict[str, str] = {
    "test_llm_factory": "covers pyutilz.llm.factory",
    "test_llm_base": "covers pyutilz.llm.base",
    "test_llm_providers": "cross-cutting — covers ALL provider modules together",
    "test_llm_openai_compat": "covers pyutilz.llm.openai_compat",
    "test_llm_deepseek": "covers pyutilz.llm.deepseek_provider",
    "test_llm_xai": "covers pyutilz.llm.xai_provider",
    "test_llm_retry": "covers pyutilz.llm._retry (private module)",
    "test_hardware_detection": "covers pyutilz.system.hardware_monitor",
    "test_proxy": "covers pyutilz.web.proxy/* sub-package",
    # Meta-tests test infrastructure, not a single source module.
    "test_provider_registration": "meta-test (PT-1)",
    "test_module_alias_integrity": "meta-test (PT-2)",
    "test_test_source_parity": "meta-test (PT-3) — this file",
    "test_todo_hygiene": "meta-test (PT-4)",
    "test_dead_helpers": "meta-test (PT-5)",
    "test_api_stability": "meta-test (PT-6)",
    "test_lazy_import_safety": "meta-test (PT-7)",
    "test_optional_deps_isolation": "meta-test (PT-8)",
    "test_no_top_level_side_effects": "meta-test (PT-9)",
    "test_deferred_drift": "meta-test (A2) — drift tracker for deferred lists",
    "test_provider_contract": "meta-test (D1) — provider interface contract",
    "test_encoding_consistency": "meta-test (D2) — open() encoding kwargs",
    "test_provider_cache_concurrency": "meta-test (D3) — thread safety",
    "test_public_docstrings": "meta-test (E1) — docstring coverage snapshot",
    "test_public_annotations": "meta-test (E2) — type annotation snapshot",
    "test_version_consistency": "meta-test (E3) — version source-of-truth parity",
    "test_no_import_cycles": "meta-test (E4) — import cycle detector",
    "test_no_unicode_in_console_output": "meta-test (E5) — non-ASCII console output",
    "test_meta_meta": "meta-test (F1+F2+F3) — meta-tests on meta-tests",
    "test_logger_lazy_formatting": "meta-test (H1) — eager-format logger.debug/info",
    "test_resource_handle_safety": "meta-test (H2) — open/Popen outside with-block",
    "test_smoke_untested_modules": "smoke suite — covers 8 untested modules under one parametrize",
    "test_no_bare_except": "meta-test (H3) — bare except / except BaseException without re-raise",
    "test_no_mutable_defaults": "meta-test (H4) — mutable default arguments (Python footgun)",
    "test_llm_account_credits": "cross-cutting — covers OpenRouter check_account_limits across providers",
    "test_llm_supports_json_mode": "cross-cutting — pins supports_json_mode() across all providers",
}

# Source modules surfaced by the forward-direction audit but explicitly
# deferred — maintainer is aware, will add tests later. Drain over time.
# All 8 modules surfaced 2026-04-28 are now covered by the smoke
# suite at ``tests/test_smoke_untested_modules.py`` (parametrized
# import-and-public-surface check). Each will graduate to a dedicated
# ``test_X.py`` over time when behavioural coverage is added.
_USER_DEFERRED_MODULES: dict[str, str] = {}


def _production_modules() -> list[Path]:
    """Every .py under src/pyutilz/ excluding __init__.py / __pycache__ /
    private modules / .old backups."""
    out: list[Path] = []
    for py in PYUTILZ_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if py.name in {"__init__.py", "__main__.py"}:
            continue
        if py.name.startswith("_"):
            continue  # Private modules don't need standalone tests.
        if py.suffix != ".py":
            continue
        if py.name.endswith(".py.old"):
            continue
        out.append(py)
    return out


def _test_basenames() -> set[str]:
    """All test_*.py basenames in the tests/ tree (recursive)."""
    return enumerate_test_files(TESTS_DIR)


def _candidate_test_names(source: Path) -> list[str]:
    """All plausible test-file names that could test ``source``."""
    name = source.stem
    rel = source.relative_to(PYUTILZ_DIR)
    parts = rel.parts
    sub = parts[0] if len(parts) > 1 else None

    cands = [
        f"test_{name}",
        f"test_{name}_extra",
        f"test_{name}_extra2",
    ]
    if sub:
        cands.append(f"test_{sub}_{name}")
        cands.append(f"test_{sub}")

    # Aliases — if this module is exposed via _MODULE_ALIASES, the test
    # may be named after the alias instead of the canonical path.
    for alias, real_path in pyutilz._MODULE_ALIASES.items():
        if real_path.endswith(f".{name}"):
            cands.append(f"test_{alias}")
    return cands


def test_every_production_module_has_a_test_file():
    test_names = _test_basenames()
    if not test_names:
        pytest.skip("no test files found — repo layout broken?")

    untested: list[str] = []
    audited = 0
    for source in _production_modules():
        rel = source.relative_to(PYUTILZ_DIR).as_posix()
        if source.stem in _TEST_EXEMPT_MODULES:
            continue
        if rel in _USER_DEFERRED_MODULES:
            continue
        audited += 1
        cands = _candidate_test_names(source)
        if any(c in test_names for c in cands):
            continue
        untested.append(f"{rel} (candidates: {', '.join(cands[:4])})")

    assert audited > 10, f"only {audited} modules audited — discovery broken?"
    if untested:
        pytest.fail(
            f"{len(untested)} production module(s) without a corresponding "
            f"test file. Either add ``tests/test_<name>.py``, OR document "
            f"the exemption in _TEST_EXEMPT_MODULES with reasoning:\n  "
            + "\n  ".join(untested)
        )


def test_every_test_file_has_a_target():
    """Reverse direction — flag test files whose name doesn't correspond
    to a real production module. Looser test (warning-only at present —
    intentional, covered cases are real)."""
    sources_by_stem = {p.stem: p for p in _production_modules()}
    aliases = set(pyutilz._MODULE_ALIASES.keys())
    orphan: list[str] = []
    for stem in sorted(_test_basenames()):
        if stem in _TEST_FILES_WITHOUT_SOURCE:
            continue
        if not stem.startswith("test_"):
            continue
        # Strip the leading "test_" and any "_extra" / "_extra2" suffix.
        name = stem[len("test_"):]
        for suffix in ("_extra2", "_extra"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        if name in sources_by_stem:
            continue
        if name in aliases:
            continue
        # Sub-prefix collapse: ``test_pyutilz_pythonlib`` → ``pythonlib``;
        # also try the LAST segment after the last underscore to catch
        # ``test_llm_factory`` → ``factory``.
        parts = name.split("_")
        if any(p in sources_by_stem for p in parts):
            continue
        # Try suffix-style match: source ``deepseek_provider``,
        # test ``test_llm_deepseek`` — match if any source has a
        # name containing one of the parts.
        if any(p and any(p in s for s in sources_by_stem) for p in parts):
            continue
        orphan.append(f"tests/{stem}.py")

    if orphan:
        pytest.fail(
            f"{len(orphan)} test file(s) without an apparent target module "
            f"(or alias). Either add the target, OR document the cross-"
            f"cutting test in _TEST_FILES_WITHOUT_SOURCE:\n  "
            + "\n  ".join(orphan)
        )
