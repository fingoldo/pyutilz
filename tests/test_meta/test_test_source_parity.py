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
  - ``tests/test_<sub>.py``                   (ONLY for the sub-package's
                                               namesake module, e.g.
                                               ``system/system.py``)
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
# A key is either a bare module stem (exempts every module with that name) or a
# path relative to ``src/pyutilz/`` (exempts exactly that one module) -- use the
# path form whenever the stem is generic enough to collide, e.g. ``base``.
_TEST_EXEMPT_MODULES: dict[str, str] = {
    "version": "single-line constant; no behaviour to test",
    "benchmark": "performance/kernel_tuning/benchmark.py — covered by tests/test_kernel_tuning_benchmark.py, renamed 2026-07-21 to stop being confusable with test_dev_benchmarking.py (pyutilz.dev.benchmarking)",
    "config": "module-level constants/literals only — no functions",
    # Sub-package boundary modules are pure namespaces; behaviour lives
    # in the modules under them.
    "cloud": "namespace stub with one helper covered indirectly",
    "system": "namespace; behaviour split across distributed/parallel/monitoring",
    "web": "namespace; behaviour in browser/proxy/web sub-modules",
    # LLM provider modules are covered jointly by tests/test_llm_providers.py
    # which exercises every provider through a shared parametrize matrix.
    "claude_code_provider": "covered by test_llm_providers.py",
    "gemini_provider": "covered by test_llm_providers.py",
    "openai_provider": "covered by test_llm_providers.py",
    "meta_test_utils": "tooling for the meta-test suite — exercised by every meta-test that imports it",
    # Modules covered jointly by ``tests/test_smoke_untested_modules.py``
    # (parametrized smoke suite). Each will graduate to a dedicated
    # test file when behavioural coverage is added.
    "filemaker": "covered by test_smoke_untested_modules.py",
    "db": "covered by test_smoke_untested_modules.py",
    "deltalakes": "covered by test_smoke_untested_modules.py",
    "redislib": "covered by test_smoke_untested_modules.py",
    "dashlib": "covered by test_smoke_untested_modules.py",
    "notebook_init": "covered by test_smoke_untested_modules.py",
    # dev/code_audit was split from a single >1000-LOC module into a
    # subpackage; the tests were split the same way, into the
    # tests/code_audit/ package of one test_<family>.py per scanner
    # family. The positive+negative-case-per-scanner discipline these
    # exemptions rest on is no longer prose: it is enforced for every
    # registered scanner by
    # tests/test_meta/test_scanner_positive_and_negative_cases.py.
    "packages": "core/pythonlib subpackage split; covered jointly by test_pythonlib*.py",
    "objects": "core/pythonlib subpackage split; covered jointly by test_pythonlib*.py",
    "filesystem": "core/pythonlib subpackage split; covered jointly by test_pythonlib*.py",
    "hardware": "core/pythonlib subpackage split; covered by test_pythonlib*.py + test_cuda_home_autodetect.py",
    "columns": "data/polarslib subpackage split; covered jointly by test_polarslib*.py",
    "aggregations": "data/polarslib subpackage split; covered jointly by test_polarslib*.py",
    "binning": "data/polarslib subpackage split; covered jointly by test_polarslib*.py",
    "connection": "database/db subpackage split; covered jointly by test_db_extra.py + test_database_extra2.py",
    "execution": "database/db subpackage split; covered jointly by test_db_extra.py + test_database_extra2.py",
    "schema": "database/db subpackage split; covered jointly by test_db_extra.py + test_database_extra2.py",
    "regjobs": "database/db subpackage split; covered jointly by test_db_extra.py",
    "legacy": "database/db subpackage split (PascalCase compatibility wrappers); covered jointly by test_db_extra.py",
    "sentences": "text/similarity subpackage split; covered jointly by test_similarity.py",
    "openai_tokens": "moved out of core/openai.py; covered by test_llm_core_openai.py",
    # database/db was split from a single >1000-LOC module into a subpackage;
    # the pure/stateless helper submodules are exercised jointly by
    # tests/test_db_extra.py (facade re-export sensor + helper behaviour) and
    # tests/test_smoke_untested_modules.py (via the ``db`` facade).
    "sql_helpers": "database/db subpackage; covered by test_db_extra.py + test_smoke_untested_modules.py",
    "upsert": "database/db subpackage; covered by test_db_extra.py + test_smoke_untested_modules.py",
    # data/pandaslib was split from a single >1000-LOC module into a subpackage;
    # every submodule is exercised jointly through the ``pandaslib`` facade by
    # tests/test_pandaslib.py (+ test_pandaslib_extra.py / test_pandaslib_extra2.py).
    "dtypes": "data/pandaslib subpackage; covered by test_pandaslib.py",
    "frames": "data/pandaslib subpackage; covered by test_pandaslib.py",
    "io_ops": "data/pandaslib subpackage; covered by test_pandaslib.py",
    "benchmarks": "data/pandaslib subpackage; covered by test_pandaslib.py",
    # performance/kernel_tuning/cache was split from a single >1000-LOC module
    # into a subpackage; every submodule is exercised jointly through the
    # ``cache`` facade by tests/performance/kernel_tuning/* (test_cache.py,
    # test_cache_concurrency.py, test_cache_facade.py re-export sensor).
    "cache_base": "kernel_tuning/cache subpackage; covered by tests/performance/kernel_tuning/*",
    "cache_class": "kernel_tuning/cache subpackage; covered by tests/performance/kernel_tuning/*",
    "cache_hooks": "kernel_tuning/cache subpackage; covered by tests/performance/kernel_tuning/*",
    "cache_persistence": "kernel_tuning/cache subpackage; covered by tests/performance/kernel_tuning/*",
    "cache_sweeping": "kernel_tuning/cache subpackage; covered by tests/performance/kernel_tuning/*",
    "cache_tuning": "kernel_tuning/cache subpackage; covered by tests/performance/kernel_tuning/*",
    # text/strings was split from a single >1000-LOC module into a subpackage;
    # every submodule is exercised jointly through the ``strings`` facade by
    # tests/test_strings*.py.
    "jsonutils": "text/strings subpackage; covered by test_strings.py",
    # Path-keyed exemptions: modules that used to satisfy the gate only through
    # the (now removed) bare ``test_<sub>`` fallback -- ``tests/test_system.py`` /
    # ``tests/test_web.py`` waved through EVERY module under system/ and web/.
    # Each is really covered by the cited file(s), verified by word-boundary
    # references to the module's own public names.
    "system/hardware_monitor.py": "covered by tests/system/test_hardware_detection.py (get_hardware_info) + test_system_audit_fixes.py",
    "system/scheduling/prefect.py": "covered by test_system_audit_fixes.py (get_running_flows / wait_for_absense_of_tasks / get_schema)",
    "system/system/fsutils.py": "system/system subpackage split; covered by test_system.py + test_system_extra2.py + test_system_domain_extra.py",
    "system/system/memory.py": "system/system subpackage split; covered by test_system.py + test_system_extra2.py + test_system_extra3.py",
    "system/system/misc.py": "system/system subpackage split; covered by test_system.py + test_system_extra.py + test_system_extra2.py",
    "system/system/probing.py": "system/system subpackage split; covered by test_system_extra2.py + test_system_extra3.py + tests/system/test_hardware_detection.py",
    "system/system/sysinfo.py": "system/system subpackage split; get_system_info covered by test_system.py + test_dev_fixes_regression.py + test_distributed.py",
    "web/proxy/base.py": "web/proxy subpackage; covered by test_proxy.py + test_domain_db_web_cloud_llm_text_audit_20260903_web.py",
    "web/proxy/decodo.py": "web/proxy subpackage; covered by test_proxy.py (TestDecodoProvider / TestDecodoSubscription / TestDecodoTrafficReport)",
    "web/proxy/ip_check.py": "web/proxy subpackage; covered by test_proxy.py + test_domain_db_web_cloud_llm_text_audit_20260903_web.py",
    "web/proxy/session.py": "web/proxy subpackage; covered by test_proxy.py + test_domain_db_web_cloud_llm_text_audit_20260903_web.py",
    "web/web/downloads.py": "web/web subpackage split; covered by test_web_extra.py + test_domain_db_web_cloud_llm_text_audit_20260903_web.py",
    "web/web/fetching.py": "web/web subpackage split; covered by test_web_extra.py + test_web_concurrency.py",
    "web/web/ipinfo.py": "web/web subpackage split; covered by test_web.py + test_web_extra.py",
    "web/web/proxy_pool.py": "web/web subpackage split; covered by test_web_extra.py + test_web.py",
}

# Test files (by stem — no .py) that don't have a 1:1 source counterpart
# but cover a real concern. Reverse-direction whitelist.
_TEST_FILES_WITHOUT_SOURCE: dict[str, str] = {
    "test_audit_20261015_runtime": "cross-cutting regression tests for the 2026-10-15 audit findings C-01, C-02, C-03, C-07, C-08, C-09, R-01, R-02, R-03, S-10, X-07, X-18, P-05 over the database, config, cache and resilience modules",
    "test_printed_advice_messages": "cross-cutting: the refusal messages that advise a keyword (load_file, _alternate_values_clause, build_upsert_query, _normalise_encryption), each pinned by the printed-advice gate",
    "test_audit_20261015_unassigned": "regression tests for the 2026-10-15 audit findings C-04 (system.parallel mmap unlink) and R-04 (pandaslib benchmark temp folder)",
    "test_pricing_tiers_20261003": "cross-cutting: pyutilz.llm._pricing (cache-write rates, long-context tiers) through the xAI, Gemini, OpenAI, Anthropic, DeepSeek and OpenRouter providers",
    "test_providers_live_20261003": "cross-cutting regression tests from the 2026-10-03 live verification of the PROV-* fixes, replaying recorded responses through five providers",
    "test_general_audit_20260926": "cross-cutting regression tests for audits/implemented/2026-09-26/50_general.md (GEN-1..15) over nine modules; one class per finding",
    "test_shared_gates_adopted": "meta-test running py_ci_shared gates over src/ and tests/ -- no single production module to pair with",
    "test_mypy_gate_floor": "meta-test over .pre-commit-config.yaml and mypy-full.yml -- the mypy_gate floor tracks the source-file count; no production module to pair with",
    "test_audit_rounds_countable": "meta-test over audits/ -- every finding has a disposition and rounds are filed by state, via py_ci_shared.audit_round_format; no production module to pair with",
    "test_gate_integrity": "meta-test over .pre-commit-config.yaml / .github/workflows / pyproject.toml -- no production module to pair with",
    "test_optional_dep_preconditions": "meta-test over the TEST suite itself -- pins that a test states the optional extra it needs; no production module to pair with",
    "test_markers_registered": "meta-test over the TEST suite and pyproject -- every pytest marker used is registered, via py_ci_shared.pytest_markers; no production module to pair with",
    "test_llm_factory": "covers pyutilz.llm.factory",
    "test_llm_base": "covers pyutilz.llm.base",
    "test_llm_providers": "cross-cutting — covers ALL provider modules together",
    "test_llm_openai_compat": "covers pyutilz.llm.openai_compat",
    "test_llm_retry": "covers pyutilz.llm._retry (private module)",
    "test_llm_live": "live LLM-provider smoke tests (opt-in via --run-live, cross-cutting)",
    "test_sibling_api_parity": "cross-cutting — covers pyutilz.data.pandaslib and pyutilz.data.polarslib API/mutation-contract parity together",
    "test_retry_predicate_matches_sdk_hierarchy": "cross-cutting — covers retry predicates in llm.anthropic_provider/gemini_provider and database.redislib together",
    "test_hardware_detection": "covers pyutilz.system.hardware_monitor",
    "test_proxy": "covers pyutilz.web.proxy/* sub-package",
    "test_dev_fixes_regression": "cross-cutting — regression tests spanning dev/logginglib, dev/notebook_init, dev/meta_test_utils, system/monitoring, system/hardware_monitor, system/distributed",
    "test_llm_provider_audit_fixes": "cross-cutting — regression tests for the llm-providers quality audit (spans anthropic/gemini/openai-compat/claude-code provider modules)",
    "test_strings_facade": "sensor test for the text/strings subpackage split (facade re-export identity)",
    "test_core_audit_fixes": "cross-cutting - 2026-09-02 audit regression tests spanning core/pythonlib, core/serialization, core/matrix",
    "test_dev_audit_fixes": "cross-cutting - 2026-09-02 audit regression tests spanning dev/logginglib and dev/benchmarking",
    "test_system_audit_fixes": "cross-cutting - 2026-09-02 audit regression tests spanning system/config, system/parallel, system/resilience, system/distributed, system/single_flight_cache, system/hardware_monitor, system/scheduling/prefect",
    "test_data_domain_extra": "cross-cutting — 2026-07-21 audit regression tests spanning data/polarslib, data/pandaslib, data/numpylib, data/numbalib, stats/normality",
    "test_database_extra2": "cross-cutting — 2026-07-21 audit regression tests spanning database/db/sql_helpers, database/db/upsert, database/db/sqlite, database/redislib",
    "test_llm_domain_extra": "cross-cutting — 2026-07-21 audit regression tests spanning llm/gemini_provider, llm/anthropic_provider, llm/openai_provider, llm/base, llm/token_counter",
    "test_system_domain_extra": "cross-cutting — 2026-07-21 audit regression tests spanning system/parallel, system/system/fsutils, system/system/misc, system/config, performance/kernel_tuning/benchmark",
    "test_complexity_ratchet": "cross-cutting - ratchets ruff C901 complexity findings over src/, tests/ and scripts/",
    # system.py was split into the system/system/ subpackage; test_system*.py cover it jointly.
    "test_system": "covers pyutilz.system.system/* sub-package",
    "test_system_extra": "covers pyutilz.system.system/* sub-package",
    "test_system_extra2": "covers pyutilz.system.system/* sub-package",
    "test_system_extra3": "covers pyutilz.system.system/* sub-package",
    "test_architecture_audit_20260902": "cross-cutting - architecture-audit regression tests spanning the package facade, llm/database/web exception hierarchies, core/openai relocation, text/strings star-export and dev/code_audit registry encapsulation",
    "test_architecture_audit_20260903": "cross-cutting - 2026-09-03 architecture-audit regression tests spanning subpackage attribute resolution, domain exception roots, provider pricing tuples and the code_audit test-file predicate",
    "test_similarity_kernel_twin_parity": "cross-cutting - pins the sequential/prange numba twins in text/similarity/_numba_kernels against each other",
    "test_similarity_coverage_gate_batch_kernel": "covers the batched with-matches kernel path in pyutilz.text.similarity",
    "test_performance_audit_20260902": "cross-cutting - performance-audit regression tests",
    "test_security_audit_20260902": "cross-cutting - security-audit regression tests",
    "test_security_audit_20260903": "cross-cutting - 2026-09-03 security-audit regression tests spanning web/browser, web/web/fetching and dev/logginglib",
    # Meta-tests test infrastructure, not a single source module.
    "test_provider_registration": "meta-test (PT-1)",
    "test_module_alias_integrity": "meta-test (PT-2)",
    "test_test_source_parity": "meta-test (PT-3) — this file",
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
    "test_meta_meta": "meta-test (F1+F2+F3) — meta-tests on meta-tests",
    "test_logger_lazy_formatting": "meta-test (H1) — eager-format logger.debug/info",
    "test_resource_handle_safety": "meta-test (H2) — open/Popen outside with-block",
    "test_unused_parameter_baseline": "meta-test - ruff ARG001/ARG002 snapshot; the rule is ruff's, not one source module's",
    "test_smoke_untested_modules": "smoke suite — covers 8 untested modules under one parametrize",
    "test_llm_account_credits": "cross-cutting — covers OpenRouter check_account_limits across providers",
    "test_llm_supports_json_mode": "cross-cutting — pins supports_json_mode() across all providers",
    "test_publish_workflow_version_check": "CI-process test — covers .github/workflows/publish.yml's tag-vs-version check, not a source module",
    "test_verbose_param_consistency": "cross-cutting — pins `verbose: bool` annotation across pandaslib.dtypes/frames and polarslib's truthiness-only verbose family",
    "test_no_module_reload": "meta-test — bans importlib.reload() outside a reviewed whitelist (module-identity-corruption guard)",
    "test_cache_bounded_or_invalidatable": "meta-test — flags unbounded/TTL-less @lru_cache functions across the whole src tree",
    "test_dict_cache_has_eviction": "meta-test — flags unbounded module-level dict/OrderedDict caches across the whole src tree",
    "test_exception_logging_preserves_traceback": "meta-test — flags logger.exception() calls outside an active except handler across the whole src tree",
    "test_thread_daemon_or_lifecycle_managed": "meta-test — flags non-daemon, unjoined, unmanaged threading.Thread() creations across the whole src tree",
    "test_no_reiterated_iterable_params": "meta-test — flags Iterable[...] params consumed more than once without materializing, across the whole src tree",
    "test_lock_discipline_consistency": "meta-test — flags globals/self.attrs assigned both under and outside their own lock, across the whole src tree",
    "test_resource_reassignment_without_close": "meta-test — flags resource-holding globals/self.attrs reassigned without closing the previous value, across the whole src tree",
    "test_consistent_param_naming_per_annotated_type": "meta-test — flags a concrete annotated type spelled with different first-param names within one module",
    "test_naming_convention_consistency": "meta-test — flags non-snake_case public functions mixed into an otherwise snake_case module",
    "test_deprecation_warnings_present": "meta-test — flags docstring-only deprecation notices with no warnings.warn() call",
    "test_docs_inventory_parity": "meta-test over the documented inventories (extras descriptions, README tables, audit index) against their sources of truth; no production module to pair with",
    "test_resource_release_gate": "py-ci-shared resource-release gate applied to three reviewed modules; cross-cutting by design",
    "test_shared_checks_wired": "consumption point of the cross-project py-ci-shared checks plus this repository's allowlists; no single production module",
    "test_uncached_constant_cost_probe_warn": "meta-test (warn-only) over the whole src/pyutilz tree for constant-cost work on dispatch paths; cross-cutting",
    "test_uncalled_functions_gate": "meta-test over the whole src/pyutilz tree and its uncalled-functions classification file; cross-cutting",
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
        # ``tests/test_<sub>.py`` only counts for the module that IS the
        # sub-package's namesake (e.g. system/system.py <- test_system.py).
        # Accepting it for every module under the sub-package waved through
        # any module whose own name appeared nowhere in tests/ -- that is how
        # system/psutil_compat.py stayed wholly untested behind test_system.py.
        if name == sub:
            cands.append(f"test_{sub}")

    # Aliases — if this module is exposed via _MODULE_ALIASES, the test
    # may be named after the alias instead of the canonical path.
    for alias, real_path in pyutilz._MODULE_ALIASES.items():
        if real_path.endswith(f".{name}"):
            cands.append(f"test_{alias}")
    return cands


def _is_moved_alias(source: Path) -> bool:
    """A module whose whole job is to alias a module that moved to py-ci-shared (its text says so)."""
    return "now lives in py_ci_shared." in source.read_text(encoding="utf-8")[:300]


def test_every_production_module_has_a_test_file():
    test_names = _test_basenames()
    if not test_names:
        pytest.skip("no test files found — repo layout broken?")

    untested: list[str] = []
    audited = 0
    for source in _production_modules():
        rel = source.relative_to(PYUTILZ_DIR).as_posix()
        if source.stem in _TEST_EXEMPT_MODULES or rel in _TEST_EXEMPT_MODULES:
            continue
        if _is_moved_alias(source):
            continue  # its code and its tests live in py-ci-shared; this file only aliases the old path
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
            f"the exemption in _TEST_EXEMPT_MODULES with reasoning:\n  " + "\n  ".join(untested)
        )


def test_every_test_file_has_a_target():
    """Reverse direction — flag test files whose name doesn't correspond
    to a real production module. Looser test (warning-only at present —
    intentional, covered cases are real)."""
    # A test may be named for a private module (``_retry`` -> ``test_retry``): private modules are matched here, not required.
    private = [py for py in PYUTILZ_DIR.rglob("*.py") if py.name.startswith("_") and py.name != "__init__.py" and "__pycache__" not in py.parts]
    sources_by_stem = {p.stem.lstrip("_"): p for p in [*_production_modules(), *private]}
    aliases = set(pyutilz._MODULE_ALIASES.keys())
    orphan: list[str] = []
    for stem in sorted(_test_basenames()):
        if stem in _TEST_FILES_WITHOUT_SOURCE:
            continue
        if not stem.startswith("test_"):
            continue
        # Strip the leading "test_" and any "_extra" / "_extra2" suffix.
        name = stem[len("test_") :]
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
            f"cutting test in _TEST_FILES_WITHOUT_SOURCE:\n  " + "\n  ".join(orphan)
        )
