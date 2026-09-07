"""Wire the cross-project checks py-ci-shared already ships but pyutilz never ran.

Every rule below is implemented and unit-tested in ``py_ci_shared``; this module is
purely the consumption point plus this repo's own allowlists. Keeping them in one
file makes the "which shared checks does this repo actually run?" question answerable
by reading a single import block, which is the gap that let seven maintained,
directly-applicable checks sit unused while findings they cover kept landing.

Runtime: file reads plus one TOML parse, well under a second in total. No network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

py_ci_shared = pytest.importorskip("py_ci_shared", reason="py-ci-shared is a dev-only git dependency (requirements-dev.txt)")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The four prose files the repo actually maintains. mkdocs' own ``docs/`` tree is
# deliberately out of scope for the link check: mkdocs resolves relative links against
# the built site, not the repo root, so a valid mkdocs link can look dead to a
# filesystem-relative resolver.
PROSE_FILES = ("README.md", "CONTRIBUTING.md", "TESTING.md", "CHANGELOG.md")

# ``ci-required`` is a pure aggregation job over ``needs`` results; it runs one bash
# comparison and nothing else. It is NOT exempt from the timeout rule -- it carries its
# own ``timeout-minutes`` -- and this set is deliberately empty so that any future job
# without one has to be justified here rather than silently inheriting the 6-hour
# platform default.
_JOBS_EXEMPT_FROM_TIMEOUT: frozenset[str] = frozenset()

# No workflow step in this repo is allowed to be advisory: the advisory lint bundle is
# a SEPARATE workflow (lint-advisory.yml, excluded from ci-required's needs) rather than
# a ``continue-on-error`` step inside a blocking job. An empty allowlist therefore means
# "any continue-on-error appearing in a blocking workflow is a finding".
_REVIEWED_ADVISORY_STEPS: set[str] = set()

# ``py-ci-shared`` is a first-party upstream owned by this repo's own maintainer, so the
# supply-chain threat a commit-SHA pin defends against does not apply: whoever could move
# that ref could push here directly. It is also deliberately kept OUT of [project]
# metadata (a git+https direct reference makes the distribution unpublishable on PyPI) --
# hence the two separate assertions below, with the pyproject one allowing NO exemption at
# all so a git URL reappearing there fails immediately.
_FIRST_PARTY_GIT_PREFIXES = ("git+https://github.com/fingoldo/py-ci-shared",)

# ``tests/test_meta`` and the three domain subdirs are all reached by ci.yml's pathless
# ``pytest -m "not gpu" ...`` run, which collects from rootdir. Nothing is intentionally
# excluded from CI here.
_INTENTIONALLY_UNREACHED_TEST_DIRS: set[str] = set()


def _production_py_files() -> list[Path]:
    """Every shipped module: ``src/`` only, so tests/benchmarks/scripts are out of scope."""
    return sorted((REPO_ROOT / "src").rglob("*.py"))


def _test_py_files() -> list[Path]:
    """Every test module. The module-size rule is about what a human can hold in their head
    while editing, which does not stop at the ``src/`` boundary: ``tests/test_code_audit.py``
    reached 11306 lines while the production side of the same feature was being deliberately
    split into one module per scanner under ``src/pyutilz/dev/code_audit/`` to respect it."""
    return sorted((REPO_ROOT / "tests").rglob("*.py"))


def test_no_new_file_over_1k_loc():
    """CLAUDE.md's module-size rule, enforced instead of remembered.

    ``src/`` and ``tests/`` share one budget and one baseline. The baseline grandfathers the
    four test files already over 1000 lines at their measured size -- reviewed individually,
    each is a flat list of independent test functions with no cheap seam, unlike the scanner
    suite -- so none of them may grow, and no NEW oversized file may appear on either side.

    The captured sizes were re-taken at the END of the 2026-09-03 test-quality wave: three of the
    four grew while that wave was in flight, entirely from adding the missing assertion to tests
    that already existed (the F01 drain), not from new bulk. Growth from here needs a split, not
    another refresh.
    """
    from py_ci_shared.loc_budget import assert_no_new_oversized_file

    assert_no_new_oversized_file(
        files=_production_py_files() + _test_py_files(),
        root=REPO_ROOT,
        baseline_path=Path(__file__).resolve().parent / "_loc_over_1k_baseline.json",
        growth_slack=0,
    )


@pytest.mark.parametrize("workflow_name", sorted(p.name for p in WORKFLOWS_DIR.glob("*.yml")))
def test_every_workflow_job_declares_a_timeout(workflow_name: str):
    """A job with no ``timeout-minutes`` inherits GitHub's 6-hour default."""
    from py_ci_shared.ci_workflow_timeout_gate import assert_all_jobs_have_timeout

    assert_all_jobs_have_timeout(WORKFLOWS_DIR / workflow_name, exempt_jobs=_JOBS_EXEMPT_FROM_TIMEOUT)


@pytest.mark.parametrize("workflow_name", sorted(p.name for p in WORKFLOWS_DIR.glob("*.yml")))
def test_no_undeclared_continue_on_error(workflow_name: str):
    """``continue-on-error: true`` turns a blocking gate into a green no-op."""
    from py_ci_shared.ci_workflow_gate import assert_continue_on_error_is_reviewed

    assert_continue_on_error_is_reviewed(WORKFLOWS_DIR / workflow_name, reviewed_advisory_steps=_REVIEWED_ADVISORY_STEPS)


def test_declared_entry_points_resolve():
    """Every console script / entry point imports and exposes its named attribute."""
    from py_ci_shared.entry_points_resolvable import assert_all_entry_points_resolvable

    assert_all_entry_points_resolvable(PYPROJECT)


def test_no_phantom_markdown_links():
    """Every markdown link in the maintained prose files resolves to a real file."""
    from py_ci_shared.phantom_markdown_links import assert_no_phantom_markdown_links

    assert_no_phantom_markdown_links(md_files=[REPO_ROOT / name for name in PROSE_FILES], repo_root=REPO_ROOT)


def test_pyproject_declares_no_git_dependency_at_all():
    """A git+https direct reference anywhere in [project] makes the sdist/wheel
    unpublishable on PyPI, which is why py-ci-shared lives in requirements-dev.txt
    instead. Allowing no exemption here means a git URL reappearing in project
    metadata fails on the commit that introduces it."""
    from py_ci_shared.git_dependency_pins import assert_all_git_dependencies_pinned

    assert_all_git_dependencies_pinned(PYPROJECT)


def test_dev_requirements_git_dependencies_are_pinned_or_first_party():
    """Any THIRD-PARTY git dependency added to requirements-dev.txt must carry a full
    commit SHA; only the maintainer's own upstreams may float."""
    from py_ci_shared.git_dependency_pins import assert_all_git_dependencies_pinned

    assert_all_git_dependencies_pinned(REPO_ROOT / "requirements-dev.txt", allow_unpinned_url_prefixes=_FIRST_PARTY_GIT_PREFIXES)


def test_every_test_subdir_is_reachable_from_ci():
    """A new tests/<subdir> that no CI job collects gates nothing."""
    from py_ci_shared.ci_test_dir_reachability import assert_every_test_subdir_reachable

    assert_every_test_subdir_reachable(
        repo_root=REPO_ROOT,
        workflows_dir=WORKFLOWS_DIR,
        intentionally_unreached=_INTENTIONALLY_UNREACHED_TEST_DIRS,
    )


# ── 2026-09-07: seven more shared checks, measured on this tree before wiring ──────────────
#
# Five reported nothing and are wired bare. Two reported, and what they reported is recorded
# below rather than hidden behind an allowlist with no reason.


def test_no_latched_availability_flags():
    """An optional dependency's availability must not be decided once and cached forever.

    A module-level `HAS_X = _probe()` latches whatever was true at import time, so a dependency
    that arrives (or a device that appears) later in the process is invisible until a restart --
    and the symptom is a silent fallback path, not an error.
    """
    from py_ci_shared.latched_availability_flags import assert_no_latched_availability_flags

    assert_no_latched_availability_flags([REPO_ROOT / "src"])


def test_no_epsilon_padded_power_denominators():
    """`x / (y + eps)` inside a power changes the result rather than guarding it.

    The padding is added to stop a division by zero and silently shifts every non-zero case too,
    which in an exponent compounds. Nothing here does it today; this keeps it that way.
    """
    from py_ci_shared.epsilon_padded_denominators import assert_no_epsilon_padded_power_denominators

    assert_no_epsilon_padded_power_denominators([REPO_ROOT / "src"])


def test_no_hash_fed_by_a_full_array_copy():
    """`h.update(a.tobytes())` allocates a second copy of the array purely to feed the hash.

    disk_cache is the reason this is gated here rather than left to review: it is the cache-KEY
    computation, so the copy is paid on every lookup including the hits, on arrays this library is
    pointed at in the tens of gigabytes. The eleven sites it had were rewritten to feed the buffer
    (`_buffer`), with every digest pinned unchanged in
    tests/test_disk_cache_digests_are_a_compatibility_contract.py.
    """
    from py_ci_shared.hash_fed_by_array_copy import assert_no_hash_fed_by_array_copy

    assert_no_hash_fed_by_array_copy([REPO_ROOT / "src"])


def test_no_stale_todos():
    """A TODO older than 30 days is a decision nobody made, wearing the clothes of one that was."""
    from py_ci_shared.stale_comment_age import assert_no_stale_todos

    assert_no_stale_todos(REPO_ROOT, ["src"])


def test_no_tracked_generated_files():
    """A committed `__pycache__` or `.pyc` shadows the source it was built from."""
    from py_ci_shared.repo_hygiene import find_tracked_generated_files

    tracked = find_tracked_generated_files(REPO_ROOT)
    assert tracked == [], f"generated files are tracked: {tracked}"


# Three optional numbers are tested for truth rather than for None, and all three are deliberate.
# Listed verbatim with the reason rather than left out of the check: the point of a baseline is that
# a FOURTH one has to be argued for here instead of appearing silently.
#
#   connection.py  `itersize`  -- psycopg2 rejects a named-cursor itersize of 0, so 0 and None both
#                                 correctly mean "leave the driver default alone".
#   _health.py     `group_by_provider` -- the count of rows to keep per provider. 0 requests an empty
#                                 result, which no caller can mean; "no grouping" is the documented
#                                 behaviour.
#   basics.py      `base`      -- the base64 variant (16/32/64). The docstring states that a falsy
#                                 base returns the hex digest, so 0 is the documented "off" value.
#
# Six were reported when this first ran. The other three were redundant double-guards whose inner
# `> 0` already excluded zero, so the outer truthiness test decided nothing and only made the intent
# read wrong; those are simplified in the source.
_TRUTHINESS_BASELINE = (
    "connection.py:203: `itersize` is an optional number tested for TRUTH; 0 is a value a caller can mean, and this reads it as absent. Use `itersize is not None`.",
    "_health.py:578: `group_by_provider` is an optional number tested for TRUTH; 0 is a value a caller can mean, and this reads it as absent. Use `group_by_provider is not None`.",
    "basics.py:128: `base` is an optional number tested for TRUTH; 0 is a value a caller can mean, and this reads it as absent. Use `base is not None`.",
)


def test_optional_numbers_are_tested_for_none():
    """`if x:` on an optional number reads 0 as absent, and 0 is a value callers mean.

    The three that remain are deliberate and carry their reasons above. The value of the gate is the
    fourth: a new `if limit:` or `if seed:` on a parameter whose 0 is meaningful now has to be
    argued for in this file rather than landing unnoticed.
    """
    from py_ci_shared.optional_truthiness import assert_optionals_test_for_none

    assert_optionals_test_for_none(
        files=_production_py_files(),
        repo_root=REPO_ROOT,
        baseline=_TRUTHINESS_BASELINE,
        min_subjects=50,
    )

# `__dir__` and `__getattr__` are the PEP 562 lazy-module pattern, which every package __init__ in
# this repo implements. The five `__getattr__` variants were read side by side: they differ in what
# they resolve (submodules, a typed exception, a module-alias table) and each carries its own
# docstring saying why. The one real difference is that pyutilz/data caches the resolved module into
# globals() and the others re-enter import_module, which after the first call is a sys.modules dict
# lookup -- not worth converging five deliberate implementations over.
_DUPLICATE_FUNCTION_ALLOW = ("__dir__", "__getattr__")


def test_no_drifted_duplicate_functions():
    """One function copied into several files, then edited in some of them and not others."""
    from py_ci_shared.drifted_duplicate_functions import assert_no_drifted_duplicate_functions

    assert_no_drifted_duplicate_functions([REPO_ROOT / "src"], allow=_DUPLICATE_FUNCTION_ALLOW)


# NOT WIRED YET, and named here rather than forgotten: the effect-assertion parity gate.
# This repo is at zero -- the five effects it reported are asserted in
# tests/test_db_execution_effects_reach_the_cursor.py and
# tests/test_pool_commits_and_probes_reach_the_connection.py -- and the gate passes against
# py-ci-shared master. What is missing is the dependency: resolving a src layout at all (so the scan
# is not vacuous) landed in py-ci-shared after the editable checkout this environment imports, so the
# gate's own population assertion fires locally while CI, which installs from git, would be green.
# Wire it with an empty `accepted` and a `len(import_map) > 100` population guard once the checkout is
# current. The guard is not optional: an empty map is exactly how this check passes while measuring
# nothing, which is how this repo read as clean until today.
