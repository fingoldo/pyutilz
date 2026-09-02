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


def test_no_new_file_over_1k_loc():
    """CLAUDE.md's module-size rule, enforced instead of remembered."""
    from py_ci_shared.loc_budget import assert_no_new_oversized_file

    assert_no_new_oversized_file(
        files=_production_py_files(),
        root=REPO_ROOT,
        baseline_path=Path(__file__).resolve().parent / "_loc_over_1k_baseline.json",
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
