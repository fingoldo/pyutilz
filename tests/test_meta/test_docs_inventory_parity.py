"""Documented inventories are computed from the thing they document.

Eleven findings in one documentation audit came from three inventories that had drifted
apart from their source of truth: extras-group descriptions against
``[project.optional-dependencies]``, module-orientation docs against the shipped tree, and
paths/markers named in prose against the tree and the pytest config. All three are
comparisons between two in-repo declarations, so none of them needs a human to notice.

Rule (b) -- module coverage -- warns rather than blocks: a deliberately-undocumented private
helper is shaped exactly like an oversight, and only the ``_UNDOCUMENTED_BY_DESIGN`` set
below can tell them apart. It reports zero on the current tree, so the warning is live
rather than pre-suppressed.

Runtime: one TOML parse plus reads of five prose files and a walk of src/ -- about 0.1 s.
No network, no subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("py_ci_shared", reason="py-ci-shared is a dev-only git dependency (requirements-dev.txt)")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
README = REPO_ROOT / "README.md"
PROSE_FILES = [REPO_ROOT / name for name in ("README.md", "CONTRIBUTING.md", "TESTING.md", "CHANGELOG.md")]
MODULE_DOCS = [REPO_ROOT / "README.md", REPO_ROOT / "docs" / "modules.md", REPO_ROOT / "docs" / "index.md", REPO_ROOT / "CHANGELOG.md"]

# One documented extras bullet in README's install block.
_EXTRAS_BULLET_RE = r"pip install pyutilz\[(\w+)\]\s*#\s*(.*)"
# The prose sentence stating what the aggregate [all] group resolves to.
_AGGREGATE_RE = r"`\[(all)\]` = `([\w,]+)`"

# python-dateutil is a shared parsing pin carried by five separate groups; naming it in
# every bullet would bury the members that actually characterise each group.
_NOT_ENUMERATED_IN_PROSE = ("python-dateutil",)

# Groups deliberately absent from the install block, each because something else documents
# them: [pandas] and [polars] are the halves of [dataframes], which IS documented and is the
# form contributors are told to install; [all] is documented by the prose sentence right
# below the block, which test_aggregate_extras_group_composition_is_accurate checks instead.
_GROUPS_DOCUMENTED_ELSEWHERE = ("pandas", "polars", "all")

# Modules intentionally absent from the orientation docs. Empty today -- the twelve-module
# backlog the audit found has been closed, so anything appearing here from now on is a
# deliberate decision someone had to write down.
_UNDOCUMENTED_BY_DESIGN: tuple[str, ...] = ()


def test_extras_group_descriptions_match_pyproject():
    """Every declared extras group is documented, and names exactly its own members."""
    from py_ci_shared.docs_inventory_parity import assert_no_inventory_drift, find_extras_documentation_drift

    problems = find_extras_documentation_drift(
        PYPROJECT,
        README,
        _EXTRAS_BULLET_RE,
        ignore_packages=_NOT_ENUMERATED_IN_PROSE,
        undocumented_groups=_GROUPS_DOCUMENTED_ELSEWHERE,
    )
    assert_no_inventory_drift(problems, "extras-group documentation drift(s)")


def test_aggregate_extras_group_composition_is_accurate():
    """README calls `[all,dev]` the recommended install; what `[all]` pulls in has to be right."""
    from py_ci_shared.docs_inventory_parity import assert_no_inventory_drift, find_aggregate_group_drift

    assert_no_inventory_drift(find_aggregate_group_drift(PYPROJECT, README, _AGGREGATE_RE), "aggregate extras-group composition drift(s)")


def test_documented_paths_exist():
    """A backtick-quoted repo path or glob in prose resolves to something real."""
    from py_ci_shared.docs_inventory_parity import assert_no_inventory_drift, find_phantom_doc_paths

    problems = find_phantom_doc_paths(PROSE_FILES, REPO_ROOT, search_roots=[REPO_ROOT / "src", REPO_ROOT / "src" / "pyutilz"])
    assert_no_inventory_drift(problems, "documented path(s) that do not exist")


def test_documented_pytest_markers_are_declared():
    """--strict-markers turns a documented-but-undeclared marker into a collection ERROR."""
    from py_ci_shared.docs_inventory_parity import assert_no_inventory_drift, find_undeclared_markers

    assert_no_inventory_drift(find_undeclared_markers(PROSE_FILES, PYPROJECT), "documented pytest marker(s) that are not declared")


def test_every_shipped_module_appears_in_some_documentation():
    """WARN-only: a module documented nowhere is a module nobody can find."""
    from py_ci_shared.docs_inventory_parity import find_undocumented_modules

    undocumented = find_undocumented_modules(REPO_ROOT / "src" / "pyutilz", MODULE_DOCS, undocumented_by_design=_UNDOCUMENTED_BY_DESIGN)
    if undocumented:
        import warnings

        warnings.warn(f"{len(undocumented)} shipped module(s) appear in no module-orientation doc:\n  " + "\n  ".join(undocumented), UserWarning, stacklevel=2)
