"""A gate declared blocking must actually be able to block.

Three failure modes, all of them observed here across three audit waves:

1. **The gate did not finish.** ``mypy src/pyutilz`` aborted with an ``INTERNAL ERROR``
   inside a third-party stub and exited 2; the hook checked the exit code and nothing
   else, so which errors it reported depended on traversal order.
2. **The gate's scope was narrowed and nobody remembered.** ``--ignore C901``,
   ``exclude = ["tests"]``, a ``files=`` regex covering two subpackages out of thirty.
   Each was a deliberate decision at the time; none was written anywhere a reviewer or a
   check could see it, so none was ever revisited.
3. **The gate's threshold drifted below what it defends.** A CI ``--cov-fail-under``
   twenty points under measured coverage lets coverage collapse by a fifth, silently.

The declaration map below is the whole mechanism for (2): the check does not judge whether
a narrowing is right, only whether a human wrote down why. A new narrowing therefore fails
on the commit that introduces it, and a narrowing that is removed leaves a stale entry that
also fails -- so the map cannot quietly become a place findings go to be forgotten.

Runtime: one YAML parse, one TOML parse, and a line scan of eight workflow files -- under
0.1 s. No network, no subprocess.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("py_ci_shared", reason="py-ci-shared is a dev-only git dependency (requirements-dev.txt)")
pytest.importorskip("yaml", reason="PyYAML is required to parse .pre-commit-config.yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The pyproject tables that configure a BLOCKING gate. A narrowing key appearing in any of
# them has to be declared below, exactly like a CLI flag.
_GATE_CONFIG_TABLES = (
    "tool.ruff",
    "tool.ruff.lint",
    "tool.mypy",
    "tool.interrogate",
    "tool.deptry",
    "tool.coverage.report",
)

# Every scope narrowing and every lowered bar on a blocking gate, with the reason it exists.
# Key shape: "<venue>::<gate>::<knob>" -- deliberately not line-anchored, so a reformat or an
# insertion above does not invalidate a reviewed decision.
_DECLARED_NARROWINGS = {
    # -- ruff -------------------------------------------------------------------------
    "pre-commit::ruff-real-bugs::--ignore=C901": (
        "C901 (mccabe complexity) is advisory, not a bug class: 17 findings survive at the "
        "measured threshold of 20 and each is a judgement call about a dispatch function's "
        "shape. Surfaced by the full-select advisory pass instead."
    ),
    "ci.yml::with::ignore=C901": "The CI half of the same decision; the two venues must stay in step, which is why both keys are listed.",
    "pre-commit::ruff-real-bugs::exclude=(^|/)_benchmarks/": (
        "_benchmarks scripts print timing results to stdout by design, so T201/PERF401 there "
        "are the interface, not bugs. Mirrors [tool.ruff] exclude, which pre-commit bypasses "
        "when it passes explicit file paths."
    ),
    "pyproject::[tool.ruff]::exclude": "Same _benchmarks exemption plus VCS/build artifact dirs. tests/ and scripts/ were REMOVED from this list; they are linted.",
    "pyproject::[tool.ruff.lint]::per-file-ignores": "Each entry carries its own inline reason at the call site in pyproject.toml; the exemptions are per-file and per-code, never whole-directory.",
    # -- mypy -------------------------------------------------------------------------
    "pre-commit::mypy::files=^src/pyutilz/(dev/code_audit|stats)/": (
        "The mirrors-mypy hook is the fast STRICT-mode beachhead, deliberately scoped to the two "
        "subpackages verified clean under strict settings. The whole-project gate is a separate, "
        "unscoped hook (mypy-full-blocking) that checks the whole tree."
    ),
    # -- interrogate ------------------------------------------------------------------
    "pre-commit::interrogate-blocking::--fail-under=100": "A ceiling, not a narrowing: 100% is the maximum the tool can demand.",
    "ci.yml::with::interrogate-fail-under=100": "The CI half of the same 100% requirement.",
    "pyproject::[tool.interrogate]::exclude": "Docstring coverage is scoped to shipped code (src/pyutilz); tests and scripts document themselves through their names and assertions.",
    "pyproject::[tool.interrogate]::ignore-init-module": (
        "KNOWN GAP, not a clean decision: a module docstring on any __init__.py is unenforced, so "
        "a package carved out of a monolith inherits no documentation requirement. Kept because "
        "flipping it would demand a docstring on every re-export shim in one pass."
    ),
    # -- other blocking gates ---------------------------------------------------------
    "pre-commit::vulture-blocking::--min-confidence=80": "Vulture below 80% confidence is dominated by false positives on dynamically-referenced names; 80 is the value CI's lint-blocking job uses too.",
    "pre-commit::detect-secrets::exclude=\\.secrets\\.baseline$|tests/test_meta/_api_snapshot\\.json$": (
        "The API snapshot holds symbol NAMES, not values, and trips the keyword heuristic; the " "baseline file is the scanner's own state."
    ),
    "pre-commit::yamllint-blocking::files=^(\\.github/workflows/.*\\.ya?ml|\\.pre-commit-config\\.yaml)$": "yamllint is scoped to the YAML this repo authors; it is not a general-purpose YAML gate.",
    "pre-commit::zizmor-blocking::files=^\\.github/workflows/.*\\.ya?ml$": "zizmor only understands GitHub Actions workflow files.",
    "pyproject::[tool.deptry]::exclude": "Dependency drift is a question about shipped imports; tests/scripts may import dev-only tooling. .claude excludes nested agent worktrees whose duplicate source trees deptry's non-git-aware scan would misattribute.",
    # -- coverage ---------------------------------------------------------------------
    "pyproject::[tool.coverage.report]::fail_under": "The coverage ratchet, measured 84.87% on 2026-09-02 and set at 82 to absorb the Windows/Linux platform delta. Raise it when coverage rises.",
    "ci.yml::run::--cov-fail-under=82": "The CI half of the same ratchet; test_ci_coverage_gate_matches_pyproject below asserts the two numbers stay equal rather than trusting this note.",
}

# Tools whose exit code alone cannot certify that they ran to completion, mapped to the
# wrapper that requires their own success terminator instead.
_COMPLETION_WRAPPED_TOOLS = {"python -m mypy": "py_ci_shared.mypy_gate"}

# A gate that installs a different set of optional dependencies than the suite it claims to
# mirror is checking a different program. Measured 2026-09-03: mypy-full.yml and
# numba-coverage.yml both omitted `speedups`, so the blocking mypy gate type-checked every
# orjson call site in the orjson-ABSENT branch (where the import reports `import-not-found`)
# while ci.yml ran the tests in the orjson-PRESENT one (`no-redef`) -- see CLAUDE.md's
# "Never trim a `# type: ignore` code" section for why those two branches differ. The extras
# set below is ci.yml's; a workflow that deliberately installs a different one belongs in
# _DECLARED_DIFFERENT_EXTRAS with the reason, exactly like a gate narrowing.
_CANONICAL_EXTRAS = frozenset("pandas polars database system llm nlp cloud speedups dev".split())

# Key shape "<workflow>::<comma-joined sorted extras>", so exempting one install command in a
# file does not exempt the file's other install commands too.
_DECLARED_DIFFERENT_EXTRAS = {
    "publish.yml::all,dev": "A release smoke test must import what `pip install pyutilz[all]` gets, not just what the test matrix exercises.",
    "ci.yml::all,dev": "deptry's own install: the dependency-drift scan has to see every declared extra, or it reports each unseen one as unused.",
    "docs.yml::docs": "mkdocs builds prose; it imports no runtime extra, and installing them would add minutes to every docs run.",
}

_EXTRAS_INSTALL_RE = re.compile(r"""pip install (?:--system )?-e ["']?\.\[([a-z0-9,_-]+)\]["']?""")


def _workflow_extras_sets():
    """Every ``pip install -e ".[...]"`` in .github/workflows, as {filename: [set, ...]}."""
    found = {}
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        sets = [frozenset(m.group(1).split(",")) for m in _EXTRAS_INSTALL_RE.finditer(path.read_text(encoding="utf-8"))]
        if sets:
            found[path.name] = sets
    return found


def test_every_workflow_installs_the_same_extras_or_says_why():
    """One codebase, one dependency graph -- or a written reason for the difference."""
    found = _workflow_extras_sets()
    assert found, 'no workflow declares a `pip install -e ".[...]"`; the regex above has gone stale'

    mismatches = []
    seen_keys = set()
    for name, sets in found.items():
        for extras in sets:
            key = f"{name}::{','.join(sorted(extras))}"
            seen_keys.add(key)
            if extras == _CANONICAL_EXTRAS or key in _DECLARED_DIFFERENT_EXTRAS:
                continue
            missing = sorted(_CANONICAL_EXTRAS - extras)
            unexpected = sorted(extras - _CANONICAL_EXTRAS)
            mismatches.append(f"{key}: missing {missing or 'nothing'}, unexpected {unexpected or 'nothing'}")

    assert not mismatches, "Workflow extras differ from ci.yml's without a declared reason:\n  " + "\n  ".join(mismatches)

    # A declaration that no longer matches any install command is a reviewed decision that has
    # silently expired, the same failure mode _DECLARED_NARROWINGS guards against above.
    stale = sorted(set(_DECLARED_DIFFERENT_EXTRAS) - seen_keys)
    assert not stale, "Declared extras exceptions match no install command any more: " + ", ".join(stale)


def test_every_blocking_gate_narrowing_is_declared():
    """A gate that inspects less than it appears to must say so, in writing, here."""
    from py_ci_shared.gate_integrity import assert_narrowings_declared

    assert_narrowings_declared(
        precommit_path=PRECOMMIT,
        workflows_dir=WORKFLOWS_DIR,
        declared=_DECLARED_NARROWINGS,
        pyproject_path=PYPROJECT,
        pyproject_tables=_GATE_CONFIG_TABLES,
    )


def test_blocking_gates_assert_completion_not_just_exit_code():
    """mypy exiting 0 is not proof mypy finished. The wrapper demands the terminator."""
    from py_ci_shared.gate_integrity import assert_blocking_gates_assert_completion

    assert_blocking_gates_assert_completion(PRECOMMIT, _COMPLETION_WRAPPED_TOOLS)


def test_ci_coverage_gate_matches_pyproject():
    """Two venues, one ratchet: when they desync the lower one is the real gate."""
    from py_ci_shared.gate_integrity import assert_coverage_gate_parity

    assert_coverage_gate_parity(PYPROJECT, WORKFLOWS_DIR)
