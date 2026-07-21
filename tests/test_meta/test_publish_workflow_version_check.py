"""Regression test for .github/workflows/publish.yml's tag-vs-version safety check.

Regression (2026-07-21 audit round 2, MEDIUM): the publish workflow never verified the pushed
git tag matched pyproject.toml's [project].version before building/publishing -- a maintainer
who bumped the CHANGELOG heading and tagged v2.0.0 but forgot the pyproject.toml edit would get
a wheel silently built and published under the OLD version, permanently mismatching the git tag
(and, since PyPI is immutable per-version, only fixable by picking a fresh, still-higher version
by hand). This test extracts the actual shell snippet from the workflow YAML and RUNS it (not
just checks for a string) so a future edit that breaks the comparison logic is caught.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "publish.yml"
_STEP_NAME = "Verify tag matches pyproject.toml version"


def _find_step(name: str) -> dict:
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    for step in workflow["jobs"]["build"]["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"step {name!r} not found in {_WORKFLOW_PATH}")


def _current_project_version() -> str:
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version = "(.*)"$', text, re.MULTILINE)
    assert m is not None, 'pyproject.toml has no top-level version = "..." line'
    return m.group(1)


def _run_check_script(script: str, ref_name: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["GITHUB_REF_NAME"] = ref_name
    return subprocess.run(
        ["bash", "-c", script],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestPublishWorkflowVersionCheck:
    def test_step_exists_and_references_the_right_inputs(self):
        step = _find_step(_STEP_NAME)
        assert "GITHUB_REF_NAME" in step["run"]
        assert "pyproject.toml" in step["run"]

    def test_matching_tag_and_version_passes(self):
        step = _find_step(_STEP_NAME)
        current = _current_project_version()
        result = _run_check_script(step["run"], ref_name=f"v{current}")
        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    def test_mismatched_tag_and_version_fails(self):
        step = _find_step(_STEP_NAME)
        current = _current_project_version()
        mismatched = f"{current}-does-not-exist"
        result = _run_check_script(step["run"], ref_name=f"v{mismatched}")
        assert result.returncode != 0
        assert mismatched in result.stdout or mismatched in result.stderr

    def test_tag_missing_v_prefix_is_treated_as_the_version_itself(self):
        """``${GITHUB_REF_NAME#v}`` only strips a LEADING literal "v" -- document that a
        malformed ref without the prefix is compared as-is (workflow triggers only on `v*` tags,
        so this is an edge case, not the primary contract)."""
        step = _find_step(_STEP_NAME)
        current = _current_project_version()
        result = _run_check_script(step["run"], ref_name=current)
        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
