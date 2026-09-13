"""Meta-test: every ``py_ci_shared`` module this repo imports must exist in the pinned version.

py-ci-shared is a git dependency, so the pin in the workflows can fall behind its master while the
dev box - which has a working checkout on PYTHONPATH - keeps importing whatever landed there
yesterday. That gap is invisible here and fatal in CI: the import fails at collection and every leg
goes red at once. It has happened twice (``mypy_gate``, then ``value_bearing_asserts``), so the pin
is now something a test can read rather than a string nobody compares against anything.
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec B404 - runs git against this repo's own pinned tag, no external input
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Resolved, never hardcoded: PY_CI_SHARED_DIR is what the ruff/black configs already extend from,
# and a sibling checkout is the layout every consumer of this package uses.
_SHARED_CHECKOUT = Path(os.environ.get("PY_CI_SHARED_DIR") or (_REPO_ROOT.parent / "py-ci-shared"))
_PIN_RE = re.compile(r"py-ci-shared\.git@(?P<tag>[\w.\-]+)")
_IMPORT_RE = re.compile(r"py_ci_shared\.(?P<module>[a-z_]+)")


def _pinned_tags() -> dict[str, str]:
    """Maps each workflow that installs py-ci-shared to the tag it pins."""
    out: dict[str, str] = {}
    for wf in sorted((_REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        for m in _PIN_RE.finditer(wf.read_text(encoding="utf-8")):
            out[wf.name] = m.group("tag")
    return out


def _imported_modules() -> set[str]:
    """Every ``py_ci_shared.<module>`` named anywhere in this repo's own sources and tests."""
    found: set[str] = set()
    for base in ("src", "tests", "scripts"):
        for path in (_REPO_ROOT / base).rglob("*.py"):
            found.update(m.group("module") for m in _IMPORT_RE.finditer(path.read_text(encoding="utf-8", errors="replace")))
    for extra in (".pre-commit-config.yaml",):
        found.update(m.group("module") for m in _IMPORT_RE.finditer((_REPO_ROOT / extra).read_text(encoding="utf-8")))
    return found


def test_every_workflow_pins_the_same_tag():
    """Two workflows installing two different versions is how one of them silently stops matching."""
    pins = _pinned_tags()
    assert pins, "no workflow installs py-ci-shared any more; this test and its comment are stale"
    assert len(set(pins.values())) == 1, f"workflows disagree on the py-ci-shared version: {pins}"


def test_the_pin_carries_every_module_we_import():
    """The pin must contain what this repo imports, or CI dies at collection while this box is green."""
    if not (_SHARED_CHECKOUT / ".git").is_dir():
        pytest.skip("no local py-ci-shared checkout to resolve the tag against")
    tag = next(iter(_pinned_tags().values()))
    listing = subprocess.run(  # nosec B603 B607 - fixed argv against a local checkout
        ["git", "ls-tree", "--name-only", f"{tag}:src/py_ci_shared"],
        cwd=str(_SHARED_CHECKOUT),
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        pytest.skip(f"tag {tag} is not present in the local checkout: {listing.stderr.strip()}")
    in_tag = {line[:-3] for line in listing.stdout.split() if line.endswith(".py")}
    missing = sorted(m for m in _imported_modules() if m not in in_tag)
    assert not missing, f"imported but absent from the pinned {tag}: {missing} -- bump the pin in .github/workflows/"
