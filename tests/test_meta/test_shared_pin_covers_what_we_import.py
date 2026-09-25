"""Meta-test: py-ci-shared is pinned to one full commit SHA everywhere, and that commit carries every module this repo imports.

py-ci-shared is a git dependency, so the pin can fall behind its master while the dev box - which has a working checkout on
PYTHONPATH - keeps importing whatever landed there yesterday. That gap is invisible here and fatal in CI: the import fails at
collection and every leg goes red at once. It has happened twice (``mypy_gate``, then ``value_bearing_asserts``).

The package reaches this repo through four doors: requirements-dev.txt (which CI installs), the reusable workflows and actions in
``uses:`` refs, the ``py-ci-shared-ref`` input those workflows fetch configs at, and the pre-commit ``rev:``. They sat on a tag, the
moving ``@v1``, two different upload-codecov commits and an unpinned master install at once. A bump is one search-and-replace of the
SHA; this fails a bump that misses a line, a ref that is a tag or branch, and a ``# vX.Y.Z`` comment that disagrees.
"""

from __future__ import annotations

import re
import subprocess  # nosec B404 - runs git against the local py-ci-shared checkout with a fixed argv, no external input
from pathlib import Path

from py_ci_shared.git_dependency_pins import assert_installed_includes_pin

_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMPORT_RE = re.compile(r"py_ci_shared\.(?P<module>[a-z][a-z_]*)")
_SHA = re.compile(r"[0-9a-f]{40}")
_TAG = re.compile(r"#\s*(v\d+\.\d+\.\d+)\b")
_REF_PATTERNS = (
    re.compile(r"uses:\s*fingoldo/py-ci-shared/\.github/[\w./-]+@(?P<ref>[^\s#]+)"),
    re.compile(r"py-ci-shared-ref:\s*['\"]?(?P<ref>[^\s'\"#]+)"),
    re.compile(r"py-ci-shared\s*@\s*git\+https://github\.com/fingoldo/py-ci-shared(?:\.git)?(?:@(?P<ref>[^\s;#'\"]+))?"),
)
_PRECOMMIT_REPO = re.compile(r"-\s*repo:\s*https://github\.com/fingoldo/py-ci-shared(?:\.git)?\s*\n\s*rev:\s*(?P<ref>[^\s#]+)(?P<rest>[^\n]*)")


def _pin_files() -> list[Path]:
    """Every file that can carry a py-ci-shared ref."""
    files = sorted((_REPO_ROOT / ".github").rglob("*.yml")) + sorted((_REPO_ROOT / ".github").rglob("*.yaml"))
    files += [_REPO_ROOT / name for name in ("requirements-dev.txt", "pyproject.toml", ".pre-commit-config.yaml") if (_REPO_ROOT / name).exists()]
    return files


def _pins() -> list[tuple[str, str, str]]:
    """``(where, ref, tag comment)`` for every py-ci-shared ref; an unpinned requirement has ref ``""``."""
    found: list[tuple[str, str, str]] = []
    for path in _pin_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern in _REF_PATTERNS:
                m = pattern.search(line)
                if m:
                    tag = _TAG.search(line[m.end() :])
                    found.append((f"{rel}:{lineno}", m.group("ref") or "", tag.group(1) if tag else ""))
                    break
        for m in _PRECOMMIT_REPO.finditer(text):
            tag = _TAG.search(m.group("rest"))
            found.append((f"{rel}:{text.count(chr(10), 0, m.start('ref')) + 1}", m.group("ref"), tag.group(1) if tag else ""))
    return found


def _the_pin() -> str:
    """The one SHA every py-ci-shared ref names; fails on a second SHA, a non-SHA ref or a missing/disagreeing release comment."""
    pins = _pins()
    # The requirement, the pre-commit rev and the workflow refs: fewer means the patterns stopped matching this repo's spellings.
    assert len(pins) >= 8, f"found only {len(pins)} py-ci-shared refs: {pins}"
    bad = [p for p in pins if not _SHA.fullmatch(p[1])]
    assert not bad, "every py-ci-shared ref must be a full 40-character commit SHA, not a tag, branch or nothing:\n  " + "\n  ".join(f"{w}: {r or '(unpinned)'}" for w, r, _ in bad)
    shas = {r for _, r, _ in pins}
    assert len(shas) == 1, "py-ci-shared is pinned to several commits; update every ref to one:\n  " + "\n  ".join(f"{w}: {r[:12]}" for w, r, _ in pins)
    untagged = [w for w, _, t in pins if not t]
    assert not untagged, f"every pin carries the release it is as a `# vX.Y.Z` comment; missing at: {untagged}"
    tags = {t for _, _, t in pins}
    assert len(tags) == 1, f"one SHA commented as several releases: {sorted(tags)}"
    return shas.pop()


def _imported_modules() -> set[str]:
    """Every ``py_ci_shared.<module>`` named anywhere in this repo's own sources and tests."""
    found: set[str] = set()
    for base in ("src", "tests", "scripts"):
        for path in (_REPO_ROOT / base).rglob("*.py"):
            found.update(m.group("module") for m in _IMPORT_RE.finditer(path.read_text(encoding="utf-8", errors="replace")))
    for extra in (".pre-commit-config.yaml",):
        found.update(m.group("module") for m in _IMPORT_RE.finditer((_REPO_ROOT / extra).read_text(encoding="utf-8")))
    return found


def test_every_py_ci_shared_ref_is_one_full_sha_with_one_release_comment():
    """One commit for the requirement, every workflow and action ref, every py-ci-shared-ref input and the pre-commit rev."""
    assert _SHA.fullmatch(_the_pin())


def test_the_installed_py_ci_shared_includes_the_pin():
    """The gate code this interpreter runs contains the pinned commit, so a local run and CI judge with the same rules."""
    assert_installed_includes_pin("py_ci_shared", _the_pin(), dist="py-ci-shared")


def test_the_pin_carries_every_module_we_import():
    """The pinned commit itself must contain what this repo imports, or CI dies at collection while a newer local checkout is green."""
    import py_ci_shared

    sha = _the_pin()
    package_dir = Path(py_ci_shared.__file__).resolve().parent
    checkout = package_dir.parent.parent  # src/py_ci_shared -> the checkout root, when the package is a git checkout
    listing = subprocess.run(  # nosec B603 B607 - fixed argv against the checkout that owns the imported package
        ["git", "ls-tree", "--name-only", f"{sha}:src/py_ci_shared"], cwd=str(checkout), capture_output=True, text=True, check=False
    )
    if listing.returncode != 0:
        # CI installs from the pin into site-packages, where the installed package IS the pinned commit; the import check covers it.
        missing = sorted(m for m in _imported_modules() if not (package_dir / f"{m}.py").exists())
        assert not missing, f"imported but absent from the installed py-ci-shared: {missing} -- bump the pin"
        return
    in_pin = {line[:-3] for line in listing.stdout.split() if line.endswith(".py")}
    missing = sorted(m for m in _imported_modules() if m not in in_pin)
    assert not missing, f"imported but absent from the pinned {sha[:12]}: {missing} -- bump the pin"
