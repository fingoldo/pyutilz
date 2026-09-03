"""Assert that the tools the blocking pre-commit hooks actually invoke match this repo's pins.

WHY THIS EXISTS. The `ruff-real-bugs` hook in `.pre-commit-config.yaml` is `language: system`, so
pre-commit builds no isolated environment: `python -m ruff` resolves to whatever ruff happens to be
installed in the interpreter, NOT to the `ruff==<x>` pinned in `pyproject.toml`'s `[dev]` extra and
run by `ruff-blocking.yml` on CI. Nothing reported the mismatch, so a local `All checks passed!`
was not evidence that CI would be green -- and ruff adds rules to already-selected families
(`RUF`, `B`, `PERF`) between patch releases, which is exactly how mlframe's pre-commit-vs-CI drift
went unnoticed. mypy already has an equivalent assertion via `py_ci_shared.mypy_gate`; this is
ruff's. Keeping it as a `repo: local` hook preserves the "one resolved config, no --select"
property the ruff hook's own comment is careful about.

Reads the pin straight out of `pyproject.toml` so there is one source of truth: bumping the pin
there is all a version upgrade needs. Exits non-zero, naming both versions and the fix, when they
differ. Python 3.8 compatible (no tomllib): the pin is read with a regex, not a TOML parser.
"""

import re
import subprocess  # nosec B404 - runs `python -m <tool> --version` with a fixed argv, no shell
import sys
from pathlib import Path

# tool import name -> the exact-pin spelling to look for in pyproject.toml
_PINNED_TOOLS = {"ruff": "ruff"}


def _pinned_version(pyproject_text, dist_name):
    """The exact version from a `"<dist>==<ver>"` requirement line in pyproject.toml, or None."""
    match = re.search(r'"' + re.escape(dist_name) + r'==([0-9][^"]*)"', pyproject_text)
    return match.group(1) if match else None


def _installed_version(module_name):
    """`python -m <module> --version`'s version token, or None when the tool is not importable."""
    try:
        out = subprocess.run(  # nosec B603 - fixed argv (sys.executable plus literal flags), shell=False
            [sys.executable, "-m", module_name, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    tokens = (out.stdout or out.stderr).split()
    return tokens[-1] if tokens else None


def main():
    root = Path(__file__).resolve().parent.parent
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    problems = []
    for module_name, dist_name in _PINNED_TOOLS.items():
        pinned = _pinned_version(pyproject_text, dist_name)
        if pinned is None:
            problems.append(f"{dist_name}: no exact pin ({dist_name}==<version>) found in pyproject.toml")
            continue
        installed = _installed_version(module_name)
        if installed is None:
            problems.append(f"{dist_name}: pinned at {pinned} but not importable as `python -m {module_name}` in this interpreter")
        elif installed != pinned:
            problems.append(
                f"{dist_name}: this interpreter has {installed}, pyproject.toml pins {pinned}. The blocking hook is "
                f"`language: system`, so it runs {installed} while CI runs {pinned}. Fix: pip install {dist_name}=={pinned}"
            )
    if problems:
        for problem in problems:
            print("pinned-tool-version mismatch: " + problem)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
