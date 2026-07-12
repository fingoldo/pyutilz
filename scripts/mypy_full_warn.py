#!/usr/bin/env python
"""Warn-only whole-project mypy check for the pre-commit hook.

Runs ``mypy src/pyutilz`` (the exact invocation mypy-full.yml's reusable workflow runs) and
ALWAYS exits 0. This mirrors mypy-full.yml's own ``continue-on-error: ${{ inputs.advisory }}``
posture: pyutilz's mypy-full.yml sets ``advisory: true`` because the whole-project backlog is
not yet closed to 0 (unlike mlframe, which sets ``advisory: false`` and is a real blocking gate
there), so this is genuinely NOT a hard-blocking CI gate right now -- a local hard blocker here
would be STRICTER than CI itself, not matching it.

Previously the only whole-project mypy hook (``mypy-full-manual`` from py-ci-shared) was
``stages: [manual]`` only, i.e. it never ran automatically at all, so mypy errors were invisible
locally until someone ran it by hand. This script makes the same check run on every commit like
CI does, surfacing findings as warnings, without blocking.

Once the backlog is triaged to 0, flip ``advisory: false`` in mypy-full.yml AND replace this
warn-only hook with a real blocking one (mirroring the black-filtered-blocking pattern in
.pre-commit-config.yaml).
"""
import subprocess
import sys


def main() -> int:
    """Run whole-project mypy, print any findings as warnings, and always return 0."""
    result = subprocess.run([sys.executable, "-m", "mypy", "src/pyutilz"])
    if result.returncode != 0:
        print(
            "\n[mypy-full-warn] Findings above are WARNINGS ONLY (mirrors mypy-full.yml's "
            "advisory: true in CI) -- the commit is NOT blocked.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
