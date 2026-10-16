"""Every code-audit scanner must run to completion on the production tree (audit finding S-01).

``run_all`` turns a crashed scanner into an empty result, so a broken scanner used to look like a clean repo.
``run_all_checked`` reports the crashed scanners instead; this gate fails if any of them crashed.
"""

from pathlib import Path

import pytest

PYUTILZ_DIR = Path(__file__).resolve().parents[2] / "src" / "pyutilz"


# Runs every scanner over the whole production tree, which took over 120 s on the macOS CI runner (the global limit).
# The timeout is raised for this one gate only, so a hang still fails instead of running forever.
@pytest.mark.timeout(400)
def test_no_code_audit_scanner_crashes_on_the_production_tree():
    from pyutilz.dev.code_audit.registry import run_all_checked

    _, failed = run_all_checked(PYUTILZ_DIR)
    assert not failed, "code-audit scanners that raised (their findings are unknown, not clean):\n  " + "\n  ".join(
        f"{name}: {error}" for name, error in sorted(failed.items())
    )


def test_a_crashing_scanner_is_reported_as_failed(tmp_path, monkeypatch):
    """Teeth check: the gate above is only meaningful if a raising scanner is actually reported."""
    from pyutilz.dev.code_audit import registry

    def boom(*args, **kwargs):
        raise RuntimeError("planted scanner crash")

    scanner_names = sorted(registry._SCANNERS)
    monkeypatch.setitem(registry._SCANNERS, scanner_names[0], boom)
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    _, failed = registry.run_all_checked(tmp_path, checks=[scanner_names[0]], parallel=False)
    assert scanner_names[0] in failed and "planted scanner crash" in failed[scanner_names[0]]
