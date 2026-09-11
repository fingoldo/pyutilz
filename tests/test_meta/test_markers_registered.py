"""Every pytest marker the suite uses is registered.

Without ``--strict-markers`` an unregistered ``@pytest.mark.<name>`` is a silent no-op, so ``-m "not slow"`` keeps
running a test whose ``slow`` was misspelled. ``py_ci_shared.pytest_markers`` reads what pyproject and every conftest
register and what the tests use; ``expect_registered`` names the markers this repo is known to register, so a parser
that stopped reading pyproject cannot report a clean tree.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.pytest_markers import assert_markers_registered

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_used_marker_is_registered():
    """Every marker used under tests/ is registered, and the parser still sees the known registrations."""
    assert_markers_registered(REPO_ROOT, expect_registered=("gpu", "live", "slow"))
