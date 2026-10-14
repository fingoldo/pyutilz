"""A resource acquired in a module must be released on every path, including the exception path.

The gate looks for a release call that runs only on the success path. Three modules were reviewed and are correct
as written: each release sits in an ``except`` branch that runs only for the resource it opened, or in a ``finally``.
They are exempt by file, with the reason recorded here; every other module is held to the rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "pyutilz"

_REVIEWED_EXEMPT = {
    "core/serialization.py": "os.close(fd) runs in the except branch, only when os.fdopen never adopted the descriptor.",
    "database/db/connection.py": "close runs in the except of connect_to_db, only for a connection opened in that iteration.",
    "database/psycopg2_pool.py": "putconn calls sit in except branches or a finally; the one success-path return is intentional.",
}

# (constructor name, release method name) pairs the gate checks. Only connect has subjects the gate can see
# outside the reviewed modules. getconn (psycopg2_pool.py) and fdopen (serialization.py) have their only
# subject in a reviewed-exempt module, so the scan would have nothing to check; those modules are covered by their
# own tests. database/db/execution.py gets its cursors through a facade (_facade.get_cursor), which this scan cannot
# see; its named-cursor release is covered by tests/test_db_execution_effects_reach_the_cursor.py.
_PAIRS = [
    ("connect", "close"),
]


def _production_files() -> list[Path]:
    return [p for p in sorted(SRC_ROOT.rglob("*.py")) if "__pycache__" not in p.parts]


def test_resources_are_released_on_every_path():
    # py_ci_shared is imported inside the test: conftest forbids a module-level import of it.
    # It requires python>=3.9 and the requirements marker skips it on the 3.8 legs, so skip there explicitly;
    # on 3.9+ a missing install still fails the import.
    if sys.version_info < (3, 9):
        pytest.skip("py-ci-shared requires python>=3.9")
    from py_ci_shared.resource_release_paths import assert_released_on_every_path

    exempt = [SRC_ROOT / rel for rel in _REVIEWED_EXEMPT]
    for constructor, release in _PAIRS:
        assert_released_on_every_path(
            files=_production_files(),
            constructors=[constructor],
            release=release,
            repo_root=REPO_ROOT,
            ignore=exempt,
        )
