"""A test module must not write ``os.environ`` at import time.

Under pytest-xdist one worker imports many test modules and then runs tests from all of them, so a module-level
``os.environ[...] = ...`` or ``os.environ.setdefault(...)`` applies to every later test that worker runs, and which
tests those are changes with the distribution: it fails a different innocent file each run. Use
``monkeypatch.setenv`` in the test or a fixture; if a variable must be set before a module-scope import, set it in
conftest.py or move the import into the test.

The scan is ``py_ci_shared.import_side_effects``. There are none here, so there is no baseline: the first one fails.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.import_side_effects import assert_no_new_import_time_env_mutations

TESTS_DIR = Path(__file__).resolve().parents[1]


def test_no_import_time_env_mutation():
    """No test module writes os.environ while it is being imported."""
    # ~298 test modules today; a scan that parsed fewer than 100 has stopped reaching the tree.
    assert_no_new_import_time_env_mutations(TESTS_DIR, min_files=100)
