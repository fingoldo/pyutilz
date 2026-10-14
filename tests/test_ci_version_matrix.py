"""The version-matrix helper picks the legs the pandas and polars matrices run, and the badge document they publish.

The network lookup is replaced by a stub, so these tests run offline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from packaging.version import Version

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_version_matrix.py"


@pytest.fixture(scope="module")
def ci_version_matrix():
    spec = importlib.util.spec_from_file_location("ci_version_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as mp:
        # The script's dataclass looks its module up in sys.modules while the class is being built, so register it first;
        # setitem undoes the registration at teardown.
        mp.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def _versions(*texts: str) -> list[Version]:
    return [Version(t) for t in texts]


def test_plan_gives_floor_newest_of_its_series_and_latest(ci_version_matrix):
    releases = _versions("1.5.0", "1.5.3", "2.0.0", "2.2.3", "3.0.6")
    legs = ci_version_matrix.plan("1.5", releases)
    assert [leg.version for leg in legs] == ["1.5", "1.5.3", "3.0.6"]
    assert legs[0].label == "1.5 (min)"
    assert legs[-1].label == "3.0.6 (latest)"


def test_plan_runs_equal_versions_once(ci_version_matrix):
    releases = _versions("1.5.0", "1.6.0")
    legs = ci_version_matrix.plan("1.5", releases)
    assert [leg.version for leg in legs] == ["1.5", "1.6.0"]


def test_plan_with_a_single_release_is_one_leg(ci_version_matrix):
    legs = ci_version_matrix.plan("1.5", _versions("1.5.0"))
    assert [leg.version for leg in legs] == ["1.5"]
    assert legs[0].label == "1.5 (min, latest)"


def test_plan_rejects_no_releases(ci_version_matrix):
    with pytest.raises(ValueError):
        ci_version_matrix.plan("1.5", [])


def test_usable_releases_skips_prereleases_and_fully_yanked_versions(ci_version_matrix):
    releases = {
        "1.5.0": [{"yanked": False}],
        "2.0.0rc1": [{"yanked": False}],
        "2.0.0.dev1": [{"yanked": False}],
        "2.1.0": [{"yanked": True}],
        "2.2.0": [{"yanked": False}, {"yanked": True}],
        "not-a-version": [{"yanked": False}],
    }
    assert [str(v) for v in ci_version_matrix.usable_releases(releases)] == ["1.5.0", "2.2.0"]


def test_lock_floor_takes_the_lowest_resolved_version(ci_version_matrix):
    lock = (
        '[[package]]\nname = "pandas"\nversion = "2.1.4"\n\n'
        '[[package]]\nname = "pandas"\nversion = "1.5.3"\n\n'
        '[[package]]\nname = "numpy"\nversion = "1.26.0"\n'
    )
    assert ci_version_matrix.lock_floor(lock, "pandas") == "1.5.3"


def test_lock_floor_fails_for_a_package_not_in_the_lock(ci_version_matrix):
    with pytest.raises(LookupError):
        ci_version_matrix.lock_floor('[[package]]\nname = "numpy"\nversion = "1.26.0"\n', "pandas")


def test_matrix_document_has_one_include_entry_per_leg(ci_version_matrix):
    legs = ci_version_matrix.plan("1.5", _versions("1.5.0", "2.0.0"))
    document = ci_version_matrix.matrix(legs, "3.12")
    assert document["include"][0] == {"version": "1.5", "label": "1.5 (min)", "python-version": "3.12"}
    assert all(entry["python-version"] == "3.12" for entry in document["include"])


def test_badge_lists_the_tested_versions_and_colours_by_result(ci_version_matrix):
    legs = [
        {"version": "1.5", "label": "1.5 (min)"},
        {"version": "2.2.3", "label": "2.2.3"},
        {"version": "3.0.6", "label": "3.0.6 (latest)"},
    ]
    document = ci_version_matrix.badge("pandas", legs, "success")
    assert document == {"schemaVersion": 1, "label": "pandas", "message": "1.5 | 2.2.3 | 3.0.6 (latest)", "color": "brightgreen"}
    assert ci_version_matrix.badge("pandas", legs, "failure")["color"] == "red"
    assert ci_version_matrix.badge("pandas", [], "success")["message"] == "unknown"


def test_plan_command_with_an_explicit_floor_writes_the_matrix(ci_version_matrix, monkeypatch, tmp_path):
    monkeypatch.setattr(ci_version_matrix, "fetch_releases", lambda package: {"1.5.0": [{"yanked": False}], "3.0.6": [{"yanked": False}]})
    output = tmp_path / "github_output"
    code = ci_version_matrix.main(["plan", "--package", "pandas", "--floor", "1.5", "--python", "3.12", "--github-output", str(output)])
    assert code == 0
    written = output.read_text(encoding="utf-8")
    assert written.startswith("matrix=")
    document = json.loads(written.split("=", 1)[1])
    assert [leg["version"] for leg in document["include"]] == ["1.5", "3.0.6"]


def test_plan_needs_exactly_one_floor_source(ci_version_matrix):
    with pytest.raises(SystemExit):
        ci_version_matrix.main(["plan", "--package", "pandas", "--python", "3.12"])
