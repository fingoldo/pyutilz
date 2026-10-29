"""Behaviour of ``pyutilz.core.pythonlib.packages``: a package is present when its import name resolves, a check that raises is
reported as unknown rather than as present or missing, only the packages that are missing are installed (one pip call each,
with a failed install logged and collected), and a failing install never aborts the remaining ones. Pip is replaced by a
recorder, so nothing is installed."""

from __future__ import annotations

import subprocess

import pytest

from pyutilz.core.pythonlib import packages


class _PipRecorder:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def __call__(self, argv, *args, **kwargs):
        name = argv[-1]
        self.calls.append(name)
        if name in self.fail:
            raise subprocess.CalledProcessError(1, argv)


@pytest.fixture
def pip(monkeypatch):
    recorder = _PipRecorder()
    monkeypatch.setattr(packages.subprocess, "check_call", recorder)
    return recorder


class TestPresence:
    def test_an_importable_package_is_present(self):
        assert packages._package_present("json") is True

    def test_an_absent_package_is_not_present(self):
        assert packages._package_present("nosuch_pkg_zz") is False

    def test_a_check_that_raises_is_unknown(self):
        assert packages._package_present("nosuch_pkg_zz.child") is None


class TestInstallEach:
    def test_each_package_is_installed_with_the_running_interpreter(self, pip):
        assert packages._install_each(["alpha", "beta"]) == []
        assert pip.calls == ["alpha", "beta"]

    def test_a_failed_install_is_collected_and_the_rest_still_run(self, monkeypatch):
        recorder = _PipRecorder(fail={"alpha"})
        monkeypatch.setattr(packages.subprocess, "check_call", recorder)
        assert packages._install_each(["alpha", "beta"]) == ["alpha"]
        assert recorder.calls == ["alpha", "beta"]


class TestEnsureInstalled:
    def test_present_packages_trigger_no_install(self, pip):
        packages.ensure_installed(["json", "os"])
        assert pip.calls == []

    def test_only_the_missing_packages_are_installed(self, pip):
        packages.ensure_installed(["json", "nosuch_pkg_zz"])
        assert pip.calls == ["nosuch_pkg_zz"]

    def test_a_separated_string_names_several_packages(self, pip):
        packages.ensure_installed("json nosuch_pkg_zz", sep=" ")
        assert pip.calls == ["nosuch_pkg_zz"]

    def test_a_single_name_is_accepted(self, pip):
        packages.ensure_installed("nosuch_pkg_zz")
        assert pip.calls == ["nosuch_pkg_zz"]

    def test_an_empty_request_does_nothing(self, pip):
        packages.ensure_installed([])
        assert pip.calls == []

    def test_a_known_abbreviation_is_checked_under_its_import_name(self, pip, monkeypatch):
        seen = []
        monkeypatch.setattr(packages, "_package_present", lambda name: seen.append(name) or True)
        packages.ensure_installed(["scikit-learn"])
        assert seen == ["sklearn"]
        assert pip.calls == []
