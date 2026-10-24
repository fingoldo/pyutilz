from __future__ import annotations
import logging
import pytest


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_s02_failed_install_is_warned_and_not_raised(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.core.pythonlib import packages

    class _FailingSubprocess:
        @staticmethod
        def check_call(cmd: list) -> None:
            raise OSError("pip exploded")

    monkeypatch.setattr(packages, "subprocess", _FailingSubprocess)
    caplog.set_level(logging.DEBUG)
    packages.ensure_installed("pyutilz_no_such_package_s02")  # must not raise
    warned = _warnings(caplog)
    assert any("pyutilz_no_such_package_s02" in m and "still not installed" in m for m in warned), warned
