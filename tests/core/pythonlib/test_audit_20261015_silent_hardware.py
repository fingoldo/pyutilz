from __future__ import annotations
import logging
import sys
import types
import pytest


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_s04_cpuinfo_probe_failure_is_warned(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.core.pythonlib import hardware

    fake = types.ModuleType("cpuinfo")

    def _broken() -> dict:
        raise RuntimeError("cpuid trap")

    fake.get_cpu_info = _broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cpuinfo", fake)
    caplog.set_level(logging.DEBUG)
    assert hardware.check_cpu_flag("avx2") is False
    assert any("avx2" in m for m in _warnings(caplog)), _warnings(caplog)


def test_s04_missing_cpuinfo_stays_debug(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.core.pythonlib import hardware

    monkeypatch.setitem(sys.modules, "cpuinfo", None)  # import raises ImportError
    caplog.set_level(logging.DEBUG)
    assert hardware.check_cpu_flag("avx2") is False
    assert not _warnings(caplog)
