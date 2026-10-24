from __future__ import annotations
import logging
import os
from typing import Any
import pytest


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_s11_unexpected_probe_failure_warned_and_assumed_alive(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.performance.kernel_tuning.cache import cache_base

    def _broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("probe broke")

    if os.name == "nt":
        monkeypatch.setattr(cache_base, "_kernel32", _broken)
    else:
        monkeypatch.setattr(cache_base.os, "kill", _broken)
    caplog.set_level(logging.DEBUG)
    assert cache_base._pid_alive(os.getpid() if os.name != "nt" else 4242) is True
    assert any("probe broke" in m for m in _warnings(caplog)), _warnings(caplog)
