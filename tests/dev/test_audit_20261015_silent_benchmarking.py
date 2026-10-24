from __future__ import annotations
import logging
import pytest


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_s05_mean_ranking_warns_the_crashing_candidate(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.dev import benchmarking

    def _boom() -> None:
        raise ValueError("kernel crashed")

    caplog.set_level(logging.DEBUG)
    out = benchmarking._rank_mean({"bad": _boom, "ok": lambda: None}, ["bad", "ok"], 1, False)
    assert out["bad"] == float("inf")
    assert any("bad" in m and "kernel crashed" in m for m in _warnings(caplog)), _warnings(caplog)
