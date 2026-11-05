"""Throttle timing pins for web.cached_client.CachedHttpClient (mutation survivors)."""

from __future__ import annotations


import pytest

from pyutilz.web import cached_client as cc


@pytest.fixture
def client(tmp_path):
    return cc.CachedHttpClient(tmp_path, min_interval=0.2)


def test_an_unseen_host_waits_the_full_gap_measured_from_zero(monkeypatch, client):
    clock = {"now": 0.05}
    slept = []

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(cc.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(cc.time, "sleep", fake_sleep)
    client._throttle("https://host.example/x")
    assert slept == [pytest.approx(0.15)]


def test_a_request_exactly_one_gap_after_the_last_is_not_delayed(monkeypatch, client):
    client.min_interval = 2.0
    monkeypatch.setattr(cc.time, "monotonic", lambda: 10.0)
    slept = []
    monkeypatch.setattr(cc.time, "sleep", lambda s: slept.append(s))
    client._last_call["host.example"] = 8.0
    client._throttle("https://host.example/x")
    assert slept == []
