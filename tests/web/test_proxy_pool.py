"""Behaviour of ``pyutilz.web.web.proxy_pool``: a fixed port is used as given and a range picks a port inside it, a proxy
used within its interval is on cooldown while one used long ago is not, a wait that outlives its budget raises, and only the
Smartproxy DC gateway on its single fixed port is reported as a rotating proxy. Pure helpers only, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pyutilz.web.web import proxy_pool


class TestPortPick:
    def test_a_fixed_port_is_used_as_given(self):
        assert proxy_pool._pick_port(8080, 20000, 30000) == 8080

    def test_a_range_gives_a_port_inside_it(self):
        for _ in range(20):
            assert 20000 <= proxy_pool._pick_port(None, 20000, 30000) < 30000


class TestCooldown:
    def test_a_proxy_used_recently_is_on_cooldown(self):
        now = datetime(2026, 1, 1, 12, 0)
        last_used = {"p": now - timedelta(minutes=1)}
        assert proxy_pool._is_on_cooldown("p", now, ((last_used, 5.0),), False, "h", 1) is True

    def test_a_proxy_used_long_ago_is_free(self):
        now = datetime(2026, 1, 1, 12, 0)
        last_used = {"p": now - timedelta(minutes=30)}
        assert proxy_pool._is_on_cooldown("p", now, ((last_used, 5.0),), False, "h", 1) is False

    def test_a_proxy_absent_from_every_policy_is_free(self):
        now = datetime(2026, 1, 1, 12, 0)
        assert proxy_pool._is_on_cooldown("p", now, ((None, 5.0), ({}, 5.0)), False, "h", 1) is False


class TestWaitBudget:
    def test_the_first_call_starts_the_clock(self):
        started = proxy_pool._wait_budget_step(None, 60.0, 20000, 30000, 5.0)
        assert isinstance(started, datetime)

    def test_no_budget_means_no_clock(self):
        assert proxy_pool._wait_budget_step(None, None, 20000, 30000, 5.0) is None

    def test_an_exhausted_budget_raises(self):
        long_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        with pytest.raises(TimeoutError, match="no eligible proxy"):
            proxy_pool._wait_budget_step(long_ago, 60.0, 20000, 30000, 5.0)


class TestRotatingProxy:
    def test_the_dc_gateway_on_its_fixed_port_is_rotating(self):
        cfg = {"PROXY_HOST": "gate.dc.smartproxy.com", "PROXY_MIN_PORT": 20000, "PROXY_MAX_PORT": 20000}
        assert proxy_pool.is_rotating_proxy(cfg) is True

    def test_a_port_range_on_the_gateway_is_not_reported(self):
        cfg = {"PROXY_HOST": "gate.dc.smartproxy.com", "PROXY_MIN_PORT": 20001, "PROXY_MAX_PORT": 37960}
        assert proxy_pool.is_rotating_proxy(cfg) is None

    def test_another_host_is_not_reported(self):
        assert proxy_pool.is_rotating_proxy({"PROXY_HOST": "example.com"}) is None
