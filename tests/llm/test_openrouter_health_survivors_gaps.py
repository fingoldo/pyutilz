"""Branch, argument-order and fallback pins for the OpenRouter health helpers (mutation survivors, gap batch)."""

from __future__ import annotations

import logging

import pytest

import pyutilz.llm.openrouter_provider as pkg
from pyutilz.llm.openrouter_provider import _health as h


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(pkg, "_HEALTH_CACHE", {})


class TestUptimeFields:
    def test_the_one_day_uptime_field_alone_marks_a_percentage_payload(self):
        assert h._uptime_payload_is_percentage([{"uptime_last_1d": 50}]) is True


class TestFetchFreshHealthWarning:
    def test_the_aggregated_warning_counts_failures_then_fetched_models(self, monkeypatch, caplog):
        def fake(model, api_key=None, timeout=30.0, client=None):
            if model == "a":
                raise RuntimeError("boom")
            return []

        monkeypatch.setattr(h, "_fetch_endpoints_for_model", fake)
        caplog.set_level(logging.WARNING)
        h._fetch_fresh_health([{"id": "a"}, {"id": "b"}], "KEY", 2, 5.0)
        assert caplog.records[-1].getMessage() == "Health check failed for 1/2 models, e.g. a: boom"


class TestEnrichArguments:
    def test_fetch_receives_the_worker_count_and_timeout_in_their_own_slots(self, monkeypatch):
        seen = []

        def fake(to_fetch, api_key, max_workers, timeout):
            seen.append((api_key, max_workers, timeout))
            return {}

        monkeypatch.setattr(h, "_fetch_fresh_health", fake)
        h._enrich_with_health([{"id": "m"}], "KEY", 0.5, 3, 7.0, 0.0)
        assert seen == [("KEY", 3, 7.0)]


class TestCacheWriteGate:
    def test_a_zero_ttl_never_writes_the_cache(self, monkeypatch):
        monkeypatch.setattr(h, "_fetch_fresh_health", lambda to_fetch, api_key, max_workers, timeout: {"m": {"best_uptime_30m": 0.99}})
        h._enrich_with_health([{"id": "m"}], "KEY", 0.5, 2, 5.0, 0.0)
        assert pkg._HEALTH_CACHE == {}

    def test_a_ttl_of_half_a_second_still_writes_the_cache(self, monkeypatch):
        monkeypatch.setattr(h, "_fetch_fresh_health", lambda to_fetch, api_key, max_workers, timeout: {"m": {"best_uptime_30m": 0.99}})
        h._enrich_with_health([{"id": "m"}], "KEY", 0.5, 2, 5.0, 0.5)
        assert "m" in pkg._HEALTH_CACHE


class TestMinUptimeFilter:
    def test_a_row_below_the_floor_is_dropped_and_the_scan_goes_on(self, monkeypatch):
        health = {
            "a": {"best_uptime_30m": 0.1},
            "b": {"best_uptime_30m": 0.99},
            "c": {"best_uptime_30m": 0.95},
        }
        monkeypatch.setattr(h, "_fetch_fresh_health", lambda to_fetch, api_key, max_workers, timeout: health)
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        out = h._enrich_with_health(rows, "KEY", 0.95, 2, 5.0, 0.0)
        assert [r["id"] for r in out] == ["b", "c"]


class TestMetricRaw:
    def test_an_empty_completion_price_reads_as_zero(self):
        assert h._metric_raw({"pricing": {"prompt": "0", "completion": ""}}, "output_price") == 0.0

    def test_an_unparsable_price_is_infinitely_expensive_on_both_sides(self):
        row = {"pricing": {"prompt": "abc", "completion": "1"}}
        assert h._metric_raw(row, "input_price") == float("inf")
        assert h._metric_raw(row, "output_price") == float("inf")

    def test_an_unknown_field_falls_back_to_the_row_id(self):
        assert h._metric_raw({"id": "m/1"}, "bogus") == "m/1"
