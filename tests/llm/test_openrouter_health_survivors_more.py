"""Summary, cache-split, fetch, write, enrich, price and metric pins for the OpenRouter health helpers (second batch)."""

from __future__ import annotations

import logging

import pytest

import pyutilz.llm.openrouter_provider as pkg
from pyutilz.llm.openrouter_provider import _health as h


class TestSummarizeEndpoints:
    def test_each_normalised_field_is_carried_through_under_its_key(self):
        ep = {
            "provider_name": "P",
            "name": "Up",
            "status": "ok",
            "context_length": 8000,
            "max_completion_tokens": 512,
            "max_prompt_tokens": 7000,
            "pricing": {"prompt": "0.1"},
            "supported_parameters": ["temperature"],
            "quantization": "fp8",
            "supports_implicit_caching": True,
        }
        row = h._summarize_endpoints([ep])["endpoints"][0]
        assert row["name"] == "Up"
        assert row["status"] == "ok"
        assert row["max_completion_tokens"] == 512
        assert row["max_prompt_tokens"] == 7000
        assert row["pricing"] == {"prompt": "0.1"}
        assert row["quantization"] == "fp8"

    def test_the_uptime_windows_are_read_from_their_keys(self):
        ep = {"uptime_last_5m": 0.5, "uptime_last_30m": 0.9, "uptime_last_1d": 0.7}
        row = h._summarize_endpoints([ep])["endpoints"][0]
        assert (row["uptime_5m"], row["uptime_30m"], row["uptime_1d"]) == (0.5, 0.9, 0.7)

    def test_the_best_uptime_is_the_maximum_over_upstreams(self):
        eps = [{"uptime_last_30m": 0.9}, {"uptime_last_30m": 0.97}]
        assert h._summarize_endpoints(eps)["best_uptime_30m"] == pytest.approx(0.97)

    def test_the_latency_p95_falls_back_to_p90_only_when_p95_is_none(self):
        eps = [{"latency_last_30m": {"p95": 0, "p90": 9}}]
        assert h._summarize_endpoints(eps)["endpoints"][0]["latency_p95_ms"] == 0
        eps = [{"latency_last_30m": {"p90": 9}}]
        assert h._summarize_endpoints(eps)["endpoints"][0]["latency_p95_ms"] == 9

    def test_no_endpoints_gives_empty_aggregates(self):
        out = h._summarize_endpoints([])
        assert out == {
            "endpoints": [],
            "best_uptime_30m": None,
            "best_latency_p50_ms": None,
            "best_throughput_p50_tps": None,
        }


class TestSplitCachedRows:
    def test_a_fresh_hit_is_served_from_the_cache(self, monkeypatch):
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"m": (900.0, {"best_uptime_30m": 1.0})})
        cached, to_fetch = h._split_cached_rows([{"id": "m"}], 300.0, 1000.0)
        assert cached == {"m": {"best_uptime_30m": 1.0}}
        assert to_fetch == []

    def test_an_entry_exactly_at_the_ttl_is_refetched(self, monkeypatch):
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"m": (700.0, {})})
        cached, to_fetch = h._split_cached_rows([{"id": "m"}], 300.0, 1000.0)
        assert cached == {}
        assert [r["id"] for r in to_fetch] == ["m"]

    def test_a_zero_ttl_bypasses_the_cache_entirely(self, monkeypatch):
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"m": (999.9, {})})
        cached, to_fetch = h._split_cached_rows([{"id": "m"}], 0, 1000.0)
        assert cached == {}
        assert len(to_fetch) == 1

    def test_a_tiny_positive_ttl_still_uses_the_cache(self, monkeypatch):
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"m": (999.9, {"x": 1})})
        cached, _ = h._split_cached_rows([{"id": "m"}], 0.5, 1000.0)
        assert cached == {"m": {"x": 1}}


class TestFetchFreshHealth:
    def test_the_fetch_receives_the_row_id_none_key_timeout_and_client(self, monkeypatch):
        seen = []

        def fake(model_id, api_key=None, timeout=30.0, client=None):
            seen.append((model_id, api_key, timeout, client is not None))
            return []

        monkeypatch.setattr(h, "_fetch_endpoints_for_model", fake)
        h._fetch_fresh_health([{"id": "m1"}], "KEY", 1, 12.5)
        assert seen == [("m1", None, 12.5, True)]

    def test_a_failed_row_does_not_stop_the_others(self, monkeypatch):
        def fake(model_id, api_key=None, timeout=30.0, client=None):
            if model_id == "bad":
                raise RuntimeError("boom")
            return [{"uptime_last_30m": 0.99}]

        monkeypatch.setattr(h, "_fetch_endpoints_for_model", fake)
        out = h._fetch_fresh_health([{"id": "bad"}, {"id": "good"}], "KEY", 1, 5.0)
        assert list(out) == ["good"]

    def test_a_failure_is_logged_at_debug_per_model(self, monkeypatch, caplog):
        def fake(model_id, api_key=None, timeout=30.0, client=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(h, "_fetch_endpoints_for_model", fake)
        caplog.set_level(logging.DEBUG)
        h._fetch_fresh_health([{"id": "bad"}], "KEY", 1, 5.0)
        assert "Health check failed for bad (boom); excluding from results." in caplog.text

    def test_the_aggregate_warning_names_the_counts_and_the_example(self, monkeypatch, caplog):
        def fake(model_id, api_key=None, timeout=30.0, client=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(h, "_fetch_endpoints_for_model", fake)
        caplog.set_level(logging.WARNING)
        h._fetch_fresh_health([{"id": "bad"}, {"id": "worse"}], "KEY", 1, 5.0)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].startswith("Health check failed for 2/2 models, e.g. bad: boom")


class TestWriteHealthCache:
    def test_a_newer_entry_written_by_a_parallel_call_is_kept(self, monkeypatch):
        monkeypatch.setattr(h.time, "monotonic", lambda: 1000.0)
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"m": (1000.5, {"from": "parallel"})})
        h._write_health_cache({"m": {"from": "mine"}})
        assert pkg._HEALTH_CACHE["m"] == (1000.5, {"from": "parallel"})

    def test_an_older_entry_is_overwritten(self, monkeypatch):
        monkeypatch.setattr(h.time, "monotonic", lambda: 1000.0)
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"m": (999.0, {"from": "old"})})
        h._write_health_cache({"m": {"from": "mine"}})
        assert pkg._HEALTH_CACHE["m"] == (1000.0, {"from": "mine"})

    def test_an_entry_stamped_at_the_write_time_is_kept_not_replaced(self, monkeypatch):
        monkeypatch.setattr(h.time, "monotonic", lambda: 1000.0)
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"m": (1000.0, {"from": "same"})})
        h._write_health_cache({"m": {"from": "mine"}})
        assert pkg._HEALTH_CACHE["m"][1] == {"from": "same"}

    def test_a_newer_entry_for_one_id_does_not_stop_the_others_being_written(self, monkeypatch):
        monkeypatch.setattr(h.time, "monotonic", lambda: 1000.0)
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"a": (1000.5, {"from": "parallel"})})
        h._write_health_cache({"a": {"from": "mine"}, "b": {"from": "mine"}})
        assert pkg._HEALTH_CACHE["b"] == (1000.0, {"from": "mine"})

    def test_the_write_runs_the_stale_sweep(self, monkeypatch):
        monkeypatch.setattr(h.time, "monotonic", lambda: 200_000.0)
        monkeypatch.setattr(pkg, "_HEALTH_CACHE_MAX_SIZE", 200)
        entries = {f"old{i}": (-100_000.0, {}) for i in range(201)}
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", entries)
        h._write_health_cache({"new": {}})
        assert len(pkg._HEALTH_CACHE) <= 200
        assert "new" in pkg._HEALTH_CACHE


class TestEnrichWithHealth:
    def test_a_cache_hit_is_logged_at_debug(self, monkeypatch, caplog):
        monkeypatch.setattr(h.time, "monotonic", lambda: 1000.0)
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"m": (999.0, {"best_uptime_30m": 1.0})})
        caplog.set_level(logging.DEBUG)
        out = h._enrich_with_health([{"id": "m"}], "KEY", 0.5, 1, 5.0, 300.0)
        assert [r["id"] for r in out] == ["m"]
        assert "OpenRouter health cache: 1 hit, 0 to fetch" in caplog.text

    def test_a_row_without_health_is_skipped_and_the_rest_still_come_through(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("x")

        monkeypatch.setattr(h, "_fetch_endpoints_for_model", boom)
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", {"good": (999.0, {"best_uptime_30m": 1.0})})
        monkeypatch.setattr(h.time, "monotonic", lambda: 1000.0)
        out = h._enrich_with_health([{"id": "missing"}, {"id": "good"}], "KEY", 0.5, 1, 5.0, 300.0)
        assert [r["id"] for r in out] == ["good"]


class TestPricePer1m:
    def test_the_price_is_scaled_to_per_million(self):
        assert h._price_per_1m({"pricing": {"prompt": "0.000001"}}, "prompt") == pytest.approx(1.0)

    def test_an_empty_price_string_reads_as_zero(self):
        assert h._price_per_1m({"pricing": {"prompt": ""}}, "prompt") == 0.0

    def test_a_missing_price_reads_as_zero(self):
        assert h._price_per_1m({"pricing": {}}, "prompt") == 0.0

    def test_a_non_numeric_price_is_infinitely_expensive_and_logged(self, caplog):
        caplog.set_level(logging.WARNING)
        assert h._price_per_1m({"pricing": {"prompt": "abc"}}, "prompt") == float("inf")
        assert "_price_per_1m: pricing is not numeric, endpoint ranked as infinitely expensive: ValueError: " in caplog.text


class TestMetricRaw:
    @staticmethod
    def _row(**kw):
        base = {"id": "vendor/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "context_length": 8000}
        base.update(kw)
        return base

    def test_the_name_field_is_the_row_id(self):
        assert h._metric_raw(self._row(), "name") == "vendor/model"

    def test_a_missing_context_is_zero(self):
        assert h._metric_raw({"id": "x"}, "context") == 0

    def test_the_input_and_output_prices_are_per_token_values(self):
        assert h._metric_raw(self._row(), "input_price") == pytest.approx(0.000001)
        assert h._metric_raw(self._row(), "output_price") == pytest.approx(0.000002)

    def test_an_empty_prompt_price_reads_as_zero_not_infinite(self):
        assert h._metric_raw(self._row(pricing={"prompt": ""}), "input_price") == 0.0

    def test_a_missing_uptime_is_zero(self):
        assert h._metric_raw(self._row(), "uptime") == 0.0

    def test_the_uptime_is_read_from_the_health_block(self):
        assert h._metric_raw(self._row(health={"best_uptime_30m": 0.99}), "uptime") == pytest.approx(0.99)

    def test_a_missing_throughput_is_zero(self):
        assert h._metric_raw(self._row(health={}), "throughput") == 0.0

    def test_the_throughput_is_read_from_the_health_block(self):
        assert h._metric_raw(self._row(health={"best_throughput_p50_tps": 42.0}), "throughput") == 42.0

    def test_a_missing_latency_is_infinite_and_a_zero_latency_is_zero(self):
        assert h._metric_raw(self._row(health={}), "latency") == float("inf")
        assert h._metric_raw(self._row(health={"best_latency_p50_ms": 0}), "latency") == 0.0


class TestTopRowsPerProvider:
    def test_only_the_top_rows_of_each_provider_are_kept(self):
        rows = [
            {"id": "a/1", "pricing": {"prompt": "3"}},
            {"id": "a/2", "pricing": {"prompt": "1"}},
            {"id": "a/3", "pricing": {"prompt": "2"}},
            {"id": "b/1", "pricing": {"prompt": "5"}},
        ]
        out = h._top_rows_per_provider(rows, 2, "input_price", False)
        assert sorted(r["id"] for r in out) == ["a/2", "a/3", "b/1"]

    def test_the_provider_is_the_text_before_the_first_slash(self):
        rows = [{"id": "x/y/z", "pricing": {"prompt": "1"}}, {"id": "x/w", "pricing": {"prompt": "2"}}]
        out = h._top_rows_per_provider(rows, 1, "input_price", False)
        assert [r["id"] for r in out] == ["x/y/z"]
