"""Branch, bound, log and URL pins for the OpenRouter health helpers (mutation survivors, first batch)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest

import pyutilz.llm.openrouter_provider as pkg
from pyutilz.llm.openrouter_provider import _health as h


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(pkg, "get_llm_settings", lambda: SimpleNamespace(openrouter_api_key=None))


class TestResolveApiKey:
    def test_an_explicit_key_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        assert h._resolve_or_api_key("explicit") == "explicit"

    def test_the_environment_key_is_used_when_no_explicit_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        monkeypatch.setattr(pkg, "get_llm_settings", lambda: SimpleNamespace(openrouter_api_key=_Secret("from-settings")))
        assert h._resolve_or_api_key(None) == "from-env"

    def test_the_settings_key_is_used_when_nothing_else_is_set(self, monkeypatch):
        monkeypatch.setattr(pkg, "get_llm_settings", lambda: SimpleNamespace(openrouter_api_key=_Secret("from-settings")))
        assert h._resolve_or_api_key(None) == "from-settings"

    def test_nothing_configured_gives_none(self):
        assert h._resolve_or_api_key(None) is None

    def test_a_settings_failure_is_logged_at_debug_and_gives_none(self, monkeypatch, caplog):
        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(pkg, "get_llm_settings", boom)
        caplog.set_level(logging.DEBUG)
        assert h._resolve_or_api_key(None) is None
        assert "Could not load OR API key from settings: boom" in caplog.text


class TestSweepHealthCache:
    @staticmethod
    def _fill(monkeypatch, entries: dict, max_size: int) -> dict:
        monkeypatch.setattr(pkg, "_HEALTH_CACHE_MAX_SIZE", max_size)
        monkeypatch.setattr(pkg, "_HEALTH_CACHE", entries)
        return entries

    def test_a_cache_under_the_cap_is_left_alone(self, monkeypatch):
        cache = self._fill(monkeypatch, {"a": (0.0, {}), "b": (1.0, {})}, 10)
        h._sweep_health_cache_locked(1_000_000.0)
        assert set(cache) == {"a", "b"}

    def test_a_cache_exactly_at_the_cap_keeps_its_stale_entries(self, monkeypatch):
        now = 200_000.0
        cache = self._fill(monkeypatch, {"stale": (now - 86_400.0 - 1.0, {}), "fresh": (now, {})}, 2)
        h._sweep_health_cache_locked(now)
        assert set(cache) == {"stale", "fresh"}

    def test_an_entry_older_than_a_day_is_evicted_when_over_the_cap(self, monkeypatch):
        now = 200_000.0
        entries = {"stale": (now - 90_000.0, {})}
        entries.update({f"f{i}": (now, {}) for i in range(200)})
        cache = self._fill(monkeypatch, entries, 200)
        h._sweep_health_cache_locked(now)
        assert "stale" not in cache
        assert len(cache) == 200

    def test_the_day_boundary_is_strict(self, monkeypatch):
        now = 200_000.0
        entries = {"edge": (now - 86_400.0, {})}
        entries.update({f"f{i}": (now, {}) for i in range(200)})
        cache = self._fill(monkeypatch, entries, 200)
        h._sweep_health_cache_locked(now)
        assert len(cache) == 200 - 128

    def test_a_just_older_than_a_day_entry_is_evicted(self, monkeypatch):
        now = 200_000.0
        entries = {"just": (now - 86_400.05, {})}
        entries.update({f"f{i}": (now, {}) for i in range(200)})
        cache = self._fill(monkeypatch, entries, 200)
        h._sweep_health_cache_locked(now)
        assert "just" not in cache
        assert len(cache) == 200

    def test_an_over_cap_cache_is_trimmed_to_the_cap_less_128(self, monkeypatch):
        now = 200_000.0
        entries = {f"k{i}": (now - i, {}) for i in range(201)}
        cache = self._fill(monkeypatch, entries, 200)
        h._sweep_health_cache_locked(now)
        assert len(cache) == 200 - 128

    def test_the_oldest_entries_are_dropped_first(self, monkeypatch):
        now = 200_000.0
        entries = {f"k{i}": (now - i, {}) for i in range(201)}
        cache = self._fill(monkeypatch, entries, 200)
        h._sweep_health_cache_locked(now)
        assert "k0" in cache
        assert "k200" not in cache

    def test_exactly_at_the_cap_after_the_stale_pass_drops_nothing_more(self, monkeypatch):
        now = 200_000.0
        entries = {"stale": (now - 90_000.0, {})}
        entries.update({f"f{i}": (now, {}) for i in range(200)})
        cache = self._fill(monkeypatch, entries, 200)
        h._sweep_health_cache_locked(now)
        assert len(cache) == 200
        assert "stale" not in cache


class TestEndpointsFetch:
    @staticmethod
    def _client(payload, status: int = 200):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(status, json=payload)

        return httpx.Client(transport=httpx.MockTransport(handler)), seen

    def test_a_supplied_client_is_used_and_its_endpoints_are_returned(self, monkeypatch):
        client, seen = self._client({"data": {"endpoints": [{"x": 1}]}})
        monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("module-level get must not be used"))
        out = h._fetch_endpoints_for_model("m1", client=client)
        assert out == [{"x": 1}]
        assert seen[0].url.path == "/api/v1/models/m1/endpoints"

    def test_without_a_client_an_api_key_drives_a_one_shot_get_with_the_bearer_header(self, monkeypatch):
        calls = []

        def fake_get(url, headers, timeout):
            calls.append((url, headers, timeout))
            return httpx.Response(200, json={"data": {"endpoints": []}}, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        assert h._fetch_endpoints_for_model("m1", api_key="KEY") == []  # pragma: allowlist secret
        assert calls == [(pkg._ENDPOINTS_URL_TEMPLATE.format(model="m1"), {"Authorization": "Bearer KEY"}, 30.0)]

    def test_without_a_client_or_a_key_the_error_names_both_options(self):
        with pytest.raises(ValueError) as info:
            h._fetch_endpoints_for_model("m1")
        assert str(info.value) == "_fetch_endpoints_for_model: pass either api_key= or a preconfigured client= with auth headers."

    def test_an_http_error_is_raised(self):
        client, _ = self._client({}, status=500)
        with pytest.raises(httpx.HTTPStatusError):
            h._fetch_endpoints_for_model("m1", client=client)

    def test_a_payload_without_endpoints_gives_an_empty_list(self):
        client, _ = self._client({"data": {}})
        assert h._fetch_endpoints_for_model("m1", client=client) == []


class TestUptimeShape:
    def test_a_value_above_one_marks_the_whole_payload_as_percentages(self):
        assert h._uptime_payload_is_percentage([{"uptime_last_5m": 1.05}]) is True

    def test_a_value_of_exactly_one_is_not_a_percentage_shape(self):
        assert h._uptime_payload_is_percentage([{"uptime_last_5m": 1.0}]) is False

    def test_a_non_numeric_field_is_skipped_and_the_scan_goes_on(self):
        endpoints = [{"uptime_last_5m": "n/a", "uptime_last_30m": 99.5}]
        assert h._uptime_payload_is_percentage(endpoints) is True


class TestNormalizeUptime:
    def test_a_non_numeric_value_is_none_and_logged_at_debug(self, caplog):
        caplog.set_level(logging.DEBUG)
        assert h._normalize_uptime("abc") is None
        assert "_normalize_uptime: uptime is not numeric: ValueError: " in caplog.text

    def test_a_value_above_one_hundred_is_out_of_range(self):
        assert h._normalize_uptime(150) is None

    def test_a_value_just_above_one_hundred_is_out_of_range(self):
        assert h._normalize_uptime(100.05) is None

    def test_a_value_of_exactly_one_hundred_is_a_full_fraction_of_one(self):
        assert h._normalize_uptime(100.0) == pytest.approx(1.0)

    def test_an_out_of_range_value_is_warned_with_the_exact_message(self, caplog):
        caplog.set_level(logging.WARNING)
        assert h._normalize_uptime(150) is None
        assert caplog.records[-1].getMessage() == "OpenRouter uptime out of expected [0, 100] range: 150; treating as None."

    def test_a_fraction_payload_is_returned_unscaled(self):
        assert h._normalize_uptime(0.98, is_percentage=False) == pytest.approx(0.98)

    def test_a_percentage_payload_is_divided_by_one_hundred(self):
        assert h._normalize_uptime(98.0, is_percentage=True) == pytest.approx(0.98)

    def test_a_missing_value_is_none(self):
        assert h._normalize_uptime(None) is None
