"""Exact-text, keyword-only and payload pins for the OpenRouter route fingerprint (mutation survivors)."""

from __future__ import annotations

import pytest

from pyutilz.llm.openrouter_provider._route import (
    _MAPPING_KWARGS,
    _ROUTE_FINGERPRINT_EXCLUDED,
    OpenRouterRouteMixin,
    route_plain,
)


class _Route(OpenRouterRouteMixin):
    def __init__(self, model_name: str, kwargs: dict | None = None) -> None:
        self.model_name = model_name
        self._route_kwargs = kwargs or {}


class TestExcludedReasons:
    @pytest.mark.parametrize(
        "name, reason",
        [
            ("api_key", "credentials, not routing; must never enter a cache key"),
            ("max_concurrent", "client-side concurrency only"),
            ("routing_404_pause_sec", "client-side retry policy"),
        ],
    )
    def test_each_excluded_keyword_keeps_its_reason(self, name, reason):
        assert _ROUTE_FINGERPRINT_EXCLUDED[name] == reason


class TestMappingKeywords:
    def test_the_price_throughput_and_latency_keywords_accept_pair_tuples(self):
        assert {"provider_preferred_min_throughput", "provider_preferred_max_latency", "extra_body"} <= _MAPPING_KWARGS

    @pytest.mark.parametrize("name", ["provider_preferred_min_throughput", "provider_preferred_max_latency", "extra_body"])
    def test_a_pair_tuple_and_a_mapping_fingerprint_identically(self, name):
        as_pairs = _Route("m", {name: (("p50", 1),)})._route_payload()
        as_mapping = _Route("m", {name: {"p50": 1}})._route_payload()
        assert as_pairs == as_mapping
        assert as_pairs[name] == {"p50": 1}


class TestRoutePlainSignature:
    def test_the_mapping_flag_is_keyword_only(self):
        with pytest.raises(TypeError):
            route_plain({"a": 1}, True)

    def test_the_mapping_flag_defaults_to_off_so_pairs_stay_lists(self):
        assert route_plain((("a", 1),)) == [["a", 1]]

    def test_the_mapping_flag_turns_pairs_into_a_dict(self):
        assert route_plain((("a", 1),), mapping=True) == {"a": 1}


class TestPayloadHeader:
    def test_the_payload_names_the_provider_and_the_live_model(self):
        payload = _Route("openai/gpt-x")._route_payload()
        assert payload["provider"] == "openrouter"
        assert payload["model"] == "openai/gpt-x"

    def test_a_reassigned_model_name_is_read_live(self):
        route = _Route("a/first")
        route.model_name = "a/second"
        assert route._route_payload()["model"] == "a/second"
