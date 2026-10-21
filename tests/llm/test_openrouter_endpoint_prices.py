"""Per-endpoint OpenRouter prices: a pinned provider is priced at its own rate, not the cheapest listing's."""

import time

import pytest

from pyutilz.llm.openrouter_provider import _endpoint_prices as ep
from pyutilz.llm.openrouter_provider._endpoint_prices import claimed_quantization, parse_endpoint_prices


def _payload():
    return {
        "data": {
            "endpoints": [
                {
                    "provider_name": "OpenInference",
                    "tag": "open-inference/fp4",
                    "pricing": {"prompt": "0.0000000352", "completion": "0.0000000112", "input_cache_read": "0.0000000016"},
                },
                {
                    "provider_name": "Together",
                    "tag": "together",
                    "pricing": {"prompt": "0.0000003", "completion": "0.0000012", "input_cache_read": "0.000000006"},
                },
                {"provider_name": "Broken", "tag": "broken", "pricing": {"prompt": "n/a", "completion": "0.0000012"}},
                {"provider_name": "Free-looking", "tag": "missing", "pricing": {}},
            ]
        }
    }


def test_each_priced_endpoint_keeps_its_own_per_million_rate():
    prices = {p.tag: p for p in parse_endpoint_prices(_payload())}
    assert prices["together"].output_per_1m == 1.2
    assert prices["together"].input_per_1m == 0.3
    assert prices["open-inference/fp4"].output_per_1m == 0.0112


def test_an_endpoint_without_a_numeric_price_is_skipped_not_priced_at_zero():
    tags = [p.tag for p in parse_endpoint_prices(_payload())]
    assert "broken" not in tags and "missing" not in tags


def test_the_cache_read_rate_is_kept_when_listed_and_none_when_not():
    prices = {p.tag: p for p in parse_endpoint_prices(_payload())}
    assert prices["together"].cache_read_per_1m == 0.006
    assert prices["open-inference/fp4"].cache_read_per_1m == 0.0016


def _quant_payload():
    def one(provider, tag, quant):
        item = {"provider_name": provider, "tag": tag, "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}}
        if quant is not None:
            item["quantization"] = quant
        return item

    return {
        "data": {
            "endpoints": [
                one("DeepInfra", "deepinfra/fp8", "fp8"),
                one("DeepInfra", "deepinfra/fp8/flex", "fp8"),
                one("DeepInfra", "deepinfra/fp4", "fp4"),
                one("Together", "together", "unknown"),
                one("Mystery", "mystery", None),
            ]
        }
    }


@pytest.fixture
def warm_cache(monkeypatch):
    monkeypatch.setattr(ep, "_CACHE", {"vendor/model": (time.monotonic(), parse_endpoint_prices(_quant_payload()))})


def test_the_claimed_quantization_is_parsed_and_an_absent_field_stays_none():
    by_tag = {p.tag: p.quantization for p in parse_endpoint_prices(_quant_payload())}
    assert by_tag["deepinfra/fp8"] == "fp8" and by_tag["together"] == "unknown" and by_tag["mystery"] is None


def test_a_pinned_tag_names_one_endpoint_and_a_slug_names_every_endpoint_under_it(warm_cache):
    assert claimed_quantization("vendor/model", tag="deepinfra/fp8") == "fp8"
    assert claimed_quantization("vendor/model", tag="deepinfra") == "fp4,fp8"


def test_an_unpinned_call_is_looked_up_by_provider_name_and_unknown_is_a_real_answer(warm_cache):
    assert claimed_quantization("vendor/model", provider_name="together") == "unknown"
    assert claimed_quantization("vendor/model", provider_name="DeepInfra") == "fp4,fp8"


def test_nothing_known_is_none_never_a_guess_and_a_cold_cache_makes_no_request(warm_cache, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("claimed_quantization must not touch the network")

    monkeypatch.setattr(ep.httpx, "get", forbidden)
    assert claimed_quantization("vendor/model", provider_name="Mystery") is None
    assert claimed_quantization("other/model", tag="deepinfra/fp8") is None
    assert claimed_quantization("vendor/model") is None


def test_the_provider_warms_its_own_endpoint_cache_so_a_recorded_call_names_the_quantization(monkeypatch):
    import asyncio

    from pyutilz.llm.openrouter_provider import _provider

    monkeypatch.setattr(ep, "_CACHE", {})
    monkeypatch.setattr(_provider, "_ensure_catalogue_warm_async", lambda: asyncio.sleep(0))
    monkeypatch.setattr(ep, "endpoint_prices", lambda model, timeout=10.0: ep._CACHE.setdefault(model, (time.monotonic(), parse_endpoint_prices(_quant_payload())))[1])
    p = _provider.OpenRouterProvider(api_key="k", model="vendor/model", provider_only=("deepinfra",), provider_allow_fallbacks=False)
    assert p.last_quantization is None
    asyncio.run(p._async_prepare())
    assert p.last_quantization == "fp4,fp8"


def test_a_failing_endpoint_fetch_never_breaks_the_call(monkeypatch):
    import asyncio

    from pyutilz.llm.openrouter_provider import _provider

    def boom(model, timeout=10.0):
        raise RuntimeError("network down")

    monkeypatch.setattr(_provider, "_ensure_catalogue_warm_async", lambda: asyncio.sleep(0))
    monkeypatch.setattr(ep, "endpoint_prices", boom)
    p = _provider.OpenRouterProvider(api_key="k", model="vendor/model")
    asyncio.run(p._async_prepare())
    assert p.last_quantization is None
