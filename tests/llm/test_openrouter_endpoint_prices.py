"""Per-endpoint OpenRouter prices: a pinned provider is priced at its own rate, not the cheapest listing's."""

from pyutilz.llm.openrouter_provider._endpoint_prices import parse_endpoint_prices


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
