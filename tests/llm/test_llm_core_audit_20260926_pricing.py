from __future__ import annotations
from typing import Any
import pytest


def _compat_provider(**totals: int) -> Any:
    from pyutilz.llm._pricing import Pricing
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _P(OpenAICompatibleProvider):
        _base_url = "https://example.test"
        _provider_name = "T"

        def _resolve_pricing(self, model: str) -> Pricing:
            return Pricing(2.0, 10.0, cache_hit=0.2, cache_write=2.5)

        def _input_cost_per_1m(self, model: str) -> float:
            return 2.0

        def _output_cost_per_1m(self, model: str) -> float:
            return 10.0

    p = _P(api_key="k", model="m")
    p.total_prompt_tokens = totals.get("prompt", 0)
    p.total_cache_hit_tokens = totals.get("hit", 0)
    p.total_completion_tokens = 0
    p.total_reasoning_tokens = 0
    if "write" in totals:
        p.total_cache_write_tokens = totals["write"]
    return p


class TestCacheWritePricing:
    def test_pricing_has_optional_cache_write(self):
        from pyutilz.llm._pricing import Pricing

        assert Pricing(1.0, 2.0).cache_write is None
        assert Pricing(1.0, 2.0, 0.1, 1.25).cache_write == 1.25

    def test_session_cost_bills_writes_at_the_write_rate(self):
        p = _compat_provider(prompt=3_000_000, hit=1_000_000, write=1_000_000)
        cost = p.get_session_cost()
        # 1M miss at 2.0 + 1M hit at 0.2 + 1M write at 2.5
        assert cost["input_cost_usd"] == pytest.approx(4.7)
        assert cost["cache_miss_tokens"] == 1_000_000

    def test_no_write_tracking_is_unchanged(self):
        p = _compat_provider(prompt=2_000_000, hit=1_000_000)
        assert p.get_session_cost()["input_cost_usd"] == pytest.approx(2.2)

    def test_write_rate_falls_back_to_input(self):
        from pyutilz.llm._pricing import Pricing

        p = _compat_provider()
        p._resolve_pricing = lambda model: Pricing(2.0, 10.0)  # type: ignore[method-assign]
        assert p._cache_write_cost_per_1m("m") == 2.0
