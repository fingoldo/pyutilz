from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch
import pytest
def _deepseek(model: str = "deepseek-flash", **kw: Any) -> Any:
    from pyutilz.llm.deepseek_provider import DeepSeekProvider

    return DeepSeekProvider(api_key="sk-fake", model=model, **kw)  # pragma: allowlist secret -- dummy key
class TestProv17DeepSeekOffPeak:
    @pytest.mark.parametrize(
        ("when", "mult"),
        [
            (datetime(2026, 9, 28, 2, 30, tzinfo=timezone.utc), 1.0),  # Monday, inside 01-04
            (datetime(2026, 9, 28, 5, 0, tzinfo=timezone.utc), 0.5),  # Monday, the 04-06 gap
            (datetime(2026, 9, 28, 9, 59, tzinfo=timezone.utc), 1.0),  # Monday, inside 06-10
            (datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc), 0.5),
            (datetime(2026, 9, 26, 2, 30, tzinfo=timezone.utc), 0.5),  # Saturday
        ],
    )
    def test_multiplier(self, when: datetime, mult: float) -> None:
        from pyutilz.llm.deepseek_provider import deepseek_price_multiplier

        assert deepseek_price_multiplier(when) == mult

    def test_off_peak_call_is_billed_at_half(self) -> None:
        p = _deepseek()
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "prompt_cache_hit_tokens": 0}
        with patch("pyutilz.llm.deepseek_provider.deepseek_price_multiplier", return_value=0.5):
            p._record_usage(usage)
        cost = p.get_session_cost()
        assert cost["total_cost_usd"] == pytest.approx((0.30 + 1.20) / 2)
        assert cost["offpeak_discount_usd"] == pytest.approx((0.30 + 1.20) / 2)

    def test_peak_call_is_billed_in_full(self) -> None:
        p = _deepseek()
        with patch("pyutilz.llm.deepseek_provider.deepseek_price_multiplier", return_value=1.0):
            p._record_usage({"prompt_tokens": 1_000_000, "completion_tokens": 0})
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.30)
class TestProv16Prov18DeepSeekFlash:
    def test_flash_limits(self) -> None:
        p = _deepseek()
        # GET /models, 2026-10-03: max_output_tokens 393,216, context_window 1,048,576 (the page rounds to 384K / 1M).
        assert (p.max_output_tokens, p.context_window) == (393_216, 1_048_576)
        assert p._resolve_pricing("deepseek-flash").input == 0.30

    def test_unknown_model_defaults_to_the_current_generation(self) -> None:
        p = _deepseek("deepseek-v5-something")
        assert (p.max_output_tokens, p.context_window) == (393_216, 1_048_576)

    def test_flash_takes_the_thinking_toggle(self) -> None:
        assert _deepseek()._thinking_request_field(True) == {"thinking": {"type": "enabled"}}
        # Live 2026-10-03 the legacy alias is served by deepseek-flash and honours the toggle.
        assert _deepseek("deepseek-chat")._thinking_request_field(True) == {"thinking": {"type": "enabled"}}
