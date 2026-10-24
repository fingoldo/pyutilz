from __future__ import annotations
from pyutilz.dev.logginglib import reset_log_throttles
import logging
from typing import Any
import pytest


class TestProv11OpenAIReasoningEffort:
    @pytest.mark.parametrize(
        ("model", "thinking", "expected"),
        [
            ("gpt-5-mini", True, {"reasoning_effort": "medium"}),
            ("gpt-5-mini", "high", {"reasoning_effort": "high"}),
            ("gpt-5-mini", False, {"reasoning_effort": "minimal"}),
            ("gpt-5.1", False, {"reasoning_effort": "none"}),
            ("o3", False, {"reasoning_effort": "low"}),
            ("gpt-6-astra", False, {"reasoning_effort": "low"}),
            ("gpt-4o", "high", None),
            ("gpt-5-chat-latest", "high", None),
        ],
    )
    def test_mapping(self, model: str, thinking: Any, expected: Any) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p.model_name = model
        assert p._thinking_request_field(thinking) == expected


class TestProv13Prov14OpenAIAccounting:
    @pytest.mark.asyncio
    async def test_check_account_limits_returns_the_captured_headers(self) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="sk-fake")
        p.last_rate_limits = {"x-ratelimit-remaining-requests": "99"}
        out = await p.check_account_limits()
        assert out["remaining_requests"] == "99"

    def test_dated_snapshot_prices_by_prefix_without_the_unknown_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        reset_log_throttles()
        with caplog.at_level(logging.WARNING):
            assert p._input_cost_per_1m("gpt-5-pro-2026-01-15") == 15.0
            assert p._cache_hit_cost_per_1m("gpt-5-pro-2026-01-15") == 15.0
        assert not any("unknown" in r.getMessage() for r in caplog.records)

    def test_genuine_miss_still_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        from pyutilz.llm.openai_provider import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        reset_log_throttles()
        with caplog.at_level(logging.WARNING):
            p._input_cost_per_1m("zzz-unknown")
        assert any("unknown" in r.getMessage() for r in caplog.records)
