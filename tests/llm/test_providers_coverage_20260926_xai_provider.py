from __future__ import annotations
import httpx
from typing import Any
import pytest


class TestProv21XaiEffortAndTier:
    @pytest.mark.parametrize(
        ("model", "thinking", "expected"),
        [
            ("grok-4.7", "high", {"reasoning_effort": "high"}),
            ("grok-4.7", "max", {"reasoning_effort": "xhigh"}),
            # GET /v1/language-models lists xhigh for grok-4.5 (2026-10-03).
            ("grok-4.5", "xhigh", {"reasoning_effort": "xhigh"}),
            ("grok-4.6", False, {"reasoning_effort": "low"}),
            ("grok-4.6", "minimal", {"reasoning_effort": "low"}),
            ("grok-4-1-fast-reasoning", "high", None),
        ],
    )
    def test_effort(self, model: str, thinking: Any, expected: Any) -> None:
        assert _xai(model)._thinking_request_field(thinking) == expected

    def test_new_models_are_priced(self) -> None:
        p = _xai("grok-4.7")
        assert (p._input_cost_per_1m("grok-4.7"), p._output_cost_per_1m("grok-4.7")) == (2.0, 6.0)
        assert p.context_window == 500_000

    def test_long_context_request_bills_the_doubled_tier(self) -> None:
        p = _xai("grok-4.3")
        p._record_usage({"prompt_tokens": 250_000, "completion_tokens": 0})
        assert p.get_session_cost()["total_cost_usd"] == pytest.approx(2 * 250_000 * 1.25 / 1_000_000)
        q = _xai("grok-4.3")
        q._record_usage({"prompt_tokens": 150_000, "completion_tokens": 0})
        assert q.get_session_cost()["total_cost_usd"] == pytest.approx(150_000 * 1.25 / 1_000_000)


def _xai(model: str = "grok-4.7", handler: Any = None, **kw: Any) -> Any:
    from pyutilz.llm.xai_provider import XAIProvider

    p = XAIProvider(api_key="xai-fake", model=model, **kw)
    if handler is not None:
        p._client = httpx.AsyncClient(base_url="https://api.x.ai/v1", transport=httpx.MockTransport(handler))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p
