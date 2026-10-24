from __future__ import annotations
from pyutilz.core import jsonlib
from typing import Any
import httpx
import pytest


def _openai(model: str, handler: Any) -> Any:
    from pyutilz.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider(api_key="sk-fake", model=model)
    p._client = httpx.AsyncClient(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p


def _chat_ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    })


class TestProv10Prov12OpenAIBody:
    @pytest.mark.asyncio
    async def test_reasoning_model_gets_max_completion_tokens_and_no_temperature(self) -> None:
        seen: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(jsonlib.loads(req.content))
            return _chat_ok(seen[-1])

        p = _openai("o3", handler)
        assert await p.generate("q", max_tokens=500, temperature=0.2) == "ok"
        assert seen[0]["max_completion_tokens"] == 500
        assert "max_tokens" not in seen[0]
        assert "temperature" not in seen[0]

    @pytest.mark.asyncio
    async def test_non_reasoning_model_keeps_its_temperature(self) -> None:
        seen: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(jsonlib.loads(req.content))
            return _chat_ok(seen[-1])

        p = _openai("gpt-4o", handler)
        await p.generate("q", max_tokens=50, temperature=0.2)
        assert seen[0]["temperature"] == 0.2 and seen[0]["max_completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_a_temperature_400_is_repaired_and_learned(self) -> None:
        from pyutilz.llm import openai_provider as op

        seen: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            body = jsonlib.loads(req.content)
            seen.append(body)
            if "temperature" in body:
                return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0.2"}})
            return _chat_ok(body)

        p = _openai("gpt-4.1-mini", handler)
        try:
            assert await p.generate("q", max_tokens=50, temperature=0.2) == "ok"
            assert "temperature" in seen[0] and "temperature" not in seen[1]
            assert "gpt-4.1-mini" in op._MODELS_REJECTING_TEMPERATURE
        finally:
            op._MODELS_REJECTING_TEMPERATURE.discard("gpt-4.1-mini")


def _xai(model: str = "grok-4.7", handler: Any = None, **kw: Any) -> Any:
    from pyutilz.llm.xai_provider import XAIProvider

    p = XAIProvider(api_key="xai-fake", model=model, **kw)
    if handler is not None:
        p._client = httpx.AsyncClient(base_url="https://api.x.ai/v1", transport=httpx.MockTransport(handler))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p


class TestProv20XaiSearchTools:
    @pytest.mark.asyncio
    async def test_live_search_uses_the_responses_tools_not_search_parameters(self) -> None:
        seen: list[tuple[str, dict]] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append((req.url.path, jsonlib.loads(req.content)))
            return httpx.Response(200, json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer", "annotations": []}]}],
                "citations": ["https://example.com"],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            })

        p = _xai(handler=handler, live_search=True)
        assert await p.generate("news?", max_tokens=100) == "answer"
        path, body = seen[0]
        assert path.endswith("/responses")
        assert {"type": "web_search"} in body["tools"] and "search_parameters" not in body
        assert p.last_citations == ["https://example.com"]
        assert p.total_prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_without_live_search_it_stays_on_chat_completions(self) -> None:
        seen: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req.url.path)
            return _chat_ok({})

        await _xai(handler=handler).generate("q", max_tokens=10)
        assert seen[0].endswith("/chat/completions")
