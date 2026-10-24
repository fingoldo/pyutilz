from __future__ import annotations
from pyutilz.core import jsonlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest
from pyutilz.llm.anthropic_provider import AnthropicProvider
def _message(*blocks: Any, stop: str = "end_turn", usage: Any = None, stop_details: Any = None) -> SimpleNamespace:
    usage = usage or SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage, stop_details=stop_details)
def _text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)
def _anthropic(model: str = "claude-opus-5-5", responses: list[Any] | None = None, **init: Any) -> tuple[AnthropicProvider, list[dict]]:
    p = AnthropicProvider(api_key="sk-fake-test-key-not-real", model=model, **init)
    sent: list[dict] = []
    queue = list(responses or [_message(_text("ok"))])

    async def _create(**kwargs: Any) -> Any:
        sent.append(jsonlib.loads(jsonlib.dumps(kwargs, default=str)))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        raw = MagicMock()
        raw.parse.return_value = item
        raw.headers = {}
        return raw

    p.client = MagicMock()
    p.client.messages.with_raw_response.create = AsyncMock(side_effect=_create)
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt  # type: ignore[method-assign]
    return p, sent
def _deepseek(model: str = "deepseek-flash", **kw: Any) -> Any:
    from pyutilz.llm.deepseek_provider import DeepSeekProvider

    return DeepSeekProvider(api_key="sk-fake", model=model, **kw)  # pragma: allowlist secret -- dummy key
def _xai(model: str = "grok-4.7", handler: Any = None, **kw: Any) -> Any:
    from pyutilz.llm.xai_provider import XAIProvider

    p = XAIProvider(api_key="xai-fake", model=model, **kw)
    if handler is not None:
        p._client = httpx.AsyncClient(base_url="https://api.x.ai/v1", transport=httpx.MockTransport(handler))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p
def _gemini(model: str, response: Any) -> tuple[Any, dict]:
    gp = pytest.importorskip("pyutilz.llm.gemini_provider")
    pytest.importorskip("google.genai")
    p = gp.GeminiProvider(api_key="g-fake", model=model)
    captured: dict = {}

    async def generate_content(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return response

    p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    p.fit_max_tokens_to_context = lambda mt, prompt, system=None: mt
    return p, captured
def _gem_response(text: str = "ok", finish: str = "STOP", candidates: bool = True, feedback: Any = None, prompt_tokens: int = 5) -> Any:
    cand = SimpleNamespace(finish_reason=finish, safety_ratings=[], grounding_metadata=None, citation_metadata=None, content=None)
    um = SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=3, thoughts_token_count=0, cached_content_token_count=0)
    return SimpleNamespace(candidates=[cand] if candidates else [], text=text, usage_metadata=um, prompt_feedback=feedback)
