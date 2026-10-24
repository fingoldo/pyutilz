from __future__ import annotations
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import httpx
import pytest


class TestProv23GeminiTransport:
    def test_httpx_transport_errors_are_retryable(self) -> None:
        gp = pytest.importorskip("pyutilz.llm.gemini_provider")
        assert httpx.ConnectError in gp._TRANSIENT_TRANSPORT_ERRORS or any(issubclass(httpx.ConnectError, t) for t in gp._TRANSIENT_TRANSPORT_ERRORS)
        assert not issubclass(httpx.ConnectError, OSError), "the reason OSError alone did not cover it"

    def test_client_has_an_explicit_timeout(self) -> None:
        gp = pytest.importorskip("pyutilz.llm.gemini_provider")
        pytest.importorskip("google.genai")
        with patch.object(gp.genai, "Client") as client_cls:
            gp.GeminiProvider(api_key="g-fake")  # pragma: allowlist secret -- dummy key
        options = client_cls.call_args.kwargs["http_options"]
        assert options.timeout == 600_000


class TestProv22Prov24GeminiRequest:
    @pytest.mark.asyncio
    async def test_gemini3_request_carries_thinking_level(self) -> None:
        p, captured = _gemini("gemini-3-flash-preview", _gem_response())
        await p.generate.__wrapped__(p, "q", max_tokens=100, thinking="low")
        tc = captured["config"].thinking_config
        assert str(tc.thinking_level).upper().endswith("LOW") and tc.thinking_budget is None

    @pytest.mark.asyncio
    async def test_json_schema_is_sent_as_response_json_schema(self) -> None:
        p, captured = _gemini("gemini-2.5-flash", _gem_response('{"a": 1}'))
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        assert await p.generate_json("q", json_schema={"name": "n", "strict": True, "schema": schema}) == {"a": 1}
        cfg = captured["config"]
        assert cfg.response_json_schema == schema and cfg.response_mime_type == "application/json"
        assert p.supports_json_schema() is True


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
