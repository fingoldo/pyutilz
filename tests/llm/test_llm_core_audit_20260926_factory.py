from __future__ import annotations
import asyncio
from typing import Any
import pytest
from pyutilz.llm import _progress
from pyutilz.llm.base import LLMProvider


class _Provider(LLMProvider):
    """Minimal concrete provider: records every generate() call's kwargs."""

    _PRICING = {"known-model": (1.0, 2.0), "fam-model": (3.0, 4.0)}

    def __init__(self, max_concurrent: int = 1, model: str = "known-model") -> None:
        self._max_concurrent = max_concurrent
        self.model_name = model
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt: str, system: str | None = None, temperature: float | None = 0.7, max_tokens: int = 0, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        async with self.semaphore:
            _progress.note_stream_progress(None, "x" * 10)
            await asyncio.sleep(0.01)
        return prompt

    async def count_tokens(self, text: str) -> int:
        return len(text)


class TestShutdownClose:
    def _provider_with_client(self) -> tuple[_Provider, list[str]]:
        httpx = pytest.importorskip("httpx")
        closed: list[str] = []
        provider = _Provider()
        provider._client = httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(lambda r: httpx.Response(200)))

        async def _close() -> None:
            closed.append("close")
            await provider._client.aclose()

        provider._close = _close  # type: ignore[method-assign]
        return provider, closed

    def test_unused_provider_closes_its_own_client_on_the_shutdown_loop(self, monkeypatch):
        from pyutilz.llm import factory

        provider, closed = self._provider_with_client()
        original = provider.__dict__["_client"]
        monkeypatch.setattr(factory, "_provider_cache", {("x",): provider})
        factory._close_cached_providers()
        assert closed == ["close"]
        assert original.is_closed  # the original, not a clone built for the shutdown loop

    def test_clones_are_closed_and_dead_home_loop_is_skipped(self, monkeypatch):
        from pyutilz.llm import factory

        provider, closed = self._provider_with_client()

        async def grab() -> Any:
            return provider._client

        asyncio.run(grab())  # home loop, closed after this
        loop = asyncio.new_event_loop()
        try:
            clone = loop.run_until_complete(grab())
            monkeypatch.setattr(factory, "_provider_cache", {("x",): provider})
            factory._close_cached_providers()
            assert closed == []  # its home loop is closed: its sockets cannot be closed from elsewhere
            assert clone.is_closed
        finally:
            loop.close()
