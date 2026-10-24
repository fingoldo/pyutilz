from __future__ import annotations
import asyncio
import logging
from typing import Any
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


class TestGenerateBatch:
    def test_forwards_optional_options_only_when_present(self):
        provider = _Provider(max_concurrent=4)

        async def run() -> list[dict[str, Any]]:
            reqs = [
                {"id": "a", "prompt": "pa", "images": ["data:image/png;base64,AA=="], "thinking": "high"},
                {"id": "b", "prompt": "pb", "json_mode": True, "json_schema": {"name": "s"}},
                {"id": "c", "prompt": "pc"},
            ]
            return [item async for item in provider.generate_batch(reqs)]

        out = asyncio.run(run())
        assert sorted(o["id"] for o in out if "result" in o) == ["a", "b", "c"]
        by_prompt = {c["prompt"]: c for c in provider.calls}
        assert by_prompt["pa"] == {"prompt": "pa", "images": ["data:image/png;base64,AA=="], "thinking": "high"}
        assert by_prompt["pb"] == {"prompt": "pb", "json_mode": True, "json_schema": {"name": "s"}}
        assert by_prompt["pc"] == {"prompt": "pc"}

    def test_unknown_keys_are_reported(self, caplog):
        provider = _Provider(max_concurrent=4)

        async def run() -> None:
            async for _ in provider.generate_batch([{"id": "a", "prompt": "p", "tools": []}]):
                pass

        with caplog.at_level(logging.WARNING, logger="pyutilz.llm.base"):
            asyncio.run(run())
        assert "['tools']" in caplog.text

    def test_batch_requests_do_not_share_an_outer_stream_counter(self):
        provider = _Provider(max_concurrent=4)

        async def run() -> _progress.StreamProgress:
            with _progress.track_stream_progress() as progress:
                async for _ in provider.generate_batch([{"id": str(i), "prompt": "p"} for i in range(3)]):
                    pass
                _progress.note_stream_progress(None, "own")
            return progress

        progress = asyncio.run(run())
        assert progress.answer_chars == 3
        assert progress.answer_tail == "own"
