from __future__ import annotations
import asyncio
from typing import Any
from pyutilz.llm import _progress
from pyutilz.llm.base import LLMProvider, PerCallAttr
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
class _Holder:
    value = PerCallAttr(lambda: None)
class _Slotted:
    __slots__ = ()
    value = PerCallAttr(lambda: None)
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
