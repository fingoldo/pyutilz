"""Anthropic Claude LLM provider."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import anthropic
from tenacity import retry, retry_if_exception_type

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.exceptions import LLMProviderError, JSONParsingError
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with async support and retry logic."""

    # Pricing per 1M tokens: (input, output)
    # Source: https://platform.claude.com/docs/en/about-claude/pricing
    _PRICING: dict[str, tuple[float, float]] = {
        "claude-opus-4-6-20250610":    (5.00, 25.00),
        "claude-opus-4-5-20250414":    (5.00, 25.00),
        "claude-opus-4-20250514":      (15.00, 75.00),
        "claude-sonnet-4-6-20250610":  (3.00, 15.00),
        "claude-sonnet-4-5-20250514":  (3.00, 15.00),
        "claude-sonnet-4-20250514":    (3.00, 15.00),
        "claude-haiku-4-5-20251001":   (1.00, 5.00),
        "claude-haiku-3-5-20241022":   (0.80, 4.00),
        "claude-haiku-3-20240307":     (0.25, 1.25),
    }
    _DEFAULT_PRICING = (3.00, 15.00)  # fallback = Sonnet pricing

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_concurrent: int = 5,
    ):
        settings = get_llm_settings()
        self.api_key = api_key or (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key
            else None
        )
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        # Per-call usage (read by LLMClient after generate())
        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    @property
    def max_output_tokens(self) -> int:
        # Source: https://platform.claude.com/docs/en/docs/about-claude/models
        #   Opus 4.6: 128K, Opus 4/4.1: 32K, Sonnet 4.6: 64K, Haiku 4.5: 64K
        if self.model.startswith("claude-opus"):
            return 128000 if "4-6" in self.model or "opus-4-6" in self.model else 32000
        if self.model.startswith("claude-sonnet"):
            return 64000
        if self.model.startswith("claude-haiku"):
            return 64000
        return 64000

    @property
    def context_window(self) -> int:
        return 200_000

    @retry(
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        )),
        **INFINITE_RETRY_KWARGS,
    )
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
    ) -> str:
        """Generate text using Claude."""
        if max_tokens <= 0:
            max_tokens = min(self.max_output_tokens, 21000)
        async with self.semaphore:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            if system:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            response = await self.client.messages.create(**kwargs)
            self._last_finish_reason = response.stop_reason

            self._last_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning_tokens": 0,
            }

            return response.content[0].text

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
    ) -> dict[str, Any]:
        """Generate structured JSON output."""
        json_system = (system or "") + "\n\nRespond with valid JSON only."
        text = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self.extract_json(text, "Anthropic")

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate responses in batch using concurrent requests."""
        async def process_request(req: dict) -> dict:
            request_id = req.get("id", "unknown")
            try:
                result = await self.generate(
                    prompt=req["prompt"],
                    system=req.get("system"),
                    temperature=req.get("temperature", 0.7),
                    max_tokens=req.get("max_tokens", 1024),
                )
                return {"id": request_id, "result": result}
            except anthropic.RateLimitError as e:
                logger.error(f"Batch request {request_id} rate limited: {e}")
                return {"id": request_id, "error": f"Rate limited: {e}"}
            except anthropic.APIConnectionError as e:
                logger.error(f"Batch request {request_id} connection error: {e}")
                return {"id": request_id, "error": f"Connection error: {e}"}
            except anthropic.APIStatusError as e:
                logger.error(f"Batch request {request_id} API error: {e}")
                return {"id": request_id, "error": f"API error: {e}"}

        tasks = [process_request(req) for req in requests]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    def _get_pricing(self) -> tuple[float, float]:
        """Return (input_cost_per_1m, output_cost_per_1m) for current model."""
        pricing = self._PRICING.get(self.model)
        if pricing:
            return pricing
        for key, val in self._PRICING.items():
            if self.model.startswith(key.rsplit("-", 1)[0]):
                return val
        return self._DEFAULT_PRICING

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD based on model-specific pricing."""
        inp_rate, out_rate = self._get_pricing()
        input_cost = (input_tokens / 1_000_000) * inp_rate
        output_cost = (output_tokens / 1_000_000) * out_rate
        return input_cost + output_cost

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (accurate) or len//4 fallback."""
        from pyutilz.llm.token_counter import count_tokens
        return count_tokens(text)
