"""Base class for OpenAI-compatible LLM providers (DeepSeek, xAI, etc.).

Consolidates the shared HTTP + retry + token-tracking logic that was
duplicated across deepseek_provider.py and xai_provider.py.
Subclasses override a small set of hooks for provider-specific behaviour.
"""

import asyncio
import json
import logging
from abc import abstractmethod
from typing import Any, AsyncIterator

import httpx
from tenacity import retry, retry_if_exception

from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for transient HTTP errors that should be retried infinitely.

    Non-retryable: 400 (bad request), 401 (auth), 403 (forbidden).
    Retryable: 402 (billing), 429 (rate limit), 5xx, transport errors.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in (400, 401, 403)
    return isinstance(exc, httpx.TransportError)


class OpenAICompatibleProvider(LLMProvider):
    """Base for providers exposing an OpenAI-compatible chat/completions API.

    Subclasses MUST define:
        _base_url          — e.g. "https://api.deepseek.com"
        _provider_name     — e.g. "DeepSeek"
        _max_tokens_map    — dict[model_name, max_output_tokens]
        _default_max_tokens — fallback when model not in map
        _input_cost_per_1m(model)  — input pricing
        _output_cost_per_1m(model) — output pricing
        _cache_hit_cost_per_1m(model) — cache-hit input pricing

    Subclasses MAY override:
        _handle_special_status(resp) — raise on provider-specific HTTP codes
        _compute_billed_output(compl, reasoning) — billing semantics
        _get_timeout(model) — request timeout in seconds
    """

    # ── subclass configuration (override in subclass) ────────────────
    _base_url: str
    _provider_name: str
    _max_tokens_map: dict[str, int] = {}
    _default_max_tokens: int = 8192

    def __init__(
        self,
        api_key: str,
        model: str,
        max_concurrent: int = 10,
    ):
        self.api_key = api_key
        self.model_name = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

        timeout_s = self._get_timeout(model)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )

        # Cumulative token counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cache_hit_tokens = 0
        self.total_reasoning_tokens = 0
        self._call_count = 0
        # Per-call usage (read by LLMClient after generate())
        self._last_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

    # ── hooks for subclasses ─────────────────────────────────────────

    def _get_timeout(self, model: str) -> float:
        """Return request timeout in seconds. Override for model-specific."""
        return 120.0

    def _handle_special_status(self, resp: httpx.Response) -> None:
        """Raise on provider-specific non-retryable HTTP status codes.

        Called *before* the generic raise_for_status().
        """
        pass

    def _compute_billed_output(
        self, completion_tokens: int, reasoning_tokens: int
    ) -> int:
        """Return the number of output tokens that count toward billing.

        DeepSeek: reasoning ⊂ completion → return completion_tokens
        xAI:      reasoning separate     → return completion + reasoning
        """
        return completion_tokens

    @abstractmethod
    def _input_cost_per_1m(self, model: str) -> float: ...

    @abstractmethod
    def _output_cost_per_1m(self, model: str) -> float: ...

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        """Override for providers with cache-hit pricing."""
        return self._input_cost_per_1m(model)

    # ── LLMProvider interface ────────────────────────────────────────

    @property
    def max_output_tokens(self) -> int:
        return self._max_tokens_map.get(self.model_name, self._default_max_tokens)

    # Subclasses override for per-model context windows
    _context_window_map: dict[str, int] = {}
    _default_context_window: int = 128_000

    @property
    def context_window(self) -> int:
        return self._context_window_map.get(self.model_name, self._default_context_window)

    async def _close(self):
        await self._client.aclose()

    def _build_messages(
        self,
        prompt: str,
        system: str | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @retry(
        retry=retry_if_exception(_is_retryable_http_error),
        **INFINITE_RETRY_KWARGS,
    )
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
        json_mode: bool = False,
    ) -> str:
        """Generate text using OpenAI-compatible chat/completions API."""
        if max_tokens <= 0:
            max_tokens = self.max_output_tokens
        async with self.semaphore:
            body: dict[str, Any] = {
                "model": self.model_name,
                "messages": self._build_messages(prompt, system),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}

            resp = await self._client.post("/chat/completions", json=body)

            # Provider-specific status handling (e.g. DeepSeek 402)
            self._handle_special_status(resp)

            if resp.status_code in (400, 401, 403):
                detail = resp.json().get("error", {}).get("message", resp.text)
                raise LLMProviderError(
                    f"{self._provider_name} API error {resp.status_code}: {detail}"
                )
            resp.raise_for_status()
            data = resp.json()

            # Token usage tracking
            usage = data.get("usage", {})
            if usage:
                prompt_tok = usage.get("prompt_tokens", 0)
                compl_tok = usage.get("completion_tokens", 0)
                cache_hit = usage.get("prompt_cache_hit_tokens", 0)
                details = usage.get("completion_tokens_details", {})
                reasoning_tok = (
                    details.get("reasoning_tokens", 0) if details else 0
                )

                self.total_prompt_tokens += prompt_tok
                self.total_completion_tokens += compl_tok
                self.total_cache_hit_tokens += cache_hit
                self.total_reasoning_tokens += reasoning_tok
                self._call_count += 1

                self._last_usage = {
                    "input_tokens": prompt_tok,
                    "output_tokens": self._compute_billed_output(
                        compl_tok, reasoning_tok
                    ),
                    "reasoning_tokens": reasoning_tok,
                }

                logger.info(
                    "%s [call #%d] %d prompt (%d cached) + %d completion"
                    "%s | cumulative: %d in, %d out",
                    self._provider_name,
                    self._call_count,
                    prompt_tok,
                    cache_hit,
                    compl_tok,
                    f" ({reasoning_tok} reasoning)" if reasoning_tok else "",
                    self.total_prompt_tokens,
                    self.total_completion_tokens,
                )

            choices = data.get("choices", [])
            if not choices:
                raise LLMProviderError(f"{self._provider_name} returned no choices")

            self._last_finish_reason = choices[0].get("finish_reason", "unknown")
            return choices[0]["message"]["content"]

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Generate structured JSON output."""
        json_system = (system or "") + "\n\nRespond with valid JSON only."
        text = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return self.extract_json(text, self._provider_name)

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
            except Exception as e:
                logger.error("Batch request %s failed: %s", request_id, e)
                return {"id": request_id, "error": str(e)}

        tasks = [process_request(req) for req in requests]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD (cache miss pricing)."""
        input_cost = (input_tokens / 1_000_000) * self._input_cost_per_1m(
            self.model_name
        )
        output_cost = (output_tokens / 1_000_000) * self._output_cost_per_1m(
            self.model_name
        )
        return input_cost + output_cost

    def get_session_cost(self) -> dict[str, Any]:
        """Return cumulative token usage and cost breakdown for this session."""
        cache_miss = self.total_prompt_tokens - self.total_cache_hit_tokens
        input_cost = (
            (cache_miss / 1_000_000) * self._input_cost_per_1m(self.model_name)
            + (self.total_cache_hit_tokens / 1_000_000)
            * self._cache_hit_cost_per_1m(self.model_name)
        )
        billed_output = self._compute_billed_output(
            self.total_completion_tokens, self.total_reasoning_tokens
        )
        output_cost = (billed_output / 1_000_000) * self._output_cost_per_1m(
            self.model_name
        )
        return {
            "calls": self._call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "cache_hit_tokens": self.total_cache_hit_tokens,
            "cache_miss_tokens": cache_miss,
            "completion_tokens": self.total_completion_tokens,
            "reasoning_tokens": self.total_reasoning_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (accurate) or len//4 fallback."""
        from pyutilz.llm.token_counter import count_tokens
        return count_tokens(text)
