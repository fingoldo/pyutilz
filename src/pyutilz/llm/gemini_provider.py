"""Google Gemini LLM provider using the new google.genai SDK."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

from tenacity import retry, retry_if_exception_type

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.exceptions import (
    LLMProviderError, JSONParsingError, LLMSafetyBlockError,
)
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Import google.genai at module level to check availability
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

# Google API exceptions for rate limit / server errors (retry-worthy).
try:
    from google.api_core.exceptions import (
        ResourceExhausted,
        ServiceUnavailable,
        InternalServerError as GoogleInternalServerError,
        TooManyRequests,
    )
    _GENAI_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
        ResourceExhausted, ServiceUnavailable, GoogleInternalServerError, TooManyRequests,
    )
except ImportError:
    _GENAI_RETRYABLE_EXCEPTIONS = ()
    types = None


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the new google.genai SDK."""

    # Pricing per 1M tokens: (input, output)
    # Source: https://ai.google.dev/gemini-api/docs/pricing
    # Verified 2026-05-01.
    #
    # NOTE: Gemini Pro models (2.5-pro and 3.1-pro-preview) have TIERED
    # pricing — prompts ≤200K tokens billed at the lower tier, >200K at
    # the higher tier. We bill at the lower tier here; callers issuing
    # >200K prompts should override via ``estimate_cost`` with explicit
    # rates. Tier-2 prices documented in the comments next to each entry.
    _PRICING: dict[str, tuple[float, float]] = {
        # Tier-2 (>200K): ($4.00, $18.00) — 2x input, 1.5x output.
        "gemini-3.1-pro-preview":        (2.00, 12.00),
        "gemini-3.1-flash-lite-preview": (0.25, 1.50),
        "gemini-3-flash-preview":        (0.50, 3.00),
        # Tier-2 (>200K): ($2.50, $15.00).
        "gemini-2.5-pro":               (1.25, 10.00),
        "gemini-2.5-flash":             (0.30, 2.50),
        "gemini-2.5-flash-lite":        (0.10, 0.40),
    }
    # Cached input prices per 1M tokens (90% discount on input miss).
    # Plus storage at $1-4.50/hour depending on model.
    _CACHE_HIT_COST: dict[str, float] = {
        "gemini-3.1-pro-preview":        0.20,
        "gemini-3.1-flash-lite-preview": 0.025,
        "gemini-3-flash-preview":        0.05,
        "gemini-2.5-pro":                0.125,
        "gemini-2.5-flash":              0.03,
        "gemini-2.5-flash-lite":         0.01,
    }
    _DEFAULT_PRICING = (0.25, 1.50)

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.1-flash-lite-preview",
        max_concurrent: int = 10,
    ):
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")

        settings = get_llm_settings()
        self.api_key = api_key or (
            settings.gemini_api_key.get_secret_value()
            if settings.gemini_api_key
            else None
        )
        if not self.api_key:
            raise ValueError("Gemini API key not provided")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    @property
    def context_window(self) -> int:
        for prefix, limit in [("gemini-2.5", 1_048_576), ("gemini-2.0", 1_048_576), ("gemini-3", 1_048_576)]:
            if self.model_name.startswith(prefix):
                return limit
        return 1_048_576

    @property
    def max_output_tokens(self) -> int:
        for prefix, limit in [("gemini-3.", 65536), ("gemini-2.5", 65536), ("gemini-2.0", 8192)]:
            if self.model_name.startswith(prefix):
                return limit
        return 8192

    @retry(
        retry=retry_if_exception_type((
            ConnectionError, TimeoutError, OSError,
            *_GENAI_RETRYABLE_EXCEPTIONS,
        )),
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
        """Generate text using Gemini."""
        if max_tokens <= 0:
            max_tokens = self.max_output_tokens
        async with self.semaphore:
            contents = prompt

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system if system else None,
                response_mime_type="application/json" if json_mode else None,
            )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                ),
            )

            if response.candidates:
                self._last_finish_reason = str(
                    getattr(response.candidates[0], "finish_reason", "unknown")
                )
            else:
                self._last_finish_reason = "unknown"

            um = getattr(response, "usage_metadata", None)
            if um:
                self._last_usage = {
                    "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                    "reasoning_tokens": getattr(um, "thoughts_token_count", 0) or 0,
                }

            # Safety-filter detection: Gemini returns finish_reason=SAFETY
            # when the response is blocked. response.text may be empty or
            # raise when accessed; handle both.
            _fr = self._last_finish_reason.upper() if isinstance(self._last_finish_reason, str) else ""
            if "SAFETY" in _fr or "BLOCK" in _fr:
                raise LLMSafetyBlockError(
                    f"Gemini blocked response by safety filter (finish_reason={self._last_finish_reason})",
                    raw_text=None,
                    details={"finish_reason": self._last_finish_reason},
                )
            try:
                text_out = response.text
            except (ValueError, AttributeError) as exc:
                # Accessing .text on a blocked/empty candidate raises
                raise LLMSafetyBlockError(
                    f"Gemini response has no text (likely safety block): {exc}",
                    details={"finish_reason": self._last_finish_reason},
                )
            if not text_out:
                raise LLMSafetyBlockError(
                    "Gemini returned empty text (likely safety block)",
                    details={"finish_reason": self._last_finish_reason},
                )
            return text_out

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
        return self.extract_json(text, "Gemini")

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
            except ValueError as e:
                logger.error(f"Batch request {request_id} value error: {e}")
                return {"id": request_id, "error": f"Value error: {e}"}
            except RuntimeError as e:
                logger.error(f"Batch request {request_id} runtime error: {e}")
                return {"id": request_id, "error": f"Runtime error: {e}"}

        tasks = [process_request(req) for req in requests]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    def _get_pricing(self) -> tuple[float, float]:
        """Return (input_cost_per_1m, output_cost_per_1m) for current model."""
        pricing = self._PRICING.get(self.model_name)
        if pricing:
            return pricing
        for key, val in self._PRICING.items():
            if self.model_name.startswith(key):
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
        """Count tokens using Gemini's tokenizer."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self.client.models.count_tokens(
                model=self.model_name,
                contents=text,
            ),
        )
        return result.total_tokens
