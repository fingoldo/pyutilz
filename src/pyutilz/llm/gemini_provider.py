"""Google Gemini LLM provider using the new google.genai SDK."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tenacity import retry, retry_if_exception_type

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.exceptions import LLMSafetyBlockError
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Import google.genai at module level to check availability
genai: Any
types: Any
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    # Only the genai import failure invalidates ``types``. A later failure
    # to import google.api_core must NOT clobber a valid ``types`` binding.
    types = None

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


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the new google.genai SDK."""

    _provider_name = "Gemini"

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
        "gemini-3.1-pro-preview": (2.00, 12.00),
        "gemini-3.1-flash-lite-preview": (0.25, 1.50),
        "gemini-3-flash-preview": (0.50, 3.00),
        # Tier-2 (>200K): ($2.50, $15.00).
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
    }
    # Cached input prices per 1M tokens (90% discount on input miss).
    # Plus storage at $1-4.50/hour depending on model.
    _CACHE_HIT_COST: dict[str, float] = {
        "gemini-3.1-pro-preview": 0.20,
        "gemini-3.1-flash-lite-preview": 0.025,
        "gemini-3-flash-preview": 0.05,
        "gemini-2.5-pro": 0.125,
        "gemini-2.5-flash": 0.03,
        "gemini-2.5-flash-lite": 0.01,
    }
    _DEFAULT_PRICING = (0.25, 1.50)

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.1-flash-lite-preview",
        max_concurrent: int = 10,
        candidate_count: int = 1,
        cached_content: str | None = None,
    ):
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")

        settings = get_llm_settings()
        self.api_key = api_key or (settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None)
        if not self.api_key:
            raise ValueError("Gemini API key not provided")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        # Per-call response metadata captured by ``_capture_candidate_metadata``.
        # Without these, grounding / citations / function calls / safety
        # categories silently disappear when only ``.text`` is extracted.
        self.last_safety_ratings: list[dict[str, Any]] = []
        self.last_grounding_metadata: Any = None
        self.last_citation_metadata: Any = None
        self.last_function_calls: list[dict[str, Any]] = []
        self.last_all_candidates: list[Any] = []
        self.last_cached_content_tokens = 0
        self.total_cached_content_tokens = 0
        # Phase-4 multi-candidate + cache support.
        # ``candidate_count``: how many response candidates to ask for in
        # one call (Gemini supports up to ~8). The first is returned by
        # ``generate()``; ``last_all_candidates`` holds the rest.
        # ``cached_content``: pass a Cached Content resource name (from
        # the Caching API) to skip re-paying the prompt-input rate on
        # cached prefixes; usage_metadata.cached_content_token_count
        # surfaces the saving in ``last_cached_content_tokens``.
        self._candidate_count = max(1, int(candidate_count))
        self._cached_content = cached_content

    @property
    def context_window(self) -> int:
        for prefix, limit in [("gemini-2.5", 1_048_576), ("gemini-2.0", 1_048_576), ("gemini-3", 1_048_576)]:
            if self.model_name.startswith(prefix):
                return limit
        return 1_048_576

    @property
    def max_output_tokens(self) -> int:
        for prefix, limit in [("gemini-3", 65536), ("gemini-2.5", 65536), ("gemini-2.0", 8192)]:
            if self.model_name.startswith(prefix):
                return limit
        return 8192

    def supports_json_mode(self) -> bool:
        """Gemini 2.0+ accepts ``response_mime_type="application/json"``
        (mapped from our ``json_mode=True`` kwarg in ``generate``).
        Earlier 1.x models did not — but pyutilz no longer ships
        defaults pointing there, so True is safe across the supported
        catalogue."""
        return True

    @retry(  # type: ignore[call-overload]  # tenacity's retry() overloads can't be resolved through a **dict unpack; correct at runtime
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

            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "system_instruction": system if system else None,
                "response_mime_type": "application/json" if json_mode else None,
            }
            # Phase-4 multi-candidate + cached-content support. Skip
            # ``candidate_count`` when 1 (the SDK default) so the request
            # body stays unchanged for the common path.
            if self._candidate_count > 1:
                config_kwargs["candidate_count"] = self._candidate_count
            if self._cached_content:
                config_kwargs["cached_content"] = self._cached_content
            config = types.GenerateContentConfig(**config_kwargs)

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
                self._last_finish_reason = str(getattr(response.candidates[0], "finish_reason", "unknown"))
            else:
                self._last_finish_reason = "unknown"

            # Capture safety ratings, grounding, citations, function calls
            # from candidate[0]. Saves callers from re-parsing the raw
            # response object themselves.
            self._capture_candidate_metadata(response)

            um = getattr(response, "usage_metadata", None)
            if um:
                self._last_usage = {
                    "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                    "reasoning_tokens": getattr(um, "thoughts_token_count", 0) or 0,
                    "cached_content_token_count": (getattr(um, "cached_content_token_count", 0) or 0),
                }
                self.last_cached_content_tokens = self._last_usage["cached_content_token_count"]
                self.total_cached_content_tokens += self.last_cached_content_tokens

            # Safety-filter detection: Gemini returns finish_reason=SAFETY
            # when the response is blocked. response.text may be empty or
            # raise when accessed; handle both. We surface the per-category
            # safety_ratings (HARASSMENT / HATE_SPEECH / SEXUALLY_EXPLICIT
            # / DANGEROUS_CONTENT) so callers know WHICH category fired.
            _fr = self._last_finish_reason.upper() if isinstance(self._last_finish_reason, str) else ""
            if "SAFETY" in _fr or "BLOCK" in _fr:
                raise LLMSafetyBlockError(
                    f"Gemini blocked response by safety filter (finish_reason={self._last_finish_reason})",
                    raw_text=None,
                    details={
                        "finish_reason": self._last_finish_reason,
                        "safety_ratings": self.last_safety_ratings,
                    },
                )
            try:
                text_out = response.text
            except (ValueError, AttributeError) as exc:
                # Accessing .text on a blocked/empty candidate raises
                raise LLMSafetyBlockError(
                    f"Gemini response has no text (likely safety block): {exc}",
                    details={
                        "finish_reason": self._last_finish_reason,
                        "safety_ratings": self.last_safety_ratings,
                    },
                )
            if not text_out:
                raise LLMSafetyBlockError(
                    "Gemini returned empty text (likely safety block)",
                    details={
                        "finish_reason": self._last_finish_reason,
                        "safety_ratings": self.last_safety_ratings,
                    },
                )
            return text_out  # type: ignore[no-any-return]  # untyped upstream source (google.genai response text); return value verified correct at runtime

    def _capture_candidate_metadata(self, response: Any) -> None:
        """Stash safety ratings / grounding / citation / function-call info
        from ``response.candidates[0]`` onto ``last_*`` attributes.

        Without this, Google Search grounding sources, citation segments,
        and function_calls all silently disappear -- pyutilz's previous
        wrapper extracted only ``.text``. Best-effort: any failure leaves
        the previous snapshot in place.
        """
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                self.last_safety_ratings = []
                self.last_grounding_metadata = None
                self.last_citation_metadata = None
                self.last_function_calls = []
                return
            cand = candidates[0]
            ratings = getattr(cand, "safety_ratings", None) or []
            self.last_safety_ratings = [
                {
                    "category": str(getattr(r, "category", "")),
                    "probability": str(getattr(r, "probability", "")),
                    "blocked": bool(getattr(r, "blocked", False)),
                }
                for r in ratings
            ]
            grounding = getattr(cand, "grounding_metadata", None)
            self.last_grounding_metadata = grounding
            citation = getattr(cand, "citation_metadata", None)
            self.last_citation_metadata = citation

            calls: list[dict[str, Any]] = []
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    calls.append({
                        "name": getattr(fc, "name", ""),
                        "args": dict(getattr(fc, "args", {}) or {}),
                    })
            self.last_function_calls = calls
            # Preserve raw additional candidates for callers that requested
            # multi-candidate generation.
            self.last_all_candidates = list(candidates)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Gemini candidate metadata capture failed: %s", exc)

    def _classify_batch_exception(self, exc: Exception) -> dict[str, Any] | None:
        # Safety blocks are the single most common Gemini failure mode.
        # Tag them so batch callers can branch on ``error_type`` without
        # re-parsing the error string.
        if isinstance(exc, LLMSafetyBlockError):
            return {"error_type": "safety_block"}
        return None

    async def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's tokenizer.

        Makes a network call to the Gemini API. Falls back to tiktoken on any
        transient API failure so a temporary outage doesn't block calling
        code (cl100k_base diverges from Gemini's tokenizer but is a usable
        estimate for budgeting).
        """
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.client.models.count_tokens(
                    model=self.model_name,
                    contents=text,
                ),
            )
            if result.total_tokens is None:
                raise ValueError("Gemini count_tokens returned no total_tokens")
            return int(result.total_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Gemini count_tokens API failed (%s); falling back to tiktoken.", exc)
            from pyutilz.llm.token_counter import count_tokens
            return count_tokens(text)

    async def get_account_credits(self) -> dict:
        # Gemini billing rides on Google Cloud — credit / spend lives in the
        # Cloud Billing API, which uses GCP service-account auth, not the
        # Gemini API key we're holding. Mixing the two would expand scope
        # significantly and isn't symmetric with what other providers offer.
        raise NotImplementedError(
            "Gemini billing is GCP-based — there is no balance endpoint "
            "reachable with a Gemini API key. Check console.cloud.google.com/billing "
            "or use the Cloud Billing API with separate GCP credentials."
        )

    async def check_account_limits(self) -> dict:
        raise NotImplementedError("Gemini does not expose per-key rate limits via API. " "Quotas are GCP-side at console.cloud.google.com/iam-admin/quotas.")

    async def _close(self) -> None:
        """Release the per-provider ThreadPoolExecutor.

        The factory's atexit handler (factory.py ``_close_cached_providers``)
        awaits ``_close`` on every cached provider that exposes it. Without
        this hook the executor's worker threads accumulate when a long-lived
        process builds many distinct Gemini providers (the stdlib executor's
        own atexit hook only joins threads at interpreter exit).
        ``wait=False`` so shutdown never blocks on an in-flight call.
        """
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
