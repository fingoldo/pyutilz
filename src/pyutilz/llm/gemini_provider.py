"""Google Gemini LLM provider using the new google.genai SDK."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tenacity import retry, retry_if_exception, retry_if_exception_type

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.exceptions import LLMSafetyBlockError, LLMTruncationError
from pyutilz.llm._retry import INFINITE_RETRY_KWARGS
from pyutilz.llm.base import LLMProvider, PerCallAttr

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

# Regression fix (2026-07-21 audit): the installed google-genai SDK (the only Gemini client this
# module imports -- `from google import genai` above) raises its OWN google.genai.errors
# ClientError/ServerError (both inherit directly from Exception), NOT google.api_core.exceptions
# -- that's a separate library used by older, gRPC-based Google client SDKs, and google-genai
# doesn't even depend on it. The previous predicate matched nothing a real google-genai call ever
# raises, so a rate-limited (429) or briefly-down (5xx) Gemini call was never actually retried.
try:
    from google.genai.errors import ClientError as _GenaiClientError, ServerError as _GenaiServerError

    def _is_retryable_genai_error(exc: BaseException) -> bool:
        """5xx (ServerError) always retryable; ClientError only for 429 (rate limit) -- other 4xx
        (400 bad request, 401/403 auth) are permanent and must not be retried."""
        if isinstance(exc, _GenaiServerError):
            return True
        if isinstance(exc, _GenaiClientError):
            return getattr(exc, "code", None) == 429
        return False
except ImportError:

    def _is_retryable_genai_error(exc: BaseException) -> bool:
        """google.genai.errors unavailable -- nothing to classify as retryable."""
        return False


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
    _PRICING: dict[str, tuple[float, float]] = {  # noqa: RUF012 -- intentional shared class-level pricing table, not a per-instance mutable-default bug
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
    _CACHE_HIT_COST: dict[str, float] = {  # noqa: RUF012 -- intentional shared class-level pricing table, not a per-instance mutable-default bug
        "gemini-3.1-pro-preview": 0.20,
        "gemini-3.1-flash-lite-preview": 0.025,
        "gemini-3-flash-preview": 0.05,
        "gemini-2.5-pro": 0.125,
        "gemini-2.5-flash": 0.03,
        "gemini-2.5-flash-lite": 0.01,
    }
    _DEFAULT_PRICING = (0.25, 1.50)

    # Per-call "last successful call" state -- backed by contextvars via PerCallAttr, NOT plain
    # instance attributes. Regression fix (2026-07-21 audit round 2, HIGH): see identical
    # PerCallAttr usage + docstring in openai_compat.py / base.py -- generate_batch() fires N
    # concurrent self.generate() calls on one shared/cached provider instance, so a plain
    # attribute write from one in-flight request used to be visible to every other
    # concurrently-running request. ``total_cached_content_tokens`` is NOT converted -- it is
    # intentionally cumulative/summed across all calls.
    _last_usage: PerCallAttr = PerCallAttr(lambda: {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0})
    _last_finish_reason: PerCallAttr = PerCallAttr(lambda: None)
    last_safety_ratings: PerCallAttr = PerCallAttr(list)
    last_grounding_metadata: PerCallAttr = PerCallAttr(lambda: None)
    last_citation_metadata: PerCallAttr = PerCallAttr(lambda: None)
    last_function_calls: PerCallAttr = PerCallAttr(list)
    last_all_candidates: PerCallAttr = PerCallAttr(list)
    last_cached_content_tokens: PerCallAttr = PerCallAttr(lambda: 0)

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
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY in .env or pass api_key=")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        # Per-call usage/safety/grounding/citation/function-call/candidate metadata:
        # PerCallAttr class-level descriptors (declared above __init__) provide the defaults;
        # nothing to initialize here.
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
        """Maximum input context size in tokens for the configured model."""
        for prefix, limit in [("gemini-2.5", 1_048_576), ("gemini-2.0", 1_048_576), ("gemini-3", 1_048_576)]:
            if self.model_name.startswith(prefix):
                return limit
        return 1_048_576

    @property
    def max_output_tokens(self) -> int:
        """Maximum number of output tokens supported by the configured model."""
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
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)) | retry_if_exception(_is_retryable_genai_error),
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

            # Native async client (google-genai's own .aio surface) instead of offloading the
            # sync client to a dedicated ThreadPoolExecutor: avoids consuming a thread-pool
            # worker per concurrent call and integrates more naturally with asyncio cancellation.
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
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
            if "MAX_TOKENS" in _fr:
                # Regression fix (2026-07-21 audit): LLMTruncationError was fully specified but
                # never actually raised anywhere -- see openai_compat.py's identical fix.
                raise LLMTruncationError(
                    f"Gemini response truncated by max_tokens (finish_reason={self._last_finish_reason})", finish_reason=self._last_finish_reason
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
        except Exception as exc:
            logger.debug("Gemini candidate metadata capture failed: %s", exc)

    def _classify_batch_exception(self, exc: Exception) -> dict[str, Any] | None:
        """Tag safety-block exceptions with an ``error_type`` so batch callers can branch without parsing error strings; returns None for other exception types."""
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
            result = await self.client.aio.models.count_tokens(
                model=self.model_name,
                contents=text,
            )
            if result.total_tokens is None:
                raise ValueError("Gemini count_tokens returned no total_tokens")
            return int(result.total_tokens)
        except Exception as exc:
            logger.debug("Gemini count_tokens API failed (%s); falling back to tiktoken.", exc)
            from pyutilz.llm.token_counter import count_tokens
            return count_tokens(text)

    async def get_account_credits(self) -> dict:
        """Not supported for Gemini: always raises NotImplementedError since account balance lives in GCP Cloud Billing, not the Gemini API."""
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
        """Not supported for Gemini: always raises NotImplementedError since per-key rate limits are managed via GCP quotas, not exposed by the API."""
        raise NotImplementedError("Gemini does not expose per-key rate limits via API. " "Quotas are GCP-side at console.cloud.google.com/iam-admin/quotas.")

    async def _close(self) -> None:
        """No-op: ``generate()``/``count_tokens()`` now use google-genai's native async client
        (``self.client.aio``) rather than a dedicated ThreadPoolExecutor, so there are no
        per-provider worker threads left to release. Kept as a hook for factory.py's
        ``_close_cached_providers``, which awaits ``_close`` on every cached provider that
        exposes it."""
        return None
