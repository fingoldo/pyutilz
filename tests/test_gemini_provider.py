"""Behavioral coverage for pyutilz.llm.gemini_provider (previously untested -- no test file existed
for this module at all, unlike every sibling provider in llm/)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pyutilz.llm.gemini_provider as gemini_provider_module
from pyutilz.llm.exceptions import LLMSafetyBlockError, LLMTruncationError
from pyutilz.llm.gemini_provider import GeminiProvider, _is_retryable_genai_error


@pytest.fixture(autouse=True)
def _ensure_genai_types_available():
    """``types`` (``from google.genai import types``) is a module-level global set to None when
    GENAI_AVAILABLE is False -- e.g. on any CI leg without the real google-genai package
    installed. ``generate()`` calls ``types.GenerateContentConfig(...)`` unconditionally, so
    every test exercising it needs a real (or mocked) ``types`` regardless of whether the actual
    optional dependency is present. Individual tests that patch ``types`` themselves for a
    stricter assertion (e.g. checking constructor kwargs) simply nest their own patch inside
    this one -- harmless, not a conflict."""
    # GENAI_AVAILABLE is forced True for every test in this module: the constructor refuses to run without it, and
    # google-genai is not installed on Python 3.8 (its marker is >=3.10), so classes that build a provider directly
    # (not through _make_provider) would otherwise fail there. A test that needs the unavailable path patches it False itself.
    with patch("pyutilz.llm.gemini_provider.GENAI_AVAILABLE", True):
        if gemini_provider_module.types is not None:
            yield
        else:
            with patch("pyutilz.llm.gemini_provider.types", MagicMock()):
                yield


def _make_provider(**overrides):
    # GENAI_AVAILABLE is a module-level flag set once at import time from whether the real
    # google-genai package was importable -- False on any CI leg where it's not installed
    # (e.g. an unsupported Python version). Patched True here so these tests exercise
    # GeminiProvider's own logic via a mocked `genai` client regardless of whether the real
    # optional dependency is present in this environment.
    mock_settings = MagicMock()
    mock_settings.gemini_api_key = None
    with patch("pyutilz.llm.gemini_provider.get_llm_settings", return_value=mock_settings), patch("pyutilz.llm.gemini_provider.GENAI_AVAILABLE", True), patch(
        "pyutilz.llm.gemini_provider.genai"
    ) as mock_genai:
        mock_genai.Client.return_value = MagicMock()
        p = GeminiProvider(api_key="test-key", **overrides)  # pragma: allowlist secret -- test placeholder, not a real credential
    return p


class TestInitAndConfig:
    def test_missing_api_key_raises(self):
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        with patch("pyutilz.llm.gemini_provider.get_llm_settings", return_value=mock_settings), patch("pyutilz.llm.gemini_provider.GENAI_AVAILABLE", True):
            with pytest.raises(ValueError, match="API key not provided"):
                GeminiProvider(api_key=None)

    def test_raises_when_genai_unavailable(self):
        with patch("pyutilz.llm.gemini_provider.GENAI_AVAILABLE", False):
            with pytest.raises(ImportError, match="google-genai package not installed"):
                GeminiProvider(api_key="test-key")  # pragma: allowlist secret -- test placeholder, not a real credential

    def test_default_model_and_candidate_count(self):
        p = _make_provider()
        assert p.model_name == "gemini-3.1-flash-lite-preview"
        assert p._candidate_count == 1

    def test_candidate_count_clamped_to_at_least_one(self):
        p = _make_provider(candidate_count=0)
        assert p._candidate_count == 1

    def test_supports_json_mode_true(self):
        assert _make_provider().supports_json_mode() is True


class TestContextAndOutputLimits:
    def test_context_window_gemini_2_5(self):
        p = _make_provider(model="gemini-2.5-flash")
        assert p.context_window == 1_048_576

    def test_max_output_tokens_gemini_2_0(self):
        p = _make_provider(model="gemini-2.0-flash")
        assert p.max_output_tokens == 8192

    def test_max_output_tokens_gemini_3(self):
        p = _make_provider(model="gemini-3-flash-preview")
        assert p.max_output_tokens == 65536

    def test_max_output_tokens_unknown_model_falls_back(self):
        p = _make_provider(model="some-future-model")
        assert p.max_output_tokens is None, "an unlisted model has no known ceiling: max_tokens is omitted"


class TestIsRetryableGenaiError:
    def test_connection_style_errors_not_classified_here(self):
        assert _is_retryable_genai_error(ConnectionError("x")) is False

    def test_server_error_always_retryable(self):
        errors = pytest.importorskip("google.genai.errors")

        err = errors.ServerError.__new__(errors.ServerError)
        assert _is_retryable_genai_error(err) is True

    def test_client_error_429_retryable(self):
        errors = pytest.importorskip("google.genai.errors")

        err = errors.ClientError.__new__(errors.ClientError)
        err.code = 429
        assert _is_retryable_genai_error(err) is True

    def test_client_error_400_not_retryable(self):
        errors = pytest.importorskip("google.genai.errors")

        err = errors.ClientError.__new__(errors.ClientError)
        err.code = 400
        assert _is_retryable_genai_error(err) is False


class TestGenerate:
    def _response(self, text="hello", finish_reason="STOP", usage=None, candidates=True):
        cand = SimpleNamespace(
            finish_reason=finish_reason,
            safety_ratings=[],
            grounding_metadata=None,
            citation_metadata=None,
            content=SimpleNamespace(parts=[]),
        )
        return SimpleNamespace(
            candidates=[cand] if candidates else [],
            usage_metadata=usage,
            text=text,
        )

    @pytest.mark.asyncio
    async def test_successful_generate_returns_text(self):
        p = _make_provider()
        p.client = MagicMock()
        p.client.aio.models.generate_content = AsyncMock(return_value=self._response(text="hi there"))
        out = await p.generate("prompt")
        assert out == "hi there"

    @pytest.mark.asyncio
    async def test_usage_metadata_captured(self):
        p = _make_provider()
        p.client = MagicMock()
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=2, cached_content_token_count=3)
        p.client.aio.models.generate_content = AsyncMock(return_value=self._response(usage=usage))
        await p.generate("prompt")
        assert p._last_usage["input_tokens"] == 10
        assert p._last_usage["output_tokens"] == 5
        assert p._last_usage["reasoning_tokens"] == 2
        assert p.last_cached_content_tokens == 3
        assert p.total_cached_content_tokens == 3

    @pytest.mark.asyncio
    async def test_safety_block_finish_reason_raises(self):
        p = _make_provider()
        p.client = MagicMock()
        p.client.aio.models.generate_content = AsyncMock(return_value=self._response(finish_reason="SAFETY"))
        with pytest.raises(LLMSafetyBlockError, match="blocked"):
            await p.generate("prompt")

    @pytest.mark.asyncio
    async def test_no_candidates_sets_unknown_finish_reason_then_empty_text_blocks(self):
        p = _make_provider()
        p.client = MagicMock()
        p.client.aio.models.generate_content = AsyncMock(return_value=self._response(text="", candidates=False))
        with pytest.raises(LLMSafetyBlockError):
            await p.generate("prompt")
        assert p._last_finish_reason == "unknown"

    @pytest.mark.asyncio
    async def test_accessing_text_raises_becomes_safety_block(self):
        p = _make_provider()
        p.client = MagicMock()
        resp = MagicMock()
        resp.candidates = [SimpleNamespace(finish_reason="STOP", safety_ratings=[], grounding_metadata=None, citation_metadata=None, content=SimpleNamespace(parts=[]))]
        resp.usage_metadata = None
        type(resp).text = property(lambda self: (_ for _ in ()).throw(ValueError("blocked")))
        p.client.aio.models.generate_content = AsyncMock(return_value=resp)
        with pytest.raises(LLMSafetyBlockError, match="no text"):
            await p.generate("prompt")

    @pytest.mark.asyncio
    async def test_empty_text_raises_safety_block(self):
        p = _make_provider()
        p.client = MagicMock()
        p.client.aio.models.generate_content = AsyncMock(return_value=self._response(text=""))
        with pytest.raises(LLMSafetyBlockError, match="empty text"):
            await p.generate("prompt")

    @pytest.mark.asyncio
    async def test_max_tokens_finish_reason_raises_truncation(self):
        p = _make_provider()
        p.client = MagicMock()
        p.client.aio.models.generate_content = AsyncMock(return_value=self._response(finish_reason="MAX_TOKENS"))
        with pytest.raises(LLMTruncationError):
            await p.generate("prompt")

    @pytest.mark.asyncio
    async def test_candidate_count_over_one_added_to_config(self):
        p = _make_provider(candidate_count=3)
        p.client = MagicMock()
        p.client.aio.models.generate_content = AsyncMock(return_value=self._response())
        with patch("pyutilz.llm.gemini_provider.types") as mock_types:
            mock_types.GenerateContentConfig = MagicMock()
            await p.generate("prompt")
            _, kwargs = mock_types.GenerateContentConfig.call_args
            assert kwargs["candidate_count"] == 3

    @pytest.mark.asyncio
    async def test_cached_content_added_to_config(self):
        p = _make_provider(cached_content="cachedContents/abc")
        p.client = MagicMock()
        p.client.aio.models.generate_content = AsyncMock(return_value=self._response())
        with patch("pyutilz.llm.gemini_provider.types") as mock_types:
            mock_types.GenerateContentConfig = MagicMock()
            await p.generate("prompt")
            _, kwargs = mock_types.GenerateContentConfig.call_args
            assert kwargs["cached_content"] == "cachedContents/abc"


class TestCaptureCandidateMetadata:
    def test_extracts_safety_ratings_and_function_calls(self):
        p = _make_provider()
        rating = SimpleNamespace(category="HARASSMENT", probability="LOW", blocked=False)
        fc = SimpleNamespace(name="lookup", args={"q": "x"})
        part = SimpleNamespace(function_call=fc)
        cand = SimpleNamespace(
            safety_ratings=[rating],
            grounding_metadata="grounding",
            citation_metadata="citation",
            content=SimpleNamespace(parts=[part]),
        )
        response = SimpleNamespace(candidates=[cand])
        p._capture_candidate_metadata(response)
        assert p.last_safety_ratings == [{"category": "HARASSMENT", "probability": "LOW", "blocked": False}]
        assert p.last_grounding_metadata == "grounding"
        assert p.last_citation_metadata == "citation"
        assert p.last_function_calls == [{"name": "lookup", "args": {"q": "x"}}]

    def test_no_candidates_resets_metadata(self):
        p = _make_provider()
        p.last_safety_ratings = ["stale"]
        response = SimpleNamespace(candidates=[])
        p._capture_candidate_metadata(response)
        assert p.last_safety_ratings == []
        assert p.last_grounding_metadata is None

    def test_capture_failure_is_swallowed_and_preserves_previous_snapshot(self):
        p = _make_provider()
        p.last_safety_ratings = ["kept"]
        broken_response = MagicMock()
        broken_response.candidates = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        p._capture_candidate_metadata(broken_response)
        assert p.last_safety_ratings == ["kept"]


class TestClassifyBatchException:
    def test_safety_block_tagged(self):
        p = _make_provider()
        out = p._classify_batch_exception(LLMSafetyBlockError("blocked"))
        assert out == {"error_type": "safety_block"}

    def test_other_exception_returns_none(self):
        p = _make_provider()
        assert p._classify_batch_exception(ValueError("x")) is None


class TestCountTokens:
    @pytest.mark.asyncio
    async def test_uses_gemini_api(self):
        p = _make_provider()
        p.client = MagicMock()
        p.client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=42))
        assert await p.count_tokens("hello") == 42

    @pytest.mark.asyncio
    async def test_falls_back_to_tiktoken_on_none_total(self):
        p = _make_provider()
        p.client = MagicMock()
        p.client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=None))
        out = await p.count_tokens("hello world")
        assert isinstance(out, int)
        assert out > 0

    @pytest.mark.asyncio
    async def test_falls_back_to_tiktoken_on_api_failure(self):
        p = _make_provider()
        p.client = MagicMock()
        p.client.aio.models.count_tokens = AsyncMock(side_effect=RuntimeError("down"))
        out = await p.count_tokens("hello")
        assert isinstance(out, int)
        assert out > 0


class TestUnsupportedEndpoints:
    @pytest.mark.asyncio
    async def test_get_account_credits_raises(self):
        p = _make_provider()
        with pytest.raises(NotImplementedError, match="GCP"):
            await p.get_account_credits()

    @pytest.mark.asyncio
    async def test_check_account_limits_raises(self):
        p = _make_provider()
        with pytest.raises(NotImplementedError, match="GCP"):
            await p.check_account_limits()

    @pytest.mark.asyncio
    async def test_close_is_a_noop(self):
        p = _make_provider()
        assert await p._close() is None


def test_missing_api_key_error_names_pass_api_key_fallback():
    mock_settings = MagicMock()
    mock_settings.gemini_api_key = None
    with patch("pyutilz.llm.gemini_provider.get_llm_settings", return_value=mock_settings), patch("pyutilz.llm.gemini_provider.GENAI_AVAILABLE", True):
        with pytest.raises(ValueError, match=r"pass api_key="):
            GeminiProvider(api_key=None)


_PRICING_PIN = {'gemini-2.5-flash': (0.3, 2.5),
 'gemini-2.5-flash-lite': (0.1, 0.4),
 'gemini-2.5-pro': (1.25, 10.0),
 'gemini-3-flash-preview': (0.5, 3.0),
 'gemini-3.1-flash-lite': (0.25, 1.5),
 'gemini-3.1-flash-lite-preview': (0.25, 1.5),
 'gemini-3.1-pro-preview': (2.0, 12.0),
 'gemini-3.5-flash': (1.5, 9.0),
 'gemini-3.5-flash-lite': (0.3, 2.5),
 'gemini-3.6-flash': (0.75, 3.75),
 'gemini-3.7-flash': (0.75, 3.75),
 'gemini-3.8-flash': (0.75, 3.75)}
_CACHE_HIT_PIN = {'gemini-2.5-flash': 0.03,
 'gemini-2.5-flash-lite': 0.01,
 'gemini-2.5-pro': 0.125,
 'gemini-3-flash-preview': 0.05,
 'gemini-3.1-flash-lite': 0.025,
 'gemini-3.1-flash-lite-preview': 0.025,
 'gemini-3.1-pro-preview': 0.2,
 'gemini-3.5-flash': 0.15,
 'gemini-3.5-flash-lite': 0.03,
 'gemini-3.6-flash': 0.075,
 'gemini-3.7-flash': 0.075,
 'gemini-3.8-flash': 0.075}
_TIERS_PIN = {'gemini-2.5-pro': (200000, 2.0, 1.5, 2.0, 2.0, False),
 'gemini-3.1-pro-preview': (200000, 2.0, 1.5, 2.0, 2.0, False)}


class TestGeminiPriceTablesArePinned:
    """Every Gemini price, cache rate and long-context tier is pinned as a whole table, so a changed row, a misplaced
    number or an emptied model name fails here, not only where a test names one model (gemini_provider 93-232)."""

    def test_the_pricing_table_and_its_default_are_exact(self):
        from pyutilz.llm import gemini_provider as gp

        assert gp.GeminiProvider._PRICING == _PRICING_PIN
        assert tuple(gp.GeminiProvider._DEFAULT_PRICING) == (0.25, 1.5)

    def test_the_cache_hit_table_is_exact(self):
        from pyutilz.llm import gemini_provider as gp

        assert gp.GeminiProvider._CACHE_HIT_COST == _CACHE_HIT_PIN

    def test_the_long_context_tiers_and_threshold_are_exact(self):
        from pyutilz.llm import gemini_provider as gp

        tiers = {k: (v.threshold_tokens, v.input_mult, v.output_mult, v.cached_input_mult, v.cache_write_mult, v.inclusive)
                 for k, v in gp.GeminiProvider._LONG_CONTEXT_TIERS.items()}
        assert tiers == _TIERS_PIN
        assert gp.GeminiProvider._LONG_CONTEXT_THRESHOLD == 200_000


class TestGeminiRetryPredicateAndSafetyReasons:
    """The zero-quota pattern is a whole word, the retry predicate retries server errors and an ordinary 429 only, and the
    policy-withheld finish reasons are exact (gemini_provider 35-77). Uses the real google-genai error classes."""

    def test_genai_is_available_when_the_package_is_installed(self):
        pytest.importorskip("google.genai")
        from pyutilz.llm import gemini_provider as gp

        assert gp.GENAI_AVAILABLE is True

    def test_the_zero_quota_pattern_matches_a_whole_limit_zero_only(self):
        from pyutilz.llm import gemini_provider as gp

        assert gp._ZERO_QUOTA.search("quota limit: 0") is not None
        assert gp._ZERO_QUOTA.search("xlimit: 0") is None
        assert gp._ZERO_QUOTA.search("limit: 05") is None
        assert gp._ZERO_QUOTA.search("limit: 0.5") is None

    def test_a_server_error_is_retryable_and_a_zero_quota_429_is_not(self):
        pytest.importorskip("google.genai")
        from google.genai.errors import ClientError, ServerError
        from pyutilz.llm import gemini_provider as gp

        assert gp._is_retryable_genai_error(ServerError(503, {"error": {"message": "down"}})) is True
        assert gp._is_retryable_genai_error(ClientError(429, {"error": {"message": "quota exceeded, limit: 0"}})) is False

    def test_an_ordinary_429_is_retryable_and_other_client_errors_are_not(self):
        pytest.importorskip("google.genai")
        from google.genai.errors import ClientError
        from pyutilz.llm import gemini_provider as gp

        assert gp._is_retryable_genai_error(ClientError(429, {"error": {"message": "slow down"}})) is True
        assert gp._is_retryable_genai_error(ClientError(400, {"error": {"message": "bad request"}})) is False
        assert gp._is_retryable_genai_error(ValueError("not genai")) is False

    def test_the_policy_withheld_finish_reasons_are_exact(self):
        from pyutilz.llm import gemini_provider as gp

        assert gp._BLOCKING_FINISH_REASONS == ("SAFETY", "BLOCK", "PROHIBITED_CONTENT", "RECITATION", "SPII")


class TestFreshGeminiProviderDefaults:
    """A new Gemini provider starts with zero per-call and session accounting, empty per-call collections, the documented
    per-call usage keys and the default model and concurrency (gemini_provider 160-213)."""

    def _provider(self):
        from pyutilz.llm import gemini_provider as gp

        return gp.GeminiProvider(api_key="test-key")  # pragma: allowlist secret

    def test_the_per_call_usage_has_its_three_keys_at_zero(self):
        assert self._provider()._last_usage == {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    def test_the_per_call_scalars_and_collections_start_empty(self):
        provider = self._provider()
        assert provider._last_finish_reason is None
        assert provider.last_cached_content_tokens == 0
        assert provider.last_safety_ratings == []
        assert provider.last_function_calls == []
        assert provider.last_all_candidates == []
        assert provider.last_grounding_metadata is None
        assert provider.last_citation_metadata is None

    def test_the_session_totals_start_at_zero(self):
        provider = self._provider()
        assert (provider._call_count, provider.total_prompt_tokens, provider.total_completion_tokens) == (0, 0, 0)
        assert (provider.total_reasoning_tokens, provider.total_cached_content_tokens) == (0, 0)
        assert provider._long_context_surcharge_usd == 0.0
        assert provider._long_context_calls == 0

    def test_the_default_model_and_concurrency_are_pinned(self):
        import inspect

        from pyutilz.llm import gemini_provider as gp

        params = inspect.signature(gp.GeminiProvider.__init__).parameters
        assert params["model"].default == "gemini-3.1-flash-lite-preview"
        assert params["max_concurrent"].default == 10


class _GeminiWarnLog:
    def __init__(self):
        self.calls: list = []

    def warning(self, fmt, *args, **kwargs):
        self.calls.append((fmt, args))


class TestGeminiLimitsAndUnknownPricing:
    """The context window and output ceiling come from the model's family prefix, the JSON support flags are on, and an
    unknown model's price warns once with its exact text (gemini_provider 216-258)."""

    def _provider(self, model):
        from pyutilz.llm import gemini_provider as gp

        return gp.GeminiProvider(api_key="test-key", model=model)  # pragma: allowlist secret

    def test_a_listed_family_has_its_context_window(self):
        assert self._provider("gemini-2.5-flash").context_window == 1_048_576
        assert self._provider("gemini-3.1-flash-lite-preview").context_window == 1_048_576

    def test_an_unlisted_model_gets_the_one_million_token_default_window(self):
        assert self._provider("gemma-unlisted").context_window == 1_048_576

    def test_the_output_ceiling_is_per_family_and_none_for_an_unlisted_one(self):
        assert self._provider("gemini-3.1-flash-lite-preview").max_output_tokens == 65536
        assert self._provider("gemini-2.5-flash").max_output_tokens == 65536
        assert self._provider("gemini-2.0-flash").max_output_tokens == 8192
        assert self._provider("gemini-1.5-pro").max_output_tokens is None

    def test_json_mode_and_json_schema_are_supported(self):
        provider = self._provider("gemini-2.5-flash")
        assert provider.supports_json_mode() is True
        assert provider.supports_json_schema() is True

    def test_an_unknown_model_price_warns_once_with_the_default_rates(self, monkeypatch):
        from pyutilz.llm import gemini_provider as gp

        log = _GeminiWarnLog()
        monkeypatch.setattr(gp, "logger", log)
        gp.GeminiProvider._warned_unknown_models.discard("gemma-price-unknown")
        provider = self._provider("gemma-price-unknown")
        provider._get_pricing()
        provider._get_pricing()
        assert log.calls == [
            ("Gemini pricing for %r is unknown; using the default %s per 1M. Cost estimates may be off.", ("gemma-price-unknown", (0.25, 1.5)))
        ]


class TestGeminiLongContextTierAndSurcharge:
    """Only the Pro rows and their versioned ids have the long-context tier, and a surcharge is tallied only above the
    prompt threshold, per call (gemini_provider 260-285)."""

    def _provider(self, model):
        from pyutilz.llm import gemini_provider as gp

        return gp.GeminiProvider(api_key="test-key", model=model)  # pragma: allowlist secret

    def test_a_pro_row_and_its_versioned_id_have_the_tier_and_a_flash_row_does_not(self):
        assert self._provider("gemini-2.5-pro")._long_context_tier() is not None
        assert self._provider("gemini-2.5-pro-002")._long_context_tier() is not None
        assert self._provider("gemini-2.5-flash")._long_context_tier() is None
        assert self._provider("gemini-2.5-proish")._long_context_tier() is None

    def test_the_current_pricing_carries_the_cache_rate_and_the_tier_of_its_row(self):
        priced = self._provider("gemini-2.5-pro")._current_pricing()
        assert priced.cache_hit == pytest.approx(0.125)
        assert priced.long_context is not None

    def test_no_surcharge_is_tallied_below_the_threshold(self):
        provider = self._provider("gemini-2.5-pro")
        provider._add_long_context_surcharge(100, 0, 10)
        assert provider._long_context_calls == 0

    def test_a_surcharge_is_tallied_per_call_above_the_threshold_on_pro(self):
        provider = self._provider("gemini-2.5-pro")
        provider._add_long_context_surcharge(250_000, 0, 1000)
        provider._add_long_context_surcharge(250_000, 0, 1000)
        assert provider._long_context_calls == 2
        assert provider._long_context_surcharge_usd > 0.0

    def test_a_flash_model_above_the_threshold_gets_no_surcharge(self):
        provider = self._provider("gemini-2.5-flash")
        provider._add_long_context_surcharge(250_000, 0, 1000)
        assert provider._long_context_calls == 0
        assert provider._long_context_surcharge_usd == 0.0


class TestGeminiClientTimeoutAndCandidateCount:
    """The client's timeout is the request timeout in milliseconds, and the candidate count is at least one (gemini_provider 189, 213)."""

    def test_the_client_timeout_is_the_request_timeout_in_milliseconds(self):
        from pyutilz.llm import gemini_provider as gp

        provider = gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret
        assert provider.client._api_client._http_options.timeout == int(provider._request_timeout_seconds * 1000)

    def test_a_candidate_count_below_one_is_raised_to_one(self):
        from pyutilz.llm import gemini_provider as gp

        assert gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash", candidate_count=0)._candidate_count == 1  # pragma: allowlist secret
        assert gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash", candidate_count=3)._candidate_count == 3  # pragma: allowlist secret


class TestGeminiSessionCostSummary:
    """The session summary splits cached and uncached prompt tokens at their own rates, prices completion plus reasoning at
    the output rate, adds the surcharge, and has the documented keys (gemini_provider 314-349)."""

    def _session(self):
        from pyutilz.llm import gemini_provider as gp

        provider = gp.GeminiProvider(api_key="test-key", model="gemini-2.5-pro")  # pragma: allowlist secret
        provider._call_count = 4
        provider.total_prompt_tokens = 2_000_000
        provider.total_cached_content_tokens = 500_000
        provider.total_completion_tokens = 100_000
        provider.total_reasoning_tokens = 50_000
        provider._long_context_surcharge_usd = 3.0
        provider._long_context_calls = 2
        return provider

    def test_the_session_summary_prices_cached_and_uncached_tokens_and_the_surcharge(self):
        summary = self._session().get_session_cost()
        assert summary["cache_hit_tokens"] == 500_000
        assert summary["cache_miss_tokens"] == 1_500_000
        assert summary["input_cost_usd"] == pytest.approx(1.9375, abs=1e-12)
        assert summary["output_cost_usd"] == pytest.approx(1.5, abs=1e-12)
        assert summary["long_context_surcharge_usd"] == 3.0
        assert summary["total_cost_usd"] == pytest.approx(6.4375, abs=1e-12)

    def test_the_session_summary_has_its_documented_keys_and_counts(self):
        summary = self._session().get_session_cost()
        assert set(summary) == {
            "calls",
            "prompt_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "long_context_surcharge_usd",
            "long_context_calls",
            "total_cost_usd",
        }
        assert summary["calls"] == 4
        assert summary["prompt_tokens"] == 2_000_000
        assert summary["completion_tokens"] == 100_000
        assert summary["reasoning_tokens"] == 50_000
        assert summary["long_context_calls"] == 2


class TestGeminiJsonForwarding:
    """generate_json forwards images, thinking and a schema only when given, and always in json mode (gemini_provider 314-319)."""

    def _calls(self, monkeypatch, **kwargs):
        from pyutilz.llm import gemini_provider as gp
        import asyncio

        provider = gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret
        seen: list = []

        async def spy(prompt, system, temperature, max_tokens, **extra):
            seen.append((prompt, system, temperature, max_tokens, extra))
            return {}

        monkeypatch.setattr(provider, "_generate_json_via", spy)
        asyncio.run(provider.generate_json("p", system="s", temperature=0.2, max_tokens=50, **kwargs))
        return seen[0]

    def test_a_plain_call_forwards_only_json_mode(self, monkeypatch):
        prompt, system, temperature, max_tokens, extra = self._calls(monkeypatch)
        assert (prompt, system, temperature, max_tokens) == ("p", "s", 0.2, 50)
        assert extra == {"json_mode": True}

    def test_images_thinking_and_a_schema_are_forwarded_when_given(self, monkeypatch):
        *_, extra = self._calls(monkeypatch, images=["img"], thinking=False, json_schema={"type": "object"})
        assert extra == {"json_mode": True, "images": ["img"], "thinking": False, "json_schema": {"type": "object"}}


class _UsageMeta:
    def __init__(self, prompt, candidates, thoughts, cached):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts
        self.cached_content_token_count = cached


class _UsageResponse:
    def __init__(self, usage):
        self.usage_metadata = usage


class TestGeminiUsageRecording:
    """A response's usage metadata sets the per-call usage and adds to every session total, once per call; a response with no
    metadata records nothing; a long prompt on Pro is surcharged (gemini_provider 394-402)."""

    def _provider(self, model="gemini-2.5-flash"):
        from pyutilz.llm import gemini_provider as gp

        return gp.GeminiProvider(api_key="test-key", model=model)  # pragma: allowlist secret

    def test_a_response_sets_the_per_call_usage_and_the_last_cached_count(self):
        provider = self._provider()
        provider._record_gemini_usage(_UsageResponse(_UsageMeta(1000, 200, 50, 300)))
        assert provider._last_usage == {
            "input_tokens": 1000,
            "output_tokens": 200,
            "reasoning_tokens": 50,
            "cached_content_token_count": 300,
        }
        assert provider.last_cached_content_tokens == 300

    def test_a_response_adds_each_count_to_the_session_totals_once(self):
        provider = self._provider()
        provider._record_gemini_usage(_UsageResponse(_UsageMeta(1000, 200, 50, 300)))
        provider._record_gemini_usage(_UsageResponse(_UsageMeta(500, 100, 25, 100)))
        assert provider._call_count == 2
        assert provider.total_prompt_tokens == 1500
        assert provider.total_completion_tokens == 300
        assert provider.total_reasoning_tokens == 75
        assert provider.total_cached_content_tokens == 400

    def test_a_response_without_usage_records_nothing(self):
        provider = self._provider()
        provider._record_gemini_usage(_UsageResponse(None))
        assert provider._call_count == 0
        assert provider.total_prompt_tokens == 0

    def test_a_long_prompt_on_pro_is_surcharged_and_a_flash_one_is_not(self):
        pro = self._provider("gemini-2.5-pro")
        pro._record_gemini_usage(_UsageResponse(_UsageMeta(250_000, 1000, 0, 0)))
        assert pro._long_context_calls == 1
        flash = self._provider("gemini-2.5-flash")
        flash._record_gemini_usage(_UsageResponse(_UsageMeta(250_000, 1000, 0, 0)))
        assert flash._long_context_calls == 0


class TestGeminiGenerateConfig:
    """The request config carries temperature, a token cap, the system instruction, the JSON mime type and schema, and a candidate
    count only above one, each only when set (gemini_provider 355-391)."""

    def _config(self, **kwargs):
        from pyutilz.llm import gemini_provider as gp

        provider = gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash", candidate_count=kwargs.pop("candidates", 1))  # pragma: allowlist secret
        defaults = dict(max_tokens=None, temperature=None, system=None, json_mode=False, json_schema=None, thinking=None)
        defaults.update(kwargs)
        return provider._build_generate_config(**defaults)

    def test_a_temperature_is_sent_only_when_given(self):
        assert getattr(self._config(temperature=None), "temperature", None) is None
        assert self._config(temperature=0.3).temperature == 0.3

    def test_a_token_cap_is_sent_only_when_positive(self):
        assert getattr(self._config(max_tokens=None), "max_output_tokens", None) is None
        assert getattr(self._config(max_tokens=0), "max_output_tokens", None) is None
        assert self._config(max_tokens=50).max_output_tokens == 50

    def test_the_system_instruction_is_set_only_when_non_empty(self):
        assert getattr(self._config(system=""), "system_instruction", None) is None
        assert self._config(system="be terse").system_instruction == "be terse"

    def test_json_mode_or_a_schema_sets_the_json_mime_type(self):
        assert getattr(self._config(json_mode=False), "response_mime_type", None) is None
        assert self._config(json_mode=True).response_mime_type == "application/json"
        assert self._config(json_schema={"type": "object"}).response_mime_type == "application/json"

    def test_a_wrapped_schema_is_unwrapped_only_with_its_name_or_strict_marker(self):
        wrapped = {"name": "n", "strict": True, "schema": {"type": "object"}}
        assert self._config(json_schema=wrapped).response_json_schema == {"type": "object"}
        bare = {"schema": {"type": "object"}}
        assert self._config(json_schema=bare).response_json_schema == bare

    def test_a_candidate_count_is_sent_only_above_one(self):
        assert getattr(self._config(candidates=1), "candidate_count", None) is None
        assert self._config(candidates=3).candidate_count == 3

    def test_no_thinking_config_is_sent_without_a_thinking_request(self):
        assert getattr(self._config(thinking=None), "thinking_config", None) is None


class _Feedback:
    def __init__(self, block_reason=None, message=None):
        self.block_reason = block_reason
        self.block_reason_message = message


class _ResponseWithFeedback:
    def __init__(self, candidates, feedback=None):
        self.candidates = candidates
        self.prompt_feedback = feedback


class TestGeminiSafetyBlocks:
    """A prompt blocked before any candidate raises with its block reason and message, a response that ends on a policy finish
    reason raises with the finish reason, and anything else passes; generate defaults to temperature 0.7, no token cap, not JSON
    (gemini_provider 404-440)."""

    def _provider(self):
        from pyutilz.llm import gemini_provider as gp

        return gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret

    def test_a_blocked_prompt_raises_with_its_reason_and_message(self):
        from pyutilz.llm.exceptions import LLMSafetyBlockError

        provider = self._provider()
        response = _ResponseWithFeedback([], _Feedback("SAFETY", "policy"))
        with pytest.raises(LLMSafetyBlockError) as info:
            provider._raise_if_blocked(response, "STOP")
        assert str(info.value) == "Gemini blocked the prompt (block_reason=SAFETY)"
        assert info.value.details == {
            "finish_reason": None,
            "block_reason": "SAFETY",
            "block_reason_message": "policy",
            "safety_ratings": [],
        }

    def test_a_prompt_with_no_block_reason_is_not_refused(self):
        provider = self._provider()
        provider._raise_if_blocked(_ResponseWithFeedback([], _Feedback(None)), "STOP")

    def test_a_response_on_a_policy_finish_reason_raises_with_it(self):
        from pyutilz.llm.exceptions import LLMSafetyBlockError

        provider = self._provider()
        provider._last_finish_reason = "SAFETY"
        with pytest.raises(LLMSafetyBlockError) as info:
            provider._raise_if_blocked(_ResponseWithFeedback(["candidate"]), "SAFETY")
        assert str(info.value) == "Gemini blocked response by safety filter (finish_reason=SAFETY)"
        assert info.value.details == {"finish_reason": "SAFETY", "safety_ratings": []}

    def test_an_ordinary_finish_reason_is_not_refused(self):
        provider = self._provider()
        provider._raise_if_blocked(_ResponseWithFeedback(["candidate"]), "STOP")

    def test_generate_defaults_are_temperature_point_seven_no_cap_and_not_json(self):
        import inspect

        params = inspect.signature(self._provider().generate).parameters
        assert params["temperature"].default == 0.7
        assert params["max_tokens"].default == 0
        assert params["json_mode"].default is False
