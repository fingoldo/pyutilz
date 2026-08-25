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
    if gemini_provider_module.types is not None:
        yield
        return
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
        assert p.max_output_tokens == 8192


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
