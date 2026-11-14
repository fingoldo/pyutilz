"""Gemini provider call contracts: session cost, JSON forwarding, usage recording, generate config, safety blocks,
per-call resets, response text, billing hints and candidate metadata. Split out of test_gemini_provider.py to stay under the 1000-LOC budget."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import pyutilz.llm.gemini_provider as gemini_provider_module
from pyutilz.llm.gemini_provider import GeminiProvider


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
        # genai (the client) is None on 3.8 as well: the constructor calls genai.Client(...) and types.HttpOptions(...).
        with patch.multiple(
            "pyutilz.llm.gemini_provider",
            genai=gemini_provider_module.genai if gemini_provider_module.genai is not None else MagicMock(),
            types=gemini_provider_module.types if gemini_provider_module.types is not None else MagicMock(),
        ):
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


@pytest.mark.skipif(gemini_provider_module.types is None, reason="asserts on the real google-genai types, not installed on this Python")
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
        # The contract is "returns None without raising"; a raise fails the test by itself, the assert pins the return value.
        assert provider._raise_if_blocked(_ResponseWithFeedback([], _Feedback(None)), "STOP") is None

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
        assert provider._raise_if_blocked(_ResponseWithFeedback(["candidate"]), "STOP") is None

    def test_generate_defaults_are_temperature_point_seven_no_cap_and_not_json(self):
        import inspect

        params = inspect.signature(self._provider().generate).parameters
        assert params["temperature"].default == 0.7
        assert params["max_tokens"].default == 0
        assert params["json_mode"].default is False


class _FakeCandidate:
    def __init__(self, finish_reason="STOP"):
        self.finish_reason = finish_reason
        self.safety_ratings = []


class _FakeGenResponse:
    def __init__(self, text="hello", finish_reason="STOP", candidates=True):
        self.text = text
        self.candidates = [_FakeCandidate(finish_reason)] if candidates else []
        self.prompt_feedback = None
        self.usage_metadata = None


class _FakeModels:
    def __init__(self, response):
        self._response = response
        self.calls: list = []

    async def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._response


class _FakeGenClient:
    def __init__(self, response):
        self.aio = type("Aio", (), {})()
        self.aio.models = _FakeModels(response)


class TestGeminiGenerateCall:
    """generate sends the model, the parts and the config to the native async client, caps the budget through the context fit, and
    returns the answer text (gemini_provider 467-500)."""

    def _provider(self, response, monkeypatch):
        from pyutilz.llm import gemini_provider as gp

        provider = gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret
        provider.client = _FakeGenClient(response)
        return provider

    def test_generate_sends_the_model_and_returns_the_answer_text(self, monkeypatch):
        import asyncio

        response = _FakeGenResponse(text="the answer")
        provider = self._provider(response, monkeypatch)
        out = asyncio.run(provider.generate("hi"))
        assert out == "the answer"
        call = provider.client.aio.models.calls[0]
        assert call["model"] == "gemini-2.5-flash"

    def test_generate_passes_a_zero_budget_as_no_ceiling_through_the_context_fit(self, monkeypatch):
        import asyncio

        from pyutilz.llm import gemini_provider as gp

        seen: list = []
        monkeypatch.setattr(gp.GeminiProvider, "fit_max_tokens_to_context", lambda self, budget, prompt, system: seen.append(budget) or budget)
        provider = self._provider(_FakeGenResponse(), monkeypatch)
        asyncio.run(provider.generate("hi", max_tokens=0))
        asyncio.run(provider.generate("hi", max_tokens=300))
        assert seen == [None, 300]


class TestGeminiPerCallStateResets:
    """A new generate call starts from a clean per-call state, so a count left by an earlier call does not survive a response that
    reports no usage (gemini_provider 467)."""

    def test_a_stale_cached_count_is_cleared_by_the_next_call(self):
        import asyncio

        from pyutilz.llm import gemini_provider as gp

        provider = gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret
        provider.client = _FakeGenClient(_FakeGenResponse())
        provider.last_cached_content_tokens = 7

        async def call_then_read():
            await provider.generate("hi")
            return provider.last_cached_content_tokens

        assert asyncio.run(call_then_read()) == 0


class _RaisingTextResponse(_FakeGenResponse):
    def __init__(self, error, finish_reason="STOP"):
        self.candidates = [_FakeCandidate(finish_reason)]
        self.prompt_feedback = None
        self.usage_metadata = None
        self._error = error

    @property
    def text(self):
        raise self._error


class TestGeminiResponseTextHandling:
    """A response that ran out of tokens is refused with its partial text (empty, with a warning, when that text cannot be read);
    a response with no readable text is refused as a safety block with its details (gemini_provider 511-539)."""

    def _generate(self, response):
        import asyncio

        from pyutilz.llm import gemini_provider as gp

        provider = gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret
        provider.client = _FakeGenClient(response)

        async def call():
            return await provider.generate("hi")

        return asyncio.run(call())

    def test_a_token_cutoff_is_refused_with_its_partial_text_and_finish_reason(self):
        from pyutilz.llm.exceptions import LLMTruncationError

        with pytest.raises(LLMTruncationError) as info:
            self._generate(_FakeGenResponse(text="half", finish_reason="MAX_TOKENS"))
        assert str(info.value) == "Gemini response truncated by max_tokens (finish_reason=MAX_TOKENS)"
        assert info.value.finish_reason == "MAX_TOKENS"
        assert info.value.partial_text == "half"

    def test_an_unreadable_partial_text_is_empty_and_warned_about(self, monkeypatch):
        from pyutilz.llm import gemini_provider as gp
        from pyutilz.llm.exceptions import LLMTruncationError

        recorded: list = []

        class _WarnLog:
            def warning(self, fmt, *args, **kwargs):
                recorded.append((fmt, args))

        monkeypatch.setattr(gp, "logger", _WarnLog())
        with pytest.raises(LLMTruncationError) as info:
            self._generate(_RaisingTextResponse(ValueError("no text"), finish_reason="MAX_TOKENS"))
        assert info.value.partial_text == ""
        fmt, args = recorded[0]
        assert fmt == "Could not read partial text off a truncated Gemini response; reporting it as empty: %s"
        assert str(args[0]) == "no text"

    def test_a_response_with_no_readable_text_is_refused_as_a_safety_block(self):
        from pyutilz.llm.exceptions import LLMSafetyBlockError

        with pytest.raises(LLMSafetyBlockError) as info:
            self._generate(_RaisingTextResponse(AttributeError("gone")))
        assert str(info.value) == "Gemini response has no text (likely safety block): gone"
        assert info.value.details == {"finish_reason": "STOP", "safety_ratings": []}


class _CountModels:
    def __init__(self, total=None, error=None):
        self._total = total
        self._error = error

    async def count_tokens(self, model, contents):
        if self._error is not None:
            raise self._error
        return type("Counted", (), {"total_tokens": self._total})()


class TestGeminiCountTokensAndBillingHints:
    """count_tokens asks the API and falls back to the local estimate when the API fails or reports no total; the billing and
    rate-limit queries refuse with their exact hints (gemini_provider 597-640)."""

    def _provider(self, models):
        from pyutilz.llm import gemini_provider as gp

        provider = gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret
        provider.client = type("C", (), {})()
        provider.client.aio = type("A", (), {})()
        provider.client.aio.models = models
        return provider

    def test_count_tokens_returns_the_api_total(self):
        import asyncio

        assert asyncio.run(self._provider(_CountModels(total=42)).count_tokens("hello")) == 42

    def test_a_missing_total_falls_back_to_the_local_estimate(self, monkeypatch):
        import asyncio

        import pyutilz.llm.token_counter as tc

        monkeypatch.setattr(tc, "count_tokens", lambda text: 99)
        assert asyncio.run(self._provider(_CountModels(total=None)).count_tokens("hello")) == 99

    def test_an_api_failure_falls_back_and_is_logged_at_debug(self, monkeypatch):
        import asyncio

        import pyutilz.llm.token_counter as tc
        from pyutilz.llm import gemini_provider as gp

        monkeypatch.setattr(tc, "count_tokens", lambda text: 77)
        recorded: list = []

        class _DebugLog:
            def debug(self, fmt, *args, **kwargs):
                recorded.append((fmt, args))

        monkeypatch.setattr(gp, "logger", _DebugLog())
        assert asyncio.run(self._provider(_CountModels(error=ConnectionError("down"))).count_tokens("hello")) == 77
        fmt, args = recorded[0]
        assert fmt == "Gemini count_tokens API failed (%s); falling back to tiktoken."
        assert str(args[0]) == "down"

    def test_the_billing_query_refuses_with_its_gcp_hint(self):
        import asyncio

        with pytest.raises(NotImplementedError) as info:
            asyncio.run(self._provider(_CountModels(total=1)).get_account_credits())
        assert str(info.value) == (
            "Gemini billing is GCP-based — there is no balance endpoint reachable with a Gemini API key. "
            "Check console.cloud.google.com/billing or use the Cloud Billing API with separate GCP credentials."
        )

    def test_the_rate_limit_query_refuses_with_its_quota_hint(self):
        import asyncio

        with pytest.raises(NotImplementedError) as info:
            asyncio.run(self._provider(_CountModels(total=1)).check_account_limits())
        assert str(info.value) == ("Gemini does not expose per-key rate limits via API. Quotas are GCP-side at console.cloud.google.com/iam-admin/quotas.")


class _Rating:
    def __init__(self, category, probability, blocked=None):
        self.category = category
        self.probability = probability
        if blocked is not None:
            self.blocked = blocked


class _CandidateWithRatings:
    def __init__(self, ratings):
        self.finish_reason = "STOP"
        self.safety_ratings = ratings
        self.grounding_metadata = None
        self.citation_metadata = None


class _MetaResponse:
    def __init__(self, candidates):
        self.candidates = candidates
        self.text = "x"


class TestGeminiCandidateMetadataCapture:
    """Safety ratings are recorded with their category, probability and blocked flag (false when the flag is absent), and a
    failure while reading metadata is logged at debug rather than raised (gemini_provider 565-594)."""

    def _provider(self):
        from pyutilz.llm import gemini_provider as gp

        return gp.GeminiProvider(api_key="test-key", model="gemini-2.5-flash")  # pragma: allowlist secret

    def test_a_rating_without_a_blocked_flag_is_recorded_as_not_blocked(self):
        provider = self._provider()
        provider._capture_candidate_metadata(_MetaResponse([_CandidateWithRatings([_Rating("HARM", "LOW")])]))
        assert provider.last_safety_ratings == [{"category": "HARM", "probability": "LOW", "blocked": False}]

    def test_a_rating_with_a_blocked_flag_keeps_it(self):
        provider = self._provider()
        provider._capture_candidate_metadata(_MetaResponse([_CandidateWithRatings([_Rating("HARM", "HIGH", blocked=True)])]))
        assert provider.last_safety_ratings == [{"category": "HARM", "probability": "HIGH", "blocked": True}]

    def test_a_metadata_read_failure_is_logged_at_debug_and_not_raised(self, monkeypatch):
        from pyutilz.llm import gemini_provider as gp

        recorded: list = []

        class _DebugLog:
            def debug(self, fmt, *args, **kwargs):
                recorded.append((fmt, args))

        monkeypatch.setattr(gp, "logger", _DebugLog())

        class _Broken:
            @property
            def candidates(self):
                raise RuntimeError("boom")

        self._provider()._capture_candidate_metadata(_Broken())
        fmt, args = recorded[0]
        assert fmt == "Gemini candidate metadata capture failed: %s"
        assert str(args[0]) == "boom"
