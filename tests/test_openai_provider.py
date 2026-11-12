"""Behaviour of ``pyutilz.llm.openai_provider``: the reasoning-model family test, the per-family timeouts, the request
body reshaped for OpenAI (``max_tokens`` renamed, ``temperature`` dropped for reasoning models), the reasoning-effort
mapping for ``thinking=``, the learned repair of a rejected ``temperature``, and the refusal to report credits.
No network is used: the provider is built with a test key and only its pure helpers are exercised."""

from __future__ import annotations

import asyncio

import pytest

from pyutilz.llm import openai_provider as op
from pyutilz.llm.exceptions import LLMProviderError

_FAKE_KEY = "sk-test-key"  # pragma: allowlist secret


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(op, "_MODELS_REJECTING_TEMPERATURE", set())
    return op.OpenAIProvider(api_key=_FAKE_KEY, model="gpt-5-mini")


class TestReasoningFamily:
    @pytest.mark.parametrize("model", ["o3-mini", "o1", "gpt-5-mini"])
    def test_reasoning_families_are_recognised(self, model):
        assert op.is_reasoning_model(model) is True

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1-mini"])
    def test_non_reasoning_models_are_not(self, model):
        assert op.is_reasoning_model(model) is False


class TestTimeouts:
    def test_the_o_series_gets_the_longest_non_pro_timeout(self, provider):
        assert provider._get_timeout("o3") == 1200.0

    def test_a_pro_model_gets_the_pro_timeout(self, provider):
        assert provider._get_timeout("gpt-5-pro") == 600.0

    def test_an_ordinary_model_gets_the_default_timeout(self, provider):
        assert provider._get_timeout("gpt-4o") == 240.0


class TestRequestBody:
    def test_max_tokens_is_renamed_for_the_current_api(self, provider):
        out = provider._openai_body({"model": "gpt-4o", "max_tokens": 50})
        assert out == {"model": "gpt-4o", "max_completion_tokens": 50}

    def test_temperature_is_kept_for_a_non_reasoning_model(self, monkeypatch):
        monkeypatch.setattr(op, "_MODELS_REJECTING_TEMPERATURE", set())
        plain = op.OpenAIProvider(api_key=_FAKE_KEY, model="gpt-4o")
        out = plain._openai_body({"model": "gpt-4o", "temperature": 0.7})
        assert out["temperature"] == 0.7

    def test_temperature_is_dropped_for_a_reasoning_model(self, provider):
        out = provider._openai_body({"model": "gpt-5-mini", "temperature": 0.7})
        assert "temperature" not in out

    def test_the_caller_body_is_not_mutated(self, provider):
        original = {"max_tokens": 10, "temperature": 0.1}
        provider._openai_body(original)
        assert original == {"max_tokens": 10, "temperature": 0.1}


class TestThinkingField:
    def test_thinking_on_a_reasoning_model_asks_for_medium_effort(self, provider):
        assert provider._thinking_request_field(True) == {"reasoning_effort": "medium"}

    def test_thinking_off_asks_for_the_lowest_effort_the_family_supports(self, provider):
        field = provider._thinking_request_field(False)
        assert field is not None and set(field) == {"reasoning_effort"}

    def test_a_non_reasoning_model_sends_no_reasoning_field(self, monkeypatch):
        plain = op.OpenAIProvider(api_key=_FAKE_KEY, model="gpt-4o")
        assert plain._thinking_request_field(True) is None


class TestBilledOutput:
    def test_completion_tokens_already_include_reasoning_and_are_returned_as_is(self, provider):
        assert provider._compute_billed_output(completion_tokens=120, reasoning_tokens=80) == 120


class TestCreditsRefusal:
    def test_the_credit_query_raises_rather_than_returning_a_guess(self, provider):
        with pytest.raises(NotImplementedError, match="remaining credit"):
            asyncio.run(provider.get_account_credits())


class TestLearnedTemperatureRepair:
    def test_a_400_naming_temperature_drops_it_and_remembers_the_model(self, provider):
        body = {"model": "gpt-5-mini", "temperature": 0.7, "messages": []}
        repaired = provider._body_after_rejected_request(body, 400, "Unsupported parameter: temperature")
        assert repaired == {"model": "gpt-5-mini", "messages": []}
        assert provider.model_name in op._MODELS_REJECTING_TEMPERATURE

    def test_a_non_400_status_is_not_repaired(self, provider):
        assert provider._body_after_rejected_request({"temperature": 0.7}, 500, "temperature") is None

    def test_a_400_about_something_else_is_not_repaired(self, provider):
        assert provider._body_after_rejected_request({"temperature": 0.7}, 400, "bad messages") is None


class TestConstruction:
    def test_a_missing_key_is_refused_with_a_clear_message(self, monkeypatch):
        class _Settings:
            openai_api_key = None

        monkeypatch.setattr(op, "get_llm_settings", lambda: _Settings())
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            op.OpenAIProvider(api_key=None)

class TestPriceAndLimitTablesArePinned:
    """Every price, cache rate, output cap and context window is pinned as a whole table, so a changed row or a
    misplaced number fails here, not only where a test happens to name the model (openai_provider 35-290). Values
    are the pricing page of 2026-10-03 as the module records them."""

    def test_the_price_cache_cap_and_context_tables_are_exact(self):
        expected = {
    '_PRICING': {'gpt-4.1': (2.0, 8.0),
 'gpt-4.1-mini': (0.4, 1.6),
 'gpt-4.1-nano': (0.1, 0.4),
 'gpt-4o': (2.5, 10.0),
 'gpt-4o-mini': (0.15, 0.6),
 'gpt-5': (1.25, 10.0),
 'gpt-5-chat': (1.25, 10.0),
 'gpt-5-codex': (1.25, 10.0),
 'gpt-5-mini': (0.25, 2.0),
 'gpt-5-nano': (0.05, 0.4),
 'gpt-5-pro': (15.0, 120.0),
 'gpt-5.1': (1.25, 10.0),
 'gpt-5.1-codex': (1.25, 10.0),
 'gpt-5.2': (1.75, 14.0),
 'gpt-5.2-pro': (21.0, 168.0),
 'gpt-5.4': (2.5, 15.0),
 'gpt-5.4-mini': (0.75, 4.5),
 'gpt-5.4-nano': (0.2, 1.25),
 'gpt-5.4-pro': (30.0, 180.0),
 'gpt-5.5': (5.0, 30.0),
 'gpt-5.5-pro': (30.0, 180.0),
 'gpt-5.6-luna': (0.2, 1.2),
 'gpt-5.6-sol': (4.0, 20.0),
 'gpt-5.6-terra': (2.0, 12.0),
 'gpt-6-astra': (10.0, 50.0),
 'gpt-6-luna': (0.1, 0.5),
 'gpt-6-sol': (2.0, 10.0),
 'gpt-6.1-sol': (2.0, 10.0),
 'o1': (15.0, 60.0),
 'o1-pro': (150.0, 600.0),
 'o3': (2.0, 8.0),
 'o3-mini': (1.1, 4.4),
 'o3-pro': (20.0, 80.0),
 'o4-mini': (1.1, 4.4)},
    '_CACHE_HIT_COST': {'gpt-4.1': 0.5,
 'gpt-4.1-mini': 0.1,
 'gpt-4.1-nano': 0.025,
 'gpt-4o': 1.25,
 'gpt-4o-mini': 0.075,
 'gpt-5': 0.125,
 'gpt-5-chat': 0.125,
 'gpt-5-codex': 0.125,
 'gpt-5-mini': 0.025,
 'gpt-5-nano': 0.005,
 'gpt-5-pro': 15.0,
 'gpt-5.1': 0.125,
 'gpt-5.1-codex': 0.125,
 'gpt-5.2': 0.175,
 'gpt-5.2-pro': 21.0,
 'gpt-5.4': 0.25,
 'gpt-5.4-mini': 0.075,
 'gpt-5.4-nano': 0.02,
 'gpt-5.4-pro': 30.0,
 'gpt-5.5': 0.5,
 'gpt-5.5-pro': 30.0,
 'gpt-5.6-luna': 0.02,
 'gpt-5.6-sol': 0.4,
 'gpt-5.6-terra': 0.2,
 'gpt-6-astra': 1.0,
 'gpt-6-luna': 0.01,
 'gpt-6-sol': 0.2,
 'gpt-6.1-sol': 0.1,
 'o1': 7.5,
 'o1-pro': 150.0,
 'o3': 0.5,
 'o3-mini': 0.55,
 'o3-pro': 20.0,
 'o4-mini': 0.275},
    '_CACHE_WRITE_COST': {'gpt-5.6-luna': 0.25,
 'gpt-5.6-sol': 5.0,
 'gpt-5.6-terra': 2.5,
 'gpt-6-astra': 12.5,
 'gpt-6-luna': 0.125,
 'gpt-6-sol': 2.5,
 'gpt-6.1-sol': 2.5},
    '_MAX_TOKENS': {'gpt-4.1': 32000,
 'gpt-4.1-mini': 32000,
 'gpt-4.1-nano': 32000,
 'gpt-4o': 16384,
 'gpt-4o-mini': 16384,
 'gpt-5': 128000,
 'gpt-5-chat': 128000,
 'gpt-5-codex': 128000,
 'gpt-5-mini': 128000,
 'gpt-5-nano': 128000,
 'gpt-5-pro': 128000,
 'gpt-5.1': 128000,
 'gpt-5.1-codex': 128000,
 'gpt-5.2': 128000,
 'gpt-5.2-pro': 128000,
 'gpt-5.4': 128000,
 'gpt-5.4-mini': 128000,
 'gpt-5.4-nano': 128000,
 'gpt-5.4-pro': 128000,
 'gpt-5.5': 128000,
 'gpt-5.5-pro': 128000,
 'gpt-5.6-luna': 128000,
 'gpt-5.6-sol': 128000,
 'gpt-5.6-terra': 128000,
 'gpt-6-astra': 128000,
 'gpt-6-luna': 128000,
 'gpt-6-sol': 128000,
 'gpt-6.1-sol': 128000,
 'o1': 100000,
 'o1-pro': 100000,
 'o3': 100000,
 'o3-mini': 100000,
 'o3-pro': 100000,
 'o4-mini': 100000},
    '_CONTEXT_WINDOW': {'gpt-4.1': 1000000,
 'gpt-4.1-mini': 1000000,
 'gpt-4.1-nano': 1000000,
 'gpt-4o': 128000,
 'gpt-4o-mini': 128000,
 'gpt-5': 400000,
 'gpt-5-chat': 400000,
 'gpt-5-codex': 400000,
 'gpt-5-mini': 400000,
 'gpt-5-nano': 400000,
 'gpt-5-pro': 400000,
 'gpt-5.1': 400000,
 'gpt-5.1-codex': 400000,
 'gpt-5.2': 400000,
 'gpt-5.2-pro': 400000,
 'gpt-5.4': 400000,
 'gpt-5.4-mini': 400000,
 'gpt-5.4-nano': 400000,
 'gpt-5.4-pro': 400000,
 'gpt-5.5': 400000,
 'gpt-5.5-pro': 400000,
 'gpt-5.6-luna': 400000,
 'gpt-5.6-sol': 400000,
 'gpt-5.6-terra': 400000,
 'gpt-6-astra': 1050000,
 'gpt-6-luna': 1050000,
 'gpt-6-sol': 1050000,
 'gpt-6.1-sol': 1050000,
 'o1': 200000,
 'o1-pro': 200000,
 'o3': 200000,
 'o3-mini': 200000,
 'o3-pro': 200000,
 'o4-mini': 200000},
}
        for name, value in expected.items():
            assert getattr(op, name) == value, name

    def test_the_long_context_model_set_is_exact(self):
        assert sorted(op._LONG_CONTEXT_MODELS) == ['gpt-5.4',
 'gpt-5.5',
 'gpt-5.6-luna',
 'gpt-5.6-sol',
 'gpt-5.6-terra',
 'gpt-6-astra',
 'gpt-6-luna',
 'gpt-6-sol',
 'gpt-6.1-sol']

    def test_the_long_context_tier_is_exact(self):
        assert tuple(op._LONG_CONTEXT_TIER) == (272000, 2.0, 1.5, 2.0, 2.0, False)


class TestErrorBodiesAndEndpoint:
    """The error object of a body, the 429 policy (quota versus ordinary limit), the missing-key message, the endpoint and the timeouts (openai_provider 254-300)."""

    def _resp(self, status, body=None, text=""):
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        if body is None:
            return httpx.Response(status, text=text, request=request)
        return httpx.Response(status, json=body, request=request)

    def test_a_body_that_is_not_json_gives_no_error_object_and_is_logged(self, monkeypatch):
        recorder: list = []

        class _DebugLog:
            def debug(self, fmt, *args, **kwargs):
                recorder.append((fmt, args))

        monkeypatch.setattr(op, "logger", _DebugLog())
        assert op._error_object(self._resp(500, text="<html>")) == {}
        fmt, args = recorder[0]
        assert fmt == "_error_object: error response body is not JSON: %s: %s"
        assert args[0] == "JSONDecodeError"

    def test_an_object_without_an_error_or_a_non_object_body_gives_no_error_object(self):
        assert op._error_object(self._resp(400, body={"other": 1})) == {}
        assert op._error_object(self._resp(400, body=[1, 2])) == {}

    def test_the_error_object_is_returned_as_is(self):
        assert op._error_object(self._resp(400, body={"error": {"type": "x", "code": "y"}})) == {"type": "x", "code": "y"}

    def test_an_exhausted_quota_429_is_refused_with_the_reason_and_message(self):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        with pytest.raises(LLMProviderError) as info:
            provider._handle_special_status(self._resp(429, body={"error": {"type": "insufficient_quota", "message": "top up"}}))
        assert str(info.value) == "OpenAI account has no credit left (HTTP 429 insufficient_quota): top up"

    def test_a_credit_balance_code_names_the_code_as_the_reason(self):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        with pytest.raises(LLMProviderError) as info:
            provider._handle_special_status(self._resp(429, body={"error": {"code": "credit_balance_exhausted", "message": "m"}}))
        assert "(HTTP 429 credit_balance_exhausted)" in str(info.value)

    def test_an_ordinary_429_is_not_refused(self):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        assert provider._handle_special_status(self._resp(429, body={"error": {"type": "rate_limit_exceeded"}})) is None

    def test_a_non_429_status_is_ignored_by_the_policy(self):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        assert provider._handle_special_status(self._resp(402, body={"error": {"type": "insufficient_quota"}})) is None

    def test_a_missing_key_is_refused_with_the_exact_message(self, monkeypatch):
        class _Settings:
            openai_api_key = None

        monkeypatch.setattr(op, "get_llm_settings", lambda: _Settings())
        with pytest.raises(ValueError) as info:
            op.OpenAIProvider(api_key=None)
        assert str(info.value) == "OpenAI API key not provided. Set OPENAI_API_KEY in .env or pass api_key="

    def test_the_endpoint_and_provider_name_are_pinned(self):
        assert op.OpenAIProvider._base_url == "https://api.openai.com/v1"
        assert op.OpenAIProvider._provider_name == "OpenAI"

    def test_reasoning_families_get_the_long_timeouts(self):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        assert provider._get_timeout("o3-mini") == 1200.0
        assert provider._get_timeout("gpt-5.5-pro") == 600.0
        assert provider._get_timeout("gpt-5-mini") == 240.0

    def test_the_default_model_and_concurrency_are_pinned(self):
        import inspect

        params = inspect.signature(op.OpenAIProvider.__init__).parameters
        assert params["model"].default == "gpt-5-mini"
        assert params["max_concurrent"].default == 10

    def test_dated_snapshots_and_latest_are_stripped_but_not_other_suffixes(self):
        assert op._SNAPSHOT_SUFFIX.search("gpt-5-2026-01-15")
        assert op._SNAPSHOT_SUFFIX.search("gpt-5-20260115")
        assert op._SNAPSHOT_SUFFIX.search("gpt-5-latest")
        assert op._SNAPSHOT_SUFFIX.search("gpt-5-mini") is None


class TestQuotaRefusalDetails:
    """The quota refusal carries its status and error type and code, and an ordinary 429 is only logged (openai_provider 309-312)."""

    def _resp(self, status, body):
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        return httpx.Response(status, json=body, request=request)

    def test_the_quota_refusal_details_carry_the_status_type_and_code(self):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        with pytest.raises(LLMProviderError) as info:
            provider._handle_special_status(self._resp(429, {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted", "message": "m"}}))
        assert info.value.details == {"status_code": 429, "type": "insufficient_quota", "code": "credit_balance_exhausted"}

    def test_the_quota_refusal_details_hold_none_for_absent_fields(self):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        with pytest.raises(LLMProviderError) as info:
            provider._handle_special_status(self._resp(429, {"error": {"code": "credit_balance_exhausted"}}))
        assert info.value.details == {"status_code": 429, "type": None, "code": "credit_balance_exhausted"}

    def test_an_ordinary_429_logs_the_retry_warning(self, monkeypatch):
        recorded: list = []

        class _WarnLog:
            def warning(self, msg, *args, **kwargs):
                recorded.append(msg)

        monkeypatch.setattr(op, "logger", _WarnLog())
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        provider._handle_special_status(self._resp(429, {"error": {"type": "rate_limit_exceeded"}}))
        assert recorded == ["OpenAI rate limit hit (HTTP 429). Retrying via tenacity loop."]


class _PriceWarnLog:
    def __init__(self):
        self.calls: list = []

    def warning(self, fmt, *args, **kwargs):
        self.calls.append((fmt, args))


@pytest.fixture
def price_log(monkeypatch):
    from pyutilz.dev.logginglib import reset_log_throttles

    reset_log_throttles()
    log = _PriceWarnLog()
    monkeypatch.setattr(op, "logger", log)
    yield log
    reset_log_throttles()


class TestPricingLookupAndUnknownModels:
    """A model resolves to its row exactly or as a dated/latest snapshot, a family match prices as its family, a genuine miss prices at gpt-5-mini; only an unknown model warns, once (openai_provider 408-472)."""

    def _provider(self, model="gpt-5-mini"):
        return op.OpenAIProvider(api_key="sk-test", model=model)  # pragma: allowlist secret

    def test_an_exact_row_or_its_snapshot_resolves_to_that_row(self):
        assert op.OpenAIProvider._known_row("gpt-5-mini", op._PRICING) == "gpt-5-mini"
        assert op.OpenAIProvider._known_row("gpt-5-mini-2026-01-15", op._PRICING) == "gpt-5-mini"
        assert op.OpenAIProvider._known_row("gpt-5-mini-20260115", op._PRICING) == "gpt-5-mini"
        assert op.OpenAIProvider._known_row("gpt-5-mini-latest", op._PRICING) == "gpt-5-mini"

    def test_a_family_prefix_or_a_typo_is_not_a_known_row(self):
        assert op.OpenAIProvider._known_row("gpt-5-typo", op._PRICING) is None
        assert op.OpenAIProvider._known_row("gpt-5-mini-x", op._PRICING) is None

    def test_a_known_row_is_priced_with_its_rates_and_no_warning(self, price_log):
        priced = self._provider()._resolve_pricing("gpt-5.5")
        assert (priced.input, priced.output) == (5.0, 30.0)
        assert price_log.calls == []

    def test_a_dated_snapshot_of_a_known_row_is_priced_silently(self, price_log):
        priced = self._provider()._resolve_pricing("gpt-5.5-2026-02-01")
        assert (priced.input, priced.output) == (5.0, 30.0)
        assert price_log.calls == []

    def test_a_family_match_is_priced_as_its_family_and_warned_once(self, price_log):
        provider = self._provider()
        priced = provider._resolve_pricing("gpt-5.5-typo")
        assert (priced.input, priced.output) == (5.0, 30.0)
        provider._resolve_pricing("gpt-5.5-typo")
        assert price_log.calls == [("OpenAI pricing for %r is unknown; pricing it at %s rates. Cost estimates may be off.", ("gpt-5.5-typo", "its family's"))]

    def test_a_model_with_no_known_family_is_priced_at_gpt_five_mini_and_warned(self, price_log):
        priced = self._provider()._resolve_pricing("zzz-unknown")
        assert (priced.input, priced.output) == (0.25, 2.0)
        assert price_log.calls[0][1] == ("zzz-unknown", "gpt-5-mini")

    def test_the_cache_write_rate_is_only_known_for_a_listed_row(self):
        provider = self._provider()
        assert provider._resolve_pricing("gpt-5.5").cache_write == op._CACHE_WRITE_COST.get("gpt-5.5")
        assert provider._resolve_pricing("gpt-5.5-typo").cache_write is None

    def test_the_cache_hit_rate_is_the_row_value_or_the_family_or_gpt_five_mini(self, price_log):
        provider = self._provider()
        assert provider._cache_hit_cost_per_1m("gpt-5.5") == op._CACHE_HIT_COST["gpt-5.5"]
        assert provider._cache_hit_cost_per_1m("gpt-5.5-typo") == op._CACHE_HIT_COST["gpt-5.5"]
        assert provider._cache_hit_cost_per_1m("zzz-unknown") == op._CACHE_HIT_COST["gpt-5-mini"]

    def test_the_long_context_tier_applies_to_listed_models_and_their_snapshots_only(self):
        provider = self._provider()
        assert provider._long_context_tier("gpt-5.5") is op._LONG_CONTEXT_TIER
        assert provider._long_context_tier("gpt-5.5-2026-02-01") is op._LONG_CONTEXT_TIER
        assert provider._long_context_tier("gpt-5-mini") is None


class TestRejectedParametersAndReasoningField:
    """A 400 over temperature or reasoning_effort drops that field (temperature is remembered per model), thinking becomes reasoning_effort for reasoning models, and a non-reasoning model is warned and sent nothing (openai_provider 343-377)."""

    def _provider(self, model="gpt-5.5"):
        return op.OpenAIProvider(api_key="sk-test", model=model)  # pragma: allowlist secret

    def test_a_non_400_status_repairs_nothing(self):
        assert self._provider()._body_after_rejected_request({"temperature": 0.5}, 422, "temperature") is None

    def test_a_400_over_temperature_drops_it_and_remembers_the_model(self):
        op._MODELS_REJECTING_TEMPERATURE.discard("gpt-5.5")
        out = self._provider()._body_after_rejected_request({"temperature": 0.5, "model": "gpt-5.5"}, 400, "Unsupported temperature")
        assert out == {"model": "gpt-5.5"}
        assert "gpt-5.5" in op._MODELS_REJECTING_TEMPERATURE
        op._MODELS_REJECTING_TEMPERATURE.discard("gpt-5.5")

    def test_a_400_that_names_temperature_but_the_body_has_none_repairs_nothing(self):
        assert self._provider()._body_after_rejected_request({"model": "gpt-5.5"}, 400, "temperature is wrong") is None

    def test_a_400_over_reasoning_effort_drops_it(self):
        out = self._provider()._body_after_rejected_request({"reasoning_effort": "minimal", "model": "gpt-5.5"}, 400, "reasoning_effort not supported")
        assert out == {"model": "gpt-5.5"}

    def test_an_unrelated_400_repairs_nothing(self):
        assert self._provider()._body_after_rejected_request({"model": "gpt-5.5"}, 400, "bad model") is None

    def test_thinking_on_a_reasoning_model_is_medium_unless_an_effort_is_named(self):
        provider = self._provider("gpt-5.5")
        assert provider._thinking_request_field(True) == {"reasoning_effort": "medium"}
        assert provider._thinking_request_field("high") == {"reasoning_effort": "high"}

    def test_thinking_off_on_a_reasoning_model_asks_for_the_family_lowest_effort(self):
        provider = self._provider("gpt-5.5")
        expected = str(op.longest_prefix_lookup("gpt-5.5", op._LOWEST_EFFORT, "low"))
        assert provider._thinking_request_field(False) == {"reasoning_effort": expected}

    def test_thinking_on_a_non_reasoning_model_is_warned_and_not_sent(self, monkeypatch):
        log = _PriceWarnLog()
        monkeypatch.setattr(op, "logger", log)
        assert self._provider("gpt-4.1")._thinking_request_field(True) is None
        assert log.calls == [("OpenAI %r is not a reasoning model; thinking=%r is not sent.", ("gpt-4.1", True))]

    def test_thinking_off_on_a_non_reasoning_model_is_silent(self, monkeypatch):
        log = _PriceWarnLog()
        monkeypatch.setattr(op, "logger", log)
        assert self._provider("gpt-4.1")._thinking_request_field(False) is None
        assert log.calls == []


class TestOutputAndContextLimits:
    """The output cap and context window of a model: its row, a dated snapshot of it, a family prefix, or the default for an unlisted model (openai_provider 457-463)."""

    def _provider(self, model):
        return op.OpenAIProvider(api_key="sk-test", model=model)  # pragma: allowlist secret

    def test_the_output_cap_is_the_row_value_for_a_listed_model_and_its_snapshot(self):
        assert self._provider("gpt-5.5").max_output_tokens == op._MAX_TOKENS["gpt-5.5"]
        assert self._provider("gpt-5.5-2026-02-01").max_output_tokens == op._MAX_TOKENS["gpt-5.5"]

    def test_the_output_cap_of_a_family_prefix_is_the_longest_matching_row(self):
        assert self._provider("gpt-5.5-typo").max_output_tokens == op._MAX_TOKENS["gpt-5.5"]

    def test_an_unlisted_model_has_no_output_cap(self):
        assert self._provider("zzz-unknown").max_output_tokens is None

    def test_the_context_window_is_the_row_value_for_a_listed_model_and_its_snapshot(self):
        assert self._provider("gpt-5.5").context_window == op._CONTEXT_WINDOW["gpt-5.5"]
        assert self._provider("gpt-5.5-2026-02-01").context_window == op._CONTEXT_WINDOW["gpt-5.5"]

    def test_an_unlisted_model_gets_the_class_default_context_window(self):
        assert self._provider("zzz-unknown").context_window == op.OpenAIProvider._default_context_window == 128_000


class TestOpenAIBodyShaping:
    """The chat body renames max_tokens, drops temperature for reasoning models and for models that refused it, keeps it otherwise; the streaming builder passes its keywords through; an unlisted family's reasoning effort is low (openai_provider 343-377)."""

    def _provider(self, model):
        return op.OpenAIProvider(api_key="sk-test", model=model)  # pragma: allowlist secret

    def test_max_tokens_is_renamed_to_max_completion_tokens(self):
        out = self._provider("gpt-4.1")._openai_body({"max_tokens": 50, "model": "gpt-4.1"})
        assert out == {"max_completion_tokens": 50, "model": "gpt-4.1"}

    def test_temperature_is_kept_for_a_non_reasoning_model_that_accepts_it(self):
        op._MODELS_REJECTING_TEMPERATURE.discard("gpt-4.1")
        assert self._provider("gpt-4.1")._openai_body({"temperature": 0.7})["temperature"] == 0.7

    def test_temperature_is_dropped_for_a_reasoning_model(self):
        assert "temperature" not in self._provider("gpt-5.5")._openai_body({"temperature": 0.7})

    def test_temperature_is_dropped_for_a_model_that_refused_it_before(self):
        op._MODELS_REJECTING_TEMPERATURE.add("gpt-4.1")
        try:
            assert "temperature" not in self._provider("gpt-4.1")._openai_body({"temperature": 0.7})
        finally:
            op._MODELS_REJECTING_TEMPERATURE.discard("gpt-4.1")

    def test_the_streaming_builder_passes_its_keywords_to_the_shared_builder(self):
        out = self._provider("gpt-4.1")._build_stream_body(
            prompt="hi", system=None, temperature=0.7, max_tokens=50, json_mode=False, thinking=None, json_schema=None, images=None
        )
        assert out["max_completion_tokens"] == 50
        assert "max_tokens" not in out

    def test_a_family_without_a_lowest_effort_row_asks_for_low(self, monkeypatch):
        monkeypatch.setattr(op, "_LOWEST_EFFORT", {})
        assert self._provider("gpt-5.5")._thinking_request_field(False) == {"reasoning_effort": "low"}


class TestCreditHintAndUnknownModelThrottle:
    """The credit query refuses with the exact billing hint, and an unknown model's warning is throttled per model under its own key (openai_provider 388, 393)."""

    def test_the_credit_query_refuses_with_the_exact_billing_hint(self):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        with pytest.raises(NotImplementedError) as info:
            asyncio.run(provider.get_account_credits())
        assert str(info.value) == (
            "OpenAI has no public API to fetch remaining credit. Check platform.openai.com/usage or platform.openai.com/account/billing/overview."
        )

    def test_an_unknown_model_warning_is_throttled_under_its_own_key(self, price_log):
        from pyutilz.dev import logginglib as ll

        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        provider._warn_unknown_model_once("zzz-keyed")
        assert "openai.unknown_model:zzz-keyed" in ll._log_throttle_last
        provider._warn_unknown_model_once("zzz-keyed")
        assert len(price_log.calls) == 1
        assert price_log.calls[0][1] == ("zzz-keyed", "gpt-5-mini")


class TestUnknownModelWarningPerPath:
    """The pricing path and the cache-hit path each warn about an unknown model on their own (openai_provider 429, 471)."""

    def test_the_pricing_path_warns_by_itself_when_the_cache_rate_is_stubbed(self, price_log, monkeypatch):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        monkeypatch.setattr(provider, "_cache_hit_cost_per_1m", lambda model: 0.1)
        provider._resolve_pricing("gpt-5.5-typo")
        assert price_log.calls == [("OpenAI pricing for %r is unknown; pricing it at %s rates. Cost estimates may be off.", ("gpt-5.5-typo", "its family's"))]

    def test_the_cache_hit_path_warns_by_itself_for_an_unknown_model(self, price_log):
        provider = op.OpenAIProvider(api_key="sk-test")  # pragma: allowlist secret
        provider._cache_hit_cost_per_1m("gpt-5.5-typo")
        assert price_log.calls == [("OpenAI pricing for %r is unknown; pricing it at %s rates. Cost estimates may be off.", ("gpt-5.5-typo", "its family's"))]
