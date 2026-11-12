"""Behaviour of ``pyutilz.llm.openai_provider``: the reasoning-model family test, the per-family timeouts, the request
body reshaped for OpenAI (``max_tokens`` renamed, ``temperature`` dropped for reasoning models), the reasoning-effort
mapping for ``thinking=``, the learned repair of a rejected ``temperature``, and the refusal to report credits.
No network is used: the provider is built with a test key and only its pure helpers are exercised."""

from __future__ import annotations

import asyncio

import pytest

from pyutilz.llm import openai_provider as op

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
