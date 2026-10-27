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
