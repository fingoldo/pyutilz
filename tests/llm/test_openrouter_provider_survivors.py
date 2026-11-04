"""Default-value pins for OpenRouterProvider (mutation survivors)."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from pyutilz.llm.openrouter_provider import OpenRouterProvider


def test_the_default_context_window_is_one_hundred_twenty_eight_thousand():
    assert OpenRouterProvider._default_context_window == 128_000


def test_a_model_missing_from_the_catalogue_reports_the_literal_default_window(monkeypatch):
    monkeypatch.setattr(OpenRouterProvider, "_smallest_known_limit", lambda self, index: None)
    p = OpenRouterProvider(api_key="k")
    assert p.context_window == 128_000


@pytest.fixture
def warmers(monkeypatch):
    """Record both warm-ups _async_prepare drives; the endpoint one is imported at call time, so it is patched on its module."""
    from pyutilz.llm.openrouter_provider import _endpoint_prices
    from pyutilz.llm.openrouter_provider import _provider as prov

    seen = {"catalogue": 0, "endpoint": []}

    async def fake_catalogue():
        seen["catalogue"] += 1

    def fake_endpoint(model_id):
        seen["endpoint"].append(model_id)

    monkeypatch.setattr(prov, "_ensure_catalogue_warm_async", fake_catalogue)
    monkeypatch.setattr(_endpoint_prices, "warm_endpoint_cache", fake_endpoint)
    return seen


def test_async_prepare_warms_the_catalogue_then_the_endpoint_cache_for_this_model(warmers):
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    p = OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini")
    asyncio.run(p._async_prepare())
    assert warmers["catalogue"] == 1
    assert warmers["endpoint"] == [catalogue_id("openai/gpt-4o-mini")]


def test_a_failing_endpoint_warm_up_is_swallowed_and_logged_at_debug(monkeypatch, warmers, caplog):
    from pyutilz.llm.openrouter_provider import _endpoint_prices

    def boom(model_id):
        raise RuntimeError("cache down")

    monkeypatch.setattr(_endpoint_prices, "warm_endpoint_cache", boom)
    caplog.set_level("DEBUG")
    asyncio.run(OpenRouterProvider(api_key="k")._warm_endpoint_cache_async())
    assert "endpoint cache warm-up skipped for openai/gpt-4o-mini: cache down" in caplog.text


@pytest.fixture
def unknown_ceiling(monkeypatch):
    """A model with no catalogue ceiling, and a fresh once-per-model warned set so each test starts clean."""
    from pyutilz.llm.openrouter_provider import _provider as prov

    monkeypatch.setattr(OpenRouterProvider, "_smallest_known_limit", lambda self, index: None)
    monkeypatch.setattr(OpenRouterProvider, "_warned_default_ceiling", set())
    return prov


def test_a_cold_catalogue_gives_no_ceiling_and_says_the_warm_up_has_not_run(unknown_ceiling, monkeypatch, caplog):
    monkeypatch.setattr(unknown_ceiling, "_catalogue_is_loaded", lambda: False)
    caplog.set_level("WARNING")
    assert OpenRouterProvider(api_key="k").max_output_tokens is None
    assert caplog.records[-1].getMessage() == (
        "OpenRouter catalogue has not been read yet (asked from an event loop before the warm-up ran), "
        "so no output ceiling is known for 'openai/gpt-4o-mini': max_tokens is omitted and the provider decides."
    )


def test_a_loaded_catalogue_without_a_ceiling_says_the_entry_lists_none(unknown_ceiling, monkeypatch, caplog):
    monkeypatch.setattr(unknown_ceiling, "_catalogue_is_loaded", lambda: True)
    caplog.set_level("WARNING")
    assert OpenRouterProvider(api_key="k").max_output_tokens is None
    assert "OpenRouter catalogue lists no max_completion_tokens for this model, so no output ceiling is known" in caplog.records[-1].getMessage()


def test_the_missing_ceiling_warning_is_emitted_once_per_model(unknown_ceiling, monkeypatch, caplog):
    monkeypatch.setattr(unknown_ceiling, "_catalogue_is_loaded", lambda: True)
    caplog.set_level("WARNING")
    p = OpenRouterProvider(api_key="k")
    p.max_output_tokens
    p.max_output_tokens
    assert sum("no output ceiling is known" in r.getMessage() for r in caplog.records) == 1


@pytest.fixture
def catalogue(monkeypatch):
    """Serve a chosen /models catalogue to supports_json_mode; the returned dict is filled by each test."""
    from pyutilz.llm.openrouter_provider import _provider as prov

    entries: dict = {}
    monkeypatch.setattr(prov, "_fetch_models_catalogue", lambda: entries)
    return entries


def test_a_catalogue_fetch_failure_assumes_json_mode_and_logs_at_debug(monkeypatch, caplog):
    from pyutilz.llm.openrouter_provider import _provider as prov

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(prov, "_fetch_models_catalogue", boom)
    caplog.set_level("DEBUG")
    assert OpenRouterProvider(api_key="k").supports_json_mode() is True
    assert "supports_json_mode: unexpected catalogue-fetch error (boom), assuming supported" in caplog.text


def test_a_model_missing_from_the_catalogue_is_assumed_to_support_json_mode(catalogue):
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_mode() is True


def test_a_listed_model_without_a_json_parameter_does_not_support_json_mode(catalogue):
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    catalogue[catalogue_id("openai/gpt-4o-mini")] = {"supported_parameters": ["temperature"]}
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_mode() is False


@pytest.mark.parametrize("parameter", ["response_format", "structured_outputs"])
def test_either_json_parameter_name_marks_json_mode_as_supported(catalogue, parameter):
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    catalogue[catalogue_id("openai/gpt-4o-mini")] = {"supported_parameters": [parameter]}
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_mode() is True


def test_a_non_list_supported_parameters_field_means_json_mode_is_assumed(catalogue):
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    catalogue[catalogue_id("openai/gpt-4o-mini")] = {"supported_parameters": "temperature"}
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_mode() is True


def test_a_strict_schema_catalogue_failure_is_refused_and_warned(monkeypatch, caplog):
    from pyutilz.llm.openrouter_provider import _provider as prov

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(prov, "_fetch_models_catalogue", boom)
    caplog.set_level("WARNING")
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_schema() is False
    assert caplog.records[-1].getMessage() == "OR catalogue unavailable (boom); treating openai/gpt-4o-mini as NOT supporting strict json_schema"


def test_a_strict_schema_check_refuses_a_model_missing_from_the_catalogue(catalogue, caplog):
    caplog.set_level("WARNING")
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_schema() is False
    assert caplog.records[-1].getMessage() == "openai/gpt-4o-mini absent from the OR catalogue; treating it as NOT supporting strict json_schema"


def test_strict_schema_needs_structured_outputs_not_merely_response_format(catalogue):
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    catalogue[catalogue_id("openai/gpt-4o-mini")] = {"supported_parameters": ["response_format"]}
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_schema() is False
    catalogue[catalogue_id("openai/gpt-4o-mini")] = {"supported_parameters": ["structured_outputs"]}
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_schema() is True


def test_a_non_list_supported_parameters_field_refuses_strict_schema(catalogue):
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    catalogue[catalogue_id("openai/gpt-4o-mini")] = {"supported_parameters": "structured_outputs"}
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").supports_json_schema() is False


@pytest.mark.parametrize(
    "model",
    [
        "openai/o1-preview",
        "openai/o3-mini",
        "openai/o4-mini",
        "openai/gpt-5-pro",
        "anthropic/claude-opus-4",
        "deepseek/deepseek-reasoner",
        "deepseek/deepseek-v4-pro",
        "vendor/reasoning-model",
    ],
)
def test_each_slow_reasoning_class_gets_the_long_timeout(model):
    assert OpenRouterProvider(api_key="k")._get_timeout(model) == 1200.0


def test_an_ordinary_model_gets_the_standard_timeout():
    assert OpenRouterProvider(api_key="k")._get_timeout("openai/gpt-4o-mini") == 240.0


def test_a_one_token_reasoning_budget_is_sent_as_max_tokens():
    assert OpenRouterProvider(api_key="k")._thinking_request_field(1) == {"reasoning": {"max_tokens": 1}}


@pytest.fixture
def mandatory_refusals(monkeypatch):
    from pyutilz.llm.openrouter_provider import _provider as prov

    monkeypatch.setattr(prov, "_REASONING_CANNOT_BE_DISABLED", set())


MANDATORY_DETAIL = "Reasoning is mandatory for this endpoint and cannot be disabled"


@pytest.mark.parametrize("status", [400, 404, 422])
def test_a_refused_reasoning_switch_is_repaired_for_the_parameter_refusal_statuses(mandatory_refusals, status):
    body = {"model": "m", "reasoning": {"enabled": False}}
    repaired = OpenRouterProvider(api_key="k", model="m")._body_after_rejected_request(body, status, MANDATORY_DETAIL)
    assert repaired == {"model": "m", "reasoning": {"effort": "minimal", "exclude": True}}


@pytest.mark.parametrize("detail", ["Reasoning is mandatory", "Reasoning cannot be disabled for this model"])
def test_either_mandatory_wording_with_reasoning_named_is_repaired(mandatory_refusals, detail):
    from pyutilz.llm.openrouter_provider import _provider as prov

    body = {"model": "m", "reasoning": {"enabled": False}}
    assert OpenRouterProvider(api_key="k", model="m")._body_after_rejected_request(body, 400, detail) is not None
    assert "m" in prov._REASONING_CANNOT_BE_DISABLED


@pytest.mark.parametrize("detail", ["Reasoning is off", "This setting is mandatory"])
def test_a_refusal_that_lacks_the_reasoning_or_the_mandatory_wording_is_not_repaired(mandatory_refusals, detail):
    body = {"model": "m", "reasoning": {"enabled": False}}
    assert OpenRouterProvider(api_key="k", model="m")._body_after_rejected_request(body, 400, detail) is None


@pytest.mark.parametrize("status", [403, 405, 423, 429, 500])
def test_other_statuses_are_not_repaired(mandatory_refusals, status):
    body = {"model": "m", "reasoning": {"enabled": False}}
    assert OpenRouterProvider(api_key="k", model="m")._body_after_rejected_request(body, status, MANDATORY_DETAIL) is None


def test_a_402_response_warns_about_the_empty_credit_balance(caplog):
    import httpx

    caplog.set_level("WARNING")
    OpenRouterProvider(api_key="k")._handle_special_status(httpx.Response(402))
    assert caplog.records[-1].getMessage() == (
        "OpenRouter account out of credits (HTTP 402). Top up at https://openrouter.ai/credits - "
        "retrying for a short grace window in case the balance is being topped up, "
        "then giving up (PYUTILZ_LLM_BILLING_GRACE_SECONDS)..."
    )


@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_other_statuses_do_not_trigger_the_credit_warning(caplog, status):
    import httpx

    caplog.set_level("WARNING")
    OpenRouterProvider(api_key="k")._handle_special_status(httpx.Response(status))
    assert "out of credits" not in caplog.text


def test_a_usage_block_with_a_numeric_cost_marks_the_cost_as_reported():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({"cost": 0.5})
    assert p.last_actual_cost_reported is True


def test_upstream_inference_cost_accumulates_across_posts_in_the_session_total():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({"cost_details": {"upstream_inference_cost": 0.2}})
    p._track_provider_specific_usage({"cost_details": {"upstream_inference_cost": 0.3}})
    assert p.total_upstream_inference_cost_usd == pytest.approx(0.5)


def test_a_usage_block_without_cache_fields_records_no_cache_hits():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({})
    assert p.last_cache_hit_tokens == 0


def test_a_null_cached_token_count_records_no_cache_hits():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({"prompt_tokens_details": {"cached_tokens": None}})
    assert p.last_cache_hit_tokens == 0


def test_the_top_level_deepseek_hit_count_is_the_fallback_when_the_openai_field_is_zero():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({"prompt_tokens_details": {"cached_tokens": 0}, "prompt_cache_hit_tokens": 5})
    assert p.last_cache_hit_tokens == 5


def test_the_openai_hit_count_wins_even_when_it_is_one_token():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({"prompt_tokens_details": {"cached_tokens": 1}, "prompt_cache_hit_tokens": 5})
    assert p.last_cache_hit_tokens == 1


def test_audio_tokens_accumulate_across_posts_into_the_session_total():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({"prompt_tokens_details": {"audio_tokens": 3}})
    p._track_provider_specific_usage({"prompt_tokens_details": {"audio_tokens": 3}})
    assert p.total_audio_tokens == 6


def test_cache_discount_accumulates_across_posts_in_the_session_total():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({"cache_discount": 0.25})
    p._track_provider_specific_usage({"cache_discount": 0.25})
    assert p.total_cache_discount_usd == pytest.approx(0.5)


def test_a_fresh_model_tally_starts_at_the_token_counts_of_its_first_post():
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    p = OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini")
    p._track_provider_specific_usage({"prompt_tokens": 10, "completion_tokens": 4})
    key = p._pricing_model(catalogue_id(p.model_name))
    assert p._usage_by_model[key] == [10, 0, 4, 0, 0]


def test_a_model_tally_sums_prompt_tokens_and_reasoning_tokens_across_posts():
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    p = OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini")
    p._track_provider_specific_usage({"prompt_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 7}})
    p._track_provider_specific_usage({"prompt_tokens": 5})
    key = p._pricing_model(catalogue_id(p.model_name))
    assert p._usage_by_model[key] == [15, 0, 0, 7, 0]


def test_a_model_tally_sums_cache_write_tokens_across_posts():
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    p = OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini")
    post = {"prompt_tokens_details": {"cache_write_tokens": 1}}
    p._track_provider_specific_usage(post)
    p._track_provider_specific_usage(post)
    key = p._pricing_model(catalogue_id(p.model_name))
    assert p._usage_by_model[key][4] == 2


def test_a_model_tally_sums_cache_hits_and_completion_tokens_into_their_own_slots():
    from pyutilz.llm.openrouter_provider._provider import catalogue_id

    p = OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini")
    post = {"prompt_tokens_details": {"cached_tokens": 4}, "completion_tokens": 3}
    p._track_provider_specific_usage(post)
    p._track_provider_specific_usage(post)
    key = p._pricing_model(catalogue_id(p.model_name))
    assert p._usage_by_model[key] == [0, 8, 6, 0, 0]


def test_a_string_generation_id_is_recorded_and_a_non_string_one_is_ignored():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_response({"id": "gen-1"})
    assert p.last_generation_id == "gen-1"
    p._track_provider_specific_response({"id": 123})
    assert p.last_generation_id == "gen-1"


def test_the_serving_provider_is_recorded_only_when_a_string():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_response({"provider": "Anthropic"})
    assert p.last_upstream_provider == "Anthropic"
    p._track_provider_specific_response({"provider": 7})
    assert p.last_upstream_provider == "Anthropic"


def test_the_resolved_model_and_native_finish_reason_are_recorded_from_the_response():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_response({"model": "x/y", "choices": [{"native_finish_reason": "stop"}]})
    assert p.last_upstream_model == "x/y"
    assert p.last_native_finish_reason == "stop"


def test_web_search_citations_are_read_from_the_message_and_kept_when_a_later_reply_has_none():
    p = OpenRouterProvider(api_key="k")
    annotated = {"choices": [{"message": {"annotations": [{"type": "url_citation", "url_citation": {"url": "https://a"}}]}}]}
    p._track_provider_specific_response(annotated)
    assert p.last_web_search_citations == [{"url": "https://a"}]
    p._track_provider_specific_response({"choices": [{"message": {}}]})
    assert p.last_web_search_citations == [{"url": "https://a"}]


def test_the_byok_flag_is_recorded_only_when_it_is_a_real_bool():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_response({"is_byok": True})
    assert p.last_is_byok is True
    p._track_provider_specific_response({"is_byok": "yes"})
    assert p.last_is_byok is True


def test_the_response_cache_source_is_recorded_only_when_a_string():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_response({"response_cache_source_id": "src-1"})
    assert p.last_response_cache_source_id == "src-1"
    p._track_provider_specific_response({"response_cache_source_id": 5})
    assert p.last_response_cache_source_id == "src-1"


def test_a_pinned_single_provider_with_fallbacks_off_is_the_endpoint_tag():
    p = OpenRouterProvider(api_key="k", provider_only=("Anthropic",), provider_allow_fallbacks=False)
    assert p.last_endpoint_tag == "Anthropic"


def test_two_pinned_providers_with_fallbacks_off_have_no_single_endpoint_tag():
    assert OpenRouterProvider(api_key="k", provider_only=("Anthropic", "DeepInfra"), provider_allow_fallbacks=False).last_endpoint_tag is None


def test_an_unpinned_or_fallback_enabled_call_has_no_endpoint_tag():
    assert OpenRouterProvider(api_key="k", provider_only=("Anthropic",), provider_allow_fallbacks=True).last_endpoint_tag is None
    assert OpenRouterProvider(api_key="k").last_endpoint_tag is None


def test_resolve_pricing_maps_each_catalogue_rate_to_its_own_field(monkeypatch):
    from pyutilz.llm.openrouter_provider import _provider as prov

    monkeypatch.setattr(prov, "_per_token_cost_pair", lambda m: (1.0, 2.0))
    monkeypatch.setattr(prov, "_cache_read_cost_per_1m_or_none", lambda m: 0.5)
    monkeypatch.setattr(prov, "_cache_write_cost_per_1m_or_none", lambda m: 0.7)
    pricing = OpenRouterProvider(api_key="k")._resolve_pricing("openai/gpt-4o-mini")
    assert (pricing.input, pricing.output, pricing.cache_hit, pricing.cache_write) == (1.0, 2.0, 0.5, 0.7)


def test_an_unavailable_estimate_reports_every_cost_field_as_none():
    p = OpenRouterProvider(api_key="k")
    p._estimate_by_served_model = lambda: None
    out = p.get_session_cost()
    assert out["input_cost_usd"] is None
    assert out["output_cost_usd"] is None
    assert out["total_cost_usd"] is None
    assert out["pricing_available"] is False


def test_a_known_estimate_reports_its_input_output_and_summed_total_as_available():
    p = OpenRouterProvider(api_key="k")
    p._estimate_by_served_model = lambda: (1.0, 2.0)
    out = p.get_session_cost()
    assert out["input_cost_usd"] == 1.0
    assert out["output_cost_usd"] == 2.0
    assert out["total_cost_usd"] == pytest.approx(3.0)
    assert out["pricing_available"] is True


def test_an_unpinned_call_asks_for_its_quantization_with_empty_tag_and_provider(monkeypatch):
    from pyutilz.llm.openrouter_provider import _endpoint_prices

    seen = {}

    def fake(model, *, tag="", provider_name=""):
        seen.update(tag=tag, provider_name=provider_name)
        return "fp8"

    monkeypatch.setattr(_endpoint_prices, "claimed_quantization", fake)
    assert OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini").last_quantization == "fp8"
    assert seen == {"tag": "", "provider_name": ""}


def test_a_pinned_call_asks_for_its_quantization_with_the_pinned_tag_and_serving_provider(monkeypatch):
    from pyutilz.llm.openrouter_provider import _endpoint_prices

    seen = {}

    def fake(model, *, tag="", provider_name=""):
        seen.update(tag=tag, provider_name=provider_name)
        return "fp8"

    monkeypatch.setattr(_endpoint_prices, "claimed_quantization", fake)
    p = OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini", provider_only=("deepinfra",), provider_allow_fallbacks=False)
    p._track_provider_specific_response({"provider": "DeepInfra"})
    assert p.last_quantization == "fp8"
    assert seen == {"tag": "deepinfra", "provider_name": "DeepInfra"}


def test_the_call_summary_reports_the_served_identity_under_its_own_keys():
    p = OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini")
    p._track_provider_specific_response({"id": "gen-1", "provider": "Anthropic", "model": "x/y"})
    p._last_finish_reason = "stop"
    p._track_provider_specific_response({"choices": [{"native_finish_reason": "end_turn"}]})
    p._track_provider_specific_usage({"cost": 0.25, "cost_details": {"upstream_inference_cost": 0.2}, "cache_discount": 0.1,
                                      "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 1}})
    summary = p.last_call_summary()
    assert summary["generation_id"] == "gen-1"
    assert summary["upstream_provider"] == "Anthropic"
    assert summary["upstream_model"] == "x/y"
    assert summary["requested_model"] == "openai/gpt-4o-mini"
    assert summary["finish_reason"] == "stop"
    assert summary["native_finish_reason"] == "end_turn"
    assert summary["cost_usd"] == 0.25
    assert summary["upstream_inference_cost_usd"] == 0.2
    assert summary["cache_discount_usd"] == 0.1
    assert summary["cache_hit_tokens"] == 4
    assert summary["cache_write_tokens"] == 1
    p._last_usage = {"input_tokens": 12, "output_tokens": 3, "reasoning_tokens": 2}
    p._track_provider_specific_response({"is_byok": True, "response_cache_source_id": "src-9"})
    summary = p.last_call_summary()
    assert summary["is_byok"] is True
    assert summary["response_cache_source_id"] == "src-9"
    assert summary["input_tokens"] == 12
    assert summary["output_tokens"] == 3
    assert summary["reasoning_tokens"] == 2


def test_the_session_cost_reports_upstream_and_audio_totals_under_their_keys():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_usage({"cost_details": {"upstream_inference_cost": 0.4}, "prompt_tokens_details": {"audio_tokens": 2}})
    out = p.get_session_cost()
    assert out["upstream_inference_cost_usd"] == pytest.approx(0.4)
    assert out["last_upstream_inference_cost_usd"] == pytest.approx(0.4)
    assert out["audio_tokens"] == 2


def test_the_call_summary_reports_the_web_search_citations_under_their_key():
    p = OpenRouterProvider(api_key="k")
    p._track_provider_specific_response({"choices": [{"message": {"annotations": [{"type": "url_citation", "url_citation": {"url": "https://a"}}]}}]})
    assert p.last_call_summary()["web_search_citations"] == [{"url": "https://a"}]


def test_a_call_summary_with_no_usage_reports_zero_token_counts():
    summary = OpenRouterProvider(api_key="k").last_call_summary()
    assert (summary["input_tokens"], summary["output_tokens"], summary["reasoning_tokens"], summary["audio_tokens"]) == (0, 0, 0, 0)


def test_a_fresh_provider_has_not_reported_a_cost():
    p = OpenRouterProvider(api_key="k")
    assert p.last_actual_cost_reported is False


@pytest.mark.parametrize("name", ["last_cache_write_tokens", "last_cache_hit_tokens", "last_audio_tokens"])
def test_the_token_counters_are_carried_into_the_per_request_metadata(name):
    assert name in OpenRouterProvider._PERCALL_METADATA_ATTRS


def test_the_per_request_snapshot_carries_the_token_counters():
    p = OpenRouterProvider(api_key="k")
    p.last_cache_write_tokens = 9
    p.last_cache_hit_tokens = 4
    p.last_audio_tokens = 2
    snapshot = p._capture_percall_metadata()
    assert snapshot["cache_write_tokens"] == 9
    assert snapshot["cache_hit_tokens"] == 4
    assert snapshot["audio_tokens"] == 2


def test_a_new_call_resets_the_token_counters_to_zero():
    p = OpenRouterProvider(api_key="k")
    p.last_cache_write_tokens = 9
    p.last_cache_hit_tokens = 4
    p.last_audio_tokens = 2
    p._reset_per_call_state()
    assert (p.last_cache_write_tokens, p.last_cache_hit_tokens, p.last_audio_tokens) == (0, 0, 0)


def test_the_default_concurrency_is_ten():
    assert inspect.signature(OpenRouterProvider.__init__).parameters["max_concurrent"].default == 10


def test_routing_404_retry_is_off_by_default():
    assert inspect.signature(OpenRouterProvider.__init__).parameters["retry_routing_404"].default is False


def test_a_fresh_provider_starts_its_audio_and_upstream_totals_at_zero():
    p = OpenRouterProvider(api_key="k")
    assert p.total_audio_tokens == 0
    assert p.total_upstream_inference_cost_usd == 0.0


def test_a_fresh_provider_starts_its_cache_discount_total_at_zero():
    assert OpenRouterProvider(api_key="k").total_cache_discount_usd == 0.0


def test_the_routing_404_attempt_count_keeps_a_valid_value_and_floors_at_one():
    assert OpenRouterProvider(api_key="k", routing_404_max_attempts=5)._routing_404_max_attempts == 5
    assert OpenRouterProvider(api_key="k", routing_404_max_attempts=0)._routing_404_max_attempts == 1


def test_a_negative_routing_404_pause_is_floored_at_zero():
    assert OpenRouterProvider(api_key="k", routing_404_pause_sec=-5.0)._routing_404_pause_sec == 0.0
    assert OpenRouterProvider(api_key="k", routing_404_pause_sec=2.5)._routing_404_pause_sec == 2.5


def test_the_routing_404_retry_budget_is_three_attempts_paused_sixty_seconds():
    params = inspect.signature(OpenRouterProvider.__init__).parameters
    assert params["routing_404_max_attempts"].default == 3
    assert params["routing_404_pause_sec"].default == 60.0
