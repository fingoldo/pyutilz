"""Tests for OpenRouter provider — meta-provider semantics + OR-specific bits."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from pyutilz.llm.openrouter_provider import (
    OpenRouterProvider,
    list_openrouter_models,
    _fetch_models_catalogue,
    _per_token_cost_pair,
    _resolve_model_limits,
    _summarize_endpoints,
    _normalize_uptime,
    _resolve_or_api_key,
)
import pyutilz.llm.openrouter_provider as openrouter_module


def _provider(**overrides) -> OpenRouterProvider:
    """Build a provider without hitting settings — bypasses key resolution."""
    p = OpenRouterProvider.__new__(OpenRouterProvider)
    # Run base __init__ logic via a manual setup
    import asyncio
    p.api_key = "test-key"
    p.model_name = overrides.get("model", "openai/gpt-4o-mini")
    p.semaphore = asyncio.Semaphore(10)
    p._client = httpx.AsyncClient(
        base_url=OpenRouterProvider._base_url,
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
    )
    p.total_prompt_tokens = 0
    p.total_completion_tokens = 0
    p.total_cache_hit_tokens = 0
    p.total_reasoning_tokens = 0
    p._call_count = 0
    p._last_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    p._provider_order = overrides.get("provider_order")
    p._provider_ignore = overrides.get("provider_ignore")
    p._provider_sort = overrides.get("provider_sort")
    p._provider_allow_fallbacks = overrides.get("provider_allow_fallbacks", True)
    p._models_fallback = overrides.get("models_fallback")
    p.total_actual_cost_usd = 0.0
    p.last_actual_cost_usd = 0.0
    p.total_cache_write_tokens = 0
    p.last_cache_write_tokens = 0
    p.last_cache_hit_tokens = 0
    p.total_audio_tokens = 0
    p.last_audio_tokens = 0
    p.total_upstream_inference_cost_usd = 0.0
    p.last_upstream_inference_cost_usd = None
    p.last_generation_id = None
    p.last_upstream_provider = None
    p.last_upstream_model = None
    p.last_native_finish_reason = None
    return p


@pytest.fixture(autouse=True)
def _reset_models_cache():
    """Each test starts with fresh caches so prior fixtures don't leak."""
    openrouter_module._MODELS_CATALOGUE = None
    openrouter_module._HEALTH_CACHE.clear()
    yield
    openrouter_module._MODELS_CATALOGUE = None
    openrouter_module._HEALTH_CACHE.clear()


class TestConstructor:
    def test_provider_name(self):
        assert OpenRouterProvider._provider_name == "OpenRouter"

    def test_base_url(self):
        assert OpenRouterProvider._base_url == "https://openrouter.ai/api/v1"

    def test_default_model_is_widely_available(self):
        from inspect import signature
        default = signature(OpenRouterProvider.__init__).parameters["model"].default
        # Must be a "provider/model"-shaped slug — this is the OR convention.
        assert "/" in default

    def test_missing_api_key_raises(self):
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = None
        with patch(
            "pyutilz.llm.openrouter_provider.get_llm_settings",
            return_value=mock_settings,
        ):
            with pytest.raises(ValueError, match="API key not provided"):
                OpenRouterProvider(api_key=None)

    def test_attribution_headers_set(self):
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key.get_secret_value.return_value = "key"
        with patch(
            "pyutilz.llm.openrouter_provider.get_llm_settings",
            return_value=mock_settings,
        ):
            p = OpenRouterProvider(
                api_key="k",
                app_name="MyApp",
                site_url="https://example.com",
            )
            assert p._client.headers["X-Title"] == "MyApp"
            assert p._client.headers["HTTP-Referer"] == "https://example.com"

    def test_attribution_headers_optional(self):
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key.get_secret_value.return_value = "key"
        with patch(
            "pyutilz.llm.openrouter_provider.get_llm_settings",
            return_value=mock_settings,
        ):
            p = OpenRouterProvider(api_key="k")
            assert "X-Title" not in p._client.headers
            assert "HTTP-Referer" not in p._client.headers


class TestExtraRequestBody:
    def test_no_routing_returns_empty(self):
        p = _provider()
        assert p._extra_request_body("openai/gpt-4o-mini") == {}

    def test_provider_order_emitted(self):
        p = _provider(provider_order=("anthropic", "openai"))
        body = p._extra_request_body("anthropic/claude-sonnet-4.6")
        assert body == {"provider": {"order": ["anthropic", "openai"]}}

    def test_provider_sort_emitted(self):
        p = _provider(provider_sort="throughput")
        body = p._extra_request_body("openai/gpt-4o")
        assert body == {"provider": {"sort": "throughput"}}

    def test_allow_fallbacks_default_omitted(self):
        # Default is True — match OR's default and skip the field for clean payloads.
        p = _provider()
        assert "allow_fallbacks" not in p._extra_request_body("any").get("provider", {})

    def test_allow_fallbacks_false_emitted(self):
        p = _provider(provider_allow_fallbacks=False)
        body = p._extra_request_body("any")
        assert body == {"provider": {"allow_fallbacks": False}}

    def test_models_fallback_emitted(self):
        p = _provider(models_fallback=("openai/gpt-4o", "google/gemini-2.0-flash"))
        body = p._extra_request_body("anthropic/claude-sonnet-4.6")
        assert body == {"models": ["openai/gpt-4o", "google/gemini-2.0-flash"]}

    def test_provider_ignore_emitted(self):
        p = _provider(provider_ignore=("deepinfra",))
        body = p._extra_request_body("any")
        assert body == {"provider": {"ignore": ["deepinfra"]}}

    def test_full_routing_combined(self):
        p = _provider(
            provider_order=("anthropic",),
            provider_sort="price",
            provider_allow_fallbacks=False,
            models_fallback=("openai/gpt-4o",),
        )
        body = p._extra_request_body("anthropic/claude-sonnet-4.6")
        assert body == {
            "provider": {
                "order": ["anthropic"],
                "sort": "price",
                "allow_fallbacks": False,
            },
            "models": ["openai/gpt-4o"],
        }


class TestCostTracking:
    def test_track_usage_records_actual_cost(self):
        p = _provider()
        p._track_provider_specific_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0042}
        )
        assert p.last_actual_cost_usd == pytest.approx(0.0042)
        assert p.total_actual_cost_usd == pytest.approx(0.0042)

    def test_track_usage_accumulates(self):
        p = _provider()
        p._track_provider_specific_usage({"cost": 0.01})
        p._track_provider_specific_usage({"cost": 0.02})
        assert p.total_actual_cost_usd == pytest.approx(0.03)
        assert p.last_actual_cost_usd == pytest.approx(0.02)

    def test_track_usage_missing_cost_no_op(self):
        p = _provider()
        p._track_provider_specific_usage({"prompt_tokens": 10})
        assert p.total_actual_cost_usd == 0.0

    def test_track_usage_cache_write_tokens(self):
        p = _provider()
        p._track_provider_specific_usage(
            {"prompt_tokens_details": {"cache_write_tokens": 100}}
        )
        assert p.total_cache_write_tokens == 100

    def test_track_usage_cached_tokens_fallback(self):
        # OR's modern field is prompt_tokens_details.cached_tokens; the legacy
        # prompt_cache_hit_tokens field should still take precedence if present.
        p = _provider()
        p._track_provider_specific_usage(
            {"prompt_tokens_details": {"cached_tokens": 50}}
        )
        assert p.total_cache_hit_tokens == 50

    def test_get_session_cost_includes_actual(self):
        p = _provider()
        p.total_actual_cost_usd = 1.23
        p.last_actual_cost_usd = 0.5
        p.total_cache_write_tokens = 999
        cost = p.get_session_cost()
        assert cost["actual_cost_usd"] == 1.23
        assert cost["last_actual_cost_usd"] == 0.5
        assert cost["cache_write_tokens"] == 999


class TestPricing:
    def test_per_token_cost_returns_zero_when_unknown(self):
        openrouter_module._MODELS_CATALOGUE = {}
        assert _per_token_cost_pair("nonexistent/model") == (0.0, 0.0)

    def test_per_token_cost_converts_per_token_to_per_1m(self):
        # OR publishes pricing as USD-per-token strings.
        openrouter_module._MODELS_CATALOGUE = {
            "openai/gpt-4o": {
                "id": "openai/gpt-4o",
                "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
            }
        }
        in_p, out_p = _per_token_cost_pair("openai/gpt-4o")
        assert in_p == pytest.approx(2.5)
        assert out_p == pytest.approx(10.0)

    def test_per_token_cost_handles_malformed_pricing(self):
        openrouter_module._MODELS_CATALOGUE = {
            "junk/model": {"id": "junk/model", "pricing": {"prompt": "abc"}}
        }
        assert _per_token_cost_pair("junk/model") == (0.0, 0.0)

    def test_input_cost_per_1m_uses_catalogue(self):
        openrouter_module._MODELS_CATALOGUE = {
            "openai/gpt-4o": {
                "id": "openai/gpt-4o",
                "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
            }
        }
        p = _provider(model="openai/gpt-4o")
        assert p._input_cost_per_1m("openai/gpt-4o") == pytest.approx(2.5)
        assert p._output_cost_per_1m("openai/gpt-4o") == pytest.approx(10.0)


class TestModelsCatalogueFetch:
    def test_fetch_failure_returns_empty_dict(self):
        openrouter_module._MODELS_CATALOGUE = None
        with patch(
            "pyutilz.llm.openrouter_provider.httpx.get",
            side_effect=httpx.ConnectError("nope"),
        ):
            cat = _fetch_models_catalogue()
        assert cat == {}

    def test_fetch_success_indexes_by_id(self):
        openrouter_module._MODELS_CATALOGUE = None
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "data": [
                {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025"}},
                {"id": "anthropic/claude-3.5", "pricing": {"prompt": "0.000003"}},
            ]
        }
        fake_resp.raise_for_status = MagicMock()
        with patch(
            "pyutilz.llm.openrouter_provider.httpx.get",
            return_value=fake_resp,
        ):
            cat = _fetch_models_catalogue()
        assert "openai/gpt-4o" in cat
        assert "anthropic/claude-3.5" in cat
        assert cat["openai/gpt-4o"]["pricing"]["prompt"] == "0.0000025"


class TestListModels:
    def setup_method(self):
        openrouter_module._MODELS_CATALOGUE = {
            "openai/gpt-4o": {
                "id": "openai/gpt-4o",
                "context_length": 128000,
                "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
            },
            "openai/gpt-4o-mini": {
                "id": "openai/gpt-4o-mini",
                "context_length": 128000,
                "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
            },
            "anthropic/claude-sonnet-4.6": {
                "id": "anthropic/claude-sonnet-4.6",
                "context_length": 200000,
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            },
        }

    def teardown_method(self):
        openrouter_module._MODELS_CATALOGUE = None

    def test_list_returns_all_models(self):
        # Stage-1 only — proves the offline catalogue path is intact.
        rows = list_openrouter_models(return_only_healthy=False)
        assert len(rows) == 3

    def test_filter_by_name_substring(self):
        rows = list_openrouter_models(
            name_contains="claude", return_only_healthy=False,
        )
        assert len(rows) == 1
        assert rows[0]["id"] == "anthropic/claude-sonnet-4.6"

    def test_filter_by_name_case_insensitive(self):
        rows = list_openrouter_models(
            name_contains="CLAUDE", return_only_healthy=False,
        )
        assert len(rows) == 1

    def test_filter_by_max_input_price(self):
        # 0.50 USD per 1M tokens — should match only gpt-4o-mini
        rows = list_openrouter_models(
            max_input_per_1m=0.50, return_only_healthy=False,
        )
        assert len(rows) == 1
        assert rows[0]["id"] == "openai/gpt-4o-mini"

    def test_sort_by_input_price_ascending(self):
        rows = list_openrouter_models(
            sort_by="input_price", return_only_healthy=False,
        )
        assert [r["id"] for r in rows] == [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4.6",
        ]

    def test_sort_by_context_descending(self):
        # context sort should put the BIGGEST window first (most useful default)
        rows = list_openrouter_models(
            sort_by="context", return_only_healthy=False,
        )
        assert rows[0]["id"] == "anthropic/claude-sonnet-4.6"

    def test_combined_filter_and_sort(self):
        rows = list_openrouter_models(
            name_contains="openai",
            sort_by="input_price",
            return_only_healthy=False,
        )
        assert [r["id"] for r in rows] == ["openai/gpt-4o-mini", "openai/gpt-4o"]

    def test_default_is_return_only_healthy_true(self):
        # Default has changed to True — document the contract.
        from inspect import signature
        sig = signature(list_openrouter_models)
        assert sig.parameters["return_only_healthy"].default is True

    def test_no_key_falls_back_gracefully(self, monkeypatch, caplog):
        # With return_only_healthy=True but no API key configured, must not
        # crash — should warn and return Stage-1 results unfiltered.
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch(
            "pyutilz.llm.openrouter_provider._resolve_or_api_key",
            return_value=None,
        ):
            with caplog.at_level("WARNING"):
                rows = list_openrouter_models(return_only_healthy=True)
        assert len(rows) == 3  # Stage-1 unfiltered
        assert "API key" in caplog.text


class TestModelLimitsLookup:
    def test_resolve_unknown_returns_none_pair(self):
        openrouter_module._MODELS_CATALOGUE = {}
        assert _resolve_model_limits("nonexistent/model") == (None, None)

    def test_resolve_prefers_top_provider_over_model_level(self):
        # Theoretical 1M context but upstream caps at 200K — upstream wins.
        openrouter_module._MODELS_CATALOGUE = {
            "vendor/big-model": {
                "id": "vendor/big-model",
                "context_length": 1_000_000,
                "top_provider": {
                    "context_length": 200_000,
                    "max_completion_tokens": 32_000,
                },
            }
        }
        assert _resolve_model_limits("vendor/big-model") == (200_000, 32_000)

    def test_resolve_falls_back_to_model_level_context(self):
        # No top_provider info — use the entry's own context_length.
        openrouter_module._MODELS_CATALOGUE = {
            "vendor/m": {"id": "vendor/m", "context_length": 128_000}
        }
        ctx, max_out = _resolve_model_limits("vendor/m")
        assert ctx == 128_000
        assert max_out is None

    def test_resolve_handles_missing_context_length(self):
        openrouter_module._MODELS_CATALOGUE = {
            "vendor/m": {"id": "vendor/m", "top_provider": {}}
        }
        assert _resolve_model_limits("vendor/m") == (None, None)

    def test_context_window_uses_top_provider(self):
        openrouter_module._MODELS_CATALOGUE = {
            "vendor/m": {
                "id": "vendor/m",
                "context_length": 1_000_000,
                "top_provider": {"context_length": 200_000},
            }
        }
        p = _provider(model="vendor/m")
        assert p.context_window == 200_000

    def test_context_window_fallback_when_unknown(self):
        openrouter_module._MODELS_CATALOGUE = {}
        p = _provider(model="vendor/unknown")
        assert p.context_window == OpenRouterProvider._default_context_window

    def test_max_output_tokens_uses_top_provider(self):
        openrouter_module._MODELS_CATALOGUE = {
            "vendor/m": {
                "id": "vendor/m",
                "top_provider": {
                    "context_length": 200_000,
                    "max_completion_tokens": 8_192,
                },
            }
        }
        p = _provider(model="vendor/m")
        assert p.max_output_tokens == 8_192

    def test_max_output_tokens_fallback_when_unknown(self):
        openrouter_module._MODELS_CATALOGUE = {}
        p = _provider(model="vendor/unknown")
        assert p.max_output_tokens == OpenRouterProvider._default_max_tokens


class TestSummarizeEndpoints:
    def test_empty_list_returns_safe_defaults(self):
        out = _summarize_endpoints([])
        assert out["endpoints"] == []
        assert out["best_uptime_30m"] is None
        assert out["best_latency_p50_ms"] is None
        assert out["best_throughput_p50_tps"] is None

    def test_renames_or_field_names(self):
        out = _summarize_endpoints([{
            "provider_name": "Anthropic",
            "status": "operational",
            "uptime_last_5m": 1.0,
            "uptime_last_30m": 0.998,
            "uptime_last_1d": 0.9994,
            "latency_last_30m": {"p50": 234, "p95": 540},
            "throughput_last_30m": {"p50": 89.5},
            "context_length": 200000,
            "max_completion_tokens": 8192,
            "pricing": {"prompt": "0.000003"},
            "supported_parameters": ["temperature", "tools"],
            "quantization": None,
            "supports_implicit_caching": False,
        }])
        e = out["endpoints"][0]
        assert e["provider_name"] == "Anthropic"
        assert e["uptime_30m"] == 0.998
        assert e["latency_p50_ms"] == 234
        assert e["latency_p95_ms"] == 540
        assert e["throughput_p50_tps"] == 89.5
        assert e["context_length"] == 200000
        assert e["supported_parameters"] == ["temperature", "tools"]
        assert e["supports_implicit_caching"] is False

    def test_best_aggregates_across_endpoints(self):
        out = _summarize_endpoints([
            {
                "provider_name": "A",
                "uptime_last_30m": 0.95,
                "latency_last_30m": {"p50": 500},
                "throughput_last_30m": {"p50": 50},
            },
            {
                "provider_name": "B",
                "uptime_last_30m": 0.999,
                "latency_last_30m": {"p50": 200},
                "throughput_last_30m": {"p50": 100},
            },
        ])
        assert out["best_uptime_30m"] == 0.999          # max
        assert out["best_latency_p50_ms"] == 200        # min
        assert out["best_throughput_p50_tps"] == 100    # max

    def test_p90_used_when_p95_missing(self):
        out = _summarize_endpoints([{
            "provider_name": "A",
            "latency_last_30m": {"p50": 100, "p90": 400},
        }])
        assert out["endpoints"][0]["latency_p95_ms"] == 400

    def test_percentage_uptime_normalised_to_fraction(self):
        """OpenRouter's live API has been observed serving uptime as
        a 0-100 percentage (e.g. 99.806 for 99.806% uptime) instead
        of the documented 0-1 fraction. ``_summarize_endpoints``
        normalises both shapes to fraction so downstream filters
        using a 0-1 ``min_uptime`` work consistently."""
        out = _summarize_endpoints([
            {
                "provider_name": "A",
                "uptime_last_5m": 99.5,    # percentage
                "uptime_last_30m": 99.806,  # percentage
                "uptime_last_1d": 100.0,   # 100% as a percentage
            },
            {
                "provider_name": "B",
                "uptime_last_30m": 0.95,   # already a fraction
            },
        ])
        # All three endpoint A values must end up in 0-1 range.
        eps = out["endpoints"]
        a, b = eps[0], eps[1]
        assert a["uptime_5m"] == pytest.approx(0.995)
        assert a["uptime_30m"] == pytest.approx(0.99806)
        assert a["uptime_1d"] == pytest.approx(1.0)
        # Already-fractional value is preserved.
        assert b["uptime_30m"] == pytest.approx(0.95)
        # Best aggregate is now in fraction-space.
        assert 0.99 < out["best_uptime_30m"] <= 1.0


class TestNormalizeUptime:
    def test_fraction_returned_unchanged(self):
        assert _normalize_uptime(0.998) == pytest.approx(0.998)
        assert _normalize_uptime(0.5) == pytest.approx(0.5)
        assert _normalize_uptime(1.0) == pytest.approx(1.0)

    def test_percentage_divided_by_100(self):
        assert _normalize_uptime(99.806) == pytest.approx(0.99806)
        assert _normalize_uptime(50) == pytest.approx(0.5)
        assert _normalize_uptime(100) == pytest.approx(1.0)

    def test_none_returns_none(self):
        assert _normalize_uptime(None) is None

    def test_non_numeric_returns_none(self):
        assert _normalize_uptime("bad") is None
        assert _normalize_uptime([0.5]) is None

    def test_zero_returned_as_zero(self):
        # Edge: 0 is a fraction, NOT a percentage to divide.
        assert _normalize_uptime(0) == 0.0
        assert _normalize_uptime(0.0) == 0.0


class TestResolveApiKey:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        assert _resolve_or_api_key("explicit") == "explicit"

    def test_env_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        assert _resolve_or_api_key(None) == "from-env"

    def test_returns_none_when_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Settings might still resolve from .env file in dev — patch it out.
        with patch(
            "pyutilz.llm.config.get_llm_settings",
            return_value=MagicMock(openrouter_api_key=None),
        ):
            assert _resolve_or_api_key(None) is None


class TestHealthEnrichment:
    """End-to-end: list_openrouter_models with return_only_healthy=True
    triggers concurrent /endpoints fetches and filters by min_uptime."""

    def setup_method(self):
        openrouter_module._MODELS_CATALOGUE = {
            "openai/gpt-4o-mini": {
                "id": "openai/gpt-4o-mini",
                "context_length": 128000,
                "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
            },
            "anthropic/claude-sonnet-4.6": {
                "id": "anthropic/claude-sonnet-4.6",
                "context_length": 200000,
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            },
            "broken/model": {
                "id": "broken/model",
                "context_length": 8000,
                "pricing": {"prompt": "0.0", "completion": "0.0"},
            },
        }

    def teardown_method(self):
        openrouter_module._MODELS_CATALOGUE = None

    def _make_endpoints_response(self, model_id: str) -> list:
        # gpt-4o-mini → healthy, claude → healthy, broken/model → degraded
        responses = {
            "openai/gpt-4o-mini": [{
                "provider_name": "OpenAI",
                "status": "operational",
                "uptime_last_30m": 0.999,
                "latency_last_30m": {"p50": 180},
                "throughput_last_30m": {"p50": 120},
                "context_length": 128000,
                "pricing": {"prompt": "0.00000015"},
            }],
            "anthropic/claude-sonnet-4.6": [{
                "provider_name": "Anthropic",
                "status": "operational",
                "uptime_last_30m": 1.0,
                "latency_last_30m": {"p50": 250},
                "throughput_last_30m": {"p50": 80},
                "context_length": 200000,
                "pricing": {"prompt": "0.000003"},
            }],
            "broken/model": [{
                "provider_name": "Provider",
                "status": "degraded",
                "uptime_last_30m": 0.85,  # below default 0.95 threshold
                "latency_last_30m": {"p50": 5000},
                "throughput_last_30m": {"p50": 2},
            }],
        }
        return responses.get(model_id, [])

    def _patch_endpoints_fetch(self):
        # _enrich_with_health uses a shared httpx.Client; tests need to
        # intercept its .get(). We patch the Client class in the module
        # to return a MagicMock with a .get method that fakes responses.
        test_self = self

        def _fake_client_factory(*args, **kwargs):
            client = MagicMock()
            client.__enter__ = lambda self_: self_
            client.__exit__ = lambda *a: None

            def _fake_get(url, timeout=None):
                prefix = "https://openrouter.ai/api/v1/models/"
                assert url.startswith(prefix), url
                assert url.endswith("/endpoints"), url
                model_id = url[len(prefix):-len("/endpoints")]
                resp = MagicMock()
                resp.json.return_value = {
                    "data": {"id": model_id, "endpoints": test_self._make_endpoints_response(model_id)}
                }
                resp.raise_for_status = MagicMock()
                return resp

            client.get = MagicMock(side_effect=_fake_get)
            # Capture for assertions in tests that need it
            client._captured_kwargs = kwargs
            return client

        return patch(
            "pyutilz.llm.openrouter_provider.httpx.Client",
            side_effect=_fake_client_factory,
        )

    def test_filters_unhealthy_models(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_endpoints_fetch():
            rows = list_openrouter_models(return_only_healthy=True)
        ids = sorted(r["id"] for r in rows)
        # broken/model dropped (uptime 0.85 < 0.95); 2 healthy survive
        assert ids == ["anthropic/claude-sonnet-4.6", "openai/gpt-4o-mini"]

    def test_attaches_health_block_to_each_row(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_endpoints_fetch():
            rows = list_openrouter_models(return_only_healthy=True)
        for r in rows:
            assert "health" in r
            h = r["health"]
            assert isinstance(h["endpoints"], list) and len(h["endpoints"]) >= 1
            assert h["best_uptime_30m"] is not None
            assert h["best_latency_p50_ms"] is not None

    def test_min_uptime_lower_keeps_more(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_endpoints_fetch():
            rows = list_openrouter_models(return_only_healthy=True, min_uptime=0.80)
        ids = sorted(r["id"] for r in rows)
        # broken/model now passes (uptime 0.85 ≥ 0.80)
        assert "broken/model" in ids
        assert len(rows) == 3

    def test_explicit_api_key_used(self):
        # Even without env var, explicit api_key= should drive the call.
        # Auth now lives on the shared httpx.Client (Authorization header
        # set at construction time, not per-request) so we inspect the
        # Client kwargs rather than per-get call args.
        with self._patch_endpoints_fetch() as mock_client_ctor:
            list_openrouter_models(return_only_healthy=True, api_key="explicit-key")
        ctor_kwargs = mock_client_ctor.call_args.kwargs
        headers = ctor_kwargs.get("headers") or {}
        assert headers.get("Authorization") == "Bearer explicit-key"

    def test_health_fetch_failure_drops_row(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        test_self = self

        def _flaky_client_factory(*args, **kwargs):
            client = MagicMock()
            client.__enter__ = lambda self_: self_
            client.__exit__ = lambda *a: None

            def _flaky_get(url, timeout=None):
                if "broken" in url:
                    raise httpx.ConnectError("boom")
                prefix = "https://openrouter.ai/api/v1/models/"
                model_id = url[len(prefix):-len("/endpoints")]
                resp = MagicMock()
                resp.json.return_value = {
                    "data": {"id": model_id, "endpoints": test_self._make_endpoints_response(model_id)}
                }
                resp.raise_for_status = MagicMock()
                return resp

            client.get = MagicMock(side_effect=_flaky_get)
            return client

        with patch(
            "pyutilz.llm.openrouter_provider.httpx.Client",
            side_effect=_flaky_client_factory,
        ):
            rows = list_openrouter_models(return_only_healthy=True)
        # broken/model excluded due to fetch failure; 2 healthy remain
        ids = sorted(r["id"] for r in rows)
        assert "broken/model" not in ids
        assert len(rows) == 2

    def test_sort_by_uptime_descending(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_endpoints_fetch():
            rows = list_openrouter_models(
                return_only_healthy=True, sort_by="uptime",
            )
        # claude (1.0) > gpt-4o-mini (0.999)
        assert rows[0]["id"] == "anthropic/claude-sonnet-4.6"
        assert rows[1]["id"] == "openai/gpt-4o-mini"

    def test_sort_by_latency_ascending(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_endpoints_fetch():
            rows = list_openrouter_models(
                return_only_healthy=True, sort_by="latency",
            )
        # gpt-4o-mini (180ms) < claude (250ms)
        assert rows[0]["id"] == "openai/gpt-4o-mini"

    def test_sort_by_throughput_descending(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_endpoints_fetch():
            rows = list_openrouter_models(
                return_only_healthy=True, sort_by="throughput",
            )
        # gpt-4o-mini (120 t/s) > claude (80 t/s)
        assert rows[0]["id"] == "openai/gpt-4o-mini"


class TestHealthTTLCache:
    """Stage-2 health enrichment caches per-model results in a process-wide
    TTL dict so repeated calls in a tight loop don't re-pay the TLS cost."""

    def setup_method(self):
        openrouter_module._MODELS_CATALOGUE = {
            "openai/gpt-4o-mini": {
                "id": "openai/gpt-4o-mini",
                "context_length": 128000,
                "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
            },
        }
        openrouter_module._HEALTH_CACHE.clear()

    def teardown_method(self):
        openrouter_module._MODELS_CATALOGUE = None
        openrouter_module._HEALTH_CACHE.clear()

    def _patch_one_model_healthy(self):
        def _fake_client_factory(*args, **kwargs):
            client = MagicMock()
            client.__enter__ = lambda self_: self_
            client.__exit__ = lambda *a: None

            def _fake_get(url, timeout=None):
                resp = MagicMock()
                resp.json.return_value = {
                    "data": {
                        "id": "openai/gpt-4o-mini",
                        "endpoints": [{
                            "provider_name": "OpenAI",
                            "uptime_last_30m": 0.999,
                            "latency_last_30m": {"p50": 180},
                            "throughput_last_30m": {"p50": 120},
                        }],
                    }
                }
                resp.raise_for_status = MagicMock()
                return resp

            client.get = MagicMock(side_effect=_fake_get)
            return client

        return patch(
            "pyutilz.llm.openrouter_provider.httpx.Client",
            side_effect=_fake_client_factory,
        )

    def test_second_call_within_ttl_skips_http(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_one_model_healthy() as ctor1:
            list_openrouter_models(return_only_healthy=True)
        first_call_count = ctor1.call_count
        assert first_call_count == 1

        # Second call should hit the cache and NOT construct a new Client.
        with self._patch_one_model_healthy() as ctor2:
            rows = list_openrouter_models(return_only_healthy=True)
        # Client constructor wasn't even called on the second invocation.
        assert ctor2.call_count == 0
        assert len(rows) == 1
        assert rows[0]["health"]["best_uptime_30m"] == 0.999

    def test_ttl_zero_bypasses_cache(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_one_model_healthy() as ctor1:
            list_openrouter_models(return_only_healthy=True, health_ttl_seconds=0)
        with self._patch_one_model_healthy() as ctor2:
            list_openrouter_models(return_only_healthy=True, health_ttl_seconds=0)
        assert ctor1.call_count == 1
        assert ctor2.call_count == 1  # both fetch fresh

    def test_expired_entry_refetches(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_one_model_healthy():
            list_openrouter_models(return_only_healthy=True, health_ttl_seconds=300)

        # Manually age the cached entry past the TTL.
        with openrouter_module._HEALTH_CACHE_LOCK:
            mid, (ts, h) = next(iter(openrouter_module._HEALTH_CACHE.items()))
            openrouter_module._HEALTH_CACHE[mid] = (ts - 1000.0, h)

        with self._patch_one_model_healthy() as ctor:
            list_openrouter_models(return_only_healthy=True, health_ttl_seconds=300)
        assert ctor.call_count == 1  # re-fetched

    def test_clear_openrouter_caches_empties_health(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with self._patch_one_model_healthy():
            list_openrouter_models(return_only_healthy=True)
        assert len(openrouter_module._HEALTH_CACHE) == 1

        from pyutilz.llm.openrouter_provider import clear_openrouter_caches
        clear_openrouter_caches(models=False, health=True)
        assert len(openrouter_module._HEALTH_CACHE) == 0

    def test_clear_openrouter_caches_default_clears_both(self):
        openrouter_module._HEALTH_CACHE["x"] = (0.0, {"best_uptime_30m": 1.0})
        openrouter_module._MODELS_CATALOGUE = {"a": {}}

        from pyutilz.llm.openrouter_provider import clear_openrouter_caches
        clear_openrouter_caches()
        assert openrouter_module._HEALTH_CACHE == {}
        assert openrouter_module._MODELS_CATALOGUE is None


class TestSharedClientPerf:
    """Verify the shared httpx.Client carries the correct headers and
    pool limits so we don't accidentally regress on TLS overhead."""

    def setup_method(self):
        openrouter_module._MODELS_CATALOGUE = {
            "a/b": {"id": "a/b", "pricing": {"prompt": "0", "completion": "0"}},
            "c/d": {"id": "c/d", "pricing": {"prompt": "0", "completion": "0"}},
        }

    def teardown_method(self):
        openrouter_module._MODELS_CATALOGUE = None

    def test_one_client_for_n_models(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        captured = []

        def _factory(*args, **kwargs):
            captured.append(kwargs)
            client = MagicMock()
            client.__enter__ = lambda self_: self_
            client.__exit__ = lambda *a: None
            client.get = MagicMock(return_value=MagicMock(
                json=lambda: {"data": {"endpoints": [
                    {"uptime_last_30m": 1.0, "latency_last_30m": {"p50": 1},
                     "throughput_last_30m": {"p50": 1}},
                ]}},
                raise_for_status=lambda: None,
            ))
            return client

        with patch("pyutilz.llm.openrouter_provider.httpx.Client", side_effect=_factory):
            list_openrouter_models(return_only_healthy=True, max_workers=4)

        # ONE shared client is constructed, regardless of model count.
        assert len(captured) == 1
        kwargs = captured[0]
        # Auth header set on the client itself.
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        # Limits sized to max_workers (so threads don't fight the pool).
        assert kwargs["limits"].max_keepalive_connections == 4
        assert kwargs["limits"].max_connections == 4


class TestMaxOutputFilter:
    def setup_method(self):
        openrouter_module._MODELS_CATALOGUE = {
            "cheap/model": {
                "id": "cheap/model",
                "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
            },
            "midrange/model": {
                "id": "midrange/model",
                "pricing": {"prompt": "0.0000015", "completion": "0.0000050"},
            },
            "expensive/model": {
                "id": "expensive/model",
                "pricing": {"prompt": "0.0000150", "completion": "0.0000600"},
            },
        }

    def teardown_method(self):
        openrouter_module._MODELS_CATALOGUE = None

    def test_max_output_per_1m_drops_expensive(self):
        # cap at $5/1M output -> midrange & cheap survive (5.0 + 0.5)
        rows = list_openrouter_models(
            max_output_per_1m=5.0,
            return_only_healthy=False,
        )
        ids = sorted(r["id"] for r in rows)
        assert ids == ["cheap/model", "midrange/model"]

    def test_max_input_and_max_output_combined(self):
        rows = list_openrouter_models(
            max_input_per_1m=1.0,    # only cheap/model passes
            max_output_per_1m=10.0,  # cheap & midrange pass
            return_only_healthy=False,
        )
        # Intersection = cheap/model only
        assert [r["id"] for r in rows] == ["cheap/model"]

    def test_no_output_cap_returns_all(self):
        rows = list_openrouter_models(return_only_healthy=False)
        assert len(rows) == 3


class TestLoweredMaxWorkersDefault:
    def test_default_is_eight_not_sixteen(self):
        # 16 was triggering 408 timeouts from OR's edge under load.
        from inspect import signature
        sig = signature(list_openrouter_models)
        assert sig.parameters["max_workers"].default == 8


class TestProviderHealthMethods:
    @pytest.mark.asyncio
    async def test_check_model_health_uses_self_model_by_default(self):
        p = _provider(model="anthropic/claude-sonnet-4.6")
        body = {
            "data": {
                "id": "anthropic/claude-sonnet-4.6",
                "name": "Claude Sonnet 4.6",
                "endpoints": [{
                    "provider_name": "Anthropic",
                    "status": "operational",
                    "uptime_last_30m": 0.999,
                    "latency_last_30m": {"p50": 234},
                    "throughput_last_30m": {"p50": 89},
                }],
            }
        }
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request(
                "GET",
                "https://openrouter.ai/api/v1/models/anthropic/claude-sonnet-4.6/endpoints",
            ),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.check_model_health()

        p._client.get.assert_awaited_once_with(
            "/models/anthropic/claude-sonnet-4.6/endpoints"
        )
        assert out["model"] == "anthropic/claude-sonnet-4.6"
        assert out["name"] == "Claude Sonnet 4.6"
        assert out["best_uptime_30m"] == 0.999
        assert out["best_latency_p50_ms"] == 234
        assert len(out["endpoints"]) == 1

    @pytest.mark.asyncio
    async def test_check_model_health_explicit_model(self):
        p = _provider(model="anthropic/claude-sonnet-4.6")
        body = {"data": {"endpoints": []}}
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://openrouter.ai/x"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)
        await p.check_model_health(model="openai/gpt-4o")
        p._client.get.assert_awaited_once_with("/models/openai/gpt-4o/endpoints")

    @pytest.mark.asyncio
    async def test_is_model_healthy_true_when_above_threshold(self):
        p = _provider()
        body = {
            "data": {"endpoints": [{
                "uptime_last_30m": 0.998,
            }]}
        }
        resp = httpx.Response(
            status_code=200, json=body,
            request=httpx.Request("GET", "https://openrouter.ai/x"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)
        assert await p.is_model_healthy(min_uptime=0.99) is True

    @pytest.mark.asyncio
    async def test_is_model_healthy_false_when_below_threshold(self):
        p = _provider()
        body = {"data": {"endpoints": [{"uptime_last_30m": 0.85}]}}
        resp = httpx.Response(
            status_code=200, json=body,
            request=httpx.Request("GET", "https://openrouter.ai/x"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)
        assert await p.is_model_healthy(min_uptime=0.99) is False

    @pytest.mark.asyncio
    async def test_is_model_healthy_returns_false_on_network_error(self):
        # Crashing guards are worse than conservative ones — an error
        # means "I don't know if it's healthy", which equals "no, hold off".
        p = _provider()
        p._client = AsyncMock()
        p._client.get = AsyncMock(side_effect=httpx.ConnectError("offline"))
        assert await p.is_model_healthy() is False


class TestTimeout:
    def test_reasoning_models_get_long_timeout(self):
        p = _provider()
        for slow in (
            "openai/o1-mini",
            "openai/o3",
            "anthropic/claude-opus-4.6",
            "deepseek/deepseek-reasoner",
        ):
            assert p._get_timeout(slow) == 1200.0, slow

    def test_default_timeout(self):
        p = _provider()
        assert p._get_timeout("openai/gpt-4o-mini") == 240.0


class TestThinkingRequestField:
    """The OR provider routes ``thinking=`` to OR's unified
    ``reasoning`` field. Mapping: bool ``True`` -> medium effort,
    ``False`` / empty string -> ``exclude=True``, explicit effort
    string -> pass through. Bool-flag upstreams (DeepSeek V4) coerce
    non-empty strings to ``True`` via the shared ``_normalize_thinking``
    helper."""

    def test_thinking_true_uses_medium_effort(self):
        p = _provider()
        assert p._thinking_request_field(True) == {
            "reasoning": {"effort": "medium"}
        }

    def test_thinking_false_excludes_reasoning(self):
        p = _provider()
        assert p._thinking_request_field(False) == {
            "reasoning": {"exclude": True}
        }

    def test_thinking_empty_str_excludes_reasoning(self):
        """Defensive: empty string is treated as off, matching ``False``."""
        p = _provider()
        assert p._thinking_request_field("") == {
            "reasoning": {"exclude": True}
        }

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "minimal"])
    def test_thinking_effort_string_passes_through(self, effort):
        p = _provider()
        assert p._thinking_request_field(effort) == {
            "reasoning": {"effort": effort}
        }

    def test_thinking_effort_string_lowercased(self):
        """Helper normalises case so ``"HIGH"`` and ``"high"`` agree."""
        p = _provider()
        assert p._thinking_request_field("HIGH") == {
            "reasoning": {"effort": "high"}
        }

    def test_thinking_unknown_effort_passes_through(self):
        """OR may add new effort tiers later; the provider doesn't
        gate-keep -- forward unknown strings and let OR's edge accept
        or 400 them."""
        p = _provider()
        assert p._thinking_request_field("ultra") == {
            "reasoning": {"effort": "ultra"}
        }


class TestSpecialStatus:
    def test_402_warns(self):
        p = _provider()
        resp = httpx.Response(
            status_code=402,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        # Must not raise — warn-only behaviour, retry loop handles it
        p._handle_special_status(resp)

    def test_200_no_op(self):
        p = _provider()
        resp = httpx.Response(
            status_code=200,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        p._handle_special_status(resp)


class TestIntegrationViaGenerate:
    """End-to-end: mock httpx, exercise generate() and verify the cost
    capture is wired through the base class hook (proves the
    _track_provider_specific_usage hook path works in practice)."""

    @pytest.mark.asyncio
    async def test_usage_cost_recorded_after_generate(self):
        p = _provider()
        body = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "cost": 0.0007,
                "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 50},
            },
        }
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=resp)

        out = await p.generate("hello")

        assert out == "hi"
        assert p.total_actual_cost_usd == pytest.approx(0.0007)
        assert p.last_actual_cost_usd == pytest.approx(0.0007)
        assert p.total_cache_write_tokens == 50
        # cached_tokens path populates cache_hit_tokens
        assert p.total_cache_hit_tokens == 3
        # Standard usage tracking still works
        assert p.total_prompt_tokens == 12
        assert p.total_completion_tokens == 4

    @pytest.mark.asyncio
    async def test_provider_routing_in_request_body(self):
        p = _provider(
            provider_order=("anthropic", "openai"),
            provider_sort="price",
            models_fallback=("openai/gpt-4o",),
        )
        resp_body = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        resp = httpx.Response(
            status_code=200,
            json=resp_body,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=resp)

        await p.generate("hello")

        sent = p._client.post.call_args.kwargs.get("json") \
            or p._client.post.call_args[1].get("json")
        assert sent["provider"] == {
            "order": ["anthropic", "openai"],
            "sort": "price",
        }
        assert sent["models"] == ["openai/gpt-4o"]


class TestExtendedUsageCapture:
    """Every field OR exposes on every response — track them all."""

    def test_upstream_inference_cost_recorded(self):
        p = _provider()
        p._track_provider_specific_usage(
            {"cost": 0.005, "cost_details": {"upstream_inference_cost": 0.0042}}
        )
        assert p.last_upstream_inference_cost_usd == pytest.approx(0.0042)
        assert p.total_upstream_inference_cost_usd == pytest.approx(0.0042)

    def test_upstream_inference_cost_accumulates(self):
        p = _provider()
        p._track_provider_specific_usage(
            {"cost_details": {"upstream_inference_cost": 0.001}}
        )
        p._track_provider_specific_usage(
            {"cost_details": {"upstream_inference_cost": 0.002}}
        )
        assert p.total_upstream_inference_cost_usd == pytest.approx(0.003)
        assert p.last_upstream_inference_cost_usd == pytest.approx(0.002)

    def test_upstream_inference_cost_none_when_not_byok(self):
        # Non-BYOK calls don't include cost_details — record explicit None
        # so callers can distinguish "no upstream cost reported" from "0.00".
        p = _provider()
        p._track_provider_specific_usage({"cost": 0.005})
        assert p.last_upstream_inference_cost_usd is None

    def test_audio_tokens_recorded(self):
        p = _provider()
        p._track_provider_specific_usage(
            {"prompt_tokens_details": {"audio_tokens": 128}}
        )
        assert p.last_audio_tokens == 128
        assert p.total_audio_tokens == 128

    def test_audio_tokens_accumulate(self):
        p = _provider()
        p._track_provider_specific_usage({"prompt_tokens_details": {"audio_tokens": 50}})
        p._track_provider_specific_usage({"prompt_tokens_details": {"audio_tokens": 70}})
        assert p.total_audio_tokens == 120

    def test_last_cache_hit_tokens_set_each_call(self):
        # last_cache_hit_tokens is a per-call field — should reflect the
        # latest call even when the cumulative counter doesn't grow (because
        # the same value was already counted via prompt_cache_hit_tokens).
        p = _provider()
        p._track_provider_specific_usage(
            {"prompt_tokens_details": {"cached_tokens": 30}}
        )
        assert p.last_cache_hit_tokens == 30


class TestResponseLevelMetadata:
    """OR-specific response-level fields outside the usage block."""

    def test_generation_id_captured(self):
        p = _provider()
        p._track_provider_specific_response({"id": "gen-abc123"})
        assert p.last_generation_id == "gen-abc123"

    def test_upstream_provider_captured(self):
        p = _provider()
        p._track_provider_specific_response({"provider": "Anthropic"})
        assert p.last_upstream_provider == "Anthropic"

    def test_upstream_model_captured(self):
        # Useful when models_fallback kicks in — actual model differs from
        # what we requested.
        p = _provider(model="anthropic/claude-sonnet-4.6")
        p._track_provider_specific_response(
            {"model": "openai/gpt-4o"}  # fallback fired
        )
        assert p.last_upstream_model == "openai/gpt-4o"
        assert p.model_name == "anthropic/claude-sonnet-4.6"  # requested unchanged

    def test_native_finish_reason_captured(self):
        p = _provider()
        p._track_provider_specific_response(
            {"choices": [{"native_finish_reason": "tool_calls"}]}
        )
        assert p.last_native_finish_reason == "tool_calls"

    def test_response_metadata_resilient_to_missing_fields(self):
        # Tolerate the empty / partial responses some upstreams return on errors.
        p = _provider()
        p._track_provider_specific_response({})
        p._track_provider_specific_response({"choices": []})
        assert p.last_generation_id is None
        assert p.last_upstream_provider is None
        assert p.last_native_finish_reason is None


class TestLastCallSummary:
    @pytest.mark.asyncio
    async def test_summary_after_generate(self):
        p = _provider(model="anthropic/claude-sonnet-4.6")
        body = {
            "id": "gen-xyz",
            "provider": "Anthropic",
            "model": "anthropic/claude-sonnet-4.6",
            "choices": [{
                "message": {"content": "ok"},
                "finish_reason": "stop",
                "native_finish_reason": "end_turn",
            }],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 56,
                "cost": 0.0042,
                "cost_details": {"upstream_inference_cost": 0.004},
                "prompt_tokens_details": {
                    "cached_tokens": 800,
                    "cache_write_tokens": 0,
                    "audio_tokens": 0,
                },
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=resp)

        await p.generate("hello")
        summary = p.last_call_summary()

        assert summary["generation_id"] == "gen-xyz"
        assert summary["upstream_provider"] == "Anthropic"
        assert summary["upstream_model"] == "anthropic/claude-sonnet-4.6"
        assert summary["requested_model"] == "anthropic/claude-sonnet-4.6"
        assert summary["finish_reason"] == "stop"
        assert summary["native_finish_reason"] == "end_turn"
        assert summary["cost_usd"] == pytest.approx(0.0042)
        assert summary["upstream_inference_cost_usd"] == pytest.approx(0.004)
        assert summary["input_tokens"] == 1200
        assert summary["output_tokens"] == 56
        assert summary["cache_hit_tokens"] == 800

    def test_summary_pre_generate_returns_none_defaults(self):
        # Calling .last_call_summary() before any generate() should not crash —
        # it should return a fully-populated dict with sensible defaults.
        p = _provider()
        s = p.last_call_summary()
        assert s["generation_id"] is None
        assert s["upstream_provider"] is None
        assert s["cost_usd"] == 0.0
        assert s["input_tokens"] == 0


class TestGenerationLookup:
    @pytest.mark.asyncio
    async def test_fetch_uses_last_id_by_default(self):
        p = _provider()
        p.last_generation_id = "gen-prev"
        body = {
            "data": {
                "id": "gen-prev",
                "total_cost": 0.0042,
                "latency": 1234,
                "is_byok": False,
                "finish_reason": "stop",
            }
        }
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/generation"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.fetch_generation_stats()

        p._client.get.assert_awaited_once_with("/generation", params={"id": "gen-prev"})
        assert out["total_cost"] == pytest.approx(0.0042)
        assert out["latency"] == 1234

    @pytest.mark.asyncio
    async def test_fetch_explicit_id_overrides_last(self):
        p = _provider()
        p.last_generation_id = "gen-prev"
        body = {"data": {"id": "gen-other", "total_cost": 0.001}}
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/generation"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.fetch_generation_stats("gen-other")

        p._client.get.assert_awaited_once_with("/generation", params={"id": "gen-other"})
        assert out["id"] == "gen-other"

    @pytest.mark.asyncio
    async def test_fetch_raises_without_id(self):
        p = _provider()
        # last_generation_id is None and no explicit ID given
        with pytest.raises(ValueError, match="generation_id"):
            await p.fetch_generation_stats()


class TestExtendedSessionCost:
    def test_session_cost_includes_upstream(self):
        p = _provider()
        p.total_actual_cost_usd = 1.0
        p.total_upstream_inference_cost_usd = 0.85
        p.last_upstream_inference_cost_usd = 0.05
        p.total_audio_tokens = 999
        p.total_cache_write_tokens = 500
        cost = p.get_session_cost()
        assert cost["actual_cost_usd"] == 1.0
        assert cost["upstream_inference_cost_usd"] == 0.85
        assert cost["last_upstream_inference_cost_usd"] == 0.05
        assert cost["audio_tokens"] == 999
        assert cost["cache_write_tokens"] == 500


class TestAccountIntrospection:
    @pytest.mark.asyncio
    async def test_check_account_limits_unwraps_data_envelope(self):
        # OR returns ``{"data": {...}}`` — caller wants the inner dict.
        p = _provider()
        body = {
            "data": {
                "label": "primary",
                "limit": 50.0,
                "limit_remaining": 12.5,
                "limit_reset": "monthly",
                "usage": 37.5,
                "usage_daily": 1.2,
                "is_free_tier": False,
            }
        }
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/key"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.check_account_limits()
        p._client.get.assert_awaited_once_with("/key")
        assert out["label"] == "primary"
        assert out["limit_remaining"] == 12.5
        assert out["is_free_tier"] is False

    @pytest.mark.asyncio
    async def test_check_account_limits_handles_non_envelope(self):
        # Defensive: if OR ever returns the dict without a "data" wrapper,
        # we still surface it instead of silently returning {}.
        p = _provider()
        body = {"limit": None, "is_free_tier": True}
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/key"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.check_account_limits()
        assert out["limit"] is None
        assert out["is_free_tier"] is True
        # ``raw`` always present for forward compat
        assert out["raw"] == body

    @pytest.mark.asyncio
    async def test_check_account_limits_propagates_http_errors(self):
        p = _provider()
        resp = httpx.Response(
            status_code=401,
            json={"error": {"message": "invalid key"}},
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/key"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        with pytest.raises(httpx.HTTPStatusError):
            await p.check_account_limits()

    @pytest.mark.asyncio
    async def test_get_account_credits(self):
        p = _provider()
        body = {"data": {"total_credits": 100.0, "total_usage": 37.5}}
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/credits"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.get_account_credits()
        p._client.get.assert_awaited_once_with("/credits")
        # Normalized to match base-class schema across all providers
        assert out["balance_usd"] == pytest.approx(62.5)
        assert out["total_granted"] == pytest.approx(100.0)
        assert out["total_used"] == pytest.approx(37.5)
        assert out["currency"] == "USD"
        assert out["is_available"] is True
        assert out["raw"] == {"total_credits": 100.0, "total_usage": 37.5}

    @pytest.mark.asyncio
    async def test_get_account_credits_zero_balance_unavailable(self):
        p = _provider()
        body = {"data": {"total_credits": 10.0, "total_usage": 10.0}}
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://openrouter.ai/api/v1/credits"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.get_account_credits()
        assert out["balance_usd"] == 0.0
        assert out["is_available"] is False


class TestFactoryIntegration:
    def test_canonical_name_registered(self):
        from pyutilz.llm.factory import _PROVIDER_MODULES
        assert "openrouter" in _PROVIDER_MODULES

    @pytest.mark.parametrize("alias", ["or", "router"])
    def test_aliases_resolve(self, alias):
        from pyutilz.llm.factory import _ALIASES
        assert _ALIASES[alias] == "openrouter"
