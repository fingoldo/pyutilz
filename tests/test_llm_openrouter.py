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
    return p


@pytest.fixture(autouse=True)
def _reset_models_cache():
    """Each test starts with a fresh /models cache so prior fixtures don't leak."""
    openrouter_module._MODELS_CATALOGUE = None
    yield
    openrouter_module._MODELS_CATALOGUE = None


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
        rows = list_openrouter_models()
        assert len(rows) == 3

    def test_filter_by_name_substring(self):
        rows = list_openrouter_models(name_contains="claude")
        assert len(rows) == 1
        assert rows[0]["id"] == "anthropic/claude-sonnet-4.6"

    def test_filter_by_name_case_insensitive(self):
        rows = list_openrouter_models(name_contains="CLAUDE")
        assert len(rows) == 1

    def test_filter_by_max_input_price(self):
        # 0.50 USD per 1M tokens — should match only gpt-4o-mini
        rows = list_openrouter_models(max_input_per_1m=0.50)
        assert len(rows) == 1
        assert rows[0]["id"] == "openai/gpt-4o-mini"

    def test_sort_by_input_price_ascending(self):
        rows = list_openrouter_models(sort_by="input_price")
        assert [r["id"] for r in rows] == [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4.6",
        ]

    def test_sort_by_context_descending(self):
        # context sort should put the BIGGEST window first (most useful default)
        rows = list_openrouter_models(sort_by="context")
        assert rows[0]["id"] == "anthropic/claude-sonnet-4.6"

    def test_combined_filter_and_sort(self):
        rows = list_openrouter_models(
            name_contains="openai",
            sort_by="input_price",
        )
        assert [r["id"] for r in rows] == ["openai/gpt-4o-mini", "openai/gpt-4o"]


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
