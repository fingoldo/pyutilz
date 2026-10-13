"""Continuation of test_llm_openrouter: split out to keep each test module under the 1k-LOC gate."""
import logging
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from pyutilz.llm.openrouter_provider import OpenRouterProvider, list_openrouter_models
import pyutilz.llm.openrouter_provider as openrouter_module

httpx = pytest.importorskip("httpx")

pytest.importorskip("pydantic")

def _provider(**overrides) -> OpenRouterProvider:
    """Build a fully-initialised OpenRouterProvider for tests.

    Calls the REAL ``__init__`` via a patched settings stub -- avoids
    the previous "manually set 30 attributes" pattern that silently
    drifted when ``__init__`` gained new fields. ``overrides`` accepts
    any constructor kwarg; sensible test defaults are supplied below.
    """
    mock_settings = MagicMock()
    mock_settings.openrouter_api_key = None  # force explicit api_key path
    init_kwargs = {
        "api_key": "test-key",  # pragma: allowlist secret
        "model": "openai/gpt-4o-mini",
        "max_concurrent": 10,
    }
    init_kwargs.update(overrides)
    with patch(
        "pyutilz.llm.openrouter_provider.get_llm_settings",
        return_value=mock_settings,
    ):
        return OpenRouterProvider(**init_kwargs)


@pytest.fixture(autouse=True)
def _reset_caches():
    """Each test starts with fresh module-level caches so prior fixtures
    don't leak. The catalogue + health cache are process-wide singletons
    in the implementation; clearing on entry AND exit guards both ways.
    """
    openrouter_module._MODELS_CATALOGUE = None
    openrouter_module._HEALTH_CACHE.clear()
    yield
    openrouter_module._MODELS_CATALOGUE = None
    openrouter_module._HEALTH_CACHE.clear()


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
            max_input_per_1m=1.0,  # only cheap/model passes
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

        p._client.get.assert_awaited_once_with("/models/anthropic/claude-sonnet-4.6/endpoints")
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
    ``False`` / empty string -> minimal effort AND ``exclude=True``,
    explicit effort string -> pass through. Bool-flag upstreams
    (DeepSeek V4) coerce non-empty strings to ``True`` via the shared
    ``_normalize_thinking`` helper.

    ``False`` asks for NO reasoning, not a small budget - RE-MEASURED 2026-09-02 across eight models. The
    older mapping (`effort: "minimal"` with `exclude: True`, itself a 2026-08-07 fix for `exclude` alone,
    which hides the reasoning TEXT while the model still reasons and is still billed) shrinks the budget
    without closing it: 13-348 billed reasoning tokens per call, and on z-ai/glm-4.7-flash an EMPTY answer
    cut off by `length`. `enabled: False` bills zero on every model that accepts it. Two of the eight
    (google/gemini-3.7-flash, openai/gpt-oss-120b) refuse the call outright - see
    `TestThinkingOffMeansNotBilledForThinking` for that fallback.
    """

    def test_thinking_true_uses_medium_effort(self):
        p = _provider()
        assert p._thinking_request_field(True) == {"reasoning": {"effort": "medium"}}

    def test_thinking_false_asks_for_no_reasoning(self):
        p = _provider()
        assert p._thinking_request_field(False) == {"reasoning": {"enabled": False}}

    def test_thinking_empty_str_asks_for_no_reasoning(self):
        """Defensive: empty string is treated as off, matching ``False``."""
        p = _provider()
        assert p._thinking_request_field("") == {"reasoning": {"enabled": False}}

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "minimal"])
    def test_thinking_effort_string_passes_through(self, effort):
        p = _provider()
        assert p._thinking_request_field(effort) == {"reasoning": {"effort": effort}}

    def test_thinking_effort_string_lowercased(self):
        """Helper normalises case so ``"HIGH"`` and ``"high"`` agree."""
        p = _provider()
        assert p._thinking_request_field("HIGH") == {"reasoning": {"effort": "high"}}

    def test_thinking_unknown_effort_passes_through(self):
        """OR may add new effort tiers later; the provider doesn't
        gate-keep -- forward unknown strings and let OR's edge accept
        or 400 them."""
        p = _provider()
        assert p._thinking_request_field("ultra") == {"reasoning": {"effort": "ultra"}}


class TestSpecialStatus:
    _LOGGER = "pyutilz.llm.openrouter_provider._provider"

    def test_402_warns(self, caplog):
        """402 is credit exhaustion: warn-only (the retry loop handles it), but the warning
        is the operator's only signal that a paid key has run dry, so it must actually fire."""
        p = _provider()
        resp = httpx.Response(
            status_code=402,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            p._handle_special_status(resp)  # must not raise
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "402" in warnings[0].getMessage()
        assert "credits" in warnings[0].getMessage().lower()

    def test_200_no_op(self, caplog):
        """The no-op half must be distinguishable from the 402 half: silence, not a warning."""
        p = _provider()
        resp = httpx.Response(
            status_code=200,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        with caplog.at_level(logging.DEBUG, logger=self._LOGGER):
            p._handle_special_status(resp)
        assert caplog.records == []


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

        sent = p._client.post.call_args.kwargs.get("json") or p._client.post.call_args[1].get("json")
        assert sent["provider"] == {
            "order": ["anthropic", "openai"],
            "sort": "price",
        }
        assert sent["models"] == ["openai/gpt-4o"]


class TestExtendedUsageCapture:
    """Every field OR exposes on every response — track them all."""

    def test_upstream_inference_cost_recorded(self):
        p = _provider()
        p._track_provider_specific_usage({"cost": 0.005, "cost_details": {"upstream_inference_cost": 0.0042}})
        assert p.last_upstream_inference_cost_usd == pytest.approx(0.0042)
        assert p.total_upstream_inference_cost_usd == pytest.approx(0.0042)

    def test_upstream_inference_cost_accumulates(self):
        p = _provider()
        p._track_provider_specific_usage({"cost_details": {"upstream_inference_cost": 0.001}})
        p._track_provider_specific_usage({"cost_details": {"upstream_inference_cost": 0.002}})
        assert p.total_upstream_inference_cost_usd == pytest.approx(0.003)
        assert p.last_upstream_inference_cost_usd == pytest.approx(0.003)  # both POSTs are one call context: summed (OR-5)

    def test_upstream_inference_cost_none_when_not_byok(self):
        # Non-BYOK calls don't include cost_details — record explicit None
        # so callers can distinguish "no upstream cost reported" from "0.00".
        p = _provider()
        p._track_provider_specific_usage({"cost": 0.005})
        assert p.last_upstream_inference_cost_usd is None

    def test_audio_tokens_recorded(self):
        p = _provider()
        p._track_provider_specific_usage({"prompt_tokens_details": {"audio_tokens": 128}})
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
        p._track_provider_specific_usage({"prompt_tokens_details": {"cached_tokens": 30}})
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
        p._track_provider_specific_response({"model": "openai/gpt-4o"})  # fallback fired
        assert p.last_upstream_model == "openai/gpt-4o"
        assert p.model_name == "anthropic/claude-sonnet-4.6"  # requested unchanged

    def test_native_finish_reason_captured(self):
        p = _provider()
        p._track_provider_specific_response({"choices": [{"native_finish_reason": "tool_calls"}]})
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


class TestFactoryIntegration:
    def test_canonical_name_registered(self):
        from pyutilz.llm.factory import _PROVIDER_MODULES
        assert "openrouter" in _PROVIDER_MODULES

    @pytest.mark.parametrize("alias", ["or", "router"])
    def test_aliases_resolve(self, alias):
        from pyutilz.llm.factory import _ALIASES
        assert _ALIASES[alias] == "openrouter"


class TestRoutingRetry:
    """``retry_routing_404`` opt-in: bounded retry on OR routing 404/405.

    Default OFF preserves production behaviour (instant fail on 404
    keeps the concurrency pool safe). When ON, transient routing 404s
    (provider redeploy / cold start / region failover) get a chance to
    recover before the row is recorded as a failure.
    """

    def test_default_disabled(self):
        p = _provider()
        assert p._retry_routing_404 is False
        assert p._routing_404_max_attempts == 3
        assert p._routing_404_pause_sec == 60.0

    def test_enabled_via_constructor(self):
        p = _provider(
            retry_routing_404=True,
            routing_404_max_attempts=4,
            routing_404_pause_sec=10.0,
        )
        assert p._retry_routing_404 is True
        assert p._routing_404_max_attempts == 4
        assert p._routing_404_pause_sec == 10.0

    @pytest.mark.asyncio
    async def test_disabled_by_default_no_retry(self):
        from pyutilz.llm.exceptions import LLMProviderError
        p = _provider()
        with patch(
            "pyutilz.llm.openai_compat.OpenAICompatibleProvider.generate",
            new=AsyncMock(side_effect=LLMProviderError("OpenRouter API error 404: No endpoints found for X.")),
        ) as parent:
            with pytest.raises(LLMProviderError):
                await p.generate("hi")
            assert parent.call_count == 1  # no retry path

    @pytest.mark.asyncio
    async def test_retries_on_routing_404(self):
        from pyutilz.llm.exceptions import LLMProviderError
        p = _provider(
            retry_routing_404=True,
            routing_404_max_attempts=3,
            routing_404_pause_sec=0.0,
        )
        with patch(
            "pyutilz.llm.openai_compat.OpenAICompatibleProvider.generate",
            new=AsyncMock(side_effect=LLMProviderError("OpenRouter API error 404: No endpoints found for X.")),
        ) as parent:
            with pytest.raises(LLMProviderError):
                await p.generate("hi")
            assert parent.call_count == 3  # original + 2 retries

    @pytest.mark.asyncio
    async def test_retry_recovers_when_second_attempt_succeeds(self):
        from pyutilz.llm.exceptions import LLMProviderError
        p = _provider(
            retry_routing_404=True,
            routing_404_max_attempts=3,
            routing_404_pause_sec=0.0,
        )
        side_effects = [
            LLMProviderError("OpenRouter API error 404: No endpoints found for X."),
            "second-attempt-ok",
        ]
        with patch(
            "pyutilz.llm.openai_compat.OpenAICompatibleProvider.generate",
            new=AsyncMock(side_effect=side_effects),
        ) as parent:
            result = await p.generate("hi")
            assert result == "second-attempt-ok"
            assert parent.call_count == 2  # one fail, one success

    @pytest.mark.asyncio
    async def test_non_routing_error_propagates_immediately(self):
        """A 422 ContextOverflow is NOT a routing 404 - must not retry."""
        from pyutilz.llm.exceptions import LLMProviderError
        p = _provider(
            retry_routing_404=True,
            routing_404_max_attempts=3,
            routing_404_pause_sec=0.0,
        )
        with patch(
            "pyutilz.llm.openai_compat.OpenAICompatibleProvider.generate",
            new=AsyncMock(side_effect=LLMProviderError("OpenRouter API error 422: maximum context length is 8192")),
        ) as parent:
            with pytest.raises(LLMProviderError):
                await p.generate("hi")
            assert parent.call_count == 1  # no retry on non-routing error

    @pytest.mark.asyncio
    async def test_retry_recognises_405_method_not_allowed(self):
        from pyutilz.llm.exceptions import LLMProviderError
        p = _provider(
            retry_routing_404=True,
            routing_404_max_attempts=2,
            routing_404_pause_sec=0.0,
        )
        with patch(
            "pyutilz.llm.openai_compat.OpenAICompatibleProvider.generate",
            new=AsyncMock(side_effect=LLMProviderError("OpenRouter API error 405: Method not allowed")),
        ) as parent:
            with pytest.raises(LLMProviderError):
                await p.generate("hi")
            assert parent.call_count == 2

class TestSubpackageSplitSensor:
    """Guards the ``openrouter_provider`` subpackage split (carved out of
    the former >1000-LOC single module). Asserts the facade preserves the
    public import surface, that representative symbols resolve to the
    carved-out submodules, and that the provider is still discoverable via
    the factory registry so it self-registers at import time.
    """

    def test_facade_reexports_provider_class_and_symbols(self):
        import pyutilz.llm.openrouter_provider as m

        assert m.OpenRouterProvider.__name__ == "OpenRouterProvider"
        assert callable(m.list_openrouter_models)
        assert callable(m.clear_openrouter_caches)
        for name in (
            "_fetch_models_catalogue", "_per_token_cost_pair",
            "_resolve_model_limits", "_summarize_endpoints",
            "_normalize_uptime", "_resolve_or_api_key",
        ):
            assert hasattr(m, name), f"facade lost {name} after split"
        assert hasattr(m, "_MODELS_CATALOGUE")
        assert hasattr(m, "_HEALTH_CACHE")

    def test_carved_submodules_exist(self):
        import importlib

        for sub in ("_catalogue", "_health", "_provider"):
            mod = importlib.import_module(f"pyutilz.llm.openrouter_provider.{sub}")
            assert mod is not None
        from pyutilz.llm.openrouter_provider._provider import (
            OpenRouterProvider as _Cls,
        )
        import pyutilz.llm.openrouter_provider as m
        assert m.OpenRouterProvider is _Cls

    def test_provider_still_registered_in_factory(self):
        import importlib

        from pyutilz.llm import factory as factory_module

        assert "openrouter" in factory_module._PROVIDER_MODULES
        mod_path, cls_name, _key_attr = factory_module._PROVIDER_MODULES["openrouter"]
        assert mod_path == "pyutilz.llm.openrouter_provider"
        mod = importlib.import_module(mod_path)
        assert getattr(mod, cls_name).__name__ == "OpenRouterProvider"


class TestThinkingOffMeansNotBilledForThinking:
    """Measured 2026-09-02 across eight models: `effort=minimal, exclude=true` still bills 13-348 reasoning
    tokens (and on z-ai/glm-4.7-flash returns nothing at all), while `enabled: false` bills zero - except on
    the two endpoints that refuse the whole call, which is what the fallback exists for."""

    def _provider(self, model="z-ai/glm-4.7-flash"):
        from pyutilz.llm.openrouter_provider._provider import OpenRouterProvider

        return OpenRouterProvider(api_key="test-key", model=model)  # pragma: allowlist secret

    def test_thinking_off_asks_for_no_reasoning_at_all(self):
        assert self._provider()._thinking_request_field(False) == {"reasoning": {"enabled": False}}

    def test_thinking_on_is_unchanged(self):
        assert self._provider()._thinking_request_field(True) == {"reasoning": {"effort": "medium"}}
        assert self._provider()._thinking_request_field("high") == {"reasoning": {"effort": "high"}}

    def test_an_endpoint_that_mandates_reasoning_gets_the_budget_form_instead(self):
        from pyutilz.llm.openrouter_provider import _provider as mod

        p = self._provider("google/gemini-3.7-flash")
        body = {"model": p.model_name, "reasoning": {"enabled": False}}
        mod._REASONING_CANNOT_BE_DISABLED.discard(p.model_name)
        try:
            repaired = p._body_after_rejected_request(body, 404, "Reasoning is mandatory for this endpoint and cannot be disabled")
            assert repaired["reasoning"] == {"effort": "minimal", "exclude": True}
            assert p._thinking_request_field(False) == {"reasoning": {"effort": "minimal", "exclude": True}}, "and remembered, so the refusal is paid once"
        finally:
            mod._REASONING_CANNOT_BE_DISABLED.discard(p.model_name)

    def test_an_unrelated_rejection_is_not_repaired(self):
        p = self._provider()
        assert p._body_after_rejected_request({"reasoning": {"enabled": False}}, 400, "context length exceeded") is None
        assert p._body_after_rejected_request({"temperature": 5}, 400, "reasoning is mandatory") is None


# `TestCeilingFallbackIsAnnounced` lived here as a byte-for-byte copy of the class in
# `test_llm_openrouter_ceiling.py`, whose own docstring says it exists because THIS file is at its LOC
# budget with no slack. Two copies of one guard drift apart and read as two independent confirmations.
