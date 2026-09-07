"""OpenRouter provider health checks and account introspection.

Carved out of ``test_llm_openrouter.py`` on 2026-09-07, when that module crossed the 2000-line
budget. These exercise the /models and /key endpoints, the health TTL cache and its log dedup --
a different surface from the generation-path tests that stayed behind.
"""

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
    """Build a fully-initialised OpenRouterProvider for tests.

    Calls the REAL ``__init__`` via a patched settings stub -- avoids
    the previous "manually set 30 attributes" pattern that silently
    drifted when ``__init__`` gained new fields. ``overrides`` accepts
    any constructor kwarg; sensible test defaults are supplied below.
    """
    mock_settings = MagicMock()
    mock_settings.openrouter_api_key = None  # force explicit api_key path
    init_kwargs = {
        "api_key": "test-key",  # pragma: allowlist secret -- a literal dummy in a mocked-transport test
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


class TestHealthCheckLogDedup:
    """Regression (2026-07-21 audit round 2, MEDIUM): a single shared root cause (e.g. an
    expired/revoked API key) used to log one WARNING PER failed model -- 200+ near-identical
    lines burying the actual signal. Failures are now aggregated into ONE summary WARNING after
    the fan-out completes; per-model detail moves to DEBUG."""

    def setup_method(self):
        openrouter_module._MODELS_CATALOGUE = {
            "a/model-1": {"id": "a/model-1", "pricing": {"prompt": "0", "completion": "0"}},
            "a/model-2": {"id": "a/model-2", "pricing": {"prompt": "0", "completion": "0"}},
            "a/model-3": {"id": "a/model-3", "pricing": {"prompt": "0", "completion": "0"}},
        }

    def teardown_method(self):
        openrouter_module._MODELS_CATALOGUE = None

    def _all_fail_client_factory(self, *args, **kwargs):
        client = MagicMock()
        client.__enter__ = lambda self_: self_
        client.__exit__ = lambda *a: None

        def _get(url, timeout=None):
            raise httpx.ConnectError("boom")

        client.get = MagicMock(side_effect=_get)
        return client

    def test_one_aggregated_warning_not_one_per_model(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with patch("pyutilz.llm.openrouter_provider.httpx.Client", side_effect=self._all_fail_client_factory):
            with caplog.at_level("DEBUG"):
                list_openrouter_models(return_only_healthy=True)

        warning_records = [r for r in caplog.records if r.levelname == "WARNING" and "Health check failed" in r.getMessage()]
        assert len(warning_records) == 1
        assert "3/3" in warning_records[0].getMessage()

        debug_records = [r for r in caplog.records if r.levelname == "DEBUG" and "Health check failed" in r.getMessage()]
        assert len(debug_records) == 3

    def test_no_failures_means_no_aggregated_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        def _ok_client_factory(*args, **kwargs):
            client = MagicMock()
            client.__enter__ = lambda self_: self_
            client.__exit__ = lambda *a: None

            def _get(url, timeout=None):
                resp = MagicMock()
                resp.json.return_value = {"data": {"endpoints": []}}
                resp.raise_for_status = MagicMock()
                return resp

            client.get = MagicMock(side_effect=_get)
            return client

        with patch("pyutilz.llm.openrouter_provider.httpx.Client", side_effect=_ok_client_factory):
            with caplog.at_level("WARNING"):
                list_openrouter_models(return_only_healthy=True)

        warning_records = [r for r in caplog.records if r.levelname == "WARNING" and "Health check failed" in r.getMessage()]
        assert len(warning_records) == 0


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
        # /credits AND /key are called now (the latter to read is_free_tier
        # so a free-tier user with balance=None still reports available).
        credits_body = {"data": {"total_credits": 100.0, "total_usage": 37.5}}
        key_body = {"data": {"is_free_tier": False, "limit": None}}

        async def _fake_get(path):
            body = credits_body if path == "/credits" else key_body
            return httpx.Response(
                status_code=200, json=body,
                request=httpx.Request("GET", f"https://openrouter.ai/api/v1{path}"),
            )

        p._client = AsyncMock()
        p._client.get = AsyncMock(side_effect=_fake_get)

        out = await p.get_account_credits()
        # Both endpoints hit
        assert p._client.get.await_count == 2
        called_paths = sorted(c.args[0] for c in p._client.get.await_args_list)
        assert called_paths == ["/credits", "/key"]
        assert out["balance_usd"] == pytest.approx(62.5)
        assert out["total_granted"] == pytest.approx(100.0)
        assert out["total_used"] == pytest.approx(37.5)
        assert out["currency"] == "USD"
        assert out["is_available"] is True
        assert out["is_free_tier"] is False
        assert out["raw"] == {"total_credits": 100.0, "total_usage": 37.5}

    @pytest.mark.asyncio
    async def test_get_account_credits_zero_balance_unavailable(self):
        p = _provider()
        credits_body = {"data": {"total_credits": 10.0, "total_usage": 10.0}}
        key_body = {"data": {"is_free_tier": False}}

        async def _fake_get(path):
            body = credits_body if path == "/credits" else key_body
            return httpx.Response(
                status_code=200, json=body,
                request=httpx.Request("GET", f"https://openrouter.ai/api/v1{path}"),
            )

        p._client = AsyncMock()
        p._client.get = AsyncMock(side_effect=_fake_get)

        out = await p.get_account_credits()
        assert out["balance_usd"] == 0.0
        assert out["is_available"] is False
        assert out["is_free_tier"] is False

    @pytest.mark.asyncio
    async def test_get_account_credits_free_tier_available_despite_no_balance(self):
        """A free-tier user (never purchased credits) has balance=None
        but is_available=True since free-models quota still applies."""
        p = _provider()
        credits_body = {"data": {}}  # no total_credits / total_usage
        key_body = {"data": {"is_free_tier": True}}

        async def _fake_get(path):
            body = credits_body if path == "/credits" else key_body
            return httpx.Response(
                status_code=200, json=body,
                request=httpx.Request("GET", f"https://openrouter.ai/api/v1{path}"),
            )

        p._client = AsyncMock()
        p._client.get = AsyncMock(side_effect=_fake_get)

        out = await p.get_account_credits()
        assert out["balance_usd"] is None
        assert out["is_free_tier"] is True
        assert out["is_available"] is True

    @pytest.mark.asyncio
    async def test_get_account_credits_resilient_to_key_lookup_failure(self):
        """If /key fails, fall back to strict balance check; don't blow up."""
        p = _provider()
        credits_body = {"data": {"total_credits": 50.0, "total_usage": 10.0}}

        async def _fake_get(path):
            if path == "/key":
                raise httpx.ConnectError("network glitch")
            return httpx.Response(
                status_code=200, json=credits_body,
                request=httpx.Request("GET", f"https://openrouter.ai/api/v1{path}"),
            )

        p._client = AsyncMock()
        p._client.get = AsyncMock(side_effect=_fake_get)

        out = await p.get_account_credits()
        assert out["balance_usd"] == pytest.approx(40.0)
        assert out["is_free_tier"] is None  # lookup failed
        assert out["is_available"] is True  # balance > 0 still drives this
