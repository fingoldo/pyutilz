"""Tests for OpenRouter provider — meta-provider semantics + OR-specific bits."""

from unittest.mock import patch, MagicMock
import pytest
from pyutilz.llm.openrouter_provider import OpenRouterProvider, list_openrouter_models, _fetch_models_catalogue, _per_token_cost_pair, _resolve_model_limits, _summarize_endpoints, _normalize_uptime, _resolve_or_api_key
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
    # B13 dedup: covered by integration test_usage_cost_recorded_after_generate.

    def test_track_usage_accumulates(self):
        """Two calls: the session total sums both, and each call's own figure starts from zero.

        Re-framed (realtime_applications audit 2026-09-10, LLM-7): this used to assert that a second
        POST OVERWROTE the first, without the per-call reset in between -- which is the defect when the
        two POSTs belong to one call. The reset between calls is what `generate()` does."""
        p = _provider()
        p._track_provider_specific_usage({"cost": 0.01})
        p._reset_per_call_state()
        p._track_provider_specific_usage({"cost": 0.02})
        assert p.total_actual_cost_usd == pytest.approx(0.03)
        assert p.last_actual_cost_usd == pytest.approx(0.02)

    def test_a_reissued_post_within_one_call_is_counted_too(self):
        """LLM-7: `generate()` re-issues without `response_format` (or with a repaired parameter), and
        both POSTs are billed. The call's own cost is their sum, not the second alone."""
        p = _provider()
        p._reset_per_call_state()
        p._track_provider_specific_usage({"cost": 0.01})
        p._track_provider_specific_usage({"cost": 0.01})
        assert p.last_actual_cost_usd == pytest.approx(0.02)

    def test_a_post_that_reports_no_cost_does_not_erase_an_earlier_one(self):
        p = _provider()
        p._reset_per_call_state()
        p._track_provider_specific_usage({"cost": 0.01})
        p._track_provider_specific_usage({"prompt_tokens": 10})
        assert p.last_actual_cost_usd == pytest.approx(0.01)

    def test_a_reissued_posts_tokens_are_summed_and_cache_hits_are_carried(self):
        """LLM-7 / LLM-5: `_last_usage` sums the call's POSTs and carries `cached_tokens`."""
        p = _provider()
        p._reset_per_call_state()
        p._record_usage({"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 80}})
        p._record_usage({"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 80}})
        assert p._last_usage["input_tokens"] == 200
        assert p._last_usage["cached_tokens"] == 160
        assert p._last_usage["output_tokens"] >= 30

    def test_track_usage_missing_cost_no_op(self):
        p = _provider()
        p._track_provider_specific_usage({"prompt_tokens": 10})
        assert p.total_actual_cost_usd == 0.0

    # B14 dedup: cache-write tokens covered by integration
    # test_usage_cost_recorded_after_generate which asserts the same totals.

    def test_track_usage_cached_tokens_fallback(self):
        # OR's modern field is prompt_tokens_details.cached_tokens; the legacy
        # prompt_cache_hit_tokens field should still take precedence if present.
        # Regression test (2026-07-21 audit): total_cache_hit_tokens accumulation moved to the
        # shared base class's _record_usage (which now also falls back to
        # prompt_tokens_details.cached_tokens for every OpenAI-compatible provider, not just OR) --
        # _track_provider_specific_usage alone only updates last_cache_hit_tokens, to avoid double
        # counting when both the base and this override see the same "cached_tokens" field.
        p = _provider()
        p._track_provider_specific_usage({"prompt_tokens_details": {"cached_tokens": 50}})
        assert p.total_cache_hit_tokens == 0
        assert p.last_cache_hit_tokens == 50
        p._record_usage({"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 50}})
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

    # B12 dedup: per-token to per-1M conversion covered through the
    # public-API path test_input_cost_per_1m_uses_catalogue below.

    def test_per_token_cost_handles_malformed_pricing(self):
        openrouter_module._MODELS_CATALOGUE = {"junk/model": {"id": "junk/model", "pricing": {"prompt": "abc"}}}
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


class TestModelsCatalogueTTL:
    """Regression (2026-07-21 audit round 2, MEDIUM): the catalogue used to be fetched once and
    cached indefinitely -- pricing/context-limit data went stale forever in a long-running
    process. It now expires after ``_MODELS_CATALOGUE_TTL_SECONDS`` (default 300s, override via
    ``PYUTILZ_OR_CATALOGUE_TTL_SECONDS``) and re-fetches automatically."""

    def teardown_method(self):
        openrouter_module._MODELS_CATALOGUE = None
        openrouter_module._MODELS_CATALOGUE_FETCHED_AT = float("inf")

    def _fake_fetch_response(self, model_id="fresh/model"):
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"data": [{"id": model_id, "pricing": {"prompt": "0"}}]}
        fake_resp.raise_for_status = MagicMock()
        return fake_resp

    def test_within_ttl_does_not_refetch(self):
        with patch("pyutilz.llm.openrouter_provider.httpx.get", return_value=self._fake_fetch_response()) as mock_get:
            _fetch_models_catalogue()  # real fetch #1, stamps _MODELS_CATALOGUE_FETCHED_AT
            _fetch_models_catalogue()  # within TTL -- must be a cache hit, no second HTTP call
        assert mock_get.call_count == 1

    def test_past_ttl_triggers_refetch(self):
        with patch("pyutilz.llm.openrouter_provider.httpx.get", return_value=self._fake_fetch_response()) as mock_get:
            _fetch_models_catalogue()
        assert mock_get.call_count == 1

        # Simulate TTL expiry without sleeping.
        openrouter_module._MODELS_CATALOGUE_FETCHED_AT -= openrouter_module._MODELS_CATALOGUE_TTL_SECONDS + 1.0

        with patch("pyutilz.llm.openrouter_provider.httpx.get", return_value=self._fake_fetch_response("fresh-2/model")) as mock_get2:
            cat = _fetch_models_catalogue()
        assert mock_get2.call_count == 1
        assert "fresh-2/model" in cat

    def test_manually_injected_catalogue_treated_as_fresh(self):
        """A catalogue set directly (tests, or a caller pre-populating the cache) without going
        through ``_fetch_models_catalogue`` must NOT be treated as stale just because
        ``_MODELS_CATALOGUE_FETCHED_AT`` was never stamped -- this preserves every existing
        test's ``openrouter_module._MODELS_CATALOGUE = {...}`` pattern."""
        openrouter_module._MODELS_CATALOGUE = {"injected/model": {"id": "injected/model"}}
        with patch("pyutilz.llm.openrouter_provider.httpx.get") as mock_get:
            cat = _fetch_models_catalogue()
        mock_get.assert_not_called()
        assert cat == {"injected/model": {"id": "injected/model"}}

    def test_ttl_env_var_override(self):
        """Verify ``_MODELS_CATALOGUE_TTL_SECONDS`` reads ``PYUTILZ_OR_CATALOGUE_TTL_SECONDS``
        at import time. Checked in a FRESH SUBPROCESS rather than ``importlib.reload`` in this
        process -- reloading ``openrouter_provider`` (a package facade that defines the
        ``OpenRouterProvider`` class other tests already imported) would split that class's
        identity from what's bound elsewhere in this test session."""
        import os as _os
        import pathlib
        import subprocess
        import sys as _sys

        worktree_src = str(pathlib.Path(__file__).resolve().parent.parent / "src")
        env = dict(_os.environ)
        env["PYTHONPATH"] = worktree_src + _os.pathsep + env.get("PYTHONPATH", "")
        env["PYUTILZ_OR_CATALOGUE_TTL_SECONDS"] = "5"
        result = subprocess.run(
            [_sys.executable, "-c", "import pyutilz.llm.openrouter_provider as m; print(m._MODELS_CATALOGUE_TTL_SECONDS)"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "5.0"


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


class TestListModelsGroupByProvider:
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
            "openai/gpt-3.5-turbo": {
                "id": "openai/gpt-3.5-turbo",
                "context_length": 16000,
                "pricing": {"prompt": "0.0000005", "completion": "0.0000015"},
            },
            "anthropic/claude-sonnet-4.6": {
                "id": "anthropic/claude-sonnet-4.6",
                "context_length": 200000,
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            },
            "anthropic/claude-haiku-4.5": {
                "id": "anthropic/claude-haiku-4.5",
                "context_length": 200000,
                "pricing": {"prompt": "0.0000008", "completion": "0.000004"},
            },
            "mistralai/mistral-large": {
                "id": "mistralai/mistral-large",
                "context_length": 128000,
                "pricing": {"prompt": "0.000002", "completion": "0.000006"},
            },
        }

    def teardown_method(self):
        openrouter_module._MODELS_CATALOGUE = None

    def test_zero_disables_grouping_same_as_none(self):
        all_rows = list_openrouter_models(return_only_healthy=False)
        rows_zero = list_openrouter_models(return_only_healthy=False, group_by_provider=0)
        rows_none = list_openrouter_models(return_only_healthy=False, group_by_provider=None)
        assert len(rows_zero) == len(all_rows) == 6
        assert len(rows_none) == len(all_rows) == 6

    def test_top_one_per_provider_by_output_price_descending_default(self):
        # Default group_sort_by="output_price", group_sort_desc=True: the PRICIEST model per
        # provider survives - exactly the "priciest model still under some cap" use case this
        # parameter exists for.
        rows = list_openrouter_models(return_only_healthy=False, group_by_provider=1)
        ids = {r["id"] for r in rows}
        assert ids == {"openai/gpt-4o", "anthropic/claude-sonnet-4.6", "mistralai/mistral-large"}

    def test_top_one_per_provider_cheapest_when_desc_false(self):
        rows = list_openrouter_models(
            return_only_healthy=False, group_by_provider=1, group_sort_by="output_price", group_sort_desc=False,
        )
        ids = {r["id"] for r in rows}
        assert ids == {"openai/gpt-4o-mini", "anthropic/claude-haiku-4.5", "mistralai/mistral-large"}

    def test_top_two_per_provider_keeps_the_two_priciest_openai_models(self):
        rows = list_openrouter_models(return_only_healthy=False, group_by_provider=2)
        openai_ids = {r["id"] for r in rows if r["id"].startswith("openai/")}
        # gpt-4o (priciest) and gpt-3.5-turbo (2nd priciest) survive; gpt-4o-mini (cheapest) does not.
        assert openai_ids == {"openai/gpt-4o", "openai/gpt-3.5-turbo"}

    def test_group_cap_larger_than_provider_size_keeps_everything_for_that_provider(self):
        # anthropic only has 2 models - group_by_provider=5 must not error or drop any of them.
        rows = list_openrouter_models(return_only_healthy=False, group_by_provider=5)
        anthropic_ids = {r["id"] for r in rows if r["id"].startswith("anthropic/")}
        assert anthropic_ids == {"anthropic/claude-sonnet-4.6", "anthropic/claude-haiku-4.5"}

    def test_grouping_composes_with_max_output_filter_and_final_sort(self):
        # The documented use case: priciest-but-still-under-a-cap model per provider, final list
        # sorted by output_price so the result reads cheapest-to-priciest overall.
        rows = list_openrouter_models(
            return_only_healthy=False,
            max_output_per_1m=10.0,  # excludes claude-sonnet-4.6 (15.0) before grouping ever runs
            group_by_provider=1,
            sort_by="output_price",
        )
        assert [r["id"] for r in rows] == [
            "anthropic/claude-haiku-4.5",
            "mistralai/mistral-large",
            "openai/gpt-4o",
        ]

    def test_group_by_provider_ignored_field_falls_back_like_sort_by(self):
        # An unrecognised group_sort_by falls back to the "name" behaviour _metric_raw's own
        # dict.get default provides - documents the contract rather than raising.
        rows = list_openrouter_models(return_only_healthy=False, group_by_provider=1, group_sort_by="nonsense_field")
        assert len(rows) == 3  # one per provider, no crash


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
        openrouter_module._MODELS_CATALOGUE = {"vendor/m": {"id": "vendor/m", "context_length": 128_000}}
        ctx, max_out = _resolve_model_limits("vendor/m")
        assert ctx == 128_000
        assert max_out is None

    def test_resolve_handles_missing_context_length(self):
        openrouter_module._MODELS_CATALOGUE = {"vendor/m": {"id": "vendor/m", "top_provider": {}}}
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
        """An unknown ceiling is None, so max_tokens is omitted: no number of ours is sent as a cap."""
        openrouter_module._MODELS_CATALOGUE = {}
        p = _provider(model="vendor/unknown")
        assert p.max_output_tokens is None


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
        using a 0-1 ``min_uptime`` work consistently.

        The shape is decided ONCE PER PAYLOAD, not per value: at 1.0 the two shapes are
        indistinguishable, and guessing per value read a genuine 1% uptime as 100% and kept
        routing traffic to a nearly-dead backend. So in a payload that carries 99.806, the
        0.95 below is 0.95%, not 95%."""
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
        # Same payload, same shape: 0.95 here means 0.95%, not 95%.
        assert b["uptime_30m"] == pytest.approx(0.0095)
        # Best aggregate is now in fraction-space.
        assert 0.99 < out["best_uptime_30m"] <= 1.0


class TestNormalizeUptime:
    def test_fraction_returned_unchanged(self):
        # is_percentage=False is what _uptime_payload_is_percentage returns for a payload whose
        # every uptime value is <= 1.0.
        assert _normalize_uptime(0.998, is_percentage=False) == pytest.approx(0.998)
        assert _normalize_uptime(0.5, is_percentage=False) == pytest.approx(0.5)
        assert _normalize_uptime(1.0, is_percentage=False) == pytest.approx(1.0)

    def test_one_point_zero_follows_the_payload_shape(self):
        """1.0 is the single ambiguous value; the payload's shape, not a per-value guess, decides."""
        assert _normalize_uptime(1.0, is_percentage=True) == pytest.approx(0.01)
        assert _normalize_uptime(1.0, is_percentage=False) == pytest.approx(1.0)

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
        # The module imports get_llm_settings at the top level; patch the
        # IMPORTED reference (not the source module) so the override is
        # effective. Otherwise an .env file with OPENROUTER_API_KEY set
        # leaks through (which is exactly what just happened on the dev box).
        with patch(
            "pyutilz.llm.openrouter_provider.get_llm_settings",
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

            def _fake_get(url, timeout=None):  # unused-ok: fake keeps the real API signature (timeout accepted, nothing to wait on in-memory)
                prefix = "https://openrouter.ai/api/v1/models/"
                assert url.startswith(prefix), url
                assert url.endswith("/endpoints"), url
                model_id = url[len(prefix) : -len("/endpoints")]
                resp = MagicMock()
                resp.json.return_value = {"data": {"id": model_id, "endpoints": test_self._make_endpoints_response(model_id)}}
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
            list_openrouter_models(return_only_healthy=True, api_key="explicit-key")  # pragma: allowlist secret
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

            def _flaky_get(url, timeout=None):  # unused-ok: fake keeps the real API signature (timeout accepted, nothing to wait on in-memory)
                if "broken" in url:
                    raise httpx.ConnectError("boom")
                prefix = "https://openrouter.ai/api/v1/models/"
                model_id = url[len(prefix) : -len("/endpoints")]
                resp = MagicMock()
                resp.json.return_value = {"data": {"id": model_id, "endpoints": test_self._make_endpoints_response(model_id)}}
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


def test_missing_api_key_error_names_pass_api_key_fallback():
    mock_settings = MagicMock()
    mock_settings.openrouter_api_key = None
    with patch("pyutilz.llm.openrouter_provider.get_llm_settings", return_value=mock_settings):
        with pytest.raises(ValueError, match=r"pass api_key="):
            OpenRouterProvider(api_key=None)
