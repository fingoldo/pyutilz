"""Path, guard, log and threshold pins for the OpenRouter endpoints mixin (mutation survivors)."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from pyutilz.llm.openrouter_provider._endpoints import OpenRouterEndpointsMixin

BASE = "https://openrouter.ai/api/v1"


class _Host(OpenRouterEndpointsMixin):
    def __init__(self, handler=None, model_name: str = "m1", last_generation_id: str | None = None) -> None:
        self.model_name = model_name
        self.last_generation_id = last_generation_id
        self._client = httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(handler or (lambda r: httpx.Response(200, json={}))))


def _json_handler(body, seen: list, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    return handler


def _run(coro):
    return asyncio.run(coro)


class TestFetchModelParameters:
    def test_the_default_model_goes_to_the_parameters_path(self):
        seen = []
        host = _Host(_json_handler({"data": {"temperature": 1}}, seen))
        assert _run(host.fetch_model_parameters()) == {"temperature": 1}
        assert seen[0].url.path == "/api/v1/parameters/m1"

    def test_an_explicit_model_overrides_the_default(self):
        seen = []
        host = _Host(_json_handler({"data": {}}, seen))
        _run(host.fetch_model_parameters("other/model"))
        assert seen[0].url.path == "/api/v1/parameters/other/model"

    def test_an_http_error_is_raised(self):
        host = _Host(_json_handler({}, [], status=500))
        with pytest.raises(httpx.HTTPStatusError):
            _run(host.fetch_model_parameters())

    def test_a_non_dict_payload_gives_an_empty_dict(self):
        host = _Host(_json_handler([1, 2], []))
        assert _run(host.fetch_model_parameters()) == {}

    def test_a_missing_or_non_dict_data_gives_an_empty_dict(self):
        host = _Host(_json_handler({"data": [1]}, []))
        assert _run(host.fetch_model_parameters()) == {}


class TestCheckModelHealth:
    @staticmethod
    def _body():
        return {"data": {"name": "Upstream", "endpoints": [{"uptime_last_30m": 0.995}]}}

    def test_the_default_model_goes_to_the_endpoints_path(self):
        seen = []
        host = _Host(_json_handler(self._body(), seen))
        _run(host.check_model_health())
        assert seen[0].url.path == "/api/v1/models/m1/endpoints"

    def test_the_summary_carries_the_model_and_the_name(self):
        host = _Host(_json_handler(self._body(), []))
        out = _run(host.check_model_health("x/y"))
        assert out["model"] == "x/y"
        assert out["name"] == "Upstream"

    def test_the_endpoint_list_is_summarised(self):
        host = _Host(_json_handler(self._body(), []))
        assert _run(host.check_model_health()).get("best_uptime_30m") == 0.995

    def test_a_payload_without_data_still_summarises_to_none(self):
        host = _Host(_json_handler({}, []))
        out = _run(host.check_model_health())
        assert out["name"] is None
        assert out["best_uptime_30m"] is None

    def test_an_http_error_is_raised(self):
        host = _Host(_json_handler({}, [], status=500))
        with pytest.raises(httpx.HTTPStatusError):
            _run(host.check_model_health())


class TestIsModelHealthy:
    @staticmethod
    def _with_uptime(host, uptime):
        async def fake(model=None):
            return {"best_uptime_30m": uptime}

        host.check_model_health = fake
        return host

    def test_the_default_threshold_is_ninety_nine_hundredths(self):
        host = self._with_uptime(_Host(), 0.995)
        assert _run(host.is_model_healthy()) is True

    def test_an_uptime_below_the_default_threshold_is_unhealthy(self):
        host = self._with_uptime(_Host(), 0.985)
        assert _run(host.is_model_healthy()) is False

    def test_an_uptime_exactly_at_the_threshold_is_healthy(self):
        host = self._with_uptime(_Host(), 0.99)
        assert _run(host.is_model_healthy()) is True

    def test_a_lower_threshold_is_honoured(self):
        host = self._with_uptime(_Host(), 0.96)
        assert _run(host.is_model_healthy(min_uptime=0.95)) is True

    def test_a_missing_uptime_is_unhealthy(self):
        host = _Host()

        async def fake(model=None):
            return {}

        host.check_model_health = fake
        assert _run(host.is_model_healthy()) is False

    def test_a_failed_health_check_is_unhealthy_and_logged(self, caplog):
        host = _Host()

        async def boom(model=None):
            raise RuntimeError("boom")

        host.check_model_health = boom
        caplog.set_level(logging.WARNING)
        assert _run(host.is_model_healthy()) is False
        assert "is_model_healthy: health check failed (boom)" in caplog.text
        assert caplog.records[-1].name == "pyutilz.llm.openrouter_provider._provider"


class TestFetchGenerationStats:
    def test_the_last_generation_id_is_used_when_none_is_passed(self):
        seen = []
        host = _Host(_json_handler({"data": {"id": "gen-9"}}, seen), last_generation_id="gen-9")
        _run(host.fetch_generation_stats())
        assert seen[0].url.params["id"] == "gen-9"

    def test_an_explicit_id_is_sent_under_the_id_parameter(self):
        seen = []
        host = _Host(_json_handler({"data": {}}, seen))
        _run(host.fetch_generation_stats("gen-1"))
        assert seen[0].url.path == "/api/v1/generation"
        assert seen[0].url.params["id"] == "gen-1"

    def test_no_id_at_all_is_refused_with_the_exact_message(self):
        host = _Host()
        with pytest.raises(ValueError) as info:
            _run(host.fetch_generation_stats())
        assert str(info.value) == "No generation_id passed and self.last_generation_id is unset — call generate() first or pass a known ID."

    def test_an_http_error_is_raised(self):
        host = _Host(_json_handler({}, [], status=500))
        with pytest.raises(httpx.HTTPStatusError):
            _run(host.fetch_generation_stats("gen-1"))

    def test_the_data_object_is_returned(self):
        host = _Host(_json_handler({"data": {"id": "x"}}, []))
        assert _run(host.fetch_generation_stats("gen-1")) == {"id": "x"}

    def test_a_payload_without_data_is_returned_whole(self):
        host = _Host(_json_handler({"a": 1}, []))
        assert _run(host.fetch_generation_stats("gen-1")) == {"a": 1}

    def test_a_non_dict_payload_gives_an_empty_dict(self):
        host = _Host(_json_handler([1], []))
        assert _run(host.fetch_generation_stats("gen-1")) == {}

    def test_a_non_dict_data_gives_an_empty_dict(self):
        host = _Host(_json_handler({"data": [1]}, []))
        assert _run(host.fetch_generation_stats("gen-1")) == {}


class TestGetAccountCredits:
    @staticmethod
    def _host(credits_payload, key=None, key_fails: bool = False):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/credits"):
                return httpx.Response(200, json={"data": credits_payload})
            return httpx.Response(200, json={"data": key or {}})

        host = _Host(handler)
        if key_fails:

            async def boom():
                raise RuntimeError("down")

            host.check_account_limits = boom
        return host

    def test_a_balance_is_granted_minus_used(self):
        host = self._host({"total_credits": 10, "total_usage": 4}, key={"is_free_tier": False})
        out = _run(host.get_account_credits())
        assert out["balance_usd"] == 6.0
        assert out["is_available"] is True

    def test_a_missing_usage_leaves_the_balance_unknown(self):
        host = self._host({"total_credits": 10}, key={"is_free_tier": False})
        out = _run(host.get_account_credits())
        assert out["balance_usd"] is None
        assert out["is_available"] is False

    def test_a_balance_under_one_dollar_is_still_available(self):
        host = self._host({"total_credits": 1.0, "total_usage": 0.5}, key={"is_free_tier": False})
        assert _run(host.get_account_credits())["is_available"] is True

    def test_a_free_tier_user_is_available_without_a_balance(self):
        host = self._host({}, key={"is_free_tier": True})
        out = _run(host.get_account_credits())
        assert out["is_free_tier"] is True
        assert out["is_available"] is True

    def test_an_http_error_is_raised(self):
        def handler(request):
            return httpx.Response(500)

        with pytest.raises(httpx.HTTPStatusError):
            _run(_Host(handler).get_account_credits())

    def test_a_non_numeric_credit_is_logged_at_debug_and_read_as_unknown(self, caplog):
        host = self._host({"total_credits": "abc", "total_usage": 1}, key={"is_free_tier": False})
        caplog.set_level(logging.DEBUG)
        out = _run(host.get_account_credits())
        assert out["total_granted"] is None
        assert out["balance_usd"] is None
        assert "_to_float: credits value is not numeric: ValueError" in caplog.text

    def test_a_failed_key_lookup_leaves_the_free_tier_flag_unknown_and_is_logged(self, caplog):
        host = self._host({"total_credits": 0, "total_usage": 0}, key_fails=True)
        caplog.set_level(logging.DEBUG)
        out = _run(host.get_account_credits())
        assert out["is_free_tier"] is None
        assert out["is_available"] is False
        assert "OR /key lookup for is_free_tier failed: down" in caplog.text

    def test_a_balance_of_exactly_zero_is_not_available(self):
        host = self._host({"total_credits": 2, "total_usage": 2}, key={"is_free_tier": False})
        assert _run(host.get_account_credits())["is_available"] is False


class TestLoggerName:
    def test_the_module_logger_keeps_the_pre_split_name(self):
        from pyutilz.llm.openrouter_provider import _endpoints

        assert _endpoints.logger.name == "pyutilz.llm.openrouter_provider._provider"
