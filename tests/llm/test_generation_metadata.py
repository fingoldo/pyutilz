"""fetch_generation_metadata: compact storage dict from GET /generation (MockTransport only, no network)."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List

import httpx
import pytest

from pyutilz.llm.generation_metadata import GENERATION_METADATA_KEYS, fetch_generation_metadata

GEN_ID = "gen-tts-1792299598-95fpbNDPILj8SQkhmSHL"
KEY = "sk-or-test-SECRET-KEY-VALUE"

# Copied from audits/2026-10-18_tts_synthesis_round2/generation_metadata_example.md (HTTP 200 body).
EXAMPLE_BODY: Dict[str, Any] = json.loads(
    """
{
  "data": {
    "created_at": "2026-10-18T04:59:58.329Z",
    "model": "elevenlabs/eleven-flash-v2.5-20260930",
    "app_id": null,
    "external_user": null,
    "streamed": false,
    "cancelled": false,
    "latency": 585,
    "moderation_latency": null,
    "generation_time": null,
    "tokens_prompt": 2,
    "tokens_completion": null,
    "native_tokens_prompt": 2,
    "native_tokens_completion": null,
    "native_tokens_completion_images": null,
    "native_tokens_reasoning": null,
    "native_tokens_cached": null,
    "num_media_prompt": null,
    "num_input_audio_prompt": 0,
    "num_media_completion": null,
    "num_search_results": null,
    "num_fetches": null,
    "web_search_engine": null,
    "origin": "",
    "is_byok": false,
    "finish_reason": null,
    "native_finish_reason": null,
    "service_tier": null,
    "usage": 0.00016,
    "router": null,
    "user_agent": null,
    "http_referer": null,
    "request_id": null,
    "session_id": null,
    "preset_id": null,
    "api_type": "tts",
    "id": "gen-tts-PLACEHOLDER",
    "upstream_id": "UjsPJs8Hi5rdDgzIEc7C",
    "provider_responses": [
      {
        "endpoint_id": "89a816f6-e958-4b95-b96c-290fad3fc245",
        "id": "UjsPJs8Hi5rdDgzIEc7C",
        "is_byok": false,
        "latency": 585,
        "model_permaslug": "elevenlabs/eleven-flash-v2.5-20260930",
        "provider_name": "ElevenLabs",
        "status": 200
      }
    ],
    "total_cost": 0.00016,
    "cache_discount": null,
    "upstream_inference_cost": 0,
    "provider_name": "ElevenLabs",
    "response_cache_source_id": null,
    "data_region": "global",
    "workspace_id": "f2ec5e06-a656-57cd-a267-784528295b3a"
  }
}
"""
)
EXAMPLE_BODY["data"]["id"] = GEN_ID

EXPECTED_FULL: Dict[str, Any] = {
    "provider_name": "ElevenLabs",
    "model": "elevenlabs/eleven-flash-v2.5-20260930",
    "total_cost": 0.00016,
    "usage": 0.00016,
    "latency": 585,
    "tokens_prompt": 2,
    "api_type": "tts",
    "id": GEN_ID,
    "upstream_id": "UjsPJs8Hi5rdDgzIEc7C",
    "created_at": "2026-10-18T04:59:58.329Z",
    "cancelled": False,
    "streamed": False,
    "is_byok": False,
    "provider_responses": [
        {
            "endpoint_id": "89a816f6-e958-4b95-b96c-290fad3fc245",
            "id": "UjsPJs8Hi5rdDgzIEc7C",
            "is_byok": False,
            "latency": 585,
            "model_permaslug": "elevenlabs/eleven-flash-v2.5-20260930",
            "provider_name": "ElevenLabs",
            "status": 200,
        }
    ],
}


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _json_client(body: Any, status: int = 200) -> httpx.Client:
    return _client(lambda request: httpx.Response(status, json=body))


def _body_with(**overrides: Any) -> Dict[str, Any]:
    body = copy.deepcopy(EXAMPLE_BODY)
    body["data"].update(overrides)
    return body


def test_full_example_gives_expected_dict() -> None:
    assert fetch_generation_metadata(GEN_ID, KEY, _json_client(EXAMPLE_BODY)) == EXPECTED_FULL


def test_request_shape_is_get_generation_with_bearer_key() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=EXAMPLE_BODY)

    fetch_generation_metadata(GEN_ID, KEY, _client(handler))
    assert len(seen) == 1
    req = seen[0]
    assert req.method == "GET"
    assert req.url.path == "/api/v1/generation"
    assert req.url.params["id"] == GEN_ID
    assert req.headers["Authorization"] == f"Bearer {KEY}"


def test_unlisted_keys_are_not_stored() -> None:
    result = fetch_generation_metadata(GEN_ID, KEY, _json_client(EXAMPLE_BODY))
    assert result is not None
    assert set(result) <= set(GENERATION_METADATA_KEYS)
    for key in ("workspace_id", "data_region", "cache_discount", "origin", "native_tokens_prompt", "upstream_inference_cost"):
        assert key not in result


def test_nulls_dropped_at_top_level() -> None:
    body = _body_with(total_cost=None, usage=None, api_type=None)
    result = fetch_generation_metadata(GEN_ID, KEY, _json_client(body))
    assert result is not None
    assert "total_cost" not in result
    assert "usage" not in result
    assert "api_type" not in result
    assert "generation_time" not in result  # null in the example
    assert None not in result.values()


def test_nulls_dropped_inside_provider_responses() -> None:
    entry = {"endpoint_id": "e1", "id": "u1", "is_byok": None, "latency": None, "model_permaslug": "m", "provider_name": "P", "status": None}
    result = fetch_generation_metadata(GEN_ID, KEY, _json_client(_body_with(provider_responses=[entry])))
    assert result is not None
    assert result["provider_responses"] == [{"endpoint_id": "e1", "id": "u1", "model_permaslug": "m", "provider_name": "P"}]


def test_provider_response_extra_keys_are_not_stored() -> None:
    entry = {"endpoint_id": "e1", "id": "u1", "status": 200, "secret_debug": "x"}
    result = fetch_generation_metadata(GEN_ID, KEY, _json_client(_body_with(provider_responses=[entry])))
    assert result is not None
    assert result["provider_responses"] == [{"endpoint_id": "e1", "id": "u1", "status": 200}]


def test_empty_strings_dropped_but_zero_and_false_kept() -> None:
    body = _body_with(finish_reason="", tokens_completion=0, cancelled=False, streamed=False, latency=0, usage=0.0, is_byok=False)
    result = fetch_generation_metadata(GEN_ID, KEY, _json_client(body))
    assert result is not None
    assert "finish_reason" not in result
    assert result["tokens_completion"] == 0
    assert result["cancelled"] is False
    assert result["streamed"] is False
    assert result["latency"] == 0
    assert result["usage"] == 0.0
    assert result["is_byok"] is False


def test_empty_string_dropped_inside_provider_responses_and_zero_kept() -> None:
    entry = {"endpoint_id": "", "id": "u1", "is_byok": False, "latency": 0, "status": 0}
    result = fetch_generation_metadata(GEN_ID, KEY, _json_client(_body_with(provider_responses=[entry])))
    assert result is not None
    assert result["provider_responses"] == [{"id": "u1", "is_byok": False, "latency": 0, "status": 0}]


def test_empty_provider_responses_list_is_kept() -> None:
    result = fetch_generation_metadata(GEN_ID, KEY, _json_client(_body_with(provider_responses=[])))
    assert result is not None
    assert result["provider_responses"] == []


@pytest.mark.parametrize("bad_id", ["", "bad id", "a/b", "gen;drop", "x" * 129, None])
def test_malformed_id_returns_none_without_request(bad_id: Any, caplog: pytest.LogCaptureFixture) -> None:
    calls: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=EXAMPLE_BODY)

    with caplog.at_level(logging.WARNING):
        assert fetch_generation_metadata(bad_id, KEY, _client(handler)) is None
    assert calls == []
    assert any("malformed" in rec.getMessage() for rec in caplog.records)


@pytest.mark.parametrize("status", [404, 500])
def test_http_error_status_returns_none(status: int, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert fetch_generation_metadata(GEN_ID, KEY, _json_client({"error": "nope"}, status=status)) is None
    assert any("lookup failed" in rec.getMessage() for rec in caplog.records)


@pytest.mark.parametrize("body", [{}, {"data": None}, {"error": {"message": "x"}}, {"data": "text"}, ["list"]])
def test_body_without_data_returns_none(body: Any) -> None:
    assert fetch_generation_metadata(GEN_ID, KEY, _json_client(body)) is None


def test_non_json_body_returns_none() -> None:
    assert fetch_generation_metadata(GEN_ID, KEY, _client(lambda r: httpx.Response(200, text="<html>"))) is None


def test_network_exception_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    assert fetch_generation_metadata(GEN_ID, KEY, _client(handler)) is None


def test_provider_name_missing_still_returns_other_fields() -> None:
    body = _body_with(provider_name=None)
    body["data"].pop("provider_name")
    result = fetch_generation_metadata(GEN_ID, KEY, _json_client(body))
    assert result is not None
    assert "provider_name" not in result
    assert result["model"] == "elevenlabs/eleven-flash-v2.5-20260930"
    assert result["total_cost"] == 0.00016
    assert result["provider_responses"][0]["provider_name"] == "ElevenLabs"


def test_api_key_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with caplog.at_level(logging.DEBUG):
        fetch_generation_metadata(GEN_ID, KEY, _client(handler))
        fetch_generation_metadata("bad id", KEY, _client(handler))
    assert KEY not in caplog.text
