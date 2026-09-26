"""Regression tests for audits/2026-09-26/10_openrouter_coverage.md (OR-1 .. OR-20).

Each test drives the real code path with the wire shape OpenRouter documents, and asserts on the exact field or
exception, so reverting a fix fails a test rather than just changing a count.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")

import pyutilz.llm.openai_compat as openai_compat_mod
import pyutilz.llm.openrouter_provider as openrouter_module
from pyutilz.llm._openai_compat_http import (
    LLMStreamInterruptedError,
    _is_retryable_http_error,
    error_code_status,
    merge_extra_body,
    wait_honoring_retry_after,
)
from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openrouter_provider import OpenRouterProvider
from pyutilz.llm.openrouter_provider._request import auto_require_parameters, catalogue_id

MODEL = "anthropic/claude-x"
FALLBACK = "openai/gpt-y"

CATALOGUE: dict[str, dict[str, Any]] = {
    MODEL: {
        "id": MODEL,
        "supported_parameters": ["max_tokens", "temperature", "response_format", "structured_outputs", "reasoning", "tools", "tool_choice", "seed"],
        "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000002", "input_cache_write": "0.0000025"},
        "top_provider": {"context_length": 200000, "max_completion_tokens": 64000},
    },
    FALLBACK: {
        "id": FALLBACK,
        "supported_parameters": ["max_tokens", "response_format"],
        "pricing": {"prompt": "0.000001", "completion": "0.000004"},
        "top_provider": {"context_length": 128000, "max_completion_tokens": 16000},
    },
    "openai/o-reasoner": {"id": "openai/o-reasoner", "supported_parameters": ["max_tokens", "response_format", "reasoning"], "pricing": {}},
}


@pytest.fixture(autouse=True)
def _catalogue():
    openrouter_module._MODELS_CATALOGUE = dict(CATALOGUE)
    openrouter_module._MODELS_CATALOGUE_FETCHED_AT = math.inf
    yield
    openrouter_module._MODELS_CATALOGUE = None


def _or(**kw: Any) -> OpenRouterProvider:
    settings = MagicMock()
    settings.openrouter_api_key = None
    with patch("pyutilz.llm.openrouter_provider.get_llm_settings", return_value=settings):
        return OpenRouterProvider(**{"api_key": "k", "model": MODEL, **kw})


def _body(p: OpenRouterProvider, **kw: Any) -> dict[str, Any]:
    args = {"prompt": "q", "system": "sys", "temperature": 0.2, "max_tokens": 100, "json_mode": False, "thinking": None, "json_schema": None}
    args.update(kw)
    return p._build_stream_body(**args)


def _resp(payload: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers or {}, request=httpx.Request("POST", "https://x/chat/completions"))


class _Stream:
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._lines = lines
        self.request = httpx.Request("POST", "https://x/chat/completions")

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> Any:
        for line in self._lines:
            yield line

    async def __aenter__(self) -> "_Stream":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}"


async def _drain(p: OpenRouterProvider, **kw: Any) -> str:
    out = []
    async for piece in p.generate_stream("q", max_tokens=100, **kw):
        out.append(piece)
    return "".join(out)


# ── OR-1 / OR-20: mid-stream errors and string error codes ─────────────────────────────────────────────────────────


def test_or1_documented_mid_stream_error_chunk_raises_with_partial_text() -> None:
    p = _or()
    p._client = MagicMock()
    lines = [
        _data({"id": "gen-1", "choices": [{"index": 0, "delta": {"content": "Hello, wor"}}]}),
        # verbatim shape from https://openrouter.ai/docs/api/reference/streaming
        _data(
            {
                "id": "gen-1",
                "error": {"code": "server_error", "message": "Provider disconnected unexpectedly"},
                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
            }
        ),
        "data: [DONE]",
    ]
    p._client.stream = MagicMock(return_value=_Stream(lines))
    with pytest.raises(LLMStreamInterruptedError) as info:
        asyncio.run(_drain(p))
    assert info.value.partial_text == "Hello, wor"
    assert info.value.code == "server_error" and info.value.retryable is True
    assert p._client.stream.call_count == 1  # generated already: never re-opened


def test_or1_bare_finish_reason_error_in_stream_raises() -> None:
    p = _or()
    p._client = MagicMock()
    lines = [_data({"choices": [{"delta": {"content": "part"}}]}), _data({"choices": [{"delta": {}, "finish_reason": "error"}]}), "data: [DONE]"]
    p._client.stream = MagicMock(return_value=_Stream(lines))
    with pytest.raises(LLMStreamInterruptedError) as info:
        asyncio.run(_drain(p))
    assert info.value.partial_text == "part"


def test_or1_buffered_finish_reason_error_raises_retryable() -> None:
    p = _or()
    p._client = AsyncMock()
    p._client.post = AsyncMock(return_value=_resp({"choices": [{"message": {"content": "half"}, "finish_reason": "error"}]}))
    with pytest.raises(LLMStreamInterruptedError) as info:
        asyncio.run(p._post_and_unwrap({"model": MODEL, "messages": []}))
    assert info.value.partial_text == "half"
    assert _is_retryable_http_error(info.value) is True


def test_or1_buffered_error_with_choices_is_no_longer_ignored() -> None:
    p = _or()
    p._client = AsyncMock()
    payload = {"error": {"code": "server_error", "message": "gone"}, "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    p._client.post = AsyncMock(return_value=_resp(payload))
    with pytest.raises(LLMStreamInterruptedError):
        asyncio.run(p._post_and_unwrap({"model": MODEL, "messages": []}))


@pytest.mark.parametrize("code,status", [("server_error", 502), ("rate_limit_exceeded", 429), ("timeout", 504), ("503", 503), (429, 429), ("bogus", 0), (True, 0)])
def test_or20_error_code_status(code: Any, status: int) -> None:
    assert error_code_status(code) == status


def test_or20_string_code_in_200_body_is_retryable_http_error() -> None:
    p = _or()
    p._client = AsyncMock()
    p._client.post = AsyncMock(return_value=_resp({"error": {"code": "server_error", "message": "upstream died"}}))
    with pytest.raises(httpx.HTTPStatusError) as info:
        asyncio.run(p._post_and_unwrap({"model": MODEL, "messages": []}))
    assert info.value.response.status_code == 502 and _is_retryable_http_error(info.value)


def test_or20_unknown_string_code_is_not_retried() -> None:
    p = _or()
    p._client = AsyncMock()
    p._client.post = AsyncMock(return_value=_resp({"error": {"code": "moderation", "message": "flagged"}}))
    with pytest.raises(LLMProviderError) as info:
        asyncio.run(p._post_and_unwrap({"model": MODEL, "messages": []}))
    assert not isinstance(info.value, LLMStreamInterruptedError) and not _is_retryable_http_error(info.value)


# ── OR-2 / OR-3: routing knobs and require_parameters ──────────────────────────────────────────────────────────────


def test_or3_all_routing_knobs_reach_the_provider_block() -> None:
    p = _or(
        provider_only=("anthropic",),
        provider_order=("anthropic", "google-vertex"),
        provider_data_collection="deny",
        provider_zdr=True,
        provider_quantizations=("fp8", "bf16"),
        provider_max_price=(("prompt", 1), ("completion", 2)),
        provider_preferred_min_throughput=(("p90", 50),),
        provider_preferred_max_latency=3,
        provider_require_parameters=False,
    )
    prov = _body(p)["provider"]
    assert prov == {
        "order": ["anthropic", "google-vertex"],
        "only": ["anthropic"],
        "data_collection": "deny",
        "zdr": True,
        "quantizations": ["fp8", "bf16"],
        "max_price": {"prompt": 1, "completion": 2},
        "preferred_min_throughput": {"p90": 50},
        "preferred_max_latency": 3,
    }
    hash(tuple(sorted({"provider_max_price": (("prompt", 1),), "provider_only": ("a",)}.items())))  # factory cache key stays hashable


def test_or3_bad_data_collection_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="data_collection"):
        _or(provider_data_collection="denied")


def test_or2_auto_require_parameters_on_json_schema() -> None:
    body = _body(_or(), json_schema={"name": "s", "schema": {"type": "object"}})
    assert body["provider"]["require_parameters"] is True


def test_or2_auto_require_parameters_on_reasoning_effort_but_not_on_disable() -> None:
    p = _or()
    assert _body(p, thinking="high")["provider"]["require_parameters"] is True
    assert "provider" not in _body(p, thinking=False)
    assert "provider" not in _body(p)  # plain text call: nothing to guarantee


def test_or2_auto_mode_skips_when_catalogue_lacks_a_sent_parameter() -> None:
    # o-reasoner lists no `temperature`: requiring every parameter would leave no endpoint (404).
    body = _body(_or(model="openai/o-reasoner"), json_mode=True)
    assert "provider" not in body
    assert auto_require_parameters({"response_format": {"type": "json_object"}, "max_tokens": 1}, CATALOGUE["openai/o-reasoner"]) is True
    assert auto_require_parameters({"response_format": {"type": "json_object"}}, None) is False


def test_or2_explicit_settings_win() -> None:
    assert "provider" not in _body(_or(provider_require_parameters=False), json_mode=True)
    assert _body(_or(provider_require_parameters=True))["provider"] == {"require_parameters": True}


# ── OR-4: streamed web-search citations ────────────────────────────────────────────────────────────────────────────


def test_or4_citations_from_stream_delta_annotations_accumulate() -> None:
    p = _or(enable_web_search=True)
    p._client = MagicMock()
    ann = lambda u: {"type": "url_citation", "url_citation": {"url": u, "title": u}}  # noqa: E731
    lines = [
        _data({"choices": [{"delta": {"content": "a", "annotations": [ann("https://a")]}}]}),
        _data({"choices": [{"delta": {"content": "b", "annotations": [ann("https://b"), {"type": "other"}]}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    p._client.stream = MagicMock(return_value=_Stream(lines))

    async def run() -> list:
        await _drain(p)
        return list(p.last_web_search_citations)

    assert [c["url"] for c in asyncio.run(run())] == ["https://a", "https://b"]


# ── OR-5: per-call fields summed across POSTs ──────────────────────────────────────────────────────────────────────


def test_or5_second_post_adds_to_and_never_erases_per_call_fields() -> None:
    p = _or()
    first = {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "cost": 0.01,
        "cost_details": {"upstream_inference_cost": 0.008},
        "prompt_tokens_details": {"cache_write_tokens": 40, "cached_tokens": 10, "audio_tokens": 3},
        "cache_discount": -0.001,
    }
    second = {"prompt_tokens": 50, "completion_tokens": 5, "cost": 0.02, "prompt_tokens_details": {"cached_tokens": 7}}
    p._track_provider_specific_usage(first)
    p._track_provider_specific_usage(second)
    assert p.last_actual_cost_usd == pytest.approx(0.03)
    assert p.last_upstream_inference_cost_usd == pytest.approx(0.008)  # was reset to None by the second POST
    assert p.last_cache_write_tokens == 40
    assert p.last_cache_hit_tokens == 17
    assert p.last_audio_tokens == 3
    assert p.last_cache_discount_usd == pytest.approx(-0.001)


# ── OR-6 / OR-7: estimate by served model, cache writes at the write rate ──────────────────────────────────────────


def test_or6_cache_writes_priced_at_the_write_rate() -> None:
    p = _or()
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "prompt_tokens_details": {"cache_write_tokens": 1_000_000}}
    p._record_usage(usage)
    cost = p.get_session_cost()
    assert cost["input_cost_usd"] == pytest.approx(2.5)  # 1M write tokens at $2.5/M, not the $2/M input rate


def test_or7_usage_priced_by_the_model_that_served_it() -> None:
    p = _or(models_fallback=(FALLBACK,))
    p.last_upstream_model = FALLBACK
    p._record_usage({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    cost = p.get_session_cost()
    assert cost["input_cost_usd"] == pytest.approx(1.0) and cost["output_cost_usd"] == pytest.approx(4.0)


def test_or7_buffered_path_reads_the_served_model_before_recording_usage() -> None:
    p = _or(models_fallback=(FALLBACK,))
    p._client = AsyncMock()
    payload = {"model": FALLBACK, "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    p._client.post = AsyncMock(return_value=_resp(payload))
    asyncio.run(p._post_and_unwrap({"model": MODEL, "messages": []}))
    assert p._usage_by_model[FALLBACK][0] == 1_000_000
    assert p.get_session_cost()["input_cost_usd"] == pytest.approx(1.0)


def test_or7_unlisted_snapshot_id_prices_as_the_requested_model() -> None:
    p = _or()
    p.last_upstream_model = MODEL + "-20260101"
    p._record_usage({"prompt_tokens": 1_000_000, "completion_tokens": 0})
    assert p.get_session_cost()["input_cost_usd"] == pytest.approx(2.0)


def test_or7_limits_are_the_minimum_across_the_fallback_list() -> None:
    p = _or(models_fallback=(FALLBACK,))
    assert p.max_output_tokens == 16000 and p.context_window == 128000
    assert _or().max_output_tokens == 64000


def test_online_suffix_uses_the_base_catalogue_entry() -> None:
    assert catalogue_id(MODEL + ":online") == MODEL and catalogue_id("x/y:free") == "x/y:free"
    assert _or(model=MODEL + ":online").max_output_tokens == 64000


# ── OR-8 / OR-13 / OR-14 / OR-15 / OR-11: request shaping ─────────────────────────────────────────────────────────


def test_or8_system_prompt_as_cache_breakpoint() -> None:
    msgs = _body(_or(system_cache_control=True))["messages"]
    assert msgs[0] == {"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]}
    assert _body(_or())["messages"][0] == {"role": "system", "content": "sys"}


def test_or13_14_transforms_and_plugins() -> None:
    body = _body(_or(transforms=("middle-out",), enable_web_search=True, web_search_engine="exa", web_search_max_results=3, pdf_engine="mistral-ocr"))
    assert body["transforms"] == ["middle-out"]
    assert body["plugins"] == [{"id": "web", "engine": "exa", "max_results": 3}, {"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}]
    assert "plugins" not in _body(_or()) and "transforms" not in _body(_or())


def test_or15_extra_body_constructor_and_per_call_merge_last() -> None:
    p = _or(extra_body=(("seed", 7), ("user", "u-1")), provider_zdr=True, provider_require_parameters=False)
    body = _body(p, extra_body={"seed": 9, "stop": ["END"], "provider": {"only": ["anthropic"]}})
    assert body["seed"] == 9 and body["user"] == "u-1" and body["stop"] == ["END"]
    assert body["provider"] == {"zdr": True, "only": ["anthropic"]}  # merged one level, not replaced
    assert merge_extra_body({"a": 1}, (("a", 2),)) == {"a": 2}


def test_or11_tools_are_sent_and_tool_calls_read_back() -> None:
    p = _or()
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    p._client = AsyncMock()
    p._client.post = AsyncMock(return_value=_resp({"choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}]}))

    async def run() -> tuple[str, list]:
        text = await p.generate("q", max_tokens=50, tools=tools, tool_choice="auto")
        return text, list(p.last_tool_calls)

    text, calls = asyncio.run(run())
    sent = p._client.post.call_args.kwargs["json"]
    assert sent["tools"] == tools and sent["tool_choice"] == "auto"
    assert text == "" and calls == [call]


# ── OR-9: reasoning budget ─────────────────────────────────────────────────────────────────────────────────────────


def test_or9_int_thinking_is_a_reasoning_max_tokens_budget() -> None:
    p = _or(provider_require_parameters=False)
    assert _body(p, thinking=1500)["reasoning"] == {"max_tokens": 1500}
    # "none" normalises to "disabled", which maps to the measured real off switch, not to effort "none".
    assert _body(p, thinking="none")["reasoning"] == {"enabled": False}
    assert _body(p, thinking=0)["reasoning"] == {"enabled": False}
    assert _body(p, thinking=True)["reasoning"] == {"effort": "medium"}


# Recorded 2026-09-26 (audits/2026-09-26/or9_live_results.json): the refusal every mandatory endpoint returned to both
# `enabled: false` and `effort: "none"`.
_MANDATORY_REFUSAL = "Reasoning is mandatory for this endpoint and cannot be disabled."


def test_or9_catalogue_mandatory_model_gets_minimal_effort_up_front() -> None:
    openrouter_module._MODELS_CATALOGUE["z-ai/glm-5.3-flash"] = {
        "id": "z-ai/glm-5.3-flash",
        "supported_parameters": ["max_tokens", "reasoning"],
        "reasoning": {"mandatory": True, "default_enabled": True, "supported_efforts": ["max", "high", "low"], "default_effort": "max"},
        "pricing": {},
    }
    p = _or(model="z-ai/glm-5.3-flash", provider_require_parameters=False)
    assert _body(p, thinking=False)["reasoning"] == {"effort": "minimal", "exclude": True}
    # The control: deepseek-v4.1-flash is `mandatory: false` and accepted `enabled: false` live.
    assert _body(_or(provider_require_parameters=False), thinking=False)["reasoning"] == {"enabled": False}


@pytest.mark.parametrize("sent", [{"enabled": False}, {"effort": "none"}])
def test_or9_refused_off_switch_is_repaired_to_minimal(sent: dict) -> None:
    p = _or(model="openai/gpt-oss-20b-unlisted")
    repaired = p._body_after_rejected_request({"model": "m", "reasoning": sent}, 400, _MANDATORY_REFUSAL)
    assert repaired is not None and repaired["reasoning"] == {"effort": "minimal", "exclude": True}
    assert p._body_after_rejected_request({"reasoning": {"effort": "low"}}, 400, _MANDATORY_REFUSAL) is None
    openrouter_module._provider._REASONING_CANNOT_BE_DISABLED.discard("openai/gpt-oss-20b-unlisted")


def test_or9_recorded_live_results_support_the_fallback() -> None:
    data = json.loads((Path(__file__).resolve().parents[2] / "audits" / "2026-09-26" / "or9_live_results.json").read_text(encoding="utf-8"))
    mandatory = {r["model"] for r in data["results"] if r["variant"] == "enabled_false" and r["http_status"] == 400}
    assert len(mandatory) >= 3
    for r in data["results"]:
        if r["model"] in mandatory:
            assert (r["http_status"] == 400) == (r["variant"] in ("effort_none", "enabled_false")), r


# ── OR-10: strict json_schema ──────────────────────────────────────────────────────────────────────────────────────


def test_or10_strict_defaults_true_and_explicit_false_is_reported() -> None:
    p = _or(provider_require_parameters=False)
    schema = {"name": "s", "schema": {"type": "object"}}
    body = _body(p, json_schema=schema)
    assert body["response_format"]["json_schema"]["strict"] is True and p.last_json_schema_applied is True
    assert "strict" not in schema  # the caller's dict is not mutated
    body = _body(p, json_schema={**schema, "strict": False})
    assert body["response_format"]["json_schema"]["strict"] is False and p.last_json_schema_applied is False


# ── OR-12: stream_options stays (other providers need it) ──────────────────────────────────────────────────────────


def test_or12_stream_options_still_sent() -> None:
    assert _body(_or())["stream_options"] == {"include_usage": True}


# ── OR-16: Retry-After honoured on the buffered path ───────────────────────────────────────────────────────────────


def _state(exc: BaseException) -> Any:
    outcome = MagicMock()
    outcome.failed = True
    outcome.exception.return_value = exc
    return MagicMock(outcome=outcome)


def test_or16_buffered_wait_takes_the_longer_retry_after_on_429_only() -> None:
    req = httpx.Request("POST", "https://x")
    wait = wait_honoring_retry_after(lambda _s: 5.0)
    r429 = httpx.HTTPStatusError("rl", request=req, response=httpx.Response(429, headers={"retry-after": "42"}, request=req))
    r503 = httpx.HTTPStatusError("x", request=req, response=httpx.Response(503, headers={"retry-after": "42"}, request=req))
    short = httpx.HTTPStatusError("rl", request=req, response=httpx.Response(429, headers={"retry-after": "1"}, request=req))
    assert wait(_state(r429)) == 42.0
    assert wait(_state(r503)) == 5.0
    assert wait(_state(short)) == 5.0


def test_or16_generate_retry_decorator_uses_the_retry_after_wait() -> None:
    wait = openai_compat_mod.OpenAICompatibleProvider.generate.retry.wait  # type: ignore[attr-defined]
    assert wait.__qualname__.startswith("wait_honoring_retry_after")


# ── OR-17 / OR-18: batch ───────────────────────────────────────────────────────────────────────────────────────────


def test_or18_batch_reasoning_from_any_field_and_truncation_not_ok() -> None:
    from pyutilz.llm.openrouter_batch import parse_result_item

    item = {
        "custom_id": "a",
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": "cut", "reasoning_details": [{"type": "reasoning.text", "text": "why"}]}, "finish_reason": "length"}]},
        },
    }
    res = parse_result_item(item)
    assert res.reasoning == "why" and res.text == "cut"
    assert res.truncated is True and res.ok is False


def test_or17_listed_metadata_hash_is_not_trusted(tmp_path: Path) -> None:
    from pyutilz.llm.openrouter_batch import BatchRequest, OpenRouterBatchClient

    client = OpenRouterBatchClient(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    client.list_batches = MagicMock(  # type: ignore[method-assign]
        return_value=[{"id": "other", "model": "m", "created_at": 10_000.0, "metadata": {"request_hash": "h"}}]
    )
    # A listed hash equal to ours but created far outside the window is NOT adopted: submit never sends metadata.
    assert client.recover_submit("m", [BatchRequest("a", {"messages": []})], {"submitting_at": 1.0, "request_hash": "h"}) is None
    client.close()


# ── OR-19: decisions retries ───────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", [408, 504])
def test_or19_decisions_retry_408_and_504(status: int) -> None:
    from pyutilz.llm.openrouter_decisions import OpenRouterDecisionsError

    assert OpenRouterDecisionsError("x", status_code=status).retryable is True
    assert OpenRouterDecisionsError("x", status_code=400).retryable is False


def test_or19_decisions_wait_honours_retry_after() -> None:
    from pyutilz.llm.openrouter_decisions import OpenRouterDecisionsError, _check_response, _wait_honoring_retry_after

    resp = httpx.Response(429, json={"error": {"code": 429, "message": "slow"}}, headers={"retry-after": "30"}, request=httpx.Request("POST", "https://x"))
    with pytest.raises(OpenRouterDecisionsError) as info:
        _check_response(resp)
    assert info.value.retry_after_s == 30.0
    wait = _wait_honoring_retry_after(lambda _s: 2.0)
    assert wait(_state(info.value)) == 30.0
    assert wait(_state(OpenRouterDecisionsError("x", status_code=503))) == 2.0


def test_resolve_pricing_carries_the_catalogue_cache_write_rate(monkeypatch):
    """``_resolve_pricing`` exposes ``pricing.input_cache_write`` as ``Pricing.cache_write`` (None when unpublished)."""
    from pyutilz.llm.openrouter_provider import _catalogue, _provider

    catalogue = {
        "anthropic/claude-x": {"pricing": {"prompt": "0.000003", "completion": "0.000015", "input_cache_read": "0.0000003", "input_cache_write": "0.00000375"}},
        "plain/model": {"pricing": {"prompt": "0.000001", "completion": "0.000002"}},
    }
    monkeypatch.setattr(_catalogue, "_fetch_models_catalogue", lambda *a, **k: catalogue)
    provider = _provider.OpenRouterProvider.__new__(_provider.OpenRouterProvider)
    written = provider._resolve_pricing("anthropic/claude-x")
    assert written.cache_write == pytest.approx(3.75)
    assert written.input == pytest.approx(3.0)
    assert provider._resolve_pricing("plain/model").cache_write is None
