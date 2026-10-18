"""OpenRouter text-to-speech client: request body, validation, retries, accounting and thread-safety (no network)."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

import httpx
import pytest

from pyutilz.llm import openrouter_speech as osp
from pyutilz.llm.openrouter_speech import (
    OpenRouterSpeech,
    SpeechAccounting,
    SpeechError,
    SpeechInputError,
    SpeechRequest,
)

MODEL = "elevenlabs/eleven-flash-v2.5"
PRICE = 0.00002  # USD per character, as the catalogue lists it
AUDIO = b"ID3" + b"\x01" * 100


@pytest.fixture(autouse=True)
def _fresh_catalogue() -> Any:
    osp.reset_speech_catalogue_cache()
    yield
    osp.reset_speech_catalogue_cache()


seen_params: List[Any] = []


def _client(speech: Callable[[httpx.Request], httpx.Response], models: Any = None) -> httpx.Client:
    listing = models if models is not None else {"data": [{"id": MODEL, "pricing": {"prompt": str(PRICE), "completion": "0"}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            seen_params.append(request.url.params.get("output_modalities"))  # checked by the test, not inside the handler
            return httpx.Response(200, json=listing)
        return speech(request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=AUDIO, headers={"content-type": "audio/mpeg", "x-generation-id": "gen-1"})


def _req(text: str = "привет", **kw: Any) -> SpeechRequest:
    return SpeechRequest(text=text, model=MODEL, voice="rachel", **kw)


def _speech(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> OpenRouterSpeech:
    return OpenRouterSpeech("key", client=_client(handler), sleep=kw.pop("sleep", lambda s: None), **kw)


def test_body_core_fields_only_when_no_options() -> None:
    assert _req().body() == {"model": MODEL, "input": "привет", "voice": "rachel", "response_format": "mp3"}


def test_body_puts_elevenlabs_options_under_provider_and_speed_in_voice_settings() -> None:
    body = _req(speed=0.9, language_code="ru", seed=7, voice_settings={"stability": 0.5}, apply_text_normalization="off").body()
    options = body["provider"]["options"]["elevenlabs"]
    assert options == {"seed": 7, "language_code": "ru", "apply_text_normalization": "off", "voice_settings": {"stability": 0.5, "speed": 0.9}}


def test_billed_chars_counts_unicode_code_points_not_bytes() -> None:
    assert _req("ქართული").billed_chars == 7
    assert len("ქართული".encode()) == 21


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_text_rejected_before_any_call(text: str) -> None:
    calls: List[Any] = []
    s = _speech(lambda r: calls.append(r) or _ok(r))
    with pytest.raises(SpeechInputError):
        s.synthesize(_req(text))
    assert calls == [] and s.accounting.calls == 0


def test_model_character_cap_enforced() -> None:
    SpeechRequest(text="a" * 5000, model="elevenlabs/eleven-v3", voice="v").validate()
    with pytest.raises(SpeechInputError, match="5000"):
        SpeechRequest(text="a" * 5001, model="elevenlabs/eleven-v3", voice="v").validate()


def test_unknown_model_is_not_capped() -> None:
    SpeechRequest(text="a" * 100_000, model="other/tts", voice="v").validate()
    with pytest.raises(SpeechInputError):  # the same text is refused for a model with a known cap
        SpeechRequest(text="a" * 100_000, model="elevenlabs/eleven-v3", voice="v").validate()


def test_v4_rejects_non_default_speed_but_allows_default() -> None:
    SpeechRequest(text="x", model="elevenlabs/eleven-v4", voice="v", speed=1.0).validate()
    with pytest.raises(SpeechInputError):
        SpeechRequest(text="x", model="elevenlabs/eleven-v4", voice="v", speed=1.1).validate()


@pytest.mark.parametrize("speed", [0.69, 1.21])
def test_speed_outside_range_rejected(speed: float) -> None:
    with pytest.raises(SpeechInputError):
        _req(speed=speed).validate()


def test_unsupported_response_format_rejected() -> None:
    with pytest.raises(SpeechInputError):
        _req(response_format="flac").validate()


def test_params_hash_ignores_text_and_neighbours_but_tracks_voice_and_settings() -> None:
    base = _req("один").params_hash()
    assert _req("два", previous_text="a", next_text="b").params_hash() == base
    assert SpeechRequest(text="один", model=MODEL, voice="adam").params_hash() != base
    assert _req("один", speed=0.8).params_hash() != base
    assert _req("один", response_format="pcm").params_hash() != base


def test_success_returns_audio_cost_and_ids() -> None:
    s = _speech(_ok)
    result = s.synthesize(_req("hello"))
    assert result.audio == AUDIO and result.nbytes == len(AUDIO)
    assert result.chars == 5 and result.cost_usd == pytest.approx(5 * PRICE)
    assert result.generation_id == "gen-1" and result.content_type == "audio/mpeg"


def test_request_goes_to_audio_speech_with_json_body_and_utf8() -> None:
    seen: List[httpx.Request] = []
    s = _speech(lambda r: seen.append(r) or _ok(r))
    s.synthesize(_req("привет"))
    assert seen[0].url.path == "/api/v1/audio/speech"
    assert json.loads(seen[0].content.decode("utf-8"))["input"] == "привет"
    assert "привет".encode() in seen[0].content  # not \u-escaped


def test_accounting_totals_chars_bytes_cost_and_request_bytes() -> None:
    s = _speech(_ok)
    s.synthesize(_req("ab"))
    s.synthesize(_req("cde"))
    snap = s.accounting.snapshot()
    assert (snap["calls"], snap["chars"], snap["audio_bytes"]) == (2, 5, 2 * len(AUDIO))
    upload = sum(len(json.dumps(_req(x).body(), ensure_ascii=False).encode("utf-8")) for x in ("ab", "cde"))
    assert snap["cost_usd"] == pytest.approx(5 * PRICE) and snap["request_bytes"] == upload and snap["failures"] == 0


def test_unpriced_model_reports_unknown_cost_not_zero() -> None:
    s = OpenRouterSpeech("key", client=_client(_ok, models={"data": []}), sleep=lambda s: None)
    result = s.synthesize(_req("abc"))
    assert result.cost_usd is None
    snap = s.accounting.snapshot()
    assert snap["unpriced_calls"] == 1 and snap["cost_usd"] == 0.0


def test_catalogue_outage_yields_unknown_price_not_crash() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500) if request.url.path.endswith("/models") else _ok(request)

    s = OpenRouterSpeech("key", client=httpx.Client(transport=httpx.MockTransport(boom)), sleep=lambda s: None)
    assert s.synthesize(_req("abc")).cost_usd is None


def test_catalogue_is_requested_for_speech_models_only() -> None:
    seen_params.clear()
    osp.price_per_char(MODEL, _client(lambda r: httpx.Response(500), models={"data": []}))
    assert seen_params == ["speech"]


def test_price_per_char_reads_prompt_price_and_caches() -> None:
    hits: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        return httpx.Response(200, json={"data": [{"id": MODEL, "pricing": {"prompt": "0.00004"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert osp.price_per_char(MODEL, client) == pytest.approx(4e-5)
    assert osp.price_per_char(MODEL, client) == pytest.approx(4e-5)
    assert osp.price_per_char("nope/none", client) is None
    assert len(hits) == 1


def test_429_retried_honouring_retry_after_then_succeeds() -> None:
    sleeps: List[float] = []
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(429, headers={"retry-after": "7"}) if state["n"] == 1 else _ok(request)

    s = _speech(handler, sleep=sleeps.append)
    assert s.synthesize(_req()).audio == AUDIO
    assert 7.0 <= sleeps[0] <= 7.0 * (1 + osp.RETRY_AFTER_JITTER) and state["n"] == 2
    assert s.accounting.calls == 1  # the failed attempt is not a billed call


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(status, text="nope")

    s = _speech(handler)
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert state["n"] == 1 and info.value.status == status and not info.value.retryable
    assert s.accounting.snapshot()["failures"] == 1 and s.accounting.calls == 0


def test_server_errors_exhaust_bounded_retries_then_raise() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(503)

    s = _speech(handler, max_retries=3)
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert state["n"] == 4 and info.value.retryable
    assert s.accounting.snapshot()["failures"] == 1


def test_empty_200_body_is_treated_as_transient() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(200, content=b"") if state["n"] == 1 else _ok(request)

    assert _speech(handler).synthesize(_req()).audio == AUDIO and state["n"] == 2


def test_transport_error_is_retried() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("reset")
        return _ok(request)

    assert _speech(handler).synthesize(_req()).audio == AUDIO and state["n"] == 2


def test_backoff_grows_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(osp.random, "random", lambda: 0.5)  # jitter multiplier 1.0: the sequence is exact
    sleeps: List[float] = []
    s = _speech(lambda r: httpx.Response(500), sleep=sleeps.append, max_retries=3, backoff_s=2.0)
    with pytest.raises(SpeechError):
        s.synthesize(_req())
    assert sleeps == [2.0, 4.0, 8.0]


def test_cache_hit_recorded_separately_from_calls() -> None:
    acc = SpeechAccounting()
    acc.record_cache_hit(chars=4, nbytes=900)
    snap = acc.snapshot()
    assert (snap["cache_hits"], snap["cached_chars"], snap["cached_bytes"], snap["calls"], snap["chars"]) == (1, 4, 900, 0, 0)


def test_accounting_is_exact_under_concurrent_threads() -> None:
    s = _speech(_ok)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda i: s.synthesize(_req("x" * (i % 7 + 1))), range(200)))
    snap = s.accounting.snapshot()
    expected = sum(i % 7 + 1 for i in range(200))
    assert snap["calls"] == 200 and snap["chars"] == expected and snap["audio_bytes"] == 200 * len(AUDIO)
    assert snap["cost_usd"] == pytest.approx(expected * PRICE)


def test_client_is_shared_safely_across_threads() -> None:
    seen: set = set()
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            seen.add(threading.get_ident())
        return _ok(request)

    s = _speech(handler)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: s.synthesize(_req(f"w{i}")), range(64)))
    assert s.accounting.calls == 64
    assert len(seen) > 1  # the eight workers really ran on several threads through one client


def test_missing_api_key_rejected() -> None:
    with pytest.raises(ValueError):
        OpenRouterSpeech("")


def test_context_manager_closes_owned_client_only() -> None:
    shared = _client(_ok)
    with OpenRouterSpeech("k", client=shared):
        pass
    assert not shared.is_closed


def test_owned_client_is_closed_on_exit() -> None:
    owned = OpenRouterSpeech("k")
    inner = owned._client
    with owned:
        assert not inner.is_closed
    assert inner.is_closed


def test_retries_and_error_groups_are_counted() -> None:
    responses = [
        lambda r: httpx.Response(429),
        lambda r: httpx.Response(503),
        lambda r: httpx.Response(200, content=b""),  # empty body counts as empty_body
        _ok,
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)(request)

    s = _speech(handler, max_retries=5)
    assert s.synthesize(_req()).audio == AUDIO
    snap = s.accounting.snapshot()
    assert snap["retries"] == 3
    assert snap["errors_by_kind"] == {"http_429": 1, "http_5xx": 1, "empty_body": 1}


@pytest.mark.parametrize(
    ("status", "transport", "kind"),
    [(0, True, "transport"), (200, False, "empty_body"), (429, False, "http_429"), (402, False, "http_402"), (503, False, "http_5xx"), (404, False, "http_4xx")],
)
def test_error_kind_groups(status: int, transport: bool, kind: str) -> None:
    assert osp.error_kind(status, transport) == kind


def test_terminal_failure_carries_its_kind() -> None:
    s = _speech(lambda r: httpx.Response(404, text="nope"))
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert info.value.kind == "http_4xx"


def test_fetch_credits_reads_total_usage_and_remaining() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": {"total_credits": 10.0, "total_usage": 2.5}})))
    assert osp.fetch_credits("k", client) == {"total_credits": 10.0, "total_usage": 2.5, "remaining": 7.5}


def test_fetch_credits_failure_is_none_not_an_exception() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert osp.fetch_credits("k", client) is None
