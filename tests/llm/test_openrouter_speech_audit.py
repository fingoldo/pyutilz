"""Regression tests for the 2026-10-17 TTS synthesis audit (one test per finding), pyutilz side. No network."""

from __future__ import annotations

import json
import threading
import time
from email.utils import formatdate
from typing import Any, List

import httpx
import pytest

from pyutilz.llm import openrouter_speech as osp
from pyutilz.llm.openrouter_speech import OpenRouterSpeech, SpeechError, SpeechInputError, SpeechRequest

MODEL = "elevenlabs/eleven-flash-v2.5"
TURBO = "elevenlabs/eleven-v4-turbo"
AUDIO = b"ID3" + b"\x01" * 100


@pytest.fixture(autouse=True)
def _fresh_catalogue() -> Any:
    osp.reset_speech_catalogue_cache()
    yield
    osp.reset_speech_catalogue_cache()


def _catalogue(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": MODEL, "pricing": {"prompt": "0.00002"}}, {"id": TURBO, "pricing": {"prompt": "0.00002"}}]})


def _client(speech: Any) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _catalogue(request)
        return speech(request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=AUDIO, headers={"content-type": "audio/mpeg"})


def _req(text: str = "привет", model: str = MODEL, **kw: Any) -> SpeechRequest:
    return SpeechRequest(text=text, model=model, voice="matilda", **kw)


def _speech(responses: List[Any], **kw: Any) -> OpenRouterSpeech:
    queue = list(responses)
    return OpenRouterSpeech("key", client=_client(lambda r: queue.pop(0)(r)), sleep=kw.pop("sleep", lambda s: None), **kw)


def _json_200(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"error": {"code": 500, "message": "upstream"}}, headers={"content-type": "application/json"})


# E01, C09: a 200 whose body is not audio is not stored as audio


def test_e01_json_error_envelope_with_status_200_is_not_stored_as_audio() -> None:
    s = _speech([_json_200, _ok], max_retries=2)
    assert s.synthesize(_req()).audio == AUDIO
    assert s.accounting.snapshot()["errors_by_kind"] == {"bad_body": 1}


def test_e01_non_mp3_bytes_with_status_200_are_refused_after_retries() -> None:
    s = _speech([lambda r: httpx.Response(200, content=b"<html>nope</html>")] * 2, max_retries=1)
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert info.value.kind == "bad_body" and info.value.retryable


# E13: a body shorter than its declared length is a truncated read


def test_e13_body_shorter_than_content_length_is_truncated_and_retried() -> None:
    short = lambda r: httpx.Response(200, content=AUDIO[:20], headers={"content-length": str(len(AUDIO))})  # noqa: E731
    s = _speech([short, _ok], max_retries=2)
    assert s.synthesize(_req()).audio == AUDIO
    assert s.accounting.snapshot()["errors_by_kind"] == {"truncated": 1}


# E05, E10, E17, E18: Retry-After handling


def test_e05_huge_retry_after_is_capped() -> None:
    sleeps: List[float] = []
    s = _speech([lambda r: httpx.Response(429, headers={"retry-after": "86400"}), _ok], sleep=sleeps.append, max_retries=2)
    s.synthesize(_req())
    assert osp.MAX_RETRY_AFTER_S <= sleeps[0] <= osp.MAX_RETRY_AFTER_S * (1 + osp.RETRY_AFTER_JITTER)


def test_e10_infinite_retry_after_does_not_overflow() -> None:
    sleeps: List[float] = []
    s = _speech([lambda r: httpx.Response(429, headers={"retry-after": "inf"}), _ok], sleep=sleeps.append, max_retries=2)
    s.synthesize(_req())
    assert osp.MAX_RETRY_AFTER_S <= sleeps[0] <= osp.MAX_RETRY_AFTER_S * (1 + osp.RETRY_AFTER_JITTER)


def test_e17_zero_retry_after_still_waits_a_minimum() -> None:
    sleeps: List[float] = []
    s = _speech([lambda r: httpx.Response(429, headers={"retry-after": "0"}), _ok], sleep=sleeps.append, max_retries=2)
    s.synthesize(_req())
    assert osp.MIN_RETRY_DELAY_S <= sleeps[0] <= osp.MIN_RETRY_DELAY_S * (1 + osp.RETRY_AFTER_JITTER)


def test_e18_retry_after_as_http_date_is_honoured() -> None:
    sleeps: List[float] = []
    when = formatdate(time.time() + 30, usegmt=True)  # wall clock: the date is 30 s ahead, rounded to whole seconds
    s = _speech([lambda r: httpx.Response(429, headers={"retry-after": when}), _ok], sleep=sleeps.append, max_retries=2)
    s.synthesize(_req())
    assert 28 <= sleeps[0] <= 36


def test_e18_http_date_parses_to_exact_seconds_against_a_fixed_clock() -> None:
    when = formatdate(1_000_030, usegmt=True)
    assert osp._retry_after(httpx.Response(429, headers={"retry-after": when}), now=1_000_000.0) == 30.0


def test_retry_after_forms_parse_to_seconds() -> None:
    assert osp._retry_after(httpx.Response(429, headers={"retry-after": "3"})) == 3.0
    assert osp._retry_after(httpx.Response(429, headers={"retry-after": "nan"})) is None
    assert osp._retry_after(httpx.Response(429, headers={"retry-after": "soon"})) is None


# E06, E07, E19: status classification


def test_e06_402_is_not_retried() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(402)

    with pytest.raises(SpeechError) as info:
        _speech([handler]).synthesize(_req())
    assert state["n"] == 1 and not info.value.retryable and info.value.kind == "http_402"


@pytest.mark.parametrize("status", [408, 425])
def test_e07_408_and_425_are_retried(status: int) -> None:
    s = _speech([lambda r, code=status: httpx.Response(code), _ok], max_retries=2)
    assert s.synthesize(_req()).audio == AUDIO


def test_e19_redirect_is_grouped_as_3xx_and_not_retried() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(302, headers={"location": "https://example.invalid/"})

    with pytest.raises(SpeechError) as info:
        _speech([handler]).synthesize(_req())
    assert state["n"] == 1 and info.value.kind == "http_3xx"


# E12: only transport errors are retried; a programming error propagates


def test_e12_programming_error_in_the_client_propagates_instead_of_being_retried() -> None:
    class Broken:
        def stream(self, *args: object, **kwargs: object) -> None:
            raise TypeError("bad argument")

        def close(self) -> None:
            return None

    s = OpenRouterSpeech("key", client=Broken(), sleep=lambda x: None)
    with pytest.raises(TypeError):
        s.synthesize(_req())


# E14: a timeout after the upload may have been billed


def test_e14_timeout_is_counted_as_possibly_billed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    s = _speech([handler, handler], max_retries=1)
    with pytest.raises(SpeechError):
        s.synthesize(_req())
    assert s.accounting.snapshot()["possibly_billed_attempts"] == 2


# M-06: upload bytes count every attempt


def test_m06_upload_bytes_are_counted_for_failed_attempts_too() -> None:
    s = _speech([lambda r: httpx.Response(404)], max_retries=0)
    with pytest.raises(SpeechError):
        s.synthesize(_req("привет"))
    expected = len(json.dumps(_req("привет").body(), ensure_ascii=False).encode("utf-8"))
    assert s.accounting.snapshot()["request_bytes"] == expected


# E15, C13: a failed catalogue fetch is not repeated on every call


def test_c13_e15_failed_catalogue_fetch_is_not_repeated_on_every_call() -> None:
    hits: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    for _ in range(5):
        assert osp.price_per_char(MODEL, client) is None
    assert len(hits) == 1


# E20: a price of zero is a price; E21: no httpx means an unknown price


def test_e20_zero_price_is_a_price_not_unknown() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": MODEL, "pricing": {"prompt": "0"}}]})))
    assert osp.price_per_char(MODEL, client) == 0.0


def test_e21_missing_httpx_gives_unknown_price_not_an_attribute_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(osp, "httpx", None)
    assert osp._fetch_speech_catalogue(None) is None


# C04, C05, C06, C07: request validation and body shape


def test_c05_default_speed_is_not_sent_to_models_that_reject_it() -> None:
    body = SpeechRequest(text="x", model=TURBO, voice="v", speed=1.0, voice_settings={"speed": 1.0}).body()
    assert "speed" not in json.dumps(body)


def test_c06_speed_inside_voice_settings_is_range_checked() -> None:
    with pytest.raises(SpeechInputError):
        SpeechRequest(text="x", model=MODEL, voice="v", voice_settings={"speed": 3.0}).validate()
    with pytest.raises(SpeechInputError):
        SpeechRequest(text="x", model=TURBO, voice="v", voice_settings={"speed": 0.9}).validate()


@pytest.mark.parametrize("settings", [{"stability": 1.5}, {"style": -0.1}, {"similarity_boost": "high"}])
def test_c07_out_of_range_voice_settings_rejected(settings: dict) -> None:
    with pytest.raises(SpeechInputError):
        SpeechRequest(text="x", model=MODEL, voice="v", voice_settings=settings).validate()


def test_c07_unknown_text_normalization_rejected() -> None:
    with pytest.raises(SpeechInputError):
        SpeechRequest(text="x", model=MODEL, voice="v", apply_text_normalization="maybe").validate()


def test_c04_elevenlabs_options_refused_for_other_models() -> None:
    with pytest.raises(SpeechInputError):
        SpeechRequest(text="x", model="openai/tts-1", voice="v", seed=1).validate()


# E03, E11, C12: cancellation


def test_cancel_before_send_makes_no_request() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return _ok(request)

    cancel = threading.Event()
    cancel.set()
    s = OpenRouterSpeech("key", client=_client(handler), cancel=cancel)
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert state["n"] == 0 and info.value.kind == "cancelled"


def test_cancel_ends_a_pending_retry_wait() -> None:
    waits: List[Any] = []

    class RecordingEvent(threading.Event):
        def wait(self, timeout: Any = None) -> bool:  # records the requested wait; the timer below ends it early
            waits.append(timeout)
            return super().wait(timeout)

    cancel = RecordingEvent()
    queue: List[Any] = [lambda r: httpx.Response(429, headers={"retry-after": "60"}), _ok]
    timer = threading.Timer(0.05, cancel.set)  # the retry would wait 60 s; cancellation must cut that short
    s = OpenRouterSpeech("key", client=_client(lambda r: queue.pop(0)(r)), max_retries=3, cancel=cancel)
    timer.start()
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    timer.cancel()
    assert info.value.kind == "cancelled" and len(queue) == 1
    assert len(waits) == 1 and waits[0] >= 60


def test_cancel_is_checked_between_attempts_not_only_at_start() -> None:
    cancel = threading.Event()
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        cancel.set()
        return httpx.Response(503)

    s = OpenRouterSpeech("key", client=_client(handler), sleep=lambda x: None, max_retries=5, cancel=cancel)
    with pytest.raises(SpeechError):
        s.synthesize(_req())
    assert state["n"] == 1
