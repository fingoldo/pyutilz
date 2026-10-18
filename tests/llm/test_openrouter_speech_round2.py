"""Regression tests for the round-2 TTS synthesis audit, pyutilz side: one or more tests per finding. No network."""

from __future__ import annotations

import copy
import datetime as dt
import gzip
import logging
import pickle
import threading
import time
import zlib
from dataclasses import asdict
from typing import Any, Callable, List, Optional

import httpx
import pytest

import pyutilz.llm as llm
from pyutilz.llm import openrouter_speech as osp
from pyutilz.llm.openrouter_speech import (
    OpenRouterSpeech,
    SpeechAccounting,
    SpeechError,
    SpeechInputError,
    SpeechRequest,
    SpeechResult,
)

MODEL = "elevenlabs/eleven-flash-v2.5"
TURBO = "elevenlabs/eleven-v4-turbo"
PRICE = 0.00002  # USD per character, as the catalogue lists it
AUDIO = b"ID3" + b"\x01" * 100
LISTING = {"data": [{"id": MODEL, "pricing": {"prompt": str(PRICE)}}]}


@pytest.fixture(autouse=True)
def _fresh_catalogue() -> Any:
    osp.reset_speech_catalogue_cache()
    yield
    osp.reset_speech_catalogue_cache()


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=AUDIO, headers={"content-type": "audio/mpeg"})


def _req(text: str = "привет", model: str = MODEL, **kw: Any) -> SpeechRequest:
    return SpeechRequest(text=text, model=model, voice="matilda", **kw)


def _client(speech: Callable[[httpx.Request], httpx.Response], listing: Any = LISTING, hits: Optional[List[str]] = None, delay: float = 0.0) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            if hits is not None:
                hits.append(str(request.url))
            if delay:
                time.sleep(delay)
            return httpx.Response(200, json=listing)
        return speech(request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _speech(responses: List[Any], **kw: Any) -> OpenRouterSpeech:
    queue = list(responses)
    sleep = kw.pop("sleep", lambda s: None)
    return OpenRouterSpeech("key", client=_client(lambda r: queue.pop(0)(r)), sleep=sleep, **kw)


# R2-LIB-1 and R2-FIX-3: a content-encoded 200 is judged on its decoded bytes, not on the encoded Content-Length


def test_r2_lib1_gzip_body_is_not_judged_truncated() -> None:
    gz = gzip.compress(AUDIO)
    headers = {"content-type": "audio/mpeg", "content-encoding": "gzip", "content-length": str(len(gz))}
    s = OpenRouterSpeech("key", client=_client(lambda r: httpx.Response(200, content=gz, headers=headers)), max_retries=0)
    assert s.synthesize(_req()).audio == AUDIO
    assert s.accounting.snapshot()["errors_by_kind"] == {}


def test_r2_fix3_deflate_body_is_not_judged_truncated() -> None:
    zz = zlib.compress(AUDIO)
    headers = {"content-type": "audio/mpeg", "content-encoding": "deflate", "content-length": str(len(zz))}
    s = OpenRouterSpeech("key", client=_client(lambda r: httpx.Response(200, content=zz, headers=headers)), max_retries=0)
    assert s.synthesize(_req()).audio == AUDIO


def test_r2_lib1_identity_encoding_keeps_the_length_check() -> None:
    short = lambda r: httpx.Response(200, content=AUDIO[:20], headers={"content-length": str(len(AUDIO)), "content-encoding": "identity"})  # noqa: E731
    s = _speech([short, _ok], max_retries=2)
    assert s.synthesize(_req()).audio == AUDIO
    assert s.accounting.snapshot()["errors_by_kind"] == {"truncated": 1}


# R2-LIB-2: a Unicode digit in Content-Length is not a length and must not raise out of synthesize


def test_r2_lib2_superscript_digit_content_length_does_not_raise() -> None:
    raw = [(b"content-length", "²".encode()), (b"content-type", b"audio/mpeg")]  # httpx cannot send this header as text
    odd = lambda r: httpx.Response(200, content=AUDIO, headers=raw)  # noqa: E731
    assert _speech([odd], max_retries=0).synthesize(_req()).audio == AUDIO


def test_r2_lib2_declared_length_accepts_ascii_digits_only() -> None:
    assert osp._declared_length("12") == 12
    assert osp._declared_length("²") is None
    assert osp._declared_length("-1") is None
    assert osp._declared_length(None) is None


# R2-LIB-3: an injected client with httpx missing still retries a connection error instead of raising AttributeError


def test_r2_lib3_injected_client_without_httpx_retries_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(osp, "httpx", None)
    queue: List[Any] = [ConnectionError("reset"), _ok]

    def speech(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item(request)

    s = OpenRouterSpeech("key", client=_client(speech), sleep=lambda x: None, max_retries=2)
    assert s.synthesize(_req()).audio == AUDIO
    assert s.accounting.snapshot()["errors_by_kind"] == {"transport": 1}


# R2-LIB-4: cancellation during a retry keeps the last failure; an injected sleep is honoured


def test_r2_lib4_cancel_during_retry_keeps_the_last_failure() -> None:
    cancel = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        cancel.set()
        return httpx.Response(503)

    s = OpenRouterSpeech("key", client=_client(handler), sleep=lambda x: None, max_retries=3, cancel=cancel)
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert info.value.kind == "cancelled" and "HTTP 503" in str(info.value)
    snap = s.accounting.snapshot()
    assert snap["failures"] == 1 and snap["errors_by_kind"] == {"http_5xx": 1}


def test_r2_lib4_injected_sleep_is_used_even_when_a_cancel_event_is_set() -> None:
    queue: List[Any] = [lambda r: httpx.Response(503), _ok]
    sleeps: List[float] = []
    s = OpenRouterSpeech("key", client=_client(lambda r: queue.pop(0)(r)), sleep=sleeps.append, max_retries=2, cancel=threading.Event())
    assert s.synthesize(_req()).audio == AUDIO
    assert len(sleeps) == 1 and sleeps[0] > 0


# R2-LIB-5: Retry-After waits are spread by jitter, never earlier than the server asked


def test_r2_lib5_retry_after_wait_is_jittered_within_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    def wait_for(draw: float) -> float:
        monkeypatch.setattr(osp.random, "random", lambda: draw)
        sleeps: List[float] = []
        queue: List[Any] = [lambda r: httpx.Response(429, headers={"retry-after": "7"}), _ok]
        s = OpenRouterSpeech("key", client=_client(lambda r: queue.pop(0)(r)), sleep=sleeps.append, max_retries=2)
        s.synthesize(_req())
        return sleeps[0]

    assert wait_for(0.0) == pytest.approx(7.0)
    assert wait_for(1.0) == pytest.approx(7.0 * (1 + osp.RETRY_AFTER_JITTER))


# R2-LIB-6: the catalogue follows the client's base URL and is cached per base URL


def test_r2_lib6_catalogue_request_follows_the_base_url() -> None:
    seen: List[str] = []
    queue: List[Any] = [_ok]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=LISTING)
        return queue.pop(0)(request)

    s = OpenRouterSpeech("key", base_url="https://proxy.example/api/v1", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda x: None)
    assert s.synthesize(_req()).cost_usd == pytest.approx(PRICE * len("привет"))
    models = [u for u in seen if "/models" in u]
    assert models and all(u.startswith("https://proxy.example/api/v1/models") for u in models)


def test_r2_lib6_cache_is_keyed_by_base_url() -> None:
    hits: List[str] = []
    client = _client(_ok, hits=hits)
    assert osp.price_per_char(MODEL, client, base_url="https://a.example/api/v1") == PRICE
    assert osp.price_per_char(MODEL, client, base_url="https://b.example/api/v1") == PRICE
    assert len(hits) == 2 and "a.example" in hits[0] and "b.example" in hits[1]


# R2-LIB-7: an empty but valid catalogue is cached like any other


def test_r2_lib7_empty_listing_is_fresh_and_not_refetched() -> None:
    hits: List[str] = []
    client = _client(_ok, listing={"data": []}, hits=hits)
    for _ in range(3):
        assert osp.speech_catalogue(client) == {}
    assert len(hits) == 1


# R2-LIB-8: SpeechAccounting survives asdict, deepcopy and pickle


def test_r2_lib8_accounting_survives_asdict_deepcopy_and_pickle() -> None:
    acc = SpeechAccounting()
    acc.record_upload(5)
    assert asdict(acc)["request_bytes"] == 5
    clone = copy.deepcopy(acc)
    assert clone.snapshot()["request_bytes"] == 5
    revived = pickle.loads(pickle.dumps(acc))
    revived.record_failure()
    clone.record_failure()
    assert revived.failures == 1 and clone.failures == 1 and acc.failures == 0
    assert "_lock" not in acc.snapshot()


# R2-LIB-9: voice settings, seed and language are validated by type, unknown keys are refused


@pytest.mark.parametrize(
    "build",
    [
        lambda: _req(voice_settings={"bogus": 1.0}),
        lambda: _req(voice_settings={"stability": True}),
        lambda: _req(voice_settings={"style": False}),
        lambda: _req(voice_settings={"use_speaker_boost": "yes"}),
        lambda: _req(voice_settings={"speed": True}, model=MODEL),
        lambda: _req(speed=True),
        lambda: _req(seed="5"),
        lambda: _req(seed=True),
        lambda: _req(seed=2**32),
        lambda: _req(language_code=5),
        lambda: _req(language_code="   "),
        lambda: _req(apply_language_text_normalization="yes"),
    ],
)
def test_r2_lib9_ill_typed_options_are_refused_before_any_call(build: Callable[[], SpeechRequest]) -> None:
    with pytest.raises(SpeechInputError):
        build().validate()


def test_r2_lib9_well_typed_options_are_accepted() -> None:
    _req(voice_settings={"stability": 0, "similarity_boost": 1.0, "use_speaker_boost": True}, seed=7, language_code="ru").validate()


# R2-LIB-10: 1 and 1.0 are one setting and share one cache key


def test_r2_lib10_integral_numbers_hash_the_same_as_floats() -> None:
    assert _req(speed=1).params_hash() == _req(speed=1.0).params_hash()
    assert _req(voice_settings={"stability": 0}).params_hash() == _req(voice_settings={"stability": 0.0}).params_hash()
    assert _req(speed=1.1).params_hash() != _req(speed=1.0).params_hash()


# R2-LIB-11 and R2-FIX-11: possibly billed attempts are the ones the request left for; their characters and cost are kept


@pytest.mark.parametrize("exc_type", [httpx.ConnectTimeout, httpx.PoolTimeout])
def test_r2_fix11_connect_and_pool_timeouts_are_not_possibly_billed(exc_type: type) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_type("nope", request=request)

    s = _speech([handler, handler], max_retries=1)
    with pytest.raises(SpeechError):
        s.synthesize(_req())
    snap = s.accounting.snapshot()
    assert snap["possibly_billed_attempts"] == 0 and snap["possibly_billed_chars"] == 0
    assert snap["failures"] == 1 and snap["errors_by_kind"] == {"transport": 2}


@pytest.mark.parametrize("exc_type", [httpx.ReadTimeout, httpx.WriteTimeout])
def test_r2_lib11_read_and_write_timeouts_keep_their_characters_and_cost(exc_type: type) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_type("slow", request=request)

    text = "привет мир"
    s = OpenRouterSpeech("key", client=_client(handler), sleep=lambda x: None, max_retries=0)
    with pytest.raises(SpeechError):
        s.synthesize(_req(text))
    snap = s.accounting.snapshot()
    assert snap["possibly_billed_attempts"] == 1
    assert snap["possibly_billed_chars"] == len(text)
    assert snap["possibly_billed_cost_usd"] == pytest.approx(PRICE * len(text))


# R2-LIB-12: reserved MPEG version and layer patterns are not MP3 frames


@pytest.mark.parametrize("head", [b"\xff\xe8\x00\x00", b"\xff\xe0\x00\x00"])  # version 01 and layer 00 are reserved
def test_r2_lib12_reserved_mpeg_bits_are_refused(head: bytes) -> None:
    assert osp.audio_body_problem(head + b"\x00" * 100, "mp3", None, None) == "bad_body"


def test_r2_lib12_a_real_mpeg_frame_header_is_accepted() -> None:
    assert osp.audio_body_problem(b"\xff\xfb\x90\x00" + b"\x00" * 100, "mp3", None, None) is None


# R2-LIB-13: content types are compared case-insensitively; a PCM body that is markup is refused


@pytest.mark.parametrize("media", ["Application/JSON", "TEXT/HTML", "Text/Plain"])
def test_r2_lib13_content_type_check_ignores_case(media: str) -> None:
    assert osp.audio_body_problem(AUDIO, "mp3", media, None) == "bad_body"


def test_r2_lib13_pcm_refuses_an_even_length_markup_body() -> None:
    body = b"<html><body>error</body></html> "
    assert len(body) % 2 == 0
    assert osp.audio_body_problem(body, "pcm", None, None) == "bad_body"


def test_r2_lib13_pcm_keeps_binary_samples_that_start_with_a_markup_byte() -> None:
    body = b"<" + bytes(range(128, 256)) + b"\x00"
    assert osp.audio_body_problem(body, "pcm", None, None) is None


# R2-LIB-14: a -0000 HTTP date is UTC; an unrepresentable date falls back to backoff


def test_r2_lib14_negative_zero_http_date_is_read_as_utc() -> None:
    base = 1_700_000_000.0
    stamp = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(base + 60)) + " -0000"
    resp = httpx.Response(429, headers={"retry-after": stamp})
    assert osp._retry_after(resp, now=base) == pytest.approx(60, abs=1)


class _Far(dt.datetime):
    def timestamp(self) -> float:  # stands in for a date the platform cannot represent
        raise OverflowError("date out of range")


def test_r2_lib14_unrepresentable_http_date_falls_back_to_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(osp.email.utils, "parsedate_to_datetime", lambda raw: _Far(2100, 1, 1, tzinfo=dt.timezone.utc))
    assert osp._retry_after(httpx.Response(429, headers={"retry-after": "Fri, 01 Jan 2100 00:00:00 GMT"})) is None


# R2-LIB-15: a terminal failure carries its diagnostics


def test_r2_lib15_terminal_error_carries_status_body_retry_after_and_generation_id() -> None:
    s = _speech([lambda r: httpx.Response(429, text="slow down", headers={"retry-after": "3", "x-generation-id": "gen-9"})], max_retries=0)
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert info.value.details == {
        "status": 429,
        "kind": "http_429",
        "body": "slow down",
        "retry_after_s": 3.0,
        "generation_id": "gen-9",
        "content_type": "text/plain; charset=utf-8",
    }


# R2-LIB-16: a total call budget stops retries that would run past it


def test_r2_lib16_call_budget_stops_a_retry_that_would_overrun_it() -> None:
    sleeps: List[float] = []
    s = _speech([lambda r: httpx.Response(429, headers={"retry-after": "60"})] * 6, sleep=sleeps.append, max_retries=5, max_call_s=5.0)
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert "call budget" in str(info.value) and info.value.kind == "http_429"
    assert sleeps == [] and s.accounting.snapshot()["failures"] == 1


def test_r2_lib16_zero_budget_disables_the_deadline() -> None:
    sleeps: List[float] = []
    s = _speech([lambda r: httpx.Response(503), _ok], sleep=sleeps.append, max_retries=2, max_call_s=0)
    assert s.synthesize(_req()).audio == AUDIO and len(sleeps) == 1


# R2-LIB-17: the speech names are re-exported from pyutilz.llm


def test_r2_lib17_speech_names_are_exported_from_pyutilz_llm() -> None:
    for name in ("OpenRouterSpeech", "SpeechRequest", "SpeechResult", "SpeechAccounting", "SpeechError", "SpeechInputError"):
        assert getattr(llm, name) is getattr(osp, name)


# R2-LIB-18: the accounting total has the chat providers' name too


def test_r2_lib18_total_cost_usd_is_an_alias_of_cost_usd() -> None:
    acc = SpeechAccounting()
    acc.record(SpeechResult(audio=AUDIO, request=_req(), chars=5, cost_usd=0.5, latency_s=0.0))
    assert acc.total_cost_usd == acc.cost_usd == 0.5


# R2-LIB-21: __all__ lists every public name of the module and nothing that is not defined there


def test_r2_lib21_all_lists_the_public_names_of_the_module() -> None:
    import inspect

    assert {"BASE_URL", "MODEL_MAX_CHARS", "SPEED_RANGE"} <= set(osp.__all__)
    for name in osp.__all__:
        assert hasattr(osp, name), name
    for name, obj in vars(osp).items():
        if name.startswith("_"):
            continue
        defined_here = (isinstance(obj, type) or inspect.isfunction(obj)) and obj.__module__ == osp.__name__
        if defined_here or name.isupper():
            assert name in osp.__all__, name


# R2-LIB-22 and R2-MONEY-18: one catalogue fetch serves concurrent callers, cold start and expiry alike


def test_r2_lib22_cold_start_fetches_the_catalogue_once() -> None:
    hits: List[str] = []
    client = _client(_ok, hits=hits, delay=0.2)
    barrier = threading.Barrier(8)
    results: List[Optional[float]] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        price = osp.price_per_char(MODEL, client)
        with lock:
            results.append(price)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(hits) == 1 and results == [PRICE] * 8


def test_r2_money18_expiry_refreshes_once_and_serves_the_last_listing_meanwhile(monkeypatch: pytest.MonkeyPatch) -> None:
    hits: List[str] = []
    client = _client(_ok, hits=hits, delay=0.2)
    assert osp.price_per_char(MODEL, client) == PRICE
    monkeypatch.setattr(osp, "_CATALOGUE_TTL_S", 0.0)  # the hour is over
    barrier = threading.Barrier(8)
    results: List[Optional[float]] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        price = osp.price_per_char(MODEL, client)
        with lock:
            results.append(price)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(hits) == 2 and results == [PRICE] * 8


def test_r2_money18_model_nearing_expiry_is_warned_once(caplog: pytest.LogCaptureFixture) -> None:
    soon = (dt.date.today() + dt.timedelta(days=5)).isoformat()
    listing = {"data": [{"id": MODEL, "pricing": {"prompt": str(PRICE)}, "expiration_date": soon}]}
    client = _client(_ok, listing=listing)
    with caplog.at_level(logging.WARNING, logger="pyutilz.llm.openrouter_speech"):
        assert osp.price_per_char(MODEL, client) == PRICE
        assert osp.price_per_char(MODEL, client) == PRICE
    warnings = [r for r in caplog.records if "expires" in r.getMessage()]
    assert len(warnings) == 1 and MODEL in warnings[0].getMessage()


def test_r2_money18_far_expiry_is_not_warned(caplog: pytest.LogCaptureFixture) -> None:
    far = (dt.date.today() + dt.timedelta(days=400)).isoformat()
    listing = {"data": [{"id": MODEL, "pricing": {"prompt": str(PRICE)}, "expiration_date": far}]}
    with caplog.at_level(logging.WARNING, logger="pyutilz.llm.openrouter_speech"):
        assert osp.price_per_char(MODEL, _client(_ok, listing=listing)) == PRICE
    assert not [r for r in caplog.records if "expires" in r.getMessage()]


# R2-SEC-7: a 200 body above the ceiling is refused, not stored, and not read to the end


def test_r2_sec7_oversized_audio_is_refused_without_retry() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(200, content=b"ID3" + b"\x00" * 5000, headers={"content-type": "audio/mpeg"})

    s = OpenRouterSpeech("key", client=_client(handler), sleep=lambda x: None, max_retries=3, max_audio_bytes=1000)
    with pytest.raises(SpeechError) as info:
        s.synthesize(_req())
    assert info.value.kind == "too_large" and not info.value.retryable and state["n"] == 1


def test_r2_sec7_body_at_the_ceiling_is_accepted() -> None:
    s = OpenRouterSpeech("key", client=_client(_ok), sleep=lambda x: None, max_audio_bytes=len(AUDIO))
    assert s.synthesize(_req()).audio == AUDIO


def test_r2_sec7_reading_stops_at_the_ceiling() -> None:
    class Endless(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.yielded = 0

        def __iter__(self) -> Any:
            while True:
                self.yielded += 1
                yield b"\x00" * 1000

        def close(self) -> None:
            return None

    source = Endless()
    s = OpenRouterSpeech("key", client=_client(lambda r: httpx.Response(200, stream=source)), sleep=lambda x: None, max_retries=0, max_audio_bytes=3000)
    with pytest.raises(SpeechError):
        s.synthesize(_req())
    assert source.yielded <= 4


# R2-TEST-24: fetch_credits on malformed bodies is None, not an exception; string numbers are accepted


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": {"total_credits": 5}},
        {"data": {"total_credits": "abc", "total_usage": 1}},
        {"data": {"total_credits": None, "total_usage": 1}},
        [1, 2, 3],
    ],
)
def test_r2_test24_fetch_credits_malformed_body_is_none(body: Any) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    assert osp.fetch_credits("test-key", client=client) is None


def test_r2_test24_fetch_credits_accepts_string_numbers() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": {"total_credits": "10.5", "total_usage": "2.5"}})))
    assert osp.fetch_credits("test-key", client=client) == {"total_credits": 10.5, "total_usage": 2.5, "remaining": 8.0}


def test_r2_api11_generation_lookup_returns_the_serving_endpoint() -> None:
    """the endpoint name that served a request, from the generation id the speech response carried."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": {"provider_name": "ElevenLabs", "model": "elevenlabs/eleven-flash-v2.5"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert osp.fetch_generation_provider("gen-abc_1", "k", client) == {"provider_name": "ElevenLabs", "model": "elevenlabs/eleven-flash-v2.5"}
    assert seen[0].url.path == "/api/v1/generation" and seen[0].url.params["id"] == "gen-abc_1"


def test_r2_api11_malformed_generation_id_is_never_sent() -> None:
    """an id with a slash, a query or a space could change the request path; it is refused before any call."""
    seen: list[httpx.Request] = []
    client = httpx.Client(transport=httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200, json={})))
    for bad in ("", "a/b", "x?y=1", "gen id", "g" * 129):
        assert osp.fetch_generation_provider(bad, "k", client) is None
    assert seen == []


def test_r2_api11_failed_or_empty_lookup_is_none() -> None:
    server_error = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    no_provider = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": {"model": "m"}})))
    assert osp.fetch_generation_provider("gen-1", "k", server_error) is None
    assert osp.fetch_generation_provider("gen-1", "k", no_provider) is None


# R2-API-13: apply_text_normalization="on" may need enterprise access on v2.5 and is unverified through OpenRouter.
# It is refused before any request is sent; "auto" and "off" stay accepted.


def test_r2_api13_text_normalization_on_is_refused_before_any_request() -> None:
    posts: List[str] = []

    def speech(request: httpx.Request) -> httpx.Response:
        posts.append(str(request.url))
        return _ok(request)

    s = OpenRouterSpeech("key", client=_client(speech), max_retries=0)
    with pytest.raises(SpeechInputError, match="enterprise access"):
        s.synthesize(_req(apply_text_normalization="on"))
    assert posts == []
    for accepted in ("auto", "off"):
        _req(apply_text_normalization=accepted).validate()
