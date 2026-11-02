"""Behaviour tests for the OpenRouter speech survivors in the accounting, MP3/markup sniffing, HTTP helpers and catalogue cache."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import threading
import time

import httpx
import pytest

from pyutilz.llm import openrouter_speech as sp
from pyutilz.llm.openrouter_speech import SpeechError, SpeechRequest, SpeechResult

_RealClient = httpx.Client


def _req(**overrides) -> SpeechRequest:
    base = dict(text="hello", model="elevenlabs/eleven-v3", voice="v")
    base.update(overrides)
    return SpeechRequest(**base)


def _mock(handler, **kw) -> httpx.Client:
    return _RealClient(transport=httpx.MockTransport(handler), **kw)


def _recording_client(handler, closed: list) -> httpx.Client:
    client = _mock(handler)
    client.close = lambda: closed.append(1)
    return client


@pytest.fixture(autouse=True)
def _fresh_catalogue_cache():
    sp.reset_speech_catalogue_cache()
    yield
    sp.reset_speech_catalogue_cache()


@pytest.fixture
def client_factory(monkeypatch):
    """Route every httpx.Client the module creates for itself to a MockTransport; record the kwargs and closes."""
    made = []
    closed = []

    def build(handler):
        def factory(**kw):
            made.append(kw)
            return _recording_client(handler, closed)

        monkeypatch.setattr(sp.httpx, "Client", factory)
        return made, closed

    return build


class TestFrozenRecords:
    def test_a_request_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _req().text = "other"

    def test_a_result_is_immutable(self):
        result = SpeechResult(audio=b"a", request=_req(), chars=1, cost_usd=None, latency_s=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.chars = 2


class TestSpeechErrorMessage:
    def test_the_message_attribute_is_set_from_the_base_class(self):
        assert SpeechError("m").message == "m"


class TestBodyAndHash:
    def test_pronunciation_locators_are_sent_under_their_key(self):
        body = _req(pronunciation_dictionary_locators=[{"id": "d"}]).body()
        assert body["provider"]["options"]["elevenlabs"]["pronunciation_dictionary_locators"] == [{"id": "d"}]

    @staticmethod
    def _expected(body: dict) -> str:
        return hashlib.sha256(json.dumps(sp._canonical(body), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

    def test_the_hash_is_the_sorted_unicode_json_of_the_body_without_the_input(self):
        req = _req(voice="голос", speed=1.1)
        body = _req(voice="голос", speed=1.1, text="").body()
        body.pop("input")
        assert req.params_hash() == self._expected(body)

    def test_the_hash_has_sixteen_hex_characters(self):
        assert len(_req().params_hash()) == 16


class TestAccountingCounters:
    @staticmethod
    def _result(cost_usd):
        return SpeechResult(audio=b"ab", request=_req(), chars=3, cost_usd=cost_usd, latency_s=0.0)

    def test_each_unpriced_call_is_counted(self):
        acc = sp.SpeechAccounting()
        acc.record(self._result(None))
        acc.record(self._result(None))
        assert acc.unpriced_calls == 2

    def test_each_failure_is_counted(self):
        acc = sp.SpeechAccounting()
        acc.record_failure()
        acc.record_failure()
        assert acc.failures == 2

    def test_cache_hits_accumulate(self):
        acc = sp.SpeechAccounting()
        acc.record_cache_hit(3, 5)
        acc.record_cache_hit(3, 5)
        assert (acc.cache_hits, acc.cached_chars, acc.cached_bytes) == (2, 6, 10)

    def test_possibly_billed_attempts_accumulate_chars_and_cost(self):
        acc = sp.SpeechAccounting()
        acc.record_possibly_billed(chars=2, cost_usd=0.5)
        acc.record_possibly_billed(chars=2, cost_usd=0.5)
        assert (acc.possibly_billed_attempts, acc.possibly_billed_chars, acc.possibly_billed_cost_usd) == (2, 4, 1.0)

    def test_a_possibly_billed_attempt_without_chars_adds_none(self):
        acc = sp.SpeechAccounting()
        acc.record_possibly_billed()
        assert acc.possibly_billed_chars == 0


class TestErrorKind:
    @pytest.mark.parametrize(
        "status, expected",
        [(500, "http_5xx"), (503, "http_5xx"), (300, "http_3xx"), (399, "http_3xx"), (400, "http_4xx"), (402, "http_402"), (429, "http_429"), (200, "empty_body")],
    )
    def test_each_status_group(self, status, expected):
        assert sp.error_kind(status, transport=False) == expected

    def test_a_transport_failure_is_transport(self):
        assert sp.error_kind(0, transport=True) == "transport"


class TestMp3Sniffing:
    @pytest.mark.parametrize(
        "content, expected",
        [
            (b"ID3\x03", True),
            (b"\xff", False),
            (b"\xff\xe2", True),
            (b"\xff\xe2\x00", True),
            (b"\xff\xe2\xea", True),
            (b"\xff\xe0\x02", False),
            (b"\xff\xea", False),
            (b"\xff\xe0", False),
            (b"\x00\xe2", False),
        ],
    )
    def test_the_frame_header_is_checked_on_the_right_bytes(self, content, expected):
        assert sp._is_mp3(content) is expected


class TestMarkupSniffing:
    @pytest.mark.parametrize(
        "content, expected",
        [
            (b"<a>", True),
            (b"{}", True),
            (b"[]", True),
            (b"a", False),
            (b"\xff", False),
            (b"<\x7f", False),
            (b"<\t", True),
            (b"<\n", True),
            (b"<\r", True),
            (b"{" + b"a" * 255 + b"\x01", True),
        ],
    )
    def test_only_the_first_256_printable_bytes_decide(self, content, expected):
        assert sp._is_markup(content) is expected


class _Resp:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_bytes(self):
        yield from self._chunks


class TestReadCapped:
    def test_a_body_longer_than_the_limit_is_cut_at_the_limit(self):
        assert sp._read_capped(_Resp([b"abcd", b"efgh"]), 5) == (b"abcde", True)

    def test_a_body_within_the_limit_is_returned_whole(self):
        assert sp._read_capped(_Resp([b"abc"]), 5) == (b"abc", False)

    def test_a_body_exactly_at_the_limit_is_not_cut(self):
        assert sp._read_capped(_Resp([b"abcde"]), 5) == (b"abcde", False)


class TestReplyDefaults:
    def test_a_fresh_reply_has_no_status_and_is_not_too_large(self):
        reply = sp._Reply(limit=1)
        assert (reply.status, reply.too_large) == (0, False)

    def test_the_text_is_the_first_2000_characters(self):
        assert sp._Reply(limit=1, content=b"a" * 2500).text == "a" * 2000

    def test_undecodable_bytes_become_the_replacement_character(self):
        assert sp._Reply(limit=1, content=b"\xff").text == "�"


class TestBodyProblemHeaders:
    def test_a_json_content_type_is_a_bad_body_even_with_audio_bytes(self):
        reply = sp._Reply(limit=10, status=200, headers={"content-type": "application/json"}, content=b"ID3\x03")
        assert sp._body_problem(reply, "mp3") == "bad_body"


class TestJudgeMessages:
    def test_a_transport_failure_message_names_the_error(self):
        assert sp._judge(sp._Reply(limit=1, transport_error=RuntimeError("boom")), "mp3") == (None, "speech request failed: RuntimeError: boom", True)

    def test_an_http_error_message_carries_the_first_300_characters_of_the_body(self):
        message = sp._judge(sp._Reply(limit=1, status=503, content=b"x" * 400), "mp3")[1]
        assert message == "speech request returned HTTP 503: " + "x" * 300


class TestAccountingFrozenSnapshot:
    def test_snapshot_is_a_copy_of_the_counters(self):
        acc = sp.SpeechAccounting(calls=2)
        snap = acc.snapshot()
        acc.calls = 9
        assert snap["calls"] == 2


class TestFetchCredits:
    @staticmethod
    def _handler(seen: list, status: int = 200):
        def handler(request):
            seen.append(request)
            if status != 200:
                return httpx.Response(status)
            return httpx.Response(200, json={"data": {"total_credits": 10, "total_usage": 4}})

        return handler

    def test_the_balance_is_parsed_with_the_remaining_amount(self):
        seen = []
        result = sp.fetch_credits("KEY", client=_mock(self._handler(seen)))
        assert result == {"total_credits": 10.0, "total_usage": 4.0, "remaining": 6.0}

    def test_the_request_goes_to_the_credits_url_with_the_bearer_key(self):
        seen = []
        sp.fetch_credits("KEY", client=_mock(self._handler(seen)))
        assert str(seen[0].url) == f"{sp.BASE_URL}/credits"
        assert seen[0].headers["Authorization"] == "Bearer KEY"

    def test_an_http_error_returns_none_and_logs_the_exception_type_first(self, caplog):
        caplog.set_level(logging.WARNING)
        assert sp.fetch_credits("KEY", client=_mock(self._handler([], status=500))) is None
        assert "balance check failed (HTTPStatusError: " in caplog.text

    def test_a_caller_supplied_client_is_not_closed(self):
        closed = []
        sp.fetch_credits("KEY", client=_recording_client(self._handler([]), closed))
        assert closed == []

    def test_an_owned_client_is_built_with_the_timeout_and_auth_header_and_closed(self, client_factory):
        made, closed = client_factory(self._handler([]))
        sp.fetch_credits("KEY")
        assert made == [{"timeout": 15.0, "headers": {"Authorization": "Bearer KEY"}}]
        assert closed == [1]

    def test_without_httpx_and_without_a_client_there_is_no_balance(self, monkeypatch):
        monkeypatch.setattr(sp, "httpx", None)
        assert sp.fetch_credits("KEY") is None

    def test_without_httpx_a_supplied_client_still_works(self, monkeypatch):
        monkeypatch.setattr(sp, "httpx", None)
        assert sp.fetch_credits("KEY", client=_mock(self._handler([])))["remaining"] == 6.0


class TestFetchGenerationProvider:
    @staticmethod
    def _handler(seen: list, status: int = 200):
        def handler(request):
            seen.append(request)
            if request.headers.get("Authorization") != "Bearer KEY" or status != 200:
                return httpx.Response(status if status != 200 else 401)
            return httpx.Response(200, json={"data": {"provider_name": "DeepInfra", "model": "m"}})

        return handler

    def test_the_serving_endpoint_is_returned(self):
        seen = []
        result = sp.fetch_generation_provider("gen-123", "KEY", client=_mock(self._handler(seen)))
        assert result == {"provider_name": "DeepInfra", "model": "m"}
        assert str(seen[0].url).startswith(f"{sp.BASE_URL}/generation")
        assert seen[0].url.params["id"] == "gen-123"

    def test_a_malformed_id_is_refused_without_a_request(self):
        seen = []
        assert sp.fetch_generation_provider("bad id", "KEY", client=_mock(self._handler(seen))) is None
        assert seen == []

    def test_an_http_error_returns_none_and_logs_the_exception_type_first(self, caplog):
        caplog.set_level(logging.WARNING)
        assert sp.fetch_generation_provider("gen-1", "KEY", client=_mock(self._handler([], status=500))) is None
        assert "generation lookup failed (HTTPStatusError: " in caplog.text

    def test_a_caller_supplied_client_is_not_closed(self):
        closed = []
        sp.fetch_generation_provider("gen-1", "KEY", client=_recording_client(self._handler([]), closed))
        assert closed == []

    def test_an_owned_client_is_built_with_the_timeout_and_auth_header_and_closed(self, client_factory):
        made, closed = client_factory(self._handler([]))
        sp.fetch_generation_provider("gen-1", "KEY")
        assert made == [{"timeout": 15.0, "headers": {"Authorization": "Bearer KEY"}}]
        assert closed == [1]

    def test_without_httpx_and_without_a_client_there_is_no_lookup(self, monkeypatch):
        monkeypatch.setattr(sp, "httpx", None)
        assert sp.fetch_generation_provider("gen-1", "KEY") is None


class TestCatalogueFetch:
    @staticmethod
    def _handler(seen: list, status: int = 200):
        def handler(request):
            seen.append(request)
            if status != 200:
                return httpx.Response(status)
            return httpx.Response(200, json={"data": [{"id": "m", "pricing": {"prompt": "0.1"}}, {"no_id": True}]})

        return handler

    def test_models_without_an_id_are_dropped_and_the_rest_are_keyed_by_id(self):
        out = sp._fetch_speech_catalogue(_mock(self._handler([])), sp.BASE_URL)
        assert list(out) == ["m"]

    def test_an_owned_client_fetches_the_models_with_the_timeout_and_closes(self, client_factory):
        made, closed = client_factory(self._handler([]))
        out = sp._fetch_speech_catalogue(None, sp.BASE_URL)
        assert list(out) == ["m"]
        assert made == [{"timeout": 15.0}]
        assert closed == [1]

    def test_a_caller_supplied_client_is_not_closed(self):
        closed = []
        sp._fetch_speech_catalogue(_recording_client(self._handler([]), closed), sp.BASE_URL)
        assert closed == []

    def test_a_failed_fetch_returns_none_and_logs_the_exception_type_first(self, caplog):
        caplog.set_level(logging.WARNING)
        assert sp._fetch_speech_catalogue(_mock(self._handler([], status=500)), sp.BASE_URL) is None
        assert "speech catalogue fetch failed (HTTPStatusError: " in caplog.text

    def test_without_httpx_and_without_a_client_the_fetch_is_skipped_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(sp, "httpx", None)
        caplog.set_level(logging.WARNING)
        assert sp._fetch_speech_catalogue(None, sp.BASE_URL) is None
        assert "httpx is not installed (optional extra 'llm'); speech prices are unknown" in caplog.text

    def test_without_httpx_a_supplied_client_is_still_used(self, monkeypatch):
        monkeypatch.setattr(sp, "httpx", None)
        assert list(sp._fetch_speech_catalogue(_mock(self._handler([])), sp.BASE_URL)) == ["m"]


class TestCatalogueCache:
    @staticmethod
    def _counting(seen: list, status: int = 200):
        def handler(request):
            seen.append(request)
            if status != 200:
                return httpx.Response(status)
            return httpx.Response(200, json={"data": [{"id": "m", "pricing": {"prompt": "0.1"}}]})

        return handler

    def test_a_trailing_slash_does_not_split_the_cache_key(self):
        seen = []
        client = _mock(self._counting(seen))
        sp.speech_catalogue(client, base_url="https://h/api/")
        sp.speech_catalogue(client, base_url="https://h/api")
        assert len(seen) == 1
        assert seen[0].url.path == "/api/models"

    def test_the_listing_is_reused_strictly_before_the_ttl_and_refetched_at_it(self, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr(sp.time, "monotonic", lambda: clock[0])
        seen = []
        client = _mock(self._counting(seen))
        sp.speech_catalogue(client)
        clock[0] = sp._CATALOGUE_TTL_S - 1
        sp.speech_catalogue(client)
        assert len(seen) == 1
        clock[0] = sp._CATALOGUE_TTL_S
        sp.speech_catalogue(client)
        assert len(seen) == 2

    def test_a_failed_fetch_is_remembered_strictly_before_the_backoff_and_retried_at_it(self, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr(sp.time, "monotonic", lambda: clock[0])
        seen = []
        client = _mock(self._counting(seen, status=500))
        assert sp.speech_catalogue(client) == {}
        clock[0] = sp._CATALOGUE_FAILURE_BACKOFF_S - 1
        sp.speech_catalogue(client)
        assert len(seen) == 1
        clock[0] = sp._CATALOGUE_FAILURE_BACKOFF_S
        sp.speech_catalogue(client)
        assert len(seen) == 2


class TestResetWakesWaiters:
    def test_reset_releases_a_waiter_without_waiting_for_the_in_flight_fetch(self):
        release = threading.Event()

        def blocking(request):
            release.wait(10)
            return httpx.Response(200, json={"data": [{"id": "slow"}]})

        def fast(request):
            return httpx.Response(200, json={"data": [{"id": "fast"}]})

        slow = threading.Thread(target=sp.speech_catalogue, args=(_mock(blocking),), daemon=True)
        slow.start()
        deadline = time.monotonic() + 10
        while not any(entry["inflight"] for entry in sp._catalogue_cache.values()):
            assert time.monotonic() < deadline, "the first fetch never started"
            time.sleep(0.01)

        waiter_out = []
        waiter = threading.Thread(target=lambda: waiter_out.append(sp.speech_catalogue(_mock(fast))), daemon=True)
        waiter.start()
        time.sleep(0.2)  # the waiter has no listing and a fetch is in flight, so it is parked on the condition
        try:
            sp.reset_speech_catalogue_cache()
            waiter.join(timeout=5)
            assert not waiter.is_alive(), "the waiter stayed parked after the reset"
            assert waiter_out == [{"fast": {"id": "fast"}}]
        finally:
            release.set()
            slow.join(timeout=10)
            waiter.join(timeout=10)


class TestExpiryWarning:
    @staticmethod
    def _today() -> dt.date:
        return dt.datetime.now(dt.timezone.utc).date()

    def test_a_model_expiring_exactly_at_the_warning_window_is_warned(self, caplog):
        caplog.set_level(logging.WARNING)
        expires = self._today() + dt.timedelta(days=sp._EXPIRY_WARNING_DAYS)
        sp._warn_if_expiring("m", {"expiration_date": expires.isoformat()})
        assert f"speech model m expires on {expires.isoformat()} ({sp._EXPIRY_WARNING_DAYS} days left); plan its replacement" in caplog.text

    def test_a_model_beyond_the_window_is_not_warned(self, caplog):
        caplog.set_level(logging.WARNING)
        sp._warn_if_expiring("m", {"expiration_date": (self._today() + dt.timedelta(days=sp._EXPIRY_WARNING_DAYS + 1)).isoformat()})
        assert "expires on" not in caplog.text

    def test_only_the_date_part_of_a_timestamp_is_read(self, caplog):
        caplog.set_level(logging.WARNING)
        expires = self._today() + dt.timedelta(days=10)
        sp._warn_if_expiring("m", {"expiration_date": f"{expires.isoformat()}T00:00:00Z"})
        assert f"speech model m expires on {expires.isoformat()}" in caplog.text

    def test_an_unreadable_date_is_logged_with_its_raw_value(self, caplog):
        caplog.set_level(logging.WARNING)
        sp._warn_if_expiring("m", {"expiration_date": "soon"})
        assert "speech model m has an unreadable expiration_date 'soon'; no expiry warning for it" in caplog.text

    def test_a_model_is_warned_once_until_the_cache_is_reset(self, caplog):
        caplog.set_level(logging.WARNING)
        entry = {"expiration_date": self._today().isoformat()}
        sp._warn_if_expiring("m", entry)
        sp._warn_if_expiring("m", entry)
        assert caplog.text.count("expires on") == 1
        sp.reset_speech_catalogue_cache()
        sp._warn_if_expiring("m", entry)
        assert caplog.text.count("expires on") == 2
