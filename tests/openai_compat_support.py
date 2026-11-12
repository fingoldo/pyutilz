"""Shared fixtures and fakes for the OpenAI-compatible provider tests (split out of test_openai_compat.py)."""

from __future__ import annotations
from pyutilz.core import jsonlib
import pytest
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

httpx = pytest.importorskip("httpx")

class _TestProvider(OpenAICompatibleProvider):
    _base_url = "https://test.example.com"
    _provider_name = "TestProvider"
    _max_tokens_map = {"test-model": 4096}
    _default_max_tokens = 2048

    def _input_cost_per_1m(self, model: str) -> float:
        return 1.0

    def _output_cost_per_1m(self, model: str) -> float:
        return 2.0

    def _cache_hit_cost_per_1m(self, model: str) -> float:
        return 0.1


def _make_provider(model: str = "test-model") -> _TestProvider:
    return _TestProvider(api_key="test-key", model=model)  # pragma: allowlist secret


def _mock_response(status_code: int = 200, body: dict | None = None, text: str = "") -> httpx.Response:
    if body is None:
        body = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_cache_hit_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
    resp = httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", "https://test.example.com/chat/completions"),
    )
    return resp


class _MockStreamResponse:
    """Minimal async-context-manager mock standing in for
    ``httpx.AsyncClient.stream(...)``'s ``async with ... as resp:`` result.
    """

    def __init__(self, status_code: int = 200, headers: dict | None = None, lines: list[str] | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self._lines = lines or []
        self.request = httpx.Request("POST", "https://test.example.com/chat/completions")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, headers=self.headers, request=self.request),
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeResponse:
    """The three response shapes an intermediary can return in place of the API's JSON envelope."""

    def __init__(self, text: str, status_code: int = 200, payload: object | None = None, raises: bool = False):
        self.text = text
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self) -> object:
        if self._raises:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class _WarnLog:
    def __init__(self):
        self.calls: list = []

    def warning(self, msg, *args, **kwargs):
        self.calls.append((msg, args))


class _StatusError(Exception):
    def __init__(self, response=None):
        super().__init__("status")
        self.response = response


class _FakeHttpClient:
    def __init__(self, closed: bool = False, fail: bool = False):
        self.is_closed = closed
        self.fail = fail
        self.close_calls = 0

    async def aclose(self):
        if self.fail:
            raise RuntimeError("socket already gone")
        self.close_calls += 1
        self.is_closed = True


class _FakeStreamResponse:
    def __init__(self, status: int = 200, lines: tuple = (), headers: dict | None = None):
        self.status_code = status
        self.headers = headers or {}
        self._lines = lines
        self.aread_calls = 0
        self.request = httpx.Request("POST", "https://test.example.com/chat/completions")

    async def aread(self):
        self.aread_calls += 1
        return b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, owner, response):
        self._owner = owner
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeStreamingHttpClient:
    def __init__(self, response):
        self._response = response
        self.calls: list = []

    def stream(self, method, url, json=None, timeout=None):
        self.calls.append((method, url, json, timeout))
        return _FakeStreamContext(self, self._response)


async def _drain_stream(provider, body=None, state=None):
    box = {"streamed_text": []}
    attempt_state = state if state is not None else {}
    async for _ in provider._stream_attempt(body or {"model": "test-model"}, attempt_state, box):
        pass
    return attempt_state


def _content_line(text: str) -> str:
    return "data: " + jsonlib.dumps({"choices": [{"delta": {"content": text}}]})


async def _drain_with_box(provider, lines: tuple):
    provider._client = _FakeStreamingHttpClient(_FakeStreamResponse(status=200, lines=lines))
    box: dict = {"streamed_text": []}
    attempt_state: dict = {}
    yielded = [piece async for piece in provider._stream_attempt({"model": "test-model"}, attempt_state, box)]
    return yielded, box, attempt_state


class _StatusStreamResponse(_FakeStreamResponse):
    """A failed stream open: raise_for_status raises the httpx error the retry predicate classifies."""

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://test.example.com/chat/completions")
            raise httpx.HTTPStatusError("boom", request=request, response=httpx.Response(self.status_code, request=request))


class _SequenceStreamClient:
    """Hands out one scripted response per stream() call; the last one repeats."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list = []

    def stream(self, method, url, json=None, timeout=None):
        self.calls.append(json)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _FakeStreamContext(self, response)


def _no_sleep(monkeypatch, provider, wait: float = 7.0):
    import asyncio

    waits: list = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(provider, "_stream_retry_backoff", lambda exc, attempt: wait)
    return waits


async def _collect_stream(provider, **kwargs):
    return [piece async for piece in provider.generate_stream("hi", **kwargs)]


class _PreparationOrderError(Exception):
    pass


class _WarnRecorder:
    def __init__(self):
        self.warnings: list = []

    def warning(self, fmt, *args, **kwargs):
        self.warnings.append((fmt, args))


class _JsonPostResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.headers: dict = {}
        self.text = jsonlib.dumps(payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _PostClient:
    def __init__(self, response):
        self._response = response
        self.calls: list = []

    async def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return self._response


class _StatusBodyResponse:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers: dict = {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _Tier:
    def __init__(self, applies: bool):
        self._applies = applies

    def applies(self, tokens):
        return self._applies


class _BareCloseClient:
    """A client with aclose but no is_closed flag: it must still be closed."""

    def __init__(self):
        self.close_calls = 0

    async def aclose(self):
        self.close_calls += 1


class _DyingStreamResponse(_FakeStreamResponse):
    """Yields its lines, then the connection drops before the stream finishes."""

    async def aiter_lines(self):
        for line in self._lines:
            yield line
        raise RuntimeError("connection dropped mid-answer")


class _ExitRecordingContext(_FakeStreamContext):
    def __init__(self, owner, response):
        super().__init__(owner, response)
        self.exited = False

    async def __aexit__(self, *exc):
        self.exited = True
        return False


class _ExitRecordingClient(_SequenceStreamClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.contexts: list = []

    def stream(self, method, url, json=None, timeout=None):
        self.calls.append(json)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        context = _ExitRecordingContext(self, response)
        self.contexts.append(context)
        return context


class _MidStreamFailure(_FakeStreamResponse):
    async def aiter_lines(self):
        for line in self._lines:
            yield line
        raise httpx.HTTPStatusError(
            "upstream fell over",
            request=httpx.Request("POST", "https://test.example.com/chat/completions"),
            response=httpx.Response(503, request=httpx.Request("POST", "https://test.example.com/chat/completions")),
        )


class _RequestedBodyResponse(_StatusBodyResponse):
    def __init__(self, request):
        super().__init__(200, {"error": {"code": 429, "message": "rate limited"}})
        self.request = request


class _RequestlessBodyResponse(_StatusBodyResponse):
    def __init__(self):
        super().__init__(200, {"error": {"code": 429, "message": "rate limited"}})
