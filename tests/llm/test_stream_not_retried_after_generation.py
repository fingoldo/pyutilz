"""A streamed call is never re-opened once the upstream has generated, and every abandoned attempt is recorded.

glossum refsuite audit O-P0-4: a stream dropped mid-REASONING was retried up to MAX_RETRY_ATTEMPTS times, because only
an answer delta stopped the retry loop. Each attempt was billed for the reasoning it generated, and the caller, which
reads the final attempt's usage only, recorded none of it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

httpx = pytest.importorskip("httpx")

import pyutilz.llm.openai_compat as openai_compat_mod
from pyutilz.llm.openai_compat import OpenAICompatibleProvider


class _Provider(OpenAICompatibleProvider):
    _base_url = "https://test.example.com"
    _provider_name = "TestProvider"
    _max_tokens_map = {"test-model": 4096}
    _default_max_tokens = 2048

    def _input_cost_per_1m(self, model: str) -> float:
        return 1.0

    def _output_cost_per_1m(self, model: str) -> float:
        return 2.0


class _Stream:
    """``httpx.AsyncClient.stream``'s context manager: yields ``lines``, then raises ``fail`` (if any) mid-stream."""

    def __init__(self, lines: list[str], fail: BaseException | None = None, status_code: int = 200) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._lines = lines
        self._fail = fail
        self.request = httpx.Request("POST", "https://test.example.com/chat/completions")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=self.request, response=httpx.Response(self.status_code, request=self.request))

    async def aiter_lines(self) -> Any:
        for line in self._lines:
            yield line
        if self._fail is not None:
            raise self._fail

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def _data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}"


_REASONING = _data({"id": "gen-9", "choices": [{"delta": {"reasoning": "thinking about it"}}]})
_ANSWER = [_data({"id": "gen-10", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}), "data: [DONE]"]


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> _Provider:
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(openai_compat_mod, "MAX_RETRY_ATTEMPTS", 5)
    monkeypatch.setattr(openai_compat_mod.asyncio, "sleep", _no_sleep)
    p = _Provider(api_key="k", model="test-model")
    p._client = AsyncMock()
    return p


async def _drain(p: _Provider) -> list[str]:
    return [piece async for piece in p.generate_stream("q")]


@pytest.mark.asyncio
async def test_a_stream_dropped_mid_reasoning_is_not_reopened(provider: _Provider) -> None:
    """The drop is a retryable transport error, but the upstream had generated: re-opening it pays for the reasoning again."""
    provider._client.stream = MagicMock(side_effect=[_Stream([_REASONING], fail=httpx.ReadError("connection lost")), _Stream(_ANSWER)])

    with pytest.raises(httpx.ReadError):
        await _drain(provider)

    assert provider._client.stream.call_count == 1, "a stream that has generated anything must never be opened a second time"
    [aborted] = provider.last_aborted_stream_attempts
    assert (aborted["generation_id"], aborted["response_started"], aborted["generated_chunks"]) == ("gen-9", True, 1)


@pytest.mark.asyncio
async def test_a_stream_that_failed_before_generating_is_retried_and_the_failed_attempt_recorded(provider: _Provider) -> None:
    provider._client.stream = MagicMock(side_effect=[_Stream([], fail=httpx.ReadError("reset before any data")), _Stream(_ANSWER)])

    assert await _drain(provider) == ["ok"]

    assert provider._client.stream.call_count == 2
    [aborted] = provider.last_aborted_stream_attempts
    assert (aborted["attempt"], aborted["generation_id"], aborted["generated_chunks"], aborted["response_started"]) == (1, None, 0, True)


@pytest.mark.asyncio
async def test_a_cancelled_stream_records_its_attempt(provider: _Provider) -> None:
    """A caller's timeout cancels the stream: the attempt was generating and is billed, so it is recorded like any other."""
    provider._client.stream = MagicMock(return_value=_Stream([_REASONING], fail=asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        await _drain(provider)

    assert [a["generation_id"] for a in provider.last_aborted_stream_attempts] == ["gen-9"]


@pytest.mark.asyncio
async def test_the_record_is_reset_at_the_start_of_the_next_call(provider: _Provider) -> None:
    provider._client.stream = MagicMock(side_effect=[_Stream([], fail=httpx.ReadError("x")), _Stream(_ANSWER), _Stream(_ANSWER)])
    await _drain(provider)
    assert len(provider.last_aborted_stream_attempts) == 1
    await _drain(provider)
    assert provider.last_aborted_stream_attempts == []
