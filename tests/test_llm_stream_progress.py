"""A streamed call's progress must be visible to a task other than the one consuming the stream.

The reasoning of a streamed call is stored in a ``PerCallAttr`` (a ContextVar), which isolates
concurrent calls on one shared provider. A watchdog or heartbeat runs as its own asyncio Task,
created before the stream starts, so it holds a copy of the context from before the provider reset
and filled the fragments: it read "0 reasoning chars" for the whole of a 134-second reasoning phase.
``pyutilz.llm._progress`` hands a counter over by reference instead; these tests drive the real
``OpenAICompatibleProvider._apply_stream_chunk`` and read the counter from another task.
"""

from __future__ import annotations

import asyncio

import pytest

from pyutilz.llm._progress import StreamProgress, note_stream_progress, track_stream_progress
from pyutilz.llm.deepseek_provider import DeepSeekProvider


def _reasoning_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"reasoning": text}}]}


def _content_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


@pytest.mark.asyncio
async def test_a_watcher_task_created_before_the_stream_sees_reasoning_grow() -> None:
    provider = DeepSeekProvider(api_key="sk-fake-test-key-not-real")
    progress = StreamProgress()
    seen: list[int] = []
    release = asyncio.Event()

    async def watcher() -> None:
        # Created BEFORE the consumer installs anything or the provider resets its per-call state,
        # which is exactly the position of the glossum watchdog.
        await release.wait()
        seen.append(progress.reasoning_chars)
        seen.append(len(provider.last_reasoning_text or ""))

    watch = asyncio.ensure_future(watcher())
    with track_stream_progress(progress):
        provider._reset_per_call_state()
        for _ in range(3):
            provider._apply_stream_chunk(_reasoning_chunk("think " * 10), {})
        assert provider._apply_stream_chunk(_content_chunk('{"a": 1}'), {}) == '{"a": 1}'
    release.set()
    await watch
    assert seen[0] == 180, "the shared counter carries the reasoning the consumer received"
    assert seen[1] == 0, "the ContextVar-backed fragments stay invisible from the other task, which is why the counter exists"
    assert progress.answer_chars == len('{"a": 1}')


def test_no_counter_installed_is_a_no_op() -> None:
    provider = DeepSeekProvider(api_key="sk-fake-test-key-not-real")
    provider._reset_per_call_state()
    assert provider._apply_stream_chunk(_content_chunk("x"), {}) == "x"
    note_stream_progress("abc", "def")  # nothing installed: must not raise


def test_counter_is_uninstalled_after_the_block() -> None:
    outer = StreamProgress()
    with track_stream_progress(outer):
        with track_stream_progress() as inner:
            note_stream_progress("ab", None)
        note_stream_progress(None, "xyz")
    assert (inner.reasoning_chars, inner.answer_chars) == (2, 0)
    assert (outer.reasoning_chars, outer.answer_chars) == (0, 3)
    assert outer.total_chars == 3
