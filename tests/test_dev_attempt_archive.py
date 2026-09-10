"""pyutilz.dev.attempt_archive: every paid attempt keeps its raw text, stored before anything parses it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

import pytest

from pyutilz.dev.attempt_archive import (
    AttemptRecord,
    CallableContentStore,
    DirectoryContentStore,
    JsonlAttemptSink,
    archive_provider,
    metadata_from_provider,
    sha256_text,
)


class _FakeProvider:
    """Just enough of a pyutilz.llm provider: generate, generate_json through self.generate, per-call attrs."""

    model_name = "fake/model"

    def __init__(self, answers: List[Any], finish: str = "stop", stream_chunks: Optional[List[str]] = None, stream_fail: bool = False) -> None:
        self.answers = list(answers)
        self.finish = finish
        self.stream_chunks = stream_chunks or []
        self.stream_fail = stream_fail
        self._last_usage: dict = {}
        self._last_finish_reason: Optional[str] = None
        self.last_generation_id: Optional[str] = None
        self.last_upstream_provider: Optional[str] = None
        self.last_actual_cost_usd: Optional[float] = None

    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 0) -> str:
        self.last_generation_id = f"gen-{len(self.answers)}"
        self.last_upstream_provider = "Upstream"
        self._last_usage = {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 2}
        self._last_finish_reason = self.finish
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def generate_json(self, prompt: str, system: Optional[str] = None) -> dict:
        return json.loads(await self.generate(prompt, system=system))

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        self.last_generation_id = "gen-stream"
        for chunk in self.stream_chunks:
            yield chunk
        if self.stream_fail:
            raise ConnectionError("stream reset")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _wrapped(tmp_path: Path, provider: _FakeProvider) -> tuple:
    store = DirectoryContentStore(tmp_path / "texts")
    sink = JsonlAttemptSink(tmp_path / "attempts.jsonl")
    return archive_provider(provider, store, sink), store, sink


class TestTheTextIsKeptBeforeTheCallerParses:
    def test_a_parse_failure_in_the_caller_still_leaves_the_text_stored(self, tmp_path: Path) -> None:
        """The shape that motivated it: the model answered, the caller's parse failed, and the answer was gone."""
        provider, store, sink = _wrapped(tmp_path, _FakeProvider(["not json {"]))

        with pytest.raises(json.JSONDecodeError):
            _run(provider.generate_json("p"))

        (row,) = sink.read()
        assert store.get(row["response_sha256"]) == "not json {"
        assert row["outcome"] == "accepted" and row["raw_text"] == "not json {"

    def test_telemetry_is_read_from_the_call_that_made_it(self, tmp_path: Path) -> None:
        provider, _, sink = _wrapped(tmp_path, _FakeProvider(['{"a": 1}']))

        _run(provider.generate("p"))

        (row,) = sink.read()
        assert (row["generation_id"], row["upstream_provider"], row["finish_reason"]) == ("gen-1", "Upstream", "stop")
        assert (row["prompt_tokens"], row["completion_tokens"], row["reasoning_tokens"]) == (10, 5, 2)
        assert row["cost_usd"] is None, "a cost the provider did not report must stay unknown, not 0.0"
        assert row["model"] == "fake/model" and row["provider"] == "_FakeProvider"


class TestFailedAndCutOffAttemptsAreKept:
    def test_a_call_that_raises_is_recorded_and_re_raised(self, tmp_path: Path) -> None:
        provider, _, sink = _wrapped(tmp_path, _FakeProvider([TimeoutError("upstream timed out")]))

        with pytest.raises(TimeoutError):
            _run(provider.generate("p"))

        (row,) = sink.read()
        assert row["outcome"] == "error" and row["error"] == "TimeoutError: upstream timed out" and row["raw_text"] is None

    def test_a_length_stop_is_recorded_as_truncated(self, tmp_path: Path) -> None:
        provider, _, sink = _wrapped(tmp_path, _FakeProvider(['{"partial": '], finish="length"))

        _run(provider.generate("p"))

        assert [r["outcome"] for r in sink.read()] == ["truncated"]

    def test_a_stream_that_breaks_off_keeps_its_partial_text(self, tmp_path: Path) -> None:
        provider, store, sink = _wrapped(tmp_path, _FakeProvider([], stream_chunks=["half an ", "answer"], stream_fail=True))

        async def consume() -> List[str]:
            return [c async for c in provider.generate_stream("p")]

        with pytest.raises(ConnectionError):
            _run(consume())

        (row,) = sink.read()
        assert row["outcome"] == "truncated" and store.get(row["response_sha256"]) == "half an answer"
        assert row["generation_id"] == "gen-stream"

    def test_every_attempt_gets_its_own_number(self, tmp_path: Path) -> None:
        provider, _, sink = _wrapped(tmp_path, _FakeProvider(["one", "two", "three"]))

        for _ in range(3):
            _run(provider.generate("p"))

        assert [r["attempt_number"] for r in sink.read()] == [1, 2, 3]


class TestTheStoreNeverClaimsTextItDidNotWrite:
    def test_a_failing_writer_yields_none_not_the_digest(self) -> None:
        def broken(digest: str, text: str) -> None:
            raise RuntimeError("disk full")

        assert _run(CallableContentStore(broken).put("x")) is None

    def test_an_async_writer_is_awaited(self) -> None:
        seen: List[tuple] = []

        async def writer(digest: str, text: str) -> None:
            seen.append((digest, text))

        digest = _run(CallableContentStore(writer).put("hello"))

        assert seen == [(sha256_text("hello"), "hello")] and digest == sha256_text("hello")

    def test_an_unwritable_directory_yields_none(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x", encoding="utf-8")

        assert _run(DirectoryContentStore(blocker / "texts").put("hello")) is None

    def test_the_same_text_twice_is_one_file(self, tmp_path: Path) -> None:
        store = DirectoryContentStore(tmp_path)
        first, second = _run(store.put("same")), _run(store.put("same"))

        assert first == second and [p.name for p in tmp_path.iterdir()] == [f"{first}.txt"]


class TestTheWrapperIsSafeToApply:
    def test_wrapping_twice_records_once(self, tmp_path: Path) -> None:
        store, sink = DirectoryContentStore(tmp_path / "t"), JsonlAttemptSink(tmp_path / "a.jsonl")
        provider = archive_provider(archive_provider(_FakeProvider(["x"]), store, sink), store, sink)

        _run(provider.generate("p"))

        assert len(sink.read()) == 1

    def test_a_sink_that_raises_does_not_break_the_call(self, tmp_path: Path) -> None:
        class Broken:
            async def record(self, attempt: AttemptRecord) -> None:
                raise RuntimeError("sink down")

        provider = archive_provider(_FakeProvider(["kept"]), DirectoryContentStore(tmp_path), Broken())

        assert _run(provider.generate("p")) == "kept"

    def test_an_unknown_outcome_is_refused(self) -> None:
        with pytest.raises(ValueError):
            AttemptRecord(attempt_number=1, outcome="fine")


class TestTheMetadataReader:
    def test_last_call_summary_wins_and_attributes_fill_the_gaps(self) -> None:
        class Summarising(_FakeProvider):
            def last_call_summary(self) -> dict:
                return {"generation_id": "from-summary", "cost_usd": 0.004, "input_tokens": 99}

        provider = Summarising([])
        provider.last_upstream_provider = "FromAttr"

        meta = metadata_from_provider(provider)

        assert (meta["generation_id"], meta["cost_usd"], meta["prompt_tokens"], meta["upstream_provider"]) == ("from-summary", 0.004, 99, "FromAttr")
