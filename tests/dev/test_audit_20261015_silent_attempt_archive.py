from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any


class _StatsBrokenProvider:
    model_name = "fake/model"
    last_generation_id = "gen-1"
    last_upstream_provider = "Upstream"
    _last_usage: dict = {}
    _last_finish_reason = "stop"
    last_actual_cost_usd = None

    async def generate(self, prompt: str, system: Any = None, temperature: float = 0.7, max_tokens: int = 0) -> str:
        return "answer"

    async def fetch_generation_stats(self, generation_id: str) -> dict:
        raise RuntimeError("upstream timeout")


def test_s09_failed_stats_fetch_is_marked(tmp_path: Path) -> None:
    from pyutilz.dev.attempt_archive import DirectoryContentStore, JsonlAttemptSink, archive_provider

    store = DirectoryContentStore(tmp_path / "t")
    sink = JsonlAttemptSink(tmp_path / "a.jsonl")
    wrapped = archive_provider(_StatsBrokenProvider(), store, sink, fetch_stats=True)
    asyncio.run(wrapped.generate("p"))
    attempt = sink.read()[0]
    assert attempt["stats_fetch_failed"] is True
    assert attempt["latency_ms"] is None


def test_s09_stats_not_requested_leave_marker_unset(tmp_path: Path) -> None:
    from pyutilz.dev.attempt_archive import DirectoryContentStore, JsonlAttemptSink, archive_provider

    store = DirectoryContentStore(tmp_path / "t")
    sink = JsonlAttemptSink(tmp_path / "a.jsonl")
    wrapped = archive_provider(_StatsBrokenProvider(), store, sink, fetch_stats=False)
    asyncio.run(wrapped.generate("p"))
    assert sink.read()[0]["stats_fetch_failed"] is None
