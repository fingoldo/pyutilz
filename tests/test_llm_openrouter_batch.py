"""Behavioural tests for pyutilz.llm.openrouter_batch against a fake OpenRouter (httpx.MockTransport)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from pyutilz.llm.openrouter_batch import (
    BatchRequest,
    OpenRouterBatchClient,
    OpenRouterBatchError,
    build_submit_payload,
    load_job_state,
    parse_results,
)


def _ok(cid: str, text: str, cost: float, finish: str = "stop") -> dict[str, Any]:
    return {
        "custom_id": cid,
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"content": text, "reasoning": "r-" + cid}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "cost": cost},
            },
        },
        "error": None,
    }


class FakeOpenRouter:
    def __init__(self, polls_until_done: int = 2, results: list[dict[str, Any]] | None = None, status: str = "completed"):
        self.polls_until_done = polls_until_done
        self.results = results
        self.final_status = status
        self.submits: list[dict[str, Any]] = []
        self.raw_submit: list[bytes] = []
        self.polls = 0
        self.fail_next_poll = False

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["Authorization"] == "Bearer k"
        if req.method == "POST" and req.url.path.endswith("/batches"):
            self.raw_submit.append(req.content)
            self.submits.append(json.loads(req.content))
            return httpx.Response(202, json={"id": "batch_1", "status": "validating"})
        if req.method == "GET" and req.url.path.endswith("/batches/batch_1"):
            self.polls += 1
            if self.fail_next_poll:
                self.fail_next_poll = False
                return httpx.Response(503, text="busy")
            if self.polls < self.polls_until_done:
                return httpx.Response(200, json={"id": "batch_1", "status": "in_progress"})
            return httpx.Response(200, json={"id": "batch_1", "status": self.final_status, "results": self.results})
        return httpx.Response(404)


def _client(fake: FakeOpenRouter, clock_step: float = 1.0) -> tuple[OpenRouterBatchClient, list[float]]:
    slept: list[float] = []
    t = [0.0]

    def clock() -> float:
        t[0] += clock_step
        return t[0]

    c = OpenRouterBatchClient("k", transport=httpx.MockTransport(fake.handler), sleep=slept.append, clock=clock)
    return c, slept


REQS = [
    BatchRequest("q-a", {"messages": [{"role": "user", "content": "a"}], "model": "ignored"}),
    BatchRequest("q-b", {"messages": [{"role": "user", "content": "b"}]}),
    BatchRequest("q-c", {"messages": [{"role": "user", "content": "c"}]}),
]


def test_results_map_back_by_custom_id_not_position(tmp_path: Path) -> None:
    # results arrive shuffled and one request fails; mapping must follow custom_id
    fake = FakeOpenRouter(
        results=[_ok("q-c", "C", 0.3), {"custom_id": "q-b", "response": None, "error": {"code": 400, "message": "bad"}}, _ok("q-a", "A", 0.1, "length")]
    )
    c, _ = _client(fake)
    out = c.run("m/x", REQS, state_path=tmp_path / "job.json")
    assert out["q-a"].text == "A" and out["q-a"].finish_reason == "length" and out["q-a"].cost_usd == pytest.approx(0.1)
    assert out["q-c"].text == "C" and out["q-c"].reasoning == "r-q-c" and out["q-c"].prompt_tokens == 11
    assert not out["q-b"].ok and "bad" in (out["q-b"].error or "")
    # q-a was cut by max_tokens: its text is kept but it is not a complete answer (2026-09-26 OR-18).
    assert out["q-a"].truncated and not out["q-a"].ok and out["q-c"].ok


def test_missing_and_non2xx_results_surface_as_errors(tmp_path: Path) -> None:
    bad = {"custom_id": "q-b", "response": {"status_code": 429, "body": {"error": {"message": "rate"}}}}
    fake = FakeOpenRouter(results=[_ok("q-a", "A", 0.1), bad], status="expired")
    c, _ = _client(fake)
    out = c.run("m/x", REQS, state_path=tmp_path / "job.json")
    assert out["q-a"].ok
    assert "429" in (out["q-b"].error or "") and out["q-b"].text is None
    assert "missing" in (out["q-c"].error or "") and "expired" in (out["q-c"].error or "")


def test_job_id_persisted_before_polling_and_resume_does_not_resubmit(tmp_path: Path) -> None:
    state = tmp_path / "sub" / "job.json"
    fake = FakeOpenRouter(polls_until_done=10**9)  # never finishes: simulate a crash/deadline mid-wait
    c, _ = _client(fake)
    with pytest.raises(OpenRouterBatchError, match="deadline"):
        c.run("m/x", REQS, state_path=state, deadline_s=5)
    saved = load_job_state(state)
    assert saved is not None and saved["batch_id"] == "batch_1" and saved["custom_ids"] == ["q-a", "q-b", "q-c"]
    assert len(fake.submits) == 1

    fake.polls_until_done = 0
    fake.results = [_ok(r.custom_id, r.custom_id.upper(), 0.01) for r in REQS]
    c2, _ = _client(fake)
    out = c2.run("m/x", REQS, state_path=state)
    assert len(fake.submits) == 1, "resume must not submit a second paid batch"
    assert out["q-b"].text == "Q-B"


def test_resume_refuses_mismatched_state(tmp_path: Path) -> None:
    state = tmp_path / "job.json"
    state.write_bytes(json.dumps({"batch_id": "old", "model": "m/x", "custom_ids": ["zzz"]}).encode())
    fake = FakeOpenRouter()
    c, _ = _client(fake)
    with pytest.raises(OpenRouterBatchError, match="different"):
        c.run("m/x", REQS, state_path=state)
    assert fake.submits == []


def test_submit_payload_order_and_model_stripped(tmp_path: Path) -> None:
    fake = FakeOpenRouter(polls_until_done=0, results=[])
    c, _ = _client(fake)
    c.run("m/x", REQS, state_path=tmp_path / "j.json", provider_only=["prov"])
    sent = fake.submits[0]
    assert list(sent) == ["endpoint", "model", "provider", "completion_window", "requests"]
    assert fake.raw_submit[0].index(b'"requests"') > fake.raw_submit[0].index(b'"completion_window"')
    assert sent["provider"] == {"only": ["prov"]} and sent["model"] == "m/x"
    assert [r["custom_id"] for r in sent["requests"]] == ["q-a", "q-b", "q-c"]
    assert "model" not in sent["requests"][0]["body"]


def test_payload_validation() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_submit_payload("m", [BatchRequest("x", {}), BatchRequest("x", {})])
    with pytest.raises(ValueError, match="stream"):
        build_submit_payload("m", [BatchRequest("x", {"stream": True})])
    with pytest.raises(ValueError):
        build_submit_payload("m", [])


def test_poll_backoff_grows_capped_and_survives_transient_error(tmp_path: Path) -> None:
    fake = FakeOpenRouter(polls_until_done=8, results=[])
    fake.fail_next_poll = True
    c, slept = _client(fake)
    c.wait("batch_1", initial_interval_s=10, max_interval_s=40)
    assert fake.polls == 8
    assert len(slept) == 7
    assert slept[0] < slept[2] and all(s <= 40 * 1.1 for s in slept)
    assert max(slept) > 30


def test_submit_http_error_raises_and_writes_no_state(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad model"})

    c = OpenRouterBatchClient("k", transport=httpx.MockTransport(handler))
    with pytest.raises(OpenRouterBatchError, match="400"):
        c.submit("m", REQS, state_path=tmp_path / "j.json")
    assert not (tmp_path / "j.json").exists()


def test_parse_results_content_parts_and_null_content() -> None:
    items = [
        {
            "custom_id": "p",
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": [{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}]}, "finish_reason": "stop"}]},
            },
        },
        {"custom_id": "n", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}}},
    ]
    out = parse_results({"results": items})
    assert out["p"].text == "hello" and out["p"].cost_usd is None
    assert not out["n"].ok and "length" in (out["n"].error or "")
