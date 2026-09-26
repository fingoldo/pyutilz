"""A batch job is never paid for twice, and never collected for requests it was not submitted for.

glossum refsuite audit O-P1-3: a resumed job was checked by its request ids only, so a job submitted for other prompts
under the same ids was waited on and its answers written as answers to the new prompts. O-P1-4: a submit whose POST
died on our side after the server had created the batch left no trace, and the rerun submitted (and paid for) it again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from pyutilz.llm.openrouter_batch import (
    BatchRequest,
    BatchStateMismatchError,
    OpenRouterBatchClient,
    OpenRouterBatchError,
    load_job_state,
    request_hash,
)

REQS = [BatchRequest("q-a", {"messages": [{"role": "user", "content": "a"}]}), BatchRequest("q-b", {"messages": [{"role": "user", "content": "b"}]})]


class FakeApi:
    """POST /batches, GET /batches (the listing) and GET /batches/<id>; each behaviour switchable per test."""

    def __init__(self) -> None:
        self.posts = 0
        self.post_mode = "ok"  # ok | drop | 500
        self.listing: list[dict[str, Any]] | None = []  # None: the listing endpoint answers 404
        self.gets: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST":
            self.posts += 1
            if self.post_mode == "drop":
                raise httpx.ReadTimeout("client-side timeout", request=request)
            if self.post_mode == "500":
                return httpx.Response(502, text="gateway")
            return httpx.Response(202, json={"id": f"batch_{self.posts}", "status": "validating"})
        if path.endswith("/batches"):
            return httpx.Response(404) if self.listing is None else httpx.Response(200, json={"data": self.listing})
        batch_id = path.rsplit("/", 1)[1]
        self.gets.append(batch_id)
        results = [{"custom_id": r.custom_id, "response": {"status_code": 200, "body": {"choices": [{"message": {"content": batch_id}}]}}} for r in REQS]
        return httpx.Response(200, json={"id": batch_id, "status": "completed", "results": results})


def _client(api: FakeApi) -> OpenRouterBatchClient:
    return OpenRouterBatchClient("k", transport=httpx.MockTransport(api.handler), sleep=lambda _s: None)


def _interrupted_submit(api: FakeApi, state: Path) -> dict[str, Any]:
    """A submit whose POST got no answer: it raises, and the ``submitting`` marker is what it leaves."""
    api.post_mode = "drop"
    with pytest.raises(OpenRouterBatchError, match="no answer"):
        _client(api).run("m/x", REQS, state_path=state)
    api.post_mode = "ok"
    marker = load_job_state(state)
    assert marker is not None
    return marker


def test_a_submitted_job_records_the_hash_of_its_requests(tmp_path: Path) -> None:
    state = tmp_path / "job.json"
    api = FakeApi()
    _client(api).submit("m/x", REQS, state_path=state, extra_state={"run_keys_sha256": "abc"})
    saved = load_job_state(state)
    assert saved is not None and saved["request_hash"] == request_hash("m/x", REQS) and saved["run_keys_sha256"] == "abc"


def test_a_job_for_other_bodies_under_the_same_ids_is_refused(tmp_path: Path) -> None:
    state = tmp_path / "job.json"
    api = FakeApi()
    _client(api).submit("m/x", REQS, state_path=state)
    changed = [BatchRequest("q-a", {"messages": [{"role": "user", "content": "a, reworded"}]}), REQS[1]]
    with pytest.raises(BatchStateMismatchError):
        _client(api).run("m/x", changed, state_path=state)
    assert api.gets == [] and api.posts == 1, "neither waited on nor submitted again"


def test_the_marker_is_written_before_the_post_and_kept_when_the_post_gets_no_answer(tmp_path: Path) -> None:
    marker = _interrupted_submit(FakeApi(), tmp_path / "job.json")
    assert marker["status"] == "submitting" and "batch_id" not in marker and marker["request_hash"] == request_hash("m/x", REQS)


def test_a_5xx_submit_keeps_the_marker(tmp_path: Path) -> None:
    api = FakeApi()
    api.post_mode = "500"
    with pytest.raises(OpenRouterBatchError, match="502"):
        _client(api).submit("m/x", REQS, state_path=tmp_path / "a.json")
    assert (load_job_state(tmp_path / "a.json") or {}).get("status") == "submitting", "a gateway error may still have created the batch"


def test_a_rerun_adopts_the_batch_the_interrupted_submit_created(tmp_path: Path) -> None:
    state = tmp_path / "job.json"
    api = FakeApi()
    marker = _interrupted_submit(api, state)
    api.listing = [
        {"id": "batch_other_model", "model": "m/y", "created_at": marker["submitting_at"] + 1},
        {"id": "batch_long_ago", "model": "m/x", "created_at": marker["submitting_at"] - 86_400},
        {"id": "batch_created", "model": "m/x", "created_at": marker["submitting_at"] + 2, "request_counts": {"total": 2}},
    ]
    out = _client(api).run("m/x", REQS, state_path=state)
    assert api.posts == 1, "the batch the lost POST created is adopted, not paid for a second time"
    assert out["q-a"].text == "batch_created" and (load_job_state(state) or {}).get("batch_id") == "batch_created"


def test_a_rerun_submits_when_the_interrupted_submit_created_nothing(tmp_path: Path) -> None:
    state = tmp_path / "job.json"
    api = FakeApi()
    _interrupted_submit(api, state)
    out = _client(api).run("m/x", REQS, state_path=state)
    assert api.posts == 2 and out["q-a"].text == "batch_2"


@pytest.mark.parametrize("listing", [None, "two"])
def test_a_rerun_refuses_when_it_cannot_tell(tmp_path: Path, listing: str | None) -> None:
    state = tmp_path / "job.json"
    api = FakeApi()
    marker = _interrupted_submit(api, state)
    api.listing = None if listing is None else [{"id": f"b{i}", "model": "m/x", "created_at": marker["submitting_at"] + i} for i in range(2)]
    with pytest.raises(OpenRouterBatchError):
        _client(api).run("m/x", REQS, state_path=state)
    assert api.posts == 1 and api.gets == []


def test_a_listed_request_hash_decides_over_the_time_window(tmp_path: Path) -> None:
    state = tmp_path / "job.json"
    api = FakeApi()
    marker = _interrupted_submit(api, state)
    api.listing = [
        {"id": "b_same_time_other_hash", "model": "m/x", "created_at": marker["submitting_at"], "metadata": {"request_hash": "0" * 64}},
        {"id": "b_ours", "model": "m/x", "created_at": marker["submitting_at"] + 5, "metadata": {"request_hash": marker["request_hash"]}},
    ]
    _client(api).run("m/x", REQS, state_path=state)
    assert api.posts == 1 and api.gets[-1] == "b_ours"


def test_the_listing_reads_the_data_array() -> None:
    api = FakeApi()
    api.listing = [{"id": "x"}, "not an object"]  # type: ignore[list-item]
    assert _client(api).list_batches() == [{"id": "x"}]
