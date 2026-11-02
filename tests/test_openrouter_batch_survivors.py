"""Behaviour tests for the OpenRouter batch module: result parsing, payload building, persisted state and client flows.

Every HTTP interaction goes through ``httpx.MockTransport``, and every clock and sleep is injected, so nothing touches the network.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

httpx = pytest.importorskip("httpx")

from pyutilz.llm import openrouter_batch as ob
from pyutilz.llm.openrouter_batch import (
    BatchRequest,
    BatchResult,
    BatchStateMismatchError,
    OpenRouterBatchClient,
    OpenRouterBatchError,
    build_submit_payload,
    load_job_state,
    parse_result_item,
    parse_results,
    request_hash,
)


def _request(custom_id: str = "a", **body) -> BatchRequest:
    return BatchRequest(custom_id=custom_id, body={"messages": [{"role": "user", "content": "hi"}], **body})


def _client(handler, **kwargs) -> OpenRouterBatchClient:
    kwargs.setdefault("sleep", lambda seconds: None)
    return OpenRouterBatchClient("key", transport=httpx.MockTransport(handler), **kwargs)


class TestBatchResult:
    def test_only_a_complete_answer_is_ok(self):
        assert BatchResult("a", text="done").ok is True
        assert BatchResult("a", text="done", finish_reason="length").ok is False
        assert BatchResult("a", text="done", error="boom").ok is False
        assert BatchResult("a", text=None).ok is False

    def test_length_is_the_only_truncation(self):
        assert BatchResult("a", finish_reason="length").truncated is True
        assert BatchResult("a", finish_reason="stop").truncated is False


class TestAtomicWrite:
    def test_the_payload_lands_as_sorted_indented_json(self, tmp_path):
        path = tmp_path / "state.json"
        ob._atomic_write_json(path, {"b": 1, "a": [2]})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": [2], "b": 1}
        assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index('"b"')
        assert not (tmp_path / "state.json.tmp").exists()

    def test_the_temporary_file_is_the_target_with_a_tmp_suffix(self, tmp_path, monkeypatch):
        seen = []
        real_replace = ob.os.replace

        def record(src, dst):
            seen.append((str(src), str(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(ob.os, "replace", record)
        path = tmp_path / "state.json"
        ob._atomic_write_json(path, {"x": 1})
        assert seen == [(str(path) + ".tmp", str(path))]

    def test_the_parent_directory_is_created_when_missing(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "state.json"
        ob._atomic_write_json(path, {"x": 1})
        assert path.exists()


class TestLoadJobState:
    def test_a_missing_file_is_none(self, tmp_path):
        assert load_job_state(tmp_path / "absent.json") is None

    def test_a_json_object_is_returned(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text('{"batch_id": "b1"}', encoding="utf-8")
        assert load_job_state(str(path)) == {"batch_id": "b1"}

    def test_a_non_object_json_is_none(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert load_job_state(path) is None


class TestBuildSubmitPayload:
    def test_an_empty_batch_is_refused_with_the_message(self):
        with pytest.raises(ValueError, match="a batch needs at least one request"):
            build_submit_payload("m", [])

    def test_duplicate_ids_are_listed_in_the_error(self):
        with pytest.raises(ValueError, match=r"duplicated: \['x'\]"):
            build_submit_payload("m", [_request("x"), _request("x"), _request("y")])

    def test_streaming_is_refused_naming_the_request(self):
        with pytest.raises(ValueError, match="request 'a': streaming is not supported"):
            build_submit_payload("m", [_request("a", stream=True)])

    def test_the_default_endpoint_and_window_are_set(self):
        payload = build_submit_payload("m", [_request("a")])
        assert payload["endpoint"] == "/v1/chat/completions"
        assert payload["completion_window"] == "24h"
        assert payload["model"] == "m"

    def test_provider_routing_is_added_only_when_given(self):
        assert "provider" not in build_submit_payload("m", [_request("a")])
        assert build_submit_payload("m", [_request("a")], provider_only=["Azure"])["provider"] == {"only": ["Azure"]}
        assert "provider" not in build_submit_payload("m", [_request("a")], provider_only=[])

    def test_each_request_drops_its_own_model_and_keeps_the_rest(self):
        req = BatchRequest(custom_id="a", body={"model": "other", "max_tokens": 5})
        payload = build_submit_payload("m", [req])
        assert payload["requests"] == [{"custom_id": "a", "body": {"max_tokens": 5}}]

    def test_requests_are_the_last_key_of_the_payload(self):
        assert list(build_submit_payload("m", [_request("a")]).keys())[-1] == "requests"

    def test_a_custom_window_and_endpoint_are_used(self):
        payload = build_submit_payload("m", [_request("a")], endpoint="/v1/x", completion_window="12h")
        assert payload["endpoint"] == "/v1/x"
        assert payload["completion_window"] == "12h"


class TestRequestHash:
    def test_the_hash_is_stable_for_equal_inputs(self):
        assert request_hash("m", [_request("a")]) == request_hash("m", [_request("a")])

    def test_a_different_model_or_routing_or_body_changes_it(self):
        base = request_hash("m", [_request("a")])
        assert request_hash("n", [_request("a")]) != base
        assert request_hash("m", [_request("a")], provider_only=["Azure"]) != base
        assert request_hash("m", [_request("a", max_tokens=9)]) != base

    def test_it_is_a_sha256_hex_digest(self):
        digest = request_hash("m", [_request("a")])
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")


class TestCreatedAt:
    def test_a_bool_is_not_a_timestamp(self):
        assert ob._created_at({"created_at": True}) is None

    def test_a_number_is_epoch_seconds(self):
        assert ob._created_at({"created_at": 1700000000}) == 1700000000.0

    def test_an_iso_string_with_z_is_read_as_utc(self):
        expected = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
        assert ob._created_at({"created_at": "2024-01-02T03:04:05Z"}) == pytest.approx(expected)

    def test_an_iso_string_with_an_offset_is_read(self):
        expected = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
        assert ob._created_at({"created_at": "2024-01-02T03:04:05+00:00"}) == pytest.approx(expected)

    def test_an_unreadable_string_is_none_and_logged_at_debug(self, caplog):
        caplog.set_level(logging.DEBUG, logger=ob.logger.name)
        assert ob._created_at({"created_at": "yesterday"}) is None
        assert "_created_at: timestamp is not ISO-8601" in caplog.text

    def test_missing_or_other_types_are_none(self):
        assert ob._created_at({}) is None
        assert ob._created_at({"created_at": None}) is None


class TestRequestTotal:
    def test_an_integer_total_is_returned(self):
        assert ob._request_total({"request_counts": {"total": 7}}) == 7

    def test_a_bool_total_is_refused(self):
        assert ob._request_total({"request_counts": {"total": True}}) is None

    def test_counts_that_are_not_a_mapping_give_none(self):
        assert ob._request_total({"request_counts": [1, 2]}) is None
        assert ob._request_total({}) is None


class TestTextOf:
    def test_none_stays_none_and_a_string_is_kept(self):
        assert ob._text_of(None) is None
        assert ob._text_of("plain") == "plain"

    def test_content_parts_are_joined_by_their_text(self):
        assert ob._text_of([{"text": "a"}, {"text": "b"}]) == "ab"

    def test_non_dict_parts_are_ignored(self):
        assert ob._text_of([{"text": "a"}, "junk", {"other": 1}]) == "a"

    def test_any_other_value_is_stringified(self):
        assert ob._text_of(42) == "42"


def _item(**overrides) -> dict:
    base = {
        "custom_id": "a",
        "response": {
            "status_code": 200,
            "body": {
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "cost": 0.5},
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            },
        },
    }
    base.update(overrides)
    return base


class TestParseResultItem:
    def test_a_successful_item_maps_text_usage_and_cost(self):
        res = parse_result_item(_item())
        assert res.custom_id == "a"
        assert res.text == "hello"
        assert res.prompt_tokens == 3
        assert res.completion_tokens == 4
        assert res.cost_usd == 0.5
        assert res.finish_reason == "stop"
        assert res.status_code == 200
        assert res.error is None
        assert res.raw == _item()

    def test_a_string_error_is_kept_verbatim(self):
        assert parse_result_item(_item(error="rate limited")).error == "rate limited"

    def test_a_structured_error_is_dumped_as_sorted_json(self):
        res = parse_result_item(_item(error={"b": 1, "a": 2}))
        assert res.error == '{"a":2,"b":1}'

    def test_a_missing_response_is_named(self):
        res = parse_result_item({"custom_id": "a"})
        assert res.error == "result has neither response nor error"

    def test_a_non_object_body_is_unparseable(self):
        res = parse_result_item({"custom_id": "a", "response": {"status_code": 200, "body": "oops"}})
        assert res.error == "unparseable response body: oops"

    @pytest.mark.parametrize("code", [199, 300, 404, 500])
    def test_a_status_outside_two_hundred_is_an_http_error(self, code):
        res = parse_result_item(_item(response={"status_code": code, "body": {"error": {"message": "x"}}}))
        assert res.error == f'HTTP {code}: {{"message":"x"}}'

    @pytest.mark.parametrize("code", [200, 201, 299])
    def test_a_status_inside_two_hundred_is_ok(self, code):
        assert parse_result_item(_item(response={"status_code": code, "body": _item()["response"]["body"]})).error is None

    def test_a_missing_status_skips_the_check(self):
        res = parse_result_item(_item(response={"body": _item()["response"]["body"]}))
        assert res.error is None
        assert res.status_code is None

    def test_no_choices_is_an_error_carrying_the_body_error(self):
        res = parse_result_item(_item(response={"status_code": 200, "body": {"error": "upstream down"}}))
        assert res.error == 'no choices in response body: "upstream down"'

    def test_empty_content_reports_the_finish_reason(self):
        res = parse_result_item(_item(response={"status_code": 200, "body": {"choices": [{"message": {}, "finish_reason": "stop"}]}}))
        assert res.error == "empty content (finish_reason=stop)"

    def test_a_length_finish_is_a_truncated_partial_answer(self):
        res = parse_result_item(_item(response={"status_code": 200, "body": {"choices": [{"message": {"content": "part"}, "finish_reason": "length"}]}}))
        assert res.truncated is True
        assert res.text == "part"
        assert res.ok is False

    def test_reasoning_is_read_from_the_message(self):
        res = parse_result_item(
            _item(response={"status_code": 200, "body": {"choices": [{"message": {"content": "x", "reasoning": "because"}, "finish_reason": "stop"}]}})
        )
        assert res.reasoning == "because"

    def test_missing_usage_and_cost_default_to_zero_and_none(self):
        res = parse_result_item(_item(response={"status_code": 200, "body": {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}}))
        assert (res.prompt_tokens, res.completion_tokens, res.cost_usd) == (0, 0, None)


class TestParseResults:
    def test_results_are_keyed_by_custom_id(self):
        out = parse_results({"status": "completed", "results": [_item(custom_id="a"), _item(custom_id="b")]})
        assert set(out) == {"a", "b"}

    def test_an_expected_id_missing_from_the_batch_gets_an_error_with_the_status(self):
        out = parse_results({"status": "failed", "results": []}, expected_ids=["c"])
        assert out["c"].error == "missing from batch results (batch status=failed)"

    def test_no_results_key_is_an_empty_mapping(self):
        assert parse_results({}) == {}


class TestSubmit:
    def test_the_batch_id_is_returned_and_the_payload_is_posted(self):
        sent = {}

        def handler(request):
            sent["path"] = request.url.path
            sent["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "batch-1"})

        client = _client(handler)
        assert client.submit("m", [_request("a")]) == "batch-1"
        assert sent["path"] == "/api/v1/batches"
        assert sent["body"]["model"] == "m"
        assert sent["body"]["requests"][0]["custom_id"] == "a"

    def test_the_submitted_state_is_persisted_with_the_id(self, tmp_path):
        client = _client(lambda request: httpx.Response(200, json={"id": "batch-1"}))
        path = tmp_path / "state.json"
        client.submit("m", [_request("a")], state_path=path, extra_state={"job": "x"})
        state = json.loads(path.read_text(encoding="utf-8"))
        assert state["batch_id"] == "batch-1"
        assert state["status"] == "submitted"
        assert state["custom_ids"] == ["a"]
        assert state["job"] == "x"
        assert state["model"] == "m"
        assert "submitted_at" in state

    def test_a_submitting_marker_is_written_before_the_post(self, tmp_path):
        path = tmp_path / "state.json"
        seen = {}

        def handler(request):
            seen["marker"] = json.loads(path.read_text(encoding="utf-8"))
            return httpx.Response(200, json={"id": "batch-1"})

        _client(handler).submit("m", [_request("a")], state_path=path)
        assert seen["marker"]["status"] == "submitting"
        assert "batch_id" not in seen["marker"]
        assert "request_hash" in seen["marker"]

    def test_a_refusal_by_the_server_removes_the_marker(self, tmp_path):
        path = tmp_path / "state.json"
        client = _client(lambda request: httpx.Response(400, json={"error": "bad"}))
        with pytest.raises(OpenRouterBatchError, match="submit: HTTP 400"):
            client.submit("m", [_request("a")], state_path=path)
        assert not path.exists()

    def test_a_server_error_keeps_the_marker(self, tmp_path):
        path = tmp_path / "state.json"
        client = _client(lambda request: httpx.Response(500, json={"error": "oops"}))
        with pytest.raises(OpenRouterBatchError):
            client.submit("m", [_request("a")], state_path=path)
        assert path.exists()

    def test_no_answer_at_all_names_the_rerun(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(OpenRouterBatchError, match=r"got no answer .* a rerun resolves it"):
            _client(handler).submit("m", [_request("a"), _request("b")], state_path=None)

    def test_a_response_without_an_id_is_refused(self):
        client = _client(lambda request: httpx.Response(200, json={"status": "x"}))
        with pytest.raises(OpenRouterBatchError, match="no batch id in response"):
            client.submit("m", [_request("a")])

    def test_a_non_object_response_is_refused(self):
        client = _client(lambda request: httpx.Response(200, json=[1, 2]))
        with pytest.raises(OpenRouterBatchError, match="submit: non-object response"):
            client.submit("m", [_request("a")])


class TestListAndGet:
    def test_list_batches_returns_the_data_objects_only(self):
        client = _client(lambda request: httpx.Response(200, json={"data": [{"id": "1"}, "junk"]}))
        assert client.list_batches() == [{"id": "1"}]

    def test_the_batches_key_is_accepted_too(self):
        client = _client(lambda request: httpx.Response(200, json={"batches": [{"id": "2"}]}))
        assert client.list_batches() == [{"id": "2"}]

    def test_a_listing_that_is_not_a_list_is_empty(self):
        client = _client(lambda request: httpx.Response(200, json={"data": "nope"}))
        assert client.list_batches() == []

    def test_the_limit_is_sent_as_a_query_parameter(self):
        seen = {}

        def handler(request):
            seen["limit"] = request.url.params["limit"]
            return httpx.Response(200, json={"data": []})

        _client(handler).list_batches(limit=5)
        assert seen["limit"] == "5"

    def test_get_returns_the_batch_object(self):
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": "completed"}))
        assert client.get("b") == {"id": "b", "status": "completed"}


class TestRecoverSubmit:
    def _batch(self, **overrides):
        base = {"id": "found", "model": "m", "created_at": 1000.0, "request_counts": {"total": 1}}
        base.update(overrides)
        return base

    def _state(self, since=1000.0):
        return {"submitting_at": since}

    def test_one_matching_batch_in_the_window_is_adopted(self):
        client = _client(lambda request: httpx.Response(200, json={"data": [self._batch()]}))
        assert client.recover_submit("m", [_request("a")], self._state()) == "found"

    def test_several_matches_refuse_to_guess(self):
        batches = [self._batch(id="one"), self._batch(id="two")]
        client = _client(lambda request: httpx.Response(200, json={"data": batches}))
        with pytest.raises(OpenRouterBatchError, match="matches 2 listed batches"):
            client.recover_submit("m", [_request("a")], self._state())

    def test_an_unreadable_listing_refuses_with_the_dashboard_hint(self):
        client = _client(lambda request: httpx.Response(500, json={}))
        with pytest.raises(OpenRouterBatchError, match="batch list cannot be read"):
            client.recover_submit("m", [_request("a")], self._state())

    def test_no_match_means_submit_again(self):
        client = _client(lambda request: httpx.Response(200, json={"data": []}))
        assert client.recover_submit("m", [_request("a")], self._state()) is None

    def test_another_model_is_not_ours(self):
        client = _client(lambda request: httpx.Response(200, json={"data": [self._batch(model="other")]}))
        assert client.recover_submit("m", [_request("a")], self._state()) is None

    def test_a_batch_created_before_the_window_is_not_ours(self):
        early = self._batch(created_at=1000.0 - ob.SUBMIT_MATCH_BEFORE_S - 1)
        client = _client(lambda request: httpx.Response(200, json={"data": [early]}))
        assert client.recover_submit("m", [_request("a")], self._state()) is None

    def test_a_batch_created_after_the_window_is_not_ours(self):
        late = self._batch(created_at=1000.0 + ob.SUBMIT_MATCH_AFTER_S + 1)
        client = _client(lambda request: httpx.Response(200, json={"data": [late]}))
        assert client.recover_submit("m", [_request("a")], self._state()) is None

    def test_the_window_edges_are_inclusive(self):
        edge = self._batch(created_at=1000.0 + ob.SUBMIT_MATCH_AFTER_S)
        client = _client(lambda request: httpx.Response(200, json={"data": [edge]}))
        assert client.recover_submit("m", [_request("a")], self._state()) == "found"

    def test_a_different_request_count_is_not_ours(self):
        client = _client(lambda request: httpx.Response(200, json={"data": [self._batch(request_counts={"total": 9})]}))
        assert client.recover_submit("m", [_request("a")], self._state()) is None

    def test_a_batch_without_a_count_still_matches(self):
        client = _client(lambda request: httpx.Response(200, json={"data": [self._batch(request_counts=None)]}))
        assert client.recover_submit("m", [_request("a")], self._state()) == "found"


class TestWait:
    def test_a_terminal_status_is_returned_and_reported(self):
        seen = []
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": "completed"}))
        final = client.wait("b", on_poll=lambda data: seen.append(data["status"]))
        assert final["status"] == "completed"
        assert seen == ["completed"]

    def test_polling_continues_until_a_terminal_status(self):
        statuses = iter(["in_progress", "in_progress", "completed"])
        sleeps = []
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": next(statuses)}), sleep=sleeps.append)
        assert client.wait("b")["status"] == "completed"
        assert len(sleeps) == 2

    def test_the_deadline_raises_and_the_job_is_not_cancelled(self):
        ticks = iter([0.0, 0.0, 100.0])
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": "in_progress"}), clock=lambda: next(ticks))
        with pytest.raises(OpenRouterBatchError, match="not finished after 100s"):
            client.wait("b", deadline_s=50.0)

    def test_a_transient_poll_error_is_retried(self):
        calls = iter([httpx.Response(500, json={}), httpx.Response(200, json={"id": "b", "status": "completed"})])
        client = _client(lambda request: next(calls))
        assert client.wait("b")["status"] == "completed"

    def test_the_backoff_grows_and_is_capped(self):
        sleeps = []
        statuses = iter(["in_progress"] * 4 + ["completed"])
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": next(statuses)}), sleep=sleeps.append)
        client.wait("b", initial_interval_s=10.0, max_interval_s=25.0, backoff=2.0)
        assert all(s <= 25.0 * 1.1 for s in sleeps)
        assert sleeps[0] == pytest.approx(10.0, rel=0.11)
        assert sleeps[1] == pytest.approx(20.0, rel=0.11)
        assert sleeps[2] == pytest.approx(25.0, rel=0.11)


class TestRun:
    def test_a_fresh_run_submits_waits_and_returns_results(self, tmp_path):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(200, json={"id": "batch-1"})
            return httpx.Response(200, json={"id": "batch-1", "status": "completed", "results": [_item(custom_id="a")]})

        results = _client(handler).run("m", [_request("a")], state_path=tmp_path / "s.json")
        assert results["a"].text == "hello"

    def test_a_resumed_run_polls_the_stored_batch_without_submitting(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"batch_id": "stored", "model": "m", "custom_ids": ["a"]}), encoding="utf-8")
        methods = []

        def handler(request):
            methods.append(request.method)
            return httpx.Response(200, json={"id": "stored", "status": "completed", "results": [_item(custom_id="a")]})

        _client(handler).run("m", [_request("a")], state_path=path)
        assert methods == ["GET"]

    def test_a_different_model_in_the_state_is_a_mismatch(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"batch_id": "stored", "model": "other", "custom_ids": ["a"]}), encoding="utf-8")
        with pytest.raises(BatchStateMismatchError, match="different model/request set"):
            _client(lambda request: httpx.Response(200, json={})).run("m", [_request("a")], state_path=path)

    def test_a_different_request_set_is_a_mismatch(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"batch_id": "stored", "model": "m", "custom_ids": ["zzz"]}), encoding="utf-8")
        with pytest.raises(BatchStateMismatchError):
            _client(lambda request: httpx.Response(200, json={})).run("m", [_request("a")], state_path=path)

    def test_a_changed_request_body_is_a_mismatch_when_the_hash_is_stored(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(
            json.dumps({"batch_id": "stored", "model": "m", "custom_ids": ["a"], "request_hash": "0" * 64}),
            encoding="utf-8",
        )
        with pytest.raises(BatchStateMismatchError):
            _client(lambda request: httpx.Response(200, json={})).run("m", [_request("a")], state_path=path)

    def test_a_non_completed_terminal_status_still_returns_results_and_warns(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger=ob.logger.name)

        def handler(request):
            if request.method == "POST":
                return httpx.Response(200, json={"id": "batch-1"})
            return httpx.Response(200, json={"id": "batch-1", "status": "expired", "results": []})

        results = _client(handler).run("m", [_request("a")], state_path=tmp_path / "s.json")
        assert results["a"].error == "missing from batch results (batch status=expired)"
        assert "ended with status=expired" in caplog.text

    def test_a_submitting_marker_is_resolved_by_adopting_the_found_batch(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"status": "submitting", "model": "m", "custom_ids": ["a"], "submitting_at": 1000.0}), encoding="utf-8")

        def handler(request):
            if request.url.path.endswith("/batches") and request.method == "GET" and "limit" in request.url.params:
                return httpx.Response(200, json={"data": [{"id": "found", "model": "m", "created_at": 1000.0, "request_counts": {"total": 1}}]})
            return httpx.Response(200, json={"id": "found", "status": "completed", "results": [_item(custom_id="a")]})

        results = _client(handler).run("m", [_request("a")], state_path=path)
        assert results["a"].text == "hello"
        assert json.loads(path.read_text(encoding="utf-8"))["batch_id"] == "found"


class TestClientConstruction:
    def test_an_empty_key_is_refused_with_the_setting_name(self):
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterBatchClient("", transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    def test_the_bearer_key_is_sent_on_every_request(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers["authorization"]
            return httpx.Response(200, json={"data": []})

        _client(handler).list_batches()
        assert seen["auth"] == "Bearer key"

    def test_a_context_manager_closes_the_client(self):
        client = _client(lambda request: httpx.Response(200, json={"data": []}))
        with client as entered:
            assert entered is client
        assert client._client.is_closed

    def test_a_missing_key_falls_back_to_the_settings(self, monkeypatch):
        class _Secret:
            def get_secret_value(self):
                return "from-settings"

        class _Settings:
            openrouter_api_key = _Secret()

        monkeypatch.setattr("pyutilz.llm.config.get_llm_settings", lambda: _Settings())
        client = OpenRouterBatchClient(None, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []})))
        assert client._client.headers["authorization"] == "Bearer from-settings"
        client.close()

    def test_no_key_anywhere_is_refused(self, monkeypatch):
        class _Settings:
            openrouter_api_key = None

        monkeypatch.setattr("pyutilz.llm.config.get_llm_settings", lambda: _Settings())
        with pytest.raises(ValueError, match="API key not provided"):
            OpenRouterBatchClient(None, transport=httpx.MockTransport(lambda request: httpx.Response(200)))
