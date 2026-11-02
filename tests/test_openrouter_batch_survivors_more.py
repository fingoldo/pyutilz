"""Exact-message, constant and arithmetic pins for the OpenRouter batch module (mutation survivors of the first test file)."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import FrozenInstanceError

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


def _item(custom_id: str = "a") -> dict:
    return {"custom_id": custom_id, "response": {"status_code": 200, "body": {"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}}}


def _client(handler, **kwargs) -> OpenRouterBatchClient:
    kwargs.setdefault("sleep", lambda seconds: None)
    return OpenRouterBatchClient("key", transport=httpx.MockTransport(handler), **kwargs)


class TestModuleConstants:
    def test_the_terminal_statuses_are_the_four_documented_ones(self):
        assert ob.TERMINAL_STATUSES == frozenset({"completed", "failed", "expired", "cancelled"})

    def test_the_submit_match_window_is_two_minutes_before_and_fifteen_after(self):
        assert ob.SUBMIT_MATCH_BEFORE_S == 120.0
        assert ob.SUBMIT_MATCH_AFTER_S == 900.0

    def test_a_batch_just_outside_the_window_before_is_not_ours(self):
        state = {"submitting_at": 1000.0}
        listed = [{"id": "x", "model": "m", "created_at": 1000.0 - ob.SUBMIT_MATCH_BEFORE_S - 0.05, "request_counts": {"total": 1}}]
        client = _client(lambda request: httpx.Response(200, json={"data": listed}))
        assert client.recover_submit("m", [_request("a")], state) is None

    def test_a_batch_just_outside_the_window_after_is_not_ours(self):
        state = {"submitting_at": 1000.0}
        listed = [{"id": "x", "model": "m", "created_at": 1000.0 + ob.SUBMIT_MATCH_AFTER_S + 0.05, "request_counts": {"total": 1}}]
        client = _client(lambda request: httpx.Response(200, json={"data": listed}))
        assert client.recover_submit("m", [_request("a")], state) is None

    def test_a_batch_at_the_lower_window_edge_is_ours(self):
        state = {"submitting_at": 1000.0}
        listed = [{"id": "x", "model": "m", "created_at": 1000.0 - ob.SUBMIT_MATCH_BEFORE_S, "request_counts": {"total": 1}}]
        client = _client(lambda request: httpx.Response(200, json={"data": listed}))
        assert client.recover_submit("m", [_request("a")], state) == "x"

    def test_a_missing_submitting_time_uses_zero_as_the_anchor(self):
        listed = [{"id": "x", "model": "m", "created_at": -119.95, "request_counts": {"total": 1}}]
        client = _client(lambda request: httpx.Response(200, json={"data": listed}))
        assert client.recover_submit("m", [_request("a")], {}) == "x"

    def test_the_submit_defaults_to_the_chat_endpoint(self):
        assert ob.CHAT_ENDPOINT == "/v1/chat/completions"
        assert ob.BASE_URL == "https://openrouter.ai/api/v1"


class TestSignatureDefaults:
    def test_the_client_default_timeout_is_120_seconds(self):
        assert inspect.signature(OpenRouterBatchClient.__init__).parameters["timeout"].default == 120.0

    def test_the_wait_defaults_are_the_documented_schedule(self):
        params = inspect.signature(OpenRouterBatchClient.wait).parameters
        assert params["deadline_s"].default == 26 * 3600.0
        assert params["initial_interval_s"].default == 10.0
        assert params["max_interval_s"].default == 300.0
        assert params["backoff"].default == 1.5

    def test_the_run_defaults_match_the_wait_defaults(self):
        params = inspect.signature(OpenRouterBatchClient.run).parameters
        assert params["deadline_s"].default == 26 * 3600.0
        assert params["initial_interval_s"].default == 10.0
        assert params["max_interval_s"].default == 300.0

    def test_the_list_limit_defaults_to_one_hundred(self):
        assert inspect.signature(OpenRouterBatchClient.list_batches).parameters["limit"].default == 100

    def test_keyword_only_markers_refuse_positional_options(self):
        with pytest.raises(TypeError):
            build_submit_payload("m", [_request("a")], "/v1/x")
        with pytest.raises(TypeError):
            request_hash("m", [_request("a")], "/v1/x")
        with pytest.raises(TypeError):
            _client(lambda request: httpx.Response(200, json={"id": "b"})).submit("m", [_request("a")], None)
        with pytest.raises(TypeError):
            _client(lambda request: httpx.Response(200, json={})).wait("b", 5)
        with pytest.raises(TypeError):
            _client(lambda request: httpx.Response(200, json={})).run("m", [_request("a")], "/tmp/x.json")

    def test_the_client_timeout_reaches_the_http_client(self):
        client = _client(lambda request: httpx.Response(200, json={"data": []}), timeout=120.0)
        assert client._client.timeout.read == 120.0


class TestBatchResultDefaults:
    def test_the_token_counts_default_to_zero(self):
        res = BatchResult("a")
        assert res.prompt_tokens == 0
        assert res.completion_tokens == 0

    def test_a_batch_request_is_immutable(self):
        req = _request("a")
        with pytest.raises(FrozenInstanceError):
            req.custom_id = "b"  # frozen dataclass


class TestPayloadDetails:
    def test_the_duplicate_list_is_capped_at_five_entries(self):
        dupes = [str(i) for i in range(6)]
        reqs = [_request(x) for x in dupes] + [_request(x) for x in dupes]
        with pytest.raises(ValueError) as info:
            build_submit_payload("m", reqs)
        assert str(info.value).count("'") == 2 * 5

    def test_the_hash_ignores_body_key_order(self):
        a = BatchRequest("a", {"max_tokens": 5, "temperature": 0.1, "messages": []})
        b = BatchRequest("a", {"messages": [], "temperature": 0.1, "max_tokens": 5})
        assert request_hash("m", [a]) == request_hash("m", [b])


class TestAtomicWriteFormat:
    def test_the_state_file_is_indented_over_several_lines(self, tmp_path):
        path = tmp_path / "s.json"
        ob._atomic_write_json(path, {"a": 1, "b": 2})
        assert "\n" in path.read_text(encoding="utf-8")


class TestCreatedAtLogging:
    def test_the_unreadable_timestamp_debug_names_the_exception_type(self, caplog):
        caplog.set_level(logging.DEBUG, logger=ob.logger.name)
        ob._created_at({"created_at": "yesterday"})
        assert "ValueError" in caplog.text


class TestParseResultMessages:
    def test_an_unparseable_body_message_is_cut_at_two_hundred_characters(self):
        res = parse_result_item({"custom_id": "a", "response": {"status_code": 200, "body": "x" * 300}})
        assert res.error == "unparseable response body: " + "x" * 200

    def test_an_http_error_text_is_sorted_and_cut_at_five_hundred_characters(self):
        body = {"b": "y" * 600, "a": 1}
        res = parse_result_item({"custom_id": "a", "response": {"status_code": 500, "body": body}})
        assert res.error.startswith('HTTP 500: {"a":1,"b":"yyy')
        assert len(res.error) == len("HTTP 500: ") + 500

    def test_a_no_choices_message_sorts_keys_and_is_cut_at_three_hundred(self):
        body = {"z": 1, "a": "q" * 400}
        res = parse_result_item({"custom_id": "a", "response": {"status_code": 200, "body": body}})
        assert res.error.startswith('no choices in response body: {"a":"qqq')
        assert len(res.error) == len("no choices in response body: ") + 300

    def test_a_string_error_is_passed_through_untouched(self):
        res = parse_result_item({"custom_id": "a", "error": "boom"})
        assert res.error == "boom"


class TestParseResultsMessages:
    def test_the_missing_id_error_names_the_batch_status(self):
        out = parse_results({"status": "expired"}, expected_ids=["q"])
        assert out["q"].custom_id == "q"
        assert out["q"].error == "missing from batch results (batch status=expired)"


class TestSubmitMessages:
    def test_a_submit_that_gets_no_answer_names_the_request_count_and_model(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(OpenRouterBatchError) as info:
            _client(handler).submit("m", [_request("a"), _request("b")])
        assert str(info.value).startswith("submit of 2 request(s) for m got no answer (")
        assert str(info.value).endswith("); a rerun resolves it")

    def test_the_submit_log_names_the_batch_count_and_model(self, caplog):
        caplog.set_level(logging.INFO, logger=ob.logger.name)
        _client(lambda request: httpx.Response(200, json={"id": "batch-1"})).submit("m", [_request("a"), _request("b")])
        assert "OpenRouter batch batch-1 submitted: 2 requests, model=m" in caplog.text

    def test_a_refused_submit_puts_the_status_in_the_error(self):
        with pytest.raises(OpenRouterBatchError) as info:
            _client(lambda request: httpx.Response(400, json={"error": "bad"})).submit("m", [_request("a")])
        assert str(info.value).startswith("submit: HTTP 400: ")

    def test_an_error_body_is_cut_at_five_hundred_characters(self):
        with pytest.raises(OpenRouterBatchError) as info:
            _client(lambda request: httpx.Response(500, text="z" * 900)).submit("m", [_request("a")])
        assert str(info.value) == "submit: HTTP 500: " + "z" * 500

    def test_a_missing_id_message_is_cut_at_three_hundred_characters(self):
        with pytest.raises(OpenRouterBatchError) as info:
            _client(lambda request: httpx.Response(200, json={"status": "w" * 500})).submit("m", [_request("a")])
        assert str(info.value).startswith("submit: no batch id in response: ")
        assert len(str(info.value)) == len("submit: no batch id in response: ") + 300

    def test_a_refused_submit_message_is_named_in_the_error(self):
        with pytest.raises(OpenRouterBatchError) as info:
            _client(lambda request: httpx.Response(400, json={"error": "bad"})).submit("m", [_request("a")])
        assert "submit" in str(info.value)

    def test_the_marker_carries_the_submitting_time_before_the_post(self, tmp_path):
        path = tmp_path / "s.json"
        seen = {}

        def handler(request):
            seen["marker"] = json.loads(path.read_text(encoding="utf-8"))
            return httpx.Response(200, json={"id": "batch-1"})

        _client(handler).submit("m", [_request("a")], state_path=path)
        assert "submitting_at" in seen["marker"]

    def test_a_refusal_removes_the_marker_even_when_the_file_was_already_removed(self, tmp_path):
        path = tmp_path / "s.json"

        def handler(request):
            path.unlink(missing_ok=True)
            return httpx.Response(400, json={"error": "bad"})

        with pytest.raises(OpenRouterBatchError):
            _client(handler).submit("m", [_request("a")], state_path=path)
        assert not path.exists()


class TestListMessages:
    def test_the_list_endpoint_is_batches(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            return httpx.Response(200, json={"data": []})

        _client(handler).list_batches()
        assert seen["path"] == "/api/v1/batches"

    def test_a_failed_listing_names_the_operation(self):
        with pytest.raises(OpenRouterBatchError) as info:
            _client(lambda request: httpx.Response(500, text="down")).list_batches()
        assert str(info.value) == "list batches: HTTP 500: down"


class TestRecoverMessages:
    def test_an_unreadable_listing_message_is_exact(self):
        client = _client(lambda request: httpx.Response(500, text="down"))
        with pytest.raises(OpenRouterBatchError) as info:
            client.recover_submit("m", [_request("a"), _request("b")], {"submitting_at": 1.0})
        assert str(info.value) == (
            "a submit of 2 request(s) for m was interrupted and the batch list cannot be read "
            "(list batches: HTTP 500: down); it may have been created: check the dashboard, then delete the state file to submit again"
        )

    def test_the_several_matches_message_lists_the_ids(self):
        batches = [
            {"id": "one", "model": "m", "created_at": 1000.0, "request_counts": {"total": 1}},
            {"id": "two", "model": "m", "created_at": 1000.0, "request_counts": {"total": 1}},
        ]
        client = _client(lambda request: httpx.Response(200, json={"data": batches}))
        with pytest.raises(OpenRouterBatchError) as info:
            client.recover_submit("m", [_request("a")], {"submitting_at": 1000.0})
        assert str(info.value) == "an interrupted submit for m matches 2 listed batches ['one', 'two']; resolve it by hand"

    def test_a_match_is_adopted_and_logged_as_a_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=ob.logger.name)
        batches = [{"id": "found", "model": "m", "created_at": 1000.0, "request_counts": {"total": 1}}]
        client = _client(lambda request: httpx.Response(200, json={"data": batches}))
        assert client.recover_submit("m", [_request("a")], {"submitting_at": 1000.0}) == "found"
        assert "an interrupted submit had created batch found; adopting it instead of submitting again" in caplog.text

    def test_no_match_is_logged_at_info(self, caplog):
        caplog.set_level(logging.INFO, logger=ob.logger.name)
        client = _client(lambda request: httpx.Response(200, json={"data": []}))
        assert client.recover_submit("m", [_request("a")], {"submitting_at": 1000.0}) is None
        assert "an interrupted submit for m created no batch; submitting it now" in caplog.text

    def test_a_later_batch_still_matches_after_an_earlier_one_does_not(self):
        batches = [
            {"id": "other", "model": "x", "created_at": 1000.0, "request_counts": {"total": 1}},
            {"id": "found", "model": "m", "created_at": 1000.0, "request_counts": {"total": 1}},
        ]
        client = _client(lambda request: httpx.Response(200, json={"data": batches}))
        assert client.recover_submit("m", [_request("a")], {"submitting_at": 1000.0}) == "found"


class TestGetAndWaitMessages:
    def test_the_get_path_and_its_error_name_the_batch(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            return httpx.Response(404, text="gone")

        with pytest.raises(OpenRouterBatchError) as info:
            _client(handler).get("b-9")
        assert seen["path"] == "/api/v1/batches/b-9"
        assert str(info.value) == "get b-9: HTTP 404: gone"

    def test_the_deadline_message_is_exact(self):
        ticks = iter([0.0, 0.0, 100.0])
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": "in_progress"}), clock=lambda: next(ticks))
        with pytest.raises(OpenRouterBatchError) as info:
            client.wait("b", deadline_s=50.0)
        assert str(info.value) == "batch b not finished after 100s (deadline 50s)"

    @pytest.mark.parametrize("status", ["failed", "cancelled", "expired", "completed"])
    def test_every_terminal_status_ends_the_wait_without_sleeping(self, status):
        sleeps = []
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": status}), sleep=sleeps.append)
        assert client.wait("b")["status"] == status
        assert sleeps == []

    def test_a_deadline_exactly_reached_raises(self):
        ticks = iter([0.0, 50.0])
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": "in_progress"}), clock=lambda: next(ticks))
        with pytest.raises(OpenRouterBatchError):
            client.wait("b", deadline_s=50.0)

    def test_the_first_failed_poll_is_logged_as_a_warning_and_counted_one(self, caplog):
        caplog.set_level(logging.DEBUG, logger=ob.logger.name)
        responses = iter([httpx.Response(500, text="x"), httpx.Response(200, json={"id": "b", "status": "completed"})])
        _client(lambda request: next(responses)).wait("b")
        assert "poll of batch b failed (1 in a row)" in caplog.text
        records = [r for r in caplog.records if "poll of batch" in r.getMessage()]
        assert records[0].levelno == logging.WARNING

    def test_a_second_failed_poll_is_only_debug_and_counted_two(self, caplog):
        caplog.set_level(logging.DEBUG, logger=ob.logger.name)
        responses = iter([httpx.Response(500, text="x"), httpx.Response(500, text="y"), httpx.Response(200, json={"id": "b", "status": "completed"})])
        _client(lambda request: next(responses)).wait("b")
        records = [r for r in caplog.records if "poll of batch" in r.getMessage()]
        assert "(2 in a row)" in records[1].getMessage()
        assert records[1].levelno == logging.DEBUG

    def test_a_success_resets_the_failure_count_so_the_next_failure_is_one_again(self, caplog):
        caplog.set_level(logging.DEBUG, logger=ob.logger.name)
        responses = iter(
            [
                httpx.Response(500, text="x"),
                httpx.Response(200, json={"id": "b", "status": "in_progress"}),
                httpx.Response(500, text="y"),
                httpx.Response(200, json={"id": "b", "status": "completed"}),
            ]
        )
        _client(lambda request: next(responses)).wait("b")
        records = [r for r in caplog.records if "poll of batch" in r.getMessage()]
        assert "(1 in a row)" in records[1].getMessage()


class TestWaitArithmetic:
    def test_the_sleeps_use_the_jitter_lower_bound_and_grow_by_the_backoff(self, monkeypatch):
        monkeypatch.setattr(ob.random, "uniform", lambda low, high: low)
        sleeps = []
        statuses = iter(["in_progress", "in_progress", "completed"])
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": next(statuses)}), sleep=sleeps.append)
        client.wait("b", initial_interval_s=10.0, backoff=1.5)
        assert sleeps == pytest.approx([9.0, 13.5])

    def test_the_sleep_is_capped_by_the_remaining_deadline(self, monkeypatch):
        monkeypatch.setattr(ob.random, "uniform", lambda low, high: low)
        sleeps = []
        ticks = iter([0.0, 3.0, 3.0])
        statuses = iter(["in_progress", "completed"])
        client = _client(
            lambda request: httpx.Response(200, json={"id": "b", "status": next(statuses)}),
            sleep=sleeps.append,
            clock=lambda: next(ticks),
        )
        client.wait("b", deadline_s=5.0, initial_interval_s=9.0)
        assert sleeps == pytest.approx([2.0])

    def test_the_interval_is_capped_at_the_maximum(self, monkeypatch):
        monkeypatch.setattr(ob.random, "uniform", lambda low, high: low)
        sleeps = []
        statuses = iter(["in_progress", "in_progress", "completed"])
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": next(statuses)}), sleep=sleeps.append)
        client.wait("b", initial_interval_s=10.0, max_interval_s=12.0, backoff=2.0)
        assert sleeps == pytest.approx([9.0, 10.8])

    def test_the_jitter_is_drawn_from_ninety_to_one_hundred_ten_percent(self, monkeypatch):
        drawn = []

        def record(low, high):
            drawn.append((low, high))
            return low

        monkeypatch.setattr(ob.random, "uniform", record)
        statuses = iter(["in_progress", "completed"])
        _client(lambda request: httpx.Response(200, json={"id": "b", "status": next(statuses)})).wait("b")
        assert drawn == [(0.9, 1.1)]


class TestRunMessages:
    def test_the_mismatch_message_names_the_stored_batch(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"batch_id": "stored", "model": "other", "custom_ids": ["a"]}), encoding="utf-8")
        with pytest.raises(BatchStateMismatchError) as info:
            _client(lambda request: httpx.Response(200, json={})).run("m", [_request("a")], state_path=path)
        assert str(info.value) == (
            f"state file {path} holds batch stored for a different model/request set; collect or delete it before submitting a new batch"
        )

    def test_a_resumed_job_is_logged_at_info(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger=ob.logger.name)
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"batch_id": "stored", "model": "m", "custom_ids": ["a"]}), encoding="utf-8")
        _client(lambda request: httpx.Response(200, json={"id": "stored", "status": "completed", "results": []})).run("m", [_request("a")], state_path=path)
        assert f"resuming OpenRouter batch stored from {path}" in caplog.text

    def test_a_resolved_submit_keeps_the_submitted_state_fields(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"status": "submitting", "model": "m", "custom_ids": ["a"], "submitting_at": 1000.0}), encoding="utf-8")

        def handler(request):
            if request.method == "GET" and request.url.path.endswith("/batches") and "limit" in request.url.params:
                return httpx.Response(200, json={"data": [{"id": "found", "model": "m", "created_at": 1000.0, "request_counts": {"total": 1}}]})
            return httpx.Response(200, json={"id": "found", "status": "completed", "results": []})

        _client(handler).run("m", [_request("a")], state_path=path)
        state = load_job_state(path)
        assert state["status"] == "submitted"
        assert "submitted_at" in state

    def test_a_terminal_non_completed_status_is_warned_with_its_error(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger=ob.logger.name)

        def handler(request):
            if request.method == "POST":
                return httpx.Response(200, json={"id": "batch-1"})
            return httpx.Response(200, json={"id": "batch-1", "status": "failed", "error": "quota", "results": []})

        _client(handler).run("m", [_request("a")], state_path=tmp_path / "s.json")
        assert "batch batch-1 ended with status=failed error=quota" in caplog.text


class TestAdoptionDoesNotResubmit:
    def test_an_adopted_batch_is_polled_and_no_new_batch_is_posted(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"status": "submitting", "model": "m", "custom_ids": ["a"], "submitting_at": 1000.0}), encoding="utf-8")
        posts = []

        def handler(request):
            if request.method == "POST":
                posts.append(1)
                return httpx.Response(200, json={"id": "batch-new"})
            if request.method == "GET" and "limit" in request.url.params:
                return httpx.Response(200, json={"data": [{"id": "found", "model": "m", "created_at": 1000.0, "request_counts": {"total": 1}}]})
            return httpx.Response(200, json={"id": "found", "status": "completed", "results": [_item(custom_id="a")]})

        results = _client(handler).run("m", [_request("a")], state_path=path)
        assert posts == []
        assert results["a"].text == "hello"
        assert load_job_state(path)["batch_id"] == "found"

    def test_a_completed_run_logs_no_status_warning(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger=ob.logger.name)

        def handler(request):
            if request.method == "POST":
                return httpx.Response(200, json={"id": "batch-1"})
            return httpx.Response(200, json={"id": "batch-1", "status": "completed", "results": []})

        _client(handler).run("m", [_request("a")], state_path=tmp_path / "s.json")
        assert "ended with status" not in caplog.text


class TestCreatedAtDebugArguments:
    def test_the_debug_line_names_the_exception_class_before_its_message(self, caplog):
        caplog.set_level(logging.DEBUG, logger=ob.logger.name)
        ob._created_at({"created_at": "yesterday"})
        assert "timestamp is not ISO-8601: ValueError: " in caplog.text


class TestKeywordOnlyBaseUrl:
    def test_the_base_url_is_keyword_only_on_the_client(self):
        with pytest.raises(TypeError):
            OpenRouterBatchClient("key", "https://example.invalid")


class TestWaitRemainingDeadline:
    def test_the_sleep_is_the_remaining_deadline_when_that_is_shorter(self, monkeypatch):
        monkeypatch.setattr(ob.random, "uniform", lambda low, high: low)
        sleeps = []
        ticks = iter([0.0, 49.95])
        statuses = iter(["in_progress", "completed"])
        client = _client(lambda request: httpx.Response(200, json={"id": "b", "status": next(statuses)}), sleep=sleeps.append, clock=lambda: next(ticks))
        client.wait("b", deadline_s=50.0, initial_interval_s=10.0)
        assert sleeps == pytest.approx([0.05])
