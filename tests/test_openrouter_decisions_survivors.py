"""Exact-message, boundary and state pins for the OpenRouter decisions module (mutation survivors of its first tests)."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import FrozenInstanceError

import pytest

httpx = pytest.importorskip("httpx")

from pyutilz.llm import openrouter_decisions as od
from pyutilz.llm.openrouter_decisions import (
    ChoiceQuestion,
    DecisionAnswer,
    DecisionResult,
    NoulQuestion,
    OpenRouterDecisionsClient,
    OpenRouterDecisionsError,
    ScoreQuestion,
    build_payload,
    parse_answer,
    parse_response,
)


def _ok_body(**overrides) -> dict:
    body = {"model": "m", "id": "gen-1", "provider": "p", "answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {"cost": 0.25}}
    body.update(overrides)
    return body


def _client(handler, **kwargs) -> OpenRouterDecisionsClient:
    kwargs.setdefault("retry_wait", lambda retry_state: 0.0)
    return OpenRouterDecisionsClient("key", transport=httpx.MockTransport(handler), **kwargs)


class _Outcome:
    def __init__(self, failed: bool, exc=None):
        self.failed = failed
        self._exc = exc

    def exception(self):
        return self._exc


class _RetryState:
    def __init__(self, outcome):
        self.outcome = outcome


class TestModuleConstants:
    def test_the_documented_limits_and_defaults(self):
        assert od.DEFAULT_TIMEOUT_S == 15.0
        assert od.DEFAULT_MAX_ATTEMPTS == 5
        assert od.MAX_CHOICE_OPTIONS == 255
        assert od.MAX_SCORE_LEVELS == 10
        assert od.BASE_URL == "https://openrouter.ai/api/alpha"
        assert od.DEFAULT_MODEL == "typesafe/jev-1.13"

    def test_the_retryable_statuses_are_exactly_the_transient_set(self):
        assert od.RETRYABLE_STATUSES == frozenset({408, 429, 500, 502, 503, 504, 524, 529})

    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 524, 529])
    def test_each_transient_status_is_retryable(self, status):
        assert OpenRouterDecisionsError("x", status_code=status).retryable is True

    @pytest.mark.parametrize("status", [400, 404, 409, 505])
    def test_other_statuses_are_not_retryable(self, status):
        assert OpenRouterDecisionsError("x", status_code=status).retryable is False


class TestErrorClass:
    def test_the_message_is_kept_as_the_exception_text(self):
        assert str(OpenRouterDecisionsError("boom")) == "boom"

    def test_the_status_is_keyword_only(self):
        with pytest.raises(TypeError):
            OpenRouterDecisionsError("m", 400)

    def test_the_fields_are_stored(self):
        err = OpenRouterDecisionsError("m", status_code=429, code=7, raw={"a": 1}, retry_after_s=3.0)
        assert (err.status_code, err.code, err.raw, err.retry_after_s) == (429, 7, {"a": 1}, 3.0)


class TestFrozenValues:
    @pytest.mark.parametrize(
        "value, field",
        [
            (NoulQuestion("i", "t", "f"), "instructions"),
            (ChoiceQuestion("i", {"a": "x", "b": "y"}), "instructions"),
            (ScoreQuestion("i", ["lo", "hi"]), "instructions"),
            (DecisionAnswer(type="noul", value=True, probabilities={}), "type"),
            (DecisionResult(answers={}, missing=[], model="m", generation_id="g", provider="p", input_tokens=0, output_tokens=0, cost_usd=None, latency_s=0.0), "model"),
        ],
    )
    def test_each_value_object_is_immutable(self, value, field):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, "changed")


class TestNoulPayload:
    def test_the_wire_form_has_the_criteria_pair(self):
        assert NoulQuestion("ins", "yes", "no").to_payload() == {"type": "noul", "instructions": "ins", "criteria": {"true": "yes", "false": "no"}}

    def test_a_blank_field_is_refused_with_the_message(self):
        with pytest.raises(ValueError) as info:
            NoulQuestion("  ", "t", "f").to_payload()
        assert str(info.value) == "noul question needs non-empty instructions, true and false criteria"


class TestChoicePayload:
    def test_the_wire_form_stringifies_labels_and_descriptions(self):
        payload = ChoiceQuestion("ins", {"a": "x", "b": 2}).to_payload()
        assert payload == {"type": "choice", "instructions": "ins", "criteria": {"a": "x", "b": "2"}}

    def test_one_option_is_refused_with_the_count(self):
        with pytest.raises(ValueError) as info:
            ChoiceQuestion("i", {"a": "x"}).to_payload()
        assert str(info.value) == "choice question needs 2..255 options, got 1"

    def test_too_many_options_are_refused_with_the_count(self):
        many = {str(i): "d" for i in range(256)}
        with pytest.raises(ValueError) as info:
            ChoiceQuestion("i", many).to_payload()
        assert str(info.value) == "choice question needs 2..255 options, got 256"

    def test_two_and_two_hundred_fifty_five_options_are_accepted(self):
        ChoiceQuestion("i", {"a": "x", "b": "y"}).to_payload()
        ChoiceQuestion("i", {str(i): "d" for i in range(255)}).to_payload()

    def test_blank_instructions_are_refused(self):
        with pytest.raises(ValueError) as info:
            ChoiceQuestion("  ", {"a": "x", "b": "y"}).to_payload()
        assert str(info.value) == "choice question needs non-empty instructions"

    def test_one_blank_label_among_good_ones_is_refused(self):
        with pytest.raises(ValueError) as info:
            ChoiceQuestion("i", {"a": "x", "  ": "y"}).to_payload()
        assert str(info.value) == "choice option labels must be non-empty"


class TestScorePayload:
    def test_the_wire_form_lists_the_levels_in_order(self):
        assert ScoreQuestion("ins", ["lo", "hi"]).to_payload() == {"type": "score", "instructions": "ins", "criteria": ["lo", "hi"]}

    def test_one_string_instead_of_levels_is_a_type_error(self):
        with pytest.raises(TypeError) as info:
            ScoreQuestion("i", "lo").to_payload()
        assert str(info.value) == "score levels must be a sequence of strings, not one string"

    def test_one_level_is_refused_with_the_count(self):
        with pytest.raises(ValueError) as info:
            ScoreQuestion("i", ["only"]).to_payload()
        assert str(info.value) == "score question needs 2..10 levels, got 1"

    def test_eleven_levels_are_refused_with_the_count(self):
        with pytest.raises(ValueError) as info:
            ScoreQuestion("i", [str(i) for i in range(11)]).to_payload()
        assert str(info.value) == "score question needs 2..10 levels, got 11"

    def test_two_and_ten_levels_are_accepted(self):
        ScoreQuestion("i", ["a", "b"]).to_payload()
        ScoreQuestion("i", [str(i) for i in range(10)]).to_payload()

    def test_blank_instructions_are_refused(self):
        with pytest.raises(ValueError) as info:
            ScoreQuestion(" ", ["a", "b"]).to_payload()
        assert str(info.value) == "score question needs non-empty instructions"


class TestBuildPayloadMessages:
    def test_an_empty_model_is_refused(self):
        with pytest.raises(ValueError) as info:
            build_payload("", "s", {"q": NoulQuestion("i", "t", "f")})
        assert str(info.value) == "model is required"

    def test_a_moving_alias_is_refused_naming_the_opt_in(self):
        with pytest.raises(ValueError) as info:
            build_payload("~typesafe/latest", "s", {"q": NoulQuestion("i", "t", "f")})
        assert str(info.value) == ("'~typesafe/latest' is a moving alias; pin a version such as 'typesafe/jev-1.13' or pass allow_moving_alias=True")

    def test_a_moving_alias_is_accepted_when_opted_in(self):
        payload = build_payload("~typesafe/latest", "s", {"q": NoulQuestion("i", "t", "f")}, allow_moving_alias=True)
        assert payload["model"] == "~typesafe/latest"

    def test_no_questions_is_refused(self):
        with pytest.raises(ValueError) as info:
            build_payload("m", "s", {})
        assert str(info.value) == "at least one question is required"

    def test_allow_moving_alias_is_keyword_only(self):
        with pytest.raises(TypeError):
            build_payload("m", "s", {"q": NoulQuestion("i", "t", "f")}, True)

    def test_the_payload_carries_model_state_and_questions(self):
        payload = build_payload("m", "s", {"q": NoulQuestion("i", "t", "f")})
        assert payload == {"model": "m", "state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "criteria": {"true": "t", "false": "f"}}}}


class TestStateMessages:
    def test_a_blank_string_state_is_refused(self):
        with pytest.raises(ValueError) as info:
            build_payload("m", "   ", {"q": NoulQuestion("i", "t", "f")})
        assert str(info.value) == "state must not be empty"

    def test_a_non_state_type_is_a_type_error_with_the_message(self):
        with pytest.raises(TypeError) as info:
            build_payload("m", 5, {"q": NoulQuestion("i", "t", "f")})
        assert str(info.value) == "state must be a string, a mapping or a sequence of strings"

    def test_a_sequence_with_a_non_string_is_refused(self):
        with pytest.raises(TypeError):
            build_payload("m", ["a", 1], {"q": NoulQuestion("i", "t", "f")})

    def test_a_mapping_state_is_copied_and_a_string_list_is_listed(self):
        mapping_payload = build_payload("m", {"k": 1}, {"q": NoulQuestion("i", "t", "f")})
        list_payload = build_payload("m", ("a", "b"), {"q": NoulQuestion("i", "t", "f")})
        assert mapping_payload["state"] == {"k": 1}
        assert list_payload["state"] == ["a", "b"]


class TestQuestionValidation:
    def test_a_blank_question_id_is_refused_with_its_repr(self):
        with pytest.raises(ValueError) as info:
            build_payload("m", "s", {"  ": NoulQuestion("i", "t", "f")})
        assert str(info.value) == "question ids must be non-empty strings, got '  '"

    def test_an_unsupported_question_type_is_a_type_error_naming_the_question(self):
        with pytest.raises(TypeError) as info:
            build_payload("m", "s", {"q": "text"})
        assert str(info.value) == "question 'q': unsupported type str"

    def test_a_nested_limit_error_names_the_question_and_keeps_its_type(self):
        with pytest.raises(ValueError) as info:
            build_payload("m", "s", {"q": ChoiceQuestion("i", {"a": "x"})})
        assert str(info.value) == "question 'q': choice question needs 2..255 options, got 1"


class TestHelpers:
    def test_only_real_numbers_survive_in_a_float_map(self):
        assert od._float_map({"a": True, "b": 0.5, "c": "x", "d": 2}) == {"b": 0.5, "d": 2.0}

    def test_a_non_mapping_gives_an_empty_float_map(self):
        assert od._float_map([1, 2]) == {}

    def test_the_optional_float_accepts_numbers_only(self):
        assert od._opt_float(2) == 2.0
        assert od._opt_float(True) is None
        assert od._opt_float("1") is None
        assert od._opt_float(None) is None


class TestParseAnswerNoul:
    def test_a_boolean_noul_maps_to_one_or_zero(self):
        yes = parse_answer("noul", {"noul": True})
        no = parse_answer("noul", {"noul": False})
        assert (yes.value, yes.probabilities) == (True, {"true": 1.0, "false": 0.0})
        assert (no.value, no.probabilities) == (False, {"true": 0.0, "false": 1.0})

    def test_exactly_half_is_true_and_just_below_is_false(self):
        assert parse_answer("noul", {"noul": 0.5}).value is True
        assert parse_answer("noul", {"noul": 0.49999}).value is False

    def test_a_missing_or_text_probability_is_no_answer(self):
        assert parse_answer("noul", {}) is None
        assert parse_answer("noul", {"noul": "yes"}) is None

    def test_the_item_type_overrides_the_question_type(self):
        assert parse_answer("choice", {"type": "noul", "noul": 0.7}).type == "noul"

    def test_a_falsy_item_type_falls_back_to_the_question_type(self):
        assert parse_answer("noul", {"type": "", "noul": 0.7}).type == "noul"

    def test_a_truthy_item_type_is_not_replaced_by_the_question_type(self):
        assert parse_answer("noul", {"type": "choice", "choice": "a"}).type == "choice"


class TestParseAnswerChoice:
    def test_the_chosen_label_and_probabilities_are_kept(self):
        answer = parse_answer("choice", {"type": "choice", "choice": "a", "probabilities": {"a": 0.8}, "confidence": 0.8})
        assert (answer.type, answer.value, answer.probabilities, answer.confidence) == ("choice", "a", {"a": 0.8}, 0.8)

    @pytest.mark.parametrize("choice", ["", 5, None])
    def test_an_empty_or_non_text_choice_is_no_answer(self, choice):
        assert parse_answer("choice", {"type": "choice", "choice": choice}) is None


class TestParseAnswerScore:
    def test_the_most_probable_level_index_is_the_value(self):
        answer = parse_answer("score", {"type": "score", "probabilities": {"0": 0.2, "1": 0.8}, "score": 1.2, "legend": {"0": "lo", "1": "hi"}})
        assert (answer.value, answer.expected_score, answer.legend) == (1, 1.2, {"0": "lo", "1": "hi"})

    def test_the_expected_score_is_rounded_when_there_are_no_probabilities(self):
        answer = parse_answer("score", {"type": "score", "score": 2.6})
        assert (answer.value, answer.expected_score) == (3, 2.6)

    def test_a_low_expected_score_rounds_down(self):
        assert parse_answer("score", {"type": "score", "score": 2.4}).value == 2

    def test_no_probabilities_and_no_score_is_no_answer(self):
        assert parse_answer("score", {"type": "score"}) is None

    def test_a_non_integer_level_key_is_no_answer_and_logged(self, caplog):
        caplog.set_level(logging.WARNING, logger=od.logger.name)
        assert parse_answer("score", {"type": "score", "probabilities": {"x": 0.9}}) is None
        assert "answer index is not an integer, answer is None: ValueError: " in caplog.text


class TestParseAnswerUnknownType:
    def test_an_unknown_type_is_no_answer_and_logged_with_its_repr(self, caplog):
        caplog.set_level(logging.WARNING, logger=od.logger.name)
        assert parse_answer("noul", {"type": "weird"}) is None
        assert "decisions: unknown answer type 'weird'" in caplog.text


class TestParseResponse:
    def test_an_error_key_raises_the_envelope_error(self):
        with pytest.raises(OpenRouterDecisionsError) as info:
            parse_response({"error": {"code": 400, "message": "bad"}}, {"q": NoulQuestion("i", "t", "f")})
        assert str(info.value) == "decisions: HTTP 400 code=400: bad"

    def test_latency_defaults_to_zero_and_is_keyword_only(self):
        result = parse_response(_ok_body(), {"q": NoulQuestion("i", "t", "f")})
        assert result.latency_s == 0.0
        with pytest.raises(TypeError):
            parse_response(_ok_body(), {"q": NoulQuestion("i", "t", "f")}, 1.5)
        assert inspect.signature(parse_response).parameters["latency_s"].default == 0.0

    def test_missing_usage_counts_are_zero_and_missing_cost_is_none(self):
        result = parse_response({"answers": {}}, {"q": NoulQuestion("i", "t", "f")})
        assert (result.input_tokens, result.output_tokens, result.cost_usd) == (0, 0, None)

    def test_reported_usage_is_kept_including_a_zero_cost(self):
        result = parse_response(_ok_body(usage={"input_tokens": 3, "output_tokens": 4, "cost": 0}), {"q": NoulQuestion("i", "t", "f")})
        assert (result.input_tokens, result.output_tokens, result.cost_usd) == (3, 4, 0.0)

    def test_a_non_mapping_answers_block_marks_every_question_missing(self):
        result = parse_response({"answers": "nope"}, {"q": NoulQuestion("i", "t", "f")})
        assert result.missing == ["q"]
        assert result.answers == {}


class TestErrorFromBody:
    def test_the_envelope_code_and_message_are_named(self):
        err = od._error_from_body({"error": {"code": 429, "message": "slow"}}, 429)
        assert str(err) == "decisions: HTTP 429 code=429: slow"
        assert err.status_code == 429

    def test_the_detail_envelope_is_read(self):
        err = od._error_from_body({"detail": "nope"}, None)
        assert str(err) == "decisions: HTTP None code=None: nope"

    def test_a_text_error_value_is_the_message(self):
        assert str(od._error_from_body({"error": "bad"}, 500)) == "decisions: HTTP 500 code=None: bad"

    def test_a_missing_body_gives_an_empty_message(self):
        assert str(od._error_from_body(None, 500)) == "decisions: HTTP 500 code=None: "

    def test_a_long_message_is_cut_at_five_hundred_characters(self):
        err = od._error_from_body({"error": {"message": "m" * 600}}, 500)
        assert str(err) == "decisions: HTTP 500 code=None: " + "m" * 500

    def test_the_status_falls_back_to_an_integer_code(self):
        err = od._error_from_body({"error": {"code": 502, "message": "x"}}, None)
        assert err.status_code == 502


class TestCheckResponse:
    def test_a_non_json_body_is_a_non_object_error_cut_at_three_hundred(self):
        resp = httpx.Response(200, text="z" * 600)
        with pytest.raises(OpenRouterDecisionsError) as info:
            od._check_response(resp)
        assert str(info.value) == "decisions: non-object response: " + "z" * 300

    def test_a_json_list_is_a_non_object_error_with_its_text(self):
        with pytest.raises(OpenRouterDecisionsError) as info:
            od._check_response(httpx.Response(200, json=[1, 2]))
        assert str(info.value) == "decisions: non-object response: [1, 2]"

    def test_status_four_hundred_raises_and_three_ninety_nine_does_not(self):
        with pytest.raises(OpenRouterDecisionsError):
            od._check_response(httpx.Response(400, json={"error": {"message": "x"}}))
        assert od._check_response(httpx.Response(399, json={"answers": {}})) == {"answers": {}}

    def test_a_text_error_body_is_kept_cut_at_five_hundred_characters(self):
        with pytest.raises(OpenRouterDecisionsError) as info:
            od._check_response(httpx.Response(500, text="t" * 600))
        assert len(info.value.raw) == 500

    def test_the_retry_after_header_is_carried_on_the_error(self):
        with pytest.raises(OpenRouterDecisionsError) as info:
            od._check_response(httpx.Response(429, json={"error": {"message": "slow"}}, headers={"Retry-After": "7"}))
        assert info.value.retry_after_s == 7.0


class TestRetryWait:
    def test_a_longer_server_hint_lengthens_the_backoff(self):
        base = lambda retry_state: 5.0  # noqa: E731
        exc = OpenRouterDecisionsError("x", status_code=429, retry_after_s=30.0)
        wait = od._wait_honoring_retry_after(base)
        assert wait(_RetryState(_Outcome(True, exc))) == 30.0

    def test_a_shorter_hint_keeps_the_base_backoff(self):
        base = lambda retry_state: 5.0  # noqa: E731
        exc = OpenRouterDecisionsError("x", status_code=429, retry_after_s=2.0)
        assert od._wait_honoring_retry_after(base)(_RetryState(_Outcome(True, exc))) == 5.0

    def test_no_outcome_keeps_the_base_backoff(self):
        assert od._wait_honoring_retry_after(lambda retry_state: 5.0)(_RetryState(None)) == 5.0

    def test_a_successful_outcome_ignores_any_exception_hint(self):
        exc = OpenRouterDecisionsError("x", retry_after_s=30.0)
        assert od._wait_honoring_retry_after(lambda retry_state: 5.0)(_RetryState(_Outcome(False, exc))) == 5.0

    def test_a_non_numeric_hint_keeps_the_base_backoff(self):
        class HintedError(Exception):
            retry_after_s = "soon"

        assert od._wait_honoring_retry_after(lambda retry_state: 5.0)(_RetryState(_Outcome(True, HintedError()))) == 5.0


class TestIsRetryable:
    def test_transport_errors_are_retried(self):
        assert od._is_retryable(httpx.ConnectError("reset")) is True

    def test_other_exceptions_are_not_retried(self):
        assert od._is_retryable(ValueError("no")) is False

    def test_an_envelope_error_follows_its_status(self):
        assert od._is_retryable(OpenRouterDecisionsError("x", status_code=429)) is True
        assert od._is_retryable(OpenRouterDecisionsError("x", status_code=400)) is False


class TestClientConstruction:
    def test_a_missing_key_is_refused_naming_the_setting(self, monkeypatch):
        class _Settings:
            openrouter_api_key = None

        monkeypatch.setattr("pyutilz.llm.config.get_llm_settings", lambda: _Settings())
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterDecisionsClient(None, transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    def test_zero_attempts_is_refused_and_one_is_accepted(self):
        with pytest.raises(ValueError) as info:
            OpenRouterDecisionsClient("key", max_attempts=0, transport=httpx.MockTransport(lambda request: httpx.Response(200)))
        assert str(info.value) == "max_attempts must be >= 1"
        OpenRouterDecisionsClient("key", max_attempts=1, transport=httpx.MockTransport(lambda request: httpx.Response(200))).close()

    def test_the_bearer_key_is_sent_on_every_request(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers["authorization"]
            return httpx.Response(200, json=_ok_body())

        _client(handler).decide("s", {"q": NoulQuestion("i", "t", "f")})
        assert seen["auth"] == "Bearer key"

    def test_the_documented_defaults_are_in_the_signature(self):
        params = inspect.signature(OpenRouterDecisionsClient.__init__).parameters
        assert params["timeout"].default == 15.0
        assert params["max_attempts"].default == 5

    def test_the_base_url_is_keyword_only(self):
        with pytest.raises(TypeError):
            OpenRouterDecisionsClient("key", "https://example.invalid")


class TestCloseAndContexts:
    def test_close_closes_the_sync_client(self):
        client = _client(lambda request: httpx.Response(200, json=_ok_body()))
        client.close()
        assert client._client.is_closed

    def test_an_unclosed_async_client_is_warned_about_on_close(self, caplog):
        caplog.set_level(logging.WARNING, logger=od.logger.name)

        class _Open:
            is_closed = False

        client = _client(lambda request: httpx.Response(200))
        client._aclient = _Open()
        client.close()
        assert "cannot close the async client; await aclose() instead" in caplog.text

    def test_a_closed_async_client_is_not_warned_about(self, caplog):
        caplog.set_level(logging.WARNING, logger=od.logger.name)

        class _Closed:
            is_closed = True

        client = _client(lambda request: httpx.Response(200))
        client._aclient = _Closed()
        client.close()
        assert "cannot close the async client" not in caplog.text

    def test_aclose_closes_both_clients(self):
        closed = []

        class _Async:
            async def aclose(self):
                closed.append("async")

        client = _client(lambda request: httpx.Response(200))
        client._aclient = _Async()
        asyncio.run(client.aclose())
        assert closed == ["async"]
        assert client._client.is_closed

    def test_aclose_with_no_async_client_only_closes_the_sync_one(self):
        client = _client(lambda request: httpx.Response(200))
        asyncio.run(client.aclose())
        assert client._client.is_closed

    def test_the_sync_context_manager_returns_self_and_closes(self):
        with _client(lambda request: httpx.Response(200)) as client:
            assert isinstance(client, OpenRouterDecisionsClient)
        assert client._client.is_closed

    def test_the_async_context_manager_returns_self_and_closes(self):
        async def run():
            async with _client(lambda request: httpx.Response(200)) as client:
                assert isinstance(client, OpenRouterDecisionsClient)
            return client

        assert asyncio.run(run())._client.is_closed


class TestAccounting:
    def test_reported_costs_accumulate_and_calls_count(self):
        client = _client(lambda request: httpx.Response(200, json=_ok_body(usage={"cost": 0.25})))
        client.decide("s", {"q": NoulQuestion("i", "t", "f")})
        client.decide("s", {"q": NoulQuestion("i", "t", "f")})
        assert client.total_cost_usd == pytest.approx(0.5)
        assert client.calls == 2
        assert client.unreported_cost_calls == 0

    def test_unreported_costs_are_counted_not_summed(self):
        client = _client(lambda request: httpx.Response(200, json=_ok_body(usage={})))
        client.decide("s", {"q": NoulQuestion("i", "t", "f")})
        client.decide("s", {"q": NoulQuestion("i", "t", "f")})
        assert client.unreported_cost_calls == 2
        assert client.total_cost_usd == 0.0

    def test_a_missing_answer_is_logged_with_the_generation_and_ids(self, caplog):
        caplog.set_level(logging.INFO, logger=od.logger.name)
        client = _client(lambda request: httpx.Response(200, json=_ok_body(answers={})))
        client.decide("s", {"q": NoulQuestion("i", "t", "f")})
        assert "decisions gen-1: no answer for ['q']" in caplog.text


class TestDecideCall:
    def test_the_request_goes_to_the_decisions_endpoint_with_the_payload(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_body())

        _client(handler).decide("s", {"q": NoulQuestion("i", "t", "f")})
        assert seen["path"].endswith("/decisions")
        assert seen["body"]["model"] == "typesafe/jev-1.13"

    def test_a_client_error_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(400, json={"error": {"message": "bad"}})

        with pytest.raises(OpenRouterDecisionsError):
            _client(handler).decide("s", {"q": NoulQuestion("i", "t", "f")})
        assert len(calls) == 1

    def test_decide_is_keyword_only_for_the_model(self):
        with pytest.raises(TypeError):
            _client(lambda request: httpx.Response(200, json=_ok_body())).decide("s", {"q": NoulQuestion("i", "t", "f")}, "m")

    def test_adecide_is_keyword_only_and_refuses_an_alias_by_default(self):
        client = _client(lambda request: httpx.Response(200, json=_ok_body()))
        with pytest.raises(TypeError):
            asyncio.run(client.adecide("s", {"q": NoulQuestion("i", "t", "f")}, "m"))
        with pytest.raises(ValueError, match="moving alias"):
            asyncio.run(client.adecide("s", {"q": NoulQuestion("i", "t", "f")}, model="~typesafe/latest"))

    def test_adecide_posts_to_the_decisions_endpoint(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            return httpx.Response(200, json=_ok_body())

        async def run():
            client = OpenRouterDecisionsClient("key", async_transport=httpx.MockTransport(handler), retry_wait=lambda retry_state: 0.0)
            try:
                return await client.adecide("s", {"q": NoulQuestion("i", "t", "f")})
            finally:
                await client.aclose()

        asyncio.run(run())
        assert seen["path"].endswith("/decisions")


class TestScoreTypeAndLatency:
    def test_a_score_answer_is_typed_score(self):
        answer = parse_answer("score", {"type": "score", "score": 2.6})
        assert answer.type == "score"

    def test_the_result_latency_is_the_elapsed_time_since_the_start_reading(self):
        import time

        resp = httpx.Response(200, json=_ok_body())
        t0 = time.perf_counter() - 2.0
        result = od._result_of(resp, {"q": NoulQuestion("i", "t", "f")}, t0)
        assert result.latency_s == pytest.approx(2.0, abs=0.5)


class TestCustomRetryWait:
    def test_a_supplied_retry_wait_is_the_one_used_between_attempts(self):
        waits = []

        def record(retry_state):
            waits.append(retry_state.attempt_number)
            return 0.0

        responses = iter([httpx.Response(500, json={"error": {"message": "x"}}), httpx.Response(200, json=_ok_body())])
        client = OpenRouterDecisionsClient("key", transport=httpx.MockTransport(lambda request: next(responses)), retry_wait=record)
        client.decide("s", {"q": NoulQuestion("i", "t", "f")})
        assert waits == [1]
