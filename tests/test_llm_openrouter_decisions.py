"""Behavioural tests for the OpenRouter decisions (Jev) client, driven through ``httpx.MockTransport``."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
import pytest
from tenacity import wait_none

from pyutilz.llm.openrouter_decisions import (
    DEFAULT_MODEL,
    ChoiceQuestion,
    NoulQuestion,
    OpenRouterDecisionsClient,
    OpenRouterDecisionsError,
    Question,
    ScoreQuestion,
    build_payload,
)

# Recorded from a live call on 2026-09-25 (ids shortened).
LIVE_BODY: Dict[str, Any] = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "team": {"type": "choice", "choice": "billing", "probabilities": {"technical": 0, "billing": 1}, "confidence": 1},
        "urgent": {"type": "noul", "noul": 0.58},
        "anger": {
            "type": "score",
            "score": 0.85,
            "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
            "probabilities": {"0": 0.15, "1": 0.85, "2": 0},
            "confidence": 0.77,
        },
    },
    "usage": {"input_tokens": 389, "output_tokens": 61, "cost": 0.000016338},
    "id": "gen-dec-1",
    "provider": "TypeSafe",
}

QUESTIONS: Dict[str, Question] = {
    "team": ChoiceQuestion("Which team?", {"technical": "bugs", "billing": "charges"}),
    "urgent": NoulQuestion("Urgent?", "today", "can wait"),
    "anger": ScoreQuestion("How upset?", ["calm", "annoyed", "furious"]),
}


def _client(responses: List[httpx.Response], seen: List[httpx.Request]) -> OpenRouterDecisionsClient:
    """A client whose sync and async transports replay ``responses`` in order and record requests."""
    it = iter(responses)

    def handler(req: httpx.Request) -> httpx.Response:
        """Record and answer one request."""
        seen.append(req)
        return next(it)

    async def ahandler(req: httpx.Request) -> httpx.Response:
        """Async twin of ``handler``."""
        return handler(req)

    return OpenRouterDecisionsClient(
        "k", transport=httpx.MockTransport(handler), async_transport=httpx.MockTransport(ahandler), retry_wait=wait_none(), max_attempts=3
    )


def test_live_shape_parses_all_three_types() -> None:
    """Each answer type maps to its value, probabilities and confidence; usage and cost come through."""
    seen: List[httpx.Request] = []
    c = _client([httpx.Response(200, json=LIVE_BODY)], seen)
    r = c.decide("I was charged twice", QUESTIONS)
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/alpha/decisions"
    assert sent["model"] == DEFAULT_MODEL
    assert sent["questions"]["anger"] == {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]}
    assert sent["questions"]["urgent"]["criteria"] == {"true": "today", "false": "can wait"}
    assert r.missing == []
    assert r.answers["team"].value == "billing" and r.answers["team"].probabilities == {"technical": 0.0, "billing": 1.0}
    assert r.answers["team"].confidence == 1.0
    u = r.answers["urgent"]
    assert u.value is True and u.p_true == pytest.approx(0.58) and u.probabilities["false"] == pytest.approx(0.42) and u.confidence is None
    a = r.answers["anger"]
    assert a.value == 1 and a.expected_score == pytest.approx(0.85) and a.legend["2"] == "furious" and a.confidence == pytest.approx(0.77)
    assert (r.input_tokens, r.output_tokens) == (389, 61)
    assert r.model == "typesafe/jev-1.13-20260917" and r.generation_id == "gen-dec-1" and r.provider == "TypeSafe"
    assert r.cost_reported and c.total_cost_usd == pytest.approx(0.000016338)


def test_noul_below_half_is_false() -> None:
    """A p_true under 0.5 gives False, not a truthy float."""
    body = dict(LIVE_BODY, answers={"urgent": {"type": "noul", "noul": 0.2}})
    r = _client([httpx.Response(200, json=body)], []).decide("s", {"urgent": QUESTIONS["urgent"]})
    assert r.answers["urgent"].value is False


def test_missing_and_malformed_answers_are_reported_as_missing() -> None:
    """An absent id and an answer without its value both land in ``missing`` and not in ``answers``."""
    body = dict(LIVE_BODY, answers={"team": {"type": "choice", "probabilities": {}}, "urgent": {"type": "noul", "noul": 0.9}})
    r = _client([httpx.Response(200, json=body)], []).decide("s", QUESTIONS)
    assert r.missing == ["team", "anger"]
    assert list(r.answers) == ["urgent"]


def test_cost_missing_differs_from_zero() -> None:
    """No ``usage.cost`` is None and counted as unreported; a reported 0 is a real zero."""
    no_cost = dict(LIVE_BODY, usage={"input_tokens": 5, "output_tokens": 1})
    zero = dict(LIVE_BODY, usage={"input_tokens": 5, "output_tokens": 1, "cost": 0})
    c = _client([httpx.Response(200, json=no_cost), httpx.Response(200, json=zero)], [])
    r1 = c.decide("s", QUESTIONS)
    r2 = c.decide("s", QUESTIONS)
    assert r1.cost_usd is None and not r1.cost_reported
    assert r2.cost_usd == 0.0 and r2.cost_reported
    assert c.unreported_cost_calls == 1 and c.calls == 2 and c.total_cost_usd == 0.0


def test_error_envelope_parsed_and_not_retried() -> None:
    """A 400 envelope raises with code and message after exactly one request."""
    seen: List[httpx.Request] = []
    env = {"error": {"message": "Invalid discriminator value", "code": 400}, "user_id": "u"}
    c = _client([httpx.Response(400, json=env)], seen)
    with pytest.raises(OpenRouterDecisionsError) as ei:
        c.decide("s", QUESTIONS)
    assert ei.value.status_code == 400 and ei.value.code == 400 and "Invalid discriminator" in str(ei.value)
    assert len(seen) == 1


def test_detail_envelope_and_error_in_200_body() -> None:
    """TypeSafe's ``{"detail"}`` form is read, and an error inside a 200 body still raises."""
    c = _client([httpx.Response(401, json={"detail": "bad key"}), httpx.Response(200, json={"error": {"code": 402, "message": "no credits"}})], [])
    with pytest.raises(OpenRouterDecisionsError, match="bad key"):
        c.decide("s", QUESTIONS)
    with pytest.raises(OpenRouterDecisionsError) as ei:
        c.decide("s", QUESTIONS)
    assert ei.value.status_code == 402 and not ei.value.retryable


@pytest.mark.parametrize("status", [429, 500, 502, 503, 524, 529])
def test_retries_transient_statuses(status: int) -> None:
    """A transient status is retried and the later success returned."""
    seen: List[httpx.Request] = []
    c = _client([httpx.Response(status, json={"error": {"code": status, "message": "x"}}), httpx.Response(200, json=LIVE_BODY)], seen)
    assert c.decide("s", QUESTIONS).answers["team"].value == "billing"
    assert len(seen) == 2 and c.calls == 1


def test_retry_gives_up_after_max_attempts() -> None:
    """Persistent 503 raises after ``max_attempts`` requests, with the last status attached."""
    seen: List[httpx.Request] = []
    c = _client([httpx.Response(503, text="down")] * 3, seen)
    with pytest.raises(OpenRouterDecisionsError) as ei:
        c.decide("s", QUESTIONS)
    assert ei.value.status_code == 503 and len(seen) == 3


def test_async_decide_retries_and_parses() -> None:
    """``adecide`` goes through the async transport with the same retry and parsing."""
    seen: List[httpx.Request] = []
    c = _client([httpx.Response(429, json={"error": {"code": 429, "message": "slow"}}), httpx.Response(200, json=LIVE_BODY)], seen)

    async def run() -> Any:
        """Call and close."""
        try:
            return await c.adecide(["line one", "line two"], QUESTIONS)
        finally:
            await c.aclose()

    r = asyncio.run(run())
    assert len(seen) == 2 and json.loads(seen[1].content)["state"] == ["line one", "line two"]
    assert r.answers["anger"].value == 1 and c.total_cost_usd == pytest.approx(0.000016338)


@pytest.mark.parametrize(
    "questions, model",
    [
        ({"q": ChoiceQuestion("i", {str(i): "d" for i in range(256)})}, DEFAULT_MODEL),
        ({"q": ChoiceQuestion("i", {"only": "d"})}, DEFAULT_MODEL),
        ({"q": ScoreQuestion("i", [str(i) for i in range(11)])}, DEFAULT_MODEL),
        ({"q": ScoreQuestion("i", ["one"])}, DEFAULT_MODEL),
        ({"q": NoulQuestion("i", "", "no")}, DEFAULT_MODEL),
        ({}, DEFAULT_MODEL),
        ({"": NoulQuestion("i", "y", "n")}, DEFAULT_MODEL),
        ({"q": NoulQuestion("i", "y", "n")}, "~typesafe/jev-latest"),
    ],
)
def test_limit_violations_raise_before_any_http_call(questions: Dict[str, Any], model: str) -> None:
    """Every limit violation raises ValueError and no request is sent."""
    seen: List[httpx.Request] = []
    c = _client([], seen)
    with pytest.raises(ValueError):
        c.decide("s", questions, model=model)
    assert seen == []


def test_limits_accept_boundaries_and_alias_opt_in() -> None:
    """255 options, 10 levels and an explicit moving-alias opt-in all build."""
    p = build_payload(
        "~typesafe/jev-latest",
        {"k": 1},
        {"c": ChoiceQuestion("i", {str(i): "d" for i in range(255)}), "s": ScoreQuestion("i", [str(i) for i in range(10)])},
        allow_moving_alias=True,
    )
    assert len(p["questions"]["c"]["criteria"]) == 255 and len(p["questions"]["s"]["criteria"]) == 10 and p["state"] == {"k": 1}


def test_bad_state_and_string_levels_raise_type_error() -> None:
    """A non-string sequence state and a bare-string ``levels`` are rejected."""
    with pytest.raises(TypeError):
        build_payload(DEFAULT_MODEL, [1, 2], {"q": NoulQuestion("i", "y", "n")})  # type: ignore[list-item]
    with pytest.raises(TypeError):
        build_payload(DEFAULT_MODEL, "s", {"q": ScoreQuestion("i", "abc")})
