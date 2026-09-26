"""OpenRouter decisions client for System One models such as TypeSafe's Jev.

Jev does not generate text. It takes a free-form ``state`` and a map of typed questions and returns one typed
answer per question with probabilities attached. It is served on its own endpoint, not chat completions:

* ``POST https://openrouter.ai/api/alpha/decisions`` with ``{model, state, questions}`` (the endpoint is alpha).
* ``questions`` maps an id to one of three question types: ``noul`` (yes/no, ``criteria: {"true", "false"}``),
  ``choice`` (``criteria: {label: description}``) and ``score`` (``criteria: [low, ..., high]``).
* ``usage`` carries ``input_tokens`` / ``output_tokens`` / ``cost``, not chat's ``prompt_tokens``.

Answer shapes, taken from a live call on 2026-09-25 against ``typesafe/jev-1.13`` (snapshot
``typesafe/jev-1.13-20260917``). Where they differ from the published docs, the live shape wins:

* ``choice``: ``{"type": "choice", "choice": label, "probabilities": {label: p}, "confidence": c}``, as documented.
* ``noul``: ``{"type": "noul", "noul": 0.58}``. The value is the probability of ``true``, not a boolean, and the
  answer has neither ``probabilities`` nor ``confidence``.
* ``score``: ``{"type": "score", "score": 0.85, "legend": {"0": level, ...}, "probabilities": {"0": p, ...},
  "confidence": c}``. ``score`` is the probability-weighted mean level index (a float), not a 0-based index.

A question id the model did not answer is reported in ``DecisionResult.missing``; callers treat it as an abstain.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import httpx

from pyutilz.llm.exceptions import LLMProviderError
from tenacity import AsyncRetrying, Retrying, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_base

from pyutilz.llm._retry import RETRY_WAIT, log_retry
from pyutilz.llm._openai_compat_http import parse_retry_after

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/alpha"
DEFAULT_MODEL = "typesafe/jev-1.13"
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_ATTEMPTS = 5
# 408 (request timeout) and 504 (gateway timeout) are transient per the errors doc, like the rest.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 524, 529})


class OpenRouterDecisionsError(LLMProviderError, RuntimeError):
    """A failed decisions call: an HTTP error with OpenRouter's ``{"error": {"code", "message"}}`` envelope, or a bad body.

    Rooted at ``LLMProviderError`` (with ``RuntimeError`` kept for existing callers) so the domain root catches it.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None, code: Any = None, raw: Any = None, retry_after_s: Optional[float] = None) -> None:
        """Keep the HTTP status, the envelope's ``code``, the raw body and the server's ``Retry-After`` next to the message."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.raw = raw
        self.retry_after_s = retry_after_s

    @property
    def retryable(self) -> bool:
        """True for timeouts, rate limits and transient upstream failures (408, 429, 500, 502, 503, 504, 524, 529)."""
        return self.status_code in RETRYABLE_STATUSES


@dataclass(frozen=True)
class NoulQuestion:
    """A yes/no question. ``true`` and ``false`` describe when each answer applies."""

    instructions: str
    true: str
    false: str

    def to_payload(self) -> Dict[str, Any]:
        """The wire form of the question, after checking that no field is empty."""
        if not self.instructions.strip() or not self.true.strip() or not self.false.strip():
            raise ValueError("noul question needs non-empty instructions, true and false criteria")
        return {"type": "noul", "instructions": self.instructions, "criteria": {"true": self.true, "false": self.false}}


@dataclass(frozen=True)
class ChoiceQuestion:
    """A multiple-choice question: ``criteria`` maps each option label to its description (2 to 255 options)."""

    instructions: str
    criteria: Mapping[str, str]

    def to_payload(self) -> Dict[str, Any]:
        """The wire form of the question, after the option-count and label checks."""
        n = len(self.criteria)
        if not 2 <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice question needs 2..{MAX_CHOICE_OPTIONS} options, got {n}")
        if not self.instructions.strip():
            raise ValueError("choice question needs non-empty instructions")
        if any(not str(k).strip() for k in self.criteria):
            raise ValueError("choice option labels must be non-empty")
        return {"type": "choice", "instructions": self.instructions, "criteria": {str(k): str(v) for k, v in self.criteria.items()}}


@dataclass(frozen=True)
class ScoreQuestion:
    """An ordered-score question: ``levels`` runs from lowest to highest (2 to 10 levels)."""

    instructions: str
    levels: Sequence[str]

    def to_payload(self) -> Dict[str, Any]:
        """The wire form of the question, after the level-count check."""
        if isinstance(self.levels, str):
            raise TypeError("score levels must be a sequence of strings, not one string")
        n = len(self.levels)
        if not 2 <= n <= MAX_SCORE_LEVELS:
            raise ValueError(f"score question needs 2..{MAX_SCORE_LEVELS} levels, got {n}")
        if not self.instructions.strip():
            raise ValueError("score question needs non-empty instructions")
        return {"type": "score", "instructions": self.instructions, "criteria": [str(x) for x in self.levels]}


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]
# String forward references: a runtime alias is evaluated even under `from __future__ import annotations`, and
# collections.abc classes are not subscriptable on Python 3.8 (collection error on every 3.8 CI leg).
State = Union[str, "Mapping[str, Any]", "Sequence[str]"]


@dataclass(frozen=True)
class DecisionAnswer:
    """One parsed answer.

    ``value`` is ``bool`` for noul (``p_true >= 0.5``), the chosen label for choice, and the most probable
    0-based level index for score. ``probabilities`` is keyed by label for choice, by ``"true"``/``"false"``
    for noul (derived from ``p_true``) and by level index as a string for score. ``confidence`` is what the
    model reported, a measure of how concentrated the distribution is, not of correctness; None for noul,
    which reports none.
    """

    type: str
    value: Union[bool, str, int]
    probabilities: Dict[str, float]
    confidence: Optional[float] = None
    p_true: Optional[float] = None
    expected_score: Optional[float] = None
    legend: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionResult:
    """A parsed decisions response. ``missing`` lists asked ids with no usable answer (treat as abstain)."""

    answers: Dict[str, DecisionAnswer]
    missing: List[str]
    model: str
    generation_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]
    latency_s: float
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost_reported(self) -> bool:
        """True when the response carried ``usage.cost``; a reported 0.0 is a real zero bill, None is unknown."""
        return self.cost_usd is not None


def build_payload(model: str, state: State, questions: Mapping[str, Question], *, allow_moving_alias: bool = False) -> Dict[str, Any]:
    """Build the request body, raising ``ValueError``/``TypeError`` on any limit violation before a call is spent."""
    if not model:
        raise ValueError("model is required")
    if model.startswith("~") and not allow_moving_alias:
        raise ValueError(f"{model!r} is a moving alias; pin a version such as {DEFAULT_MODEL!r} or pass allow_moving_alias=True")
    if not questions:
        raise ValueError("at least one question is required")
    wire_state: Any
    if isinstance(state, str):
        if not state.strip():
            raise ValueError("state must not be empty")
        wire_state = state
    elif isinstance(state, Mapping):
        wire_state = dict(state)
    elif isinstance(state, Sequence) and all(isinstance(s, str) for s in state):
        wire_state = list(state)
    else:
        raise TypeError("state must be a string, a mapping or a sequence of strings")
    wire_q: Dict[str, Any] = {}
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"question ids must be non-empty strings, got {qid!r}")
        if not isinstance(q, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
            raise TypeError(f"question {qid!r}: unsupported type {type(q).__name__}")
        try:
            wire_q[qid] = q.to_payload()
        except (ValueError, TypeError) as exc:
            raise type(exc)(f"question {qid!r}: {exc}") from exc
    return {"model": model, "state": wire_state, "questions": wire_q}


def _float_map(obj: Any) -> Dict[str, float]:
    """A ``{key: number}`` object as ``{str: float}``; anything else gives an empty dict."""
    if not isinstance(obj, Mapping):
        return {}
    return {str(k): float(v) for k, v in obj.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _opt_float(v: Any) -> Optional[float]:
    """A number as float; None for a missing or non-numeric value."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_answer(qtype: str, item: Mapping[str, Any]) -> Optional[DecisionAnswer]:
    """Map one ``answers[id]`` object to a ``DecisionAnswer``; None when it does not carry a usable answer."""
    atype = str(item.get("type") or qtype)
    raw = dict(item)
    if atype == "noul":
        p = _opt_float(item.get("noul"))
        if isinstance(item.get("noul"), bool):  # the docs' boolean form, in case the live shape changes back
            p = 1.0 if item["noul"] else 0.0
        if p is None:
            return None
        return DecisionAnswer(type="noul", value=p >= 0.5, probabilities={"true": p, "false": 1.0 - p}, p_true=p, raw=raw)
    probs = _float_map(item.get("probabilities"))
    conf = _opt_float(item.get("confidence"))
    if atype == "choice":
        choice = item.get("choice")
        if not isinstance(choice, str) or not choice:
            return None
        return DecisionAnswer(type="choice", value=choice, probabilities=probs, confidence=conf, raw=raw)
    if atype == "score":
        legend = {str(k): str(v) for k, v in item["legend"].items()} if isinstance(item.get("legend"), Mapping) else {}
        expected = _opt_float(item.get("score"))
        if probs:
            best = max(probs.items(), key=lambda kv: kv[1])[0]
            try:
                idx = int(best)
            except ValueError:
                return None
        elif expected is not None:
            idx = round(expected)
        else:
            return None
        return DecisionAnswer(type="score", value=idx, probabilities=probs, confidence=conf, expected_score=expected, legend=legend, raw=raw)
    logger.warning("decisions: unknown answer type %r", atype)
    return None


def parse_response(body: Mapping[str, Any], questions: Mapping[str, Question], *, latency_s: float = 0.0) -> DecisionResult:
    """Map a 200 body to a ``DecisionResult``; an ``error`` key in the body raises ``OpenRouterDecisionsError``."""
    if body.get("error"):
        raise _error_from_body(body, None)
    answers_raw = body.get("answers")
    answers_raw = answers_raw if isinstance(answers_raw, Mapping) else {}
    answers: Dict[str, DecisionAnswer] = {}
    missing: List[str] = []
    for qid, q in questions.items():
        item = answers_raw.get(qid)
        qtype = q.to_payload()["type"]
        answer = parse_answer(qtype, item) if isinstance(item, Mapping) else None
        if answer is None:
            missing.append(qid)
        else:
            answers[qid] = answer
    usage = body.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    return DecisionResult(
        answers=answers,
        missing=missing,
        model=str(body.get("model") or ""),
        generation_id=str(body.get("id") or ""),
        provider=str(body.get("provider") or ""),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cost_usd=_opt_float(usage.get("cost")),
        latency_s=latency_s,
        raw=dict(body),
    )


def _error_from_body(body: Any, status_code: Optional[int]) -> OpenRouterDecisionsError:
    """Build an error from OpenRouter's ``{"error": {"code", "message"}}`` envelope or TypeSafe's ``{"detail": ...}``."""
    code: Any = None
    message = ""
    if isinstance(body, Mapping):
        err = body.get("error")
        if isinstance(err, Mapping):
            code = err.get("code")
            message = str(err.get("message") or "")
        elif err is not None:
            message = str(err)
        elif body.get("detail") is not None:
            message = str(body.get("detail"))
    elif body is not None:
        message = str(body)
    status = status_code if status_code is not None else (code if isinstance(code, int) else None)
    return OpenRouterDecisionsError(f"decisions: HTTP {status} code={code}: {message[:500]}", status_code=status, code=code, raw=body)


def _check_response(resp: httpx.Response) -> Dict[str, Any]:
    """The response body as a dict; an HTTP error or a non-object body raises ``OpenRouterDecisionsError``."""
    try:
        data: Any = resp.json()
    except ValueError:
        data = resp.text[:500]
    if resp.status_code >= 400:
        err = _error_from_body(data, resp.status_code)
        err.retry_after_s = parse_retry_after(resp)
        raise err
    if not isinstance(data, dict):
        raise OpenRouterDecisionsError(f"decisions: non-object response: {str(data)[:300]}", status_code=resp.status_code, raw=data)
    return data


def _result_of(resp: httpx.Response, questions: Mapping[str, Question], t0: float) -> DecisionResult:
    """Check and parse one HTTP response; latency runs from ``t0`` (a ``perf_counter`` reading) to now."""
    return parse_response(_check_response(resp), questions, latency_s=time.perf_counter() - t0)


def _wait_honoring_retry_after(base: Any) -> Any:
    """``base``'s delay, lengthened to the server's ``Retry-After`` when the failed attempt carried a longer one."""

    def _wait(retry_state: Any) -> float:
        """The base backoff, raised to the server's Retry-After when it asked for longer."""
        delay = float(base(retry_state))
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None and outcome.failed else None
        hint = getattr(exc, "retry_after_s", None)
        return max(delay, float(hint)) if isinstance(hint, (int, float)) else delay

    return _wait


def _is_retryable(exc: BaseException) -> bool:
    """Retry rate limits, transient upstream statuses and transport errors (timeouts, resets); nothing else."""
    if isinstance(exc, OpenRouterDecisionsError):
        return exc.retryable
    return isinstance(exc, httpx.TransportError)


def _resolve_api_key(api_key: Optional[str]) -> str:
    """The explicit key, or ``openrouter_api_key`` from pyutilz LLM settings; raises when neither exists."""
    if api_key is None:
        from pyutilz.llm.config import get_llm_settings

        s = get_llm_settings()
        api_key = s.openrouter_api_key.get_secret_value() if s.openrouter_api_key else None
    if not api_key:
        raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY in .env or pass api_key=")
    return api_key


class OpenRouterDecisionsClient:
    """Sync and async client for ``/api/alpha/decisions``. Tests inject ``transport`` / ``async_transport``.

    ``total_cost_usd`` sums reported ``usage.cost``; ``unreported_cost_calls`` counts successful calls whose
    response carried no cost, so a missing bill never reads as a free one.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = BASE_URL,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_wait: Optional[wait_base] = None,
    ) -> None:
        """Resolve the key and prepare the HTTP clients; ``retry_wait`` defaults to the shared LLM backoff."""
        key = _resolve_api_key(api_key)
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        headers = {"Authorization": f"Bearer {key}"}
        self._client = httpx.Client(base_url=base_url, transport=transport, timeout=timeout, headers=headers)
        self._aclient_kwargs: Dict[str, Any] = dict(base_url=base_url, transport=async_transport, timeout=timeout, headers=headers)
        self._aclient: Optional[httpx.AsyncClient] = None
        self._retry_kwargs: Dict[str, Any] = dict(
            wait=_wait_honoring_retry_after(retry_wait if retry_wait is not None else RETRY_WAIT),
            stop=stop_after_attempt(max_attempts),
            retry=retry_if_exception(_is_retryable),
            before_sleep=log_retry,
            reraise=True,
        )
        self.total_cost_usd = 0.0
        self.unreported_cost_calls = 0
        self.calls = 0

    def _async_client(self) -> httpx.AsyncClient:
        """The async HTTP client, created on first async use so a sync-only caller never holds an unclosed one."""
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(**self._aclient_kwargs)
        return self._aclient

    def close(self) -> None:
        """Close the sync HTTP client; an async client opened by ``adecide`` needs ``aclose``, and a warning says so."""
        self._client.close()
        if self._aclient is not None and not self._aclient.is_closed:
            logger.warning("OpenRouterDecisionsClient.close() cannot close the async client; await aclose() instead")

    async def aclose(self) -> None:
        """Close both HTTP clients."""
        self._client.close()
        if self._aclient is not None:
            await self._aclient.aclose()

    def __enter__(self) -> OpenRouterDecisionsClient:
        """Return self for ``with`` use."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the sync client."""
        self.close()

    async def __aenter__(self) -> OpenRouterDecisionsClient:
        """Return self for ``async with`` use."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close both clients."""
        await self.aclose()

    def _account(self, result: DecisionResult) -> DecisionResult:
        """Add the call's cost to the running totals and return the result unchanged."""
        self.calls += 1
        if result.cost_usd is None:
            self.unreported_cost_calls += 1
        else:
            self.total_cost_usd += result.cost_usd
        if result.missing:
            logger.info("decisions %s: no answer for %s", result.generation_id, result.missing)
        return result

    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        *,
        model: str = DEFAULT_MODEL,
        allow_moving_alias: bool = False,
    ) -> DecisionResult:
        """Ask ``questions`` about ``state``; limits are checked before any HTTP call, transient failures retried."""
        payload = build_payload(model, state, questions, allow_moving_alias=allow_moving_alias)
        for attempt in Retrying(**self._retry_kwargs):
            with attempt:
                t0 = time.perf_counter()
                result = _result_of(self._client.post("/decisions", json=payload), questions, t0)
        return self._account(result)

    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        *,
        model: str = DEFAULT_MODEL,
        allow_moving_alias: bool = False,
    ) -> DecisionResult:
        """Async ``decide``."""
        payload = build_payload(model, state, questions, allow_moving_alias=allow_moving_alias)
        async for attempt in AsyncRetrying(**self._retry_kwargs):
            with attempt:
                t0 = time.perf_counter()
                result = _result_of(await self._async_client().post("/decisions", json=payload), questions, t0)
        return self._account(result)
