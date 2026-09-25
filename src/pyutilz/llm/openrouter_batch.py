"""OpenRouter Batch API client: asynchronous, discounted (~50%) bulk chat completions.

OpenRouter's Batch API (https://openrouter.ai/docs/batch-quickstart) is a real async job API,
not a slow synchronous tier:

* ``POST /api/v1/batches`` with ``{endpoint, model, [provider], [completion_window], requests}``
  answers ``202`` and ``status: "validating"``; ``requests`` is an inline array of
  ``{custom_id, body}`` (``body`` is an ordinary chat-completions request without ``model``).
* ``GET /api/v1/batches/{id}`` is polled until a terminal status (``completed``, ``failed``,
  ``expired``, ``cancelled``); the finished batch carries ``results`` inline, each holding
  exactly one of ``response`` (``{status_code, body}``) or ``error``.
* Only ``completion_window="24h"`` is accepted, and ``provider.only`` is the only routing
  preference (``order``/``sort``/``allow_fallbacks`` are rejected). Streaming is rejected.

A submitted batch is paid work, so the job id is written to ``state_path`` (atomically) the
moment the submit returns, before anything else can fail; ``run()`` resumes an existing job from
that file instead of submitting a second, duplicate paid batch.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from pyutilz.llm.exceptions import LLMProviderError

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
CHAT_ENDPOINT = "/v1/chat/completions"


class OpenRouterBatchError(LLMProviderError, RuntimeError):
    """A batch-level failure: rejected submit, terminal non-completed status, or deadline hit.

    Rooted at ``LLMProviderError`` so ``except LLMProviderError`` catches it like every other provider failure; the
    ``RuntimeError`` base keeps existing ``except RuntimeError`` callers working.
    """


@dataclass(frozen=True)
class BatchRequest:
    """One request of a batch. ``body`` is a chat-completions body (messages, max_tokens, ...)."""

    custom_id: str
    body: Mapping[str, Any]


@dataclass
class BatchResult:
    """Per-request outcome, keyed back to the caller by ``custom_id``."""

    custom_id: str
    text: str | None = None
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    status_code: int | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the request produced text and no error."""
        return self.error is None and self.text is not None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as JSON through a temporary file and a rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json.dumps(payload, indent=1, sort_keys=True).encode("utf-8"))
    os.replace(tmp, path)


def load_job_state(state_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return the persisted job state (``{"batch_id", "model", "custom_ids", ...}``) or None."""
    p = Path(state_path)
    if not p.exists():
        return None
    data = json.loads(p.read_bytes().decode("utf-8"))
    return data if isinstance(data, dict) else None


def build_submit_payload(
    model: str,
    requests: Sequence[BatchRequest],
    *,
    endpoint: str = CHAT_ENDPOINT,
    provider_only: Sequence[str] | None = None,
    completion_window: str = "24h",
) -> dict[str, Any]:
    """Build the submit body. Key order matters: the docs require ``requests`` to come last."""
    if not requests:
        raise ValueError("a batch needs at least one request")
    ids = [r.custom_id for r in requests]
    dupes = sorted({i for i in ids if ids.count(i) > 1}) if len(set(ids)) != len(ids) else []
    if dupes:
        raise ValueError(f"custom_id values must be unique within a batch; duplicated: {dupes[:5]}")
    payload: dict[str, Any] = {"endpoint": endpoint, "model": model}
    if provider_only:
        payload["provider"] = {"only": list(provider_only)}
    payload["completion_window"] = completion_window
    reqs = []
    for r in requests:
        body = dict(r.body)
        if body.get("stream"):
            raise ValueError(f"request {r.custom_id!r}: streaming is not supported by the Batch API")
        body.pop("model", None)  # the batch-level model is the only one honoured
        reqs.append({"custom_id": r.custom_id, "body": body})
    payload["requests"] = reqs
    return payload


def _text_of(content: Any) -> str | None:
    """Message content as plain text: a string as is, a list of content parts joined by their text."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):  # content-parts form
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content)


def parse_result_item(item: Mapping[str, Any]) -> BatchResult:
    """Map one ``results[]`` entry to a ``BatchResult``; any error lands in ``.error``, never raises."""
    cid = str(item.get("custom_id"))
    res = BatchResult(custom_id=cid, raw=dict(item))
    err = item.get("error")
    resp = item.get("response")
    if err:
        res.error = json.dumps(err, sort_keys=True) if not isinstance(err, str) else err
        return res
    if not isinstance(resp, dict):
        res.error = "result has neither response nor error"
        return res
    res.status_code = resp.get("status_code")
    body = resp.get("body") or {}
    if not isinstance(body, dict):
        res.error = f"unparseable response body: {str(body)[:200]}"
        return res
    if res.status_code is not None and not 200 <= int(res.status_code) < 300:
        res.error = f"HTTP {res.status_code}: {json.dumps(body.get('error', body), sort_keys=True)[:500]}"
        return res
    usage = body.get("usage") or {}
    res.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    res.completion_tokens = int(usage.get("completion_tokens") or 0)
    cost = usage.get("cost")
    res.cost_usd = float(cost) if cost is not None else None
    choices = body.get("choices") or []
    if not choices:
        res.error = f"no choices in response body: {json.dumps(body.get('error', body), sort_keys=True)[:300]}"
        return res
    choice = choices[0]
    msg = choice.get("message") or {}
    res.text = _text_of(msg.get("content"))
    res.reasoning = _text_of(msg.get("reasoning"))
    res.finish_reason = choice.get("finish_reason")
    if res.text is None:
        res.error = f"empty content (finish_reason={res.finish_reason})"
    return res


def parse_results(batch: Mapping[str, Any], expected_ids: Sequence[str] | None = None) -> dict[str, BatchResult]:
    """Map a finished batch's ``results`` by custom_id. Expected ids missing from the batch get an error result."""
    out: dict[str, BatchResult] = {}
    for item in batch.get("results") or []:
        r = parse_result_item(item)
        out[r.custom_id] = r
    for cid in expected_ids or ():
        if cid not in out:
            out[cid] = BatchResult(custom_id=cid, error=f"missing from batch results (batch status={batch.get('status')})")
    return out


class OpenRouterBatchClient:
    """Submit / poll / collect OpenRouter batches. ``transport`` lets tests inject ``httpx.MockTransport``."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if api_key is None:
            from pyutilz.llm.config import get_llm_settings

            s = get_llm_settings()
            api_key = s.openrouter_api_key.get_secret_value() if s.openrouter_api_key else None
        if not api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY in .env or pass api_key=")
        self._client = httpx.Client(base_url=base_url, transport=transport, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self._sleep = sleep
        self._clock = clock

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> OpenRouterBatchClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _json(self, resp: httpx.Response, what: str) -> dict[str, Any]:
        """The response body as a dict; an HTTP error or a non-object body raises ``OpenRouterBatchError`` naming ``what``."""
        if resp.status_code >= 400:
            raise OpenRouterBatchError(f"{what}: HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        if not isinstance(data, dict):
            raise OpenRouterBatchError(f"{what}: non-object response")
        return data

    def submit(
        self,
        model: str,
        requests: Sequence[BatchRequest],
        *,
        state_path: str | os.PathLike[str] | None = None,
        provider_only: Sequence[str] | None = None,
        endpoint: str = CHAT_ENDPOINT,
    ) -> str:
        """Submit and return the batch id; with ``state_path`` the id is persisted before returning."""
        payload = build_submit_payload(model, requests, endpoint=endpoint, provider_only=provider_only)
        data = self._json(self._client.post("/batches", json=payload), "submit")
        batch_id = data.get("id")
        if not batch_id:
            raise OpenRouterBatchError(f"submit: no batch id in response: {str(data)[:300]}")
        if state_path is not None:
            _atomic_write_json(
                Path(state_path),
                {
                    "batch_id": batch_id,
                    "model": model,
                    "custom_ids": [r.custom_id for r in requests],
                    "submitted_at": time.time(),
                },
            )
        logger.info("OpenRouter batch %s submitted: %d requests, model=%s", batch_id, len(requests), model)
        return str(batch_id)

    def get(self, batch_id: str) -> dict[str, Any]:
        """The current state of one batch job, results included once it has finished."""
        return self._json(self._client.get(f"/batches/{batch_id}"), f"get {batch_id}")

    def wait(
        self,
        batch_id: str,
        *,
        deadline_s: float = 26 * 3600.0,
        initial_interval_s: float = 10.0,
        max_interval_s: float = 300.0,
        backoff: float = 1.5,
        on_poll: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll with jittered exponential backoff until a terminal status or the wall-clock deadline.

        Transient HTTP/network errors while polling are logged and retried: the job lives server-side
        and a lost poll must never abandon it. Raises ``OpenRouterBatchError`` on deadline (the job is
        NOT cancelled; the persisted id lets a later run pick it up).
        """
        start = self._clock()
        interval = initial_interval_s
        failed_polls = 0
        while True:
            try:
                data = self.get(batch_id)
            except (httpx.HTTPError, OpenRouterBatchError) as exc:
                # One warning per outage: the first failed poll warns, the rest of the same run of failures go to debug.
                failed_polls += 1
                log = logger.warning if failed_polls == 1 else logger.debug
                log("poll of batch %s failed (%d in a row), will retry: %s", batch_id, failed_polls, exc)
                data = None
            if data is not None:
                failed_polls = 0
                if on_poll is not None:
                    on_poll(data)
                if data.get("status") in TERMINAL_STATUSES:
                    return data
            elapsed = self._clock() - start
            if elapsed >= deadline_s:
                raise OpenRouterBatchError(f"batch {batch_id} not finished after {elapsed:.0f}s (deadline {deadline_s:.0f}s)")
            self._sleep(min(interval * random.uniform(0.9, 1.1), max(0.0, deadline_s - elapsed)))
            interval = min(interval * backoff, max_interval_s)

    def run(
        self,
        model: str,
        requests: Sequence[BatchRequest],
        *,
        state_path: str | os.PathLike[str],
        provider_only: Sequence[str] | None = None,
        deadline_s: float = 26 * 3600.0,
        initial_interval_s: float = 10.0,
        max_interval_s: float = 300.0,
        on_poll: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, BatchResult]:
        """Submit (or resume the job persisted at ``state_path``), wait, and map results by custom_id.

        A terminal status other than ``completed`` still returns whatever results exist, with the
        missing ids carrying an error. The state file is kept; the caller deletes it once it has
        stored the results.
        """
        state = load_job_state(state_path)
        if state and state.get("batch_id"):
            if state.get("model") != model or list(state.get("custom_ids") or []) != [r.custom_id for r in requests]:
                raise OpenRouterBatchError(
                    f"state file {state_path} holds batch {state['batch_id']} for a different model/request set; "
                    "collect or delete it before submitting a new batch"
                )
            batch_id = str(state["batch_id"])
            logger.info("resuming OpenRouter batch %s from %s", batch_id, state_path)
        else:
            batch_id = self.submit(model, requests, state_path=state_path, provider_only=provider_only)
        final = self.wait(batch_id, deadline_s=deadline_s, initial_interval_s=initial_interval_s, max_interval_s=max_interval_s, on_poll=on_poll)
        if final.get("status") != "completed":
            logger.warning("batch %s ended with status=%s error=%s", batch_id, final.get("status"), final.get("error"))
        return parse_results(final, [r.custom_id for r in requests])
