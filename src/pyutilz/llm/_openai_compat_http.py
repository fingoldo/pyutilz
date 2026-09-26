"""HTTP envelope parsing and SSE fragment reassembly for OpenAI-compatible providers.

Carved out of ``openai_compat.py`` (which had grown past the repo's 1000-LOC module budget) as a
cohesive block: everything here is module-level, provider-independent and depends on nothing in
``OpenAICompatibleProvider``. The facade re-exports every name, so ``from pyutilz.llm.openai_compat
import parse_response_envelope`` keeps resolving.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from pyutilz.llm.exceptions import LLMProviderError, LLMStreamInterruptedError, LLMUnparseableResponseError

logger = logging.getLogger(__name__)

# orjson is faster than stdlib json for the per-chunk streaming parse below;
# resolved once at import time (not per-call, this loop runs per token chunk)
# and falls back to stdlib if missing (kept optional -- core has no hard
# requirements).
_json_backend: Any
try:
    import orjson as _json_backend  # type: ignore[import-not-found,no-redef]  # absent in a minimal install (import-not-found), resolvable in CI where it then redefines the annotation above (no-redef); BOTH codes are needed, which environment you check from decides which one fires

    _json_loads = _json_backend.loads
    _JSONDecodeError = _json_backend.JSONDecodeError
except ImportError:
    import json as _json_backend  # type: ignore[no-redef]

    _json_loads = _json_backend.loads
    _JSONDecodeError = _json_backend.JSONDecodeError


_NON_RETRYABLE_STATUSES: frozenset[int] = frozenset({
    400,  # bad request — body invalid, retry won't help
    401,  # unauthorized — wrong/expired API key
    403,  # forbidden — RBAC/region block
    404,  # not found — model deprecated or misspelled; OpenRouter
          # /chat/completions returns 404 even when /models/{id}/endpoints
          # still lists provider endpoints (catalog can lag). Retrying with
          # exponential backoff burns 30+ minutes per dead model before
          # the wall-clock timeout fires.
    405,  # method not allowed — endpoint doesn't accept POST. Same
          # underlying pattern as 404: catalog claims model is alive, but
          # the actual /chat/completions endpoint won't service the call.
          # Observed 2026-05-05 on llama-guard-4-12b, nemotron-3-nano-30b-a3b,
          # olmo-3.1-32b-instruct — 110+ calls each spinning through 50
          # retry attempts, blocking the concurrency pool for hours.
    410,  # gone — endpoint permanently removed; identical reasoning to 404.
    422,  # unprocessable entity — request well-formed but semantically
          # rejected (bad enum, schema violation); won't be accepted on retry.
})


#: String error codes OpenRouter (and upstreams it relays) put in a 200 body or chunk, mapped to the HTTP status that
#: carries the same meaning, so the shared retry policy treats them like their numeric forms.
_STRING_ERROR_CODE_STATUS: dict[str, int] = {
    "server_error": 502,
    "internal_error": 500,
    "internal_server_error": 500,
    "bad_gateway": 502,
    "upstream_error": 502,
    "provider_error": 502,
    "service_unavailable": 503,
    "overloaded": 503,
    "overloaded_error": 503,
    "timeout": 504,
    "gateway_timeout": 504,
    "rate_limit_exceeded": 429,
    "rate_limited": 429,
    "too_many_requests": 429,
}


def error_code_status(raw_code: Any) -> int:
    """The HTTP status an in-body error ``code`` stands for: a numeric code as is, a known transient string mapped, else 0."""
    if isinstance(raw_code, bool):
        return 0
    if isinstance(raw_code, int):
        return raw_code
    if isinstance(raw_code, str):
        text = raw_code.strip()
        if text.isdigit():
            return int(text)
        return _STRING_ERROR_CODE_STATUS.get(text.lower(), 0)
    return 0


def _status_is_transient(status: int) -> bool:
    """429, 402 and 5xx are worth another attempt; anything else (including the unknown 0) is not."""
    return status in (402, 429) or status >= 500


def raise_for_error_in_body(payload: dict[str, Any], request: httpx.Request, provider_name: str, partial_text: str = "") -> None:
    """Raise the error a 200 body or stream chunk carries; return when it carries none.

    * A chunk WITH choices and an ``error`` (OpenRouter's mid-stream error event) raises :class:`LLMStreamInterruptedError`
      carrying ``partial_text``: the upstream had started answering, so this is never an HTTP-level retry.
    * A body with no choices and a code that maps to a status of 400 or above raises ``httpx.HTTPStatusError`` with that
      status, so the shared policy retries 429/5xx and refuses 400/404 at once. String codes such as ``server_error`` map
      through :func:`error_code_status`; they used to become 0 and a non-retryable plain error.
    * Anything else raises ``LLMProviderError`` naming the message.
    """
    error = payload.get("error")
    if not isinstance(error, dict):
        return
    raw_code = error.get("code")
    status = error_code_status(raw_code)
    error_message = str(error.get("message") or error)
    if payload.get("choices"):
        raise LLMStreamInterruptedError(
            f"{provider_name} reported an error after it started answering (code={raw_code!r}): {error_message}",
            code=raw_code,
            partial_text=partial_text,
            retryable=status == 0 or _status_is_transient(status),
        )
    if status >= 400:
        raise httpx.HTTPStatusError(
            f"{provider_name} returned error {raw_code} in place of choices: {error_message}",
            request=request,
            response=httpx.Response(status, request=request, text=error_message),
        )
    raise LLMProviderError(f"{provider_name} returned no choices; error in body: {error_message}")


def merge_extra_body(body: dict[str, Any], extra: Any) -> dict[str, Any]:
    """``body`` with ``extra`` merged over it, LAST: its top-level keys win, and a dict value merges one level into a dict
    already present (so ``{"provider": {"zdr": True}}`` adds to the routing block instead of replacing it).

    ``extra`` may be a mapping or a tuple of ``(key, value)`` pairs (the hashable form a factory cache key needs).
    """
    items = extra.items() if hasattr(extra, "items") else extra
    out = dict(body)
    for key, value in items:
        current = out.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            out[key] = {**current, **value}
        else:
            out[key] = value
    return out


def wait_honoring_retry_after(base_wait: Any) -> Any:
    """A tenacity wait: ``base_wait``'s delay, or the server's ``Retry-After`` on a 429 when that is longer.

    The errors doc asks clients to respect ``Retry-After`` on 429; the buffered paths used the exponential schedule
    alone and could retry inside the window the server named, re-triggering the limit.
    """

    def _wait(retry_state: Any) -> float:
        """The base backoff, raised to the server's Retry-After when it asked for longer."""
        delay = float(base_wait(retry_state))
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None and outcome.failed else None
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 429:
            hint = parse_retry_after(response)
            if hint is not None and hint > delay:
                return hint
        return delay

    return _wait


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for transient HTTP errors that should be retried infinitely.

    Non-retryable: 400, 401, 403, 404, 410, 422 (see ``_NON_RETRYABLE_STATUSES``).
    Retryable: 402 (billing), 429 (rate limit), 5xx, transport errors.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in _NON_RETRYABLE_STATUSES
    if isinstance(exc, LLMStreamInterruptedError):
        return exc.retryable
    # An empty or non-JSON body on an otherwise-successful response is the same class of transient fault as
    # a transport error, but `resp.json()` reports it as `json.JSONDecodeError` (a `ValueError`), which
    # matches neither branch above - so it used to escape this predicate entirely and fail the call outright.
    if isinstance(exc, LLMUnparseableResponseError):
        return True
    return isinstance(exc, httpx.TransportError)


def parse_response_envelope(resp: Any, provider_name: str) -> dict[str, Any]:
    """`resp.json()` as a dict, or `LLMUnparseableResponseError` so the retry decorator can see it.

    Three failure shapes collapse into one raise, because the caller's response to all three is the same
    (re-issue): an empty body, a body that is not JSON at all (an intermediary's HTML gateway page), and
    valid JSON that is not an object (a bare string or list, which every downstream `.get` would crash on).
    The excerpt is capped rather than dropped - a log line saying only "not JSON" cannot distinguish a
    504 page from a truncated envelope, and that distinction is the whole reason to read the log.
    """
    body = resp.text or ""
    if not body.strip():
        raise LLMUnparseableResponseError(
            f"{provider_name} returned an empty body with status {resp.status_code}",
            status_code=resp.status_code,
        )
    try:
        data = resp.json()
    except (ValueError, _JSONDecodeError) as exc:
        raise LLMUnparseableResponseError(
            f"{provider_name} returned a non-JSON body with status {resp.status_code}: {exc}",
            status_code=resp.status_code,
            body_excerpt=body[:500],
        ) from exc
    if not isinstance(data, dict):
        raise LLMUnparseableResponseError(
            f"{provider_name} returned JSON of type {type(data).__name__}, not the expected object",
            status_code=resp.status_code,
            body_excerpt=body[:500],
        )
    return data


def parse_retry_after(resp: Any) -> float | None:
    """Parse ``Retry-After`` / ``retry-after-ms`` headers from an HTTP response.

    Providers (Anthropic, OpenAI, Gemini) return ``Retry-After`` on 429 —
    honouring it is cheaper than blind exponential backoff and avoids
    re-triggering the rate limit. Returns seconds (float) or None.

    Honoured by the manual retry loop in ``generate_stream`` (it takes
    ``max(server_hint, exponential_floor)`` between attempts) and, through
    :func:`wait_honoring_retry_after`, by the buffered ``generate()`` wait on
    a 429: the delay is only ever lengthened, never shortened.
    """
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    # Case-insensitive lookup via dict-like; httpx does this natively.
    for key in ("retry-after-ms", "x-retry-after-ms"):
        val = headers.get(key)
        if val:
            try:
                return float(val) / 1000.0
            except (TypeError, ValueError):
                pass
    for key in ("retry-after", "x-retry-after"):
        val = headers.get(key)
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                # RFC-7231 also allows HTTP-date; fall back to None and
                # let the generic backoff kick in.
                pass
    return None


def _accumulate_stream_tool_calls(acc: dict, fragments: object) -> None:
    """Merge one SSE delta's ``tool_calls`` fragments into ``acc``, keyed by the OpenAI ``index``.

    Streaming splits a single tool call across many chunks: the first carries ``id``/``type`` and
    the function ``name``, every later one appends a slice of the JSON ``arguments`` string. Only
    the reassembled whole is a usable tool call, which is why the streaming path used to report
    none at all (2026-09-03 audit F11). Malformed fragments are skipped rather than raised on --
    a partial tool call must never turn a successful stream into an exception.
    """
    if not isinstance(fragments, list):
        return
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        index = fragment.get("index", 0)
        if not isinstance(index, int):
            continue
        entry = acc.setdefault(index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        if fragment.get("id"):
            entry["id"] = fragment["id"]
        if fragment.get("type"):
            entry["type"] = fragment["type"]
        func = fragment.get("function")
        if isinstance(func, dict):
            if func.get("name"):
                entry["function"]["name"] = func["name"]
            args = func.get("arguments")
            if isinstance(args, str):
                entry["function"]["arguments"] += args
