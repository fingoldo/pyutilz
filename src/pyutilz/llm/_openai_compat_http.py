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

from pyutilz.llm.exceptions import LLMUnparseableResponseError

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


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for transient HTTP errors that should be retried infinitely.

    Non-retryable: 400, 401, 403, 404, 410, 422 (see ``_NON_RETRYABLE_STATUSES``).
    Retryable: 402 (billing), 429 (rate limit), 5xx, transport errors.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in _NON_RETRYABLE_STATUSES
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
    ``max(server_hint, exponential_floor)`` between attempts). The
    non-streaming ``generate()`` path uses tenacity's pure exponential+jitter
    wait and does NOT read this hint — wiring it into the shared tenacity
    wait would change retry timing for every provider and is left out
    pending a benchmark on rate-limit-heavy paths.
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
