"""Per-request metadata from ``GET /api/v1/generation?id=<generation_id>``, compacted for storage.

Every OpenRouter call returns a generation id, and the record behind it names the provider that served the request, the
real model, the billed cost and the timings. :func:`fetch_generation_metadata` reads that record and returns only the
fields we store. Empty values are dropped (a null field is not stored, to save space); zeros and False are kept.

Best-effort, like :func:`~pyutilz.llm.openrouter_speech.fetch_generation_provider`: it never raises. A malformed id, a
transport error, a non-2xx status or a body without ``data`` gives None and a logged warning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .openrouter_speech import BASE_URL, _GENERATION_ID_RE

httpx: Any
try:
    import httpx  # type: ignore[no-redef]
except ImportError:  # optional extra 'llm'
    httpx = None

logger = logging.getLogger(__name__)

__all__ = ["GENERATION_METADATA_KEYS", "PROVIDER_RESPONSE_KEYS", "fetch_generation_metadata"]

#: The only top-level keys stored. Anything else in the API record (workspace ids, cache discounts, ...) is dropped.
GENERATION_METADATA_KEYS = (
    "provider_name",
    "model",
    "total_cost",
    "usage",
    "latency",
    "generation_time",
    "tokens_prompt",
    "tokens_completion",
    "finish_reason",
    "provider_responses",
    "upstream_id",
    "created_at",
    "cancelled",
    "streamed",
    "is_byok",
    "api_type",
    "id",
)
#: The only keys stored from each entry of ``provider_responses``.
PROVIDER_RESPONSE_KEYS = ("endpoint_id", "id", "is_byok", "latency", "model_permaslug", "provider_name", "status")
_TIMEOUT_S = 15.0


def _keep(value: Any) -> bool:
    """False for None and the empty string; 0, 0.0 and False are real values and are kept."""
    return value is not None and not (isinstance(value, str) and value == "")


def _compact(record: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """The listed keys of ``record`` whose values are kept by :func:`_keep`, in the listed order."""
    return {key: record[key] for key in keys if key in record and _keep(record[key])}


def _provider_responses(entries: List[Any]) -> List[Dict[str, Any]]:
    """Compact each provider entry; non-dict entries are skipped."""
    return [_compact(entry, PROVIDER_RESPONSE_KEYS) for entry in entries if isinstance(entry, dict)]


def fetch_generation_metadata(generation_id: str, api_key: str, client: Any = None) -> Optional[Dict[str, Any]]:
    """The stored fields of one generation record, or None when the lookup fails.

    ``client`` is an optional ``httpx.Client`` (tests pass one built on ``httpx.MockTransport``); when it is omitted a
    short-lived client is opened and closed here. The key is sent as a bearer header and is never logged.
    Returns a dict with a subset of :data:`GENERATION_METADATA_KEYS`: keys whose value is None or "" are absent, and
    ``provider_responses`` is a list of dicts from which the same empty values are removed.
    """
    if not _GENERATION_ID_RE.fullmatch(generation_id or ""):
        logger.warning("generation metadata skipped: malformed generation id %r", generation_id)
        return None
    if client is None and httpx is None:
        logger.warning("generation metadata skipped: httpx is not installed")
        return None
    http = client if client is not None else httpx.Client(timeout=_TIMEOUT_S)
    try:
        resp = http.get(f"{BASE_URL}/generation", params={"id": generation_id}, headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            raise ValueError("response body has no data object")
        out = _compact(data, [key for key in GENERATION_METADATA_KEYS if key != "provider_responses"])
        if isinstance(data.get("provider_responses"), list):
            out["provider_responses"] = _provider_responses(data["provider_responses"])
        return out
    except Exception as exc:  # best-effort: a failed lookup is logged and never stops a run
        logger.warning("generation metadata lookup failed for %s (%s: %s)", generation_id, type(exc).__name__, exc)
        return None
    finally:
        if client is None:
            http.close()
