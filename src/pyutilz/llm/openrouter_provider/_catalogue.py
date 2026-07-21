"""OpenRouter model catalogue + pricing/limits lookups.

Carved from ``openrouter_provider`` (the parent monolith exceeded the
1000-LOC module rule). Holds the ``/api/v1/models`` fetch and the per-token
pricing / context-limit helpers derived from it. Re-exported unchanged from
``pyutilz.llm.openrouter_provider``.

The process-wide catalogue cache (``_MODELS_CATALOGUE``) plus its lock/URL
and the ``httpx`` name live on the package facade so callers/tests can patch
them through ``openrouter_provider.<name>``; this module resolves them LATE
via ``_pkg().<name>`` at call time so those patches are always seen here.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)


def _pkg():
    """Return the package facade (owns the mutable caches + patch targets).

    Resolved via ``sys.modules`` rather than a top-level import so the
    static import graph stays acyclic (the facade imports this submodule);
    the facade is always fully in ``sys.modules`` by the time any of these
    functions run.
    """
    return sys.modules["pyutilz.llm.openrouter_provider"]


def _catalogue_is_fresh() -> bool:
    """Return True if a cached catalogue exists and hasn't exceeded ``_MODELS_CATALOGUE_TTL_SECONDS``.

    Regression fix (2026-07-21 audit round 2, MEDIUM): the catalogue used to be cached forever
    once fetched -- pricing/context-limit data went stale for the rest of a long-running
    process's lifetime. ``_MODELS_CATALOGUE_FETCHED_AT`` defaults to +inf, so a catalogue
    injected directly (tests, or a caller pre-populating the cache) rather than through
    :func:`_fetch_models_catalogue` is treated as fresh -- only a real fetch stamps a finite
    monotonic time, after which this TTL check actually applies.
    """
    pkg = _pkg()
    if pkg._MODELS_CATALOGUE is None:
        return False
    return bool((time.monotonic() - pkg._MODELS_CATALOGUE_FETCHED_AT) < pkg._MODELS_CATALOGUE_TTL_SECONDS)


def _fetch_models_catalogue(timeout: float = 10.0) -> dict[str, dict[str, Any]]:
    """Fetch and parse the public /api/v1/models catalogue.

    Returns a dict mapping ``model_id`` to its raw entry. Returns an empty
    dict on failure WITHOUT caching the failure — a transient outage at
    startup would otherwise lock the process into zero pricing forever.
    The next call retries; once a fetch succeeds the result is cached for
    ``_MODELS_CATALOGUE_TTL_SECONDS`` (see :func:`_catalogue_is_fresh`).
    """
    if _catalogue_is_fresh():
        return _pkg()._MODELS_CATALOGUE  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
    with _pkg()._MODELS_LOCK:
        if _catalogue_is_fresh():
            return _pkg()._MODELS_CATALOGUE  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
        try:
            resp = _pkg().httpx.get(_pkg()._MODELS_URL, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            entries = payload.get("data") or payload.get("models") or []
            catalogue = {e["id"]: e for e in entries if isinstance(e, dict) and "id" in e}
        except Exception as exc:
            logger.warning(
                "OpenRouter /models catalogue fetch failed (%s); " "estimate_cost() will return 0 for this call. Next call retries.",
                exc,
            )
            return {}  # do NOT cache failure — retry on next call
        _pkg()._MODELS_CATALOGUE = catalogue
        _pkg()._MODELS_CATALOGUE_FETCHED_AT = time.monotonic()
        return _pkg()._MODELS_CATALOGUE  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


async def _ensure_catalogue_warm_async(timeout: float = 10.0) -> None:
    """Async-safe pre-warm of the catalogue cache, via ``asyncio.to_thread`` so the blocking
    ``httpx.get()`` in ``_fetch_models_catalogue`` runs off the event-loop thread.

    Regression fix (2026-07-21 audit): ``context_window``/``max_output_tokens`` are plain
    (necessarily sync) properties that call ``_fetch_models_catalogue()`` on a cache miss --
    calling that directly from an async property has NO await point, so it synchronously blocks
    the WHOLE event loop (every other in-flight coroutine) for up to ``timeout`` seconds. Callers
    on the async path should await this BEFORE the sync property is read, so the property call
    that follows is a cheap cache-hit.
    """
    if _catalogue_is_fresh():
        return
    await asyncio.to_thread(_fetch_models_catalogue, timeout)


def _per_token_cost_pair(model: str) -> tuple[float, float]:
    """Return (input_cost_per_1m, output_cost_per_1m) for ``model``.

    OpenRouter publishes pricing as USD-per-token strings under
    ``pricing.prompt`` and ``pricing.completion``. Multiply by 1e6 to match
    the per-1M-tokens convention used everywhere else in pyutilz.
    """
    catalogue = _fetch_models_catalogue()
    entry = catalogue.get(model)
    if not entry:
        return (0.0, 0.0)
    pricing = entry.get("pricing") or {}
    try:
        in_per_1m = float(pricing.get("prompt", "0") or "0") * 1_000_000
        out_per_1m = float(pricing.get("completion", "0") or "0") * 1_000_000
    except (TypeError, ValueError):
        return (0.0, 0.0)
    return (in_per_1m, out_per_1m)


def _resolve_model_limits(model: str) -> tuple[int | None, int | None]:
    """Return (context_length, max_completion_tokens) for ``model``.

    Prefers ``top_provider.context_length`` over the model-level
    ``context_length`` — the top-level field is the model's THEORETICAL
    max, but OR routes through one specific upstream which may cap shorter
    (e.g. a 1M-context model served by a provider that only exposes 200K).
    The upstream cap is what actually triggers ``HTTP 400: prompt too long``.

    Returns ``(None, None)`` for unknown models — callers fall back to
    the class-level defaults.
    """
    catalogue = _fetch_models_catalogue()
    entry = catalogue.get(model)
    if not entry:
        return (None, None)
    top = entry.get("top_provider") or {}
    ctx = top.get("context_length") or entry.get("context_length")
    max_out = top.get("max_completion_tokens")
    try:
        ctx_int = int(ctx) if ctx else None
    except (TypeError, ValueError):
        ctx_int = None
    try:
        max_out_int = int(max_out) if max_out else None
    except (TypeError, ValueError):
        max_out_int = None
    return (ctx_int, max_out_int)
