"""OpenRouter live-health enrichment + public model-catalogue listing.

Carved from ``openrouter_provider`` (the parent monolith exceeded the
1000-LOC module rule). Holds the per-model ``/endpoints`` health cache,
the concurrent health-enrichment fan-out, and the public
``list_openrouter_models`` / ``clear_openrouter_caches`` helpers.
Re-exported unchanged from ``pyutilz.llm.openrouter_provider``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from pyutilz.llm.openrouter_provider._catalogue import _fetch_models_catalogue

logger = logging.getLogger(__name__)


def _pkg():
    """Return the package facade (owns mutable caches + monkeypatch targets).

    Resolved via ``sys.modules`` (not a top-level import) so the static
    import graph stays acyclic; the facade is fully initialised by the
    time any of these functions run.
    """
    return sys.modules["pyutilz.llm.openrouter_provider"]


# The process-wide health cache + its cap/lock + the endpoints URL template
# live on the package facade (``openrouter_provider._HEALTH_CACHE`` etc.) so
# callers/tests that mutate/patch them through the facade namespace hit the
# SAME objects this module reads. Resolved LATE via ``_pkg().<name>`` below.


def clear_openrouter_caches(*, models: bool = True, health: bool = True) -> None:
    """Manually clear in-process caches for the OR catalogue and/or health.

    Useful for tests or long-running services that want to force a refresh
    on next call. ``list_openrouter_models(refresh=True)`` already drops
    the catalogue cache; this exists primarily for the health side, which
    has no other knob besides waiting out the TTL.
    """
    if models:
        with _pkg()._MODELS_LOCK:
            _pkg()._MODELS_CATALOGUE = None
    if health:
        with _pkg()._HEALTH_CACHE_LOCK:
            _pkg()._HEALTH_CACHE.clear()


def _resolve_or_api_key(explicit: str | None) -> str | None:
    """Resolve an OR API key for free helper endpoints — explicit > env > settings.

    Returns ``None`` if nothing is configured. Helpers should treat ``None``
    as "skip the optional auth-required step" rather than raise.
    """
    if explicit:
        return explicit
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    try:
        s = _pkg().get_llm_settings()
        if s.openrouter_api_key:
            return s.openrouter_api_key.get_secret_value()
    except Exception:
        pass
    return None


def _sweep_health_cache_locked(now: float) -> None:
    """Evict cached entries when ``_HEALTH_CACHE`` grows past the cap.

    Caller MUST hold ``_HEALTH_CACHE_LOCK``. Two-pass strategy:
      1. Drop entries older than 24h regardless of TTL (definitely stale).
      2. If still over cap, drop oldest entries until 128 below cap so
         we don't sweep on every subsequent miss.

    Sized for the long-lived-process case: an audit script probing 10k
    distinct models over hours would otherwise grow ``_HEALTH_CACHE``
    unboundedly. The dict survives the catalogue cache because models
    can rotate in/out of OR's listing while the same id stays valid.
    """
    cache = _pkg()._HEALTH_CACHE
    max_size = _pkg()._HEALTH_CACHE_MAX_SIZE
    if len(cache) <= max_size:
        return
    cutoff_long = now - 86400.0
    for k in [k for k, (ts, _) in cache.items() if ts < cutoff_long]:
        del cache[k]
    if len(cache) <= max_size:
        return
    sorted_keys = sorted(cache, key=lambda k: cache[k][0])
    drop_count = len(cache) - max_size + 128
    for k in sorted_keys[:drop_count]:
        del cache[k]


def _fetch_endpoints_for_model(
    model_id: str,
    api_key: str | None = None,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """``GET /api/v1/models/{model}/endpoints`` — per-upstream stats.

    Returns the raw endpoints list (one entry per upstream provider OR
    routes through). Auth required (Bearer token), but the call itself
    doesn't bill against credits.

    Pass ``client=`` (a ``httpx.Client`` with auth header preconfigured)
    when fanning out from a thread pool — the shared connection pool
    avoids 300 fresh TLS handshakes that would dominate CPU otherwise.
    Without ``client``, falls back to a one-shot ``httpx.get()`` (used
    by tests and standalone callers).
    """
    url = _pkg()._ENDPOINTS_URL_TEMPLATE.format(model=model_id)
    if client is not None:
        resp = client.get(url, timeout=timeout)
    else:
        if not api_key:
            raise ValueError(
                "_fetch_endpoints_for_model: pass either api_key= or a "
                "preconfigured client= with auth headers."
            )
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") or {}
    return data.get("endpoints") or []

def _normalize_uptime(value: Any) -> float | None:
    """Coerce uptime to 0-1 fraction.

    OpenRouter's live response sometimes returns uptime as a 0-100
    percentage (e.g. 99.806 for 99.806% uptime) AND sometimes as a
    0-1 fraction (e.g. 0.998). Both shapes appear in the wild.
    Normalise to fraction so downstream code can compare against
    a 0-1 ``min_uptime`` argument consistently.

    Values outside ``[0, 100]`` are treated as malformed (logged and
    coerced to ``None``) — silently dividing 1.5 down to 0.015 hides
    upstream schema drift.

    Returns None on missing / non-numeric / out-of-range input.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0.0 or v > 100.0:
        logger.warning(
            "OpenRouter uptime out of expected [0, 100] range: %r; treating as None.",
            value,
        )
        return None
    return v / 100.0 if v > 1.0 else v

def _summarize_endpoints(endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize the per-endpoint payload + compute best-of aggregates.

    Endpoint fields renamed to a stable shape so downstream code doesn't
    chase OR's snake_case-with-suffix conventions: ``uptime_last_30m`` →
    ``uptime_30m``, ``latency_last_30m.p50`` → ``latency_p50_ms``, etc.

    Uptime fields are normalised to 0-1 fractions via ``_normalize_uptime``
    — the live API has been observed serving both 0-1 fractions and
    0-100 percentages; this normalises to the documented fraction shape
    so downstream filters using ``min_uptime`` (0-1) work consistently.

    Best-of aggregates pick the most favourable value across all upstreams
    for the model — useful as a single sortable metric per row:
        ``best_uptime_30m``       — max uptime any backend hit in last 30m
        ``best_latency_p50_ms``   — fastest backend's median latency
        ``best_throughput_p50_tps`` — fastest backend's median throughput
    """
    if not endpoints:
        return {
            "endpoints": [],
            "best_uptime_30m": None,
            "best_latency_p50_ms": None,
            "best_throughput_p50_tps": None,
        }

    norm: list[dict[str, Any]] = []
    for e in endpoints:
        latency = e.get("latency_last_30m") or {}
        throughput = e.get("throughput_last_30m") or {}
        norm.append({
            "provider_name": e.get("provider_name"),
            "name": e.get("name"),
            "status": e.get("status"),
            "uptime_5m": _normalize_uptime(e.get("uptime_last_5m")),
            "uptime_30m": _normalize_uptime(e.get("uptime_last_30m")),
            "uptime_1d": _normalize_uptime(e.get("uptime_last_1d")),
            "latency_p50_ms": latency.get("p50"),
            "latency_p95_ms": latency.get("p95") or latency.get("p90"),
            "throughput_p50_tps": throughput.get("p50"),
            "context_length": e.get("context_length"),
            "max_completion_tokens": e.get("max_completion_tokens"),
            "max_prompt_tokens": e.get("max_prompt_tokens"),
            "pricing": e.get("pricing"),
            "supported_parameters": e.get("supported_parameters"),
            "quantization": e.get("quantization"),
            "supports_implicit_caching": e.get("supports_implicit_caching"),
        })

    def _best(field: str, op):
        vals = [n[field] for n in norm if isinstance(n[field], (int, float))]
        return op(vals) if vals else None

    return {
        "endpoints": norm,
        "best_uptime_30m": _best("uptime_30m", max),
        "best_latency_p50_ms": _best("latency_p50_ms", min),
        "best_throughput_p50_tps": _best("throughput_p50_tps", max),
    }

def _enrich_with_health(
    rows: list[dict[str, Any]],
    api_key: str,
    min_uptime: float,
    max_workers: int,
    timeout: float,
    health_ttl_seconds: float,
) -> list[dict[str, Any]]:
    """Concurrently fetch /endpoints for each row, attach ``health``,
    and drop rows where ``best_uptime_30m`` < ``min_uptime`` or fetch failed.

    Three perf optimisations vs. the naive "spawn 300 httpx.get calls":

    1. **TTL cache** keyed by model id (``health_ttl_seconds=0`` to bypass).
       Subsequent calls within the TTL skip HTTP entirely.
    2. **Shared ``httpx.Client``** with a connection pool — one TLS
       handshake per host instead of one per call. Was ~95% of the CPU
       cost on a cold run with 300 models.
    3. **Bounded concurrency** (default 8). The previous default of 16
       was triggering ``408 Request Timeout`` from OR's edge — fewer
       parallel requests = no retries, no wasted TLS.

    Uses ``ThreadPoolExecutor`` (rather than async) so the public function
    stays sync — callers don't need to switch to ``await``.
    """
    out: list[dict[str, Any]] = []
    if not rows:
        return out

    # ── Stage 2a: cache lookup ──────────────────────────────────────
    now = time.monotonic()
    cached_health: dict[str, dict[str, Any]] = {}
    to_fetch: list[dict[str, Any]] = []
    if health_ttl_seconds > 0:
        with _pkg()._HEALTH_CACHE_LOCK:
            for r in rows:
                mid = r.get("id", "")
                entry = _pkg()._HEALTH_CACHE.get(mid)
                if entry is not None and (now - entry[0]) < health_ttl_seconds:
                    cached_health[mid] = entry[1]
                else:
                    to_fetch.append(r)
    else:
        to_fetch = list(rows)

    if cached_health:
        logger.debug(
            "OpenRouter health cache: %d hit, %d to fetch",
            len(cached_health), len(to_fetch),
        )

    # ── Stage 2b: parallel fetch with a shared connection pool ──────
    fresh_health: dict[str, dict[str, Any]] = {}
    if to_fetch:
        # max_keepalive_connections >= max_workers so threads don't fight
        # over a too-small pool and force fresh handshakes.
        limits = httpx.Limits(
            max_keepalive_connections=max_workers,
            max_connections=max_workers,
        )
        client_headers = {"Authorization": f"Bearer {api_key}"}
        with httpx.Client(
            timeout=timeout, headers=client_headers, limits=limits,
        ) as client:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        _fetch_endpoints_for_model,
                        r.get("id", ""), None, timeout, client,
                    ): r
                    for r in to_fetch
                }
                for fut in as_completed(futs):
                    row = futs[fut]
                    mid = row.get("id", "")
                    try:
                        endpoints = fut.result()
                    except Exception as exc:
                        logger.warning(
                            "Health check failed for %s (%s); excluding from results.",
                            mid, exc,
                        )
                        continue
                    fresh_health[mid] = _summarize_endpoints(endpoints)

        if health_ttl_seconds > 0 and fresh_health:
            # Re-take monotonic at write time. The original `now` was taken
            # before the (potentially multi-second) HTTP fan-out, so writing
            # with `now` would age every fresh entry by however long the
            # fetch took. Plus a parallel call could have written a newer
            # entry in the meantime — preserve the newer one.
            write_time = time.monotonic()
            with _pkg()._HEALTH_CACHE_LOCK:
                for mid, h in fresh_health.items():
                    existing = _pkg()._HEALTH_CACHE.get(mid)
                    if existing is not None and existing[0] >= write_time:
                        continue
                    _pkg()._HEALTH_CACHE[mid] = (write_time, h)
                _sweep_health_cache_locked(write_time)

    # ── Stage 2c: combine + filter by min_uptime ────────────────────
    for r in rows:
        mid = r.get("id", "")
        health = cached_health.get(mid) or fresh_health.get(mid)
        if health is None:
            continue
        uptime = health.get("best_uptime_30m")
        if uptime is None or uptime < min_uptime:
            continue
        # Don't mutate the cached catalogue entry — copy first.
        enriched = dict(r)
        enriched["health"] = health
        out.append(enriched)
    return out

def list_openrouter_models(
    refresh: bool = False,
    *,
    sort_by: str | None = None,
    name_contains: str | None = None,
    max_input_per_1m: float | None = None,
    max_output_per_1m: float | None = None,
    return_only_healthy: bool = True,
    min_uptime: float = 0.95,
    api_key: str | None = None,
    max_workers: int = 8,
    timeout: float = 30.0,
    health_ttl_seconds: float = 300.0,
) -> list[dict[str, Any]]:
    """Return OpenRouter's model catalogue with pricing, limits and (optionally) live health.

    Two-stage pipeline:

    **Stage 1 — offline catalogue filter (always)**: pulls /api/v1/models
    (no auth, free), applies ``name_contains`` / ``max_input_per_1m``.

    **Stage 2 — live health enrichment (when ``return_only_healthy=True``)**:
    fetches /api/v1/models/{id}/endpoints in parallel for every row that
    passed Stage 1, drops rows whose best-upstream uptime over the last
    30 minutes is below ``min_uptime``, and attaches a ``health`` dict to
    each survivor:

        row["health"] = {
            "endpoints": [
                {"provider_name", "status", "uptime_5m"/"30m"/"1d",
                 "latency_p50_ms", "latency_p95_ms", "throughput_p50_tps",
                 "context_length", "max_completion_tokens",
                 "pricing", "supported_parameters", "quantization",
                 "supports_implicit_caching"},
                ...
            ],
            "best_uptime_30m": 0.998,
            "best_latency_p50_ms": 234,
            "best_throughput_p50_tps": 89.5,
        }

    Stage 2 needs an OR API key (resolved from ``api_key`` arg, then
    ``OPENROUTER_API_KEY`` env, then ``LLMSettings``). The endpoints
    endpoint requires Bearer auth but does NOT bill against credits —
    "free" in the only sense that matters. With NO key configured, the
    function logs a warning and degrades gracefully to Stage-1-only
    output (matches old behaviour for backward compat).

    Args:
        refresh: Drop the in-process catalogue cache and re-fetch from /models.
        sort_by: ``"input_price"`` / ``"output_price"`` / ``"context"`` /
            ``"name"`` (Stage-1 metrics, always available); or
            ``"uptime"`` / ``"latency"`` / ``"throughput"`` (Stage-2,
            requires ``return_only_healthy=True`` — silently falls back
            to ``"name"`` ordering otherwise).
        name_contains: Substring filter on ``id`` (case-insensitive).
        max_input_per_1m: Filter to models with input price <= this USD/1M.
        max_output_per_1m: Filter to models with output price <= this USD/1M.
            Set both ``max_input_per_1m`` and ``max_output_per_1m`` to drop
            expensive models BEFORE Stage 2 fires its per-model HTTP calls
            -- the most effective way to trim health-check work.
        return_only_healthy: When True, run Stage 2 health enrichment +
            filter. Default ``True`` because routing through a degraded
            upstream is the most common silent failure mode of meta-providers.
            Set ``False`` to skip the per-model HTTP call entirely.
        min_uptime: Minimum ``best_uptime_30m`` (0.0-1.0) for a row to
            survive Stage 2. Default 0.95.
        api_key: Override env-resolved key for the /endpoints calls.
        max_workers: ThreadPoolExecutor size for Stage 2 concurrent fetch.
            Default 8. Above 16 we observed 408 timeouts from OR's edge.
        timeout: Per-request HTTP timeout in seconds for Stage 2.
        health_ttl_seconds: TTL for the in-process health cache. Default 300
            (5 min) -- short enough that "model went down" detection lags
            by no more than 5 min, long enough that repeated calls in a
            tight loop reuse cache. Pass 0 to bypass cache entirely.

    Returns:
        List of dicts. Empty on catalogue fetch failure.

    Example:
        >>> # cheapest healthy Claude variant under $1/1M input
        >>> rows = list_openrouter_models(
        ...     name_contains="claude",
        ...     max_input_per_1m=1.0,
        ...     sort_by="uptime",
        ... )
        >>> top = rows[0]
        >>> print(top["id"], top["health"]["best_uptime_30m"],
        ...       top["health"]["best_latency_p50_ms"], "ms p50")
    """
    if refresh:
        with _pkg()._MODELS_LOCK:
            _pkg()._MODELS_CATALOGUE = None
    catalogue = _fetch_models_catalogue()
    rows = list(catalogue.values())

    if name_contains:
        needle = name_contains.lower()
        rows = [r for r in rows if needle in str(r.get("id", "")).lower()]

    if max_input_per_1m is not None:
        def _input_per_1m(r: dict) -> float:
            try:
                return float((r.get("pricing") or {}).get("prompt", "0") or "0") * 1_000_000
            except (TypeError, ValueError):
                return float("inf")
        rows = [r for r in rows if _input_per_1m(r) <= max_input_per_1m]

    if max_output_per_1m is not None:
        def _output_per_1m(r: dict) -> float:
            try:
                return float((r.get("pricing") or {}).get("completion", "0") or "0") * 1_000_000
            except (TypeError, ValueError):
                return float("inf")
        rows = [r for r in rows if _output_per_1m(r) <= max_output_per_1m]

    if return_only_healthy and rows:
        key = _pkg()._resolve_or_api_key(api_key)
        if not key:
            logger.warning(
                "list_openrouter_models(return_only_healthy=True) needs an "
                "OpenRouter API key (env OPENROUTER_API_KEY or api_key=); "
                "skipping health filter -- returning Stage-1 results."
            )
        else:
            rows = _enrich_with_health(
                rows, key, min_uptime, max_workers, timeout, health_ttl_seconds,
            )

    if sort_by:
        def _key(r: dict):
            pricing = r.get("pricing") or {}
            try:
                in_price = float(pricing.get("prompt", "0") or "0")
                out_price = float(pricing.get("completion", "0") or "0")
            except (TypeError, ValueError):
                in_price = out_price = float("inf")
            ctx = r.get("context_length") or 0
            health = r.get("health") or {}
            return {
                "input_price": in_price,
                "output_price": out_price,
                "context": -int(ctx),                          # biggest context first
                "name": str(r.get("id", "")),
                # Stage-2 sorts. Missing health → sort to the end.
                "uptime":     -(health.get("best_uptime_30m") or 0.0),
                "latency":    health.get("best_latency_p50_ms")    or float("inf"),
                "throughput": -(health.get("best_throughput_p50_tps") or 0.0),
            }.get(sort_by, str(r.get("id", "")))
        rows.sort(key=_key)
    return rows

