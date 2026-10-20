"""Per-endpoint OpenRouter prices, the rate a call is actually billed at for the provider that served it.

The model-level ``pricing`` on ``/api/v1/models`` is one listing's price, not the price every route charges: for
``deepseek/deepseek-v4.1-flash`` it equals the OpenInference fp4 endpoint's rate ($0.011 output per million) while
the typical endpoints charge about $1.20 output. Cost estimates that read only the model-level figure therefore
understate a call routed to any other provider. This module reads ``/api/v1/models/{id}/endpoints`` so a caller can
price the endpoint it pinned, or see the whole spread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
_TTL_SECONDS = 3600.0
_CACHE: dict[str, tuple[float, list[EndpointPrice]]] = {}


@dataclass(frozen=True)
class EndpointPrice:
    """One provider's price for one model, USD per million tokens."""

    provider_name: str
    tag: str
    input_per_1m: float
    output_per_1m: float
    cache_read_per_1m: float | None


def _per_1m(raw: Any) -> float | None:
    """A per-token USD string as a per-million float, or None when it is missing or not numeric."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw) * 1_000_000
    except (TypeError, ValueError):
        return None


def parse_endpoint_prices(payload: dict[str, Any]) -> list[EndpointPrice]:
    """Every priced endpoint in an ``/models/{id}/endpoints`` response, in the order OpenRouter lists them.

    An endpoint whose input or output price is missing or not numeric is skipped rather than priced at zero: a zero
    would make the cheapest route look free.
    """
    data = payload.get("data") or {}
    prices: list[EndpointPrice] = []
    for ep in data.get("endpoints") or []:
        pricing = ep.get("pricing") or {}
        in_per_1m = _per_1m(pricing.get("prompt"))
        out_per_1m = _per_1m(pricing.get("completion"))
        if in_per_1m is None or out_per_1m is None:
            continue
        prices.append(
            EndpointPrice(
                provider_name=str(ep.get("provider_name") or ""),
                tag=str(ep.get("tag") or ""),
                input_per_1m=in_per_1m,
                output_per_1m=out_per_1m,
                cache_read_per_1m=_per_1m(pricing.get("input_cache_read")),
            )
        )
    return prices


def endpoint_prices(model: str, *, timeout: float = 10.0) -> list[EndpointPrice]:
    """Every priced endpoint for ``model``, cached for an hour. Empty when the fetch fails; a failure is not cached."""
    now = time.monotonic()
    cached = _CACHE.get(model)
    if cached is not None and now - cached[0] < _TTL_SECONDS:
        return cached[1]
    try:
        resp = httpx.get(_ENDPOINTS_URL.format(model=model), timeout=timeout)
        resp.raise_for_status()
        prices = parse_endpoint_prices(resp.json())
    except Exception:
        return []
    _CACHE[model] = (now, prices)
    return prices


def endpoint_price(model: str, tag: str, *, timeout: float = 10.0) -> EndpointPrice | None:
    """The price of the endpoint with this ``tag`` (e.g. ``"open-inference/fp4"``), or None when it is not listed."""
    for price in endpoint_prices(model, timeout=timeout):
        if price.tag == tag:
            return price
    return None
