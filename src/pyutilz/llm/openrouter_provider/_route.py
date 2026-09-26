"""``OpenRouterProvider.route_fingerprint()``: one stable digest of everything that decides WHICH upstream answers and how.

Downstream response caches and benchmark ledgers key on it (glossum ``response_cache.route_of``, llm_bench
``provider_identity``), so two providers that could be served by different upstreams, quantisations, plugins or
request extras must never share a key, while two identically configured ones always do.

The contract is enforced by a meta test: every ``OpenRouterProvider.__init__`` keyword is either in
:data:`_ROUTE_FINGERPRINT_ATTRS` or in :data:`_ROUTE_FINGERPRINT_EXCLUDED` with a reason. A new routing knob
added to the constructor therefore fails that test until someone decides which side it belongs on, instead of
silently pooling cache entries across routes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Constructor keywords that change the route or the request body the upstream sees. ``model`` is read live from
#: ``model_name`` at fingerprint time, the rest as passed to ``__init__``.
_ROUTE_FINGERPRINT_ATTRS: tuple[str, ...] = (
    "model",
    "provider_order",
    "provider_ignore",
    "provider_only",
    "provider_sort",
    "provider_allow_fallbacks",
    "provider_require_parameters",
    "provider_data_collection",
    "provider_zdr",
    "provider_quantizations",
    "provider_max_price",
    "provider_preferred_min_throughput",
    "provider_preferred_max_latency",
    "models_fallback",
    "enable_web_search",
    "web_search_engine",
    "web_search_max_results",
    "pdf_engine",
    "transforms",
    "anthropic_top_level_cache",
    "system_cache_control",
    "extra_body",
)

#: Constructor keywords that cannot change what answers or what it is asked, each with the reason.
_ROUTE_FINGERPRINT_EXCLUDED: dict[str, str] = {
    "api_key": "credentials, not routing; must never enter a cache key",  # pragma: allowlist secret -- a reason string, not a key
    "max_concurrent": "client-side concurrency only",
    "app_name": "attribution header (X-Title) for OpenRouter's rankings only",
    "site_url": "attribution header (HTTP-Referer) only",
    "retry_routing_404": "client-side retry policy",
    "routing_404_max_attempts": "client-side retry policy",
    "routing_404_pause_sec": "client-side retry policy",
}

#: Keywords that accept a mapping OR a tuple of ``(key, value)`` pairs: both spellings must digest identically.
_MAPPING_KWARGS = frozenset({"provider_max_price", "provider_preferred_min_throughput", "provider_preferred_max_latency", "extra_body"})


def route_plain(value: Any, *, mapping: bool = False) -> Any:
    """``value`` as JSON-ready plain data: tuples and lists both become lists, mappings (or pair tuples when
    ``mapping``) become dicts, recursively, so ``("a", "b")`` and ``["a", "b"]`` fingerprint the same."""
    if mapping and value is not None and not isinstance(value, (int, float, str)):
        value = dict(value)
    if isinstance(value, Mapping):
        return {str(k): route_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [route_plain(v) for v in value]
    return value


class OpenRouterRouteMixin:
    """Supplies the OpenRouter-specific route payload; ``LLMProvider.route_fingerprint`` digests it."""

    model_name: str
    _route_kwargs: dict[str, Any]

    def _route_payload(self) -> dict[str, Any]:
        """Every routing input, normalised; the model is read live so a reassigned ``model_name`` is honoured."""
        kwargs = getattr(self, "_route_kwargs", None) or {}
        payload: dict[str, Any] = {"provider": "openrouter", "model": self.model_name}
        for name in _ROUTE_FINGERPRINT_ATTRS:
            if name == "model":
                continue
            payload[name] = route_plain(kwargs.get(name), mapping=name in _MAPPING_KWARGS)
        return payload
