"""OpenRouter request shaping: the ``provider`` routing block, plugins, transforms, ``require_parameters``, and prompt caching.

Split out of ``_provider.py`` for the module line budget. Every knob here is documented at
https://openrouter.ai/docs/guides/routing/provider-selection (routing), .../plugins/web-search, .../multimodal/pdfs,
and .../best-practices/prompt-caching; the field names are the wire names, verified 2026-09-26.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional, Tuple

from pyutilz.llm._openai_compat_http import merge_extra_body

logger = logging.getLogger("pyutilz.llm.openrouter_provider")

#: Routing suffixes OpenRouter accepts on a model slug that are not catalogue ids of their own (``:free`` is one, so it
#: stays). Catalogue lookups (pricing, limits, supported parameters) use the slug without them.
_ROUTING_SUFFIXES = (":online", ":nitro", ":floor")

#: Body keys that are not model parameters, so ``require_parameters`` never needs an endpoint to "support" them.
_NON_PARAMETER_KEYS = frozenset(
    {"model", "models", "messages", "stream", "stream_options", "provider", "plugins", "transforms", "cache_control", "usage", "user", "route"}
)


def catalogue_id(model: str) -> str:
    """``model`` without an OpenRouter routing suffix (``openai/gpt-5:online`` -> ``openai/gpt-5``)."""
    for suffix in _ROUTING_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def as_plain(value: Any) -> Any:
    """A mapping or a tuple of ``(key, value)`` pairs as a plain dict; a number (or None) unchanged."""
    if value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return dict(value)


def build_provider_field(
    *,
    order: Optional[Tuple[str, ...]],
    ignore: Optional[Tuple[str, ...]],
    only: Optional[Tuple[str, ...]],
    sort: Optional[str],
    allow_fallbacks: bool,
    data_collection: Optional[str],
    zdr: Optional[bool],
    quantizations: Optional[Tuple[str, ...]],
    max_price: Any,
    preferred_min_throughput: Any,
    preferred_max_latency: Any,
    require_parameters: Optional[bool],
) -> dict[str, Any]:
    """The ``provider`` object; only fields the caller set are sent, so a default request stays minimal.

    ``require_parameters`` is sent here only when it is explicitly True; the automatic mode (None) is decided per request
    by :func:`auto_require_parameters`, which needs the whole body.
    """
    field: dict[str, Any] = {}
    if order:
        field["order"] = list(order)
    if only:
        field["only"] = list(only)
    if ignore:
        field["ignore"] = list(ignore)
    if sort:
        field["sort"] = sort
    # OR's default is true, so only a disabled fallback is sent.
    if not allow_fallbacks:
        field["allow_fallbacks"] = False
    if require_parameters is True:
        field["require_parameters"] = True
    if data_collection is not None:
        if data_collection not in ("allow", "deny"):
            raise ValueError(f"provider_data_collection must be 'allow' or 'deny', got {data_collection!r}")
        field["data_collection"] = data_collection
    if zdr is not None:
        field["zdr"] = bool(zdr)
    if quantizations:
        field["quantizations"] = list(quantizations)
    if max_price is not None:
        field["max_price"] = as_plain(max_price)
    if preferred_min_throughput is not None:
        field["preferred_min_throughput"] = as_plain(preferred_min_throughput)
    if preferred_max_latency is not None:
        field["preferred_max_latency"] = as_plain(preferred_max_latency)
    return field


def build_plugins(enable_web_search: bool, web_search_engine: Optional[str], web_search_max_results: Optional[int], pdf_engine: Optional[str]) -> list:
    """The ``plugins`` array: the web plugin with its options, and the file-parser plugin for PDFs."""
    plugins: list = []
    if enable_web_search:
        web: dict[str, Any] = {"id": "web"}
        if web_search_engine:
            web["engine"] = web_search_engine
        if web_search_max_results is not None:
            web["max_results"] = int(web_search_max_results)
        plugins.append(web)
    if pdf_engine:
        plugins.append({"id": "file-parser", "pdf": {"engine": pdf_engine}})
    return plugins


def _parameter_names(body: Mapping[str, Any]) -> set:
    """The catalogue ``supported_parameters`` names the body's parameters need."""
    names = set()
    for key, value in body.items():
        if key in _NON_PARAMETER_KEYS:
            continue
        if key == "response_format" and isinstance(value, Mapping) and value.get("type") == "json_schema":
            names.add("structured_outputs")
        else:
            names.add(key)
    return names


def _asks_for_reasoning(body: Mapping[str, Any]) -> bool:
    """True when the body asks the model to REASON (an effort or a budget), not when it switches reasoning off."""
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, Mapping) or reasoning.get("enabled") is False or reasoning.get("exclude"):
        return False
    asks_effort = reasoning.get("effort") not in (None, "none", "minimal")
    asks_budget = bool(reasoning.get("max_tokens"))
    explicitly_on = reasoning.get("enabled") is True
    return asks_effort or asks_budget or explicitly_on


def auto_require_parameters(body: Mapping[str, Any], catalogue_entry: Optional[Mapping[str, Any]]) -> bool:
    """Whether ``provider.require_parameters`` should be sent for this body when the caller left it unset.

    True when the body carries a parameter whose SILENT drop changes the answer (``response_format``, or a request to
    reason) AND the catalogue says the model supports every parameter in the body. The second half is what keeps the
    default safe: ``require_parameters`` filters on EVERY parameter, so an o-series model that lists no ``temperature``
    would otherwise have no endpoint left and the call would 404. An unknown model gets no requirement for the same reason.
    """
    if not ("response_format" in body or _asks_for_reasoning(body)):
        return False
    if not catalogue_entry:
        return False
    supported = catalogue_entry.get("supported_parameters")
    if not isinstance(supported, list):
        return False
    have = {str(p).lower() for p in supported}
    missing = _parameter_names(body) - have
    if missing:
        logger.debug("not requiring parameters for %s: catalogue lacks %s", body.get("model"), sorted(missing))
        return False
    return True


def with_require_parameters(body: dict[str, Any]) -> dict[str, Any]:
    """``body`` with ``provider.require_parameters: true`` added to whatever routing block it has."""
    return merge_extra_body(body, {"provider": {"require_parameters": True}})


def system_with_cache_control(messages: list) -> list:
    """The messages with the system prompt sent as one text part carrying ``cache_control: {"type": "ephemeral"}``.

    A per-block breakpoint (the caching doc's form for Gemini and for Anthropic outside automatic caching): the system
    prompt is the stable prefix worth caching. A message whose content is already a list is left alone.
    """
    out = []
    for message in messages:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            message = {**message, "content": [{"type": "text", "text": message["content"], "cache_control": {"type": "ephemeral"}}]}
        out.append(message)
    return out


def url_citations(annotations: Any) -> list:
    """The ``url_citation`` payloads of an ``annotations`` list (message or stream delta)."""
    if not isinstance(annotations, list):
        return []
    return [ann.get("url_citation") or {} for ann in annotations if isinstance(ann, dict) and ann.get("type") == "url_citation"]


class OpenRouterRequestMixin:
    """The OpenRouter request-body overrides; mixed in ahead of ``OpenAICompatibleProvider`` by ``OpenRouterProvider``."""

    model_name: str
    last_web_search_citations: Any
    _provider_field: dict
    _models_fallback: Optional[Tuple[str, ...]]
    _plugins: list
    _transforms: Optional[Tuple[str, ...]]
    _anthropic_top_level_cache: bool
    _system_cache_control: bool
    _provider_require_parameters: Optional[bool]
    _default_extra_body: Optional[dict]

    def _catalogue_entry(self) -> Optional[Mapping[str, Any]]:  # pragma: no cover - supplied by OpenRouterProvider
        """The model's catalogue record; OpenRouterProvider supplies it."""
        raise NotImplementedError

    def _extra_request_body(self, model: str) -> dict[str, Any]:
        """The OpenRouter fields of every request: ``provider``, ``models``, ``plugins``, ``transforms``, top-level
        ``cache_control``, then the constructor's ``extra_body`` (per-call ``extra_body`` is merged after all of these)."""
        body: dict[str, Any] = {}
        if self._provider_field:
            body["provider"] = dict(self._provider_field)
        if self._models_fallback:
            body["models"] = list(self._models_fallback)
        if self._plugins:
            body["plugins"] = [dict(p) for p in self._plugins]
        if self._transforms:
            # Opt-in only: middle-out DROPS the middle of an over-long prompt, which is lossy by design.
            body["transforms"] = list(self._transforms)
        if self._anthropic_top_level_cache:
            body["cache_control"] = {"type": "ephemeral"}
        if self._default_extra_body:
            body = merge_extra_body(body, self._default_extra_body)
        return body

    def _finalize_request_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Add ``provider.require_parameters`` in automatic mode (``provider_require_parameters=None``) when warranted."""
        provider = body.get("provider")
        if self._provider_require_parameters is not None or (isinstance(provider, Mapping) and "require_parameters" in provider):
            return body
        if auto_require_parameters(body, self._catalogue_entry()):
            return with_require_parameters(body)
        return body

    def _build_messages(self, prompt: str, system: Optional[str] = None, images: Optional[list] = None) -> list:
        """The chat messages, with the system prompt as a cache breakpoint when ``system_cache_control`` is on."""
        messages = super()._build_messages(prompt, system, images) if images else super()._build_messages(prompt, system)  # type: ignore[misc]
        return system_with_cache_control(messages) if self._system_cache_control else messages

    def _apply_stream_chunk(self, chunk: dict[str, Any], tool_call_fragments: dict) -> Optional[str]:
        """The base fold, plus web-search citations, which arrive on ``delta.annotations`` when streaming."""
        content = super()._apply_stream_chunk(chunk, tool_call_fragments)  # type: ignore[misc]
        choices = chunk.get("choices") or []
        if choices and isinstance(choices[0], dict):
            found = url_citations((choices[0].get("delta") or {}).get("annotations"))
            if found:
                self.last_web_search_citations = [*(self.last_web_search_citations or []), *found]
        return content  # type: ignore[no-any-return]
