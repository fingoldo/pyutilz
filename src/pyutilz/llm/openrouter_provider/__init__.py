"""OpenRouter LLM provider — meta-provider exposing 200+ models behind one API.

This package was carved out of the former single-file
``openrouter_provider.py`` (which had grown past the project's 1000-LOC
module limit). The public import surface is preserved 100% — every name
that used to be importable from ``pyutilz.llm.openrouter_provider`` is
re-exported here (including the private helpers and the module-level
mutable caches that the test-suite patches), and ``OpenRouterProvider``
still resolves via the factory's lazy ``(module_path, class_name)`` lookup.

Design note — why the module-level caches live HERE and not in the
submodules: ``_MODELS_CATALOGUE`` and ``_HEALTH_CACHE`` are process-wide
singletons, patched/mutated by callers (and tests) through this module's
namespace (``openrouter_provider._MODELS_CATALOGUE = ...``). To keep that
contract 100% intact after the split, this facade owns the single copy of
those globals plus the ``get_llm_settings`` / ``httpx`` names that get
monkeypatched, and every submodule resolves them LATE (``_pkg.X`` at call
time) so a patch on this module is seen everywhere. Submodules:

  * ``_catalogue`` — ``/api/v1/models`` cache + per-token pricing / limits.
  * ``_health``    — live ``/endpoints`` health cache + concurrent
    enrichment + ``list_openrouter_models`` / ``clear_openrouter_caches``.
  * ``_provider``  — the ``OpenRouterProvider`` class itself.
"""

from __future__ import annotations

# ── stdlib / third-party names that were importable from the old module ──
# Re-exported verbatim to preserve the historical public surface AND to
# serve as the single late-binding target for monkeypatching (``httpx``,
# ``get_llm_settings``).
import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from pyutilz.llm.config import get_llm_settings
from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger("pyutilz.llm.openrouter_provider")

# ── process-wide mutable caches (owned here; see module docstring) ────────
# Catalogue side.
_MODELS_CATALOGUE: dict[str, dict[str, Any]] | None = None
_MODELS_LOCK = threading.Lock()
_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Health side.
_HEALTH_CACHE_MAX_SIZE = 1024
_ENDPOINTS_URL_TEMPLATE = "https://openrouter.ai/api/v1/models/{model}/endpoints"
_HEALTH_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_HEALTH_CACHE_LOCK = threading.Lock()

# ── carved-out implementation ─────────────────────────────────────────────
# Imported AFTER the globals above exist, because the submodules bind
# ``import pyutilz.llm.openrouter_provider as _pkg`` at import time and would
# otherwise race the globals (they only READ ``_pkg.X`` at call time, so the
# partially-initialised module object is fine as long as the names exist).
from pyutilz.llm.openrouter_provider._catalogue import (
    _fetch_models_catalogue,
    _per_token_cost_pair,
    _resolve_model_limits,
)
from pyutilz.llm.openrouter_provider._health import (
    _enrich_with_health,
    _fetch_endpoints_for_model,
    _normalize_uptime,
    _resolve_or_api_key,
    _summarize_endpoints,
    _sweep_health_cache_locked,
    clear_openrouter_caches,
    list_openrouter_models,
)
from pyutilz.llm.openrouter_provider._provider import OpenRouterProvider

__all__ = [
    "OpenRouterProvider",
    "list_openrouter_models",
    "clear_openrouter_caches",
]
