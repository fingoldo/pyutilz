"""The provider-independent pricing record every OpenAI-compatible provider returns.

Carved out of ``openai_compat.py`` (which re-exports it) to keep that module under the 1,000-line
budget; the providers and tests that import ``Pricing`` from ``openai_compat`` keep working.
"""

from __future__ import annotations

from typing import NamedTuple, Optional


class Pricing(NamedTuple):
    """One provider-independent pricing record, USD per 1M tokens.

    The ONE tuple contract every provider's ``_resolve_pricing`` returns. It exists because the
    same private method name used to carry two different shapes in sibling providers -- xAI's
    ``(input, output)`` and DeepSeek's ``(input, cache_hit, output)`` -- so the accessors indexed
    ``[1]`` and ``[2]`` for the same quantity. Nothing raises when those shapes get copied across:
    both positions hold a float, and the only symptom is a silently wrong USD figure. Named fields
    make the mix-up unrepresentable.

    ``cache_hit`` is None when the provider publishes no cached-input rate; the base
    ``_cache_hit_cost_per_1m`` then falls back to the uncached input rate.
    """

    input: float
    output: float
    cache_hit: Optional[float] = None
