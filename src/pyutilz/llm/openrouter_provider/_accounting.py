"""OpenRouter session-cost estimate by the model that SERVED each call, with cache writes at their own rate.

Split out of ``_provider.py`` for the module line budget; mixed into ``OpenRouterProvider``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from pyutilz.llm.openrouter_provider._catalogue import (
    _cache_read_cost_per_1m_or_none,
    _cache_write_cost_per_1m_or_none,
    _fetch_models_catalogue,
    _per_token_cost_pair_or_none,
)
from pyutilz.llm.openrouter_provider._request import catalogue_id


class OpenRouterAccountingMixin:
    """Per-served-model token tallies priced at each model's own catalogue rates."""

    model_name: str
    _usage_by_model: Dict[str, List[int]]
    total_prompt_tokens: int
    total_cache_hit_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    total_cache_write_tokens: int

    if TYPE_CHECKING:  # supplied by OpenRouterProvider / OpenAICompatibleProvider

        def _served_models(self) -> List[str]:
            """The requested model plus its fallbacks (typing stub)."""
            ...

        def _compute_billed_output(self, completion_tokens: int, reasoning_tokens: int) -> int:
            """Output tokens as billed, reasoning included or not (typing stub)."""
            ...

    def _pricing_model(self, served: str) -> str:
        """The catalogue id to price a served model at: itself when listed, else the requested model.

        OpenRouter may answer with a dated snapshot id (``anthropic/claude-x-20260101``) that the catalogue does not list;
        pricing that as "unknown" would null the whole session estimate over a naming detail. A served id that is one of
        the fallbacks, or listed, is priced as itself.
        """
        if served in self._served_models() or _fetch_models_catalogue().get(served):
            return served
        return catalogue_id(self.model_name)

    def _estimate_by_served_model(self) -> Optional[Tuple[float, float]]:
        """(input, output) USD estimate, each served model at its own rates and cache writes at the write rate; None when
        any model with usage has no catalogue pricing.

        Tokens the per-model tally does not cover (counters set directly, say) are attributed to the requested model, so
        the totals always reconcile with ``total_prompt_tokens`` / ``total_completion_tokens``. Cache writes are part of
        ``prompt_tokens`` and disjoint from cache hits, so they are carved out of the miss count and priced at
        ``pricing.input_cache_write`` (the plain input rate when the catalogue has none): they cost 1.25x input on
        Anthropic and used to be estimated at 1x (OR-6).
        """
        # getattr: an instance built with __new__ (tests, pickles from before the tally existed) has none yet.
        buckets = {m: list(t) for m, t in (getattr(self, "_usage_by_model", None) or {}).items()}
        totals = (self.total_prompt_tokens, self.total_cache_hit_tokens, self.total_completion_tokens, self.total_reasoning_tokens, self.total_cache_write_tokens)
        residual = [total - sum(t[i] for t in buckets.values()) for i, total in enumerate(totals)]
        if any(residual):
            own = buckets.setdefault(catalogue_id(self.model_name), [0, 0, 0, 0, 0])
            for i, r in enumerate(residual):
                own[i] += max(0, r)
        if not buckets:
            buckets[catalogue_id(self.model_name)] = [0, 0, 0, 0, 0]
        input_cost = output_cost = 0.0
        for model, (prompt, hit, completion, reasoning, write) in buckets.items():
            pair = _per_token_cost_pair_or_none(model)
            if pair is None:
                return None
            in_rate, out_rate = pair
            hit_rate = _cache_read_cost_per_1m_or_none(model)
            write_rate = _cache_write_cost_per_1m_or_none(model)
            miss = max(0, prompt - hit - write)
            input_cost += (miss * in_rate + hit * (in_rate if hit_rate is None else hit_rate) + write * (in_rate if write_rate is None else write_rate)) / 1_000_000
            output_cost += self._compute_billed_output(completion, reasoning) * out_rate / 1_000_000
        return input_cost, output_cost
