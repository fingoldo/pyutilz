"""The provider-independent pricing record, and the one function that prices a single request with it.

Carved out of ``openai_compat.py`` (which re-exports ``Pricing``) to keep that module under the 1,000-line budget; the
providers and tests that import ``Pricing`` from ``openai_compat`` keep working.

Design (2026-10-03, research in ``audits/implemented/2026-09-26/provider_pricing_tiers.md``):

* **Cache writes** are a rate on :class:`Pricing` (``cache_write``, USD per 1M), like ``cache_hit``. Every vendor that
  charges for writes publishes them as a multiple of the uncached input rate (Anthropic 1.25x for the 5-minute TTL and
  2x for 1 hour, OpenAI 1.25x on GPT-5.6 and later), so one optional rate per model is enough; Anthropic's second
  (1-hour) rate is :data:`CACHE_WRITE_1H_MULT` applied by its provider, the only one that reports that split.
* **Long-context tiers** are a :class:`LongContextTier` attached to ``Pricing.long_context`` (default None, so every
  positional ``Pricing(...)`` call and every 2/3/4-field use keeps working). A tier is a prompt-size threshold and four
  multipliers on the short-context rates. Every vendor that has one prices the WHOLE request at the long rates once
  that request's own prompt crosses the threshold: xAI says so in words ("billed at the higher rate for all tokens in
  the request") and Gemini's Pro rows are documented the same way. OpenAI's page lists a short and a long column per
  model (">272K input tokens") without saying whether the long rate covers the whole request or only the excess; the
  whole-request reading is the one implemented, and it is UNVERIFIED for OpenAI.
* **Where the tier is decided.** It depends on ONE request's prompt size, so a session total cannot price it: two
  150K calls and one 300K call have the same 300K total and different bills. Below the threshold pricing is linear in
  the token counts, so pricing the pooled totals at the short rates equals summing per-call costs exactly; the tier is
  the only non-linear part. The providers therefore keep pricing the totals at the short rates (which also keeps every
  caller that sets the ``total_*`` counters directly working) and accumulate, per call, only what the tier adds:
  :func:`long_context_surcharge`. The session cost is then exactly the sum of per-call :func:`price_call` results.
* ``inclusive`` on the tier records the vendor's boundary: xAI bills a prompt that REACHES 200K at the long rate,
  Gemini and OpenAI one that EXCEEDS 200K / 272K.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

#: Anthropic's 1-hour cache-write price as a multiple of input (the 5-minute one, 1.25x, is ``Pricing.cache_write``).
CACHE_WRITE_1H_MULT = 2.0
#: The 5-minute cache-write multiple, Anthropic and OpenAI (GPT-5.6 and later) alike.
CACHE_WRITE_5M_MULT = 1.25


class LongContextTier(NamedTuple):
    """The long-context rates of one model, as multipliers on its short-context :class:`Pricing` rates.

    ``threshold_tokens`` is the prompt size the tier starts at: a request is long when its prompt EXCEEDS it, or, with
    ``inclusive``, when it REACHES it. Once it is long, every token of the request is billed at the multiplied rates.
    """

    threshold_tokens: int
    input_mult: float
    output_mult: float
    cached_input_mult: float
    cache_write_mult: float
    inclusive: bool = False

    def applies(self, prompt_tokens: int) -> bool:
        """True when a request with ``prompt_tokens`` prompt tokens is billed at this tier."""
        return prompt_tokens >= self.threshold_tokens if self.inclusive else prompt_tokens > self.threshold_tokens


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

    ``cache_write`` is the price of WRITING a prompt-cache entry (Anthropic-family routes bill it at
    1.25x input for the 5-minute TTL). None means the provider publishes no separate rate, and
    ``_cache_write_cost_per_1m`` falls back to the uncached input rate.

    ``long_context`` is the model's long-context tier, None when it has none (see the module docstring).
    """

    input: float
    output: float
    cache_hit: Optional[float] = None
    cache_write: Optional[float] = None
    long_context: Optional[LongContextTier] = None


def _split(pricing: Pricing, prompt_tokens: int, cache_read: int, cache_write: int) -> Tuple[int, int, int, float, float]:
    """``(miss, read, write, read_rate, write_rate)``: cache reads and writes are part of ``prompt_tokens``, so they are
    carved out of it (clamped, so a counter larger than the prompt never yields negative miss tokens)."""
    read = max(0, min(cache_read, prompt_tokens))
    write = max(0, min(cache_write, prompt_tokens - read))
    read_rate = pricing.input if pricing.cache_hit is None else pricing.cache_hit
    write_rate = pricing.input if pricing.cache_write is None else pricing.cache_write
    return prompt_tokens - read - write, read, write, read_rate, write_rate


def long_context_surcharge(pricing: Pricing, prompt_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> Tuple[float, float]:
    """``(input_usd, output_usd)`` the long-context tier adds to ONE request on top of its short-context price.

    ``(0.0, 0.0)`` when ``pricing`` has no tier or the request's prompt does not reach it. ``output_tokens`` are the
    BILLED output tokens (reasoning included wherever the provider bills it on top of the completion).
    """
    tier = pricing.long_context
    if tier is None or not tier.applies(prompt_tokens):
        return 0.0, 0.0
    miss, read, write, read_rate, write_rate = _split(pricing, prompt_tokens, cache_read, cache_write)
    extra_in = (miss * pricing.input * (tier.input_mult - 1) + read * read_rate * (tier.cached_input_mult - 1)) / 1_000_000
    if write:
        extra_in += write * write_rate * (tier.cache_write_mult - 1) / 1_000_000
    return extra_in, output_tokens * pricing.output * (tier.output_mult - 1) / 1_000_000


def price_call(
    pricing: Pricing,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
    reasoning: int = 0,
) -> Tuple[float, float]:
    """``(input_usd, output_usd)`` for ONE request, long-context tier included.

    ``cache_read`` and ``cache_write`` are the parts of ``prompt_tokens`` read from / written to the prompt cache, each
    billed at its own rate (the input rate when the record has none). ``reasoning`` is billed at the output rate on TOP
    of ``completion_tokens``: pass 0 for a provider whose completion count already includes it. When the request's
    prompt crosses ``pricing.long_context``, every token of it is billed at the tier's multiplied rates.
    """
    miss, read, write, read_rate, write_rate = _split(pricing, prompt_tokens, cache_read, cache_write)
    output = completion_tokens + reasoning
    input_usd = (miss * pricing.input + read * read_rate + write * write_rate) / 1_000_000
    output_usd = output * pricing.output / 1_000_000
    extra_in, extra_out = long_context_surcharge(pricing, prompt_tokens, output, cache_read, cache_write)
    return input_usd + extra_in, output_usd + extra_out
