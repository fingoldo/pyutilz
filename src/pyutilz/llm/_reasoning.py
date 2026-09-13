"""Reading the thinking a call was billed for, in the three shapes upstreams deliver it.

Reasoning is output: upstreams charge for it as output tokens, and OpenRouter counts them INSIDE
``completion_tokens``, so a caller holding only that sum cannot say what it paid for thinking rather
than for an answer. One ``z-ai/glm-5.3-flash`` enrichment call on 2026-09-13 spent 39,641 of its
65,577 output tokens on 118,004 characters of reasoning, and every character was discarded because
each extraction site read the answer and walked past the thinking beside it.

The shapes differ by transport and not by meaning, which is why they live together here rather than
as three private copies: a buffered OpenAI-compatible reply carries ``message.reasoning`` whole, a
stream delivers it as thousands of ``delta.reasoning`` fragments, and the Claude Code SDK returns
ThinkingBlocks beside the text ones.
"""

from __future__ import annotations

from typing import Any

__all__ = ["appended", "from_message", "estimate_tokens"]


def appended(existing: str | None, fragment: Any) -> str | None:
    """Add one streamed reasoning fragment to what has arrived so far.

    Returns ``existing`` unchanged for anything that is not a non-empty string, so an upstream that
    sends the key with nothing in it cannot manufacture an empty record. Nothing is capped: a model
    can emit tens of thousands of fragments before its first content byte, and a truncated record
    would silently become a sample.
    """
    if not isinstance(fragment, str) or not fragment:
        return existing
    return (existing or "") + fragment


def from_message(message: dict[str, Any]) -> str | None:
    """The reasoning of a buffered OpenAI-compatible reply, or ``None`` when it carries none.

    Call this BEFORE any check that raises on the reply -- a truncation, say. A response cut off
    mid-thought is precisely the one whose reasoning explains where the budget went, and raising
    first threw it away.
    """
    reasoning = message.get("reasoning")
    return reasoning if isinstance(reasoning, str) and reasoning else None


def estimate_tokens(text: str | None) -> int:
    """How many tokens a piece of reasoning is, for upstreams that bill it without counting it.

    DERIVED, not reported. Anthropic bills thinking inside ``output_tokens`` and publishes no
    separate figure, so measuring the text is the only way to say how much of a bill bought
    thinking. The tokeniser approximates Claude's own -- it is the same one the Claude Code provider
    already falls back to for its output count -- so this is an estimate of a real quantity rather
    than an exact figure. Zero, for a call that did no thinking, is exact.
    """
    if not text:
        return 0
    from pyutilz.llm.token_counter import count_tokens

    return int(count_tokens(text))
