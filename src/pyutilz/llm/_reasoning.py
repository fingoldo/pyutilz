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

from pyutilz.llm.base import PerCallAttr

__all__ = ["ReasoningCaptureMixin", "collect", "estimate_tokens", "from_message", "joined"]


class ReasoningCaptureMixin:
    """Holds this call's reasoning fragments and serves them joined.

    Both providers need the same two members and the scanner is right that two copies drift; the
    storage is fragments rather than a growing string because appending is O(1) and concatenating is
    not -- see ``collect``.
    """

    #: Fragments of THIS call's reasoning. PerCallAttr, so a `generate_batch` sibling cannot read
    #: another call's thoughts, and assigned rather than mutated for the reason `collect` gives.
    _reasoning_fragments: PerCallAttr = PerCallAttr(list)

    @property
    def last_reasoning_text(self) -> str | None:
        """The reasoning of the most recent call, or None when the model emitted none."""
        return joined(self._reasoning_fragments)


def collect(fragments: list[str], fragment: Any) -> list[str]:
    """Keep one streamed reasoning fragment, in O(1), for joining once at the end.

    Appends rather than concatenating, because concatenating is quadratic and this runs on every
    delta of a stream that can carry tens of thousands of them before the first content byte.
    Measured 2026-09-13, z-ai/glm-5.3-flash on the enrichment prompt: building the text by
    `text = text + fragment` dropped throughput from ~300 to ~60 characters per second -- the copying
    blocked the event loop while the model was waiting, and a call that answers in 538 s was still
    unfinished at 2,999 s. The model was not slow; we were.

    Returns the list so the caller can ASSIGN it back. The providers hold it in a ``PerCallAttr``,
    whose ``__get__`` builds a fresh default on every read when nothing has been stored yet, so an
    in-place append to what that read returned would vanish -- the same reason every other PerCallAttr
    beside it (``last_tool_calls``, ``last_citations``) is assigned rather than mutated.

    Anything that is not a non-empty string is ignored, so an upstream sending the key with nothing
    in it cannot manufacture an empty record. Nothing is capped: a truncated record would silently
    become a sample.
    """
    if isinstance(fragment, str) and fragment:
        fragments.append(fragment)
    return fragments


def joined(fragments: list[str]) -> str | None:
    """The whole reasoning, or None when the model emitted none."""
    return "".join(fragments) if fragments else None


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
