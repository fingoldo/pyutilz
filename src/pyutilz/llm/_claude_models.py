"""Per-model facts for Claude, shared by the Anthropic API provider and the Claude Code provider.

A module of its own, with no SDK import, because both providers need the same numbers and the Claude
Code provider must stay importable without the ``anthropic`` package. One table instead of the three
the API provider used to keep (pricing, output limit, and a context window hard-coded to 200K for every
model), which drifted apart: the pricing table named IDs that do not exist (``claude-haiku-3-20240307``
for the real ``claude-3-haiku-20240307``), so Haiku 3 was billed at Sonnet rates, and no current model
was listed at all.

Sources, fetched 2026-09-26:
    https://platform.claude.com/docs/en/about-claude/models/overview   (IDs, context, max output, thinking mode)
    https://platform.claude.com/docs/en/about-claude/pricing            (base, cache-read multipliers)
    https://platform.claude.com/docs/en/build-with-claude/effort        (which models take ``output_config.effort``)
    https://platform.claude.com/docs/en/build-with-claude/structured-outputs  (which take ``output_config.format``)
    https://platform.claude.com/docs/en/models/{opus-4-5,sonnet-4-5,sonnet-4-6,opus-5,fable-5}/overview
The retired Claude 3.x rows are the last published figures, kept so historical logs still price.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaudeModelSpec:
    """What the API documents for one Claude model."""

    input_per_1m: float
    output_per_1m: float
    max_output: int
    context_window: int
    # Cache hits are 0.1x base input on most models, 0.05x on Opus 5.5 and 0.025x on Fable/Mythos 5.1.
    cache_read_multiplier: float = 0.10
    # Adaptive thinking (``thinking: {"type": "adaptive"}`` + ``output_config.effort``). Models from 4.7 on
    # reject the manual ``{"type": "enabled", "budget_tokens": N}`` form with a 400; 4.6 accepts it but it is
    # deprecated; 4.5 and earlier accept ONLY the manual form and 400 on ``adaptive``.
    adaptive_thinking: bool = True
    supports_effort: bool = True
    supports_structured_output: bool = True


_1M = 1_000_000
_200K = 200_000

# Keys are the dateless API IDs (from 4.6 on the dateless ID IS the pinned snapshot) or, for older models,
# the alias a dated snapshot extends: ``claude-opus-4-5`` matches ``claude-opus-4-5-20251101``.
CLAUDE_MODELS: dict[str, ClaudeModelSpec] = {
    "claude-fable-5-1": ClaudeModelSpec(10.0, 50.0, 128_000, _1M, cache_read_multiplier=0.025),
    "claude-mythos-5-1": ClaudeModelSpec(10.0, 50.0, 128_000, _1M, cache_read_multiplier=0.025),
    "claude-fable-5": ClaudeModelSpec(10.0, 50.0, 128_000, _1M),
    "claude-mythos-5": ClaudeModelSpec(10.0, 50.0, 128_000, _1M),
    "claude-opus-5-5": ClaudeModelSpec(4.0, 20.0, 128_000, _1M, cache_read_multiplier=0.05),
    "claude-opus-5": ClaudeModelSpec(5.0, 25.0, 128_000, _1M),
    "claude-opus-4-8": ClaudeModelSpec(5.0, 25.0, 128_000, _1M),
    "claude-opus-4-7": ClaudeModelSpec(5.0, 25.0, 128_000, _1M),
    "claude-opus-4-6": ClaudeModelSpec(5.0, 25.0, 128_000, _1M),
    "claude-sonnet-5": ClaudeModelSpec(2.0, 10.0, 128_000, _1M),
    "claude-sonnet-4-6": ClaudeModelSpec(3.0, 15.0, 128_000, _1M),
    # Extended-thinking-only generation. Opus 4.5 is the one of them that also takes ``effort``.
    "claude-opus-4-5": ClaudeModelSpec(5.0, 25.0, 64_000, _200K, adaptive_thinking=False),
    "claude-sonnet-4-5": ClaudeModelSpec(3.0, 15.0, 64_000, _200K, adaptive_thinking=False, supports_effort=False),
    "claude-haiku-4-5": ClaudeModelSpec(1.0, 5.0, 64_000, _200K, adaptive_thinking=False, supports_effort=False),
    # Retired on the first-party API (still served on Bedrock / Google Cloud).
    "claude-opus-4-1": ClaudeModelSpec(15.0, 75.0, 32_000, _200K, adaptive_thinking=False, supports_effort=False, supports_structured_output=False),
    "claude-opus-4": ClaudeModelSpec(15.0, 75.0, 32_000, _200K, adaptive_thinking=False, supports_effort=False, supports_structured_output=False),
    "claude-sonnet-4": ClaudeModelSpec(3.0, 15.0, 64_000, _200K, adaptive_thinking=False, supports_effort=False, supports_structured_output=False),
    "claude-3-7-sonnet": ClaudeModelSpec(3.0, 15.0, 64_000, _200K, adaptive_thinking=False, supports_effort=False, supports_structured_output=False),
    "claude-3-5-haiku": ClaudeModelSpec(0.80, 4.0, 8_192, _200K, adaptive_thinking=False, supports_effort=False, supports_structured_output=False),
    "claude-3-haiku": ClaudeModelSpec(0.25, 1.25, 4_096, _200K, adaptive_thinking=False, supports_effort=False, supports_structured_output=False),
    "claude-3-opus": ClaudeModelSpec(15.0, 75.0, 4_096, _200K, adaptive_thinking=False, supports_effort=False, supports_structured_output=False),
}

# Used for an ID the table does not know, always with a warning. The current-generation shape: every model
# released after this table was written is adaptive, 1M-context and at least 64K-output, and the price is the
# current Opus tier, so an unknown model is neither silently capped at a legacy limit nor priced at zero.
UNKNOWN_CLAUDE_MODEL = ClaudeModelSpec(4.0, 20.0, 64_000, _200K)

# Claude Code's model aliases, as the CLI resolves them (``claude --model opus``). Verified 2026-09-26 against
# CLI 2.1.263 by reading ``modelUsage`` off a live ``--output-format json`` result: ``opus`` is Opus 5, not the
# newer Opus 5.5, so the alias cannot be derived from the API lineup and is pinned from the measurement.
CLAUDE_CODE_ALIASES: dict[str, str] = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
    "fable": "claude-fable-5-1",
}

# A suffix that still names the SAME model: a snapshot date, ``-latest``, a Vertex ``@date`` or a Bedrock ``-v1:0``.
_SNAPSHOT_SUFFIX = re.compile(r"^(?:-\d{8}|-latest)?(?:-v\d+(?::\d+)?)?$")
_PLATFORM_PREFIX = re.compile(r"^(?:[a-z]{2,4}\.)?anthropic[./]")

_warned_unknown: set[str] = set()


def resolve_claude_model(model: str) -> tuple[str, ClaudeModelSpec] | None:
    """The table key and spec for ``model``, or None when the table does not know it.

    Matches the key exactly or followed by a snapshot suffix only. Deliberately NOT a bare longest-prefix match:
    ``claude-opus-4`` is a prefix of ``claude-opus-4-9``, and pricing a future model from the legacy $15/$75 row
    without a word is the failure this table exists to stop.
    """
    name = _PLATFORM_PREFIX.sub("", (model or "").strip().lower()).replace("@", "-")
    name = CLAUDE_CODE_ALIASES.get(name, name)
    spec = CLAUDE_MODELS.get(name)
    if spec is not None:
        return name, spec
    best: tuple[str, ClaudeModelSpec] | None = None
    for key, candidate in CLAUDE_MODELS.items():
        if name.startswith(key) and _SNAPSHOT_SUFFIX.match(name[len(key) :]) and (best is None or len(key) > len(best[0])):
            best = (key, candidate)
    return best


def claude_model_spec(model: str, *, provider_label: str = "Claude") -> ClaudeModelSpec:
    """The spec for ``model``; an unknown ID gets :data:`UNKNOWN_CLAUDE_MODEL` and ONE warning per ID."""
    resolved = resolve_claude_model(model)
    if resolved is not None:
        return resolved[1]
    if model not in _warned_unknown:
        _warned_unknown.add(model)
        logger.warning(
            "%s model %r is not in the Claude model table; assuming $%s/$%s per 1M, %d max output, %d context. "
            "Add it to pyutilz.llm._claude_models.CLAUDE_MODELS to price and size it correctly.",
            provider_label,
            model,
            UNKNOWN_CLAUDE_MODEL.input_per_1m,
            UNKNOWN_CLAUDE_MODEL.output_per_1m,
            UNKNOWN_CLAUDE_MODEL.max_output,
            UNKNOWN_CLAUDE_MODEL.context_window,
        )
    return UNKNOWN_CLAUDE_MODEL
