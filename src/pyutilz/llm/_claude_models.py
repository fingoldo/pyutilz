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

Re-checked 2026-10-03 against ``GET /v1/models`` (``max_input_tokens``, ``max_tokens``, ``capabilities.effort`` and
``capabilities.thinking.types`` per model) and the pricing page: Sonnet 5.5 was missing (priced as an unknown model at
the Opus fallback, twice its real rate), Sonnet 4.5's context is 1M, and the effort levels differ per model.
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
    # The ``output_config.effort`` levels the model accepts, as ``GET /v1/models`` reports them under
    # ``capabilities.effort`` (read live 2026-10-03): 4.6 has no ``xhigh``, Opus 4.5 neither ``xhigh`` nor ``max``.
    effort_levels: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


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
    "claude-opus-4-6": ClaudeModelSpec(5.0, 25.0, 128_000, _1M, effort_levels=("low", "medium", "high", "max")),
    "claude-sonnet-5-5": ClaudeModelSpec(2.0, 10.0, 128_000, _1M),
    "claude-sonnet-5": ClaudeModelSpec(2.0, 10.0, 128_000, _1M),
    "claude-sonnet-4-6": ClaudeModelSpec(3.0, 15.0, 128_000, _1M, effort_levels=("low", "medium", "high", "max")),
    # Extended-thinking-only generation. Opus 4.5 is the one of them that also takes ``effort``.
    "claude-opus-4-5": ClaudeModelSpec(5.0, 25.0, 64_000, _200K, adaptive_thinking=False, effort_levels=("low", "medium", "high")),
    # ``GET /v1/models`` reports ``max_input_tokens: 1000000`` for claude-sonnet-4-5-20250929 (2026-10-03).
    "claude-sonnet-4-5": ClaudeModelSpec(3.0, 15.0, 64_000, _1M, adaptive_thinking=False, supports_effort=False),
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

# Not listed on purpose: ``claude-haiku-5-5``. The CLI reports its limits (``modelUsage``, 2026-10-12, CLI 2.1.296):
# ``maxOutputTokens`` 128,000 and ``contextWindow`` 1,000,000, accepts ``--effort`` (low and max verified, thinking
# 394 and 1,322 tokens on one question), and the table has no documented price for it. An invented price would misprice
# every call, so a model the table lacks is priced from the OpenRouter catalogue (the live listing of the same model,
# :func:`_openrouter_claude_spec`); only when that lists no row either does the unknown-model default apply. The default's
# own max-output and context figures are ours and are NOT a cap sent to the model; the Claude Code provider never sends max_tokens.

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


#: OpenRouter lookups already made, by the normalised model name; None is a name the catalogue does not list.
_OPENROUTER_LOOKUPS_MAX_SIZE = 256
_openrouter_specs: dict[str, tuple[str, ClaudeModelSpec] | None] = {}


def _normalised_name(model: str) -> str:
    """The lower-case, platform-prefix-free name the table and the OpenRouter listing are both matched by."""
    return _PLATFORM_PREFIX.sub("", (model or "").strip().lower()).replace("@", "-")


def _listing_spec(entry: dict) -> ClaudeModelSpec | None:
    """The spec of one OpenRouter listing entry: per-token prices scaled to per 1M, the ceiling and context it states; None without prices."""
    pricing = entry.get("pricing") or {}
    try:
        input_per_1m = float(pricing["prompt"]) * _1M
        output_per_1m = float(pricing["completion"]) * _1M
    except (KeyError, TypeError, ValueError):
        return None
    top = entry.get("top_provider") or {}
    max_output = int(top.get("max_completion_tokens") or UNKNOWN_CLAUDE_MODEL.max_output)
    context = int(entry.get("context_length") or UNKNOWN_CLAUDE_MODEL.context_window)
    return ClaudeModelSpec(input_per_1m, output_per_1m, max_output, context)


def _openrouter_claude_spec(name: str) -> tuple[str, ClaudeModelSpec] | None:
    """The OpenRouter listing of Claude model ``name`` as ``(listing id, spec)``, or None.

    Matches only ``anthropic/<name>`` (dots read as dashes, so ``claude-haiku-5.5`` is ``claude-haiku-5-5``), with an optional
    snapshot suffix. Variants such as ``:batch`` and aliases such as ``~anthropic/claude-haiku-latest`` are NOT the model's own
    price and never match. A failed catalogue fetch is a miss for this call only: the fetch itself is not cached on failure.
    """
    if name in _openrouter_specs:
        return _openrouter_specs[name]
    from pyutilz.llm.openrouter_provider._catalogue import _fetch_models_catalogue  # lazy: the catalogue module imports this package

    try:
        catalogue = _fetch_models_catalogue()
    except Exception as exc:  # a failed lookup falls back to the default row, it never breaks a call
        logger.debug("OpenRouter catalogue unavailable for %r: %s", name, exc)
        return None
    found = None
    for listing_id, entry in catalogue.items():
        if not listing_id.startswith("anthropic/"):
            continue
        bare = listing_id[len("anthropic/") :].replace(".", "-")
        if bare != name and not (bare.startswith(name) and _SNAPSHOT_SUFFIX.match(bare[len(name) :])):
            continue
        spec = _listing_spec(entry)
        if spec is not None:
            found = (listing_id, spec)
            break
    if len(_openrouter_specs) >= _OPENROUTER_LOOKUPS_MAX_SIZE:
        _openrouter_specs.pop(next(iter(_openrouter_specs)))  # bounded FIFO: evict the oldest name, which costs at most one re-lookup
    _openrouter_specs[name] = found
    return found


def claude_model_listed_price(model: str) -> tuple[float, float] | None:
    """``(input, output)`` USD per 1M for ``model``: its table row, else OpenRouter's listing of it; None when neither lists it.

    Unlike :func:`claude_model_spec` there is no default, so a caller that prices a call can tell "not listed" from a price.
    """
    resolved = resolve_claude_model(model)
    if resolved is not None:
        return resolved[1].input_per_1m, resolved[1].output_per_1m
    listed = _openrouter_claude_spec(_normalised_name(model))
    return None if listed is None else (listed[1].input_per_1m, listed[1].output_per_1m)


def claude_model_spec(model: str, *, provider_label: str = "Claude") -> ClaudeModelSpec:
    """The spec for ``model``: the table's row, else OpenRouter's listing of it, else :data:`UNKNOWN_CLAUDE_MODEL` (warned once per ID)."""
    resolved = resolve_claude_model(model)
    if resolved is not None:
        return resolved[1]
    listed = _openrouter_claude_spec(_normalised_name(model))
    if listed is not None:
        if model not in _warned_unknown:
            _warned_unknown.add(model)
            logger.info(
                "%s model %r is not in the Claude model table; priced from the OpenRouter listing %r at $%.4g/$%.4g per 1M, %d max output.",
                provider_label, model, listed[0], listed[1].input_per_1m, listed[1].output_per_1m, listed[1].max_output,
            )  # fmt: skip
        return listed[1]
    if model not in _warned_unknown:
        _warned_unknown.add(model)
        logger.warning(
            "%s model %r is not in the Claude model table and OpenRouter does not list it; assuming $%s/$%s per 1M, %d max output, %d context. "
            "Add it to pyutilz.llm._claude_models.CLAUDE_MODELS to price and size it correctly.",
            provider_label,
            model,
            UNKNOWN_CLAUDE_MODEL.input_per_1m,
            UNKNOWN_CLAUDE_MODEL.output_per_1m,
            UNKNOWN_CLAUDE_MODEL.max_output,
            UNKNOWN_CLAUDE_MODEL.context_window,
        )
    return UNKNOWN_CLAUDE_MODEL
