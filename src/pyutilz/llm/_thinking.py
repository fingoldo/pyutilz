"""How a caller's ``thinking=`` becomes whatever the upstream actually wants.

Carved out of ``openai_compat.py`` when that module reached this project's 1,000-line split
threshold (``tests/test_architecture_audit_20260902.py``). One subject, two seams, and both are
overridden by the same subclasses -- OpenRouter maps an effort string onto ``reasoning.effort``,
DeepSeek V4 collapses it to a ``thinking.type`` flag.

A MIXIN rather than free functions, because ``_thinking_request_field`` is a hook: subclasses
override it and callers reach it as ``self._thinking_request_field(...)``. Moving it to a module
function would have been a rename disguised as a refactor.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pyutilz.llm.base import normalize_thinking

logger = logging.getLogger(__name__)

#: Extended-thinking budget per effort level, shared by the two Claude providers (API and Claude Code) of the shared vocabulary (see ``pyutilz.llm.base.normalize_thinking``).
#: Anthropic takes a token BUDGET where the rest of this package takes an effort string. ``True`` means "on, provider
#: default", which normalises to medium.
THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    # The two levels above ``high`` that the effort API added (``output_config.effort`` on Claude, ``--effort`` on the
    # Claude Code CLI). Budget-only models need a number for them too, or ``thinking="xhigh"`` would be refused there.
    "xhigh": 16384,
    "max": 32768,
}
#: Anthropic's minimum accepted budget; the answer needs at least as much again.
MIN_THINKING_BUDGET = 1024


#: The caller's effort vocabulary onto Claude's ``output_config.effort`` / ``--effort`` levels
#: (https://platform.claude.com/docs/en/build-with-claude/effort: low, medium, high, xhigh, max). Claude has no
#: ``minimal``; its nearest level is ``low``.
CLAUDE_EFFORTS: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def claude_effort(effort: str | None, *, model: str = "") -> str | None:
    """Claude's effort level for a normalised effort string; None for "provider default" or an unknown effort.

    An unknown effort is warned about and dropped rather than guessed: sending a different level than the caller
    asked for bills reasoning they did not request while they believe their setting took effect.
    """
    if effort is None:
        return None
    level = CLAUDE_EFFORTS.get(effort)
    if level is None:
        logger.warning("Unknown thinking effort %r for %s; not sending an effort level. Known: %s", effort, model if model else "Claude", sorted(CLAUDE_EFFORTS))
    return level


def claude_code_effort(thinking: bool | str | int | None) -> str | None:
    """The ``--effort`` level for a ``thinking=`` request, or None when there is no explicit effort to send.

    The Claude Code CLI (2.1.263) takes ``--effort low|medium|high|xhigh|max``, and the models behind its aliases think
    adaptively, where a token budget is not the control. ``True`` (on, provider default) and off both return None here:
    off is expressed through ``claude_code_thinking_tokens`` returning 0.
    """
    if thinking is None:
        return None
    enabled, effort = normalize_thinking(thinking)
    if not enabled:
        return None
    return claude_effort(effort, model="claude-code")


def claude_code_thinking_tokens(thinking: bool | str | int | None) -> int | None:
    """The MAX_THINKING_TOKENS value for a ``thinking=`` request, or None to leave the CLI's own default.

    The budgets are the Anthropic API provider's (``THINKING_BUDGETS``), so one effort name means one budget whichever
    of the two Claude providers serves it. An effort the table does not know takes the medium budget and says so,
    rather than silently running at the CLI default the caller did not ask for.
    """
    if thinking is None:
        return None
    enabled, effort = normalize_thinking(thinking)
    if not enabled:
        return 0
    budget = THINKING_BUDGETS.get("medium" if effort is None else effort)
    if budget is None:
        logger.warning("Unknown thinking effort %r for claude-code; using the medium budget %d", effort, THINKING_BUDGETS["medium"])
        return THINKING_BUDGETS["medium"]
    return budget


#: Gemini 3 ``thinking_level`` values (https://ai.google.dev/gemini-api/docs/thinking). Levels above ``high`` have no
#: Gemini counterpart and clamp to it.
GEMINI_THINKING_LEVELS: dict[str, str] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
#: Gemini 3 models whose documented levels start at ``low`` (no ``minimal``), per the thinking guide fetched 2026-09-26.
_GEMINI_NO_MINIMAL_PREFIXES = ("gemini-3-pro", "gemini-3.1-pro", "gemini-3.7-flash", "gemini-3.8-flash")


def _uses_thinking_level(model: str) -> bool:
    """Gemini 3 and later are configured by ``thinking_level``; 2.x by ``thinking_budget``."""
    match = re.match(r"(?:models/)?gemini-(\d+)", model.lower())
    return match is not None and int(match.group(1)) >= 3


def gemini_thinking_config(thinking: bool | str | int | None, model: str) -> dict[str, Any] | None:
    """The ``ThinkingConfig`` keyword arguments for a ``thinking=`` request, or None to send no thinking config.

    Gemini 3 is steered by ``thinking_level`` and cannot switch thinking off: "off" becomes the lowest level the model
    documents (``minimal``, or ``low`` where there is none). Gemini 2.5 keeps the budget form, see
    :func:`gemini_thinking_budget`.
    """
    if thinking is None:
        return None
    if not _uses_thinking_level(model):
        budget = gemini_thinking_budget(thinking, model)
        return None if budget is None else {"thinking_budget": budget}
    lowest = "low" if model.lower().startswith(_GEMINI_NO_MINIMAL_PREFIXES) else "minimal"
    enabled, effort = normalize_thinking(thinking)
    if not enabled:
        return {"thinking_level": lowest}
    if effort is None:
        return {"thinking_level": "medium"}
    level = GEMINI_THINKING_LEVELS.get(effort)
    if level is None:
        logger.warning("Unknown thinking effort %r for %s; using medium", effort, model)
        level = "medium"
    if level == "minimal" and lowest != "minimal":
        level = lowest
    return {"thinking_level": level}


def gemini_thinking_budget(thinking: bool | str | int | None, model: str) -> int | None:
    """The ``thinking_budget`` for a ``thinking=`` request, or None to send no thinking config.

    The effort budgets are the Claude providers' (``THINKING_BUDGETS``), so one effort name costs about the same
    reasoning on each. Off is budget 0, except on a Pro model, which cannot turn thinking off (the API rejects 0):
    there the request is not sent and a warning says the model thinks anyway.
    """
    if thinking is None:
        return None
    enabled, effort = normalize_thinking(thinking)
    if not enabled:
        # Only 2.5 Pro refuses budget 0 among the budget-configured models; a substring test for "pro" also caught
        # any future id that merely contains it.
        if model.lower().startswith("gemini-2.5-pro"):
            logger.warning("%s cannot turn thinking off; thinking=%r is not sent and the model uses its default", model, thinking)
            return None
        return 0
    budget = THINKING_BUDGETS.get("medium" if effort is None else effort)
    if budget is None:
        logger.warning("Unknown thinking effort %r for %s; using the medium budget %d", effort, model, THINKING_BUDGETS["medium"])
        return THINKING_BUDGETS["medium"]
    return budget


class ThinkingControlMixin:
    """The reasoning-effort seams shared by every OpenAI-compatible provider."""

    def _thinking_request_field(self, thinking: bool | str) -> dict[str, Any] | None:
        """Return the request-body fragment that toggles thinking mode.

        ``thinking`` accepts BOTH a plain bool (legacy) AND an effort
        string (``"low"`` / ``"medium"`` / ``"high"`` / ``"minimal"``).
        Subclasses normalise to the upstream's actual schema:

          * Effort-string upstreams (OpenRouter's unified ``reasoning``
            field, OpenAI ``reasoning_effort``) consume the literal
            string; ``True`` is mapped to a sensible default
            (``"medium"``).
          * Boolean-flag upstreams (DeepSeek V4 ``thinking.type``)
            coerce a non-empty effort string to ``True`` so
            ``thinking="high"`` still enables on those models.

        Provider-specific. Default returns ``None`` so callers see
        vanilla OpenAI-compatible behavior (no thinking control).
        DeepSeek V4 overrides to return ``{"thinking": {"type": ...}}``;
        OpenRouter overrides to return
        ``{"reasoning": {"effort": ...}}``.
        """
        return None

    @staticmethod
    def _normalize_thinking(thinking: bool | str | int) -> tuple[bool, str | None]:
        """Delegates to :func:`pyutilz.llm.base.normalize_thinking` -- kept as a
        method so existing subclass overrides and call sites keep working."""
        return normalize_thinking(thinking)
