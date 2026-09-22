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
}
#: Anthropic's minimum accepted budget; the answer needs at least as much again.
MIN_THINKING_BUDGET = 1024


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
        if "pro" in model.lower():
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
