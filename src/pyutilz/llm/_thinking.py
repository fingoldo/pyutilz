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

from typing import Any

from pyutilz.llm.base import normalize_thinking


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
