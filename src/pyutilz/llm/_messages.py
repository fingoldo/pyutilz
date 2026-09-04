"""Assembling an OpenAI-style chat ``messages`` list, with or without images.

Carved out of ``openai_compat.py`` when that module crossed this repo's 1,000-line ceiling
(``tests/test_meta/test_shared_checks_wired.py``), by the sibling route the convention prescribes.

The seam is real rather than convenient: this is a pure function of its arguments with no provider
state, no HTTP and no configuration, which is exactly what the rest of ``openai_compat`` is not.
``OpenAICompatibleProvider._build_messages`` stays where it is and delegates, because it is a
documented override point and moving it would break every subclass that customises the message
shape.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_chat_messages"]


def build_chat_messages(
    prompt: str,
    system: str | None = None,
    images: list[str] | None = None,
) -> list[dict[str, Any]]:
    """The chat ``messages`` list, optionally carrying images.

    ``images`` are URLs or ``data:`` URIs. Given any, the user message's ``content`` becomes the
    OpenAI multimodal LIST form (one text part, then one ``image_url`` part each) that OpenRouter and
    every OpenAI-compatible vision endpoint accept; a ``data:image/png;base64,...`` URI is inlined,
    so there is no upload step and no public URL is needed.

    With ``images=None`` the body is byte-identical to the pre-vision one -- ``content`` stays a
    plain string. That is load-bearing: a model without vision rejects the list form outright, and
    nearly every call through this provider is text-only.

    The text part comes FIRST, which is not cosmetic. The instruction telling the model what to do
    with the picture should precede the picture, or a long image sequence pushes the question out of
    the model's attention.
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if images:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages
