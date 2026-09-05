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

import base64
import contextlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

__all__ = ["build_chat_messages"]


def _report_dropped(dropped: list[str], total: int, provider: str) -> None:
    """One warning for the whole batch, naming how many of how many were not sent.

    Per-iteration warnings are log spam by construction here: a caller passing the wrong shape
    passes it for EVERY picture, so a posting with twelve attachments emits twelve identical lines
    that say nothing the first one did not. The COUNT is the part that matters -- "3 of 3" is a
    caller bug and "1 of 12" is one bad attachment.
    """
    if not dropped:
        return
    logger.warning("%s: %d of %d image(s) not sent -- %s", provider, len(dropped), total, "; ".join(sorted(set(dropped))))


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


def build_anthropic_content(prompt: str, images: list[str] | None = None) -> str | list[dict[str, Any]]:
    """The Anthropic user-message ``content``, optionally carrying images.

    Anthropic does NOT accept the OpenAI ``image_url`` part shape. Its blocks are
    ``{"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}`` for inline
    bytes, or ``{"type": "url", "url": ...}`` for a fetchable address, so a caller that forwards the
    OpenAI form gets a 400 rather than a picture.

    With ``images=None`` the return value is the plain STRING the provider has always sent, so a
    text-only call produces a byte-identical request body.

    The text block comes first, for the same reason it does in `build_chat_messages`: the
    instruction should precede the pictures, or a long image sequence pushes the question out of the
    model's attention.

    A ``data:`` URI is split into its media type and payload. Anything else is passed as a URL
    block; a malformed entry is DROPPED with a warning rather than sent, because one bad attachment
    must not 400 the whole evaluation -- the letter is still worth writing without that picture.
    """
    if not images:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    dropped: list[str] = []
    for url in images:
        if url.startswith("data:"):
            split = split_data_uri(url)
            if split is None:
                dropped.append("a malformed data URI")
                continue
            media_type, payload = split
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": payload}})
        elif url.startswith(("http://", "https://")):
            content.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            dropped.append("a reference that is neither a data URI nor an http URL")
    _report_dropped(dropped, len(images), "Anthropic")
    return content if len(content) > 1 else prompt


def split_data_uri(url: str) -> "tuple[str, str] | None":
    """`(media_type, base64_payload)` for a ``data:`` URI, or None if it is not one or is malformed.

    Shared by every provider that has to inline image bytes, because each of the three APIs wants
    the same two pieces in a different envelope and parsing the URI three times invites three
    slightly different notions of "malformed".
    """
    if not url.startswith("data:"):
        return None
    header, _, payload = url.partition(",")
    media_type = header[5:].split(";", 1)[0]
    if not payload or not media_type.startswith("image/"):
        return None
    return media_type, payload


def build_gemini_parts(prompt: str, images: "list[str] | None" = None) -> "str | list[Any]":
    """Gemini ``contents``: the prompt alone, or a parts list carrying inline image bytes.

    google-genai wants `types.Part.from_bytes(data=..., mime_type=...)`, which is neither the
    OpenAI `image_url` shape nor Anthropic's `source` block. Returns the plain STRING when there are
    no images, so a text-only request body is byte-identical to the one sent before vision existed.

    An http(s) URL is NOT passed through: Gemini's inline path takes bytes, and its file API needs an
    upload step this call has no business performing. Such an entry is dropped with a warning, which
    is the same choice the other two providers make about an input they cannot honour -- one bad
    attachment must not fail the whole evaluation.
    """
    if not images:
        return prompt
    import base64

    from google.genai import types

    parts: list[Any] = [types.Part.from_text(text=prompt)]
    dropped: list[str] = []
    for url in images:
        split = split_data_uri(url)
        if split is None:
            dropped.append("a reference Gemini cannot inline (it takes image BYTES)")
            continue
        media_type, payload = split
        try:
            raw = base64.b64decode(payload)
        except Exception:  # a malformed part is dropped, never fatal
            dropped.append("an image whose base64 payload will not decode")
            continue
        parts.append(types.Part.from_bytes(data=raw, mime_type=media_type))
    _report_dropped(dropped, len(images), "Gemini")
    return parts if len(parts) > 1 else prompt


@contextlib.contextmanager
def images_on_disk(images: list[str] | None) -> Iterator[tuple[str, list[Path]]]:
    """Write `data:` URIs to temporary files and yield the prompt fragment that points at them.

    The Claude Code CLI is the odd one out among these providers: it accepts a prompt string and
    nothing else, but the agent behind it can OPEN FILES. So the way to show it a picture is to put
    the picture somewhere and say where -- which is what a person using the CLI would do.

    Yields ``("", [])`` when there is nothing to show, so the caller's prompt is unchanged. The
    directory is removed on the way out whether or not the call succeeded: a client's attachment
    must not be left behind on a failure.

    The caller must keep the block OPEN for the whole request. Closing it around a prompt-building
    line would delete every picture before the subprocess started.
    """
    if not images:
        yield "", []
        return
    directory = Path(tempfile.mkdtemp(prefix="pyutilz_llm_img_"))
    written: list[Path] = []
    dropped: list[str] = []
    try:
        for index, url in enumerate(images):
            split = split_data_uri(url)
            if split is None:
                dropped.append("a reference that cannot be written as a file")
                continue
            media_type, payload = split
            try:
                raw = base64.b64decode(payload)
            except Exception:  # a malformed part is dropped, never fatal
                dropped.append("an image whose base64 payload will not decode")
                continue
            extension = media_type.split("/", 1)[1].split("+")[0]
            path = directory / f"attachment_{index + 1}.{extension}"
            path.write_bytes(raw)
            written.append(path)
        _report_dropped(dropped, len(images), "claude-code")
        if not written:
            yield "", []
            return
        listing = "\n".join(f"- {p}" for p in written)
        yield (
            "\n\nThe following image file(s) accompany this request. READ THEM before answering; " "they are part of the input, not a reference:\n" + listing
        ), written
    finally:
        shutil.rmtree(directory, ignore_errors=True)
