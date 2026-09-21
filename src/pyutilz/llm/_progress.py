"""How much a streamed call has produced so far, readable from a task other than the one streaming it.

A streamed call's reasoning lives in the provider's ``_reasoning_fragments``, a ``PerCallAttr`` backed by
a ContextVar. That is right for isolating concurrent calls on one shared provider, and it is exactly why a
watchdog or a heartbeat running as its own asyncio Task cannot see it: the Task copied the context when it
was created, before the stream reset and started filling the fragments, so every read returns an empty
default. Measured 2026-09-19 against z-ai/glm-5.3-flash: a heartbeat reported "0 reasoning char(s)" for
134 seconds while the model reasoned through its whole 12,000-token budget.

This module hands the counts over BY REFERENCE instead. The caller creates a :class:`StreamProgress`,
installs it with :func:`track_stream_progress` in the task that consumes the stream, and gives the same
object to whatever watches. The provider's stream loop, which runs in the consuming task, finds the
installed object through a ContextVar and increments it; the watchers read the plain integers.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = ["StreamProgress", "note_stream_progress", "track_stream_progress"]


@dataclass
class StreamProgress:
    """Characters received on one streamed call, split into reasoning and answer."""

    reasoning_chars: int = 0
    answer_chars: int = 0

    @property
    def total_chars(self) -> int:
        """Everything received; growth in either half means the model is still producing."""
        return self.reasoning_chars + self.answer_chars


_CURRENT: contextvars.ContextVar[StreamProgress | None] = contextvars.ContextVar("pyutilz_llm_stream_progress", default=None)


@contextlib.contextmanager
def track_stream_progress(progress: StreamProgress | None = None) -> Iterator[StreamProgress]:
    """Install ``progress`` (or a fresh one) as the counter for streams consumed inside this block."""
    target = progress if progress is not None else StreamProgress()
    token = _CURRENT.set(target)
    try:
        yield target
    finally:
        _CURRENT.reset(token)


def note_stream_progress(reasoning: str | None, answer: str | None) -> None:
    """Count one stream delta against the installed counter; a no-op when nobody installed one."""
    progress = _CURRENT.get()
    if progress is None:
        return
    if isinstance(reasoning, str):
        progress.reasoning_chars += len(reasoning)
    if isinstance(answer, str):
        progress.answer_chars += len(answer)
