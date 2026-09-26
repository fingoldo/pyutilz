"""What happened to the streamed attempts of one call that did not end in an answer.

Carved out of ``openai_compat`` for the 1000-line budget. Mixed into ``OpenAICompatibleProvider``, which supplies
``_body_after_rejected_request``. A streamed attempt the upstream had started generating for is billed for what it
generated, whether or not an answer reached the caller; a caller that reads only the final attempt's usage never sees
that money. ``last_aborted_stream_attempts`` keeps each such attempt with its generation id, so the caller can reconcile
the bill through OpenRouter's ``/generation``, or at least knows that an attempt was answered and then lost.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pyutilz.llm.base import PerCallAttr
from pyutilz.llm.exceptions import LLMTruncationError

logger = logging.getLogger(__name__)


class StreamAttemptsMixin:
    """Per-call record of abandoned streamed attempts, and the one repaired re-issue of a refused stream."""

    # Every streamed attempt of this call that ended without an answer, oldest first: ``{"attempt", "generation_id",
    # "response_started", "generated_chunks", "error"}``.
    last_aborted_stream_attempts: PerCallAttr = PerCallAttr(list)

    def _body_after_rejected_request(self, _body: dict[str, Any], _status: int, _detail: str) -> dict[str, Any] | None:  # pragma: no cover - supplied by the provider
        """The provider's repaired body for a refused request; the provider this mixes into supplies it."""
        raise NotImplementedError

    def _note_stream_chunk(self, state: dict[str, Any], chunk: Mapping[str, Any], first: bool) -> None:
        """Fold one parsed SSE chunk into an attempt's ``state``: the generation id of its first chunk, and a count of the chunks that carried a choice."""
        if first:
            gen_id = chunk.get("id")
            state["generation_id"] = gen_id if isinstance(gen_id, str) else None
        if chunk.get("choices"):
            state["generated_chunks"] = int(state.get("generated_chunks") or 0) + 1

    def _note_aborted_stream_attempt(self, attempt: int, state: Mapping[str, Any], exc: BaseException) -> None:
        """Append one abandoned streamed attempt to ``last_aborted_stream_attempts`` (reassigned, so the per-call copy is the one changed).

        A truncation is not one: it is an answer cut by ``max_tokens``, and its usage is the call's own.
        """
        if isinstance(exc, LLMTruncationError):
            return
        record = {
            "attempt": attempt,
            "generation_id": state.get("generation_id"),
            "response_started": bool(state.get("response_started")),
            "generated_chunks": int(state.get("generated_chunks") or 0),
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }
        self.last_aborted_stream_attempts = [*(self.last_aborted_stream_attempts or []), record]

    async def _repaired_stream_body(self, exc: Exception, body: dict[str, Any]) -> dict[str, Any] | None:
        """Return a repaired request body for a STREAM the upstream refused over a parameter, else ``None``.

        Streaming counterpart of the ``_body_after_rejected_request`` consultation in
        ``_post_and_unwrap``: a streamed error response's body has not been read yet, so it is
        pulled here before the detail is handed to the same provider hook. No delta can have been
        yielded on a 4xx, so re-opening the stream cannot duplicate already-emitted tokens.
        """
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        if resp is None or status is None:
            return None
        detail = ""
        try:
            raw = await resp.aread()
            detail = raw.decode("utf-8", errors="replace")
        except Exception as read_exc:
            logger.debug("Could not read the refused stream's error body: %s", read_exc)
        return self._body_after_rejected_request(body, int(status), detail)
