"""Request-body assembly for OpenAI-compatible providers: response_format, messages, extra fields, the stream body.

Carved out of ``openai_compat.py`` for the repo's 1000-line module budget. Mixed into ``OpenAICompatibleProvider``, which
supplies the attributes declared below; every method here is an override point a provider may replace.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from pyutilz.llm._messages import build_chat_messages
from pyutilz.llm._openai_compat_http import merge_extra_body, raise_for_error_in_body

logger = logging.getLogger("pyutilz.llm.openai_compat")


class RequestBodyMixin:
    """Builds the ``/chat/completions`` body; see the module docstring."""

    model_name: str
    _provider_name: str
    _last_json_schema_applied: Any

    if TYPE_CHECKING:  # supplied by OpenAICompatibleProvider and its other mixins

        @property
        def max_output_tokens(self) -> int:
            """The provider's output-token ceiling (typing stub)."""
            ...

        def fit_max_tokens_to_context(self, max_tokens: int, prompt: str, system: str | None = None) -> int:
            """The output budget clamped to the context window (typing stub)."""
            ...

        def _thinking_request_field(self, thinking: Any) -> dict[str, Any] | None:
            """The request field that carries ``thinking`` (typing stub)."""
            ...

    def supports_json_mode(self) -> bool:
        """All OpenAI-compatible Chat Completions endpoints accept
        ``response_format={"type": "json_object"}`` since 2023-11.
        Subclasses with model-specific gating (notably OpenRouter, where
        per-model support varies) override this with a catalogue check.
        """
        return True

    def supports_json_schema(self) -> bool:
        """OpenAI-compatible endpoints have accepted strict ``json_schema`` response formats since
        2024-08. Support is per-model in practice, so routers (OpenRouter) override with a catalogue
        check; a direct single-vendor endpoint can assume its own models.
        """
        return True

    def _response_format(self, json_mode: bool, json_schema: dict[str, Any] | None) -> dict[str, Any] | None:
        """Choose the strictest response_format the model actually supports, and record which applied.

        A caller passing ``json_schema`` to a model without strict-schema support degrades to plain JSON
        mode rather than failing the call — a mixed-model sweep stays runnable — but the degradation is
        logged and exposed via ``last_json_schema_applied`` so the caller can tell a guaranteed-shape
        response from a merely-hopeful one instead of assuming the guarantee held.
        """
        self._last_json_schema_applied = False
        if json_schema is not None:
            if self.supports_json_schema():
                # `strict` defaults to True on a copy: without it the upstream treats the schema as a hint, and the flag
                # below would claim a guarantee nobody enforced. An explicit `strict: False` is sent as given and reported.
                schema = {**json_schema, "strict": json_schema.get("strict", True)}
                self._last_json_schema_applied = schema["strict"] is True
                return {"type": "json_schema", "json_schema": schema}
            logger.warning(
                "%s/%s does not support strict json_schema; falling back to json_object (enums NOT enforced)",
                self._provider_name,
                self.model_name,
            )
        if json_mode:
            return {"type": "json_object"}
        return None

    @property
    def last_json_schema_applied(self) -> bool:
        """Whether the most recent call actually constrained generation to the caller's JSON schema."""
        return getattr(self, "_last_json_schema_applied", False)

    def _messages_for(self, prompt: str, system: str | None, images: "list[str] | None") -> list[dict[str, Any]]:
        """``_build_messages``, called with the arity the request needs.

        The third argument goes only when there ARE images: ``_build_messages`` is an override point
        that a subclass or test double may still define as ``(self, prompt, system)``, so a text-only
        call must reach it with two arguments and an unchanged body.
        """
        return self._build_messages(prompt, system, images) if images else self._build_messages(prompt, system)

    def _build_messages(
        self,
        prompt: str,
        system: str | None = None,
        images: "list[str] | None" = None,
    ) -> list[dict[str, Any]]:
        """The chat ``messages`` list -- see :func:`pyutilz.llm._messages.build_chat_messages`.

        Kept as a delegating METHOD because it is a documented override point.
        """
        return build_chat_messages(prompt, system, images)

    def _extra_request_body(self, model: str) -> dict[str, Any]:
        """Return provider-specific extra fields to merge into the request body.

        Subclasses override for things like vendor-specific defaults.
        Defaults to empty so callers see vanilla OpenAI-compatible behavior.
        """
        return {}

    def _finalize_request_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Last look at the complete body (thinking field and per-call ``extra_body`` included); default unchanged.

        OpenRouter decides ``provider.require_parameters`` here, because that depends on every parameter sent.
        """
        return body

    def _request_body(
        self,
        prompt: str,
        system: str | None,
        temperature: float | None,
        max_tokens: int,
        rf: dict[str, Any] | None,
        thinking: bool | str | None,
        images: list[str] | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The ``/chat/completions`` body shared by both paths; ``extra_body`` is merged LAST (see ``merge_extra_body``)."""
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._messages_for(prompt, system, images),
            # `None` means DO NOT SEND the field, so the upstream applies its own default. `0.0` is a real
            # temperature and the most deterministic one, which a truthiness test would silently turn off.
            **({} if temperature is None else {"temperature": temperature}),
            "max_tokens": max_tokens,
        }
        if rf is not None:
            body["response_format"] = rf
        if tools:
            body["tools"] = list(tools)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        body.update(self._extra_request_body(self.model_name))
        if thinking is not None:
            tf = self._thinking_request_field(thinking)
            if tf is not None:
                body.update(tf)
        if extra_body:
            body = merge_extra_body(body, extra_body)
        return self._finalize_request_body(body)

    def _build_stream_body(
        self,
        prompt: str,
        system: str | None,
        temperature: float | None,
        max_tokens: int,
        json_mode: bool,
        thinking: bool | str | None,
        json_schema: dict[str, Any] | None,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the ``/chat/completions`` request body for a STREAMING call.

        Split out of ``generate_stream`` so the streaming state machine stays within the project's
        C901 budget; the caller must already have awaited ``_async_prepare()``, because the
        ``max_output_tokens`` / ``context_window`` properties read here can hit the catalogue.
        """
        if max_tokens <= 0:
            max_tokens = self.max_output_tokens
        # `fit_max_tokens_to_context`'s reserve now scales with input size (see `base.py`'s
        # `_context_reserve_tokens`), not a flat 1024-token constant - fixes a measured incident
        # (2026-08-08, autopsia pilot): a large prompt's real token count exceeded `count_tokens`'s
        # estimate enough to overflow the context window even after clamping. Omitting `max_tokens`
        # entirely instead of clamping was tried and reverted: OpenRouter's own docs confirm there is no
        # universal fixed default across models when the field is absent, so widening the reserve keeps
        # the guaranteed, provider-agnostic behavior instead of an unproven cross-model assumption.
        max_tokens = self.fit_max_tokens_to_context(max_tokens, prompt, system)
        rf = self._response_format(json_mode, json_schema)
        body = self._request_body(prompt, system, temperature, max_tokens, rf, thinking, images, tools, tool_choice, extra_body)
        body["stream"] = True
        # OpenAI and several OpenAI-compatible upstreams publish streamed usage only when ``stream_options:
        # {"include_usage": true}`` is set. OpenRouter documents the field as deprecated with no effect: it always
        # sends usage on the final chunk. Kept for the other providers; harmless there.
        body.setdefault("stream_options", {"include_usage": True})
        return body

    def _raise_for_error_in_body(self, payload: dict[str, Any], request: httpx.Request, partial: list[str] | None = None) -> None:
        """Raise the error an HTTP 200 body or stream chunk carries in place of choices; return if it carries none.

        Measured on OpenRouter 2026-09-15: `{"error": {"code": 429, "message": "openai/gpt-5.6-luna is temporarily
        rate-limited upstream"}}` arrived with status 200 and no choices. The buffered path reported it as "returned no
        choices", which the retry predicate does not match, and the stream path skipped the chunk and ended empty - so a
        transient rate limit was neither retried nor readable, and looked like a dead route. A numeric code of 400 or above
        is raised as `httpx.HTTPStatusError` with that status, so the shared policy retries 429/5xx and refuses 400/404 at
        once; anything else is an `LLMProviderError` naming the message. A chunk carrying choices AND an error is the
        documented mid-stream failure and raises `LLMStreamInterruptedError` with the text so far (``partial``); it used
        to be skipped, so a failed stream returned its fragment as a success. See `raise_for_error_in_body`.
        """
        if isinstance(payload.get("error"), dict):
            raise_for_error_in_body(payload, request, self._provider_name, "".join(partial) if partial else "")
