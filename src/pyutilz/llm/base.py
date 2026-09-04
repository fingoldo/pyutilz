"""Abstract LLM provider interface."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import weakref
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable

from pyutilz.llm.exceptions import (
    JSONParsingError,
    LLMRefusalError,
)

logger = logging.getLogger(__name__)


class PerCallAttr:
    """Descriptor for a provider's "last successful call" state (e.g. ``last_tool_calls``,
    ``_last_usage``), backed by a private ``contextvars.ContextVar`` per (instance, attribute).

    Regression fix (2026-07-21 audit round 2, HIGH): these used to be plain instance attributes,
    written unconditionally at the end of every ``generate()`` call. ``generate_batch()`` fires
    N concurrent ``self.generate()`` calls on ONE shared, cached provider instance (see
    ``llm.factory.get_llm_provider``'s whole reason for existing) via ``asyncio.create_task`` --
    a write from one in-flight task was visible to every other task reading the same plain
    attribute, so a caller reading e.g. ``provider.last_tool_calls`` right after a batch item is
    yielded could silently get a DIFFERENT request's data (verified with a standalone repro:
    ``generate_batch()`` yielded ``id='req-0'`` while ``provider.last_tool_calls`` already
    reflected ``id='req-2'``).

    ``asyncio.create_task()`` gives each Task its own COPY of the current context, so a
    ``contextvars.ContextVar`` set inside one task is invisible to every other task -- this
    closes the cross-task race. A direct (non-batched) ``await provider.generate(...)`` keeps
    working exactly as before: no task boundary is crossed, so the write and the caller's
    immediately-following read share the same context.

    Does NOT fix (by design -- this is the correct, intentional behavior): the outer caller of
    ``generate_batch()`` runs in yet another context than any individual request task, so it can
    no longer read a completed batch item's metadata off the provider instance at all -- it must
    come from the yielded dict instead (see ``generate_batch``'s ``usage``/``tool_calls``/etc.
    keys), which is exactly the fix the audit's own report recommended.
    """

    def __init__(self, default_factory: Callable[[], Any]) -> None:
        self._default_factory = default_factory
        self._name = "_unnamed"
        # One ContextVar per DESCRIPTOR (i.e. per class attribute), created here rather than
        # lazily per (instance, attribute) on first touch. Two independent bugs are closed by
        # that single change:
        #  * the lazy per-instance creation was a check-then-act with no lock, so two threads
        #    first touching the same attribute on one shared (factory-cached) provider could each
        #    build a DISTINCT ContextVar; the loser wrote to a var no longer reachable and its own
        #    next read fell through to the default -- indistinguishable from "the model returned
        #    nothing" (2026-09-03 audit F10, reproduced in 287 of 3000 threaded trials).
        #  * CPython documents ContextVars as objects to create at module/class scope and never
        #    inside a function, because live Context objects keep strong references to them. One
        #    var per instance meant a service constructing a provider per request (the
        #    unhashable-kwargs path in llm.factory bypasses the LRU) minted vars nothing could
        #    reclaim (F41). The count is now bounded by the number of declared class attributes.
        # The var's VALUE is a per-instance mapping keyed by id(), validated through a weakref so
        # a recycled id can never surface a dead instance's state. It is replaced copy-on-write on
        # every set, never mutated in place, so a value set inside an asyncio Task stays invisible
        # to every other Task -- the whole point of this descriptor.
        self._var: contextvars.ContextVar[dict[int, Any]] = contextvars.ContextVar("PerCallAttr._unnamed")

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name
        self._var = contextvars.ContextVar(f"{owner.__name__}.{name}")

    def default(self) -> Any:
        """Return a freshly built default value for this attribute (used by ``_reset_per_call_state``)."""
        return self._default_factory()

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        entry = self._var.get({}).get(id(instance))
        if entry is not None:
            ref, value = entry
            if ref() is instance:
                return value
        return self._default_factory()

    def __set__(self, instance: Any, value: Any) -> None:
        ref: Callable[[], Any]
        try:
            ref = weakref.ref(instance)
        except TypeError:  # instance's type forbids weak references
            ref = _StrongRef(instance)
        store = dict(self._var.get({}))
        store[id(instance)] = (ref, value)
        self._var.set(store)


class _StrongRef:
    """``weakref.ref``-compatible fallback for instances whose type disallows weak references."""

    __slots__ = ("_obj",)

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __call__(self) -> Any:
        return self._obj


class LazySemaphore:
    """Descriptor for a provider's concurrency-limiting ``asyncio.Semaphore``, constructed lazily
    on first access rather than eagerly in ``__init__``.

    On Python 3.9, ``asyncio.Semaphore.__init__`` eagerly calls ``asyncio.get_event_loop()`` to
    bind a loop; 3.10+ removed that eager bind in favour of binding lazily on first
    ``acquire()``/``release()``. A provider is normally constructed synchronously (e.g. via
    ``llm.factory.get_llm_provider``, in a sync test, or at import time) with no loop running, so
    the eager bind raised ``RuntimeError: There is no current event loop`` on 3.9 only -- every
    other supported version masked the bug. Deferring construction to first ``__get__`` (which
    only happens inside ``async with self.semaphore:``, i.e. while a loop IS running) reproduces
    3.10+'s lazy-bind behavior on 3.9 too. Backed by instance ``__dict__`` so a test can still
    assign a replacement semaphore directly (``provider.semaphore = asyncio.Semaphore(1)``).
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        value = instance.__dict__.get(self._name)
        if value is None:
            value = asyncio.Semaphore(instance._max_concurrent)
            instance.__dict__[self._name] = value
        return value

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self._name] = value


def _longest_prefix_pricing(
    model: str,
    pricing_table: dict[str, tuple[float, float]],
    default: tuple[float, float],
    provider_label: str = "LLM",
) -> tuple[float, float]:
    """Return ``(input_per_1m, output_per_1m)`` for ``model`` from ``pricing_table``.

    Resolution order:
      1. Exact key match.
      2. LONGEST matching prefix — a more specific tier wins over a shorter,
         overlapping one. Dict iteration order is arbitrary, so a naive
         first-``startswith`` match can silently mis-price (e.g. a future
         ``claude-opus-4-2-YYYYMMDD`` inheriting ``claude-opus-4-7``'s cheaper
         tier just because that entry iterated first). We always compare
         prefix lengths, mirroring the algorithm used by every provider.
      3. ``default`` fallback (with a single warning).

    Two prefix forms are tried, most-precise first, so both the
    Anthropic (date-suffixed KEY, date-less model) and the Gemini
    (semantic-suffixed keys like ``...-flash`` vs ``...-flash-lite``) naming
    schemes resolve correctly:

      a. Longest FULL key that ``model`` starts with (handles date-suffixed
         model ids against date-less keys, and keeps ``-flash-lite`` from
         being captured by the shorter ``-flash`` entry).
      b. If none, longest key with its trailing ``-<segment>`` dropped
         (``key.rsplit("-", 1)[0]``) — lets a date-less canonical model id
         (``claude-opus-4-6``) match a date-suffixed key
         (``claude-opus-4-6-20250610``).
    """
    exact = pricing_table.get(model)
    if exact:
        return exact
    # (a) longest full-key prefix.
    best_val: tuple[float, float] | None = None
    best_len = -1
    for key, val in pricing_table.items():
        if model.startswith(key) and len(key) > best_len:
            best_len = len(key)
            best_val = val
    # (b) fall back to trailing-segment-trimmed prefixes.
    if best_val is None:
        for key, val in pricing_table.items():
            prefix = key.rsplit("-", 1)[0]
            # The trimmed prefix must still name a MODEL, not just a vendor family: every xAI key
            # trims to a bare ``grok``, so any future ``grok-5`` matched some arbitrary row and
            # was priced from it, with the unknown-model warning unreachable inside the vendor's
            # own namespace (2026-09-03 audit F19). Requiring a surviving ``-`` keeps the cases
            # this stage exists for (``claude-opus-4-6-20250610`` -> ``claude-opus-4-6``) and
            # rejects the vendor-prefix degenerate one.
            if "-" in prefix and model.startswith(prefix) and len(prefix) > best_len:
                best_len = len(prefix)
                best_val = val
    if best_val is not None:
        logger.warning(
            "%s model %r not pinned in pricing table; falling back to the "
            "longest-prefix match (%s/%s per 1M). Pin its exact id to avoid "
            "silent mispricing.",
            provider_label, model, best_val[0], best_val[1],
        )
        return best_val
    return default


def longest_prefix_lookup(model: str, table: dict[str, Any], default: Any) -> Any:
    """Same longest-matching-prefix resolution algorithm as :func:`_longest_prefix_pricing`
    (exact match, then longest full-key prefix, then longest trailing-segment-trimmed prefix),
    but for an arbitrary per-model VALUE table (e.g. max-output-token limits) rather than a
    fixed ``(input, output)`` pricing pair. Silent, no warning log -- callers wanting the
    mispricing-style warning should log it themselves.
    """
    exact = table.get(model)
    if exact is not None:
        return exact
    best_val = None
    best_len = -1
    for key, val in table.items():
        if model.startswith(key) and len(key) > best_len:
            best_len = len(key)
            best_val = val
    if best_val is None:
        for key, val in table.items():
            prefix = key.rsplit("-", 1)[0]
            # The trimmed prefix must still name a MODEL, not just a vendor family: every xAI key
            # trims to a bare ``grok``, so any future ``grok-5`` matched some arbitrary row and
            # was priced from it, with the unknown-model warning unreachable inside the vendor's
            # own namespace (2026-09-03 audit F19). Requiring a surviving ``-`` keeps the cases
            # this stage exists for (``claude-opus-4-6-20250610`` -> ``claude-opus-4-6``) and
            # rejects the vendor-prefix degenerate one.
            if "-" in prefix and model.startswith(prefix) and len(prefix) > best_len:
                best_len = len(prefix)
                best_val = val
    return best_val if best_val is not None else default


# ── Refusal detection ─────────────────────────────────────────────────────
# Patterns for common LLM refusal sentinels across providers (Anthropic,
# OpenAI, Gemini). Intentionally conservative — we only match phrases that
# BEGIN a refusal ("I cannot help", "I won't do that"), not content words
# that might appear inside a normal answer. False positives here silently
# degrade a valid evaluation to a fallback, so err on the strict side.
_REFUSAL_PATTERNS = [
    re.compile(r"\bI (cannot|can't|am unable|am not able) (help|assist|comply)\b", re.I),
    re.compile(r"\bI (will not|won't) (help|do that|comply)\b", re.I),
    re.compile(r"\bI'm (not able|unable) to (help|process|assist)\b", re.I),
    re.compile(r"\bI (cannot|can't) (provide|generate|create|produce) (that|this)\b", re.I),
]


def normalize_thinking(thinking: bool | str | int) -> tuple[bool, str | None]:
    """Normalise a ``thinking=`` argument into ``(enabled, effort)``.

    - ``False`` / empty string -> ``(False, None)`` (explicitly off)
    - ``True`` -> ``(True, None)`` (on, provider picks its default effort)
    - non-empty str -> ``(True, str.lower())`` (on with an explicit effort)

    Lives here rather than on one provider class because every provider that
    supports reasoning has to agree on what the caller's argument MEANS, even
    though each maps it to a different request field (OpenRouter's
    ``reasoning.effort``, DeepSeek's ``thinking.type``, Anthropic's
    ``thinking.budget_tokens``). A second copy of this contract is how the two
    halves drift apart.
    """
    if thinking is False or thinking == "":
        return (False, None)
    if thinking is True:
        return (True, None)
    if isinstance(thinking, str):
        return (True, thinking.lower())
    # The annotation admits int precisely because callers do pass one (0 -> (False, None));
    # a narrower `bool | str` made this line read as dead code when it is reached and covered.
    return (bool(thinking), None)


def is_llm_refusal(text: str) -> bool:
    """Return True if ``text`` contains a recognizable LLM refusal sentinel.

    Used by parsers / retry loops to raise :class:`LLMRefusalError` instead
    of retrying blindly — the same prompt will refuse again.
    """
    if not text or not isinstance(text, str):
        return False
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


def _require_json_object(obj: Any, provider_name: str) -> dict[str, Any]:
    """Return ``obj`` when it is a JSON object, else raise :class:`JSONParsingError`.

    ``extract_json`` is annotated ``-> dict[str, Any]`` and its callers index the result. A model
    asked for "valid JSON only" very often answers with a top-level array (or a bare string or
    number), which parses fine but breaks that contract: the caller's first ``result["field"]``
    then raised ``TypeError`` far from the parse site, past the retry layer that catches
    JSONParsingError.
    """
    if isinstance(obj, dict):
        return obj
    raise JSONParsingError(f"Expected a JSON object from {provider_name}, got {type(obj).__name__}")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    semaphore = LazySemaphore()

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens this provider/model supports."""
        return 4096  # safe fallback

    @property
    def context_window(self) -> int:
        """Maximum context window (input + output) in tokens."""
        return 200_000  # safe default for most models

    def supports_json_schema(self) -> bool:
        """Return True if this provider+model accepts a STRICT JSON schema, not merely JSON mode.

        ``response_format={"type": "json_object"}`` only promises syntactically valid JSON — the model
        stays free to invent fields and enum values. ``{"type": "json_schema", "json_schema": {...,
        "strict": true}}`` constrains generation to the schema, which is what makes a closed enum (a
        fixed relation vocabulary, a fixed set of severity levels) impossible to violate rather than
        merely discouraged by the prompt.

        Default: False — a provider must opt in, since sending the param to an upstream that ignores it
        gives a false sense of guarantee. Subclasses override (OpenAI-compatible endpoints; OpenRouter
        consults its catalogue per model).
        """
        return False

    # Flat headroom left for the chat envelope (role/formatting tokens), on top of the proportional
    # tokeniser-error margin below — see ``_context_reserve_tokens`` for why a flat constant alone is
    # not enough.
    _CONTEXT_RESERVE_TOKENS = 1024

    # ``count_tokens``'s own tokeniser is an APPROXIMATION of whatever the real upstream uses per model,
    # and the two can disagree by a large ABSOLUTE margin on a large prompt even when they agree closely
    # on a small one. Measured 2026-08-08 (autopsia, OpenRouter/deepseek-v3.2): a ~23.6k-token real prompt
    # was undercounted by ``count_tokens`` at ~20.9k - a ~2.75k-token gap, 2.7x the flat 1024-token
    # reserve alone. The resulting clamp forwarded a `max_tokens` that, added to the REAL input count,
    # exceeded the context window by ~1.7k tokens, and the upstream rejected the whole call with HTTP 400
    # BEFORE generating anything - clamping to fit the window is worthless if the margin it clamps by is
    # itself smaller than the counting error it exists to absorb. A fixed fraction of the estimated input
    # (not a fixed token count) scales the margin with the very thing that is under-measured.
    #
    # The fraction is set from the WORST undercount measured so far, not from the average, because a single
    # underestimate is enough to lose the whole call to an HTTP 400. Two real incidents, both expressed as a
    # fraction of what ``count_tokens`` returned: 13.2% (2026-08-08, deepseek-v3.2, 20,864 estimated vs 23,617
    # real) and 20.9% (2026-08-26, gpt-oss-120b, 19,920 estimated vs 24,077 real - the 0.15 in force at the
    # time was BELOW this and the call was rejected). 0.30 clears the worse of the two by ~44%. Raising it is
    # close to free: it only binds when the input is large, and even then the surviving budget stays far above
    # any output these callers actually request - whereas being a few hundred tokens short costs the call.
    _CONTEXT_RESERVE_FRACTION = 0.30

    def _context_reserve_tokens(self, input_tokens: int) -> int:
        """Headroom for one `fit_max_tokens_to_context` clamp: the larger of the flat envelope reserve
        and a fraction of the estimated input length, so the margin grows with the prompt instead of
        staying fixed while the tokeniser-estimation error it must absorb grows right along with it."""
        return max(self._CONTEXT_RESERVE_TOKENS, int(input_tokens * self._CONTEXT_RESERVE_FRACTION))

    def fit_max_tokens_to_context(self, max_tokens: int, prompt: str, system: str | None = None) -> int:
        """Clamp an output budget so ``input + output`` fits the model's context window.

        ``max_output_tokens`` is a per-model output ceiling that can EXCEED
        ``context_window - input`` — llama-3.3-70b, for instance, advertises a 128k output cap inside
        a 131k window, so requesting the full cap alongside any real prompt makes the upstream reject
        the whole call (HTTP 400 "maximum context length is ... you requested ...") before generating
        a single token. Providers therefore clamp here instead of forwarding an impossible budget.

        Returns ``max_tokens`` unchanged when it already fits, or when the prompt alone leaves no
        usable room — in that case the upstream's own context error is the more informative signal
        than a silently truncated budget.
        """
        if max_tokens <= 0:
            return max_tokens
        ctx = self.context_window
        if ctx <= 0:
            return max_tokens
        from pyutilz.llm.token_counter import count_tokens

        # Providers name the active model differently (``model_name`` on OpenAI-compat/Gemini,
        # ``model`` on Anthropic); resolved explicitly rather than via `or`, which would also swallow
        # a legitimately empty value.
        model = getattr(self, "model_name", "")
        if not isinstance(model, str) or not model:
            fallback = getattr(self, "model", "")
            model = fallback if isinstance(fallback, str) else ""
        input_tokens = count_tokens(prompt, model=model) + (count_tokens(system, model=model) if system else 0)
        room = ctx - input_tokens - self._context_reserve_tokens(input_tokens)
        if room <= 0 or room >= max_tokens:
            return max_tokens
        logger.debug(
            "Clamping max_tokens %d -> %d to fit context window (%d) after ~%d input tokens",
            max_tokens,
            room,
            ctx,
            input_tokens,
        )
        return room

    def supports_json_mode(self) -> bool:
        """Return True if this provider+model accepts a structured
        JSON-mode parameter (``response_format={"type":"json_object"}``,
        ``response_mime_type="application/json"``, or equivalent).

        Callers should consult this before sending ``json_mode=True`` as
        a kwarg to ``generate()`` — passing it to a provider that
        doesn't support it either silently no-ops or raises (depending
        on the upstream). For OpenRouter-routed models, support varies
        per model; the override consults the catalogue's
        ``supported_parameters``.

        Default: False. Subclasses that support a hard JSON-mode toggle
        override to True (or model-aware logic). Soft "system-prompt
        hint" implementations (claude-code) keep False — the hint helps
        but doesn't guarantee, and callers may want a stronger signal.
        """
        return False

    @staticmethod
    def extract_json(text: str, provider_name: str = "LLM") -> dict[str, Any]:
        """Extract and parse JSON from LLM response, handling code blocks.

        Strategy: prefer markdown-fenced JSON, then scan the body with
        ``json.JSONDecoder.raw_decode`` from each ``{`` candidate. The
        scan-based approach replaces brittle regexes — a lazy ``\\{...?\\}``
        stops at the first ``}`` even mid-object, while a greedy form
        crosses multiple objects. ``raw_decode`` knows JSON's grammar.

        Args:
            text: Raw LLM response that may contain JSON in code blocks.
            provider_name: Name of the provider for error messages.

        Returns:
            Parsed JSON dict.

        Raises:
            JSONParsingError: If JSON parsing fails, or if the parsed value is not an object
                (a top-level array/string/number is very common under "respond with valid JSON
                only", and returning it would break the ``-> dict[str, Any]`` contract far from
                the parse site, past the retry layer that catches JSONParsingError).
        """

        try:
            text = text.strip()

            # 1. Markdown-fenced JSON wins. The outer fences delimit the
            #    payload unambiguously, so we extract via regex (only the
            #    wrapper is regex; the JSON itself is parsed properly).
            fence_match = re.search(
                r"```(?:json)?\s*\n?(\{.*?\}|\[.*?\])\s*\n?```",
                text,
                re.DOTALL,
            )
            if fence_match:
                # A fenced payload that does not parse must DEGRADE into paths 2-5 rather than
                # abort the whole extraction (2026-09-03 audit F13): models routinely emit a
                # broken sketch in a fence followed by the real, clean object. Only the decode
                # error is swallowed -- _require_json_object's raise still propagates, so a fenced
                # top-level array remains a hard error exactly as before.
                try:
                    return _require_json_object(json.loads(fence_match.group(1)), provider_name)
                except json.JSONDecodeError:
                    pass

            # 2. Strip leading fence even without a closing fence (some
            #    streaming LLMs forget to close).
            stripped = text
            if stripped.startswith("```json"):
                stripped = stripped[7:]
            elif stripped.startswith("```"):
                stripped = stripped[3:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()

            # 3. Try the whole stripped text first (cheap path: most
            #    json_mode= responses are pure JSON).
            try:
                return _require_json_object(json.loads(stripped), provider_name)
            except json.JSONDecodeError:
                pass

            # 4. Scan for the first parseable JSON object via raw_decode.
            #    Walks each ``{`` candidate, asks the decoder to consume
            #    a single JSON value, and returns the first one that
            #    parses cleanly. Robust against prose-before-JSON,
            #    JSON-with-trailing-prose, and nested-object boundaries.
            decoder = json.JSONDecoder()
            for i, ch in enumerate(stripped):
                if ch != "{":
                    continue
                try:
                    obj, _end = decoder.raw_decode(stripped, i)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    return obj

            # 5. Last resort — re-raise via the original strict parse so
            #    the JSONDecodeError handler below produces a clean error.
            return _require_json_object(json.loads(stripped), provider_name)
        except json.JSONDecodeError as e:
            # Before reporting as malformed JSON, check whether the model
            # simply refused to answer — that's a distinct error class with
            # a distinct retry policy (do NOT retry refusals).
            if is_llm_refusal(text):
                logger.warning(
                    "%s refused to answer (no JSON): %.200s",
                    provider_name, text,
                )
                raise LLMRefusalError(
                    f"{provider_name} refused to produce JSON",
                    raw_text=text,
                )
            logger.error("Failed to parse JSON from %s: %s\nResponse: %s", provider_name, e, text)
            raise JSONParsingError(f"Invalid JSON response from {provider_name}: {e}")

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (0 = use provider max).

        Returns:
            Generated text.
        """
        pass

    # ── shared implementation hooks ──────────────────────────────────
    # Subclasses that use the default estimate_cost / _get_pricing below
    # supply a per-1M pricing table and default via these class attrs.
    _PRICING: dict[str, tuple[float, float]] = {}  # noqa: RUF012 -- intentional shared class-level pricing table (subclasses override with their own), not a per-instance mutable-default bug
    _DEFAULT_PRICING: tuple[float, float] = (0.0, 0.0)

    def _pricing_model_id(self) -> str:
        """Return the model id used for pricing lookup.

        Providers store the current model under different attribute names
        (``self.model`` on Anthropic, ``self.model_name`` on Gemini /
        OpenAI-compat). Resolve either so the shared ``_get_pricing`` works
        for both without per-provider overrides.
        """
        return str(getattr(self, "model", None) or getattr(self, "model_name", ""))

    def _get_pricing(self) -> tuple[float, float]:
        """Return ``(input_per_1m, output_per_1m)`` for the current model.

        Default uses longest-prefix matching against ``self._PRICING`` with
        ``self._DEFAULT_PRICING`` as fallback. Override for bespoke pricing.
        """
        return _longest_prefix_pricing(
            self._pricing_model_id(),
            self._PRICING,
            self._DEFAULT_PRICING,
            provider_label=self.__class__.__name__,
        )

    @property
    def _provider_display_name(self) -> str:
        """Human-readable provider name used in JSON-parse error messages."""
        return getattr(self, "_provider_name", self.__class__.__name__)

    # Attribute names read by ``_capture_percall_metadata`` -- the union of every "last
    # successful call" attribute any built-in provider sets (openai_compat: usage/tool_calls/
    # citations/finish_reason; anthropic: + cache/thinking tokens; gemini: + safety_ratings/
    # grounding/function_calls). Missing on a given provider -> simply not included, via getattr's
    # default; harmless for providers that don't set some of these (e.g. Anthropic has no
    # ``last_tool_calls``).
    _PERCALL_METADATA_ATTRS: tuple[str, ...] = (
        "_last_usage",
        "_last_finish_reason",
        "last_tool_calls",
        "last_citations",
        "last_cache_creation_input_tokens",
        "last_cache_read_input_tokens",
        "last_thinking_tokens",
        "last_thinking_tokens_estimated",
        "last_safety_ratings",
        "last_grounding_metadata",
        "last_citation_metadata",
        "last_function_calls",
        "last_cached_content_tokens",
        "last_all_candidates",
        # Response-scoped too, despite reading like instance-wide "most recent response" state:
        # both are written from EVERY response on a shared, factory-cached provider, so during a
        # generate_batch() ``check_account_limits()`` used to return whichever request answered
        # last -- plausibly a failed request's 429 window, attributed to the batch as a whole
        # (2026-09-03 audit F32).
        "last_rate_limits",
        "last_organization_id",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register every ``PerCallAttr`` the subclass declares into ``_PERCALL_METADATA_ATTRS``.

        The tuple used to be hand-maintained, and every mechanism keyed off it -- the per-call
        reset and the batch snapshot -- silently skipped whatever a provider author forgot to add.
        That is not a fault a test can catch at the moment the attribute is written, so the union
        is computed here instead: declaring the descriptor IS the registration, and an explicit
        tuple entry only fixes the ORDER of names that are already known. Names inherited from a
        base class are already in the resolved tuple and are not duplicated.
        """
        super().__init_subclass__(**kwargs)
        names: list[str] = list(getattr(cls, "_PERCALL_METADATA_ATTRS", ()))
        seen = set(names)
        for klass in cls.__mro__:
            for name, value in vars(klass).items():
                if isinstance(value, PerCallAttr) and name not in seen:
                    seen.add(name)
                    names.append(name)
        cls._PERCALL_METADATA_ATTRS = tuple(names)

    def _reset_per_call_state(self) -> None:
        """Reset every ``PerCallAttr`` this class declares to its default, at the START of every
        ``generate()`` / ``generate_stream()``.

        DERIVED from ``_PERCALL_METADATA_ATTRS`` rather than hand-written per provider: the
        hand-written variants drifted (``last_json_mode_fallback`` was declared and listed but
        never reset, so it latched ``True`` for the rest of the context -- 2026-09-03 audit F09),
        and three providers never got the hook at all (F31). Iterating the tuple means a newly
        declared per-call attribute is reset the moment it is registered for batch capture, with
        no second list to keep in sync. Names not declared as a ``PerCallAttr`` on this class are
        skipped (the tuple is the UNION across providers). Cumulative ``total_*`` counters are
        deliberately NOT reset.
        """
        cls = type(self)
        for name in self._PERCALL_METADATA_ATTRS:
            descriptor = getattr(cls, name, None)
            if isinstance(descriptor, PerCallAttr):
                setattr(self, name, descriptor.default())

    def _capture_percall_metadata(self) -> dict[str, Any]:
        """Snapshot this provider's "last successful call" attributes, called immediately after
        ``await self.generate(...)`` returns -- i.e. still within the SAME asyncio Task/context
        that just set them, so this read is race-free regardless of how many other concurrent
        ``generate_batch()`` tasks are in flight on the same shared provider instance. The
        snapshot is returned for inclusion in the per-request result dict -- reading the
        instance attribute again later, from a DIFFERENT task/context (e.g. the caller iterating
        ``generate_batch()``'s yielded results), is exactly the race this exists to avoid.

        Keys drop both the leading underscore and the ``last_`` prefix (``_last_usage`` ->
        ``usage``, ``last_tool_calls`` -> ``tool_calls``) -- "last" describes the now-superseded
        shared-instance-attribute semantics, not this per-request snapshot.
        """
        out: dict[str, Any] = {}
        for name in self._PERCALL_METADATA_ATTRS:
            if hasattr(self, name):
                key = name.lstrip("_")
                if key.startswith("last_"):
                    key = key[len("last_") :]
                out[key] = getattr(self, name)
        return out

    def _classify_batch_exception(self, exc: Exception) -> dict[str, Any] | None:
        """Return extra fields to merge into a per-request batch error dict.

        Hook for provider-specific exception classification (e.g. Gemini
        tags safety blocks with ``error_type="safety_block"``). Return
        ``None`` for the generic path. The base ``generate_batch`` always
        catches every exception per-request so one bad request never aborts
        the whole batch.
        """
        return None

    async def _generate_json_via(self, prompt: str, system: str | None, temperature: float, max_tokens: int, **generate_kwargs: Any) -> dict[str, Any]:
        """Shared ``generate_json`` body: steer the system prompt toward JSON-only output, call
        ``self.generate`` (forwarding any provider-specific kwargs, e.g. OpenAI-compat's
        ``json_mode``/``json_schema``), then parse via ``extract_json``.

        Factored out of ``generate_json`` so a provider that needs to forward extra kwargs to
        ``generate`` (``OpenAICompatibleProvider.generate_json``) can reuse this exact body
        instead of duplicating it near-verbatim -- a duplicate-function-body-subset finding in
        the 2026-08-04 code-audit pass caught the two copies drifting only in which kwargs got
        forwarded.
        """
        json_system = (system or "") + "\n\nRespond with valid JSON only."
        text = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
            **generate_kwargs,
        )
        return self.extract_json(text, self._provider_display_name)

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output.

        Appends a "respond with valid JSON only" steer to the system prompt,
        calls ``generate``, then parses via ``extract_json``. Providers with a
        hard JSON-mode toggle (OpenAI-compat) override to pass it through.
        """
        # `images` is forwarded ONLY when non-empty. A provider that does not implement vision has
        # no `images` parameter on its `generate`, and passing `images=None` to it would be a
        # TypeError on every ordinary text call -- so the default path must not mention it at all.
        extra = {"images": images} if images else {}
        return await self._generate_json_via(prompt, system, temperature, max_tokens, **extra)

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate responses for multiple requests concurrently.

        Every per-request failure (including a malformed request dict raising
        ``KeyError`` on ``req["prompt"]``) becomes a per-request ``error``
        entry rather than aborting the whole batch. Provider-specific error
        classification is delegated to ``_classify_batch_exception``.

        Yields:
            Response dicts with ``id`` and either ``result`` or ``error``. On success, also
            includes this request's own ``usage``/``tool_calls``/``citations``/``finish_reason``
            (etc., provider-dependent) metadata -- captured within this request's own task, so
            it is NOT subject to the cross-task "last_*" attribute race (see ``PerCallAttr``).
        """
        async def process_request(req: dict) -> dict[str, Any]:
            """Run a single batch request via ``generate``, returning a result/error dict tagged with its id."""
            request_id = req.get("id", "unknown")
            try:
                result = await self.generate(
                    prompt=req["prompt"],
                    system=req.get("system"),
                    temperature=req.get("temperature", 0.7),
                    # 0 == "use the model's own max_output_tokens, clamped to the context window",
                    # the same default generate() applies. The former hard-coded 1024 silently
                    # truncated batch responses that the identical single call returned in full
                    # (2026-09-03 audit F39).
                    max_tokens=req.get("max_tokens", 0),
                )
                out = {"id": request_id, "result": result}
                out.update(self._capture_percall_metadata())
                return out
            except Exception as e:
                logger.error("Batch request %s failed: %s", request_id, e)
                out = {"id": request_id, "error": str(e)}
                extra = self._classify_batch_exception(e)
                if extra:
                    out.update(extra)
                return out

        # Wrap as Tasks explicitly. ``asyncio.as_completed`` over raw
        # coroutines emits a DeprecationWarning in 3.11 and breaks in 3.12+
        # — feeding Tasks instead works across versions.
        tasks = [asyncio.create_task(process_request(req)) for req in requests]
        try:
            for coro in asyncio.as_completed(tasks):
                yield await coro
        finally:
            # If the caller stops consuming early (a `break`, an exception in the loop body, or
            # simply not draining the generator), Python delivers GeneratorExit at the `yield`
            # above -- without this, every already-scheduled-but-not-yet-completed task keeps
            # running to completion in the background, invisibly making real (billable) LLM API
            # calls the caller has no way to observe or cancel.
            for t in tasks:
                if not t.done():
                    t.cancel()

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD for token counts.

        Default multiplies token counts by the per-1M rates from
        ``_get_pricing``. Providers with cache-tier or per-token-map pricing
        (OpenAI-compat) override.
        """
        inp_rate, out_rate = self._get_pricing()
        return (input_tokens / 1_000_000) * inp_rate + (output_tokens / 1_000_000) * out_rate

    async def get_account_credits(self) -> dict[str, Any]:
        """Return account billing snapshot.

        Common keys (all optional — provider-dependent presence):
            ``balance_usd``     — remaining credit in USD (None when not exposed)
            ``total_granted``   — total ever-granted credit in USD
            ``total_used``      — lifetime spend in USD
            ``is_available``    — whether the account can issue billable calls
            ``currency``        — native currency if not USD
            ``raw``             — provider's full raw response

        Default raises :class:`NotImplementedError` — most upstream APIs
        don't publish a balance endpoint for regular (non-admin) keys.
        Concrete providers override when their API exposes it.

        Raises:
            NotImplementedError: provider doesn't expose balance via API.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not expose account balance via API. " "Check the provider's web console.")

    async def check_account_limits(self) -> dict[str, Any]:
        """Return account-level rate limit / quota / usage snapshot.

        Default raises :class:`NotImplementedError`. 6 of 7 concrete providers override this
        (only DeepSeek relies on the inherited OpenAI-compat header-fallback without its own
        override): OpenRouter has a real dedicated-endpoint implementation; Anthropic and
        DeepSeek fall back to captured rate-limit response headers; OpenAI, xAI, and Gemini
        deliberately re-raise :class:`NotImplementedError` even though OpenAI/xAI already capture
        the same headers internally; Claude Code shells out to the CLI (no HTTP headers to
        capture at all).

        Common keys (provider-dependent):
            ``limit_remaining`` — credits left under the cap (USD)
            ``usage_daily/weekly/monthly``
            ``rate_limit``      — requests-per-interval shape
            ``is_free_tier``
            ``raw``             — provider's full raw response

        Raises:
            NotImplementedError: provider doesn't expose this via API.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not expose account limits via API. " "Check the provider's web console.")

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        pass
