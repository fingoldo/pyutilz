"""Text-to-speech through OpenRouter's ``POST /api/v1/audio/speech`` (ElevenLabs and the other speech models it lists).

The chat providers in this package speak ``/chat/completions`` and return text; speech is a different content type:
the response body is the audio itself, the price is per INPUT CHARACTER (Unicode code points, audio tags included), and
every model has a per-request character cap. This module is deliberately self-contained and synchronous so a plain
``ThreadPoolExecutor`` can drive many concurrent downloads: one :class:`OpenRouterSpeech` owns one ``httpx.Client``
(thread-safe) and one :class:`SpeechAccounting` (lock-protected).

Pricing: ``GET /api/v1/models`` omits speech models unless asked (``?output_modalities=speech``), so the chat catalogue
helpers in ``openrouter_provider`` cannot price them; :func:`speech_catalogue` fetches that listing itself.
``pricing.prompt`` is USD per character for these models.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .exceptions import LLMProviderError

httpx: Any
try:
    import httpx  # type: ignore[no-redef]
except ImportError:  # optional extra 'llm'
    httpx = None

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
#: Per-request input cap in characters (OpenRouter's ElevenLabs announcement). Unknown models are not capped here.
MODEL_MAX_CHARS: Dict[str, int] = {
    "elevenlabs/eleven-v4": 10_000,
    "elevenlabs/eleven-v4-turbo": 10_000,
    "elevenlabs/eleven-v3": 5_000,
    "elevenlabs/eleven-v3-conversational": 5_000,
    "elevenlabs/eleven-multilingual-v2": 10_000,
    "elevenlabs/eleven-flash-v2.5": 40_000,
    "elevenlabs/eleven-flash-v2": 30_000,
    "elevenlabs/eleven-turbo-v2.5": 40_000,
    "elevenlabs/eleven-turbo-v2": 30_000,
}
#: Models whose ``speed`` must stay at the default (the API rejects anything else).
SPEED_REJECTING_MODELS = frozenset({"elevenlabs/eleven-v4", "elevenlabs/eleven-v4-turbo"})
SPEED_RANGE = (0.7, 1.2)
RESPONSE_FORMATS = frozenset({"mp3", "pcm"})
_NON_RETRYABLE = frozenset({400, 401, 403, 404, 405, 410, 413, 422})
_CATALOGUE_TTL_S = 3600.0


class SpeechError(LLMProviderError, RuntimeError):
    """A speech request failed; ``status`` is the HTTP status (0 when none), ``retryable`` whether another try could help."""

    def __init__(self, message: str, status: int = 0, retryable: bool = False, kind: str = "other") -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.kind = kind  # one of error_kind()'s groups, for reporting


class SpeechInputError(LLMProviderError, ValueError):
    """The request is invalid before any network call (empty text, over the model cap, bad speed). Nothing is billed."""


@dataclass(frozen=True)
class SpeechRequest:
    """One synthesis request. Only ElevenLabs-specific fields go under ``provider.options.elevenlabs``."""

    text: str
    model: str
    voice: str
    response_format: str = "mp3"
    speed: Optional[float] = None
    language_code: Optional[str] = None
    seed: Optional[int] = None
    previous_text: Optional[str] = None
    next_text: Optional[str] = None
    voice_settings: Optional[Mapping[str, Any]] = None  # stability, similarity_boost, style, use_speaker_boost
    apply_text_normalization: Optional[str] = None  # "auto" | "on" | "off"
    apply_language_text_normalization: Optional[bool] = None
    pronunciation_dictionary_locators: Optional[Sequence[Mapping[str, Any]]] = None

    def validate(self) -> None:
        """Raise :class:`SpeechInputError` for any input the API would refuse, before a request is sent."""
        if not self.text or not self.text.strip():
            raise SpeechInputError("speech input text is empty")
        cap = MODEL_MAX_CHARS.get(self.model)
        if cap is not None and len(self.text) > cap:
            raise SpeechInputError(f"{self.model} accepts at most {cap} characters per request, got {len(self.text)}")
        if self.response_format not in RESPONSE_FORMATS:
            raise SpeechInputError(f"response_format must be one of {sorted(RESPONSE_FORMATS)}, got {self.response_format!r}")
        if self.speed is not None:
            if self.model in SPEED_REJECTING_MODELS and self.speed != 1.0:
                raise SpeechInputError(f"{self.model} rejects a non-default speed")
            if not SPEED_RANGE[0] <= self.speed <= SPEED_RANGE[1]:
                raise SpeechInputError(f"speed must be within {SPEED_RANGE}, got {self.speed}")

    @property
    def billed_chars(self) -> int:
        """Unicode code points: what ElevenLabs bills (audio tags such as ``[whispering]`` count too)."""
        return len(self.text)

    def body(self) -> Dict[str, Any]:
        """The JSON body for ``/audio/speech``: ElevenLabs options go under ``provider.options.elevenlabs``, unset ones are left out."""
        body: Dict[str, Any] = {"model": self.model, "input": self.text, "voice": self.voice, "response_format": self.response_format}
        options: Dict[str, Any] = {}
        for key in ("seed", "previous_text", "next_text", "language_code", "apply_text_normalization", "apply_language_text_normalization"):
            value = getattr(self, key)
            if value is not None:
                options[key] = value
        if self.pronunciation_dictionary_locators:
            options["pronunciation_dictionary_locators"] = [dict(x) for x in self.pronunciation_dictionary_locators]
        settings = dict(self.voice_settings or {})
        if self.speed is not None:
            settings["speed"] = self.speed
        if settings:
            options["voice_settings"] = settings
        if options:
            body["provider"] = {"options": {"elevenlabs": options}}
        return body

    def params_hash(self) -> str:
        """Stable digest of everything that changes the audio EXCEPT the text (voice, model, format, settings)."""
        body = replace(self, text="", previous_text=None, next_text=None).body()
        body.pop("input", None)
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SpeechResult:
    """One completed synthesis: the audio, the request that produced it, and what it cost."""

    audio: bytes
    request: SpeechRequest
    chars: int
    cost_usd: Optional[float]  # None when the model has no catalogue price: unknown, never silently zero
    latency_s: float
    generation_id: Optional[str] = None
    content_type: Optional[str] = None

    @property
    def nbytes(self) -> int:
        """Size of the returned audio in bytes."""
        return len(self.audio)


@dataclass
class SpeechAccounting:
    """Thread-safe running totals of what a synthesis run spent. ``cost_usd`` covers calls whose price was known."""

    calls: int = 0
    failures: int = 0
    cache_hits: int = 0
    chars: int = 0
    audio_bytes: int = 0
    request_bytes: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    cached_chars: int = 0
    cached_bytes: int = 0
    retries: int = 0
    errors_by_kind: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, result: SpeechResult, request_bytes: int = 0) -> None:
        """Add one completed synthesis to the totals; a call with no known price counts as unpriced, not as free."""
        with self._lock:
            self.calls += 1
            self.chars += result.chars
            self.audio_bytes += result.nbytes
            self.request_bytes += request_bytes
            if result.cost_usd is None:
                self.unpriced_calls += 1
            else:
                self.cost_usd += result.cost_usd

    def record_failure(self) -> None:
        """Count one synthesis that ended in a failure after its retries."""
        with self._lock:
            self.failures += 1

    def record_cache_hit(self, chars: int, nbytes: int) -> None:
        """Count one request served from the cache, with the characters and bytes it would otherwise have cost."""
        with self._lock:
            self.cache_hits += 1
            self.cached_chars += chars
            self.cached_bytes += nbytes

    def record_retry(self) -> None:
        """Count one retry (a new attempt after a retryable failure)."""
        with self._lock:
            self.retries += 1

    def record_attempt_error(self, kind: str) -> None:
        """Count one failed HTTP attempt or transport error under its group (see :func:`error_kind`)."""
        with self._lock:
            self.errors_by_kind[kind] = self.errors_by_kind.get(kind, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        """A copy of every total, taken under the lock so a concurrent ``record`` is never half-visible."""
        with self._lock:
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items() if not k.startswith("_")}


def error_kind(status: int, transport: bool) -> str:
    """Reporting group of one failed attempt: transport, empty_body, http_429, http_402, http_5xx or http_4xx."""
    if transport:
        return "transport"
    if status == 200:
        return "empty_body"
    if status in (402, 429):
        return f"http_{status}"
    return "http_5xx" if status >= 500 else "http_4xx"


_catalogue_lock = threading.Lock()
_catalogue_cache: Dict[str, Any] = {"at": 0.0, "data": {}}


def speech_catalogue(client: Any = None, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Speech models by id from ``/models?output_modalities=speech`` (cached for an hour; empty dict on an outage)."""
    with _catalogue_lock:
        fresh = bool(_catalogue_cache["data"]) and time.monotonic() - _catalogue_cache["at"] < _CATALOGUE_TTL_S
        if not force and fresh:
            return dict(_catalogue_cache["data"])
    data = _fetch_speech_catalogue(client)  # network I/O runs outside the lock: other threads keep reading the cache
    with _catalogue_lock:
        if data is not None:
            _catalogue_cache.update(at=time.monotonic(), data=data)
        return dict(_catalogue_cache["data"])


def _fetch_speech_catalogue(client: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch the speech models from OpenRouter, keyed by id; None when the fetch fails (costs then stay unknown)."""
    http = client if client is not None else httpx.Client(timeout=15.0)
    try:
        resp = http.get(f"{BASE_URL}/models", params={"output_modalities": "speech"})
        resp.raise_for_status()
        return {m["id"]: m for m in resp.json().get("data", ()) if "id" in m}
    except Exception as exc:  # best-effort: an outage leaves prices unknown (cost_usd None, counted), it must not stop synthesis
        logger.warning("speech catalogue fetch failed (%s: %s); costs will be reported as unknown", type(exc).__name__, exc)
        return None
    finally:
        if client is None:
            http.close()


def price_per_char(model: str, client: Any = None) -> Optional[float]:
    """USD per input character for ``model``, or None when the catalogue does not price it."""
    entry = speech_catalogue(client).get(model)
    if not entry:
        return None
    try:
        return float((entry.get("pricing") or {}).get("prompt") or "")
    except (TypeError, ValueError):  # best-effort: a malformed price is reported as unknown, never as zero
        return None


class OpenRouterSpeech:
    """Synchronous, thread-safe OpenRouter speech client with bounded retries and exact accounting."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout_s: float = 180.0,  # upstream times out at 180 s
        max_retries: int = 5,
        backoff_s: float = 2.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        accounting: Optional[SpeechAccounting] = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key not provided")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.accounting = accounting if accounting is not None else SpeechAccounting()
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=httpx.Timeout(timeout_s, connect=10.0))

    def __enter__(self) -> "OpenRouterSpeech":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client, but only when this object created it."""
        if self._owns_client:
            self._client.close()

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        """Synthesize one request. Raises :class:`SpeechInputError` before any call, :class:`SpeechError` after retries."""
        request.validate()
        payload = json.dumps(request.body(), ensure_ascii=False).encode("utf-8")
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            retry_after: Optional[float] = None
            transport_error: Optional[Exception] = None
            resp: Any = None
            try:
                resp = self._client.post(f"{self.base_url}/audio/speech", content=payload, headers={"Content-Type": "application/json"})
            except Exception as exc:  # transport errors and timeouts are transient; carried into the failure message below
                transport_error = exc
            if transport_error is not None:
                message, status, retryable = f"speech request failed: {type(transport_error).__name__}: {transport_error}", 0, True
            else:
                if resp.status_code == 200 and resp.content:
                    return self._finish(request, resp, time.perf_counter() - started, len(payload))
                status = resp.status_code
                retryable = status not in _NON_RETRYABLE and (status in (402, 429) or status >= 500 or status == 200)
                message = f"speech request returned HTTP {status}: {resp.text[:300] if status != 200 else 'empty audio body'}"
                retry_after = _retry_after(resp)
            kind = error_kind(status, transport_error is not None)
            self.accounting.record_attempt_error(kind)
            if not retryable or attempt > self.max_retries:
                self.accounting.record_failure()
                raise SpeechError(message, status, retryable, kind=kind)
            delay = retry_after if retry_after is not None else self.backoff_s * 2 ** (attempt - 1) * (0.5 + random.random())
            logger.warning("speech attempt %d failed (%s); retrying in %.1fs", attempt, message, delay)
            self.accounting.record_retry()
            self._sleep(delay)

    def _finish(self, request: SpeechRequest, resp: Any, latency: float, request_bytes: int) -> SpeechResult:
        """Build the result for a successful response, price it from the catalogue and record it in the accounting."""
        rate = price_per_char(request.model, self._client)
        result = SpeechResult(
            audio=resp.content,
            request=request,
            chars=request.billed_chars,
            cost_usd=None if rate is None else rate * request.billed_chars,
            latency_s=latency,
            generation_id=resp.headers.get("x-generation-id"),
            content_type=resp.headers.get("content-type"),
        )
        self.accounting.record(result, request_bytes)
        return result


def _retry_after(resp: Any) -> Optional[float]:
    """The seconds a ``Retry-After`` header asks for, floored at zero; None when absent or not a number."""
    raw = resp.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:  # best-effort: an unparseable Retry-After falls back to exponential backoff
        return None
