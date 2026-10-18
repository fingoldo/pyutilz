"""Text-to-speech through OpenRouter's ``POST /api/v1/audio/speech`` (ElevenLabs and the other speech models it lists).

The chat providers in this package speak ``/chat/completions`` and return text; speech is a different content type:
the response body is the audio itself, the price is per INPUT CHARACTER (Unicode code points, audio tags included), and
every model has a per-request character cap. This module is deliberately self-contained and synchronous so a plain
``ThreadPoolExecutor`` can drive many concurrent downloads: one :class:`OpenRouterSpeech` owns one ``httpx.Client``
(thread-safe) and one :class:`SpeechAccounting` (lock-protected).

Pricing: ``GET /api/v1/models`` omits speech models unless asked (``?output_modalities=speech``), so the chat catalogue
helpers in ``openrouter_provider`` cannot price them; :func:`speech_catalogue` fetches that listing itself.
``pricing.prompt`` is USD per character for these models.

Retry policy: deliberately not the chat providers' ``_retry.py`` policy (tenacity, 50 attempts, no call deadline). Every
speech call is billed per character, so this client makes at most ``max_retries + 1`` attempts and stops at a total
wall-clock budget (``max_call_s``). Retried: transport errors, timeouts, 408, 425, 429, 5xx and 200 bodies that are not audio.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import json
import logging
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set, Tuple, Type

from .exceptions import LLMProviderError

httpx: Any
try:
    import httpx  # type: ignore[no-redef]
except ImportError:  # optional extra 'llm'
    httpx = None

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_URL",
    "MODEL_MAX_CHARS",
    "SPEED_REJECTING_MODELS",
    "SPEED_RANGE",
    "UNIT_RANGE",
    "TEXT_NORMALIZATION",
    "UNVERIFIED_TEXT_NORMALIZATION",
    "RESPONSE_FORMATS",
    "VOICE_SETTING_KEYS",
    "SEED_RANGE",
    "MAX_RETRY_AFTER_S",
    "MIN_RETRY_DELAY_S",
    "RETRY_AFTER_JITTER",
    "MAX_AUDIO_BYTES",
    "MAX_CALL_S",
    "fetch_generation_provider",
    "SpeechError",
    "SpeechInputError",
    "SpeechRequest",
    "SpeechResult",
    "SpeechAccounting",
    "OpenRouterSpeech",
    "error_kind",
    "audio_body_problem",
    "reset_speech_catalogue_cache",
    "speech_catalogue",
    "fetch_credits",
    "price_per_char",
]

BASE_URL = "https://openrouter.ai/api/v1"
_GENERATION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
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
UNIT_RANGE = (0.0, 1.0)  # stability, similarity_boost and style are fractions
VOICE_SETTING_KEYS = frozenset({"stability", "similarity_boost", "style", "use_speaker_boost", "speed"})
SEED_RANGE = (0, 2**32 - 1)
TEXT_NORMALIZATION = frozenset({"auto", "on", "off"})
UNVERIFIED_TEXT_NORMALIZATION = frozenset({"on"})  # documented as possibly enterprise-only on v2.5; refused until checked live
RESPONSE_FORMATS = frozenset({"mp3", "pcm"})
#: Statuses that will not change on a second try. 402 (out of credit) is fatal for a run, so it is not retried either.
_NON_RETRYABLE = frozenset({400, 401, 402, 403, 404, 405, 410, 413, 422})
_RETRYABLE_4XX = frozenset({408, 425, 429})
MAX_RETRY_AFTER_S = 120.0  # a server asking for longer is treated as this long: one worker must not stall for hours
MIN_RETRY_DELAY_S = 1.0  # a Retry-After of 0 (or none) still waits, so many workers do not retry in lockstep
#: A Retry-After wait is stretched by up to this fraction: never earlier than the server asked, but not all at the same moment.
RETRY_AFTER_JITTER = 0.1
#: A 200 body larger than this is refused and read no further (a per-lemma clip is far below it).
MAX_AUDIO_BYTES = 64 * 1024 * 1024
#: Wall-clock budget of one :meth:`OpenRouterSpeech.synthesize` call across its attempts and waits. 0 disables it.
MAX_CALL_S = 900.0
_EXPIRY_WARNING_DAYS = 30
_CATALOGUE_TTL_S = 3600.0
_CATALOGUE_FAILURE_BACKOFF_S = 60.0  # after a failed fetch, do not ask /models again for this long (one outage, one fetch)
_DIGITS = re.compile(r"[0-9]+")
_default_sleep = time.sleep


class SpeechError(LLMProviderError, RuntimeError):
    """A speech request failed; ``status`` is the HTTP status (0 when none), ``retryable`` whether another try could help.

    ``details`` carries the diagnostics of the last attempt: ``status``, ``kind``, ``body`` (first 2000 characters),
    ``retry_after_s`` (None when absent), ``generation_id`` and ``content_type``.
    """

    def __init__(self, message: str, status: int = 0, retryable: bool = False, kind: str = "other", details: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.kind = kind  # one of error_kind()'s groups, for reporting
        self.details: Dict[str, Any] = dict(details or {})


class SpeechInputError(LLMProviderError, ValueError):
    """The request is invalid before any network call (empty text, over the model cap, bad speed). Nothing is billed."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical(value: Any) -> Any:
    """Numbers as floats, so ``1`` and ``1.0`` serialise the same; containers recursively. Booleans and strings are kept."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


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
    voice_settings: Optional[Mapping[str, Any]] = None  # stability, similarity_boost, style, use_speaker_boost, speed
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
        if not self.model.startswith("elevenlabs/") and self._has_elevenlabs_options():
            raise SpeechInputError(f"ElevenLabs options were given for {self.model}, which is not an ElevenLabs model")
        if self.apply_text_normalization is not None and self.apply_text_normalization not in TEXT_NORMALIZATION:
            raise SpeechInputError(f"apply_text_normalization must be one of {sorted(TEXT_NORMALIZATION)}")
        if self.apply_text_normalization in UNVERIFIED_TEXT_NORMALIZATION:
            raise SpeechInputError(
                f"apply_text_normalization={self.apply_text_normalization!r} is refused: it may need enterprise access on v2.5 "
                "and has not been verified through OpenRouter; use 'auto' or 'off'"
            )
        self._check_options()
        self._check_voice_settings()
        if self.speed is not None:
            self._check_speed(self.speed)

    def _check_options(self) -> None:
        if self.seed is not None and (not _is_int(self.seed) or not SEED_RANGE[0] <= self.seed <= SEED_RANGE[1]):
            raise SpeechInputError(f"seed must be an integer within {SEED_RANGE}, got {self.seed!r}")
        if self.language_code is not None and (not isinstance(self.language_code, str) or not self.language_code.strip()):
            raise SpeechInputError(f"language_code must be a non-empty string, got {self.language_code!r}")
        if self.apply_language_text_normalization is not None and not isinstance(self.apply_language_text_normalization, bool):
            raise SpeechInputError("apply_language_text_normalization must be a bool")

    def _check_voice_settings(self) -> None:
        """Every voice setting is checked here, including a speed given inside ``voice_settings``; unknown keys are refused."""
        for name, value in dict(self.voice_settings or {}).items():
            if name not in VOICE_SETTING_KEYS:
                raise SpeechInputError(f"unknown voice_settings key {name!r}; allowed: {sorted(VOICE_SETTING_KEYS)}")
            if name == "speed":
                self._check_speed(value)
            elif name == "use_speaker_boost":
                if not isinstance(value, bool):
                    raise SpeechInputError(f"voice_settings.use_speaker_boost must be a bool, got {value!r}")
            elif not _is_number(value) or not UNIT_RANGE[0] <= value <= UNIT_RANGE[1]:
                raise SpeechInputError(f"voice_settings.{name} must be a number within {UNIT_RANGE}, got {value!r}")

    def _check_speed(self, speed: Any) -> None:
        if not _is_number(speed) or not math.isfinite(speed):
            raise SpeechInputError(f"speed must be a finite number, got {speed!r}")
        if self.model in SPEED_REJECTING_MODELS and speed != 1.0:
            raise SpeechInputError(f"{self.model} rejects a non-default speed")
        if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
            raise SpeechInputError(f"speed must be within {SPEED_RANGE}, got {speed}")

    def _has_elevenlabs_options(self) -> bool:
        return any(
            value is not None
            for value in (
                self.seed, self.previous_text, self.next_text, self.language_code, self.apply_text_normalization,
                self.apply_language_text_normalization, self.pronunciation_dictionary_locators, self.voice_settings, self.speed,
            )
        )

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
        if self.model in SPEED_REJECTING_MODELS:
            settings.pop("speed", None)  # these models reject any speed field, a default one included
        elif self.speed is not None:
            settings["speed"] = self.speed
        if settings:
            options["voice_settings"] = settings
        if options:
            body["provider"] = {"options": {"elevenlabs": options}}
        return body

    def params_hash(self) -> str:
        """Stable digest of everything that changes the audio EXCEPT the text (voice, model, format, settings).

        Numbers are hashed as floats, so ``speed=1`` and ``speed=1.0`` (one setting) share one cache key.
        """
        body = replace(self, text="", previous_text=None, next_text=None).body()
        body.pop("input", None)
        return hashlib.sha256(json.dumps(_canonical(body), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


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
    """Thread-safe running totals of what a synthesis run spent. ``cost_usd`` covers calls whose price was known.

    ``request_bytes`` counts every upload attempt, failed ones included. ``possibly_billed_attempts`` counts attempts that
    timed out after the request was sent (read or write timeout): the upstream may have generated and billed them, and no
    response says so. ``possibly_billed_chars`` and ``possibly_billed_cost_usd`` are what those attempts would have cost.

    The lock is not a field: :func:`dataclasses.asdict` and :func:`copy.deepcopy` work, and pickling drops and recreates it.
    :meth:`snapshot` is the export: a plain dict copy of the totals.
    """

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
    possibly_billed_attempts: int = 0
    possibly_billed_chars: int = 0
    possibly_billed_cost_usd: float = 0.0
    errors_by_kind: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @property
    def total_cost_usd(self) -> float:
        """Alias of :attr:`cost_usd`, the name the chat providers' cost summaries use."""
        return self.cost_usd

    def record(self, result: SpeechResult) -> None:
        """Add one completed synthesis to the totals; a call with no known price counts as unpriced, not as free."""
        with self._lock:
            self.calls += 1
            self.chars += result.chars
            self.audio_bytes += result.nbytes
            if result.cost_usd is None:
                self.unpriced_calls += 1
            else:
                self.cost_usd += result.cost_usd

    def record_upload(self, nbytes: int) -> None:
        """Count the bytes sent for one attempt, whatever its outcome."""
        with self._lock:
            self.request_bytes += nbytes

    def record_failure(self) -> None:
        """Count one synthesis that ended in a failure (after its retries, or cancelled)."""
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

    def record_possibly_billed(self, chars: int = 0, cost_usd: Optional[float] = None) -> None:
        """Count one attempt that timed out after the upload, with the characters and cost it may have billed (cost None: no price)."""
        with self._lock:
            self.possibly_billed_attempts += 1
            self.possibly_billed_chars += chars
            if cost_usd is not None:
                self.possibly_billed_cost_usd += cost_usd

    def record_attempt_error(self, kind: str) -> None:
        """Count one failed HTTP attempt or transport error under its group (see :func:`error_kind`)."""
        with self._lock:
            self.errors_by_kind[kind] = self.errors_by_kind.get(kind, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        """A copy of every total, taken under the lock so a concurrent ``record`` is never half-visible."""
        with self._lock:
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items() if not k.startswith("_")}


def error_kind(status: int, transport: bool, body_problem: Optional[str] = None) -> str:
    """Reporting group of one failed attempt.

    ``transport`` (no response); ``empty_body``, ``bad_body``, ``truncated`` or ``too_large`` (a 200 whose body is not
    usable audio); otherwise ``http_429``, ``http_402``, ``http_5xx``, ``http_3xx`` or ``http_4xx``.
    """
    if transport:
        return "transport"
    if body_problem is not None:
        return body_problem
    if status == 200:
        return "empty_body"
    if status in (402, 429):
        return f"http_{status}"
    if status >= 500:
        return "http_5xx"
    if 300 <= status < 400:
        return "http_3xx"
    return "http_4xx"


def _is_mp3(content: bytes) -> bool:
    """An ID3 tag, or an MPEG frame header whose version and layer bits are not the reserved patterns."""
    if content[:3] == b"ID3":
        return True
    if len(content) < 2 or content[0] != 0xFF or (content[1] & 0xE0) != 0xE0:
        return False
    version = (content[1] >> 3) & 0x03  # 01 is reserved
    layer = (content[1] >> 1) & 0x03  # 00 is reserved
    return version != 0x01 and layer != 0x00


def _is_markup(content: bytes) -> bool:
    """A body that opens like JSON or markup and is printable ASCII throughout its first 256 bytes (random PCM never is)."""
    head = content[:256]
    return head[:1] in (b"<", b"{", b"[") and all(32 <= b < 127 or b in (9, 10, 13) for b in head)


def audio_body_problem(content: bytes, response_format: str, content_type: Optional[str], declared_length: Optional[int]) -> Optional[str]:
    """Why a 200 body is not complete audio, or None when it looks like the requested format.

    A JSON or text error envelope returned with status 200, a body shorter than its ``Content-Length``, and bytes without
    the MP3 frame or ID3 header are all refused. The failure then counts as ``bad_body`` / ``truncated`` and is retried.
    ``declared_length`` must be None when the body was content-encoded (the declared size is then the encoded size).
    """
    if declared_length is not None and declared_length != len(content):
        return "truncated"
    media = (content_type or "").lower()
    if media.startswith("application/json") or media.startswith("text/"):
        return "bad_body"
    if response_format == "mp3":
        return None if _is_mp3(content) else "bad_body"
    if len(content) % 2 != 0 or _is_markup(content):
        return "bad_body"
    return None


#: Guards the catalogue cache and the single-flight fetches (waiters sleep on it until a fetch ends).
_catalogue_cond = threading.Condition(threading.Lock())
#: One entry per base URL: ``at`` (monotonic time of the last good fetch, None before any), ``data``, ``failed_at``, ``inflight``.
_catalogue_cache: Dict[str, Dict[str, Any]] = {}
_expiry_warned: Set[str] = set()


def reset_speech_catalogue_cache() -> None:
    """Forget the cached speech catalogue (and any remembered failure), so the next call fetches it again."""
    with _catalogue_cond:
        _catalogue_cache.clear()
        _expiry_warned.clear()
        _catalogue_cond.notify_all()


def speech_catalogue(client: Any = None, force: bool = False, base_url: str = BASE_URL) -> Dict[str, Dict[str, Any]]:
    """Speech models by id from ``{base_url}/models?output_modalities=speech``.

    Cached per base URL for an hour; an empty listing is a valid listing and is cached too. After a failed fetch the failure
    is remembered for a minute, so an outage costs one request per minute rather than one per call. Concurrent callers share
    one fetch: while a refresh is in flight, callers get the last listing, and callers with no listing at all wait for it.
    """
    key = base_url.rstrip("/")
    with _catalogue_cond:
        while True:
            entry = _catalogue_cache.setdefault(key, {"at": None, "data": {}, "failed_at": None, "inflight": False})
            now = time.monotonic()
            if not force:
                if entry["at"] is not None and now - entry["at"] < _CATALOGUE_TTL_S:
                    return dict(entry["data"])
                if entry["failed_at"] is not None and now - entry["failed_at"] < _CATALOGUE_FAILURE_BACKOFF_S:
                    return dict(entry["data"])
            if not entry["inflight"]:
                entry["inflight"] = True
                break
            if not force and (entry["at"] is not None or entry["failed_at"] is not None):
                return dict(entry["data"])  # another thread is refreshing: serve the last listing meanwhile
            _catalogue_cond.wait()
    data: Optional[Dict[str, Dict[str, Any]]] = None
    try:
        data = _fetch_speech_catalogue(client, key)  # network I/O runs outside the lock: other threads keep reading the cache
    finally:
        with _catalogue_cond:
            entry["inflight"] = False
            if data is not None:
                entry.update(at=time.monotonic(), data=data, failed_at=None)
            else:
                entry["failed_at"] = time.monotonic()
            _catalogue_cond.notify_all()
    with _catalogue_cond:
        return dict(entry["data"])


def _fetch_speech_catalogue(client: Any, base_url: str = BASE_URL) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch the speech models from OpenRouter, keyed by id; None when the fetch fails (costs then stay unknown)."""
    if client is None and httpx is None:
        logger.warning("httpx is not installed (optional extra 'llm'); speech prices are unknown")
        return None
    http = client if client is not None else httpx.Client(timeout=15.0)
    try:
        resp = http.get(f"{base_url}/models", params={"output_modalities": "speech"})
        resp.raise_for_status()
        return {m["id"]: m for m in resp.json().get("data", ()) if "id" in m}
    except Exception as exc:  # best-effort: an outage leaves prices unknown (cost_usd None, counted), it must not stop synthesis
        logger.warning("speech catalogue fetch failed (%s: %s); costs will be reported as unknown", type(exc).__name__, exc)
        return None
    finally:
        if client is None:
            http.close()


def fetch_credits(api_key: str, client: Any = None) -> Optional[Dict[str, float]]:
    """The account balance from ``GET /api/v1/credits``: ``total_credits``, ``total_usage`` and ``remaining`` in USD.

    None when the request fails or the body is malformed (best-effort: a balance check must never stop a synthesis run).
    """
    if client is None and httpx is None:
        return None
    http = client if client is not None else httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {api_key}"})
    try:
        resp = http.get(f"{BASE_URL}/credits", headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        data = resp.json()["data"]
        total, used = float(data["total_credits"]), float(data["total_usage"])
        return {"total_credits": total, "total_usage": used, "remaining": total - used}
    except Exception as exc:  # best-effort: a failed balance check is logged by the caller and never stops a run
        logger.warning("balance check failed (%s: %s)", type(exc).__name__, exc)
        return None
    finally:
        if client is None:
            http.close()


def fetch_generation_provider(generation_id: str, api_key: str, client: Any = None) -> Optional[Dict[str, str]]:
    """The serving endpoint of one request from ``GET /api/v1/generation?id=``: ``provider_name`` and ``model``.

    The speech response carries the generation id (``x-generation-id``) but not the endpoint that served it; this is the
    lookup for it, one call per request, so a caller decides when to pay for it. None when the id is malformed, the request
    fails or the body has no provider name. Best-effort, like :func:`fetch_credits`.
    """
    if not _GENERATION_ID_RE.fullmatch(generation_id or ""):
        return None
    if client is None and httpx is None:
        return None
    http = client if client is not None else httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {api_key}"})
    try:
        resp = http.get(f"{BASE_URL}/generation", params={"id": generation_id}, headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        data = resp.json()["data"]
        provider = str(data.get("provider_name") or "")
        if not provider:
            return None
        return {"provider_name": provider, "model": str(data.get("model") or "")}
    except Exception as exc:  # best-effort: a failed lookup is logged and never stops a run
        logger.warning("generation lookup failed (%s: %s)", type(exc).__name__, exc)
        return None
    finally:
        if client is None:
            http.close()


def _warn_if_expiring(model: str, entry: Mapping[str, Any]) -> None:
    """Log once per model when the catalogue's ``expiration_date`` is within the warning window (or already past)."""
    raw = entry.get("expiration_date")
    if not raw or model in _expiry_warned:
        return
    try:
        expires = dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return
    days_left = (expires - dt.datetime.now(dt.timezone.utc).date()).days
    if days_left <= _EXPIRY_WARNING_DAYS:
        _expiry_warned.add(model)
        logger.warning("speech model %s expires on %s (%d days left); plan its replacement", model, expires.isoformat(), days_left)


def price_per_char(model: str, client: Any = None, base_url: str = BASE_URL) -> Optional[float]:
    """USD per input character for ``model``, or None when the catalogue does not price it. A price of zero is a price."""
    entry = speech_catalogue(client, base_url=base_url).get(model)
    if not entry:
        return None
    _warn_if_expiring(model, entry)
    raw = (entry.get("pricing") or {}).get("prompt")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):  # best-effort: a malformed price is reported as unknown, never as zero
        return None


def _transport_errors() -> Tuple[Type[BaseException], ...]:
    """The exceptions that mean no response arrived: httpx's when httpx is installed, else OS-level socket errors."""
    return (httpx.RequestError,) if httpx is not None else (OSError,)


def _billable_timeout(exc: BaseException) -> bool:
    """A read or write timeout: the request left, so the upstream may have billed it. Connect and pool timeouts never sent it."""
    return httpx is not None and isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout))


def _declared_length(raw: Optional[str]) -> Optional[int]:
    """A Content-Length that is plain ASCII digits as an int; anything else (including Unicode digits) is None."""
    if raw is None:
        return None
    text = raw.strip()
    return int(text) if _DIGITS.fullmatch(text) else None


def _read_capped(resp: Any, limit: int) -> Tuple[bytes, bool]:
    """The decoded body, read at most ``limit`` bytes; the flag says whether the body was longer and was cut."""
    buf = bytearray()
    for chunk in resp.iter_bytes():
        buf.extend(chunk)
        if len(buf) > limit:
            return bytes(buf[:limit]), True
    return bytes(buf), False


@dataclass
class _Reply:
    """What one attempt returned: status, headers and the body read so far; or the transport error that ended it."""

    limit: int
    status: int = 0
    headers: Any = field(default_factory=dict)
    content: bytes = b""
    too_large: bool = False
    transport_error: Optional[BaseException] = None

    @property
    def text(self) -> str:
        """The first 2000 characters of the body, for diagnostics."""
        return self.content[:2000].decode("utf-8", errors="replace")


def _body_problem(reply: _Reply, response_format: str) -> Optional[str]:
    """Why a 200 reply is not usable audio, or None when it is."""
    if reply.too_large:
        return "too_large"
    if not reply.content:
        return "empty_body"
    encoding = str(reply.headers.get("content-encoding", "")).strip().lower()
    declared = _declared_length(reply.headers.get("content-length")) if encoding in ("", "identity") else None
    return audio_body_problem(reply.content, response_format, reply.headers.get("content-type"), declared)


def _judge(reply: _Reply, response_format: str) -> Tuple[Optional[str], str, bool]:
    """Classify one attempt as (body problem, message, retryable). A usable 200 has no body problem and no message."""
    if reply.transport_error is not None:
        return None, f"speech request failed: {type(reply.transport_error).__name__}: {reply.transport_error}", True
    if reply.status != 200:
        retryable = reply.status not in _NON_RETRYABLE and (reply.status >= 500 or reply.status in _RETRYABLE_4XX)
        return None, f"speech request returned HTTP {reply.status}: {reply.text[:300]}", retryable
    problem = _body_problem(reply, response_format)
    if problem == "too_large":
        return problem, f"speech response is larger than {reply.limit} bytes", False
    if problem is None:
        return None, "", False
    return problem, f"speech request returned HTTP 200 with a {problem} body", True


def _retry_delay(attempt: int, retry_after: Optional[float], backoff_s: float) -> float:
    """Seconds before the next attempt: the server's Retry-After (stretched by jitter, never shorter), else jittered exponential backoff."""
    if retry_after is not None:
        return max(MIN_RETRY_DELAY_S, min(retry_after, MAX_RETRY_AFTER_S)) * (1.0 + RETRY_AFTER_JITTER * random.random())
    return float(backoff_s * 2 ** (attempt - 1) * (0.5 + random.random()))


class OpenRouterSpeech:
    """Synchronous, thread-safe OpenRouter speech client with bounded retries and exact accounting.

    ``cancel`` is an optional :class:`threading.Event`. Once it is set no new attempt starts and a pending retry wait ends
    at once; the call then raises :class:`SpeechError` with kind ``cancelled`` (naming the last failed attempt, if any).
    A request already in flight is not interrupted. ``max_call_s`` bounds the whole call: a retry that would start after
    the budget raises the last failure instead. ``max_audio_bytes`` caps a 200 body; a larger one is refused, not retried.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout_s: float = 180.0,  # upstream times out at 180 s
        max_retries: int = 5,
        backoff_s: float = 2.0,
        client: Any = None,
        sleep: Callable[[float], None] = _default_sleep,
        accounting: Optional[SpeechAccounting] = None,
        cancel: Optional[threading.Event] = None,
        max_audio_bytes: int = MAX_AUDIO_BYTES,
        max_call_s: float = MAX_CALL_S,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key not provided")
        if client is None and httpx is None:
            raise ImportError("OpenRouterSpeech needs httpx: install the optional extra 'llm'")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.accounting = accounting if accounting is not None else SpeechAccounting()
        self.cancel = cancel
        self.max_audio_bytes = max_audio_bytes
        self.max_call_s = max_call_s
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

    def _cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def _wait(self, delay: float) -> None:
        """Sleep for ``delay`` seconds, or until cancelled. A cancelled client does not sleep; an injected ``sleep`` is honoured."""
        if self._cancelled():
            return
        if self.cancel is not None and self._sleep is _default_sleep:
            self.cancel.wait(delay)
        else:
            self._sleep(delay)

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        """Synthesize one request. Raises :class:`SpeechInputError` before any call, :class:`SpeechError` after retries."""
        request.validate()
        payload = json.dumps(request.body(), ensure_ascii=False).encode("utf-8")
        deadline = time.monotonic() + self.max_call_s if self.max_call_s > 0 else math.inf
        attempt = 0
        last_message: Optional[str] = None
        while True:
            if self._cancelled():
                self.accounting.record_failure()
                detail = f" after a failed attempt: {last_message}" if last_message else " before it was sent"
                raise SpeechError(f"speech request cancelled{detail}", 0, False, kind="cancelled")
            attempt += 1
            started = time.perf_counter()
            self.accounting.record_upload(len(payload))
            reply = self._send(request, payload)
            problem, message, retryable = _judge(reply, request.response_format)
            if reply.transport_error is None and reply.status == 200 and problem is None:
                return self._finish(request, reply.content, reply.headers, time.perf_counter() - started)
            transport = reply.transport_error is not None
            kind = error_kind(reply.status, transport, problem)
            self.accounting.record_attempt_error(kind)
            retry_after = None if transport else _parse_retry_after(reply.headers.get("retry-after"))
            details = {
                "status": reply.status,
                "kind": kind,
                "body": reply.text,
                "retry_after_s": retry_after,
                "generation_id": reply.headers.get("x-generation-id"),
                "content_type": reply.headers.get("content-type"),
            }
            if not retryable or attempt > self.max_retries:
                self.accounting.record_failure()
                raise SpeechError(message, reply.status, retryable, kind=kind, details=details)
            delay = _retry_delay(attempt, retry_after, self.backoff_s)
            if time.monotonic() + delay > deadline:
                self.accounting.record_failure()
                raise SpeechError(f"{message}; no retry fits the {self.max_call_s:g} s call budget", reply.status, retryable, kind=kind, details=details)
            logger.warning("speech attempt %d failed (%s); retrying in %.1fs", attempt, message, delay)
            self.accounting.record_retry()
            last_message = message
            self._wait(delay)

    def _send(self, request: SpeechRequest, payload: bytes) -> _Reply:
        """One POST, body read up to ``max_audio_bytes``. A transport error is returned in the reply, not raised."""
        reply = _Reply(limit=self.max_audio_bytes)
        try:
            with self._client.stream("POST", f"{self.base_url}/audio/speech", content=payload, headers={"Content-Type": "application/json"}) as resp:
                reply.status = resp.status_code
                reply.headers = resp.headers
                reply.content, reply.too_large = _read_capped(resp, self.max_audio_bytes)
        except _transport_errors() as exc:  # transport errors and timeouts are transient; programming errors propagate
            reply.transport_error = exc
            if _billable_timeout(exc):
                rate = price_per_char(request.model, self._client, self.base_url)
                self.accounting.record_possibly_billed(request.billed_chars, None if rate is None else rate * request.billed_chars)
        return reply

    def _finish(self, request: SpeechRequest, content: bytes, headers: Any, latency: float) -> SpeechResult:
        """Build the result for a successful response, price it from the catalogue and record it in the accounting."""
        rate = price_per_char(request.model, self._client, self.base_url)
        result = SpeechResult(
            audio=content,
            request=request,
            chars=request.billed_chars,
            cost_usd=None if rate is None else rate * request.billed_chars,
            latency_s=latency,
            generation_id=headers.get("x-generation-id"),
            content_type=headers.get("content-type"),
        )
        self.accounting.record(result)
        return result


def _parse_retry_after(raw: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """Seconds a ``Retry-After`` value asks for: a number of seconds or an HTTP date. None when absent or unusable."""
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(raw)
            if when.tzinfo is None:  # "-0000" parses as naive; it means UTC, not the local time timestamp() would assume
                when = when.replace(tzinfo=dt.timezone.utc)
            seconds = when.timestamp() - (time.time() if now is None else now)
        except (TypeError, ValueError, OverflowError, OSError):  # best-effort: an unreadable Retry-After falls back to exponential backoff
            return None
    if not math.isfinite(seconds):
        return MAX_RETRY_AFTER_S if seconds > 0 else None
    return max(0.0, seconds)


def _retry_after(resp: Any, now: Optional[float] = None) -> Optional[float]:
    """Seconds the response's ``Retry-After`` header asks for (see :func:`_parse_retry_after`)."""
    return _parse_retry_after(resp.headers.get("retry-after"), now)
