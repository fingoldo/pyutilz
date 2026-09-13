"""Keep every paid LLM attempt: its raw text, before anything parses it, and what it cost.

WHY
---
Generation is the expensive part of an LLM pipeline and storage is nearly free, yet the raw answer is the
first thing to go missing. glossum found three shapes of it: a retry loop that overwrote every earlier
attempt (65 parsed example sentences lost on one run), a truncated answer raised away before anything kept
it, and an archive written inside the caller's transaction, so a rollback took the copy along. autopsia's
extraction path saved no raw text at all and kept no cost or partial text for a failed call.

WHAT
----
* :class:`AttemptRecord` -- one attempt: raw or partial text, the reasoning stored beside it by digest,
  outcome, tokens, reasoning tokens, cost
  (``None`` when unknown, never 0.0), generation id, upstream provider, finish reasons, duration, error.
* A content-addressed text store with a pluggable backend. :class:`DirectoryContentStore` writes
  ``<sha256>.txt`` files; :class:`CallableContentStore` adapts any callable (a SQL insert, say). The rule
  every backend follows: on failure, return ``None``, never the digest, so nothing is left pointing at text
  that was not written.
* :class:`JsonlAttemptSink` -- an append-only JSONL log of attempts that never raises.
* :func:`archive_provider` -- wraps a ``pyutilz.llm`` provider's ``generate`` (and ``generate_stream`` when it
  has one) on the instance, so every call through it, ``generate_json`` and ``generate_batch`` included,
  stores its text before returning it and records the attempt, including failed and interrupted ones.

Metadata comes from what the provider already exposes per call: ``last_call_summary()`` when it has one
(OpenRouter), otherwise the individual ``last_*`` / ``_last_*`` attributes. They are read in the same task
that made the call, where the per-call attributes are still this call's.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    "AttemptRecord",
    "CallableContentStore",
    "DirectoryContentStore",
    "JsonlAttemptSink",
    "archive_provider",
    "attempts_missing_generation_stats",
    "fetch_missing_generation_stats",
    "metadata_from_provider",
    "sha256_text",
]

OUTCOMES = ("accepted", "superseded", "parse_failed", "truncated", "error")
_ARCHIVED_FLAG = "_pyutilz_attempt_archive_installed"


def sha256_text(text: str) -> str:
    """The content address of ``text``: hex SHA-256 of its UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class AttemptRecord:
    """One LLM attempt, including the ones a later attempt superseded or that failed.

    ``raw_text`` is the full answer, or the partial text of one that was cut off. ``response_sha256`` is set
    only when the text was actually stored. ``cost_usd`` is ``None`` when the provider did not report one;
    0.0 would claim the call was free.
    """

    attempt_number: int
    outcome: str
    raw_text: Optional[str] = None
    response_sha256: Optional[str] = None
    # The thinking the caller was billed for, stored like the answer and referenced by digest: one glm-5.3-flash
    # call spent 39,641 of 65,577 output tokens on 118,004 characters of reasoning, which would bloat this log
    # line by line. Kept because a model that thinks for 393 seconds before its first content byte is
    # indistinguishable from a dead route until someone can read what it was doing.
    reasoning_sha256: Optional[str] = None
    reasoning_chars: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    generation_id: Optional[str] = None
    upstream_provider: Optional[str] = None
    finish_reason: Optional[str] = None
    native_finish_reason: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    duration_seconds: Optional[float] = None
    # Measured here, not asked of anyone: the wrapper sees the first chunk arrive. `None` on a buffered call,
    # where the whole answer lands at once and there is no first token to time.
    time_to_first_token_s: Optional[float] = None
    # From the upstream's own record, when `archive_provider(fetch_stats=True)` asks for it. `latency_ms` is
    # ITS time to first token and `generation_time_ms` the whole run, which together separate "the route never
    # started" from "the model thought for nine minutes" - the question this archive exists to settle.
    latency_ms: Optional[int] = None
    generation_time_ms: Optional[int] = None
    cancelled: Optional[bool] = None
    # True when the upstream had no record of the generation WHEN ASKED, which is moments after the call. The
    # record is written asynchronously, so this is also True for a call that succeeded and simply has not been
    # filed yet; only a 404 that persists later means the generation never finished.
    generation_record_missing: Optional[bool] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse an outcome outside ``OUTCOMES``, so a typo cannot pass for a real outcome."""
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")

    def to_dict(self) -> Dict[str, Any]:
        """The record as a plain dict, for a JSON line or a row."""
        return dataclasses.asdict(self)


class DirectoryContentStore:
    """``<root>/<sha256>.txt`` per distinct text; writing the same text twice is one file.

    The write goes to a temporary file in the same directory and is renamed into place, so a reader never
    sees half a file. Any OS error is logged and yields ``None``.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        """Store texts under ``root``, created on first write."""
        self.root = Path(root)

    def path_for(self, digest: str) -> Path:
        """Where the text with this digest is, or would be, stored."""
        return self.root / f"{digest}.txt"

    async def put(self, text: str) -> Optional[str]:
        """Store ``text`` and return its digest, or ``None`` if the write failed."""
        digest = sha256_text(text)
        target = self.path_for(digest)
        try:
            if target.exists():
                return digest
            self.root.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=f".{digest}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(text.encode("utf-8"))
                os.replace(tmp, target)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.warning("could not store response text %s under %s: %s", digest[:12], self.root, exc)
            return None
        return digest

    def get(self, digest: str) -> Optional[str]:
        """The stored text for ``digest``, or ``None`` if there is none."""
        path = self.path_for(digest)
        return path.read_text(encoding="utf-8") if path.exists() else None


class CallableContentStore:
    """Adapt a caller-supplied writer ``fn(digest, text)`` (sync or async) into a content store.

    For a SQL table: ``fn`` runs ``INSERT ... ON CONFLICT (hash) DO NOTHING``. If it raises, the failure is
    logged and ``None`` is returned, so the caller never records a digest for text that was not stored.
    """

    def __init__(self, fn: Callable[[str, str], Any]) -> None:
        """Store texts through ``fn(digest, text)``, sync or async."""
        self.fn = fn

    async def put(self, text: str) -> Optional[str]:
        """Hand ``text`` to the writer and return its digest, or ``None`` if the writer raised."""
        digest = sha256_text(text)
        try:
            outcome = self.fn(digest, text)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as exc:
            logger.warning("could not store response text %s: %s: %s", digest[:12], type(exc).__name__, exc)
            return None
        return digest


class JsonlAttemptSink:
    """Append one JSON line per attempt to ``path``. Never raises; a failed write is logged and counted."""

    def __init__(self, path: Union[str, Path]) -> None:
        """Append attempts to ``path``, counting lines written and dropped."""
        self.path = Path(path)
        self.written = 0
        self.dropped = 0

    async def record(self, attempt: AttemptRecord) -> None:
        """Append one attempt as a JSON line; a failed write is logged and counted, never raised."""
        line = json.dumps(attempt.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            self.dropped += 1
            logger.warning("could not append attempt %s to %s: %s", attempt.attempt_number, self.path, exc)
            return
        self.written += 1

    def read(self) -> List[Dict[str, Any]]:
        """Every attempt written so far, oldest first."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _int_or_none(value: Any) -> Optional[int]:
    """``value`` as an int, or ``None`` when it is missing or not a number."""
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` when it is missing or not a number."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> Optional[str]:
    """``value`` as a string, or ``None`` when it is missing or empty."""
    return None if value is None or value == "" else str(value)


def metadata_from_provider(provider: Any) -> Dict[str, Any]:
    """The per-call metadata a provider exposes after a call, as :class:`AttemptRecord` field values.

    Reads ``last_call_summary()`` when the provider has one, and fills anything it lacks from the individual
    attributes. A cost the provider did not report stays ``None``.
    """
    summary: Dict[str, Any] = {}
    fn = getattr(provider, "last_call_summary", None)
    if callable(fn):
        try:
            got = fn()
            summary = dict(got) if isinstance(got, dict) else {}
        except Exception as exc:  # a summary is a convenience; its failure must not lose the attempt
            logger.debug("last_call_summary() failed on %s: %s", type(provider).__name__, exc)
    usage = getattr(provider, "_last_usage", None)
    usage = usage if isinstance(usage, dict) else {}

    def pick(summary_key: str, *fallbacks: Any) -> Any:
        """The summary's value for ``summary_key``, else the first fallback that is set."""
        if summary.get(summary_key) not in (None, ""):
            return summary[summary_key]
        for value in fallbacks:
            if value not in (None, ""):
                return value
        return None

    cost = pick("cost_usd", getattr(provider, "last_actual_cost_usd", None), getattr(provider, "last_cost_usd", None))
    return {
        "prompt_tokens": _int_or_none(pick("input_tokens", usage.get("input_tokens"), usage.get("prompt_tokens"))),
        "completion_tokens": _int_or_none(pick("output_tokens", usage.get("output_tokens"), usage.get("completion_tokens"))),
        "reasoning_tokens": _int_or_none(pick("reasoning_tokens", usage.get("reasoning_tokens"), getattr(provider, "last_thinking_tokens", None))),
        "cost_usd": _float_or_none(cost),
        "generation_id": _str_or_none(pick("generation_id", getattr(provider, "last_generation_id", None))),
        "upstream_provider": _str_or_none(pick("upstream_provider", getattr(provider, "last_upstream_provider", None))),
        "finish_reason": _str_or_none(pick("finish_reason", getattr(provider, "_last_finish_reason", None))),
        "native_finish_reason": _str_or_none(pick("native_finish_reason", getattr(provider, "last_native_finish_reason", None))),
        "model": _str_or_none(pick("requested_model", getattr(provider, "model_name", None), getattr(provider, "model", None))),
    }


def attempts_missing_generation_stats(attempts: Iterable[Dict[str, Any]]) -> List[str]:
    """Generation ids whose attempt row has no upstream timings yet, oldest first, de-duplicated.

    The record is written asynchronously, so the archive's own query - made moments after the call - comes back
    empty for anything short. Measured 2026-09-14: a call that returned 61 parsed rows and billed $0.031 was
    filed minutes later, and the same id then answered in full. So the timings are BACKFILLED rather than
    retried at the call, and this names what to ask for.
    """
    seen: Dict[str, None] = {}
    for attempt in attempts:
        gid = attempt.get("generation_id")
        if not gid or attempt.get("latency_ms") is not None:
            continue
        seen.setdefault(str(gid), None)
    return list(seen)


async def fetch_missing_generation_stats(provider: Any, attempts: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """`{generation_id: record}` for every attempt still missing its timings, skipping the ones still absent.

    Returns rather than writes: an attempt log is append-only, and rewriting a line in place to enrich it would
    trade the one property that makes it trustworthy for a column. The caller decides where the answers live.
    """
    fetch = getattr(provider, "fetch_generation_stats", None)
    if not callable(fetch):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    unfiled, failed = 0, []
    for gid in attempts_missing_generation_stats(attempts):
        try:
            record = await fetch(gid)
        except Exception as exc:
            # Counted here and reported ONCE below: a backfill over a long log asks for hundreds of ids, and a
            # per-id warning turns one upstream outage into a wall of identical lines.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                unfiled += 1
            else:
                failed.append(f"{gid} ({type(exc).__name__}: {exc})")
            continue
        if isinstance(record, dict) and record:
            out[gid] = record
    if failed:
        logger.warning("could not backfill %d generation record(s): %s", len(failed), "; ".join(failed[:5]))
    if unfiled:
        logger.info("%d generation record(s) not filed upstream yet; ask again later", unfiled)
    return out


def _error_text(exc: BaseException) -> str:
    """``Type: message`` for an exception, capped so a huge message cannot bloat the record."""
    return f"{type(exc).__name__}: {exc}"[:2000]


class _Archiver:
    """The state one :func:`archive_provider` installation shares between ``generate`` and ``generate_stream``."""

    def __init__(self, provider: Any, store: Any, sink: Any, is_truncated: Callable[[Dict[str, Any]], bool], fetch_stats: bool = False) -> None:
        """Share one attempt counter between a provider's ``generate`` and ``generate_stream``."""
        self.provider = provider
        self.store = store
        self.sink = sink
        self.fetch_stats = fetch_stats
        self.is_truncated = is_truncated
        self.attempts = 0

    async def _upstream_timings(self, generation_id: Optional[str]) -> Dict[str, Any]:
        """The upstream's own record for this call, or the fact that it has none.

        Opt-in because it costs a round trip per attempt. A 404 is recorded rather than swallowed, but read it
        carefully: the record is written asynchronously, so asking THIS soon after the call also returns 404
        for a generation that succeeded - measured 2026-09-14 on a call that had just returned 61 parsed rows
        and billed $0.031. A 404 that still holds minutes later is the real "it never finished" signal; this
        field is the timestamped observation, not the verdict.
        """
        fetch = getattr(self.provider, "fetch_generation_stats", None)
        if not (self.fetch_stats and generation_id and callable(fetch)):
            return {}
        try:
            stats = await fetch(generation_id)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                return {"generation_record_missing": True}
            logger.warning("could not fetch generation stats for %s: %s: %s", generation_id, type(exc).__name__, exc)
            return {}
        if not isinstance(stats, dict):
            return {}
        return {
            "latency_ms": _int_or_none(stats.get("latency")),
            "generation_time_ms": _int_or_none(stats.get("generation_time")),
            "cancelled": stats.get("cancelled") if isinstance(stats.get("cancelled"), bool) else None,
            "generation_record_missing": False,
        }

    async def keep(self, text: Optional[str], started: float, started_at: str, error: Optional[BaseException] = None, first_token_at: Optional[float] = None) -> None:
        """Store the attempt's text, then record the attempt with the provider's metadata for this call."""
        self.attempts += 1
        meta = metadata_from_provider(self.provider)
        digest = await self.store.put(text) if text else None
        reasoning = getattr(self.provider, "last_reasoning_text", None)
        reasoning_digest = await self.store.put(reasoning) if isinstance(reasoning, str) and reasoning else None
        if error is not None:
            outcome = "truncated" if text else "error"
        else:
            outcome = "truncated" if self.is_truncated(meta) else "accepted"
        record = AttemptRecord(
            attempt_number=self.attempts,
            outcome=outcome,
            raw_text=text,
            response_sha256=digest,
            reasoning_sha256=reasoning_digest,
            reasoning_chars=len(reasoning) if isinstance(reasoning, str) and reasoning else None,
            provider=type(self.provider).__name__,
            duration_seconds=round(time.monotonic() - started, 3),
            time_to_first_token_s=round(first_token_at - started, 3) if first_token_at is not None else None,
            **(await self._upstream_timings(meta.get("generation_id"))),
            error=_error_text(error) if error is not None else None,
            started_at=started_at,
            **meta,
        )
        try:
            await self.sink.record(record)
        except Exception as exc:  # the sink contract is "never raises"; a broken one must not break the call
            logger.warning("attempt sink %s raised: %s", type(self.sink).__name__, exc)


def _default_is_truncated(meta: Dict[str, Any]) -> bool:
    """Whether the call stopped on the output limit rather than finishing."""
    return (meta.get("finish_reason") or "").lower() in ("length", "max_tokens")


def archive_provider(
    provider: Any,
    store: Any,
    sink: Any,
    *,
    is_truncated: Callable[[Dict[str, Any]], bool] = _default_is_truncated,
    fetch_stats: bool = False,
) -> Any:
    """Make every call through ``provider`` keep its raw text and an attempt record; returns the same object.

    ``store`` has ``async put(text) -> digest | None`` (:class:`DirectoryContentStore`,
    :class:`CallableContentStore`); ``sink`` has ``async record(AttemptRecord)`` (:class:`JsonlAttemptSink`).
    The text is stored before ``generate`` returns it, so a caller that then fails to parse it has already
    kept it. A call that raises is recorded with its error and then re-raised; a stream that breaks off is
    recorded with the partial text it produced. Idempotent: a provider already wrapped is returned as is.
    """
    if provider is None or getattr(provider, _ARCHIVED_FLAG, False):
        return provider
    if not hasattr(provider, "__dict__"):
        # A stand-in with no instance dict - `object()`, a slotted double - cannot be wrapped, and crashing on
        # it would make every test that stubs the factory fail for a reason that has nothing to do with what it
        # tests. Found 2026-09-13: nine autopsia tests broke exactly this way the moment their production call
        # site started archiving. A real provider always has one, so this only ever declines a double.
        logger.warning("%s cannot take attributes, so its calls are not archived", type(provider).__name__)
        return provider
    archiver = _Archiver(provider, store, sink, is_truncated, fetch_stats=fetch_stats)
    generate = getattr(provider, "generate", None)
    if callable(generate):

        @functools.wraps(generate)
        async def archived_generate(*args: Any, **kwargs: Any) -> Any:
            """``generate``, with its text stored and its attempt recorded before it returns or raises."""
            started, started_at = time.monotonic(), datetime.now(timezone.utc).isoformat()
            try:
                result = await generate(*args, **kwargs)
            except BaseException as exc:
                partial = getattr(provider, "last_partial_text", None)
                await archiver.keep(partial if isinstance(partial, str) else None, started, started_at, error=exc)
                raise
            await archiver.keep(result if isinstance(result, str) else None, started, started_at)
            return result

        provider.generate = archived_generate
    stream = getattr(provider, "generate_stream", None)
    if callable(stream):

        @functools.wraps(stream)
        async def archived_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            """``generate_stream``, recording the whole text, or the part produced before it broke off."""
            started, started_at = time.monotonic(), datetime.now(timezone.utc).isoformat()
            parts: List[str] = []
            failure: Optional[BaseException] = None
            first_token_at: Optional[float] = None
            try:
                async for chunk in stream(*args, **kwargs):
                    if isinstance(chunk, str):
                        if first_token_at is None:
                            # Time to first token, measured rather than asked for: a route that never starts
                            # and a model that thinks for minutes both look like silence until this is on the row.
                            first_token_at = time.monotonic()
                        parts.append(chunk)
                    yield chunk
            except BaseException as exc:
                failure = exc
                raise
            finally:
                await archiver.keep("".join(parts) or None, started, started_at, error=failure, first_token_at=first_token_at)

        provider.generate_stream = archived_stream
    setattr(provider, _ARCHIVED_FLAG, True)
    return provider


# Kept for callers that type-annotate stores and sinks without importing a Protocol (3.8 has typing.Protocol,
# but a structural check is all anything here needs).
ContentPut = Callable[[str], Awaitable[Optional[str]]]
