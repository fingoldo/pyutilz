"""Cached, rate-limited HTTP client for polite, resumable API/ingestion workloads.

Extracted from a pattern duplicated 3x across autopsia (``vocab/_http.py``, ``ingest/fetch.py``,
and ``ingest/sim_brazil.py``'s FTP variant): a GET is served from an on-disk cache when present,
throttled PER HOST (not globally -- serializing every request across N unrelated services turns
an I/O-bound job into hours of mostly waiting), retried with exponential backoff (honoring a
``Retry-After`` header when the server sends one), and can run fully offline (cache-only, never
dials out) for reproducible test/CI runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, cast

from pyutilz.core.serialization import atomic_write_bytes
from pyutilz.web.url_guard import ALLOWED_SCHEMES, urlopen_checked

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

# `Retry-After` is attacker/remote-controlled: a hostile or misconfigured host answering 429 with
# `Retry-After: 999999999` would otherwise park the calling thread in a synchronous, uncancellable
# time.sleep() for decades, hanging a whole batch job on one bad URL. Anything above this cap falls
# back to the local exponential backoff instead.
MAX_RETRY_AFTER_SECONDS = 120.0

# Response bodies are read from the same untrusted party `Retry-After` comes from, so they get the
# same treatment: a bound, not a promise. 64 MiB is far above any JSON/text API payload this client
# targets, and far below what would OOM a worker or fill a cache volume.
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 256 * 1024

# A cache `tag` becomes a DIRECTORY component of the cache path, so it must not be able to escape
# cache_dir (`..`, absolute paths, separators). Same containment concern as
# ``core.disk_cache._key_path``; here it is enforced by grammar since a tag is meant to be a short
# human-readable label, not a path.
_SAFE_TAG_RE = re.compile(r"\A[A-Za-z0-9_.-]+\Z")


class CachedHttpClient:
    """A GET client with disk caching, per-host throttling, and offline mode.

    Each instance owns its own cache directory and throttle state, so multiple clients (e.g. one
    per external service, or one per test) don't contend on a shared lock or collide on cache
    keys. Thread-safe: the per-host throttle is guarded by a lock, so many threads can safely
    share one instance.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        user_agent: str = "pyutilz-cached-client/0.1",
        min_interval: float = 0.12,
        offline_env_var: Optional[str] = None,
        allowed_schemes: FrozenSet[str] = ALLOWED_SCHEMES,
        max_bytes: Optional[int] = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        """
        Args:
            cache_dir: root directory for cache entries (one JSON file per URL+tag).
            user_agent: sent as the ``User-Agent`` header on every request.
            min_interval: minimum seconds between two requests to the same host.
            offline_env_var: name of an environment variable that, when truthy
                (``1``/``true``/``yes``, case-insensitive), makes :meth:`offline` return True --
                only the cache may be consulted, a cache miss returns ``None`` instead of dialing
                out. Pass ``None`` to disable offline mode entirely (always live).
            allowed_schemes: forwarded to :func:`pyutilz.web.url_guard.urlopen_checked` -- rejects
                any URL whose scheme isn't in this set before a socket is opened.
            max_bytes: reject (and do not cache) any response body larger than this many bytes;
                ``None`` disables the cap and restores the historical unbounded read. Defaults to
                ``DEFAULT_MAX_RESPONSE_BYTES``, comfortably above any JSON/text API payload this
                client is meant for.
        """
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.offline_env_var = offline_env_var
        self.allowed_schemes = allowed_schemes
        self.max_bytes = max_bytes
        self._last_call: Dict[str, float] = {}
        self._throttle_lock = threading.Lock()

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the unpicklable lock; ``_last_call`` (per-host throttle timestamps) is meaningless
        across a pickle boundary too, so it's cleared rather than shipped."""
        state = self.__dict__.copy()
        del state["_throttle_lock"]
        state["_last_call"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._throttle_lock = threading.Lock()

    def offline(self) -> bool:
        """True when only the on-disk cache may be consulted (see ``offline_env_var``)."""
        if self.offline_env_var is None:
            return False
        return os.environ.get(self.offline_env_var, "").strip().lower() in {"1", "true", "yes"}

    def _throttle(self, url: str) -> None:
        """Block until ``min_interval`` seconds have passed since the last call to ``url``'s host."""
        host = urllib.parse.urlsplit(url).netloc
        while True:
            with self._throttle_lock:
                now = time.monotonic()
                wait = self.min_interval - (now - self._last_call.get(host, 0.0))
                if wait <= 0:
                    self._last_call[host] = now
                    return
            time.sleep(wait)

    def _cache_path(self, url: str, tag: str, kind: str = "json") -> Path:
        """Content-hash-keyed cache file path for ``url`` under this client's ``tag`` subdirectory.

        ``kind`` participates in the key: get_text() and get_json() store different payload
        shapes ({"text": ...} vs {"data": ...}), so a single URL-only key let a get_text() entry
        satisfy a later get_json() lookup and return a permanent ``None`` for a URL that answers.
        The legacy "json" key stays digest-only so existing on-disk caches remain valid.
        """
        if not _SAFE_TAG_RE.match(tag) or tag in {".", ".."}:
            raise ValueError(f"CachedHttpClient: unsafe cache tag {tag!r}; expected [A-Za-z0-9_.-]+ (it is used as a directory name under cache_dir)")
        keyed = url if kind == "json" else f"{kind}:{url}"
        digest = hashlib.sha256(keyed.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / tag / f"{digest}.json"

    def _write_cache_entry(self, path: Path, payload: Dict[str, Any]) -> None:
        """Atomically write a JSON cache entry, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        atomic_write_bytes(str(path), lambda f: f.write(body))

    def _read_capped(self, resp: Any, url: str) -> Optional[bytes]:
        """Read at most ``self.max_bytes`` from ``resp``, returning None (with a warning) if the body
        is larger.

        A bare ``resp.read()`` sizes the allocation from the remote side: a hostile or merely broken
        endpoint answering with a multi-gigabyte chunked body (or an effectively endless stream)
        allocates until the process is OOM-killed or the cache volume fills, and neither the per-host
        throttle nor the retry cap helps because it all happens inside ONE successful response. This
        is the same remote-controlled-quantity cap as ``MAX_RETRY_AFTER_SECONDS`` above. The
        ``Content-Length`` pre-check is an optimisation only -- it is remote-supplied and may be
        absent or lie, so the streaming loop is what actually enforces the bound.
        """
        if self.max_bytes is None:
            return cast(bytes, resp.read())
        declared = resp.headers.get("Content-Length") if getattr(resp, "headers", None) else None
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            logger.warning("cached_client: %s declares Content-Length=%s > max_bytes=%s; refusing", url, declared, self.max_bytes)
            return None
        chunks: List[bytes] = []
        total = 0
        # One byte past the cap is read deliberately: it is what distinguishes "exactly at the limit"
        # (valid) from "truncated at the limit" (must be rejected, since a silently short body would
        # be cached as if complete).
        while total <= self.max_bytes:
            chunk = resp.read(min(_READ_CHUNK_BYTES, self.max_bytes - total + 1))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
        logger.warning("cached_client: response from %s exceeds max_bytes=%s; discarding", url, self.max_bytes)
        return None

    def _fetch_bytes(self, url: str, headers: Optional[Dict[str, str]], timeout: float, retries: int) -> Optional[bytes]:
        """Throttled GET with exponential backoff; returns None on a permanent failure."""
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, **(headers or {})})
        for attempt in range(retries):
            self._throttle(url)
            try:
                with urlopen_checked(req, timeout=timeout, allowed_schemes=self.allowed_schemes) as resp:
                    return self._read_capped(resp, url)
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_HTTP_CODES and attempt < retries - 1:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    backoff = float(2**attempt)
                    delay = backoff
                    if retry_after and retry_after.isdigit():
                        requested = float(retry_after)
                        if requested > MAX_RETRY_AFTER_SECONDS:
                            logger.warning(
                                "cached_client: server asked for Retry-After=%ss (> cap %ss) on %s; using local backoff %.0fs instead",
                                retry_after, MAX_RETRY_AFTER_SECONDS, url, backoff,
                            )
                        else:
                            delay = requested
                    time.sleep(delay)
                    continue
                logger.debug("cached_client: HTTP %s for %s", exc.code, url)
                return None
            # `OSError` catches a connection reset DURING the response read (WinError 10054 on
            # Windows), which surfaces as a bare `ConnectionResetError` -- an `OSError` that is
            # neither a `URLError` nor a `TimeoutError` and would otherwise escape this handler and
            # abort the caller on a transient network blip. `URLError` is listed explicitly rather
            # than relying on its `OSError` ancestry, since that hierarchy isn't a guaranteed
            # contract to depend on silently.
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                logger.debug("cached_client: %s for %s", exc, url)
                return None
        return None

    def get_json(
        self,
        url: str,
        tag: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30,
        retries: int = 3,
        cache_failures: bool = True,
    ) -> Any:
        """GET a JSON document, served from cache when present.

        Returns ``None`` on a permanent failure (404, malformed body, offline+no-cache) so
        callers can treat "no result" and "service unavailable" the same way -- neither should
        abort a batch job. A negative result is cached too by default (``cache_failures=True``),
        so a term/URL no service has an answer for doesn't re-pay the round trip every run; pass
        ``cache_failures=False`` when a failure is expected to be transient (e.g. an integrity
        check that must not mistake a temporary outage for a durable "no").
        """
        path = self._cache_path(url, tag)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload.get("data")
            except (json.JSONDecodeError, OSError):
                pass  # a corrupt cache entry is refetched rather than trusted
        if self.offline():
            logger.debug("cached_client %s: offline, no cache entry for %s", tag, url)
            return None
        raw = self._fetch_bytes(url, headers={"Accept": "application/json", **(headers or {})}, timeout=timeout, retries=retries)
        data: Any = None
        if raw is not None:
            try:
                # errors="replace": a mis-encoded body is a permanent failure for this URL, not a
                # reason to abort the caller's batch with an escaping UnicodeDecodeError.
                data = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                logger.debug("cached_client %s: invalid JSON for %s: %s", tag, url, exc)
        if data is not None or cache_failures:
            self._write_cache_entry(path, {"url": url, "data": data})
        return data

    def get_text(
        self,
        url: str,
        tag: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30,
        retries: int = 3,
        cache_failures: bool = False,
    ) -> Optional[str]:
        """GET a non-JSON document (XML, CSV, HTML), same cache/throttle/offline/retry rules as
        :meth:`get_json`. ``cache_failures`` defaults to False here: unlike a vocabulary lookup
        where "no result" is a stable fact, a plain-text fetch failure is more often transient and
        caching it would make an outage look like a durable answer.
        """
        path = self._cache_path(url, tag, kind="text")
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                body = payload.get("text")
                if isinstance(body, str):
                    return body
            except (json.JSONDecodeError, OSError):
                pass
        if self.offline():
            logger.debug("cached_client %s: offline, no cache entry for %s", tag, url)
            return None
        raw = self._fetch_bytes(url, headers=headers, timeout=timeout, retries=retries)
        text = raw.decode("utf-8", errors="replace") if raw is not None else None
        if text is not None or cache_failures:
            self._write_cache_entry(path, {"url": url, "text": text})
        return text
