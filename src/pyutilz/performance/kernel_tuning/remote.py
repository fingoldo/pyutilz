"""Pluggable remote backend for kernel-tuning caches (shared across machines).

A team can share HW-calibrated tunings: each host's per-fingerprint payload is
an object in a remote store (one object per ``hw_fingerprint``). A machine with
no local entry for its fingerprint reads-through to the remote; a machine that
tunes writes-through so peers with identical HW reuse the measurement.

The backend is pluggable (``RemoteBackend`` ABC). ``S3Backend`` is the default,
selected via ``PYUTILZ_KERNEL_REMOTE=s3://bucket/prefix``; boto3 is imported
lazily so it is not a hard dependency. Any backend error degrades silently to
local-only (logged at debug) -- a flaky network never breaks dispatch.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["RemoteBackend", "S3Backend", "get_remote_backend"]

_REMOTE_ENV = "PYUTILZ_KERNEL_REMOTE"


class RemoteBackend(ABC):
    """Read/write per-fingerprint tuning payloads to a shared store."""

    @abstractmethod
    def read(self, fingerprint: str) -> Optional[dict]:
        """Return the payload for ``fingerprint``, or None if absent/unreachable."""

    @abstractmethod
    def write(self, fingerprint: str, payload: dict) -> bool:
        """Persist ``payload`` for ``fingerprint``. Return True on success."""


class S3Backend(RemoteBackend):
    """One S3 object per fingerprint under ``s3://<bucket>/<prefix>/<fp>.json``.

    boto3 is imported lazily on first use; if it (or credentials) is missing the
    backend reports unavailable and the cache degrades to local-only.
    """

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = None
        self._unavailable = False

    def _key(self, fingerprint: str) -> str:
        """Build the S3 object key for ``fingerprint``, prefixed by ``self.prefix`` when set."""
        return f"{self.prefix}/{fingerprint}.json" if self.prefix else f"{fingerprint}.json"

    def _get_client(self):
        """Lazily create and cache the boto3 S3 client; return None (and latch ``_unavailable``) if boto3/credentials/config setup fails."""
        if self._client is None and not self._unavailable:
            try:
                import boto3  # lazy: not a hard dependency

                # Explicit connect/read timeouts + bounded retries (D9): an
                # unbounded S3 call could otherwise hang the kernel-cache save
                # for minutes on a flaky network. Overridable via env.
                try:
                    from botocore.config import Config

                    connect_t = float(os.environ.get("PYUTILZ_KERNEL_REMOTE_CONNECT_TIMEOUT", "3") or 3)
                    read_t = float(os.environ.get("PYUTILZ_KERNEL_REMOTE_READ_TIMEOUT", "5") or 5)
                    cfg = Config(connect_timeout=connect_t, read_timeout=read_t, retries={"max_attempts": 1})
                    self._client = boto3.client("s3", config=cfg)
                except Exception:
                    # botocore Config missing/odd -> still create a client (better
                    # than no remote); it just lacks the explicit timeouts.
                    self._client = boto3.client("s3")
            except Exception as e:  # boto3 absent / no creds / config error
                logger.debug("S3Backend unavailable (%s); kernel cache stays local-only", e)
                self._unavailable = True
        return self._client

    def read(self, fingerprint: str) -> Optional[dict]:
        """Fetch and JSON-decode the object for ``fingerprint``; return None on any miss/error (client unavailable, missing key, network, parse failure)."""
        client = self._get_client()
        if client is None:
            return None
        try:
            import orjson

            obj = client.get_object(Bucket=self.bucket, Key=self._key(fingerprint))
            return orjson.loads(obj["Body"].read())  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
        except Exception as e:  # NoSuchKey, network, parse -- all degrade to miss
            logger.debug("S3Backend.read(%s) miss/error: %s", fingerprint, e)
            return None

    def write(self, fingerprint: str, payload: dict) -> bool:
        """JSON-encode and upload ``payload`` for ``fingerprint``; return False on any failure (client unavailable, network, etc.) instead of raising."""
        client = self._get_client()
        if client is None:
            return False
        try:
            import orjson

            client.put_object(
                Bucket=self.bucket,
                Key=self._key(fingerprint),
                Body=orjson.dumps(payload, option=orjson.OPT_SORT_KEYS),
                ContentType="application/json",
            )
            return True
        except Exception as e:
            logger.debug("S3Backend.write(%s) error: %s", fingerprint, e)
            return False


def get_remote_backend(spec: Optional[str] = None) -> Optional[RemoteBackend]:
    """Build a backend from a spec string (or the PYUTILZ_KERNEL_REMOTE env).

    Currently supports ``s3://bucket/prefix``. Returns None if no spec is set
    or the scheme is unrecognized (local-only). Never raises.
    """
    # `is not None`, not `or`: a caller passing spec="" to explicitly force local-only
    # regardless of the environment must not be silently overridden by PYUTILZ_KERNEL_REMOTE.
    spec = spec if spec is not None else os.environ.get(_REMOTE_ENV)
    if not spec:
        return None
    try:
        if spec.startswith("s3://"):
            rest = spec[len("s3://") :]
            bucket, _, prefix = rest.partition("/")
            if not bucket:
                logger.warning("%s=%r has no bucket; ignoring", _REMOTE_ENV, spec)
                return None
            return S3Backend(bucket=bucket, prefix=prefix)
        logger.warning("Unrecognized %s scheme: %r (local-only)", _REMOTE_ENV, spec)
        return None
    except Exception as e:
        logger.warning("Failed to build remote backend from %r: %s", spec, e)
        return None
