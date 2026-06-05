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
        return f"{self.prefix}/{fingerprint}.json" if self.prefix else f"{fingerprint}.json"

    def _get_client(self):
        if self._client is None and not self._unavailable:
            try:
                import boto3  # lazy: not a hard dependency

                self._client = boto3.client("s3")
            except Exception as e:  # boto3 absent / no creds / config error
                logger.debug("S3Backend unavailable (%s); kernel cache stays local-only", e)
                self._unavailable = True
        return self._client

    def read(self, fingerprint: str) -> Optional[dict]:
        client = self._get_client()
        if client is None:
            return None
        try:
            import json

            obj = client.get_object(Bucket=self.bucket, Key=self._key(fingerprint))
            return json.loads(obj["Body"].read())
        except Exception as e:  # NoSuchKey, network, parse -- all degrade to miss
            logger.debug("S3Backend.read(%s) miss/error: %s", fingerprint, e)
            return None

    def write(self, fingerprint: str, payload: dict) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            import json

            client.put_object(
                Bucket=self.bucket,
                Key=self._key(fingerprint),
                Body=json.dumps(payload, sort_keys=True).encode("utf-8"),
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
    spec = spec or os.environ.get(_REMOTE_ENV)
    if not spec:
        return None
    try:
        if spec.startswith("s3://"):
            rest = spec[len("s3://"):]
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
