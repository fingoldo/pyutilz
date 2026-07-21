"""Centralised sha256-sidecar verification + safe pickle helpers, shared across projects.

Originally built inside mlframe (four separate pickle entry points had converged on the
same sidecar pattern independently -- an attacker who could plant a file in a cache
directory would be deserialised on the next load without any gate). Promoted here so any
project depending on pyutilz gets the same verified-load primitive instead of
re-implementing it, mlframe included (``mlframe.utils.safe_pickle`` is now a thin
backward-compatible re-export of this module).

THREAT MODEL CAVEAT (read before relying on this for untrusted inputs): the sha256 sidecar is a CORRUPTION / INTEGRITY check, NOT an
authenticity / tamper-resistance control. An attacker who can WRITE to the directory holding the payload can trivially rewrite BOTH the
payload and its ``.sha256`` sidecar so the digest matches again -- the load then succeeds and arbitrary pickle executes. The sidecar only
defends against accidental corruption (truncated copy, mid-rename crash, bit-rot) and against a payload swapped WITHOUT a matching sidecar
rewrite. It buys nothing against an adversary with write access to the same directory. Callers that must load pickles from a directory an
untrusted party can write to need a KEYED integrity control instead -- an HMAC (sidecar = ``HMAC-SHA256(secret_key, payload)``) or a
detached cryptographic signature whose key the attacker does not hold. This module deliberately does not implement that: the keying / key
distribution is the caller's responsibility and out of scope here.

Public surface:

* :func:`verify_sidecar` -- given a path, returns True iff the ``.sha256`` sidecar
  exists and matches the file digest. Missing sidecar fails closed by default;
  the env var named by ``env_var`` (default ``PYUTILZ_ALLOW_UNVERIFIED_PICKLE``) set
  truthy opts back into the legacy permissive behaviour with a WARN log.
* :func:`write_sidecar` -- computes sha256 of a path and writes the
  ``<path>.sha256`` companion file. Idempotent.
* :func:`safe_load` -- verifies + ``pickle.load``. Raises
  :class:`PickleVerificationError` if the sidecar is missing/corrupt (unless
  the env opt-in is set). Returns the unpickled object on success.
* :func:`safe_dump` -- ``pickle.dump`` + auto-writes the sidecar so callers of
  :func:`safe_load` will succeed on subsequent reads.

Env vars:

* ``PYUTILZ_ALLOW_UNVERIFIED_PICKLE`` -- truthy ("1", "true", "yes") permits a
  missing-sidecar load with a WARN. Corrupt-sidecar (digest mismatch) always
  raises regardless. Callers embedding this in a project with its own historical
  env var name (e.g. mlframe's ``MLFRAME_ALLOW_UNVERIFIED_PICKLE``) can pass
  ``env_var=`` to :func:`verify_sidecar` / :func:`safe_load` to check that name instead.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import pickle  # nosec B403 - this module's whole purpose is guarding pickle.load behind sha256 sidecar verification (see safe_load below and the module-level THREAT MODEL CAVEAT docstring); callers needing tamper-resistance against a co-located attacker must layer keyed integrity on top
import threading
import time
from os.path import isfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "PickleVerificationError",
    "verify_sidecar",
    "write_sidecar",
    "safe_load",
    "safe_dump",
]

DEFAULT_ALLOW_UNVERIFIED_ENV_VAR = "PYUTILZ_ALLOW_UNVERIFIED_PICKLE"

# Per-path locks so a (payload replace + sidecar write) pair is atomic AS A UNIT across threads
# in this process. Without this, two threads writing the same path can interleave such that the
# temporally-earlier thread's write_sidecar() call (which re-reads the CURRENT on-disk payload,
# not necessarily its own) lands its file write AFTER the later thread's, leaving a sidecar that
# describes a payload no longer on disk -- observed intermittently on Linux CI (safe_load then
# raises PickleVerificationError: sidecar mismatch). The lock only covers this process; it is not
# a cross-process guarantee (same scope the module already limits itself to -- see safe_dump's
# docstring on os.replace being the cross-process-safe primitive).
_path_locks: Dict[str, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _get_path_lock(path: str) -> threading.Lock:
    """Return a process-wide lock keyed by the normalized absolute path, creating it on first use."""
    key = os.path.abspath(path)
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _path_locks[key] = lock
        return lock


class PickleVerificationError(RuntimeError):
    """Raised when a pickle file fails its sha256 sidecar verification.

    Distinguishes verification failures from arbitrary ``pickle.UnpicklingError``
    so callers can surface a security-relevant message in logs without having
    to inspect the error string.
    """


def _truthy(value: str) -> bool:
    """Return True if ``value`` (stripped, lowercased) is one of "1", "true", "yes", "on"."""
    return value.strip().lower() in ("1", "true", "yes", "on")


def _allow_unverified(env_var: str = DEFAULT_ALLOW_UNVERIFIED_ENV_VAR) -> bool:
    """Check whether ``env_var`` is set in the environment to a truthy value."""
    raw = os.environ.get(env_var, "")
    return bool(raw) and _truthy(raw)


def _sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    """Compute the lowercase hex sha256 digest of the file at ``path``, reading in ``chunk``-byte blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_sidecar(path: str, *, allow_unverified: Optional[bool] = None, env_var: str = DEFAULT_ALLOW_UNVERIFIED_ENV_VAR) -> bool:
    """Default-strict (fail-CLOSED) sha256 sidecar check.

    Contract:

    * sidecar present + digest matches -> True
    * sidecar present + digest does NOT match -> False
    * sidecar missing -> False by default (RCE-bypass guard)
    * sidecar missing + the ``env_var`` env var truthy -> True with WARN

    ``allow_unverified`` defaults to ``None`` (consult the env var). Pass ``True`` /
    ``False`` to override the env var explicitly -- callers in tests use this to
    pin behaviour regardless of ambient environment. ``env_var`` lets an embedding
    project keep its own historical opt-in variable name.
    """
    sidecar = path + ".sha256"
    if not isfile(sidecar):
        allow = _allow_unverified(env_var) if allow_unverified is None else bool(allow_unverified)
        if allow:
            logger.warning(
                "verify_sidecar: no .sha256 sidecar for %s -- %s "
                "is set so load proceeds WITHOUT content verification (RCE risk if path is "
                "attacker-reachable). Generate the sidecar with `sha256sum <path> > <path>.sha256` "
                "and unset the env var to restore default fail-closed behaviour.",
                path,
                env_var,
            )
            return True
        logger.error(
            "verify_sidecar: no .sha256 sidecar for %s -- refusing to load (default-strict). "
            "Generate the sidecar with `sha256sum <path> > <path>.sha256` or set "
            "%s=1 for legacy un-sidecar'd bundles.",
            path,
            env_var,
        )
        return False
    try:
        with open(sidecar, encoding="utf-8") as f:
            expected = f.read().strip().split()[0].lower()
    except (OSError, UnicodeDecodeError, IndexError) as exc:
        logger.error("verify_sidecar: could not read sidecar %s: %s", sidecar, exc)
        return False
    actual = _sha256_of_file(path).lower()
    return hmac.compare_digest(expected, actual)


def write_sidecar(path: str) -> None:
    """Compute sha256 of ``path`` and write the ``<path>.sha256`` companion.

    The on-disk format mirrors the GNU ``sha256sum`` tool: one line, the
    lowercase hex digest, two spaces, the basename of the file. Older sidecars
    written as just the digest still parse correctly via the
    ``f.read().strip().split()[0]`` parse path in :func:`verify_sidecar`.
    """
    digest = _sha256_of_file(path)
    sidecar = path + ".sha256"
    basename = os.path.basename(path)
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {basename}\n")


def safe_load(path: str, *, allow_unverified: Optional[bool] = None, env_var: str = DEFAULT_ALLOW_UNVERIFIED_ENV_VAR) -> Any:
    """Verify the sidecar, then ``pickle.load(path)``.

    Raises :class:`PickleVerificationError` if verification fails. Pass
    ``allow_unverified=True`` to bypass the verification (with a WARN) without
    setting an env var -- useful in tests that exercise the legacy fallback.
    """
    if not verify_sidecar(path, allow_unverified=allow_unverified, env_var=env_var):
        raise PickleVerificationError(
            f"safe_load: refusing to unpickle {path!r}; sha256 sidecar missing or mismatch. "
            f"Run write_sidecar(path) on a trusted copy or set {env_var}=1 to bypass (loud WARN)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec B301 - verify_sidecar() above is this module's whole purpose; this is the necessarily-unsafe primitive it gates


def safe_dump(obj: Any, path: str, *, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """``pickle.dump`` to ``path`` ATOMICALLY and auto-write the matching ``.sha256`` sidecar.

    Writes to a per-process-and-thread temp file, fsyncs, then ``os.replace`` onto ``path`` (atomic rename on
    POSIX and Windows for a same-dir target), so a crash mid-dump leaves the previous file intact instead of a
    truncated one, and two concurrent same-key writers -- even two threads in the same process -- never
    interleave into a corrupt file (last full writer wins). The whole payload-replace + sidecar-write sequence
    is additionally serialized per-path (see :func:`_get_path_lock`) so the sidecar written by one thread can
    never be overwritten by a stale digest from another thread that started earlier but finishes writing later
    -- without this, "last full writer wins" held for the payload file alone but not for the (payload, sidecar)
    pair together, since each writer's sidecar step re-reads whatever payload happens to be on disk at that
    moment.

    On Windows, ``os.replace`` onto a target that a second thread is renaming onto in the same instant can
    transiently raise ``PermissionError`` (sharing violation, WinError 5) even though each writer has its own
    temp file -- this is an OS-level race in ``MoveFileEx`` itself, not a corruption. That transient error is
    retried with a short backoff so "last full writer wins" holds without a spurious raise; a still-failing
    replace after the retry budget is a real error and propagates.
    """
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with _get_path_lock(path):
        try:
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=protocol)
                f.flush()
                os.fsync(f.fileno())
            _replace_with_retry(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        write_sidecar(path)


def _replace_with_retry(src: str, dst: str, *, attempts: int = 10, base_delay: float = 0.01) -> None:
    """``os.replace(src, dst)`` retrying transient Windows sharing-violation ``PermissionError``.

    Only ``PermissionError`` is retried (the concurrent-replace race); any other exception propagates
    immediately on the first attempt.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:  # noqa: PERF203 -- per-attempt retry loop; the try/except IS the retry mechanism
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
