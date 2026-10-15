"""Helpers for safely reading/writing Delta Lake tables, including local-path detection and file-locked writes."""

import os
import tempfile
import hashlib
import logging
from typing import Any, Callable
from urllib.parse import urlparse

FileLock: Any
Timeout: Any
try:
    from filelock import FileLock, Timeout  # type: ignore[no-redef]
except ImportError:
    FileLock = None
    Timeout = None

logger = logging.getLogger(__name__)


def is_local_path(path: str) -> bool:
    """Determine whether a path points to a local filesystem location rather than a remote store (e.g. s3://, azure://)."""
    parsed = urlparse(path)
    # If there's no scheme or it's explicitly "file"
    if parsed.scheme in ("", "file"):
        return not path.startswith(("s3://", "azure://"))

    # Special case: Windows drive letter (e.g., "R:\...")
    if os.name == "nt" and len(parsed.scheme) == 1 and parsed.scheme.isalpha():
        return True

    return False


def _lock_dir() -> str:
    """Per-user directory for lock files, created mode 0700 under the temp dir (X-12).

    A predictable lock name directly in the shared temp dir let another local user pre-create it as a
    symlink or hold it to block writers. On POSIX the directory must be owned by the current user and
    must not be a symlink, otherwise a RuntimeError is raised rather than locking inside a foreign dir.
    """
    if hasattr(os, "getuid"):
        suffix = str(os.getuid())  # type: ignore[attr-defined]  # POSIX only, guarded by hasattr
    else:
        suffix = os.environ.get("USERNAME", "user")
    base = os.path.join(tempfile.gettempdir(), f"pyutilz-delta-locks-{suffix}")
    os.makedirs(base, mode=0o700, exist_ok=True)
    if os.path.islink(base):
        raise RuntimeError(f"safe_delta_write: lock directory {base!r} is a symlink; refusing to lock inside it")
    if hasattr(os, "getuid"):
        st = os.stat(base)
        if st.st_uid != os.getuid():  # type: ignore[attr-defined]  # POSIX only, guarded by hasattr
            raise RuntimeError(f"safe_delta_write: lock directory {base!r} is owned by another user; refusing to use it")
        os.chmod(base, 0o700)
    return base


def safe_delta_write(path: str, delta_op_func: Callable[[], Any], *, lock_timeout: int = 1200, lock_suffix: str = ".lock") -> Any:
    """
    Wraps any Delta Lake operation (write_deltalake, merge+execute) with local file locking.

    Lock files live in a per-user, mode-0700 directory under ``tempfile.gettempdir()``.

    Args:
        path: Delta table path.
        delta_op_func: A function that performs the actual Delta operation.
        lock_timeout: Seconds to wait for the lock. When it expires, a ``TimeoutError`` is raised
            (the operation is not run); no value is returned in that case.
        lock_suffix: Suffix for the lock filename.

    Returns:
        Whatever ``delta_op_func`` returns.

    Raises:
        ImportError: ``path`` is local and the optional ``filelock`` package is not installed.
        TimeoutError: the lock was not acquired within ``lock_timeout`` seconds (A-04).
        Exception: any exception raised by ``delta_op_func`` is re-raised unchanged.

    Examples:
        🔁 For .merge().when_not_matched_insert_all().execute():

        def merge_ads_static():
            return DeltaTable(ADS_STATIC_PATH).merge(
                static_df,
                predicate="t.id = s.id",
                source_alias="s",
                target_alias="t",
                writer_properties=DELTALAKE_OPTIONS.get("writer_properties")
            ).when_not_matched_insert_all().execute()

        safe_delta_write(ADS_STATIC_PATH, merge_ads_static)

        📝 For write_deltalake() appends:

        def write_market_ads():
            return write_deltalake(
                MARKET_ADS_PATH,
                market_df,
                mode="append",
                partition_by=["date"],
                **DELTALAKE_OPTIONS
            )

        safe_delta_write(MARKET_ADS_PATH, write_market_ads)
    """
    if is_local_path(path):
        if FileLock is None:
            raise ImportError("safe_delta_write on a local path requires the 'filelock' package; pip install filelock")

        # Regression fix: keying the lock filename on basename(path) alone means any two DISTINCT
        # Delta tables that happen to share a basename (e.g. /data/region-us/orders and
        # /data/region-eu/orders) map to the exact same OS-wide lock file in the shared temp
        # directory, forcing one to block on the other's lock (up to lock_timeout, default 20
        # minutes) despite having no real data-race between them. Hashing the full normalized
        # path makes the lock filename unique per actual table.
        #
        # os.path.normcase() (2026-07-21 audit round 2): os.path.abspath() never changes case,
        # but NTFS (Windows' default filesystem) is case-INSENSITIVE-but-preserving -- without
        # this, "C:\Data\orders" and "c:\data\ORDERS" (the exact same table on disk) hash to two
        # DIFFERENT lock files, so two writers could both acquire "their own" lock and commit to
        # the same underlying Delta log concurrently, exactly the corruption this function exists
        # to prevent. normcase() is a no-op on POSIX (case-sensitive filesystems) and lowercases +
        # backslash-normalizes on Windows, so both spellings collapse to one lock file everywhere.
        path_hash = hashlib.sha256(os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()
        lock_file = os.path.join(_lock_dir(), f"{path_hash}{lock_suffix}")
        lock = FileLock(lock_file)

        try:
            with lock.acquire(timeout=lock_timeout):
                logger.debug("Acquired lock for local Delta path: %s", path)
                return delta_op_func()
        except Timeout:
            # Regression fix: previously fell through to an implicit `return None` here,
            # indistinguishable from delta_op_func() itself legitimately returning None (which
            # write_deltalake() does, per this function's own docstring example) -- a caller
            # could not tell "the write succeeded" apart from "the write was silently skipped
            # because another process held the lock for the full timeout window".
            logger.warning("Timeout while waiting for lock on %s. Skipping operation.", path)
            raise TimeoutError(f"safe_delta_write: could not acquire lock for {path!r} within {lock_timeout}s; operation was skipped") from None
        except Exception as e:
            logger.exception("Delta operation failed on %s: %s", path, e)
            raise (e)
    else:
        logger.warning("Delta operation on non-local path: %s. Proceeding without lock.", path)
        try:
            return delta_op_func()
        except Exception as e:
            logger.exception("Delta operation failed on %s: %s", path, e)
            raise
