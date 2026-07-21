"""Helpers for safely reading/writing Delta Lake tables, including local-path detection and file-locked writes."""

import os
import tempfile
import hashlib
import logging
from typing import Any
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


def safe_delta_write(path: str, delta_op_func, *, lock_timeout: int = 1200, lock_suffix=".lock"):
    """
    Wraps any Delta Lake operation (write_deltalake, merge+execute) with local file locking.

    Parameters:
        path (str): Delta table path.
        delta_op_func (callable): A function that performs the actual Delta operation.
        lock_timeout (int): How many seconds to wait for the lock before skipping.
        lock_suffix (str): Suffix for the lock filename.

        Usage Examples
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
        path_hash = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()
        lock_file = os.path.join(tempfile.gettempdir(), f"{path_hash}{lock_suffix}")
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
