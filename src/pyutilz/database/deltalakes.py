"""Helpers for safely reading/writing Delta Lake tables, including local-path detection and file-locked writes."""

import os
import tempfile
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

        lock_file = os.path.join(tempfile.gettempdir(), f"{os.path.basename(path).replace('/', '_')}{lock_suffix}")
        lock = FileLock(lock_file)

        try:
            with lock.acquire(timeout=lock_timeout):
                logger.debug("Acquired lock for local Delta path: %s", path)
                return delta_op_func()
        except Timeout:
            logger.warning("Timeout while waiting for lock on %s. Skipping operation.", path)
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
