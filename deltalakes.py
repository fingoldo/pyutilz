import os
import logging
from urllib.parse import urlparse
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


def is_local_path(path: str) -> bool:
    parsed = urlparse(path)
    # If there's no scheme or it's explicitly "file"
    if parsed.scheme in ("", "file"):
        return not path.startswith(("s3://", "azure://"))

    # Special case: Windows drive letter (e.g., "R:\...")
    if os.name == "nt" and len(parsed.scheme) == 1 and parsed.scheme.isalpha():
        return True

    return False


def safe_delta_write(path: str, delta_op_func, *, lock_timeout: int = 120, lock_suffix=".lock"):
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
        lock_file = os.path.join("/tmp", f"{os.path.basename(path).replace('/', '_')}{lock_suffix}")
        lock = FileLock(lock_file)

        try:
            with lock.acquire(timeout=lock_timeout):
                logger.debug(f"Acquired lock for local Delta path: {path}")
                return delta_op_func()
        except Timeout:
            logger.warning(f"Timeout while waiting for lock on {path}. Skipping operation.")
        except Exception as e:
            logger.exception(f"Delta operation failed on {path}: {e}")
            raise (e)
    else:
        logger.warning(f"Delta operation on non-local path: {path}. Proceeding without lock.")
        try:
            return delta_op_func()
        except Exception as e:
            logger.exception(f"Delta operation failed on {path}: {e}")
