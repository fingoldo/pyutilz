# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed
ensure_installed("redis")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any

import redis
from time import sleep
from random import random
from redis.exceptions import ConnectionError

rc = None

def rconnect (redis_host:str, redis_port:int, redis_db_name:str, redis_db_pwd:str, decode_responses:bool=True):
    global rc
    rc = redis.Redis(host=redis_host, port=redis_port, db=int(redis_db_name), password=redis_db_pwd, decode_responses=decode_responses)

    return rc

def rclose() -> None:
    """Close the global Redis connection and release its pool, if any."""
    global rc
    if rc is not None:
        try:
            rc.close()
        except Exception as e:
            logger.exception(e)
        finally:
            rc = None

def rexecute (method_name:str,*args,**kwargs) -> Any:
    """
    Safely execute any Redis command, not worrying about temporary network/server issues.

    Transient ConnectionErrors are retried with a small random backoff. A permanent error
    (e.g. missing connection, unknown method) is logged and re-raised instead of busy-looping forever.
    """
    if rc is None:
        raise RuntimeError("Redis connection is not established. Call rconnect(...) first.")

    try:
        method = getattr(rc, method_name)
    except AttributeError as e:
        logger.exception(e)
        raise

    res = None
    while True:
        try:
            res = method(*args, **kwargs)
        except ConnectionError as e:
            # Transient: retry with backoff.
            logger.exception(e)
            sleep(1 * random())  # nosec B311 - jitter for retry backoff timing, not a security/crypto use
        except Exception as e:
            # Permanent error: log and propagate instead of busy-looping forever.
            logger.exception(e)
            raise
        else:
            break
    return res
