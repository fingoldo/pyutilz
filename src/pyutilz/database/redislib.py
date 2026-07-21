"""Thin wrapper around a single global Redis connection, with retrying command execution."""
# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any

import redis
from time import sleep
from random import random
from redis.exceptions import AuthenticationError as RedisAuthenticationError
from redis.exceptions import ConnectionError as RedisConnectionError

from pyutilz.database.exceptions import DatabaseConnectionError

rc = None

def rconnect (redis_host:str, redis_port:int, redis_db_name:str, redis_db_pwd:str, decode_responses:bool=True):
    """
    Create a Redis connection with the given credentials, store it as the module-level global ``rc``, and return it.
    """
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

def rexecute(method_name: str, *args, max_retries: Any = None, **kwargs) -> Any:
    """
    Safely execute any Redis command, not worrying about temporary network/server issues.

    Transient ConnectionErrors are retried with a small random backoff (up to ``max_retries``
    times; ``None``, the default, retries indefinitely -- pass an int to bound it). A permanent
    error (e.g. missing connection, unknown method, or authentication failure) is logged and
    re-raised instead of retrying.

    WARNING: ``method_name``/``args``/``kwargs`` must never be built from external/user-controlled
    input -- this is a generic reflection-based dispatcher onto the ENTIRE redis-py client API
    (including e.g. ``eval``/``evalsha`` for arbitrary Lua, ``config_set``, ``flushall``,
    ``flushdb``, ``shutdown``), with no allow-list. "Safely" in this docstring refers only to the
    retry-on-transient-ConnectionError behaviour below, not command authorization.
    """
    if rc is None:
        raise DatabaseConnectionError("Redis connection is not established. Call rconnect(...) first.")

    try:
        method = getattr(rc, method_name)
    except AttributeError as e:
        logger.exception(e)
        raise

    res = None
    attempt = 0
    while True:
        try:
            res = method(*args, **kwargs)
        except RedisAuthenticationError:  # noqa: PERF203 -- per-attempt retry loop; the try/except IS the retry mechanism
            # Regression fix: AuthenticationError (wrong password) is a SUBCLASS of
            # ConnectionError in redis-py -- a permanent, non-transient failure was previously
            # caught by the same "retry with jittered backoff" branch as a genuine network
            # blip, looping forever with no exception ever propagating to the caller.
            logger.exception("Redis authentication failed -- not retrying (permanent error)")
            raise
        except RedisConnectionError as e:
            # Transient: retry with backoff, bounded by max_retries if given.
            attempt += 1
            # Regression fix: this used to log a full ERROR-level traceback on EVERY attempt --
            # with max_retries=None (the documented default, "retries indefinitely"), a real
            # outage produced roughly one full traceback per second, forever, drowning out every
            # other log signal and potentially blowing through a log-aggregator's ingestion
            # quota. Only the first failure of a streak gets the full traceback; subsequent
            # identical retries log a one-line DEBUG summary until success or the final give-up.
            if attempt == 1:
                logger.exception(e)
            else:
                logger.debug("rexecute: still failing (attempt %d): %s", attempt, e)
            if max_retries is not None and attempt >= max_retries:
                logger.error("rexecute: giving up after %d attempts", attempt)
                raise
            sleep(1 * random())  # nosec B311 - jitter for retry backoff timing, not a security/crypto use
        except Exception as e:
            # Permanent error: log and propagate instead of busy-looping forever.
            logger.exception(e)
            raise
        else:
            break
    return res
