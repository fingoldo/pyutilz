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

from typing import Any, Optional, Union

import redis
from time import sleep
from random import random
from redis.exceptions import AuthenticationError as RedisAuthenticationError
from redis.exceptions import ConnectionError as RedisConnectionError

from pyutilz.database.exceptions import DatabaseConnectionError

# Annotated: an unannotated `rc = None` makes mypy infer the type as exactly `None`, which types
# away the `if old_rc is not None: old_rc.close()` connection-leak guard below and removes all
# checking of redis method names on this module's central object.
rc: Optional[redis.Redis] = None

def rconnect (redis_host:str, redis_port:int, redis_db_name:Union[int,str], redis_db_pwd:str, decode_responses:bool=True):
    """
    Create a Redis connection with the given credentials, store it as the module-level global ``rc``, and return it.

    ``redis_db_name`` is a Redis database INDEX, not a name: Redis databases are numbered (0..15 by
    default), so the value must be an int or a string of digits. Anything else raises ValueError
    before any state is touched.
    """
    global rc
    # Regression fix: int(redis_db_name) used to happen inside the redis.Redis(...) call, i.e. AFTER
    # old_rc had been read but BEFORE the new rc was bound -- a config value like "sessions" raised a
    # bare `invalid literal for int()` naming neither the argument nor the "index, not a name" rule,
    # leaving the previous connection installed and un-closed. Convert first, with a message that says
    # which argument is wrong.
    try:
        redis_db_index = int(redis_db_name)
    except (TypeError, ValueError):
        raise ValueError(f"redis_db_name must be a Redis database INDEX (an integer, e.g. 0), got {redis_db_name!r}") from None
    # Regression fix (meta-test-driven finding, proactive resource-lifecycle audit): a second
    # rconnect() call (e.g. reconnecting with new credentials) used to drop the previous
    # connection with no close(), leaking its connection pool -- same bug class fixed in
    # web.init_vars()/get_new_session() this round.
    old_rc = rc
    rc = redis.Redis(host=redis_host, port=redis_port, db=redis_db_index, password=redis_db_pwd, decode_responses=decode_responses)
    if old_rc is not None:
        try:
            old_rc.close()
        except Exception as e:
            logger.exception(e)

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
