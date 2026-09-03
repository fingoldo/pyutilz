"""Filemaker integration utils."""

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import Optional

from time import sleep
from http import HTTPStatus
from base64 import b64encode

from pyutilz import web
from pyutilz.core.pythonlib import get_attr

filemaker_url, filemaker_username, filemaker_password = None, None, None


def init(m_filemaker_url: str, m_filemaker_username: str, m_filemaker_password: str) -> None:
    """Store FileMaker connection credentials in module globals and open a session.

    Must be called once before any other function; subsequent calls re-authenticate.

    Raises:
        RuntimeError: no session token could be obtained. The token was previously discarded, so
            a failed init() returned normally while ``web.connect``'s Basic-auth headers stayed in
            place -- every later post_filemaker_record then 401'd with a misleading ValueError.
    """
    global filemaker_url, filemaker_username, filemaker_password
    filemaker_url = m_filemaker_url
    filemaker_username = m_filemaker_username
    filemaker_password = m_filemaker_password
    token = get_session_token(username=filemaker_username, password=filemaker_password)
    if token is None:
        raise RuntimeError(f"filemaker init: could not obtain a session token from {m_filemaker_url} -- not initialized.")


def get_session_token(username: Optional[str] = None, password: Optional[str] = None, max_retries: int = 10, sleep_int_seconds: int = 10) -> Optional[str]:
    """Obtain a FileMaker Data API session token, retrying on transient failures.

    Falls back to module-level credentials when username/password are not supplied.
    Returns the bearer token, or None if all attempts fail.
    """
    # Resolve to module globals at call time (def-time binding would capture the pre-init() None values).
    if username is None:
        username = filemaker_username
    if password is None:
        password = filemaker_password
    if username is None or password is None or filemaker_url is None:
        raise ValueError("filemaker username/password/url not configured; call init() first or pass them explicitly")
    web.connect(
        m_template_headers={
            "Authorization": "Basic " + b64encode((username + ":" + password).encode()).decode(),
            "Content-Type": "application/json",
        },
    )
    for attempt in range(max_retries):
        res = web.get_url(filemaker_url + "/sessions", b_random_ua=False, verb="post")
        if res is None:
            logger.warning("No response while getting filemaker session token")
        elif res.status_code != HTTPStatus.OK:
            logger.warning("Error %s while getting filemaker session token: %s", res.status_code, res.text)
        else:
            res = res.json()
            response_field = get_attr(res, "response", {})
            def_token = get_attr(response_field if isinstance(response_field, dict) else {}, "token")
            if not def_token or not isinstance(def_token, str):
                # Log the response's SHAPE only, never its content: this branch fires exactly when the
                # token is not a str at response.token -- which includes a deployment returning a live
                # token under a different key/nesting, so `res` may still carry the bearer token itself.
                # Logs are durable and aggregated; credentials must not reach them.
                logger.warning(
                    "Empty/invalid filemaker session token; response keys=%s",
                    sorted(res) if isinstance(res, dict) else type(res).__name__,
                )
            else:
                web.connect(m_template_headers={"Authorization": "Bearer " + def_token, "Content-Type": "application/json"})
                return def_token
        if attempt < max_retries - 1:
            # No sleep after the LAST attempt: there is nothing left to wait for, and the defaults
            # added a full 10 s of dead time to every failed boot.
            sleep(sleep_int_seconds)
    return None


def simplify_types(obj: dict, sep=",") -> dict:
    """FM does not accept fields that are lists & dicts easily.

    Returns a NEW dict; ``obj`` is left untouched. ``obj.copy()`` used to copy only the iteration
    view, so the caller's own dict silently lost its list/dict fields and its None-valued keys.
    """
    res = {}
    for key, val in obj.items():
        if isinstance(val, list):
            res[key] = sep.join([str(el) for el in val])
        elif isinstance(val, dict):
            res[key] = str(val)
        elif val is not None:
            res[key] = val
    return res


def post_filemaker_record(filemaker_url: str, layout: str, data: dict, num_attempts: int = 3) -> Optional[bool]:
    """Attempts to post a record several times, refreshing the auth token if needed.

    Returns True on success; raises ValueError if the token cannot be refreshed or the
    request keeps failing.
    """
    last_error = None
    for _ in range(num_attempts):
        try:
            res = web.get_url(filemaker_url + f"/layouts/{layout}/records/", verb="post", json=data)
        except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional (retry next attempt)
            last_error = e
            logger.error("Exception %s when inserting into filemaker object %s", e, data)
        else:
            if res is None:
                last_error = ValueError("web.get_url() returned no response")
                logger.error("No response when inserting into filemaker object %s", data)
                continue
            if res.status_code == HTTPStatus.OK:
                return True
            else:
                if res.status_code == 401 and "Invalid FileMaker Data API token" in res.text:
                    logger.warning("Refreshing the filemaker token...")
                    if not get_session_token(username=filemaker_username, password=filemaker_password):
                        raise ValueError("Could not refresh the token")
                else:
                    logger.error("Exception %s when inserting object details %s", str(res.status_code) + ": " + res.text, data)
                    raise ValueError(f"Filemaker insert failed with status {res.status_code}: {res.text}")
    # All attempts exhausted via transient errors: surface the failure instead of returning None silently.
    raise ValueError(f"Filemaker insert failed after {num_attempts} attempts: {last_error}")
