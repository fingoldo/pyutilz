""" Filemaker integration utils.
"""
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

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

from time import sleep
from http import HTTPStatus
from base64 import b64encode

from pyutilz import web
from pyutilz.pythonlib import get_attr

filemaker_url, filemaker_username, filemaker_password = None, None, None


def init(m_filemaker_url: str, m_filemaker_username: str, m_filemaker_password: str):
    global filemaker_url, filemaker_username, filemaker_password
    filemaker_url = m_filemaker_url
    filemaker_username = m_filemaker_username
    filemaker_password = m_filemaker_password
    get_session_token(username=filemaker_username, password=filemaker_password)


def get_session_token(username: str = filemaker_username, password: str = filemaker_password, max_retries: int = 10, sleep_int_seconds: int = 10) -> str:
    web.connect(
        m_template_headers={
            "Authorization": "Basic " + b64encode((username + ":" + password).encode()).decode(),
            "Content-Type": "application/json",
        },
    )
    for _ in range(max_retries):
        res = web.get_url(filemaker_url + "/sessions", b_random_ua=False, verb="post")
        if res.status_code != HTTPStatus.OK:
            logger.warning("Error %s while getting filemaker session token: %s", res.status_code, res.text)
        else:
            res = res.json()
            def_token = get_attr(get_attr(res, "response", {}), "token")
            if not def_token:
                logger.warning("Empty filemaker session token: %s", res)
            else:
                web.connect(m_template_headers={"Authorization": "Bearer " + def_token, "Content-Type": "application/json"})
                return def_token
        sleep(sleep_int_seconds)


def simplify_types(obj: dict, sep=",") -> dict:
    """FM does not accept fields that are lists & dicts easily"""

    for key, val in obj.copy().items():
        if isinstance(val, list):
            obj[key] = sep.join([str(el) for el in val])
        elif isinstance(val, dict):
            obj[key] = str(val)
        if val is None:
            del obj[key]
    return obj


def post_filemaker_record(filemaker_url: str, layout: str, data: dict, num_attempts: int = 3):
    """Attempts to post a record several times, refreshing the auth token if needed."""
    for _ in range(num_attempts):
        try:
            res = web.get_url(filemaker_url + f"/layouts/{layout}/records/", verb="post", json=data)
        except Exception as e:
            logger.error("Exception %s when inserting into filemaker object %s", e, data)
        else:
            if res.status_code == HTTPStatus.OK:
                return True
            else:
                if res.status_code == 401 and "Invalid FileMaker Data API token" in res.text:
                    logger.warning("Refreshing the filemaker token...")
                    if not get_session_token(username=filemaker_username, password=filemaker_password):
                        raise ValueError("Could not refresh the token")
                else:
                    logger.error("Exception %s when inserting object details %s", str(res.status_code) + ": " + res.text, data)
                    raise ValueError
