# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

# from .pythonlib import ensure_installed
# ensure_installed("joblib")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from time import perf_counter
import functools
import requests


def job_completed(job_id: str, status: int = 0, data: str = None, provider: str = "healthchecks.io", api_key: str = None):

    endpoint = ""
    params = None

    if provider == "healthchecks.io":
        if data:
            data = str(data)
        if api_key:
            endpoint = f"https://hc-ping.com/{api_key}/{job_id}/{status}"
        else:
            endpoint = f"https://hc-ping.com/{job_id}/{status}"
    elif provider == "cronitor.io":
        endpoint = f"https://cronitor.link/p/{api_key}/{job_id}"
        if data:
            params = dict(msg=str(data))

    if endpoint:
        try:
            res = requests.post(endpoint, data=data, params=params)
            print(res)
            if res.status_code not in (200, 403, 429):
                # 403=blocked in your country
                # 429=rate limit exceeded
                logger.warning(f"Problem {res.status_code} while sending hearbeat to {provider} on job {job_id}: {res.text}")

        except Exception as e:
            logger.warning(f"Error while sending hearbeat to {provider} on monitor {job_id}: {e}")
    else:
        logger.info(f"Job {job_id} completed with result {e}")


def monitored(
    job_id: str = None,
    status: int = 0,
    log_data: bool = True,
    should_have_data: bool = False,
    duration_field: str = "duration",
    duration_rounding: int = 4,
    provider: str = "healthchecks.io",
    api_key: str = None,
):
    def decorator_logged(func):
        @functools.wraps(func)
        def wrapper_logged(*args, **kwargs):

            if duration_field:
                start_time = perf_counter()

            data = func(*args, **kwargs)
            if should_have_data and not data:
                return data

            assert isinstance(data, (type(None), dict))

            if duration_field:
                if data is None:
                    data = {}
                data[duration_field] = round(perf_counter() - start_time, duration_rounding)

            if log_data:
                logger.info(data)

            local_job_id = job_id
            if not local_job_id:
                local_job_id = func.__name__
            job_completed(job_id=local_job_id, status=status, data=data, provider=provider, api_key=api_key)

            return data

        return wrapper_logged

    return decorator_logged
