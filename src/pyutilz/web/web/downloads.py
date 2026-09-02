"""Bulk and streaming download helpers: a ``grequests``-backed parallel fetcher and a retrying
streaming file downloader.

``grequests`` stays a function-local import: it monkeypatches sockets via gevent at import time,
so pulling it in at module scope would impose that on every consumer of ``pyutilz.web``.
"""

import http
import os
from typing import Callable, List, Optional, Sequence

# Imported for the ``Optional[requests.Response]`` annotation only -- the actual GET goes through
# ``_facade.requests`` so that patching ``pyutilz.web.web.requests`` is honoured here.
import requests

from ._common import _error_log_throttle, logger

# See ipinfo.py for why the parent is reached through `import <parent> as _facade`.
import pyutilz.web.web as _facade


def download_in_parallel(
    urls_to_process: Optional[Sequence],
    func: Callable,
    headers: Optional[dict] = None,
    nparallel_downloads: int = 3,
    report_each: int = 10,
    request_timeout: Optional[float] = 30,
) -> Optional[List]:
    """Fetch ``urls_to_process`` concurrently via ``grequests`` and call ``func(response, url)`` on each successful (HTTP 200) response.

    Returns the list of URLs that errored (non-200 status, exception in ``func``, or None
    response), or None if ``urls_to_process`` is None/empty.
    """
    import grequests

    if urls_to_process is None:
        return None
    if len(urls_to_process) == 0:
        return None

    # The aggregate "Processed N urls, n_errored=M" line (every report_each iterations, already
    # unconditional) and the final "Finished!" summary still report the true total either way.
    _MAX_LOGGED_ERRORS = 20
    n_processed = 0
    errored_urls = []
    # request_timeout=: grequests.get() previously had no timeout at all -- a single stalled URL
    # (server accepts the connection but never responds) blocked grequests.map() from ever
    # returning, hanging the whole batch regardless of how many OTHER urls had already succeeded.
    rs = (grequests.get(sub_url, verify=True, allow_redirects=True, headers=headers, timeout=request_timeout) for sub_url in urls_to_process)
    logger.info("Started crawling! nlinks=%d", len(urls_to_process))
    # pbar=tqdmu(urls_to_process)
    for resp, sub_url in zip(grequests.map(rs, size=nparallel_downloads), urls_to_process):
        n_processed = n_processed + 1
        # None-check MUST come before touching any attribute of resp: grequests.map() puts None
        # into the results for a request that raised (no exception_handler passed here), exactly
        # the case this function's own docstring says it handles ("None response"). Previously
        # `resp.history` was dereferenced first, raising an uncaught AttributeError that aborted
        # the ENTIRE batch (including URLs not yet processed) instead of recording just this one
        # URL as errored.
        if resp is None:
            errored_urls.append(sub_url)
            if _error_log_throttle(len(errored_urls), _MAX_LOGGED_ERRORS):
                logger.error("Response is None for url %s", sub_url)
        else:
            final_status_code = resp.history[-1].status_code if len(resp.history) > 0 else resp.status_code
            if resp.status_code == http.HTTPStatus.OK:
                try:
                    func(resp, sub_url)
                except Exception as e:
                    errored_urls.append(sub_url)
                    if _error_log_throttle(len(errored_urls), _MAX_LOGGED_ERRORS):
                        logger.error("Error processing url %s: %s", sub_url, e)
            else:
                errored_urls.append(sub_url)
                if _error_log_throttle(len(errored_urls), _MAX_LOGGED_ERRORS):
                    logger.error("Error fetching url %s: status_code=%s", sub_url, final_status_code)
        if (n_processed % report_each) == 0:
            # pbar.update(report_each)
            logger.info("Processed %d urls,n_errored=%d", n_processed, len(errored_urls))
    logger.info("Finished! n_processed=%d,n_errored=%d", n_processed, len(errored_urls))
    # pbar.close()
    return errored_urls


def download_to_file(
    url: str,
    filename: str,
    rewrite_existing: bool = True,
    timeout: int = 100,
    chunk_size: int = 1024,
    max_attempts: int = 5,
    headers: Optional[dict] = None,
    exit_codes: tuple = (),
):
    """Dropin replacement for urllib.request.urlretrieve(url, filename) that can hand for indefinitely long."""
    if headers is None:
        headers = {}
    # Make the actual request, set the timeout for no data to 10 seconds and enable streaming responses so we don't have to keep the large files in memory

    last_error: Optional[BaseException] = None
    for attempt in range(max_attempts):
        request: Optional[requests.Response] = None
        try:
            request = _facade.requests.get(url, timeout=timeout, headers=headers, stream=True)
            if request.status_code in exit_codes:
                # A caller-designated fatal status (e.g. 404): requests.get() does not raise on
                # 4xx/5xx, so without this the error page's body would be written as the file.
                return None
            # The GET must be re-issued on every attempt: a requests body is a single-use stream,
            # so retrying only the write loop iterates an already-consumed response, yields
            # nothing, and leaves the (already truncated by open(..., "wb")) file at 0 bytes while
            # the function returns the same value as a successful download.
            with open(filename, "wb") as fh:
                for chunk in request.iter_content(chunk_size * 1024):
                    fh.write(chunk)
        except Exception as e:
            last_error = e
            logger.exception(e)
            if attempt < max_attempts - 1:
                _facade.sleep(10 * _facade.random())  # nosec B311 - random jitter on the download-retry backoff sleep, not security-sensitive
                logger.info("Making another attempt")
        else:
            return None
        finally:
            if request is not None:
                request.close()

    # Every attempt failed. Remove whatever partial/truncated bytes are on disk so that
    # "no file" reliably signals failure instead of a plausible-looking empty artifact.
    try:
        os.remove(filename)
    except OSError:
        pass
    logger.error("download_to_file: all %d attempts failed for %s: %s", max_attempts, url, last_error)
    return None
