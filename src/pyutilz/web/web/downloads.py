"""Bulk and streaming download helpers: a ``grequests``-backed parallel fetcher and a retrying
streaming file downloader.

``grequests`` stays a function-local import: it monkeypatches sockets via gevent at import time,
so pulling it in at module scope would impose that on every consumer of ``pyutilz.web``.
"""

import http
import os
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

# Imported for the ``Optional[requests.Response]`` annotation only -- the actual GET goes through
# ``_facade.requests`` so that patching ``pyutilz.web.web.requests`` is honoured here.
if TYPE_CHECKING:
    import requests

from ..exceptions import UnsafeURLError
from ..url_guard import require_http_url
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
    errored_urls: List = []
    # request_timeout=: grequests.get() previously had no timeout at all -- a single stalled URL
    # (server accepts the connection but never responds) blocked grequests.map() from ever
    # returning, hanging the whole batch regardless of how many OTHER urls had already succeeded.
    # X-16: a non-http(s) URL is reported as errored and never requested.
    fetchable: List = []
    for sub_url in urls_to_process:
        try:
            require_http_url(sub_url)
        except UnsafeURLError:  # noqa: PERF203 -- per-URL validation, one check per entry is intentional
            if _error_log_throttle(len(errored_urls), _MAX_LOGGED_ERRORS):
                logger.warning("download_in_parallel: skipping non-http(s) URL %r", sub_url)
            errored_urls.append(sub_url)
        else:
            fetchable.append(sub_url)
    rs = (grequests.get(sub_url, verify=True, allow_redirects=True, headers=headers, timeout=request_timeout) for sub_url in fetchable)
    logger.info("Started crawling! nlinks=%d", len(fetchable))
    # pbar=tqdmu(urls_to_process)
    for resp, sub_url in zip(grequests.map(rs, size=nparallel_downloads), fetchable):
        n_processed = n_processed + 1
        _handle_parallel_response(resp, sub_url, func, errored_urls, _MAX_LOGGED_ERRORS)
        if (n_processed % report_each) == 0:
            logger.info("Processed %d urls,n_errored=%d", n_processed, len(errored_urls))
    logger.info("Finished! n_processed=%d,n_errored=%d", n_processed, len(errored_urls))
    return errored_urls


MAX_REDIRECTS = 5  # X-16: a download follows at most this many redirects


class RedirectRefusedError(Exception):
    """A redirect chain that would send a download to a non-https hop, or that is longer than MAX_REDIRECTS."""


def _check_redirect_chain(resp, requested_url: str) -> None:
    """Refuse a redirect chain that is too long or that steps from https down to http (audit finding X-16)."""
    # Only str URLs count: a response object without a real URL (a test double, for one) is not a redirect chain.
    history = getattr(resp, "history", None)
    history = history if isinstance(history, (list, tuple)) else []
    urls = [getattr(h, "url", None) for h in history] + [getattr(resp, "url", None)]
    hops = [u for u in urls if isinstance(u, str)]
    if not hops:
        return
    if len(hops) - 1 > MAX_REDIRECTS:
        raise RedirectRefusedError(f"more than {MAX_REDIRECTS} redirects for {requested_url}")
    if requested_url.lower().startswith("https://") and any(h.lower().startswith("http://") for h in hops):
        raise RedirectRefusedError(f"redirect from https to http for {requested_url}")


def _handle_parallel_response(resp, sub_url, func: Callable, errored_urls: List, max_logged_errors: int) -> None:
    """Route one grequests response: call ``func`` on HTTP 200, otherwise record ``sub_url`` in ``errored_urls``."""
    # None-check MUST come before touching any attribute of resp: grequests.map() puts None
    # into the results for a request that raised (no exception_handler passed here), exactly
    # the case this function's own docstring says it handles ("None response"). Previously
    # `resp.history` was dereferenced first, raising an uncaught AttributeError that aborted
    # the ENTIRE batch (including URLs not yet processed) instead of recording just this one
    # URL as errored.
    if resp is None:
        errored_urls.append(sub_url)
        if _error_log_throttle(len(errored_urls), max_logged_errors):
            logger.error("Response is None for url %s", sub_url)
        return
    final_status_code = resp.history[-1].status_code if len(resp.history) > 0 else resp.status_code
    try:
        _check_redirect_chain(resp, sub_url)
    except RedirectRefusedError as e:
        errored_urls.append(sub_url)
        if _error_log_throttle(len(errored_urls), max_logged_errors):
            logger.error("Refused url %s: %s", sub_url, e)
        return
    if resp.status_code == http.HTTPStatus.OK:
        try:
            func(resp, sub_url)
        except Exception as e:
            errored_urls.append(sub_url)
            if _error_log_throttle(len(errored_urls), max_logged_errors):
                logger.error("Error processing url %s: %s", sub_url, e)
    else:
        errored_urls.append(sub_url)
        if _error_log_throttle(len(errored_urls), max_logged_errors):
            logger.error("Error fetching url %s: status_code=%s", sub_url, final_status_code)


def download_to_file(
    url: str,
    filename: str,
    rewrite_existing: bool = True,
    timeout: int = 100,
    chunk_size: int = 1024,
    max_attempts: int = 5,
    headers: Optional[dict] = None,
    exit_codes: tuple = (),
    retry_jitter_seconds: float = 10.0,
) -> bool:
    """Dropin replacement for urllib.request.urlretrieve(url, filename) that can hand for indefinitely long.

    Returns True when ``filename`` holds the requested content (including the "already there and
    ``rewrite_existing=False``" skip), False on a fatal ``exit_codes`` status or when all
    ``max_attempts`` failed. Previously every exit returned None, so a caller could not tell a
    completed download from a wholly failed one without stat-ing the file itself; True/False is
    backwards-compatible for the historic `if download_to_file(...) is None` / falsy-value checks
    only in the failure direction, so callers branching on the result should be reviewed.

    ``rewrite_existing=False`` skips the download entirely when ``filename`` already exists, returning True
    without touching the file - the point of the flag is to make a re-run of a large batch cheap, so the
    request must not be issued at all rather than issued and its body discarded.

    Args:
        url: Address to GET.
        filename: Local path the body is written to.
        rewrite_existing: When False and ``filename`` exists, skip the request entirely.
        timeout: Socket timeout in seconds for each request.
        chunk_size: Read size in KiB per streamed chunk.
        max_attempts: Total attempts before giving up.
        headers: Extra request headers.
        exit_codes: HTTP statuses that fail immediately without retrying.
        retry_jitter_seconds: Upper bound of the random backoff slept between failed attempts (the sleep is
            ``retry_jitter_seconds * random()``). The default of 10.0 is the historical behaviour.

    Returns:
        True when ``filename`` holds the requested content, False otherwise.
    """
    # X-16: only http/https are fetched. A file:// or ftp:// URL built from external data must not be read or
    # fetched. Raises UnsafeURLError before any request is made. Redirect targets are not re-checked (see audit).
    require_http_url(url)
    if not rewrite_existing and os.path.exists(filename):
        logger.debug("Skipping download of %s: %s already exists and rewrite_existing=False", url, filename)
        return True
    if headers is None:
        headers = {}
    # Make the actual request, set the timeout for no data to 10 seconds and enable streaming responses so we don't have to keep the large files in memory

    last_error: Optional[BaseException] = None
    for attempt in range(max_attempts):
        written, error = _try_attempt(url, filename, timeout, chunk_size, headers, exit_codes)
        if error is None:
            return written
        last_error = error
        logger.error(error, exc_info=error)
        if attempt < max_attempts - 1:
            _facade.sleep(retry_jitter_seconds * _facade.random())  # nosec B311 - random jitter on the download-retry backoff sleep, not security-sensitive
            logger.info("Making another attempt")

    # Every attempt failed. Remove whatever partial/truncated bytes are on disk so that
    # "no file" reliably signals failure instead of a plausible-looking empty artifact.
    try:
        os.remove(filename)
    except OSError:
        pass
    logger.error("download_to_file: all %d attempts failed for %s: %s", max_attempts, url, last_error)
    return False


def _try_attempt(
    url: str,
    filename: str,
    timeout: int,
    chunk_size: int,
    headers: dict,
    exit_codes: tuple,
) -> Tuple[bool, Optional[Exception]]:
    """``(result, None)`` from :func:`_download_attempt`, or ``(False, error)`` when it raised."""
    try:
        return _download_attempt(url, filename, timeout, chunk_size, headers, exit_codes), None
    except Exception as e:
        logger.debug("_try_attempt: download attempt failed, error returned to caller: %s: %s", type(e).__name__, e)
        return False, e


def _download_attempt(
    url: str,
    filename: str,
    timeout: int,
    chunk_size: int,
    headers: dict,
    exit_codes: tuple,
) -> bool:
    """One GET-and-write attempt: True once ``filename`` is written, False on a fatal ``exit_codes`` status; raises otherwise."""
    request: Optional[requests.Response] = None
    try:
        request = _facade.requests.get(url, timeout=timeout, headers=headers, stream=True)
        _check_redirect_chain(request, url)
        if request.status_code in exit_codes:
            # A caller-designated fatal status (e.g. 404): requests.get() does not raise on
            # 4xx/5xx, so without this the error page's body would be written as the file.
            logger.error("download_to_file: fatal status %s for %s", request.status_code, url)
            return False
        if not (200 <= request.status_code < 300):
            # Every OTHER non-2xx used to fall straight into the write loop, so a 500/502
            # error PAGE became the file and the function reported exactly what a successful
            # download reports -- a batch behind a flaky gateway ended up with a directory of
            # small HTML files that rewrite_existing=False then skipped forever. Treat it as
            # an attempt failure so the retry/cleanup path below applies.
            raise requests.HTTPError(f"unexpected status {request.status_code} for {url}")
        # The GET must be re-issued on every attempt: a requests body is a single-use stream,
        # so retrying only the write loop iterates an already-consumed response, yields
        # nothing, and leaves the (already truncated by open(..., "wb")) file at 0 bytes while
        # the function returns the same value as a successful download.
        with open(filename, "wb") as fh:
            for chunk in request.iter_content(chunk_size * 1024):
                fh.write(chunk)
        return True
    finally:
        if request is not None:
            request.close()
