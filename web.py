# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

ensure_installed("joblib grequests fake_useragent")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import requests
import urllib.request
from random import random, shuffle
from datetime import datetime
from joblib import hash
from time import sleep
import warnings
import http
import ssl

delay = 1
max_ip_queries = 0
last_used_dict = None
min_idle_interval_minutes = None
failed_dict = None
min_failed_idle_interval_minutes = None

IP_PROVIDERS = ["https://api.ipify.org/", "https://ident.me/", "http://icanhazip.com/"]
cur_max_ip_queries = -1
proxy_server = None
was_blocked = False


def init_vars():
    global sess, num_ip_queries, template_headers, headers, proxies, timeout
    sess = None
    num_ip_queries = 0
    template_headers = None
    headers = {}
    proxies = None
    timeout = 10
    logger.debug("Session cleared")


init_vars()


def get_external_ip(
    proxy_user: Optional[str] = None, proxy_pass: Optional[str] = None, proxy_server: Optional[str] = None, proxy_port: Optional[int] = None
) -> str:
    global IP_PROVIDERS
    shuffle(IP_PROVIDERS)

    for source in IP_PROVIDERS:
        try:
            resp = urllib.request.urlopen(source)
        except ssl.SSLCertVerificationError:
            pass
        except Exception as e:
            logger.exception(e)
        else:
            if resp.status == http.HTTPStatus.OK:
                res = resp.read().decode("utf8").strip()
                if "." in res or ":" in res:
                    return res
                else:
                    logger.warning(f"Weird IP address received from provider {source}: {res}")


def get_ipinfo(use_urllib: bool = False, url="https://api.ipify.org?format=json"):
    import json

    if use_urllib:
        try:
            resp = urllib.request.urlopen(url)
        except Exception as e:
            logger.exception(e)
        else:
            if resp.status == http.HTTPStatus.OK:
                return json.loads(resp.read().decode("utf8").strip())
            else:
                return {}
    else:
        res = get_url(url, target="ipinfo", inject_headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            res = res.json()
        except Exception as e:
            logger.exception(e)
            res = None
        return res


def download_in_parallel(
    urls_to_process: Sequence, func: Callable, headers: Optional[dict] = None, nparallel_downloads: int = 3, report_each: int = 10
) -> Optional[List]:
    import grequests

    if urls_to_process is None:
        return
    if len(urls_to_process) == 0:
        return

    n_processed = 0
    errored_urls = []
    rs = (grequests.get(sub_url, verify=True, allow_redirects=True, headers=headers) for sub_url in urls_to_process)
    logger.info("Started crawling! nlinks=%d" % (len(urls_to_process)))
    # pbar=tqdmu(urls_to_process)
    for resp, sub_url in zip(grequests.map(rs, size=nparallel_downloads), urls_to_process):
        n_processed = n_processed + 1
        if len(resp.history) > 0:
            final_status_code = resp.history[-1].status_code
        else:
            final_status_code = resp.status_code
        if not (resp is None):
            if resp.status_code == http.HTTPStatus.OK:
                try:
                    func(resp, sub_url)
                except Exception as e:
                    errored_urls.append(sub_url)
                    logger.error("Error processing url %s: %s" % (sub_url, e))
            else:
                errored_urls.append(sub_url)
                logger.error("Error fetching url %s: status_code=%s" % (sub_url, final_status_code))
        else:
            errored_urls.append(sub_url)
            logger.error("Response is None for url %s" % sub_url)
        if (n_processed % report_each) == 0:
            # pbar.update(report_each)
            logger.info("Processed %d urls,n_errored=%d" % (n_processed, len(errored_urls)))
    logger.info("Finished! n_processed=%d,n_errored=%d" % (n_processed, len(errored_urls)))
    # pbar.close()
    return errored_urls


# ***************************************************************************************************************************
# Proxy downloads
# ***************************************************************************************************************************


def connect(
    m_proxy_user: Optional[str] = None,
    m_proxy_pass: Optional[str] = None,
    m_proxy_server: Optional[str] = None,
    m_proxy_min_port: Optional[str] = None,
    m_proxy_max_port: Optional[str] = None,
    m_template_headers: Optional[str] = None,
    m_timeout: Optional[int] = 10,
    m_proxy_port: Optional[str] = None,
    m_proxy_type: str = "http",
    **kwargs,
) -> None:
    global proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, template_headers, timeout, proxy_port, proxy_type

    init_vars()
    proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, template_headers, timeout, proxy_port, proxy_type = (
        m_proxy_user,
        m_proxy_pass,
        m_proxy_server,
        m_proxy_min_port,
        m_proxy_max_port,
        m_template_headers,
        m_timeout,
        m_proxy_port,
        m_proxy_type,
    )


def set_proxy(
    m_proxy_user: Optional[str] = None,
    m_proxy_pass: Optional[str] = None,
    m_proxy_server: Optional[str] = None,
    m_proxy_min_port: Optional[str] = None,
    m_proxy_max_port: Optional[str] = None,
    m_proxy_port: Optional[str] = None,
    m_proxy_type: str = "http",
) -> None:
    global proxies
    global proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, proxy_port, proxy_type
    proxy_user, proxy_pass, proxy_server, proxy_min_port, proxy_max_port, proxy_port, proxy_type = (
        m_proxy_user,
        m_proxy_pass,
        m_proxy_server,
        m_proxy_min_port,
        m_proxy_max_port,
        m_proxy_port,
        m_proxy_type,
    )
    proxies = get_new_smartproxy(
        proxy_user,
        proxy_pass,
        proxy_server,
        proxy_min_port,
        proxy_max_port,
        last_used_dict=last_used_dict,
        min_idle_interval_minutes=min_idle_interval_minutes,
        failed_dict=failed_dict,
        min_failed_idle_interval_minutes=min_failed_idle_interval_minutes,
        proxy_port=proxy_port,
        proxy_type=proxy_type,
    )


def set_params(
    m_delay: Optional[int] = 0,
    m_max_ip_queries: Optional[int] = 0,
    m_last_used_dict: Optional[dict] = None,
    m_min_idle_interval_minutes: Optional[int] = None,
    m_failed_dict: Optional[dict] = None,
    m_min_failed_idle_interval_minutes: Optional[int] = None,
) -> None:
    global delay, max_ip_queries, last_used_dict, min_idle_interval_minutes, failed_dict, min_failed_idle_interval_minutes

    delay = m_delay
    max_ip_queries = m_max_ip_queries
    last_used_dict = m_last_used_dict
    min_idle_interval_minutes = m_min_idle_interval_minutes
    failed_dict = m_failed_dict
    min_failed_idle_interval_minutes = m_min_failed_idle_interval_minutes


def set_proxy_last_use_time(last_used_dict: dict, proxies: dict) -> None:
    if type(last_used_dict) == dict:
        last_used_dict[hash(proxies)] = datetime.utcnow()


def make_proxies_dict(proxy_user: str, proxy_pass: str, proxy_server: str, proxy_port: int, proxy_type: str = "https") -> dict:
    if proxy_user and proxy_pass:
        proxy_url = "%s:%s@%s:%s" % (proxy_user, proxy_pass, proxy_server, proxy_port)
    else:
        proxy_url = "%s:%s" % (proxy_server, proxy_port)
    # return {"http": f"http://{proxy_url}", "https": f"https://{proxy_url}"}
    return {"http": f"{proxy_type}://{proxy_url}", "https": f"{proxy_type}://{proxy_url}"}


def get_new_smartproxy(
    proxy_user: str,
    proxy_pass: str,
    proxy_server: str,
    proxy_min_port: int = 20001,
    proxy_max_port: int = 37960,
    job_desc: str = "",
    last_used_dict: dict = {},
    min_idle_interval_minutes: float = 0,
    failed_dict: dict = {},
    min_failed_idle_interval_minutes: float = 60 * 24,
    warn_after_n_failures: int = 5,
    delay: int = 5,
    proxy_port: Optional[int] = None,
    proxy_type: str = "http",
    verbose=False,
) -> dict:
    n = 0
    now_time = datetime.utcnow()
    while True:
        # ----------------------------------------------------------------------------------------------------------------------------
        # Get random port
        # ----------------------------------------------------------------------------------------------------------------------------
        if proxy_port is None:
            proxy_port = int(proxy_min_port) + int(random() * (int(proxy_max_port) - int(proxy_min_port)))

        proxies = make_proxies_dict(proxy_user, proxy_pass, proxy_server, proxy_port, proxy_type)

        proxy_key = hash(proxies)
        # ----------------------------------------------------------------------------------------------------------------------------
        # Check if it's allowed for immediate use by the policies
        # ----------------------------------------------------------------------------------------------------------------------------
        b_time_to_check_now = True
        for dict_to_check in (failed_dict, last_used_dict):
            if dict_to_check is not None:
                if proxy_key in dict_to_check:
                    if (now_time - dict_to_check[proxy_key]).total_seconds() / 60 < min_idle_interval_minutes:
                        if verbose:
                            logger.info("Skipping proxy %s:%s, touched recently" % (proxy_server, proxy_port))
                        b_time_to_check_now = False
                        break

        if b_time_to_check_now:
            if verbose:
                logger.info("Got new proxy: %s:%s" % (proxy_server, proxy_port))
            return proxies
        else:
            n = n + 1
            if n > warn_after_n_failures:
                if verbose:
                    logger.info("Could not get an untouched proxy%s, sleeping %s sec." % ("" if job_desc == "" else " for " + job_desc, delay))
                sleep(delay)
                n = 0


def report_params(url, proxies, params, data, json, headers_to_use, timeout):
    logger.info("url=%s, proxies=%s, params=%s, data=%s, json=%s, headers=%s, timeout=%s" % (url, str(proxies), params, data, json, headers_to_use, timeout))


def get_url(
    url: str,
    target: str = "",
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    json: Optional[dict] = None,
    max_retries: int = 10,
    exit_statuses: Sequence = (),
    blocking_statuses: Sequence = (),
    retry_statuses: Sequence = (),
    quit_on_blocking: bool = False,
    blocking_errors: Sequence = (),
    verb: str = "get",
    b_random_ua=True,
    b_use_proxy=True,
    b_use_session=True,
    verbose=False,
    custom_headers: dict = None,
    inject_headers: dict = None,
    sort_headers: bool = True,
    lowercase_headers: bool = True,
    ratelimited_sleep_interval: int = 30,
    ratelimited_proxy_sleep_interval: int = 0,
    ratelimiting_statuses: Sequence = (429,),
    session_expired_statuses: Sequence = (),
) -> object:
    global proxies
    global was_blocked
    global num_ip_queries, cur_max_ip_queries

    n_retries = 0
    res = None

    while n_retries < max_retries:
        try:
            n_retries = n_retries + 1
            # print("Getting url %s,headers=%s,params=%s,proxies=%s,timeout=%s,cookies=%s" % (url,headers,params,proxies,timeout,sess.cookies.get_dict()))

            # We are trying to fetch some url. Do we need to create new proxy session?
            if sess is None or (max_ip_queries > 0 and num_ip_queries > cur_max_ip_queries):
                # If there is no session yet or we have downloaded too many items withing current session alrady
                get_new_session(b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
                if max_ip_queries > 0:
                    cur_max_ip_queries = int(max_ip_queries * (0.6 + 0.4 * random()))
                    logger.info("cur_max_ip_queries set to %d" % cur_max_ip_queries)
            headers_to_use = custom_headers if custom_headers else headers
            if inject_headers:
                if headers_to_use:
                    headers_to_use = headers_to_use.copy()
                else:
                    headers_to_use = {}
                for header, value in inject_headers.items():
                    headers_to_use[header] = value

            if headers_to_use:
                if sort_headers:
                    headers_to_use = dict(sorted(headers_to_use.items()))  # need this to avoid bot detection
                if lowercase_headers:
                    headers_to_use = {key.lower(): value for key, value in headers_to_use.items()}

            if verbose:
                report_params(url, proxies, params, data, json, headers_to_use, timeout)

            if b_use_session:
                obj = sess
            else:
                obj = requests

            method = getattr(obj, verb)

            res = method(url, headers=headers_to_use, params=params, data=data, json=json, proxies=proxies, timeout=timeout)

            num_ip_queries = num_ip_queries + 1

        except Exception as e:
            se = str(e)
            if verbose:
                logger.exception(e)
            se = se.lower()
            if "proxy" in se or "timed out" in se or "bad handshake" in se or "connection broken" in se or "sslerror" in se:
                if b_use_proxy:
                    if proxy_server:
                        if verbose:
                            logger.warning("Seems to be a bad proxy. Receiving new proxy for %s" % target)
                        proxies = get_new_smartproxy(
                            proxy_user,
                            proxy_pass,
                            proxy_server,
                            proxy_min_port,
                            proxy_max_port,
                            last_used_dict=last_used_dict,
                            min_idle_interval_minutes=min_idle_interval_minutes,
                            failed_dict=failed_dict,
                            min_failed_idle_interval_minutes=min_failed_idle_interval_minutes,
                            proxy_port=proxy_port,
                            proxy_type=proxy_type,
                        )
        else:
            if res.status_code not in (http.HTTPStatus.OK, http.HTTPStatus.PARTIAL_CONTENT):
                if res.status_code in blocking_statuses:
                    logger.info("Error %s while getting %s" % (res.status_code, url))
                    report_params(url, proxies, params, data, json, headers_to_use, timeout)
                    handle_blocking(target, b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
                    was_blocked = True
                    if quit_on_blocking:
                        break
                elif res.status_code in session_expired_statuses:
                    logger.warning("Session expired while getting url=%s, code=%s, response=%s", url, res.status_code, res.text)
                    break
                elif res.status_code in exit_statuses:
                    if verbose:
                        logger.info("status_code %s" % res.status_code)
                    break
                elif res.status_code in ratelimiting_statuses:
                    if verbose:
                        logger.warning("Ratelimited [%s] while getting url %s: %s" % (res.status_code, url, res.text))
                    if proxy_server:
                        if verbose:
                            logger.warning("Seems to be a bad proxy. Receiving new proxy for %s" % target)
                        if ratelimited_proxy_sleep_interval:
                            sleep(ratelimited_proxy_sleep_interval * random())
                        proxies = get_new_smartproxy(
                            proxy_user,
                            proxy_pass,
                            proxy_server,
                            proxy_min_port,
                            proxy_max_port,
                            last_used_dict=last_used_dict,
                            min_idle_interval_minutes=min_idle_interval_minutes,
                            failed_dict=failed_dict,
                            min_failed_idle_interval_minutes=min_failed_idle_interval_minutes,
                            proxy_port=proxy_port,
                            proxy_type=proxy_type,
                        )
                    else:
                        sleep(ratelimited_sleep_interval)
                else:
                    logger.warning("Error %s while getting url %s: %s" % (res.status_code, url, res.text))

                    # if blocking or exit satuses are specified, we keep retrying on any error (after small pause)

                    if len(blocking_statuses) == 0 and len(exit_statuses) == 0:
                        # unless retry on this status is permitted explicitly
                        if res.status_code not in retry_statuses:
                            break
            else:
                err_found = False
                for t in blocking_errors:
                    if t in res.text:
                        err_found = True
                        break
                if err_found:
                    handle_blocking(target, b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)
                    was_blocked = True
                    if quit_on_blocking:
                        break
                else:
                    set_proxy_last_use_time(last_used_dict, proxies)
                    was_blocked = False
                    break
    if delay:
        sleep(delay * random())

    if res is None:
        logger.warning(f"Could not get url {url}")
    return res


def get_new_session(b_random_ua: bool = True, b_use_proxy: bool = True) -> None:
    global sess, proxies, headers
    global num_ip_queries

    sess = requests.Session()
    num_ip_queries = 0

    logger.debug(f"Created new web session")

    headers = template_headers
    if b_random_ua:
        if headers is None:
            headers = dict()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # from fake_useragent import UserAgent
            # ua = UserAgent(verify_ssl=False)
            # headers['user-agent']=ua.random

            headers["user-agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2762.73 Safari/537.36"

    if b_use_proxy:
        if proxy_server:
            proxies = get_new_smartproxy(
                proxy_user,
                proxy_pass,
                proxy_server,
                proxy_min_port,
                proxy_max_port,
                last_used_dict=last_used_dict,
                min_idle_interval_minutes=min_idle_interval_minutes,
                failed_dict=failed_dict,
                min_failed_idle_interval_minutes=min_failed_idle_interval_minutes,
                proxy_port=proxy_port,
                proxy_type=proxy_type,
            )
            logger.info(f"proxy_server={proxy_server}")


def handle_blocking(target: str, b_random_ua: bool = True, b_use_proxy: bool = True) -> None:
    if proxies is not None:
        logger.warning("IP %s blocked. Receiving new proxy/session for %s" % (proxies["https"].split("@")[1], target))
    else:
        logger.warning("IP blocked.")

    sleep(delay * random())
    set_proxy_last_use_time(failed_dict, proxies)
    get_new_session(b_random_ua=b_random_ua, b_use_proxy=b_use_proxy)


def is_rotating_proxy(proxy_server: dict) -> bool:
    # {"PROXY_HOST": "gate.dc.smartproxy.com","PROXY_MIN_PORT": 20001,"PROXY_MAX_PORT": 37960}
    if proxy_server.get("PROXY_HOST", "").lower() == "gate.dc.smartproxy.com":
        if proxy_server.get("PROXY_MIN_PORT") == 20000:
            if proxy_server.get("PROXY_MAX_PORT") == 20000:
                return True


def download_to_file(
    url: str,
    filename: str,
    rewrite_existing: bool = True,
    timeout: int = 100,
    chunk_size: int = 1024,
    max_attempts: int = 5,
    headers: dict = {},
    exit_codes: tuple = (),
):
    """Dropin replacement for urllib.request.urlretrieve(url, filename) taht can hand for indefinitely long."""
    # Make the actual request, set the timeout for no data to 10 seconds and enable streaming responses so we don't have to keep the large files in memory

    nattempts = 0
    while nattempts < max_attempts:
        try:
            request = requests.get(url, timeout=timeout, headers=headers, stream=True)
        except Exception as e:
            if request is not None and request.status_code in exit_codes:
                return
            logger.exception(e)
            sleep(10 * random())
            logger.info("Making another attempt")
            nattempts += 1
        else:
            break

    nattempts = 0
    while nattempts < max_attempts:
        try:
            # Open the output file and make sure we write in binary mode
            with open(filename, "wb") as fh:
                # Walk through the request response in chunks of chunk_size * 1024 bytes
                for chunk in request.iter_content(chunk_size * 1024):
                    # Write the chunk to the file
                    fh.write(chunk)
                    # Optionally we can check here if the download is taking too long
        except Exception as e:
            logger.exception(e)
            sleep(10 * random())
            logger.info("Making another attempt")
            nattempts += 1
        else:
            break
