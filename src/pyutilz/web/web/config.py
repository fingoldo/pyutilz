"""Configuration entry points that install the package-level scraping state: session reset
(``init_vars``), proxy credentials/ports (``connect``/``set_proxy``) and throttling parameters
(``set_params``).

Every global these write lives on the package object, so they are assigned through ``_facade``
rather than with a ``global`` statement -- that is what makes ``pyutilz.web.web.sess`` and friends
observable (and patchable) from the outside exactly as they were before the split.
"""

from typing import Any, Optional

from ._common import logger

# See ipinfo.py for why the parent is reached through `import <parent> as _facade`.
import pyutilz.web.web as _facade


def init_vars():
    """Reset the module-level session state (session, IP-query counter, headers, proxies, timeout) to defaults."""
    # Regression fix: the old Session (a requests.Session owns a urllib3 connection pool --
    # potentially several open keep-alive sockets) was previously just dropped here with no
    # .close(), leaking its connection pool every time init_vars() runs (e.g. every
    # web.connect() call, including FileMaker's two-calls-per-authentication pattern).
    #
    # Regression fix (meta-test-driven finding, proactive lock-discipline audit): these globals
    # used to be reset here with no lock, unlike get_url()'s snapshot read of the same sess/
    # proxies/headers globals (which does take _state_lock) -- a concurrent get_url() call could
    # observe a torn mix of old and new state. The lock only needs to cover the state mutation,
    # not old_sess.close() (an I/O call with no reason to hold the lock during it).
    with _facade._state_lock:
        old_sess = _facade.sess
        _facade.sess = None
        _facade.num_ip_queries = 0
        _facade.template_headers = None
        _facade.headers = {}
        _facade.proxies = None
        _facade.timeout = 10
    if old_sess is not None:
        try:
            old_sess.close()
        except Exception as e:
            logger.exception(e)
    logger.debug("Session cleared")


def connect(
    m_proxy_user: Optional[str] = None,
    m_proxy_pass: Optional[str] = None,
    m_proxy_server: Optional[str] = None,
    m_proxy_min_port: Optional[str] = None,
    m_proxy_max_port: Optional[str] = None,
    m_template_headers: Optional[Any] = None,
    m_timeout: Optional[int] = 10,
    m_proxy_port: Optional[str] = None,
    m_proxy_type: str = "http",
    **kwargs,
) -> None:
    """Reset session state and set the module-level proxy/header/timeout config from the ``m_*`` arguments (used to configure proxy credentials before fetching)."""
    _facade.init_vars()
    # Regression fix (meta-test-driven finding, proactive lock-discipline audit): this used to
    # write proxy_user/proxy_pass/proxy_server/etc. with no lock, unlike set_proxy()'s write of
    # the identical field group (which does take _state_lock) and get_url()'s locked read of it.
    with _facade._state_lock:
        (
            _facade.proxy_user,
            _facade.proxy_pass,
            _facade.proxy_server,
            _facade.proxy_min_port,
            _facade.proxy_max_port,
            _facade.template_headers,
            _facade.timeout,
            _facade.proxy_port,
            _facade.proxy_type,
        ) = (
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
    m_proxy_min_port: Optional[int] = None,
    m_proxy_max_port: Optional[int] = None,
    m_proxy_port: Optional[int] = None,
    m_proxy_type: str = "http",
) -> None:
    """Set module-level proxy credentials/port range and immediately obtain a fresh proxy from ``get_new_smartproxy``, storing it in the module-level ``proxies`` dict.

    Raises ValueError if ``m_proxy_user``, ``m_proxy_pass`` or ``m_proxy_server`` is None.
    """
    with _facade._state_lock:
        (
            _facade.proxy_user,
            _facade.proxy_pass,
            _facade.proxy_server,
            _facade.proxy_min_port,
            _facade.proxy_max_port,
            _facade.proxy_port,
            _facade.proxy_type,
        ) = (
            m_proxy_user,
            m_proxy_pass,
            m_proxy_server,
            m_proxy_min_port,
            m_proxy_max_port,
            m_proxy_port,
            m_proxy_type,
        )
        if _facade.proxy_user is None or _facade.proxy_pass is None or _facade.proxy_server is None:
            raise ValueError("set_proxy: proxy_user, proxy_pass and proxy_server are required (got None)")
        local_proxy_user, local_proxy_pass, local_proxy_server = _facade.proxy_user, _facade.proxy_pass, _facade.proxy_server
        local_proxy_min_port, local_proxy_max_port, local_proxy_port, local_proxy_type = (
            _facade.proxy_min_port,
            _facade.proxy_max_port,
            _facade.proxy_port,
            _facade.proxy_type,
        )

    new_proxies = _facade.get_new_smartproxy(
        local_proxy_user,
        local_proxy_pass,
        local_proxy_server,
        int(local_proxy_min_port) if local_proxy_min_port is not None else 20001,
        int(local_proxy_max_port) if local_proxy_max_port is not None else 37960,
        last_used_dict=_facade.last_used_dict,
        min_idle_interval_minutes=_facade.min_idle_interval_minutes if _facade.min_idle_interval_minutes is not None else 0,
        failed_dict=_facade.failed_dict,
        min_failed_idle_interval_minutes=_facade.min_failed_idle_interval_minutes if _facade.min_failed_idle_interval_minutes is not None else 60 * 24,
        proxy_port=local_proxy_port,
        proxy_type=local_proxy_type if local_proxy_type is not None else "http",
    )
    with _facade._state_lock:
        _facade.proxies = new_proxies


def set_params(
    m_delay: Optional[int] = 0,
    m_max_ip_queries: Optional[int] = 0,
    m_last_used_dict: Optional[dict] = None,
    m_min_idle_interval_minutes: Optional[int] = None,
    m_failed_dict: Optional[dict] = None,
    m_min_failed_idle_interval_minutes: Optional[int] = None,
) -> None:
    """Set module-level rate-limiting/throttling parameters: inter-request delay, max IP queries per session, and the proxy-touched/proxy-failed tracking dicts and idle intervals."""
    _facade.delay = m_delay
    _facade.max_ip_queries = m_max_ip_queries
    _facade.last_used_dict = m_last_used_dict
    _facade.min_idle_interval_minutes = m_min_idle_interval_minutes
    _facade.failed_dict = m_failed_dict
    _facade.min_failed_idle_interval_minutes = m_min_failed_idle_interval_minutes
