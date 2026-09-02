"""Web scraping helpers: proxy rotation, session/header management, retrying URL fetches and parallel/streaming downloads.

Historically one flat 1030-LOC module; carved into cohesive submodules once it passed the
1000-LOC limit, with the entire original surface (public AND private names alike) re-exported
here so ``from pyutilz.web.web import X`` / ``from pyutilz.web import X`` keep resolving unchanged:

  _common     - logger, URL-scheme guard, credential/header redaction, report_params, Retry-After parsing
  ipinfo      - external-IP discovery and IP-to-country geolocation
  proxy_pool  - proxies-dict construction, last-use/failure bookkeeping, eligible-port selection
  config      - init_vars / connect / set_proxy / set_params state installers
  fetching    - get_url retry loop, session rotation, block handling
  downloads   - grequests-backed parallel fetch and retrying streaming file download

ALL mutable scraping state (``sess``, ``proxies``, ``headers``, the ``proxy_*`` fields, the
throttling parameters, the provider lists) plus the patchable third-party aliases (``requests``,
``urllib``, ``sleep``, ``random``) live HERE rather than in a submodule, and every submodule
reads and writes them through ``import pyutilz.web.web as _facade``. That is deliberate: with the
state on the package object, ``pyutilz.web.web.<name>`` remains the single authoritative binding
for callers and for ``mock.patch``/``monkeypatch.setattr``, exactly as it was before the split --
a submodule-local copy would silently diverge from it.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import json
import logging
import os

from ._common import logger

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Callable, List, Optional, Sequence

import requests
import urllib.request
import urllib.parse
from random import random, shuffle
from datetime import datetime
from joblib import hash as joblib_hash
from time import sleep
import warnings
import http
import ssl
import threading
from dataclasses import dataclass

# Guards read-then-mutate access to the module-level scraping state below (sess, proxies,
# headers, num_ip_queries, cur_max_ip_queries, was_blocked, proxy_* fields) when this module
# is driven from multiple threads. Only the shared-state read/mutate sections are locked --
# actual network I/O (the HTTP request itself) happens outside the lock.
_state_lock = threading.Lock()

delay: Optional[int] = 1
max_ip_queries: Optional[int] = 0
last_used_dict: Optional[dict] = None
min_idle_interval_minutes: Optional[int] = None
failed_dict: Optional[dict] = None
min_failed_idle_interval_minutes: Optional[int] = None

IP_PROVIDERS = ["https://api.ipify.org/", "https://ident.me/", "http://icanhazip.com/"]
cur_max_ip_queries: int = -1
proxy_server: Optional[str] = None
was_blocked = False

# Bound for real by init_vars() / set_proxy_credentials(); declared here so module-level
# references resolve before the first call (and so static analysis can see these names exist).
sess: Optional[Any] = None
num_ip_queries: int = 0
# template_headers/proxy_min_port/proxy_max_port/proxy_port are genuinely inconsistently
# typed across call sites in this file today (some pass str, some pass int, with defensive
# int(...) casts at the point of use) -- Any reflects that honestly rather than picking a
# specific type that would be wrong for half the callers.
template_headers: Optional[Any] = None
headers: Optional[dict] = None
proxies: Optional[dict] = None
timeout: Optional[int] = 10
proxy_user: Optional[str] = None
proxy_pass: Optional[str] = None
proxy_min_port: Optional[Any] = None
proxy_max_port: Optional[Any] = None
proxy_port: Optional[Any] = None
proxy_type: Optional[str] = None

# Geolocation providers, tried in order until one answers with a country. All four are keyless and free
# for low volume; each states a DIFFERENT JSON shape, so the field names to read are carried alongside the
# URL rather than guessed. `{ip}` is substituted with the address to look up, or removed for a self-lookup
# (every one of these resolves the CALLER's own address when the path is empty).
#
# Deliberately NOT shuffled the way IP_PROVIDERS is: those three are interchangeable echo services, while
# these differ in accuracy and rate limit, so the order is a preference and shuffling would throw it away.
COUNTRY_PROVIDERS: List[dict] = [
    {"url": "https://ipapi.co/{ip}/json/", "code": "country_code", "name": "country_name", "continent": "continent_code"},
    # `fields=` is required, not decoration: ip-api's DEFAULT response omits `continent` entirely, so the
    # unqualified URL silently returned a blank continent for every lookup this provider answered.
    {
        "url": "http://ip-api.com/json/{ip}?fields=status,country,countryCode,continent,query",
        "code": "countryCode",
        "name": "country",
        "continent": "continent",
    },
    {"url": "https://ipwho.is/{ip}", "code": "country_code", "name": "country", "continent": "continent"},
    {"url": "https://freeipapi.com/api/json/{ip}", "code": "countryCode", "name": "countryName", "continent": "continent"},
]

# ----------------------------------------------------------------------------------------------------------------------------
# Re-exports from the cohesive submodules. Private names are re-exported too, on purpose: callers and tests
# have imported `_ensure_http_scheme`/`_parse_retry_after`/`_redact_*` from this module path for years.
# ----------------------------------------------------------------------------------------------------------------------------

from ._common import (
    _ALLOWED_URL_SCHEMES,
    _SENSITIVE_HEADER_NAMES,
    _ensure_http_scheme,
    _error_log_throttle,
    _parse_retry_after,
    _redact_headers,
    _redact_proxies_dict,
    _redact_proxy_url,
    report_params,
)
from .ipinfo import IpGeolocation, get_country_by_ip, get_external_ip, get_ipinfo
from .proxy_pool import get_new_smartproxy, is_rotating_proxy, make_proxies_dict, set_proxy_last_use_time
from .config import connect, init_vars, set_params, set_proxy
from .fetching import get_new_session, get_url, handle_blocking
from .downloads import download_in_parallel, download_to_file

# Preserved from the flat module: the session state must be in its documented default shape
# (headers={}, timeout=10, ...) the moment the module finishes importing, since callers may fetch
# without ever calling connect()/set_proxy() first.
init_vars()

# Mirrors what `from pyutilz.web.web import *` produced before the split -- i.e. every non-underscore
# module-level name, imported third-party aliases included, because `pyutilz/web/__init__.py` does
# `from .web import *` and downstream code has long reached e.g. `pyutilz.web.requests` through it.
# The submodule names themselves are deliberately NOT listed: they are new, and leaking them into
# `pyutilz.web` would add package attributes that never existed.
__all__ = [
    "json",
    "logging",
    "os",
    "logger",
    "Any",
    "Callable",
    "List",
    "Optional",
    "Sequence",
    "requests",
    "urllib",
    "random",
    "shuffle",
    "datetime",
    "joblib_hash",
    "sleep",
    "warnings",
    "http",
    "ssl",
    "threading",
    "dataclass",
    "delay",
    "max_ip_queries",
    "last_used_dict",
    "min_idle_interval_minutes",
    "failed_dict",
    "min_failed_idle_interval_minutes",
    "IP_PROVIDERS",
    "COUNTRY_PROVIDERS",
    "cur_max_ip_queries",
    "proxy_server",
    "was_blocked",
    "sess",
    "num_ip_queries",
    "template_headers",
    "headers",
    "proxies",
    "timeout",
    "proxy_user",
    "proxy_pass",
    "proxy_min_port",
    "proxy_max_port",
    "proxy_port",
    "proxy_type",
    "report_params",
    "IpGeolocation",
    "get_country_by_ip",
    "get_external_ip",
    "get_ipinfo",
    "get_new_smartproxy",
    "is_rotating_proxy",
    "make_proxies_dict",
    "set_proxy_last_use_time",
    "connect",
    "init_vars",
    "set_params",
    "set_proxy",
    "get_new_session",
    "get_url",
    "handle_blocking",
    "download_in_parallel",
    "download_to_file",
]
