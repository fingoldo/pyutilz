"""External-IP discovery and IP-to-country geolocation over the free keyless providers.

The provider lists themselves (``IP_PROVIDERS``/``COUNTRY_PROVIDERS``) live in the package
``__init__`` with the rest of the mutable module-level state and are read back through
``_facade`` so a caller (or a test) that reassigns them on ``pyutilz.web.web`` is honoured here.
"""

import http
import json
import ssl
from dataclasses import dataclass
from random import shuffle
from typing import Any, Optional, Sequence

from ..url_guard import _CheckedRedirectHandler, urlopen_checked
from ._common import _ensure_http_scheme, logger

# PROJECT IDIOM for a re-export package's submodules: `import <parent> as _facade` is ALLOWED and
# load-bearing, while `from <parent> import <name>` at module top level is FORBIDDEN. The package
# __init__ imports this submodule, so importing the parent back is a genuine cycle; plain `import x`
# binds the PARTIALLY-INITIALISED sys.modules entry and defers every attribute lookup to call time,
# which both survives the cycle and keeps the name patchable (a test setting
# ``pyutilz.web.web.timeout`` / ``.IP_PROVIDERS`` / ``.get_ipinfo`` is seen here, where a from-import
# would have snapshotted the original value).
import pyutilz.web.web as _facade


def _direct_urlopen(url: str, timeout: Optional[float] = None) -> Any:
    """Non-proxied fetch for this module, with the http(s) allow-list applied to the URL AND to
    every redirect hop.

    A bare ``urllib.request.urlopen`` was used here before: its stock redirect handler follows a
    hop to ``http``, ``https`` OR ``ftp``, so the caller-/config-supplied provider URL was checked
    only at hop 0 and a 302 to ``ftp://internal-host/...`` was followed and its body parsed.
    """
    return urlopen_checked(url, timeout=timeout if timeout is not None else 30)


def _proxy_opener(proxy_user: Optional[str], proxy_pass: Optional[str], proxy_server: Optional[str], proxy_port: Optional[int]) -> Optional[Any]:
    """Build a urllib opener routing through ``proxy_server:proxy_port``, or None when no server was given.

    Credentials go into the proxy URL itself rather than through a password manager because a forward
    proxy answers 407 on the CONNECT, before any realm exists for urllib to match a stored password to.
    """
    if not proxy_server:
        return None
    # `is not None`, not `or`: a caller that deliberately configured an empty proxy_type must not be
    # silently rewritten to http -- the same reasoning as the timeout guard below.
    proxy_type = _facade.proxy_type if _facade.proxy_type is not None else "http"
    proxies = _facade.make_proxies_dict(proxy_user, proxy_pass, proxy_server, proxy_port if proxy_port is not None else 80, proxy_type=proxy_type)
    # _CheckedRedirectHandler, not urllib's stock one: the stock handler follows a redirect to
    # http, https OR ftp, so _ensure_http_scheme()'s allow-list held for hop 0 only and a provider
    # (or a hijacked/expired provider domain) answering 302 -> ftp://internal-host/... was followed
    # and its body parsed as geolocation data. Same guard cached_client.py already routes through.
    return _facade.urllib.request.build_opener(_facade.urllib.request.ProxyHandler(proxies), _CheckedRedirectHandler())


def get_external_ip(
    proxy_user: Optional[str] = None, proxy_pass: Optional[str] = None, proxy_server: Optional[str] = None, proxy_port: Optional[int] = None
) -> Optional[str]:
    """Return this machine's external IP by querying ``IP_PROVIDERS`` (shuffled) until one responds with a plausible address, or None if all fail.

    When ``proxy_server`` is given the query goes THROUGH that proxy, so the answer is the proxy's exit
    address - the only reason a caller hands proxy credentials to an external-IP lookup at all. With no
    proxy_server the request goes out directly, as before.
    """
    # list(), not the module-level object itself: shuffle() permutes IN PLACE, so every consumer
    # of pyutilz.web.web.IP_PROVIDERS saw the order change on each call, and a concurrent
    # get_external_ip() in another thread could iterate the list mid-permutation.
    providers = list(_facade.IP_PROVIDERS)
    shuffle(providers)
    opener = _proxy_opener(proxy_user, proxy_pass, proxy_server, proxy_port)
    # urlopen_checked (not bare urllib.request.urlopen) on the direct path: it re-applies the
    # scheme allow-list to every redirect hop, matching the proxied opener built above.
    urlopen = opener.open if opener is not None else _direct_urlopen

    for source in providers:
        try:
            # timeout= is required: urlopen's default is socket._GLOBAL_DEFAULT_TIMEOUT, which
            # blocks forever unless something else in the process called
            # socket.setdefaulttimeout() (nothing in this package does) -- a single stalled
            # provider would otherwise hang this whole function indefinitely instead of moving
            # on to the next shuffled IP_PROVIDERS entry.
            resp = urlopen(  # nosec B310 - scheme validated above -- `is not None`, not `or`: a caller-set timeout=0 must not be silently rewritten to 10
                _ensure_http_scheme(source), timeout=_facade.timeout if _facade.timeout is not None else 10
            )
        except ssl.SSLCertVerificationError:  # noqa: PERF203 -- per-iteration fault isolation is intentional (skip this provider, try the next)
            pass
        except Exception as e:
            logger.exception(e)
        else:
            if resp.status == http.HTTPStatus.OK:
                res = resp.read().decode("utf8").strip()
                if "." in res or ":" in res:
                    return res  # type: ignore[no-any-return]  # resp comes from an untyped pooled-HTTP helper, so .read() is Any
                else:
                    logger.warning("Weird IP address received from provider %s: %s", source, res)
    return None


@dataclass
class IpGeolocation:
    """What a geolocation lookup actually established, and from where.

    `provider` is not decoration: these services disagree, and a caller acting on a country (pricing,
    routing, a clinical prior) needs to be able to say which one said so. Country-level only - the finer
    fields these APIs return (city, coordinates, ISP) are deliberately not surfaced, because they are the
    fields with the worst accuracy and the highest privacy cost, and no caller in this codebase needs them.
    """

    country_code: str  # ISO 3166-1 alpha-2, uppercased
    country_name: str  # the provider's own English name; spellings differ BETWEEN providers
    continent: str = ""  # a code ("EU") from some providers and a name ("Europe") from others - see get_country_by_ip
    ip: str = ""
    provider: str = ""


def get_country_by_ip(ip: Optional[str] = None, providers: Optional[Sequence[dict]] = None) -> Optional[IpGeolocation]:
    """Resolve an IP address to a country, trying ``COUNTRY_PROVIDERS`` in order until one answers.

    ``ip=None`` looks up the CALLER's own address (every provider here does that for an empty path), so no
    separate `get_external_ip()` round trip is needed first.

    Returns ``None`` when every provider failed or none of them stated a country - never a guess and never
    a partial record. A geolocation miss is ordinary: private/loopback addresses, VPNs, corporate egress
    and rate limits all produce one, so callers must treat ``None`` as "unknown", not as an error.

    ACCURACY, stated plainly because callers keep assuming otherwise: IP geolocation is reliable at country
    level and unreliable below it, and it reports where the ADDRESS is, not where the person is - a VPN,
    a mobile carrier's national egress or a cloud host will all answer confidently and wrongly. Use it to
    PRE-FILL a field a human can correct, never as an established fact.

    ``continent`` is passed through unnormalised on purpose: ip-api and ipwho.is state a name ("Europe"),
    ipapi.co states a code ("EU"). Normalising here would mean this function owning a continent table;
    a caller that needs one shape should map the value it got, knowing which provider answered.
    """
    for provider in providers if providers is not None else _facade.COUNTRY_PROVIDERS:
        url = provider["url"].format(ip=ip or "")
        try:
            data = _facade.get_ipinfo(use_urllib=True, url=url)
        except Exception as e:
            # debug, not exception: a provider failing is the EXPECTED case this chain exists for (all four
            # rate-limit routinely), and logging a traceback per provider turns one successful lookup into
            # three stack traces in the caller's log.
            logger.debug("country lookup via %s failed: %s", url, e)
            continue
        if not isinstance(data, dict):
            # get_ipinfo swallows its own exceptions and returns None, so this -- not the
            # except branch above -- is where a failing provider actually shows up.
            logger.debug("country lookup via %s returned no usable payload", url)
            continue
        code = data.get(provider["code"])
        name = data.get(provider["name"])
        # Both, not either: a provider that answered with an error body ({"success": false, ...}) supplies
        # neither, and one that supplies only a code leaves the caller with nothing to show a human.
        if not isinstance(code, str) or not isinstance(name, str) or not code.strip() or not name.strip():
            continue
        continent = data.get(provider["continent"])
        return IpGeolocation(
            country_code=code.strip().upper(),
            country_name=name.strip(),
            continent=continent.strip() if isinstance(continent, str) else "",
            ip=str(data.get("ip") or data.get("query") or ip or ""),
            provider=url.split("/")[2],
        )
    return None


def get_ipinfo(use_urllib: bool = False, url: str = "https://api.ipify.org?format=json") -> Optional[Any]:
    """Fetch JSON IP info from ``url``, either directly via ``urllib`` (``use_urllib=True``) or via the module's ``get_url`` (session/proxy-aware) fetcher."""
    json_loads: Any
    try:
        import orjson

        json_loads = orjson.loads
    except ImportError:
        json_loads = json.loads

    if use_urllib:
        try:
            resp = _direct_urlopen(  # nosec B310 - scheme validated here AND on every redirect hop by _direct_urlopen; timeout= avoids blocking forever, see get_external_ip's identical fix -- `is not None`, not `or`: a caller-set timeout=0 must not be silently rewritten to 10
                _ensure_http_scheme(url), timeout=_facade.timeout if _facade.timeout is not None else 10
            )
        except Exception as e:
            logger.exception(e)
            return None
        else:
            if resp.status == http.HTTPStatus.OK:
                return json_loads(resp.read().decode("utf8").strip())
            # Regression fix (2026-07-21 audit round 2, LOW): a non-200 response used to return
            # `{}` here while an exception (above) returned None -- two different failure values
            # for the same "the request failed" outcome, unlike get_external_ip's single None
            # convention and this function's own use_urllib=False branch below.
            return None
    else:
        res = _facade.get_url(url, target="ipinfo", inject_headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            res = res.json() if res is not None else None
        except Exception as e:
            logger.exception(e)
            res = None
        return res
