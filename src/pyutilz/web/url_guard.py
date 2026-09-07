"""A single checked entry point for outbound HTTP requests built from untrusted data.

Any pipeline that builds request URLs from data it did not author (an API payload's embedded
``image_url``, a registry's ``Source.url``, a query-string built by interpolating an external id)
is one `file:///etc/passwd` or `ftp://` value away from turning "fetch a resource" into a local
file read -- ``urllib.request.urlopen`` honours whatever scheme it is handed. Route every such
call through :func:`urlopen_checked` (or check with :func:`require_http_url` first) instead of
adding a scheme check at each call site individually -- a per-site check is a rule the next call
site silently opts out of.

Ported from a downstream project's own single-purpose SSRF guard module.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any, Dict, FrozenSet, Optional, Union

# Deliberately narrow: not a denylist of known-bad schemes (which a new one walks around) but the
# complete list of schemes a typical HTTP-fetch pipeline has a reason to speak.
ALLOWED_SCHEMES: FrozenSet[str] = frozenset({"http", "https"})


# Re-exported (not defined here): the class now lives with its siblings in pyutilz.web.exceptions,
# while ``from pyutilz.web.url_guard import UnsafeURLError`` -- the historic path -- keeps working.
from .exceptions import UnsafeURLError


def require_http_url(url: str, allowed_schemes: FrozenSet[str] = ALLOWED_SCHEMES) -> str:
    """Return ``url`` unchanged when its scheme is in ``allowed_schemes`` (default http/https),
    else raise :class:`UnsafeURLError`.

    Parsed with ``urllib.parse.urlsplit`` rather than a string-prefix test, so ``HTTP://``,
    surrounding whitespace, and a scheme-relative ``//host/path`` (no scheme at all -- ``urlopen``
    would reject it later with a message naming urllib rather than the caller) all resolve to the
    same verdict here.
    """
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in allowed_schemes:
        raise UnsafeURLError(f"refusing to open non-HTTP(S) URL (scheme {scheme!r}): {url!r}")
    if not parsed.netloc:
        raise UnsafeURLError(f"refusing to open URL with no host: {url!r}")
    return url


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-applies the scheme allow-list to every redirect hop, not just the first URL.

    ``urllib``'s stock handler permits a redirect to ``http``, ``https`` OR ``ftp``, so a checked
    ``https://`` fetch could still be bounced to ``ftp://internal-host/...`` -- the allow-list this
    module exists to enforce would hold for hop 0 only. Host/IP-range filtering (localhost, RFC1918,
    169.254.169.254) is explicitly OUT of scope here: this module gates schemes, and a redirect to
    an internal ``http://`` address is still allowed.

    It ALSO strips credential-bearing request headers on a CROSS-HOST hop. urllib's stock
    ``redirect_request`` copies every header off the original request onto the new one, dropping only
    ``content-length``/``content-type`` and comparing no hosts at all -- so a caller passing
    ``headers={"Authorization": "Bearer <token>"}`` had that token re-sent verbatim to whatever host
    the origin server named in ``Location``, with no error and no log line. Same-host hops (the
    ordinary ``http->https`` or trailing-slash redirect) keep their headers, so authenticated APIs
    that redirect within their own host keep working.
    """

    # Matched case-insensitively against the copied header names. ``Cookie`` is included alongside
    # the two auth headers because a session cookie is a bearer credential in every way that matters
    # here; browsers likewise never send one cross-origin on a redirect.
    SENSITIVE_HEADERS: FrozenSet[str] = frozenset({"authorization", "proxy-authorization", "cookie", "cookie2"})

    def __init__(self, allowed_schemes: FrozenSet[str] = ALLOWED_SCHEMES) -> None:
        self.allowed_schemes = allowed_schemes

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        """Validates ``newurl`` against this handler's scheme allow-list, delegates to urllib's stock
        redirect handling, then removes :attr:`SENSITIVE_HEADERS` from the resulting request when the
        hop crosses to a different host -- so a hop to a disallowed scheme raises
        :class:`UnsafeURLError` and a hop to another host cannot carry the caller's credentials."""
        require_http_url(newurl, self.allowed_schemes)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        old_host = urllib.parse.urlsplit(req.full_url).netloc.lower()
        new_host = urllib.parse.urlsplit(new_req.full_url).netloc.lower()
        if old_host != new_host:
            # ``Request.headers`` keys are capitalized by ``Request.add_header``; compare lowercased
            # so a caller-supplied "authorization" and urllib's "Authorization" are both caught.
            for name in [k for k in new_req.headers if k.lower() in self.SENSITIVE_HEADERS]:
                del new_req.headers[name]
            for name in [k for k in getattr(new_req, "unredirected_hdrs", {}) if k.lower() in self.SENSITIVE_HEADERS]:
                del new_req.unredirected_hdrs[name]
        return new_req


def urlopen_checked(
    target: Union[str, urllib.request.Request],
    timeout: float = 30,
    allowed_schemes: FrozenSet[str] = ALLOWED_SCHEMES,
    *,
    proxies: Optional[Dict[str, str]] = None,
) -> Any:
    """``urllib.request.urlopen`` with the scheme validated first -- on the supplied URL AND on every
    redirect hop. Accepts a URL string or a prepared ``Request``.

    Returns the raw ``urlopen`` context manager, so an existing ``with urlopen(...) as resp:``
    call site keeps working unchanged once swapped to call this instead.

    ``proxies`` controls the ROUTE, which for a metered pool is a billing decision rather than a
    connectivity one:

    * ``None`` (default) -- unchanged behaviour: urllib's usual ``ProxyHandler`` reads
      ``http_proxy``/``https_proxy`` from the environment.
    * ``{}`` -- NO proxy, whatever the environment says. This is urllib's equivalent of
      ``requests``' ``proxies={}, trust_env=False``, and it is the only way a caller can make
      "direct" a property of the call rather than of the machine it happens to run on.
    * a mapping -- exactly those proxies, environment ignored.
    """
    url = target if isinstance(target, str) else target.full_url
    require_http_url(url, allowed_schemes)
    # A dedicated opener (not the module-level default) so the redirect check applies here without
    # mutating global urllib state that other libraries in the process share.
    handlers: list = [_CheckedRedirectHandler(allowed_schemes)]
    if proxies is not None:
        # FIRST in the chain and explicit: `build_opener` adds a default `ProxyHandler` only when
        # the caller supplies none, so passing one -- including an empty one -- is what suppresses
        # the environment lookup.
        handlers.insert(0, urllib.request.ProxyHandler(proxies))
    opener = urllib.request.build_opener(*handlers)
    # nosec B310 - the call above raises UnsafeURLError unless the scheme is allowed, and the opener
    # re-checks each redirect target; this is the one audited urlopen call bandit cannot see the
    # guard through.
    return opener.open(target, timeout=timeout)  # nosec B310
