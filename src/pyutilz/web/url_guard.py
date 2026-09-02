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
from typing import Any, FrozenSet, Union

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
    """

    def __init__(self, allowed_schemes: FrozenSet[str] = ALLOWED_SCHEMES) -> None:
        self.allowed_schemes = allowed_schemes

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        """Validates ``newurl`` against this handler's scheme allow-list before delegating to urllib's
        stock redirect handling, so a hop to a disallowed scheme raises :class:`UnsafeURLError` instead
        of being followed."""
        require_http_url(newurl, self.allowed_schemes)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen_checked(target: Union[str, urllib.request.Request], timeout: float = 30, allowed_schemes: FrozenSet[str] = ALLOWED_SCHEMES) -> Any:
    """``urllib.request.urlopen`` with the scheme validated first -- on the supplied URL AND on every
    redirect hop. Accepts a URL string or a prepared ``Request``.

    Returns the raw ``urlopen`` context manager, so an existing ``with urlopen(...) as resp:``
    call site keeps working unchanged once swapped to call this instead.
    """
    url = target if isinstance(target, str) else target.full_url
    require_http_url(url, allowed_schemes)
    # A dedicated opener (not the module-level default) so the redirect check applies here without
    # mutating global urllib state that other libraries in the process share.
    opener = urllib.request.build_opener(_CheckedRedirectHandler(allowed_schemes))
    # nosec B310 - the call above raises UnsafeURLError unless the scheme is allowed, and the opener
    # re-checks each redirect target; this is the one audited urlopen call bandit cannot see the
    # guard through.
    return opener.open(target, timeout=timeout)  # nosec B310
