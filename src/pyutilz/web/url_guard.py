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


class UnsafeURLError(ValueError):
    """Raised when a URL's scheme is not one the caller is permitted to open.

    A ``ValueError`` subclass because that is what callers building a URL from untrusted data
    already guard against, and because it must NOT be an ``OSError``: retry-on-transient-network-
    fault logic typically retries ``OSError``, and retrying a rejected scheme would only delay the
    same refusal.
    """


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


def urlopen_checked(target: Union[str, urllib.request.Request], timeout: float = 30, allowed_schemes: FrozenSet[str] = ALLOWED_SCHEMES) -> Any:
    """``urllib.request.urlopen`` with the scheme validated first. Accepts a URL string or a
    prepared ``Request``.

    Returns the raw ``urlopen`` context manager, so an existing ``with urlopen(...) as resp:``
    call site keeps working unchanged once swapped to call this instead.
    """
    url = target if isinstance(target, str) else target.full_url
    require_http_url(url, allowed_schemes)
    # nosec B310 - the line above raises UnsafeURLError unless the scheme is allowed; this is the
    # one audited urlopen call bandit cannot see the guard through.
    return urllib.request.urlopen(target, timeout=timeout)  # nosec B310
