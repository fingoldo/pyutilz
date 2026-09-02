"""Typed exceptions for the web domain, mirroring pyutilz.llm.exceptions's pattern.

Lets callers discriminate error conditions via ``except SpecificError`` instead of string-matching
a generic ``RuntimeError``/``ValueError`` message. Not a full hierarchy for every condition --
just the highest-traffic ones.
"""

from __future__ import annotations


class WebError(Exception):
    """Root of the web domain's exception hierarchy.

    Exists so ``except WebError`` reliably catches every error this package raises on purpose.
    Each concrete subclass keeps its original builtin base as a SECOND base, so code written
    against ``except RuntimeError``/``except ValueError`` keeps working unchanged.
    """


class ProxyConfigurationError(WebError, RuntimeError):
    """A required proxy credential/setting (API key, host, port) is missing or malformed."""


class ProxyFetchError(WebError, RuntimeError):
    """A proxy-provider API call (subscriptions, endpoint list) failed against every endpoint tried."""


class UnsafeURLError(WebError, ValueError):
    """Raised when a URL's scheme is not one the caller is permitted to open.

    A ``ValueError`` subclass because that is what callers building a URL from untrusted data
    already guard against, and because it must NOT be an ``OSError``: retry-on-transient-network-
    fault logic typically retries ``OSError``, and retrying a rejected scheme would only delay the
    same refusal. Lives here rather than in ``url_guard`` so the web domain has exactly one
    exception home; ``pyutilz.web.url_guard.UnsafeURLError`` still resolves via re-export.
    """
