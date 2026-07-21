"""Typed exceptions for the web domain, mirroring pyutilz.llm.exceptions's pattern.

Lets callers discriminate error conditions via ``except SpecificError`` instead of string-matching
a generic ``RuntimeError``/``ValueError`` message. Not a full hierarchy for every condition --
just the highest-traffic ones.
"""

from __future__ import annotations


class ProxyConfigurationError(RuntimeError):
    """A required proxy credential/setting (API key, host, port) is missing or malformed."""


class ProxyFetchError(RuntimeError):
    """A proxy-provider API call (subscriptions, endpoint list) failed against every endpoint tried."""
