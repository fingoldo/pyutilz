"""One parser for boolean environment flags.

Every PyUtilz boolean switch reads through :func:`env_flag`, so ``FLAG=0`` means off and ``FLAG=on`` means on no matter
which switch is being set. Accepted spellings are case-insensitive and surrounding whitespace is ignored:

* on: ``1``, ``true``, ``yes``, ``on``
* off: ``0``, ``false``, ``no``, ``off``

An unset, empty or unrecognised value returns the caller's ``default``.
"""

from __future__ import annotations

import os

__all__ = ["env_flag", "TRUE_SPELLINGS", "FALSE_SPELLINGS"]

TRUE_SPELLINGS = frozenset({"1", "true", "yes", "on"})
FALSE_SPELLINGS = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, default: bool = False) -> bool:
    """Return the boolean value of environment variable ``name``, or ``default`` when it is unset or unrecognised."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in TRUE_SPELLINGS:
        return True
    if raw in FALSE_SPELLINGS:
        return False
    return default
