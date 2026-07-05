"""PyUtilz web subpackage."""

# Re-export all from web.web for backward compatibility
from .web import *  # noqa: F401,F403

# Explicit submodule imports so every name __all__ promises is actually bound
# on the package (relying on the from-import-* submodule fallback for these
# is fragile under static analysis and lazy-import edge cases).
from . import browser, graphql, proxy, web  # noqa: F401

__all__ = ['browser', 'web', 'graphql', 'proxy']
