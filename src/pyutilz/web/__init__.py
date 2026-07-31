"""PyUtilz web subpackage."""

# Re-export all from web.web for backward compatibility
from .web import *

# Explicit submodule imports so every name __all__ promises is actually bound
# on the package (relying on the from-import-* submodule fallback for these
# is fragile under static analysis and lazy-import edge cases).
from . import browser, graphql, proxy, web, url_guard

# cached_client is NOT eagerly imported here (unlike its siblings above): it pulls in
# pyutilz.core.serialization -> pyutilz.system.system, a heavier transitive chain that broke
# `import pyutilz.web.web` under the [web]-only optional-dep isolation test (parent-package
# __init__.py always runs before any leaf submodule import). `from pyutilz.web import
# cached_client` still works via Python's normal submodule-import fallback -- no __getattr__
# needed, since nothing here ever raises for the name.
__all__ = ["browser", "web", "graphql", "proxy", "url_guard", "cached_client"]
