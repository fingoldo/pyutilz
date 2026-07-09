"""Logging proxy that forwards attribute access to `pyutilz.text.strings`'s current `logger`.

The former flat strings.py exposed a single module-level `logger`. After the subpackage split each submodule needs a
`logger`, but callers (and tests) still patch `pyutilz.text.strings.logger`. This proxy forwards every attribute access
to the parent package's current `logger` attribute at call time, so patching the facade's `logger` transparently
affects all submodules — preserving the historic behavior exactly.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING PROXY
# ----------------------------------------------------------------------------------------------------------------------------

import logging


class _FacadeLoggerProxy:
    """Attribute-forwarding stand-in for a `logging.Logger`, resolving to the parent package's live `logger` on every access."""

    __slots__ = ()

    def _resolve(self):
        """Returns the current `logger` attribute of `pyutilz.text.strings`, falling back to a plain named logger if unset."""
        import pyutilz.text.strings as _pkg  # lazy: avoids import cycle at submodule load time

        return getattr(_pkg, "logger", None) or logging.getLogger("pyutilz.text.strings")

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


logger = _FacadeLoggerProxy()
