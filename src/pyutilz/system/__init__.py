"""PyUtilz system subpackage."""

# Re-export all from system.system for backward compatibility
from .system import *
from .hardware_monitor import UtilizationMonitor

# Explicit submodule imports so every name __all__ promises is actually bound
# on the package (relying on the from-import-* submodule fallback for these
# is fragile under static analysis and lazy-import edge cases). `distributed`
# is the one exception (PEP 562 lazy __getattr__ below): per its own module
# docstring it registers the current process as a "distributed SCRAPER node",
# and transitively imports pyutilz.web (requests, selenium, etc. -- all under
# the optional [web] extra) purely for one get_external_ip() call. Importing
# it eagerly here forced every caller of an unrelated pyutilz.system submodule
# (memory, parallel, monitoring...) to have [web] installed too -- found
# 2026-07-09 via mlframe's CI failing ~1300 unrelated tests at collection time
# on ModuleNotFoundError: selenium, reached through this exact chain.
from . import parallel, monitoring, hardware_monitor, system

__all__ = ["system", "parallel", "monitoring", "distributed", "hardware_monitor"]


def __getattr__(name):
    # importlib.import_module (not `from . import distributed`): the latter resolves via
    # getattr(this_module, "distributed"), which re-enters THIS __getattr__ and recurses
    # infinitely since "distributed" is never a plain module attribute. import_module goes
    # straight through sys.modules by fully-qualified name, side-stepping that. Caching the
    # result on the module dict (not just returning it) means __getattr__ only fires once.
    if name == "distributed":
        import importlib
        mod = importlib.import_module(".distributed", __name__)
        globals()["distributed"] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
