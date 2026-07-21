"""PyUtilz system subpackage."""

# Re-export all from system.system for backward compatibility. system.system is comparatively
# lightweight (psutil/platform/os-level helpers) so it stays eager -- it's what
# core.serialization's `pyutilz.system.system.ensure_dir_exists()` and data.polarslib's
# `pyutilz.system.system.clean_ram()` actually need, and eagerly importing IT does not force
# any of the heavier submodules below to load.
from . import system
from .system import *

# parallel/monitoring/hardware_monitor/distributed are all lazy (PEP 562 __getattr__ below) --
# NOT eagerly imported here. `distributed` was the original fix for this bug class (found
# 2026-07-09: it transitively imports pyutilz.web -- requests, selenium, etc, all under the
# optional [web] extra -- purely for one get_external_ip() call; importing it eagerly forced
# every caller of an unrelated pyutilz.system submodule to have [web] installed too, breaking
# mlframe's CI at collection time on ModuleNotFoundError: selenium). parallel/monitoring/
# hardware_monitor had the identical bug for psutil/pandas/tqdm (all under [system]) -- found
# 2026-07-21 audit: `pyutilz.system.system.ensure_dir_exists`/`clean_ram` callers (core.serialization,
# data.polarslib) were forced to have the WHOLE [system] extras stack installed just to import
# THIS package's __init__.py, even though neither of them touches parallel/monitoring/hardware_monitor.
__all__ = ["system", "parallel", "monitoring", "distributed", "hardware_monitor", "UtilizationMonitor"]

_LAZY_SUBMODULES = frozenset({"parallel", "monitoring", "hardware_monitor", "distributed"})


def __getattr__(name):
    # importlib.import_module (not `from . import parallel`): the latter resolves via
    # getattr(this_module, "parallel"), which re-enters THIS __getattr__ and recurses
    # infinitely since "parallel" is never a plain module attribute until cached below.
    # import_module goes straight through sys.modules by fully-qualified name, side-stepping
    # that. Caching the result on the module dict (not just returning it) means __getattr__
    # only fires once per name.
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name == "UtilizationMonitor":
        from .hardware_monitor import UtilizationMonitor as _UtilizationMonitor
        globals()["UtilizationMonitor"] = _UtilizationMonitor
        return _UtilizationMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
