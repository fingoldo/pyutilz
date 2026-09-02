"""Feature detection for psutil APIs that exist only on some platforms.

psutil defines several of its module-level functions *conditionally*: ``psutil/__init__.py`` wraps
them in ``if hasattr(_psplatform, "<name>")`` blocks (``cpu_freq``, ``sensors_temperatures``,
``sensors_fans``, ``sensors_battery``). On a platform whose backend does not implement one, the
attribute is therefore missing from the ``psutil`` module altogether -- calling it raises
``AttributeError`` rather than returning a documented "unknown" value the caller can branch on.

The case that motivated this module: macOS has no ``psutil.cpu_freq``, so ``psutil.cpu_freq()``
raised on *every* hardware sample and on every ``get_system_info()`` call there. A missing
platform capability is a known, expected absence and must be reported as such -- not surfaced as a
per-sample exception, and not swallowed by a broad ``except``.

Two distinct "no value" cases exist and are deliberately collapsed into ``None`` by the getters
here, because callers must treat them identically: the function is absent (platform lacks the
capability) or the function exists but returns ``None``/empty (kernel or VM does not expose the
counter). What differs is the diagnostics: absence is reported once via
:func:`missing_psutil_functions` and a single log line, never once per sample.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import psutil

from types import ModuleType
from typing import Any, Optional, Set, Tuple

# psutil's own optional (platform-gated) module-level functions, in the order they appear in its
# __init__.py. Keep in sync with psutil upstream if new gated functions are adopted here.
OPTIONAL_PSUTIL_FUNCTIONS: Tuple[str, ...] = ("cpu_freq", "sensors_temperatures", "sensors_fans", "sensors_battery")

# Names already reported as absent, so the warning is emitted once per process rather than once per
# sample -- a monitor sampling at 1 Hz would otherwise fill the log with the same platform fact.
_ABSENCE_LOGGED: Set[str] = set()


def has_psutil_function(name: str, psutil_module: Optional[ModuleType] = None) -> bool:
    """Return whether this platform's psutil exposes ``name``.

    Evaluated on every call rather than cached at import time: the answer is constant for a real
    process, but tests simulate other platforms by deleting attributes from the psutil module, and
    a cached snapshot would make that simulation a no-op.
    """
    return hasattr(psutil_module if psutil_module is not None else psutil, name)


def missing_psutil_functions(psutil_module: Optional[ModuleType] = None) -> Tuple[str, ...]:
    """Return the subset of :data:`OPTIONAL_PSUTIL_FUNCTIONS` this platform does not provide.

    Intended for inclusion in reported results, so a consumer can tell "this metric is not
    measurable here" apart from "this metric measured zero".
    """
    return tuple(name for name in OPTIONAL_PSUTIL_FUNCTIONS if not has_psutil_function(name, psutil_module))


def _log_absence_once(name: str) -> None:
    """Reports a platform-absent psutil function the first time it is asked for, and stays silent after.

    The absence is a fixed property of the platform, so repeating it once per sample would bury every other line in the log.
    """
    if name not in _ABSENCE_LOGGED:
        _ABSENCE_LOGGED.add(name)
        logger.info("psutil.%s() is not available on this platform; the metrics derived from it will be reported as unavailable", name)


def get_cpu_freq(percpu: bool = False, psutil_module: Optional[ModuleType] = None) -> Optional[Any]:
    """``psutil.cpu_freq()`` that returns ``None`` instead of raising where it does not exist.

    Args:
        percpu: Passed straight through to ``psutil.cpu_freq``.
        psutil_module: psutil reference to query. Call sites pass their own module-level ``psutil``
            name so that a test patching *that* name (rather than the real psutil module) still
            drives this helper; ``None`` means the psutil imported here.

    Returns:
        The ``scpufreq`` namedtuple (or list of them for ``percpu=True``), or ``None`` when the
        platform has no ``cpu_freq`` at all or psutil itself reports no frequency information.
    """
    module = psutil_module if psutil_module is not None else psutil
    if not has_psutil_function("cpu_freq", module):
        _log_absence_once("cpu_freq")
        return None
    return module.cpu_freq(percpu=percpu)


def get_cpu_freq_current_mhz(psutil_module: Optional[ModuleType] = None) -> Optional[float]:
    """Current aggregate CPU clock in MHz, or ``None`` if this platform cannot report one.

    ``None`` rather than ``0.0``: a zero would be averaged in downstream as a genuine measurement of
    an idle-at-0-MHz CPU, which is never true and silently corrupts a run profile.
    """
    freq = get_cpu_freq(psutil_module=psutil_module)
    if freq is None:
        return None
    current = getattr(freq, "current", None)
    return None if current is None else float(current)
