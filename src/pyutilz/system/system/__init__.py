"""pyutilz.system.system subpackage (split from a 2024-LOC monolith).

Behaviour-preserving carve into cohesive submodules. This __init__ re-exports
EVERY public name that ``pyutilz.system.system`` exposed before the split, so
``from pyutilz.system import system`` / ``system.ensure_dir_exists`` etc. keep
resolving unchanged. Submodules:
  _common  - WMI helpers, remove_nas, decode_* , summarize_devices
  probing  - CPU / GPU / OS-power / battery / large-pages hardware probing
  memory   - RAM usage, clean_ram, tracemalloc helpers
  fsutils  - ensure_dir_exists, disk free space, list_linux_devices
  misc     - IPython/tqdm, software info, process utils, idle-device monitor, locale, beep
  sysinfo  - get_system_info orchestrator (preserves the Windows os_serial fallback)
"""

from ._common import *
from .probing import *
from .memory import *
from .fsutils import *
from .misc import *
from .sysinfo import *

# The five ``from . import *`` lines above also bind the submodule objects
# themselves (``_common``, ``probing``, ``memory``, ``fsutils``, ``misc``,
# ``sysinfo``) as attributes of this package -- ordinary Python import
# machinery behaviour. Deleting them via ``globals().pop(...)`` (the original
# approach here) is fragile: anything that re-imports a submodule afterwards
# (``importlib.import_module("pyutilz.system.system.misc")``,
# ``unittest.mock.patch("pyutilz.system.system.misc.foo")`` via
# ``pkgutil.resolve_name``, a plain ``import pyutilz.system.system.misc``
# elsewhere in the process) makes CPython's import system re-set the deleted
# attribute as a side effect -- so whether ``dir()`` includes these names
# becomes dependent on unrelated test/import ORDER, not a fixed property of
# the module. A ``__dir__`` hook (PEP 562) reports the curated surface
# without ever deleting the real attribute, so every access path (direct
# attribute, ``getattr``, ``importlib``, ``mock.patch``) stays reliable
# regardless of what else has been imported first.
_SUBMODULE_NAMES = frozenset({"_common", "probing", "memory", "fsutils", "misc", "sysinfo"})


def __dir__():
    return sorted(n for n in globals() if n not in _SUBMODULE_NAMES)
