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

from ._common import *  # noqa: F401,F403
from .probing import *  # noqa: F401,F403
from .memory import *  # noqa: F401,F403
from .fsutils import *  # noqa: F401,F403
from .misc import *  # noqa: F401,F403
from .sysinfo import *  # noqa: F401,F403

# Drop submodule names bound by the ``from . import`` machinery so that
# dir(pyutilz.system.system) matches the pre-split public surface exactly.
for _submod in ("_common", "probing", "memory", "fsutils", "misc", "sysinfo"):
    globals().pop(_submod, None)
del _submod
