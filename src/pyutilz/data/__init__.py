"""PyUtilz data subpackage."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .git_checkpoint_cache import read_bytes_with_checkpoint_fallback, read_json_with_checkpoint_fallback, write_bytes_checkpoint, write_json_checkpoint
    from .numbalib import arr2str, generate_combinations_recursive_njit, python_dict_2_numba_dict, set_random_seed
    from .numpylib import get_topk_indices, smart_ratios

# Resolved on first attribute access by __getattr__ (lazy: the defining modules may need optional extras).
_REEXPORTS = {
    "read_bytes_with_checkpoint_fallback": "git_checkpoint_cache",
    "read_json_with_checkpoint_fallback": "git_checkpoint_cache",
    "write_bytes_checkpoint": "git_checkpoint_cache",
    "write_json_checkpoint": "git_checkpoint_cache",
    "arr2str": "numbalib",
    "generate_combinations_recursive_njit": "numbalib",
    "python_dict_2_numba_dict": "numbalib",
    "set_random_seed": "numbalib",
    "get_topk_indices": "numpylib",
    "smart_ratios": "numpylib",
}


__all__ = ["pandaslib", "polarslib", "numpylib", "numbalib", "git_checkpoint_cache", "read_bytes_with_checkpoint_fallback", "read_json_with_checkpoint_fallback", "write_bytes_checkpoint", "write_json_checkpoint", "arr2str", "generate_combinations_recursive_njit", "python_dict_2_numba_dict", "set_random_seed", "get_topk_indices", "smart_ratios"]


def __getattr__(name):
    """Bind this package's submodules on attribute access (PEP 562), matching `pyutilz.performance`.

    Without it the three spellings of the same import disagree: `from pyutilz.data import pandaslib` and
    `from pyutilz.data import *` both work, while `import pyutilz.data as p; p.pandaslib` raises AttributeError -- so a
    downstream module that only ever does the third fails at runtime with no import-time warning.
    Lazy rather than an eager `from . import ...` so importing the package costs nothing until a
    submodule is actually touched (several pull optional third-party dependencies).
    """
    if name in _REEXPORTS:
        import importlib

        value = getattr(importlib.import_module(f".{_REEXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value

    if name in __all__:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
