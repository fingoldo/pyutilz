"""PyUtilz core subpackage."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .array_summary import column_sum_min_max
    from .disk_cache import compose_key, hash_object
    from .filemaker import init, post_filemaker_record, simplify_types
    from .image import get_image_properties
    from .matrix import get_sparse_memory_usage
    from .safe_pickle import safe_dump
    from .serialization import serialize, str_to_class, unserialize

# Resolved on first attribute access by __getattr__ (lazy: the defining modules may need optional extras).
_REEXPORTS = {
    "column_sum_min_max": "array_summary",
    "compose_key": "disk_cache",
    "hash_object": "disk_cache",
    "init": "filemaker",
    "post_filemaker_record": "filemaker",
    "simplify_types": "filemaker",
    "get_image_properties": "image",
    "get_sparse_memory_usage": "matrix",
    "safe_dump": "safe_pickle",
    "serialize": "serialization",
    "str_to_class": "serialization",
    "unserialize": "serialization",
}


__all__ = [
    "pythonlib",
    "serialization",
    "image",
    "openai",
    "filemaker",
    "matrix",
    "disk_cache",
    "PickleVerificationError",
    "column_sum_min_max",
    "compose_key",
    "hash_object",
    "init",
    "post_filemaker_record",
    "simplify_types",
    "get_image_properties",
    "get_sparse_memory_usage",
    "safe_dump",
    "serialize",
    "str_to_class",
    "unserialize",
]

def __getattr__(name):
    """Resolve the domain's typed exception and this package's submodules at package level (PEP 562).

    Lazy rather than an eager `from .safe_pickle import ...` / `from . import ...`: safe_pickle
    pulls the whole serialization stack, and this package's __init__ must stay importable with no
    optional dependency installed.

    The submodule arm exists because the three spellings of the same import used to disagree:
    `from pyutilz.core import serialization` and `from pyutilz.core import *` both worked, while
    `import pyutilz.core as c; c.serialization` raised AttributeError -- a runtime failure with no
    import-time warning for any caller that only ever writes the third.
    """
    if name in _REEXPORTS:
        import importlib

        value = getattr(importlib.import_module(f".{_REEXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value

    if name == "PickleVerificationError":
        from .safe_pickle import PickleVerificationError as _PickleVerificationError

        globals()[name] = _PickleVerificationError
        return _PickleVerificationError
    if name in __all__:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
