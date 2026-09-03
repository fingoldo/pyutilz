"""PyUtilz core subpackage."""

__all__ = [
    "pythonlib",
    "serialization",
    "image",
    "openai",
    "filemaker",
    "matrix",
    "disk_cache",
    "PickleVerificationError",
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
