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
    """Resolve the domain's typed exception at package level (PEP 562).

    Lazy rather than an eager `from .safe_pickle import ...`: safe_pickle pulls the whole
    serialization stack, and this package's __init__ must stay importable with no optional
    dependency installed.
    """
    if name == "PickleVerificationError":
        from .safe_pickle import PickleVerificationError as _PickleVerificationError

        globals()[name] = _PickleVerificationError
        return _PickleVerificationError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
