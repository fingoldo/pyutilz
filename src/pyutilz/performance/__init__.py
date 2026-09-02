"""Performance tooling: per-host kernel auto-tuning + benchmarking helpers."""

__all__ = ["kernel_tuning"]


def __getattr__(name):
    """Bind the subpackage's submodules on attribute access (PEP 562), matching every sibling
    subpackage. Lazy rather than an eager `from . import kernel_tuning` so that importing this
    package costs nothing until a tuner is actually touched.
    """
    if name in __all__:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
