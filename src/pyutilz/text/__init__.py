"""PyUtilz text subpackage."""

__all__ = ["strings", "tokenizers", "similarity", "humanizer", "secrets_scrub"]


def __getattr__(name):
    """Bind this package's submodules on attribute access (PEP 562), matching `pyutilz.performance`.

    Without it the three spellings of the same import disagree: `from pyutilz.text import strings` and
    `from pyutilz.text import *` both work, while `import pyutilz.text as p; p.strings` raises AttributeError -- so a
    downstream module that only ever does the third fails at runtime with no import-time warning.
    Lazy rather than an eager `from . import ...` so importing the package costs nothing until a
    submodule is actually touched (several pull optional third-party dependencies).
    """
    if name in __all__:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
