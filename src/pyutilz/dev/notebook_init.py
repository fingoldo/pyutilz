"""
Jupyter notebook initialization utilities
"""
import logging
import os
import sys
from typing import Any

import psutil

logger = logging.getLogger(__name__)

try:
    from IPython import get_ipython
except ImportError:
    # Outside a Jupyter/IPython environment real get_ipython() returns None too,
    # so this dummy matches that contract exactly.
    def get_ipython():
        """Fallback used when IPython is unavailable; matches real get_ipython()'s behavior outside a notebook (returns None)."""
        return None

# Correct imports for display and HTML
display: Any
HTML: Any
try:
    from IPython.display import display, HTML  # type: ignore[no-redef]
except ImportError:
    # Fallback for older IPython versions or when not in notebook
    try:
        from IPython.core.display import HTML  # type: ignore[no-redef]
        from IPython.display import display  # type: ignore[no-redef]
    except ImportError:
        # Create dummy functions if IPython not available
        def display(*args, **kwargs):
            """No-op fallback used when IPython.display is unavailable."""
            pass
        def HTML(data):
            """No-op fallback used when IPython.display is unavailable; returns data unchanged."""
            return data

def setup_polars_config():
    """Setup Polars threading and memory configuration.

    ``POLARS_MAX_THREADS`` is read by polars ONCE, at import; setting it afterwards has no effect on
    the already-built thread pool, so this reports honestly instead of printing a thread count polars
    is not using.
    """
    physical_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    os.environ["POLARS_MAX_THREADS"] = str(max(1, int(physical_cores / 2)))
    if "polars" in sys.modules:
        print(f"[!] polars is already imported: POLARS_MAX_THREADS={os.environ['POLARS_MAX_THREADS']} will NOT apply to this session; call setup_polars_config() before importing polars")
    else:
        print(f"Using {os.environ['POLARS_MAX_THREADS']} polars threads")
    os.environ["_RJEM_MALLOC_CONF"] = "muzzy_decay_ms:500"

def _run_magic(ipython, line: str) -> None:
    """Run one IPython line magic, preferring ``run_line_magic`` over the removed ``magic()``.

    ``InteractiveShell.magic()`` was deprecated for years and REMOVED in IPython 9.0, so every call
    site raised there -- and each is wrapped in a handler that only appends to a `failed` list, so
    init_notebook still printed "[OK]Notebook initialization complete!" with autoreload silently off.
    """
    magic_name, _, rest = line.partition(" ")
    run_line_magic = getattr(ipython, "run_line_magic", None)
    if run_line_magic is not None:
        run_line_magic(magic_name, rest)
        return
    ipython.magic(line)


def setup_jupyter_display():
    """Setup Jupyter notebook display options"""
    try:
        display(HTML("<style>.container { width:100% !important; }</style>"))
        print("[OK]Jupyter display configured")
    except Exception as e:
        print(f"[!] Display setup failed: {e}")

def load_jupyter_extensions():
    """Load common Jupyter extensions using IPython API"""
    ipython = get_ipython()
    if not ipython:
        print("[!] Not running in IPython environment")
        return

    extensions_to_load = [
        "line_profiler",
        "autoreload",
        "autotime",
        # 'nb_black'  # uncomment if needed
    ]

    loaded = []
    failed = []

    for ext in extensions_to_load:
        try:
            _run_magic(ipython, f"load_ext {ext}")
            loaded.append(ext)
        except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional (one extension failing shouldn't skip the rest)
            failed.append((ext, str(e)))

    # Set autoreload
    try:
        _run_magic(ipython, "autoreload 2")
        loaded.append("autoreload 2")
    except Exception as e:
        failed.append(("autoreload 2", str(e)))

    if loaded:
        print(f"[OK]Loaded extensions: {', '.join(loaded)}")
    if failed:
        # Both printed AND logged at WARNING: a print inside a notebook cell scrolls away, and a
        # silently-off autoreload is exactly the failure people spend an hour not noticing.
        print(f"[!] Failed to load: {[f[0] for f in failed]}")
        logger.warning("notebook_init: could not load IPython extensions: %s", failed)

def import_common_packages():
    """Import common packages and return them"""
    packages = {}

    try:
        import pandas as pd
        packages['pd'] = pd
        # Set pandas options
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
    except ImportError:
        print("[!] pandas not available")

    try:
        import numpy as np
        packages['np'] = np
    except ImportError:
        print("[!] numpy not available")

    try:
        import matplotlib.pyplot as plt
        packages['plt'] = plt
    except ImportError:
        print("[!] matplotlib not available")

    try:
        import seaborn as sns
        packages['sns'] = sns
    except ImportError:
        print("[!] seaborn not available")

    if packages:
        print(f"[pkg] Imported packages: {', '.join(packages.keys())}")

    return packages

def load_jupyter_extensions_minimal():
    """Load only built-in IPython extensions"""
    ipython = get_ipython()
    if not ipython:
        print("[!] Not running in IPython environment")
        return

    # Only load built-in extensions that should always be available
    try:
        _run_magic(ipython, "load_ext autoreload")
        _run_magic(ipython, "autoreload 2")
        print("[OK]Loaded autoreload extension")
    except Exception as e:
        print(f"[!] Failed to load autoreload: {e}")

def init_notebook(include_imports=True, inject_globals=False, use_simple_extensions=True):
    """
    Complete notebook initialization

    Args:
        include_imports (bool): Whether to import common packages
        inject_globals (bool): Whether to inject packages into global namespace
        use_simple_extensions (bool): Use simple extension loading method

    Returns:
        dict: Dictionary of imported packages if include_imports=True
    """
    print("[*] Initializing notebook environment...")

    setup_polars_config()
    setup_jupyter_display()

    # Choose extension loading method
    if use_simple_extensions:
        load_jupyter_extensions_minimal()
    else:
        load_jupyter_extensions()

    if include_imports:
        packages = import_common_packages()

        if inject_globals:
            try:
                import inspect
                _this_frame = inspect.currentframe()
                frame = _this_frame.f_back if _this_frame is not None else None
                if frame is None:
                    raise RuntimeError("no caller frame available")
                frame.f_globals.update(packages)
                print("[*] Packages injected into global namespace")
            except Exception as e:
                # Bind and report the reason, like every other handler in this module: a bare
                # `except Exception:` left the caller with no way to tell what actually failed.
                print(f"[!] Could not inject into global namespace: {e}")
                logger.warning("notebook_init: global-namespace injection failed: %s", e)

        print("[OK]Notebook initialization complete!")
        return packages
    else:
        print("[OK]Notebook initialization complete!")
        return {}

# For %run -m compatibility
def main():
    """Main function for command-line execution"""
    packages = init_notebook(include_imports=True, inject_globals=True)
    return packages

if __name__ == "__main__":
    main()
