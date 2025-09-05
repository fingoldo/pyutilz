"""Repeatable code to init almost all my Jupyter notebooks ;-)"""

# =============================================================================
# FIXED VERSION: pyutilz/notebook_init.py
# =============================================================================

"""
Jupyter notebook initialization utilities
"""
import os
import psutil
from IPython.display import display, HTML
from IPython import get_ipython

def setup_polars_config():
    """Setup Polars threading and memory configuration"""
    os.environ["POLARS_MAX_THREADS"] = str(max(1, int(psutil.cpu_count(logical=False)/2)))
    print(f"Using {os.environ['POLARS_MAX_THREADS']} polars threads")
    os.environ["_RJEM_MALLOC_CONF"] = "muzzy_decay_ms:500"

def setup_jupyter_display():
    """Setup Jupyter notebook display options"""
    try:
        display(HTML("<style>.container { width:100% !important; }</style>"))
        print("✅ Jupyter display configured")
    except Exception as e:
        print(f"⚠️  Display setup failed: {e}")

def load_jupyter_extensions():
    """Load common Jupyter extensions using IPython API"""
    ipython = get_ipython()
    if not ipython:
        print("⚠️  Not running in IPython environment")
        return
    
    extensions_to_load = [
        'line_profiler',
        'autoreload', 
        'autotime',
        # 'nb_black'  # uncomment if needed
    ]
    
    loaded = []
    failed = []
    
    for ext in extensions_to_load:
        try:
            # Use magic() method instead of % syntax
            ipython.magic(f'load_ext {ext}')
            loaded.append(ext)
        except Exception as e:
            failed.append((ext, str(e)))
    
    # Set autoreload
    try:
        ipython.magic('autoreload 2')
        loaded.append('autoreload 2')
    except Exception as e:
        failed.append(('autoreload 2', str(e)))
    
    if loaded:
        print(f"✅ Loaded extensions: {', '.join(loaded)}")
    if failed:
        print(f"⚠️  Failed to load: {[f[0] for f in failed]}")

def import_common_packages():
    """Import common packages and return them"""
    packages = {}
    
    try:
        import pandas as pd
        packages['pd'] = pd
        # Set pandas options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
    except ImportError:
        print("⚠️  pandas not available")
    
    try:
        import numpy as np
        packages['np'] = np
    except ImportError:
        print("⚠️  numpy not available")
    
    try:
        import matplotlib.pyplot as plt
        packages['plt'] = plt
    except ImportError:
        print("⚠️  matplotlib not available")
    
    try:
        import seaborn as sns
        packages['sns'] = sns
    except ImportError:
        print("⚠️  seaborn not available")
    
    if packages:
        print(f"📦 Imported packages: {', '.join(packages.keys())}")
    
    return packages

def init_notebook(include_imports=True, inject_globals=False):
    """
    Complete notebook initialization
    
    Args:
        include_imports (bool): Whether to import common packages
        inject_globals (bool): Whether to inject packages into global namespace
    
    Returns:
        dict: Dictionary of imported packages if include_imports=True
    """
    print("🚀 Initializing notebook environment...")
    
    setup_polars_config()
    setup_jupyter_display()
    load_jupyter_extensions()
    
    if include_imports:
        packages = import_common_packages()
        
        # Inject into global namespace if requested
        if inject_globals:
            try:
                # Get the calling frame's globals
                import inspect
                frame = inspect.currentframe().f_back
                frame.f_globals.update(packages)
                print("📌 Packages injected into global namespace")
            except:
                print("⚠️  Could not inject into global namespace")
        
        print("✅ Notebook initialization complete!")
        return packages
    else:
        print("✅ Notebook initialization complete!")
        return {}

# For %run -m compatibility
def main():
    """Main function for command-line execution"""
    packages = init_notebook(include_imports=True, inject_globals=True)
    return packages

if __name__ == "__main__":
    main()