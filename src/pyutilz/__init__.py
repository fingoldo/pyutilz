"""PyUtilz - Comprehensive Python utilities."""

from .version import __version__

import sys
import types
from importlib import import_module

__all__ = ['__version__', 'core', 'data', 'database', 'web', 'cloud', 'text', 'system', 'dev']

# Module aliases for backward compatibility
# NOTE: Don't create aliases for names that conflict with subpackages (system, web, cloud)
_MODULE_ALIASES = {
    'pythonlib': 'pyutilz.core.pythonlib',
    'serialization': 'pyutilz.core.serialization',
    'image': 'pyutilz.core.image',
    'openai': 'pyutilz.core.openai',
    'filemaker': 'pyutilz.core.filemaker',
    'matrix': 'pyutilz.core.matrix',
    'pandaslib': 'pyutilz.data.pandaslib',
    'polarslib': 'pyutilz.data.polarslib',
    'numpylib': 'pyutilz.data.numpylib',
    'numbalib': 'pyutilz.data.numbalib',
    'db': 'pyutilz.database.db',
    'redislib': 'pyutilz.database.redislib',
    'deltalakes': 'pyutilz.database.deltalakes',
    'browser': 'pyutilz.web.browser',
    'graphql': 'pyutilz.web.graphql',
    'strings': 'pyutilz.text.strings',
    'tokenizers': 'pyutilz.text.tokenizers',
    'similarity': 'pyutilz.text.similarity',
    'parallel': 'pyutilz.system.parallel',
    'monitoring': 'pyutilz.system.monitoring',
    'distributed': 'pyutilz.system.distributed',
    'logginglib': 'pyutilz.dev.logginglib',
    'benchmarking': 'pyutilz.dev.benchmarking',
    'dashlib': 'pyutilz.dev.dashlib',
    'notebook_init': 'pyutilz.dev.notebook_init',
}

def _create_lazy_module(real_module_name):
    """Create a lazy-loading proxy module."""
    def __getattr__(name):
        # Skip dunder attributes to avoid triggering imports from IPython autoreload,
        # inspect.getmodule, hasattr(module, '__file__'), etc.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        real_mod = import_module(real_module_name)
        sys.modules[proxy_mod.__name__] = real_mod
        return getattr(real_mod, name)

    proxy_mod = types.ModuleType(real_module_name.split('.')[-1])
    proxy_mod.__getattr__ = __getattr__
    return proxy_mod

# Register lazy proxy modules
for alias, real_name in _MODULE_ALIASES.items():
    alias_fullname = f'pyutilz.{alias}'
    if alias_fullname not in sys.modules:
        proxy = _create_lazy_module(real_name)
        sys.modules[alias_fullname] = proxy

def __getattr__(name):
    """Lazy import for package-level attributes."""
    if name in ('core', 'data', 'database', 'web', 'cloud', 'text', 'system', 'dev'):
        return import_module(f'.{name}', __name__)
    if name in _MODULE_ALIASES:
        return import_module(_MODULE_ALIASES[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
