"""PyUtilz cloud subpackage."""

# Re-export all from cloud.cloud for backward compatibility
from .cloud import *

# Explicit submodule import so the name __all__ promises is actually bound
# on the package (relying on the from-import-* submodule fallback is fragile
# under static analysis and lazy-import edge cases).
from . import cloud

# mail.ru Cloud public folders: a credential-free way to publish and restore bulk build inputs. Kept a
# SUBMODULE rather than star-exported, because its names (`download_host`, `weblink_of`) are meaningful
# only in that context and would read as generic cloud helpers on the package.
from . import mailru

__all__ = ["cloud", "mailru"]
