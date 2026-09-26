"""The "argument omitted" marker for ``get_llm_settings(env_file=...)``, kept out of ``config`` on purpose.

``importlib.reload(pyutilz.llm.config)`` re-executes that module; a marker defined there would be minted anew, while
anything bound before the reload (``factory.py``'s ``from pyutilz.llm.config import get_llm_settings``) still holds the
old one, so an ``is`` test against it would miss. Living in its own module, the marker survives a reload of ``config``.
"""

from enum import Enum


class _Unset(Enum):
    """The one "argument omitted" marker for ``get_llm_settings(env_file=...)``."""

    ENV_FILE = 0


UNSET_ENV_FILE = _Unset.ENV_FILE
