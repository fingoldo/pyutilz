"""One-line ``logging.basicConfig`` setup for standalone CLI entry points.

Extracted from autopsia, where 11 independent ``if __name__ == "__main__":`` scripts each called
``logging.basicConfig`` with a slightly different, hand-copied format string (some plain
``"%(message)s"``, some timestamped, a couple with no format at all) -- harmless individually
(each runs as its own process, so there's no ``basicConfig``-is-global collision), but needless
repetition of the same one-liner with no single source of truth for the project's preferred format.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TextIO

DEFAULT_FORMAT = "%(asctime)s %(message)s"


def setup_cli_logging(
    level: int = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    stream: Optional[TextIO] = None,
    force: bool = True,
) -> None:
    """Call once at the top of a CLI script's ``if __name__ == "__main__":`` block.

    Thin wrapper around ``logging.basicConfig`` -- exists so every entry point in a project shares
    one format by default instead of each hand-copying a slightly different string, not to hide
    what it does. ``stream`` defaults to ``basicConfig``'s own default (stderr) when ``None``.

    ``basicConfig`` silently no-ops if the root logger already has a handler -- harmless for a
    genuinely standalone CLI process, but the same call inside a pytest process where an earlier
    test already triggered ``basicConfig`` becomes a silent no-op: the emitted log line goes
    nowhere and the caller sees empty output with no error. ``force=True`` (the default here,
    unlike raw ``basicConfig``) always replaces the root handlers so this function's own contract
    -- "call this and logging is configured" -- holds regardless of prior process state. Pass
    ``force=False`` to restore ``basicConfig``'s own opt-out-if-already-configured behaviour.
    """
    kwargs: Dict[str, Any] = {"level": level, "format": fmt, "force": force}
    if stream is not None:
        kwargs["stream"] = stream
    logging.basicConfig(**kwargs)
