"""Stand-ins for claude-code-sdk types, bound only when the SDK is not installed.

The isinstance check in ``ClaudeCodeProvider._assistant_message_text`` needs a ``TextBlock`` name either way. The
type-ignore codes on the import in ``claude_code_provider`` cover both environments: ``no-redef`` when the SDK import
failed and nothing above bound the name, and ``assignment`` when the SDK is installed and mypy sees its real type.
"""

from __future__ import annotations


class TextBlock:
    """Text block placeholder carrying the one attribute the provider reads, ``text``."""

    def __init__(self, text: str = "") -> None:
        self.text = text
