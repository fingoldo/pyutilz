"""Pure guards for the Claude Code CLI path: attempt budget, JSON fence stripping and argv safety."""

import re


def _check_attempt_budget(attempt: int, max_attempts: int) -> None:
    """Raise once ``attempt`` exceeds ``max_attempts`` (0 means unlimited)."""
    if max_attempts != 0 and attempt > max_attempts:
        raise RuntimeError(f"ClaudeCodeProvider: exceeded {max_attempts} retry attempts")


def _strip_json_fence(text: str) -> str:
    """Return the body of a leading ```json fence in ``text`` when one is present, else ``text`` unchanged."""
    if not text:
        return text
    stripped = text.strip()
    fence_match = re.search(
        r'```(?:json)?\s*([\s\S]*?)\s*```', stripped,
    )
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _assert_argv_shell_safe(cmd: list) -> None:
    """Refuse a CLI argv carrying a shell-significant character (the prompt and system prompt never go in argv)."""
    unsafe_argv = set('&|<>^"')
    for arg in cmd[1:]:
        if unsafe_argv & set(arg):
            raise ValueError(
                f"refusing to spawn the CLI with a shell-significant character in argv: {arg!r}. "
                "Prompt content belongs on stdin and the system prompt in a file."
            )
