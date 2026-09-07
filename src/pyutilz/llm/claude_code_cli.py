"""The Claude Code CLI backend: find the binary, spawn it, and read its stream-json output.

Carved out of ``claude_code_provider`` on 2026-09-07, when that module crossed the 1000-LOC
budget and ``_generate_cli`` crossed the C901 ceiling in the same change. Everything here is
about running one subprocess and turning its event stream into an answer plus its usage; the
provider keeps the policy (retries, cost accounting, the LLMProvider contract).

The names are re-exported from ``claude_code_provider`` for existing importers.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import queue
import shutil
import subprocess  # nosec B404 - only used to spawn the trusted `claude` CLI (resolved via shutil.which / fixed install paths, never a user-supplied path), always with shell=False
import sys
import threading
import time
from types import SimpleNamespace

logger = logging.getLogger(__name__)


from pyutilz.llm.exceptions import ClaudeCodeToolUseError


def _find_claude_executable() -> str:
    """Find the claude executable path safely."""
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path

    if sys.platform == "win32":
        npm_prefix = os.environ.get("APPDATA", "")
        possible_paths = [
            os.path.join(npm_prefix, "npm", "claude.cmd"),
            os.path.join(npm_prefix, "npm", "claude"),
            os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm", "claude.cmd"),
        ]
        for path in possible_paths:
            if os.path.isfile(path):
                return path

    raise FileNotFoundError("Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")


# Windows and POSIX error codes that say "this call will fail identically every time": the binary
# or a path is missing, an argument is malformed, the command line is too long. Retrying any of
# these buys nothing and costs the full backoff schedule.
#
# 2026-09-07: the retry arm below caught OSError wholesale. A 48378-character system prompt in argv
# raised WinError 206 ("the filename or extension is too long") on every attempt; it was retried
# thirteen times with growing backoff and burned 2457 seconds before the caller gave up. WinError
# 206 is fixed (the system prompt goes in a file now), the predicate is the general case: a missing
# `claude` binary would have run the schedule to its 50-attempt end, roughly four hours of sleeping.
_PERMANENT_WINERRORS = frozenset({
    2,    # ERROR_FILE_NOT_FOUND -- no claude binary
    3,    # ERROR_PATH_NOT_FOUND
    5,    # ERROR_ACCESS_DENIED
    8,    # ERROR_NOT_ENOUGH_MEMORY on spawn
    87,   # ERROR_INVALID_PARAMETER -- a malformed argument
    206,  # ERROR_FILENAME_EXCED_RANGE -- the command line is over 32767 characters
    267,  # ERROR_DIRECTORY -- a bad cwd
})
_PERMANENT_ERRNOS = frozenset({
    errno.ENOENT,
    errno.EACCES,
    errno.EPERM,
    errno.E2BIG,
    errno.ENAMETOOLONG,
    errno.EINVAL,
    errno.ENOEXEC,
    errno.EISDIR,
    errno.ENOTDIR,
})

# A CLI hang is transient in kind, but self.timeout is measured in tens of minutes: fifty of them
# is more than a day of wall clock spent on a run nobody is watching. Three is enough to ride out a
# genuine stall and short enough to fail while the failure still means something.
MAX_TIMEOUT_RETRIES = 3


def _is_transient_subprocess_error(exc: BaseException) -> bool:
    """Is ``exc`` worth retrying, or will the next attempt fail exactly as this one did?

    ``ConnectionError`` and ``TimeoutError`` are ``OSError`` subclasses, so the retry arm's tuple
    was in effect ``(OSError, subprocess.TimeoutExpired)`` -- everything a spawn can raise.
    """
    if isinstance(exc, subprocess.TimeoutExpired):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True  # a genuine network blip, whatever its errno
    if isinstance(exc, OSError):
        winerror = getattr(exc, "winerror", None)
        if winerror is not None:
            return winerror not in _PERMANENT_WINERRORS
        if exc.errno is not None:
            return exc.errno not in _PERMANENT_ERRNOS
        # FileNotFoundError from _find_claude_executable carries neither: it is constructed with a
        # message alone, and a missing binary never becomes present by waiting.
        return not isinstance(exc, FileNotFoundError)
    return False


def _raise_on_cli_tool_use(event: dict) -> None:
    """Raise :class:`ClaudeCodeToolUseError` if a CLI ``assistant`` event carries a tool-use block.

    The CLI counterpart of the SDK path's ToolUseBlock tripwire: both backends run with permissions
    bypassed, so a tool-use block means the sandbox they rely on (``--tools ""`` +
    ``--strict-mcp-config``) is not holding -- most plausibly prompt injection in caller-supplied
    text. It raised loudly on the SDK path and passed completely unnoticed here, the caller
    receiving a normal-looking result with nothing recording that a tool ran."""
    message = event.get("message") or {}
    content = message.get("content") or []
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            raise ClaudeCodeToolUseError(f"Claude Code returned a tool-use block ({block.get('name', '?')!r}); this provider is text-generation-only")


def _consume_cli_stream(
    line_q: "queue.Queue[str | None]",
    timeout: float,
    cancel_evt: "threading.Event | None" = None,
) -> "tuple[str | None, str | None, bool, dict | None]":
    """Read stream-json events off ``line_q`` until a ``result`` event, the reader's EOF sentinel, or ``timeout``.

    Returns ``(result_text, error_text, timed_out, result_event)``: exactly one of the first two is
    set unless the deadline expired first, in which case ``timed_out`` is True and both are None.

    ``result_event`` is the raw ``result`` event dict. The CLI puts real ``usage`` (input, output
    and BOTH cache-token counts) and a real ``total_cost_usd`` on it even on a subscription plan,
    and this function used to read only its ``result`` field and drop the rest -- so every CLI-path
    call fell through to the tiktoken estimate in ``generate()``, reported zero cache tokens and
    added nothing to ``total_cost_usd``. The SDK path has always captured this; the CLI path is the
    one actually in use. Split out of
    ``ClaudeCodeProvider._generate_cli`` -- this event dispatch is the whole of that function's
    branching, and inlining it kept the enclosing coroutine at the top of the C901 budget where any
    further edit (this one included) pushed it over.
    """
    result_text: "str | None" = None
    error_text: "str | None" = None
    result_event: "dict | None" = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancel_evt is not None and cancel_evt.is_set():
            return None, None, False, None
        try:
            raw = line_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if raw is None:
            break
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            # Dropped, but not in silence: a CLI that changed its stream format would otherwise
            # surface only as "produced no result", with the actual cause invisible. Debug
            # rather than warning because a partial line at the tail of a killed stream is
            # normal and would otherwise be noise on every cancelled call.
            logger.debug("Ignoring a stream line that is not JSON (%d chars): %.120s", len(raw), raw)
            continue

        etype = event.get("type")
        if etype == "result":
            result_event = event
            subtype = event.get("subtype", "")
            if subtype == "success":
                # An empty success is not an answer. The SDK path raises "produced no result"
                # for exactly this shape, and returning "" here handed the caller a blank
                # string that then failed JSON parsing with a far less informative message.
                text_out = event.get("result", "")
                if isinstance(text_out, str) and text_out.strip():
                    result_text = text_out
                else:
                    error_text = f"the CLI reported success with an empty result (subtype={subtype!r})"

            else:
                error_text = event.get("result") or event.get("error") or subtype
            break
        elif etype == "assistant":
            _raise_on_cli_tool_use(event)
        elif etype == "rate_limit_event":
            logger.debug("Claude CLI rate_limit_event (continuing)")
        elif etype == "system" and event.get("subtype") == "init":
            logger.debug("Claude CLI initialized")
    else:
        return None, None, True, None
    return result_text, error_text, False, result_event


class _CliResultMessage:
    """The CLI's ``result`` event in the shape ``generate()`` already reads off the SDK's
    ``ResultMessage``: a ``.usage`` object with the four token counts and a ``.total_cost_usd``.

    One consumer, two producers -- the alternative was a second usage-extraction branch in
    ``generate()`` reading the same four fields out of a dict.
    """

    __slots__ = ("duration_ms", "num_turns", "session_id", "total_cost_usd", "usage")

    def __init__(self, event: dict) -> None:
        raw = event.get("usage") or {}
        self.usage = SimpleNamespace(
            input_tokens=int(raw.get("input_tokens", 0) or 0),
            output_tokens=int(raw.get("output_tokens", 0) or 0),
            cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
        )
        self.total_cost_usd = float(event.get("total_cost_usd", 0.0) or 0.0)
        self.session_id = event.get("session_id")
        self.num_turns = event.get("num_turns")
        self.duration_ms = event.get("duration_ms")



def run_cli(cmd: "list[str]", prompt: str, sub_env: dict, child_cwd: str, timeout: float, proc_holder: "list", cancel_evt: "threading.Event") -> "tuple[int, str, str, dict | None]":
    """Spawn the ``claude`` CLI as a subprocess, stream its stdout via a background reader thread, and parse the stream-json events until a ``result`` event, timeout, or EOF; returns ``(returncode, result_text, stderr_text, result_event)``."""
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        shell=False,  # nosec B603 - cmd is a fixed argv list: the executable is resolved via shutil.which / known install paths (never attacker-controlled), remaining args are literal flags plus the model name and a file path (prompt content goes through stdin, not argv, and nothing is shell-interpreted)
        env=sub_env,
        cwd=child_cwd,
        # Without errors=, one stray byte from a CLI writing in the console codepage
        # raises UnicodeDecodeError inside the daemon reader thread, where nothing
        # catches it; the call then fails as "produced no result", naming neither the
        # decode nor the byte.
        errors="replace",
    ) as proc:
        proc_holder.append(proc)
        return _stream_one_call(proc, prompt, timeout, cancel_evt)


def _stream_one_call(
    proc: "subprocess.Popen",
    prompt: str,
    timeout: float,
    cancel_evt: "threading.Event",
) -> "tuple[int, str, str, dict | None]":
    """Write the prompt, drain both pipes in parallel, and turn the event stream into a result.

    Split from ``run_cli`` so the spawn can be a ``with`` block whose body is one call: the
    stream handling is long enough that leaving it under the context manager would put the
    function straight back over the C901 ceiling the carve just brought it under.
    """
    try:
        assert proc.stdin is not None  # guaranteed by stdin=subprocess.PIPE above
        proc.stdin.write(prompt)
        proc.stdin.close()
    except BrokenPipeError:
        pass

    line_q: queue.Queue[str | None] = queue.Queue()

    def _reader():
        """Drain ``proc.stdout`` line-by-line into ``line_q``, pushing ``None`` as a sentinel once the pipe closes."""
        assert proc.stdout is not None  # guaranteed by stdout=subprocess.PIPE above
        try:
            for line in proc.stdout:
                line_q.put(line)
        finally:
            line_q.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # stderr used to be read only after the run, with a single blocking read(500) and
    # no timeout. Nothing consumed it meanwhile, so a CLI that wrote more than one
    # pipe buffer's worth of warnings blocked on its own stderr write and hung until
    # the timeout -- which then looked transient and was retried. Draining it in
    # parallel also means the message is intact when a failure needs explaining.
    stderr_chunks: "list[str]" = []

    def _stderr_reader():
        """Drain ``proc.stderr`` so the child never blocks writing to it."""
        try:
            if proc.stderr is not None:
                for line in proc.stderr:
                    stderr_chunks.append(line)
        except (ValueError, OSError):  # pragma: no cover -- pipe closed under us
            pass

    stderr_thread = threading.Thread(target=_stderr_reader, daemon=True)
    stderr_thread.start()

    # The _reader thread is the SOLE consumer of stdout. Do NOT
    # call proc.communicate() here -- it would race the reader on
    # the same pipe (ValueError: I/O operation on closed file, or
    # partial reads). For reaping we kill + wait; stderr is read
    # directly (the reader never touches it).
    try:
        result_text, error_text, timed_out, result_event = _consume_cli_stream(line_q, timeout, cancel_evt)
    finally:
        try:
            proc.kill()
        except OSError:
            pass
        # Reap the process WITHOUT reading the pipes (the reader
        # owns stdout). wait() avoids the communicate() double-drain.
        try:
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
        # Join the reader so stdout is fully drained/closed before
        # we read stderr below -- prevents a half-open-pipe race.
        reader_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    stderr_data = "".join(stderr_chunks)[-2000:]

    if cancel_evt.is_set():
        raise asyncio.CancelledError

    if timed_out:
        raise subprocess.TimeoutExpired(proc.args, timeout)

    if error_text is not None:
        raise RuntimeError(f"Claude CLI error: {error_text}")
    if result_text is None:
        raise RuntimeError(f"Claude CLI produced no result. stderr: {stderr_data}")

    # A parsed ``result`` success event is the authoritative
    # success signal for the stream-json protocol, so we report
    # returncode 0 here regardless of the OS exit code (the CLI
    # may exit non-zero on cleanup even after a good result). All
    # genuine failures already raised above; the downstream
    # ``if returncode != 0`` guard is intentionally kept as
    # belt-and-braces for any future non-success return path.
    return 0, result_text, stderr_data, result_event
