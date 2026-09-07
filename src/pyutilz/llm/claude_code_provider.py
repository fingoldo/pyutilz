"""Claude Code provider - uses your Max subscription through Claude Code.

Supports two backends:
1. SDK (preferred) -- uses claude-code-sdk Python package, works from nested sessions
2. CLI fallback -- spawns `claude` CLI subprocess (breaks inside VSCode extension)
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import errno
import json
import logging
import os
import queue
import re
import shutil
import subprocess  # nosec B404 - only used to spawn the trusted `claude` CLI (resolved via shutil.which / fixed install paths, never a user-supplied path), always with shell=False
import sys
import tempfile
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Any, AsyncIterator

from pyutilz.llm._messages import images_on_disk
from pyutilz.llm.base import LLMProvider, PerCallAttr
from pyutilz.llm._retry import MAX_RETRY_ATTEMPTS

# Defined in the domain's exceptions module (so `except LLMProviderError` catches it) and re-exported
# here, where it is raised, so existing `from ...claude_code_provider import ClaudeCodeToolUseError`
# imports keep resolving.
from pyutilz.llm.exceptions import ClaudeCodeToolUseError  # also re-exported for back-compat; no noqa needed, the name is raised below

logger = logging.getLogger(__name__)

# Carved out 2026-09-07 (see claude_code_cli's docstring); re-exported here so existing importers
# of these names keep working.
from pyutilz.llm.claude_code_cli import (  # noqa: E402 -- re-export, placed after the logger it shares
    MAX_TIMEOUT_RETRIES,
    _CliResultMessage,
    _consume_cli_stream,
    _find_claude_executable,
    _is_transient_subprocess_error,
    _raise_on_cli_tool_use,
    run_cli,
)

# ---------------------------------------------------------------------------
# SDK monkey-patches (claude-code-sdk 0.0.25)
# ---------------------------------------------------------------------------
try:
    from claude_code_sdk import query as cc_query, ClaudeCodeOptions, ResultMessage, TextBlock
    import claude_code_sdk as _ccsdk
    _CCSDK_VERSION = getattr(_ccsdk, "__version__", "unknown")
    _SUPPORTED_SDK_VERSIONS = frozenset({
        "0.0.25", "0.0.26", "0.0.27", "0.0.28", "0.0.29", "0.0.30",
    })
    _PATCHES_APPLIED = False

    if _CCSDK_VERSION not in _SUPPORTED_SDK_VERSIONS:
        logger.error(
            "claude-code-sdk version %s is NOT in supported set %s. "
            "Monkey-patches SKIPPED -- using unpatched SDK. "
            "Review and update patches in claude_code_provider.py for this version.",
            _CCSDK_VERSION, sorted(_SUPPORTED_SDK_VERSIONS),
        )
    else:
        # -- PATCH 1: Unknown message types (rate_limit_event) --
        from claude_code_sdk._internal import message_parser as _mp
        from claude_code_sdk._internal import client as _cc_client
        _orig_parse = _mp.parse_message
        def _patched_parse(data):
            """Wrap ``message_parser.parse_message`` so unknown SDK message types (e.g. ``rate_limit_event``) return None instead of raising ``MessageParseError``."""
            try:
                return _orig_parse(data)
            except _mp.MessageParseError:
                return None
        _mp.parse_message = _patched_parse
        _cc_client.parse_message = _patched_parse

        # -- PATCH 2: Windows command-line length limit --
        from claude_code_sdk._internal.transport import subprocess_cli as _sc
        import anyio
        from anyio.streams.text import TextReceiveStream as _TRS
        _orig_build = _sc.SubprocessCLITransport._build_command

        def _patched_build(self):
            """Wrap ``SubprocessCLITransport._build_command`` to pass the prompt via ``--print -`` (stdin) instead of an inline argv element, avoiding Windows' command-line length limit for non-streaming calls."""
            cmd = _orig_build(self)
            if not self._is_streaming and "--print" in cmd:
                try:
                    idx = cmd.index("--print")
                    cmd = [*cmd[:idx], "--print", "-"]
                except ValueError:
                    pass
            return cmd
        _sc.SubprocessCLITransport._build_command = _patched_build  # type: ignore[method-assign]

        # -- PATCH 3: Stdin prompt delivery + stderr capture --
        async def _patched_connect(self):
            """Wrap ``SubprocessCLITransport.connect`` to spawn the CLI subprocess, write the prompt to stdin (rather than argv) for non-streaming calls, and capture stderr into a readable stream for error enrichment."""
            if self._process:
                return
            cmd = self._build_command()
            try:
                process_env = {
                    **os.environ,
                    **self._options.env,
                    "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
                }
                if self._cwd:
                    process_env["PWD"] = self._cwd
                from subprocess import PIPE as _PIPE  # nosec B404 - re-import of the same trusted-subprocess module for anyio.open_process's stdin/stdout/stderr pipe constants
                self._process = await anyio.open_process(
                    cmd, stdin=_PIPE, stdout=_PIPE, stderr=_PIPE,
                    cwd=self._cwd, env=process_env, user=self._options.user,
                )
                if self._process.stdout:
                    self._stdout_stream = _TRS(self._process.stdout)
                if self._process.stderr:
                    self._stderr_stream = _TRS(self._process.stderr)
                else:
                    self._stderr_stream = None
                if self._is_streaming and self._process.stdin:
                    from anyio.streams.text import TextSendStream as _TSS
                    self._stdin_stream = _TSS(self._process.stdin)
                elif not self._is_streaming and self._process.stdin:
                    prompt_text = str(self._prompt)
                    await self._process.stdin.send(prompt_text.encode("utf-8"))
                    await self._process.stdin.aclose()
                self._ready = True
            except FileNotFoundError as e:
                raise _sc.CLINotFoundError(f"Claude Code not found at: {cmd[0]}") from e

        _sc.SubprocessCLITransport.connect = _patched_connect  # type: ignore[method-assign]

        # -- PATCH 4: Include real stderr in ProcessError --
        _orig_read_messages = _sc.SubprocessCLITransport._read_messages_impl
        from claude_code_sdk._errors import ProcessError as _ProcessError

        async def _patched_read_messages(self):
            """Wrap ``SubprocessCLITransport._read_messages_impl`` to re-raise ``ProcessError`` with the process's real captured stderr text (up to 2000 chars) instead of the SDK's generic message."""
            try:
                async for msg in _orig_read_messages(self):
                    yield msg
            except _ProcessError as e:
                real_stderr = ""
                stderr_stream = getattr(self, "_stderr_stream", None)
                if stderr_stream:
                    try:
                        chunks = []
                        async for chunk in stderr_stream:
                            chunks.append(chunk)
                            if len(chunks) > 100:
                                break
                        real_stderr = "".join(chunks).strip()
                    except Exception as stderr_read_error:  # nosec B110 - best-effort stderr enrichment for a ProcessError we're already re-raising; failure here must not mask the original error
                        logger.debug("Could not read real stderr for ProcessError enrichment: %s", stderr_read_error)
                        pass
                if real_stderr:
                    raise _ProcessError(
                        f"Command failed with exit code {e.exit_code}",
                        exit_code=e.exit_code,
                        stderr=real_stderr[:2000],
                    ) from None
                raise
        _sc.SubprocessCLITransport._read_messages_impl = _patched_read_messages  # type: ignore[method-assign]

        _PATCHES_APPLIED = True

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False
    _PATCHES_APPLIED = False


_RATE_LIMIT_PATTERN = re.compile(
    r"(?:hit your limit|rate.?limit|quota exceeded|usage limit)",
    re.IGNORECASE,
)
_RESET_TIME_PATTERN = re.compile(
    r"resets?\s+(?:(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?P<day>\d{1,2}),?\s+)?"
    r"(?P<hour>\d{1,2})\s*(?::(?P<min>\d{2}))?\s*(?P<ampm>am|pm)?\b",
    re.IGNORECASE,
)
_TIMEZONE_PATTERN = re.compile(
    r"\(([A-Za-z]+/[A-Za-z_]+)\)",
)


def _parse_reset_wait_seconds(error_text: str) -> int | None:
    """Parse reset time from rate limit error and return seconds to wait."""
    m = _RESET_TIME_PATTERN.search(error_text)
    if not m:
        return None

    hour = int(m.group("hour"))
    minute = int(m.group("min")) if m.group("min") else 0
    ampm = m.group("ampm")

    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

    tz_match = _TIMEZONE_PATTERN.search(error_text)
    tz = None
    if tz_match:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(tz_match.group(1))
        except (ImportError, KeyError):
            logger.debug("Could not load timezone %s, using local time", tz_match.group(1))

    now = datetime.now(tz) if tz else datetime.now()  # noqa: DTZ005 -- error text has no explicit timezone; falls back to local wall-clock, consistent with reset_time being derived from the same `now` a few lines below

    month_str = m.group("month")
    day_str = m.group("day")
    if month_str and day_str:
        _MONTHS = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        month = _MONTHS[month_str.lower()]
        day = int(day_str)
        try:
            reset_time = now.replace(
                month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0,
            )
            if reset_time <= now:
                reset_time = reset_time.replace(year=now.year + 1)
        except ValueError:
            # An impossible date ("resets Feb 30", or Feb 29 rolled into a non-leap year) must not
            # turn a recoverable rate-limit pause into a hard failure of generate(); fall through
            # to the caller's default wait instead.
            logger.warning("Could not interpret rate-limit reset date %r-%r; using the default wait", month_str, day_str)
            return None
    else:
        reset_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if reset_time <= now:
            reset_time += timedelta(days=1)

    wait = (reset_time - now).total_seconds() + 60
    return int(wait)


def _is_rate_limit_error(error: BaseException) -> bool:
    """Check if an exception is a rate limit error."""
    msg = str(error)
    if _RATE_LIMIT_PATTERN.search(msg):
        return True
    stderr = getattr(error, "stderr", "")
    if stderr and _RATE_LIMIT_PATTERN.search(str(stderr)):
        return True
    return False


# One steer, used by both JSON entry points. They had different wordings, which split a prefix
# that is otherwise byte-identical across calls and so cache-shared between them.
_JSON_STEER = "\n\nRespond with valid JSON only. No markdown, no explanation. Start with { and end with }."


class ClaudeCodeProvider(LLMProvider):
    """LLM provider that uses Claude Code for LLM calls (your Max subscription).

    Two backends: SDK (preferred, via `claude-code-sdk`) or CLI fallback.
    """

    # Class-level default so a bare instance (tests build one with __new__ and set the fields
    # they care about) has a sane value: an int rebound with += becomes an instance attribute on
    # first write, so there is no shared-state hazard.
    _successful_call_count = 0

    INPUT_COST_PER_1M = 0.0
    OUTPUT_COST_PER_1M = 0.0

    # Per-call "last successful call" state -- contextvar-backed via PerCallAttr, NOT plain
    # instance attributes (2026-09-03 audit F08). generate_batch() fires N concurrent
    # self.generate() calls on ONE shared, factory-cached provider; as plain attributes the
    # ResultMessage and its cost/cache figures leaked between in-flight calls, so a call could
    # bill another call's cost twice or fall back to tiktoken estimates for its own.
    _last_usage: PerCallAttr = PerCallAttr(lambda: {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0})
    _last_result_message: PerCallAttr = PerCallAttr(lambda: None)
    last_cost_usd: PerCallAttr = PerCallAttr(lambda: 0.0)
    last_cache_creation_input_tokens: PerCallAttr = PerCallAttr(lambda: 0)
    last_cache_read_input_tokens: PerCallAttr = PerCallAttr(lambda: 0)
    last_session_id: PerCallAttr = PerCallAttr(lambda: None)
    last_num_turns: PerCallAttr = PerCallAttr(lambda: None)

    _PERCALL_METADATA_ATTRS: tuple[str, ...] = (
        *LLMProvider._PERCALL_METADATA_ATTRS,
        "_last_result_message",
        "last_cost_usd",
        "last_session_id",
        "last_num_turns",
    )

    def __init__(
        self,
        model: str = "opus",
        max_concurrent: int = 1,
        timeout: int = 2400,
    ):
        self.model = model
        self.model_name = model
        self.timeout = timeout
        self._max_concurrent = max_concurrent
        # Per-call usage/cost/cache/session state: PerCallAttr class-level descriptors (declared
        # above __init__) provide the defaults; nothing to initialize here.
        self._call_count = 0
        # Attempts and answers are different numbers: a call that raised still counted as a
        # call, so the cost summary's per-call figures were divided by a denominator that
        # included every failed attempt and every retry.
        self._successful_call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Real per-call accounting from ResultMessage when SDK path runs.
        # Max-subscription users don't get billed per call, but the SDK
        # reports the underlying API cost as if they were -- useful for
        # tracking session burn even on subscription accounts.
        self.total_cost_usd = 0.0
        self.total_cache_creation_input_tokens = 0
        self.total_cache_read_input_tokens = 0

    def get_session_cost(self) -> dict:
        """Return cost data, sourced from ResultMessage.usage when SDK path
        ran, falling back to tiktoken estimates for the CLI path.

        ``total_cost_usd`` is what Claude's SDK reports for this session,
        even on Max subscription -- subscription users still pay nothing
        out-of-pocket, but the underlying API cost is informative for
        understanding session burn against weekly limits.
        """
        return {
            "calls": self._call_count,
            "successful_calls": self._successful_call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "cache_hit_tokens": self.total_cache_read_input_tokens,
            "cache_miss_tokens": max(0, self.total_prompt_tokens - self.total_cache_read_input_tokens),
            "cache_creation_input_tokens": self.total_cache_creation_input_tokens,
            "cache_read_input_tokens": self.total_cache_read_input_tokens,
            "completion_tokens": self.total_completion_tokens,
            "reasoning_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": self.total_cost_usd,
        }

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens supported by the underlying Claude Code model."""
        return 32000

    @property
    def context_window(self) -> int:
        """Total context window size (input + output tokens) of the underlying Claude Code model."""
        return 200_000

    def supports_json_mode(self) -> bool:
        """Claude Code SDK has no hard JSON-mode toggle — the
        ``json_mode=True`` kwarg here only appends a system-prompt hint
        ("Respond with valid JSON only") and post-strips a Markdown
        ``\\`\\`\\`json`` fence. That's a SOFT guarantee: helpful but the
        model can still emit prose around the JSON. Callers wanting a
        hard guarantee should branch on this and route to a provider
        that returns True instead. Returning False here lets glossum's
        retry/parse layer apply more defensive handling for this
        provider."""
        return False

    _seen_unsupported_params: set[str] = set()  # noqa: RUF012 -- intentional shared class-level dedupe set (warn once per parameter name, across all instances), not a per-instance mutable-default bug

    def _warn_unsupported_param_once(self, param: str) -> None:
        """Log a one-time warning that ``param`` cannot be honoured by the Claude Code backend."""
        if param in ClaudeCodeProvider._seen_unsupported_params:
            return
        ClaudeCodeProvider._seen_unsupported_params.add(param)
        logger.warning(
            "ClaudeCodeProvider cannot forward %r to the Claude Code SDK; the value you passed is ignored. "
            "Route to another provider if this parameter matters for correctness.",
            param,
        )

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
        json_mode: bool = False,
        images: list[str] | None = None,
    ) -> str:
        """Generate text using Claude Code SDK (preferred) or CLI fallback.

        Args:
            prompt: The user message to send.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Output-token ceiling; 0 derives one.
            json_mode: Append a "respond with JSON only" steer to the prompt.
            images: ``data:`` URIs, written to temporary files whose paths are appended to the
                prompt -- see `pyutilz.llm._messages.images_on_disk`. The CLI has no image
                argument, but the agent behind it can open files.
        """
        if images:
            # Recursed rather than inlined so the temporary files stay on disk for the WHOLE call:
            # the `with` block must still be open when the CLI reads them, and closing it around a
            # prompt-building line would delete every picture before the subprocess started.
            with images_on_disk(images) as (image_prompt, _paths):
                return await self.generate(prompt + image_prompt, system, temperature, max_tokens, json_mode)
        if json_mode:
            json_hint = _JSON_STEER
            system = (system or "") + json_hint

        self._call_count += 1
        # The per-call reset lives INSIDE the semaphore, at the top of _generate_sdk/_generate_cli
        # (2026-09-03 audit F08): generate() itself holds no lock, so with max_concurrent=1 a
        # queued call used to clear state while the call ahead of it was still inside the
        # semaphore -- either erasing that call's ResultMessage (silent fallback to tiktoken
        # estimates, zero cache tokens, nothing added to total_cost_usd) or letting it read and
        # bill the other call's cost twice.
        # NEITHER transport forwards budget or sampling temperature: the SDK path passes no such
        # field to ClaudeCodeOptions, and the CLI has no --temperature/--max-tokens flag to pass one
        # to. This is the factory's default provider, so say so once on both paths instead of
        # silently dropping a caller's explicit determinism or length requirement -- the warning
        # used to sit inside an ``if _HAS_SDK:`` block, leaving the CLI fallback silent.
        if max_tokens > 0:
            self._warn_unsupported_param_once("max_tokens")
        if abs(temperature - 0.7) > 1e-9:
            self._warn_unsupported_param_once("temperature")
        attempt = 0
        # Regression fix (2026-07-21 audit): read the same PYUTILZ_LLM_MAX_RETRIES-configurable
        # bound every other provider uses (via _retry.py), instead of a hardcoded 20 that
        # silently diverged from the env-var-documented behavior -- since get_llm_provider()'s
        # default provider_name is "claude-code", this is the retry behavior most callers get.
        # 0 means infinite (mirrors _retry.MAX_RETRY_ATTEMPTS semantics).
        max_attempts = MAX_RETRY_ATTEMPTS
        timeout_attempts = 0
        while True:
            attempt += 1
            if max_attempts != 0 and attempt > max_attempts:
                raise RuntimeError(f"ClaudeCodeProvider: exceeded {max_attempts} retry attempts")
            try:
                if _HAS_SDK:
                    result = await self._generate_sdk(prompt, system)
                else:
                    result = await self._generate_cli(prompt, system)
                if json_mode and result:
                    stripped = result.strip()
                    fence_match = re.search(
                        r'```(?:json)?\s*([\s\S]*?)\s*```', stripped,
                    )
                    if fence_match:
                        result = fence_match.group(1).strip()
                # Prefer the SDK's ResultMessage usage / cost over a
                # tiktoken estimate (wrong tokenizer for Claude). Falls
                # back to tiktoken on the CLI path or if the SDK didn't
                # surface a ResultMessage with usage.
                rm = getattr(self, "_last_result_message", None)
                rm_usage = getattr(rm, "usage", None) if rm is not None else None
                if rm_usage is not None:
                    in_tok = int(getattr(rm_usage, "input_tokens", 0) or 0)
                    out_tok = int(getattr(rm_usage, "output_tokens", 0) or 0)
                    cache_create = int(getattr(rm_usage, "cache_creation_input_tokens", 0) or 0)
                    cache_read = int(getattr(rm_usage, "cache_read_input_tokens", 0) or 0)
                    cost = float(getattr(rm, "total_cost_usd", 0.0) or 0.0)
                    sid = getattr(rm, "session_id", None)
                    nturns = getattr(rm, "num_turns", None)
                    self.last_cache_creation_input_tokens = cache_create
                    self.last_cache_read_input_tokens = cache_read
                    self.total_cache_creation_input_tokens += cache_create
                    self.total_cache_read_input_tokens += cache_read
                    self.last_cost_usd = cost
                    self.total_cost_usd += cost
                    if isinstance(sid, str):
                        self.last_session_id = sid
                    if isinstance(nturns, int):
                        self.last_num_turns = nturns
                else:
                    from pyutilz.llm.token_counter import count_tokens as _count_tok
                    in_tok = _count_tok(prompt) + _count_tok(system or "")
                    out_tok = _count_tok(result)
                    self.last_cache_creation_input_tokens = 0
                    self.last_cache_read_input_tokens = 0
                self._last_usage = {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "reasoning_tokens": 0,
                    "cache_creation_input_tokens": self.last_cache_creation_input_tokens,
                    "cache_read_input_tokens": self.last_cache_read_input_tokens,
                }
                self.total_prompt_tokens += in_tok
                self.total_completion_tokens += out_tok
                self._successful_call_count += 1
                return result
            except (OSError, subprocess.TimeoutExpired) as e:
                # ConnectionError and TimeoutError are OSError subclasses; naming them here too
                # only made the tuple look narrower than it is. _is_transient_subprocess_error
                # is what actually narrows it.
                if not _is_transient_subprocess_error(e):
                    logger.error(
                        "Claude Code call failed permanently (%s: %s); not retrying.",
                        type(e).__name__, str(e)[:200],
                    )
                    raise
                if isinstance(e, subprocess.TimeoutExpired):
                    timeout_attempts += 1
                    if timeout_attempts >= MAX_TIMEOUT_RETRIES:
                        logger.error(
                            "Claude CLI timed out %d times at %ss; giving up rather than spending another %s seconds.",
                            timeout_attempts, self.timeout, self.timeout * (max_attempts or 0),
                        )
                        raise
                # subprocess.TimeoutExpired (raised by run_cli() on a CLI hang past self.timeout)
                # is a SubprocessError subclass, NOT a TimeoutError/OSError/ConnectionError
                # subclass -- it previously fell through to the generic `except Exception` below,
                # never matched _is_rate_limit_error, and re-raised immediately with zero retries
                # despite being exactly the transient-failure shape this except clause exists for.
                # Jitter added (matching _retry.py's RETRY_WAIT) so several ClaudeCodeProvider
                # workers hitting a transient error at the same moment (a shared network blip)
                # don't all retry in lockstep -- a classic thundering-herd pattern.
                wait = min(5 * (2 ** (attempt - 1)), 300) + random.uniform(0, 5)  # nosec B311 - retry-jitter timing only, not security/cryptographic use
                logger.warning(
                    "LLM call attempt %d failed (%s: %s), retrying in %.0fs...",
                    attempt, type(e).__name__, str(e)[:200], wait,
                )
                await asyncio.sleep(wait)
            except Exception as e:
                if not _is_rate_limit_error(e):
                    raise
                error_msg = str(e)
                stderr = getattr(e, "stderr", "")
                full_msg = f"{error_msg} {stderr}"
                wait_seconds = _parse_reset_wait_seconds(full_msg)
                if wait_seconds is None:
                    wait_seconds = 3600
                    logger.warning(
                        "[RateLimit] Hit rate limit but couldn't parse reset time: %s. " "Waiting %d seconds (default).",
                        error_msg[:200],
                        wait_seconds,
                    )
                else:
                    reset_dt = datetime.now() + timedelta(seconds=wait_seconds)  # noqa: DTZ005 -- intentional local time for a human-facing log message (logged as %H:%M:%S wall-clock, not compared/stored)
                    logger.warning(
                        "[RateLimit] Hit rate limit. Pausing until %s (%d seconds). "
                        "Will retry automatically. Error: %s",
                        reset_dt.strftime("%H:%M:%S"), wait_seconds, error_msg[:200],
                    )
                await asyncio.sleep(wait_seconds)
                logger.info("[RateLimit] Resuming after rate limit pause (attempt %d).", attempt + 1)

    _NESTED_BLOCK_VARS = frozenset({
        'CLAUDECODE',
        'CLAUDE_CODE_ENTRYPOINT',
        'CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING',
        'CLAUDE_AGENT_SDK_VERSION',
        'ANTHROPIC_API_KEY',
    })

    async def _generate_sdk(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Generate text using claude-code-sdk."""
        async with self.semaphore:
            self._reset_per_call_state()
            override_env = {k: "" for k in self._NESTED_BLOCK_VARS if k in os.environ}

            # The system prompt stays where it belongs. 2026-09-07: this used to fold it into
            # the user message above 6000 combined characters -- so on every real call, the
            # system prompt here being tens of kilobytes -- and set system to "". That throws
            # away the prompt cache: the system prompt is the prefix that gets cached, and two
            # identical CLI invocations measured 14813 cache-creation tokens then 14813
            # cache-read, $0.14824 then $0.00752. The threshold reads like a workaround for the
            # 32767-character command-line limit the CLI path hit as WinError 206, but the SDK
            # builds no command line: it passes the system prompt through its options object.

            opts = ClaudeCodeOptions(
                model=self.model,
                system_prompt=system or "",
                permission_mode="bypassPermissions",
                max_turns=1,
                env=override_env,
                # "strict-mcp-config" (a valueless CLI flag: the SDK emits `--strict-mcp-config` for a
                # None value) makes the session use ONLY MCP servers passed via --mcp-config -- none are
                # passed here, so no MCP server at all is loaded. Without it, every MCP server configured
                # in the INVOKING USER's global/project config is loaded into this session, and since
                # permissions are bypassed its tools would be auto-approved. `--tools ""` does not cover
                # them: it scopes only the built-in tool set. This provider is text-generation-only, so
                # no ambient MCP configuration is ever wanted.
                extra_args={"tools": "", "strict-mcp-config": None},
            )

            _tools_val = opts.extra_args.get("tools") if opts.extra_args else None
            if _tools_val != "":
                logger.warning(
                    'ClaudeCodeProvider: extra_args["tools"] was %r -- forcing to "" ' "to disable built-in tools and preserve JSON compliance.",
                    _tools_val,
                )
                if opts.extra_args is None:
                    # The SDK types extra_args as non-Optional, but it is a third-party object we
                    # do not control, so the defensive guard stays.
                    opts.extra_args = {}  # type: ignore[unreachable]
                opts.extra_args["tools"] = ""

            logger.info("SDK call: model=%s prompt=%d system=%d", self.model, len(prompt), len(system or ""))

            result_text = None
            msg_count = 0
            msg_types = []
            async for msg in cc_query(prompt=prompt, options=opts):
                msg_count += 1
                if msg is None:
                    # The SDK's async iterator is typed as never yielding None; the guard is
                    # defensive because a None here would crash the isinstance chain below.
                    msg_types.append("None")  # type: ignore[unreachable]
                    continue
                msg_types.append(type(msg).__name__)
                msg_type = type(msg).__name__

                if isinstance(msg, ResultMessage):
                    if isinstance(msg.result, str) and msg.result:
                        result_text = msg.result
                    # Capture the real usage / cost / session metadata
                    # from the SDK rather than inferring with tiktoken
                    # against a wrong tokenizer.
                    self._last_result_message = msg
                elif msg_type == "AssistantMessage":
                    if hasattr(msg, "content") and msg.content:
                        parts = []
                        for block in msg.content:
                            bt = type(block).__name__
                            if isinstance(block, TextBlock):
                                parts.append(block.text)
                            elif bt == "ToolUseBlock":
                                tool_name = getattr(block, "name", "?")
                                # A tool-use block must be impossible here: built-in tools are off
                                # (`--tools ""`) and no MCP server is loaded (`--strict-mcp-config`).
                                # If one appears anyway, the sandbox this provider relies on is not
                                # holding -- most plausibly prompt injection in caller-supplied text
                                # reaching a tool that permission bypass then auto-approves. Fail hard
                                # instead of logging "(blocked)" while the turn continues.
                                raise ClaudeCodeToolUseError(f"Claude Code returned a tool-use block ({tool_name!r}); this provider is text-generation-only")
                            elif hasattr(block, "text") and isinstance(getattr(block, "text"), str):
                                if bt != "ThinkingBlock":
                                    parts.append(block.text)
                        if parts:
                            text = "\n".join(parts)
                            if result_text is None:
                                result_text = text
                            else:
                                result_text += text
            if result_text is None:
                logger.error("SDK produced no result after %d messages: %s", msg_count, msg_types)
                raise RuntimeError("Claude Code SDK produced no result")
            stripped = result_text.strip()
            if not stripped:
                logger.warning("SDK returned empty response: model=%s prompt_len=%d msgs=%d types=%s", self.model, len(prompt), msg_count, msg_types)
            return stripped

    async def _generate_cli(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Generate text using Claude Code CLI (fallback).

        Takes no temperature/max_tokens: the CLI exposes no flag for either, so accepting them here
        only made the drop look deliberate. ``generate()`` warns about them once instead.
        """
        async with self.semaphore:
            self._reset_per_call_state()
            if not hasattr(self, "_claude_path"):
                self._claude_path = _find_claude_executable()

            cmd = [
                self._claude_path,
                '--print',
                '--verbose',
                '--include-partial-messages',
                '--model', self.model,
                '--output-format', 'stream-json',
                # 2026-09-07: was --dangerously-skip-permissions, which auto-approves whatever
                # reaches a permission check -- on a call whose entire input is untrusted text and
                # which wants no tools at all. --restricted is its opposite: the built-in tools
                # that run commands or code and WebFetch are removed unless --tools names them
                # (it names nothing), file tools are confined to the working directories, and
                # bypassPermissions is refused outright. --permission-prompts none denies any
                # prompt immediately rather than waiting on an answer a --print subprocess has no
                # way to give. Probed against CLI 2.1.263 with this exact flag set: is_error
                # false, num_turns 1, answer returned.
                '--restricted',
                '--permission-prompts', 'none',
                '--no-session-persistence',
                '--tools', '',
                # Use ONLY MCP servers given via --mcp-config (none are), so the invoking user's
                # global/project MCP servers are not loaded. `--tools ""` scopes the BUILT-IN tool
                # set only, and with --dangerously-skip-permissions any loaded MCP tool would be
                # auto-approved -- reachable from untrusted text passed in as the prompt.
                "--strict-mcp-config",
            ]

            # The system prompt goes in a FILE, not in argv.
            #
            # 2026-09-06: Windows caps a whole command line at 32767 characters
            # (CreateProcess), and a real system prompt blows past that on its own -- the
            # measured case was 48378 characters, which failed with WinError 206 "the
            # filename or extension is too long" on every attempt, was retried thirteen
            # times as though it were transient, and burned a 40-minute budget before the
            # caller gave up. There is no size threshold to pick here and no need for one:
            # the CLI takes --system-prompt-file, so the file path is what argv carries
            # whatever the prompt's length. The prompt itself already goes through stdin.
            system_prompt_file: str | None = None
            if system:
                # delete=False, because the subprocess has to be able to open it after this
                # block closes the handle; the finally below removes it.
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", prefix="claude-system-", delete=False) as handle:
                    # Bound before the write, not after: the write can fail (a full disk, an
                    # encoding error) and the cleanup in the finally is keyed on this variable,
                    # so binding it afterwards left the file on disk on exactly the path that
                    # created it.
                    system_prompt_file = handle.name
                    handle.write(system)
                cmd.extend(["--system-prompt-file", system_prompt_file])

            cmd.append("-")

            logger.debug("Running Claude CLI: %s...", cmd[0])

            sub_env = {k: v for k, v in os.environ.items() if k not in self._NESTED_BLOCK_VARS}

            # The child inherited the caller's working directory, which for the pipeline that uses
            # this provider is a source repository: --restricted confines file tools to the working
            # directories, so what those are matters. An empty temporary directory is the smallest
            # thing to confine them to.
            child_cwd = tempfile.mkdtemp(prefix="claude-cwd-")

            # Cancelling the future returned by run_in_executor does NOT stop the thread running
            # in it, and nothing else in this coroutine held a handle on the child: a cancelled
            # call (Ctrl-C, a task group tearing down, an outer timeout) left a `claude` process
            # running for up to self.timeout seconds, released the semaphore so the concurrency
            # cap was quietly exceeded, and unlinked the system-prompt file out from under a live
            # reader. The holder gives the coroutine the handle; the event gets the consumer out
            # of its own wait rather than leaving it to notice the killed pipe.
            proc_holder: "list[subprocess.Popen]" = []
            cancel_evt = threading.Event()

            # get_running_loop, not get_event_loop: the deprecated form emits a
            # DeprecationWarning from inside a coroutine on 3.12+, and there is always a running
            # loop here by construction.
            loop = asyncio.get_running_loop()
            try:
                returncode, stdout, stderr, result_event = await loop.run_in_executor(
                    None, run_cli, cmd, prompt, sub_env, child_cwd, self.timeout, proc_holder, cancel_evt
                )
                # Set here rather than inside ``run_cli``: that runs in an executor thread with its
                # own (empty) context, so a PerCallAttr write there would be invisible to the caller.
                if result_event is not None:
                    self._last_result_message = _CliResultMessage(result_event)
            except asyncio.CancelledError:
                # Reach into the thread's process and end it. The executor thread itself cannot be
                # cancelled, but with the child dead its stdout closes, the reader hits EOF and
                # the consumer returns within its one-second poll.
                cancel_evt.set()
                for child in proc_holder:
                    try:
                        child.kill()
                    except OSError:  # pragma: no cover -- already reaped
                        pass
                raise
            finally:
                if system_prompt_file:
                    try:
                        os.unlink(system_prompt_file)
                    except OSError:  # pragma: no cover -- best-effort cleanup
                        logger.debug("Could not remove the temporary system-prompt file %s", system_prompt_file)
                shutil.rmtree(child_cwd, ignore_errors=True)

            if returncode != 0:
                error_msg = stderr or stdout or "Unknown error"
                raise RuntimeError(f"Claude CLI failed (code {returncode}): {error_msg}")

            return stdout.strip()  # type: ignore[no-any-return]  # stdout comes from an asyncio subprocess pipe read decoded through an untyped helper

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
        images: list[str] | None = None,
        thinking: bool | str | int | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output.

        Args:
            prompt: The user message to send.
            system: Optional system prompt; the JSON steer is appended to it.
            temperature: Sampling temperature.
            max_tokens: Output-token ceiling; 0 derives one.
            images: refused, not forwarded -- this provider has no vision path.
            thinking: accepted for Liskov and not honoured; logged.
        """
        # Accepted for Liskov (the base class offers it) and NOT honoured: the Claude Code CLI exposes no reasoning-effort control,
        # so an effort request is logged rather than dropped in silence -- a caller who asked for a
        # harder think and got the ordinary one has somewhere to look.
        if thinking is not None:
            logger.info("%s ignores thinking=%r: no reasoning-effort control on this provider", type(self).__name__, thinking)

        # Declared for Liskov (the base class offers it) and REFUSED rather than ignored:
        # the Claude Code CLI takes a prompt string and no image parts, so accepting the argument and dropping it would answer a question the caller asked
        # about a picture the model never saw -- wrong, and with nothing in the output to show it.
        if images:
            raise NotImplementedError(
                f"{type(self).__name__} has no vision path; pass images to an OpenAI-compatible " "provider (e.g. OpenRouter) or send the document as text."
            )
        # The SAME steer text as ``generate(json_mode=True)`` appends, deliberately: two
        # different wordings meant two different system prompts for the same task, and the
        # prefix cache keys on the exact bytes -- so a run that used both paths paid the cache
        # write twice and read neither. Change one, change the other.
        json_system = (system or "") + _JSON_STEER

        text = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Delegates to the shared parser (as base._generate_json_via does) instead of
        # re-implementing it: the local greedy pattern spanned from the first { to the LAST },
        # merging two adjacent objects into invalid JSON, never ran the refusal check, and raised
        # a bare ValueError -- so a caller with the documented `except JSONParsingError` /
        # `except LLMRefusalError` handlers caught neither, on the factory's DEFAULT provider.
        return self.extract_json(text, self._provider_display_name)

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate responses in batch (sequential for CLI)."""
        for req in requests:
            request_id = req.get("id", "unknown")
            try:
                result = await self.generate(
                    prompt=req["prompt"],
                    system=req.get("system"),
                    temperature=req.get("temperature", 0.7),
                    max_tokens=req.get("max_tokens", 1024),
                )
                yield {"id": request_id, "result": result}
            except Exception as e:
                # debug, not error: this runs once per FAILED request in a caller-supplied batch that
                # can be arbitrarily large, so a systemic failure (e.g. the provider is down) would
                # otherwise emit one ERROR-level line per request -- a log-flood during exactly the
                # incident an operator most needs signal, not noise. The failure itself is never
                # silently dropped: it's yielded back to the caller via {"id":..., "error":...} right
                # below, which is the actual per-request error channel this generator exposes.
                logger.debug("Batch request %s failed: %s", request_id, e)
                yield {"id": request_id, "error": str(e)}

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Cost is $0 - included in Max subscription."""
        return 0.0

    async def count_tokens(self, text: str) -> int:
        """Count tokens via Anthropic's native ``messages.count_tokens`` API when a usable
        Anthropic client/API key is available -- Claude Code runs real Claude models, so tiktoken
        (OpenAI's tokenizer, wrong for Claude's BPE for >5% of typical text -- see
        AnthropicProvider.count_tokens()'s docstring) is only an approximation, not "accurate".

        Falls back to the tiktoken/len//4 approximation whenever no Anthropic API key is
        configured (the common case for Claude Code, which authenticates via CLI subscription,
        not an API key) or the API call fails for any other reason -- this must never hard-require
        an API key just to count tokens.
        """
        try:
            import anthropic as _anthropic

            client = _anthropic.AsyncAnthropic()  # picks up ANTHROPIC_API_KEY from env, if set
            result = await client.messages.count_tokens(model=self.model, messages=[{"role": "user", "content": text}])
            return int(result.input_tokens)
        except Exception as exc:
            logger.debug("Claude Code count_tokens via Anthropic API unavailable (%s); falling back to tiktoken approximation.", exc)
            from pyutilz.llm.token_counter import count_tokens
            return count_tokens(text)

    async def get_account_credits(self) -> dict:
        """Always raise ``NotImplementedError``: Claude Code runs on a Max subscription with no per-token credit balance to fetch."""
        # Claude Code uses the Max subscription rather than per-token credits —
        # there is no "balance" concept. Usage is rate-limited within the
        # subscription's session windows (5h / weekly), surfaced by the CLI
        # itself, not via an API endpoint.
        raise NotImplementedError(
            "Claude Code uses a Max subscription — no per-token balance to fetch. " "Subscription usage windows surface in the CLI's own status output."
        )

    async def check_account_limits(self) -> dict:
        """Shell out to ``claude /status`` for subscription window info.

        Claude Code uses Max-subscription session windows (5h burn,
        weekly cap) rather than per-token limits. The CLI's
        ``/status`` command renders these; we capture its raw output
        and parse the obvious numeric markers (sessions left, reset
        time).

        Returns ``{"raw_text": <CLI output>, "session_summary":
        <parsed-best-effort dict>}``. Raises NotImplementedError when
        the CLI isn't available locally.
        """
        try:
            claude_path = _find_claude_executable()
        except Exception as exc:
            raise NotImplementedError(f"Claude Code CLI not on PATH ({exc}); cannot query /status.")

        proc = await asyncio.create_subprocess_exec(
            claude_path, "/status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=15.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except Exception as reap_exc:
                logger.warning("Failed to reap timed-out claude /status process (pid=%s): %s", proc.pid, reap_exc)
            raise NotImplementedError("claude /status timed out; CLI may be hung.")
        if proc.returncode != 0:
            # Previously silently fell through to a best-effort parse of empty/garbage stdout on
            # failure -- the caller had no way to tell "no limit info in the output" apart from
            # "the CLI itself failed" (found 2026-07-09: stderr was captured but never read).
            raise NotImplementedError(f"claude /status exited with code {proc.returncode}: {(stderr or b'').decode(errors='replace').strip() or '(no stderr)'}")
        text = (stdout or b"").decode(errors="replace")

        # Best-effort parse of common markers; CLI output format isn't
        # versioned-stable so we always include the raw text.
        summary: dict[str, Any] = {}
        for line in text.splitlines():
            ln = line.strip()
            if "session" in ln.lower() and "remain" in ln.lower():
                summary["session_remaining_hint"] = ln
            elif "reset" in ln.lower():
                summary["reset_hint"] = ln

        return {"raw_text": text, "session_summary": summary}
