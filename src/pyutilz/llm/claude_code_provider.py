"""Claude Code provider - uses your Max subscription through Claude Code.

Supports two backends:
1. SDK (preferred) -- uses claude-code-sdk Python package, works from nested sessions
2. CLI fallback -- spawns `claude` CLI subprocess (breaks inside VSCode extension)
"""

import asyncio
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from pyutilz.llm.base import LLMProvider

logger = logging.getLogger(__name__)

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
            cmd = _orig_build(self)
            if not self._is_streaming and "--print" in cmd:
                try:
                    idx = cmd.index("--print")
                    cmd = cmd[:idx] + ["--print", "-"]
                except ValueError:
                    pass
            return cmd
        _sc.SubprocessCLITransport._build_command = _patched_build

        # -- PATCH 3: Stdin prompt delivery + stderr capture --
        async def _patched_connect(self):
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
                from subprocess import PIPE as _PIPE
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
                raise _sc.CLINotFoundError(
                    f"Claude Code not found at: {cmd[0]}"
                ) from e
        _sc.SubprocessCLITransport.connect = _patched_connect

        # -- PATCH 4: Include real stderr in ProcessError --
        _orig_read_messages = _sc.SubprocessCLITransport._read_messages_impl
        from claude_code_sdk._errors import ProcessError as _ProcessError

        async def _patched_read_messages(self):
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
                    except Exception:
                        pass
                if real_stderr:
                    raise _ProcessError(
                        f"Command failed with exit code {e.exit_code}",
                        exit_code=e.exit_code,
                        stderr=real_stderr[:2000],
                    ) from None
                raise
        _sc.SubprocessCLITransport._read_messages_impl = _patched_read_messages

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

    now = datetime.now(tz) if tz else datetime.now()

    month_str = m.group("month")
    day_str = m.group("day")
    if month_str and day_str:
        _MONTHS = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        month = _MONTHS[month_str.lower()]
        day = int(day_str)
        reset_time = now.replace(
            month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0,
        )
        if reset_time <= now:
            reset_time = reset_time.replace(year=now.year + 1)
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


def _find_claude_executable() -> str:
    """Find the claude executable path safely."""
    claude_path = shutil.which('claude')
    if claude_path:
        return claude_path

    if sys.platform == 'win32':
        npm_prefix = os.environ.get('APPDATA', '')
        possible_paths = [
            os.path.join(npm_prefix, 'npm', 'claude.cmd'),
            os.path.join(npm_prefix, 'npm', 'claude'),
            os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'npm', 'claude.cmd'),
        ]
        for path in possible_paths:
            if os.path.isfile(path):
                return path

    raise FileNotFoundError(
        "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
    )


class ClaudeCodeProvider(LLMProvider):
    """LLM provider that uses Claude Code for LLM calls (your Max subscription).

    Two backends: SDK (preferred, via `claude-code-sdk`) or CLI fallback.
    """

    INPUT_COST_PER_1M = 0.0
    OUTPUT_COST_PER_1M = 0.0

    def __init__(
        self,
        model: str = "opus",
        max_concurrent: int = 1,
        timeout: int = 2400,
    ):
        self.model = model
        self.model_name = model
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        self._call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def get_session_cost(self) -> dict:
        """Return estimated cost data (Max subscription -- no per-call billing)."""
        return {
            "calls": self._call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "reasoning_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }

    @property
    def max_output_tokens(self) -> int:
        return 32000

    @property
    def context_window(self) -> int:
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

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 0,
        json_mode: bool = False,
    ) -> str:
        """Generate text using Claude Code SDK (preferred) or CLI fallback."""
        if json_mode:
            json_hint = "\n\nRespond with valid JSON only. No markdown, no explanation. Start with { and end with }."
            system = (system or "") + json_hint

        self._call_count += 1
        attempt = 0
        max_attempts = 20
        while True:
            attempt += 1
            if attempt > max_attempts:
                raise RuntimeError(
                    f"ClaudeCodeProvider: exceeded {max_attempts} retry attempts"
                )
            try:
                if _HAS_SDK:
                    result = await self._generate_sdk(prompt, system)
                else:
                    result = await self._generate_cli(prompt, system, temperature, max_tokens)
                if json_mode and result:
                    stripped = result.strip()
                    fence_match = re.search(
                        r'```(?:json)?\s*([\s\S]*?)\s*```', stripped,
                    )
                    if fence_match:
                        result = fence_match.group(1).strip()
                from pyutilz.llm.token_counter import count_tokens as _count_tok
                in_tok = _count_tok(prompt) + _count_tok(system or "")
                out_tok = _count_tok(result)
                self._last_usage = {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "reasoning_tokens": 0,
                }
                self.total_prompt_tokens += in_tok
                self.total_completion_tokens += out_tok
                return result
            except (ConnectionError, TimeoutError, OSError) as e:
                wait = min(5 * (2 ** (attempt - 1)), 300)
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
                        "[RateLimit] Hit rate limit but couldn't parse reset time: %s. "
                        "Waiting %d seconds (default).",
                        error_msg[:200], wait_seconds,
                    )
                else:
                    reset_dt = datetime.now() + timedelta(seconds=wait_seconds)
                    logger.warning(
                        "[RateLimit] Hit rate limit. Pausing until %s (%d seconds). "
                        "Will retry automatically. Error: %s",
                        reset_dt.strftime("%H:%M:%S"), wait_seconds, error_msg[:200],
                    )
                await asyncio.sleep(wait_seconds)
                logger.info("[RateLimit] Resuming after rate limit pause (attempt %d).",
                           attempt + 1)

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
            override_env = {k: "" for k in self._NESTED_BLOCK_VARS if k in os.environ}

            combined_len = len(prompt) + len(system or "")
            if combined_len > 6000 and system:
                prompt = f"[System instructions]\n{system}\n\n[User request]\n{prompt}"
                system = None

            opts = ClaudeCodeOptions(
                model=self.model,
                system_prompt=system or "",
                permission_mode="bypassPermissions",
                max_turns=1,
                env=override_env,
                extra_args={"tools": ""},
            )

            _tools_val = opts.extra_args.get("tools") if opts.extra_args else None
            if _tools_val != "":
                logger.warning(
                    'ClaudeCodeProvider: extra_args["tools"] was %r -- forcing to "" '
                    "to disable built-in tools and preserve JSON compliance.",
                    _tools_val,
                )
                if opts.extra_args is None:
                    opts.extra_args = {}
                opts.extra_args["tools"] = ""

            logger.info("SDK call: model=%s prompt=%d system=%d", self.model, len(prompt), len(system or ""))

            result_text = None
            msg_count = 0
            msg_types = []
            async for msg in cc_query(prompt=prompt, options=opts):
                msg_count += 1
                if msg is None:
                    msg_types.append("None")
                    continue
                msg_types.append(type(msg).__name__)
                msg_type = type(msg).__name__

                if isinstance(msg, ResultMessage):
                    if isinstance(msg.result, str) and msg.result:
                        result_text = msg.result
                elif msg_type == "AssistantMessage":
                    if hasattr(msg, "content") and msg.content:
                        parts = []
                        for block in msg.content:
                            bt = type(block).__name__
                            if isinstance(block, TextBlock):
                                parts.append(block.text)
                            elif bt == "ToolUseBlock":
                                tool_name = getattr(block, "name", "?")
                                logger.warning("Model attempted tool use: %s (blocked)", tool_name)
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
                logger.warning("SDK returned empty response: model=%s prompt_len=%d msgs=%d types=%s",
                              self.model, len(prompt), msg_count, msg_types)
            return stripped

    async def _generate_cli(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text using Claude Code CLI (fallback)."""
        async with self.semaphore:
            if not hasattr(self, '_claude_path'):
                self._claude_path = _find_claude_executable()

            cmd = [
                self._claude_path,
                '--print',
                '--verbose',
                '--include-partial-messages',
                '--model', self.model,
                '--output-format', 'stream-json',
                '--dangerously-skip-permissions',
                '--no-session-persistence',
                '--tools', '',
            ]

            if '--tools' not in cmd:
                cmd.extend(['--tools', ''])

            if system:
                cmd.extend(['--system-prompt', system])

            cmd.append('-')

            logger.debug("Running Claude CLI: %s...", cmd[0])

            sub_env = {k: v for k, v in os.environ.items() if k not in self._NESTED_BLOCK_VARS}

            def run_cli():
                # NOTE: This Popen is NOT wrapped in a ``with`` block
                # because the surrounding logic already has a ``try /
                # finally`` (line below) that runs ``proc.kill()`` +
                # ``proc.communicate(timeout=5)`` on every exit path —
                # functionally equivalent to ``with``. The
                # ``test_resource_handle_safety`` meta-test grandfathers
                # this site via its baseline.
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    shell=False,
                    env=sub_env,
                )

                try:
                    proc.stdin.write(prompt)
                    proc.stdin.close()
                except BrokenPipeError:
                    pass

                result_text = None
                error_text = None
                line_q: queue.Queue[str | None] = queue.Queue()

                def _reader():
                    try:
                        for line in proc.stdout:
                            line_q.put(line)
                    finally:
                        line_q.put(None)

                reader_thread = threading.Thread(target=_reader, daemon=True)
                reader_thread.start()

                deadline = time.monotonic() + self.timeout
                try:
                    while time.monotonic() < deadline:
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
                            continue

                        etype = event.get('type')
                        if etype == 'result':
                            subtype = event.get('subtype', '')
                            if subtype == 'success':
                                result_text = event.get('result', '')
                            else:
                                error_text = event.get('result') or event.get('error') or subtype
                            break
                        elif etype == 'rate_limit_event':
                            logger.debug("Claude CLI rate_limit_event (continuing)")
                        elif etype == 'system' and event.get('subtype') == 'init':
                            logger.debug("Claude CLI initialized")
                    else:
                        proc.kill()
                        proc.communicate()
                        raise subprocess.TimeoutExpired(proc.args, self.timeout)
                finally:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    try:
                        proc.communicate(timeout=5)
                    except (subprocess.TimeoutExpired, ValueError, OSError):
                        pass

                if error_text is not None:
                    raise RuntimeError(f"Claude CLI error: {error_text}")
                if result_text is None:
                    stderr_data = proc.stderr.read(500) if not proc.stderr.closed else ""
                    raise RuntimeError(f"Claude CLI produced no result. stderr: {stderr_data}")

                return 0, result_text, ""

            loop = asyncio.get_event_loop()
            returncode, stdout, stderr = await loop.run_in_executor(None, run_cli)

            if returncode != 0:
                error_msg = stderr or stdout or "Unknown error"
                raise RuntimeError(f"Claude CLI failed (code {returncode}): {error_msg}")

            return stdout.strip()

    async def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
    ) -> dict[str, Any]:
        """Generate structured JSON output."""
        json_system = (system or "") + "\n\nRespond with valid JSON only. No markdown, no explanation."

        text = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            text = text.strip()
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
            if json_match:
                return json.loads(json_match.group(1))
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {text}")
            raise ValueError(f"Invalid JSON response: {e}")

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
                logger.error(f"Batch request {request_id} failed: {e}")
                yield {"id": request_id, "error": str(e)}

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Cost is $0 - included in Max subscription."""
        return 0.0

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (accurate) or len//4 fallback."""
        from pyutilz.llm.token_counter import count_tokens
        return count_tokens(text)

    async def get_account_credits(self) -> dict:
        # Claude Code uses the Max subscription rather than per-token credits —
        # there is no "balance" concept. Usage is rate-limited within the
        # subscription's session windows (5h / weekly), surfaced by the CLI
        # itself, not via an API endpoint.
        raise NotImplementedError(
            "Claude Code uses a Max subscription — no per-token balance to fetch. "
            "Subscription usage windows surface in the CLI's own status output."
        )

    async def check_account_limits(self) -> dict:
        raise NotImplementedError(
            "Claude Code rate limits follow Max-subscription session windows, "
            "not exposed via API. The CLI shows them inline as ``/status``."
        )
