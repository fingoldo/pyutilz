"""Regression tests for the 2026-09-02 domain audit, part 2 (provider/driver-level paths).

Split from the sibling module purely to keep each file under the project's size conventions.
Nothing here touches a real database, a real network endpoint, or a real LLM provider.
"""

import asyncio
from unittest.mock import Mock

import pytest

import pyutilz.database.db as dbmod
import pyutilz.llm.claude_code_provider as ccmod
import pyutilz.web.browser as browsermod

# ---------------------------------------------------------------------------
# F02 -- per-thread cursors at the two remaining read sites
# ---------------------------------------------------------------------------


class _Col:
    def __init__(self, name):
        self.name = name


def test_f02_get_table_fields_uses_the_calling_threads_own_cursor(monkeypatch):
    """These two sites still read the module-global `cur`, which another thread can rebind
    between execute() and fetchall() -- returning column names for the wrong table."""
    handed_out = []

    def fake_get_cursor(cursor_type, **kw):
        c = Mock()
        c.description = [_Col("id"), _Col("name")]
        c.fetchall.return_value = []
        handed_out.append(c)
        return c

    monkeypatch.setattr(dbmod, "get_cursor", fake_get_cursor)
    monkeypatch.setattr(dbmod, "cur", None)  # the global must not be consulted at all
    out = dbmod.get_table_fields("orders", "o")
    assert out == "o.id id,o.name name"
    assert handed_out, "get_table_fields did not go through get_cursor()"


def test_f02_fetch_db_elements_star_uses_the_calling_threads_own_cursor(monkeypatch):
    cursor = Mock()
    cursor.description = [_Col("id"), _Col("name")]
    monkeypatch.setattr(dbmod, "get_cursor", lambda *a, **kw: cursor)
    monkeypatch.setattr(dbmod, "cur", None)

    class _Target:
        pass

    target = _Target()
    dbmod.fetch_db_elements(target, [(7, "seven")], "*")
    assert target.id == 7
    assert target.name == "seven"


# ---------------------------------------------------------------------------
# F10 -- streaming usage recorded once
# ---------------------------------------------------------------------------


def test_f10_repeated_usage_blocks_are_recorded_once():
    """`_track_streaming_usage` ran for EVERY chunk carrying a usage block. Several
    OpenAI-compatible upstreams emit cumulative usage on more than the final chunk, so
    get_session_cost() and the call count over-reported spend by a multiple."""
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _Stub(OpenAICompatibleProvider):
        def _input_cost_per_1m(self, model: str) -> float:
            return 0.0

        def _output_cost_per_1m(self, model: str) -> float:
            return 0.0

    p = _Stub.__new__(_Stub)
    p.model_name = "m"
    p._provider_name = "stub"
    p.total_prompt_tokens = 0
    p.total_completion_tokens = 0
    p.total_cache_hit_tokens = 0
    p.total_reasoning_tokens = 0
    p._call_count = 0
    p._max_concurrent = 1

    usage_chunk = '{"usage": {"prompt_tokens": 100, "completion_tokens": 10}, "choices": []}'
    lines = [
        'data: {"choices": [{"delta": {"content": "hi"}}]}',
        "data: " + usage_chunk,
        "data: " + usage_chunk,
        "data: " + usage_chunk,
        "data: [DONE]",
    ]

    class _Resp:
        headers: dict = {}
        status_code = 200

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in lines:
                yield line

    class _StreamCtx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    p._client = Mock()
    p._client.stream = Mock(return_value=_StreamCtx())
    p._capture_rate_limit_headers = lambda headers: None
    p._handle_special_status = lambda resp: None
    p._track_provider_specific_response = lambda chunk: None

    async def _prepare():
        return None

    p._async_prepare = _prepare
    p.fit_max_tokens_to_context = lambda mt, prompt, system: mt

    async def drain():
        return [chunk async for chunk in p.generate_stream("hello", max_tokens=10)]

    asyncio.run(drain())
    assert p._call_count == 1
    assert p.total_prompt_tokens == 100


# ---------------------------------------------------------------------------
# F26 / F32 -- ClaudeCodeProvider per-call state and ignored parameters
# ---------------------------------------------------------------------------


def _claude_code_provider_for_generate(monkeypatch, result_message=None):
    p = ccmod.ClaudeCodeProvider.__new__(ccmod.ClaudeCodeProvider)
    p.model = "sonnet"
    p._call_count = 0
    p.total_cost_usd = 0.0
    p.last_cost_usd = 0.0
    p.total_cache_creation_input_tokens = 0
    p.total_cache_read_input_tokens = 0
    p.last_cache_creation_input_tokens = 0
    p.last_cache_read_input_tokens = 0
    p.total_input_tokens = 0
    p.total_output_tokens = 0
    p.total_prompt_tokens = 0
    p.total_completion_tokens = 0
    p.last_session_id = None
    p.last_num_turns = None
    p._last_result_message = result_message

    async def fake_sdk(prompt, system=None):
        # Stands in for the real _generate_sdk, which resets per-call state INSIDE its semaphore
        # (moved there 2026-09-03, audit F08: generate() holds no lock, so a queued call used to
        # clear the running call's state). The F26 contract asserted below is unchanged.
        p._reset_per_call_state()
        return "answer"

    p._generate_sdk = fake_sdk
    monkeypatch.setattr(ccmod, "_HAS_SDK", True)
    return p


def test_f26_stale_result_message_is_not_re_counted(monkeypatch):
    """`_last_result_message` was assigned but never reset, so a call producing none (an SDK
    error path, or the CLI fallback) re-added the PREVIOUS call's cost and cache tokens."""
    stale = Mock()
    stale.usage = Mock(input_tokens=10, output_tokens=5, cache_creation_input_tokens=3, cache_read_input_tokens=2)
    stale.total_cost_usd = 0.42
    stale.session_id = "s"
    stale.num_turns = 1

    p = _claude_code_provider_for_generate(monkeypatch, result_message=stale)
    asyncio.run(p.generate("hi"))
    assert p.total_cost_usd == 0.0, "a previous call's ResultMessage was billed again"


def test_f32_ignored_max_tokens_and_temperature_are_warned_about(monkeypatch, caplog):
    """claude-code is the factory's DEFAULT provider, and it silently dropped both -- so a
    determinism requirement (temperature=0.0) did not hold, with no warning anywhere."""
    ccmod.ClaudeCodeProvider._seen_unsupported_params.clear()
    p = _claude_code_provider_for_generate(monkeypatch)
    with caplog.at_level("WARNING"):
        asyncio.run(p.generate("hi", temperature=0.0, max_tokens=32000))
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "max_tokens" in warned
    assert "temperature" in warned


# ---------------------------------------------------------------------------
# F28 / F29 -- Gemini JSON mode and truncation classification
# ---------------------------------------------------------------------------


def test_f28_gemini_generate_json_requests_native_json_mode():
    """base.generate_json forwarded no generate_kwargs, so json_mode stayed False and
    response_mime_type was never sent -- prompt-steering only, despite supports_json_mode()."""
    from pyutilz.llm.gemini_provider import GeminiProvider

    p = GeminiProvider.__new__(GeminiProvider)
    seen = {}

    async def fake_generate(prompt, system=None, temperature=0.3, max_tokens=0, **kwargs):
        seen.update(kwargs)
        return '{"k": 1}'

    p.generate = fake_generate
    assert asyncio.run(p.generate_json("x")) == {"k": 1}
    assert seen.get("json_mode") is True


def test_f29_max_tokens_beats_the_empty_text_safety_branch():
    """A thinking-enabled model with a tight max_tokens returns finish_reason=MAX_TOKENS with
    empty text. Reporting that as LLMSafetyBlockError ("do not retry") permanently abandoned a
    call whose documented remedy is "double max_tokens and re-issue"."""
    pytest.importorskip("google.genai")
    from pyutilz.llm.exceptions import LLMSafetyBlockError, LLMTruncationError
    from pyutilz.llm.gemini_provider import GeminiProvider

    p = GeminiProvider.__new__(GeminiProvider)
    p.model_name = "gemini-2.5-flash"
    p._max_concurrent = 1
    p._candidate_count = 1
    p._cached_content = None
    p.total_cached_content_tokens = 0
    p._call_count = 0
    p.total_prompt_tokens = 0
    p.total_completion_tokens = 0
    p.total_reasoning_tokens = 0
    p.fit_max_tokens_to_context = lambda mt, prompt, system: mt
    p._capture_candidate_metadata = lambda resp: None

    response = Mock()
    response.candidates = [Mock(finish_reason="MAX_TOKENS")]
    response.usage_metadata = None
    type(response).text = property(lambda self: "")

    async def fake_call(**kwargs):
        return response

    client = Mock()
    client.aio.models.generate_content = fake_call
    p.client = client

    with pytest.raises((LLMTruncationError, LLMSafetyBlockError)) as exc:
        asyncio.run(p.generate("hi", max_tokens=8))
    assert isinstance(exc.value, LLMTruncationError), "a max_tokens cutoff must not be reported as a safety block"


# ---------------------------------------------------------------------------
# F31 / F35 -- Anthropic truncation payload and retry/timeout policy
# ---------------------------------------------------------------------------


def _anthropic_provider_with_response(response):
    from pyutilz.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = "claude-sonnet-4-5"
    p._max_concurrent = 1
    p._call_count = 0
    p.total_input_tokens = 0
    p.total_output_tokens = 0
    p.total_thinking_tokens = 0
    p.total_cache_creation_input_tokens = 0
    p.total_cache_read_input_tokens = 0
    p.fit_max_tokens_to_context = lambda mt, prompt, system: mt
    p._capture_response_headers = lambda headers: None

    raw = Mock()
    raw.headers = {}
    raw.parse = lambda: response

    async def create(**kwargs):
        return raw

    client = Mock()
    client.messages.with_raw_response.create = create
    p.client = client
    return p


def _anthropic_response(blocks, stop_reason="max_tokens"):
    response = Mock()
    response.content = blocks
    response.stop_reason = stop_reason
    response.usage = Mock(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return response


def test_f31_truncation_error_carries_the_partial_text():
    """exceptions.py documents partial_text for "a caller that catches it to keep what was
    already paid for"; the Anthropic raise site omitted it and discarded the paid-for output."""
    pytest.importorskip("anthropic")
    from pyutilz.llm.exceptions import LLMTruncationError

    text_block = Mock(type="text")
    text_block.text = "half an answer"
    p = _anthropic_provider_with_response(_anthropic_response([text_block]))
    with pytest.raises(LLMTruncationError) as exc:
        asyncio.run(p.generate("hi", max_tokens=8))
    assert exc.value.partial_text == "half an answer"


def test_f31_thinking_only_content_raises_the_typed_error_not_attributeerror():
    """`response.content[0].text` raised AttributeError/IndexError exactly when extended thinking
    consumed the whole budget -- i.e. the max_tokens case the typed error exists for."""
    pytest.importorskip("anthropic")
    from pyutilz.llm.exceptions import LLMTruncationError

    thinking_block = Mock(spec=["type", "thinking"])
    thinking_block.type = "thinking"
    thinking_block.thinking = "pondering"
    p = _anthropic_provider_with_response(_anthropic_response([thinking_block]))
    with pytest.raises(LLMTruncationError):
        asyncio.run(p.generate("hi", max_tokens=8))


def test_f31_empty_content_raises_the_typed_error_not_indexerror():
    pytest.importorskip("anthropic")
    from pyutilz.llm.exceptions import LLMTruncationError

    p = _anthropic_provider_with_response(_anthropic_response([]))
    with pytest.raises(LLMTruncationError):
        asyncio.run(p.generate("hi", max_tokens=8))


def test_f35_anthropic_client_disables_sdk_retries_and_pins_a_timeout(monkeypatch):
    """The SDK's own retries multiplied every tenacity attempt, so a sustained 529 produced far
    more upstream calls than PYUTILZ_LLM_MAX_RETRIES documents, at the SDK's default timeout."""
    anthropic = pytest.importorskip("anthropic")
    from pyutilz.llm.anthropic_provider import AnthropicProvider

    captured = {}
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **kw: captured.update(kw) or Mock())
    AnthropicProvider(api_key="k")
    assert captured["max_retries"] == 0
    assert captured["timeout"] == AnthropicProvider._request_timeout_seconds


# ---------------------------------------------------------------------------
# F34 -- catalogue fetch must never block the event loop
# ---------------------------------------------------------------------------


def test_f34_catalogue_fetch_does_not_block_a_running_event_loop(monkeypatch):
    """`_fetch_models_catalogue` held _MODELS_LOCK across a blocking httpx.get(timeout=10),
    reachable from the loop via supports_json_schema / estimate_cost / get_session_cost and
    stalling every other in-flight request for up to 10 seconds."""
    from pyutilz.llm.openrouter_provider import _catalogue as catmod

    pkg = catmod._pkg()
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", {"cached/model": {"id": "cached/model"}}, raising=False)
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 0.0, raising=False)

    def boom(*a, **kw):
        raise AssertionError("a blocking HTTP fetch was issued from the event loop")

    monkeypatch.setattr(pkg.httpx, "get", boom)

    async def main():
        return catmod._fetch_models_catalogue()

    assert asyncio.run(main()) == {"cached/model": {"id": "cached/model"}}


def test_f34_off_loop_fetch_still_refreshes_the_catalogue(monkeypatch):
    from pyutilz.llm.openrouter_provider import _catalogue as catmod

    pkg = catmod._pkg()
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", None, raising=False)
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", 0.0, raising=False)

    resp = Mock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"data": [{"id": "fresh/model"}]}
    monkeypatch.setattr(pkg.httpx, "get", lambda *a, **kw: resp)

    assert "fresh/model" in catmod._fetch_models_catalogue()


# ---------------------------------------------------------------------------
# F52 -- Selenium restart must not orphan the previous driver
# ---------------------------------------------------------------------------


def test_f52_restart_closes_the_previous_driver_first(monkeypatch):
    """On the "no such window" half of the condition the driver process is typically still
    alive, so overwriting the handle directly orphaned a chromedriver on every re-login."""
    pytest.importorskip("selenium")
    closed = []
    started = []

    fake = Mock()
    fake.title = "logged-in"
    fake.execute_script.return_value = "UA/1.0"
    fake.get_cookie.side_effect = lambda name: {"value": "v"}

    monkeypatch.setattr(browsermod, "browser", fake)
    monkeypatch.setattr(browsermod, "home_page", "http://example.invalid/")
    monkeypatch.setattr(browsermod, "required_cookies", ("sid",))
    monkeypatch.setattr(browsermod, "fixed_cookies", {})
    monkeypatch.setattr(browsermod, "logout_signs", ())
    monkeypatch.setattr(browsermod, "successful_login_signs", ("logged-in",))
    monkeypatch.setattr(browsermod.pythonlib, "imitate_delay", lambda **kw: None)
    monkeypatch.setattr(browsermod, "sleep", lambda *a, **kw: None)
    monkeypatch.setattr(browsermod, "close_browser", lambda: closed.append(True))
    monkeypatch.setattr(browsermod, "start_selenium", lambda: started.append(True) or fake)

    calls = []

    def flaky_get(url):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("no such window: target window already closed")

    monkeypatch.setattr(browsermod, "browser_get", flaky_get)

    browsermod.LoginAndGetCookies()
    assert closed, "the old driver was not closed before restarting Selenium"
    assert started
