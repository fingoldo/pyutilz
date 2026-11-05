"""Session-accounting, cleanup and model-spec pins for ClaudeCodeProvider (mutation survivors)."""

from __future__ import annotations

import logging

import pytest

from pyutilz.llm import claude_code_provider as prov
from pyutilz.llm.claude_code_provider import ClaudeCodeProvider


class TestDefaults:
    def test_a_fresh_provider_uses_the_opus_alias_one_slot_and_forty_minutes(self):
        p = ClaudeCodeProvider()
        assert (p.model, p.model_name, p.timeout, p._max_concurrent) == ("opus", "opus", 2400, 1)

    def test_a_fresh_provider_has_no_session_totals(self):
        p = ClaudeCodeProvider()
        assert (p._call_count, p._successful_call_count, p.total_prompt_tokens, p.total_completion_tokens) == (0, 0, 0, 0)
        assert (p.total_reasoning_tokens, p.total_cost_usd, p.total_cache_creation_input_tokens, p.total_cache_read_input_tokens) == (0, 0.0, 0, 0)

    def test_json_mode_is_a_soft_guarantee_so_it_is_reported_unsupported(self):
        assert ClaudeCodeProvider().supports_json_mode() is False


class TestSessionCost:
    def test_the_session_cost_maps_each_total_to_its_key(self):
        p = ClaudeCodeProvider()
        p._call_count = 5
        p._successful_call_count = 4
        p.total_prompt_tokens = 100
        p.total_completion_tokens = 20
        p.total_reasoning_tokens = 7
        p.total_cost_usd = 1.5
        p.total_cache_creation_input_tokens = 3
        p.total_cache_read_input_tokens = 40
        cost = p.get_session_cost()
        assert cost["calls"] == 5
        assert cost["successful_calls"] == 4
        assert cost["prompt_tokens"] == 100
        assert cost["completion_tokens"] == 20
        assert cost["reasoning_tokens"] == 7
        assert cost["total_cost_usd"] == 1.5
        assert cost["cache_creation_input_tokens"] == 3
        assert cost["cache_read_input_tokens"] == 40
        assert cost["cache_hit_tokens"] == 40

    def test_cache_misses_are_the_uncached_prompt_tokens(self):
        p = ClaudeCodeProvider()
        p.total_prompt_tokens = 100
        p.total_cache_read_input_tokens = 40
        assert p.get_session_cost()["cache_miss_tokens"] == 60

    def test_cache_misses_never_go_negative(self):
        p = ClaudeCodeProvider()
        p.total_prompt_tokens = 10
        p.total_cache_read_input_tokens = 40
        assert p.get_session_cost()["cache_miss_tokens"] == 0

    def test_the_cli_path_has_no_per_token_price(self):
        cost = ClaudeCodeProvider().get_session_cost()
        assert (cost["input_cost_usd"], cost["output_cost_usd"]) == (0.0, 0.0)


class TestModelSpec:
    def test_the_output_ceiling_and_context_window_come_from_the_model_table(self):
        p = ClaudeCodeProvider(model="opus")
        assert p.max_output_tokens == prov.claude_model_spec("opus", provider_label="Claude Code").max_output
        assert p.context_window == prov.claude_model_spec("opus", provider_label="Claude Code").context_window

    def test_a_missing_model_attribute_falls_back_to_opus(self):
        p = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
        assert p._spec == prov.claude_model_spec("opus", provider_label="Claude Code")


class _ResultMessage:
    def __init__(self, **fields) -> None:
        self.__dict__.update(fields)


_USAGE = {
    "input_tokens": 10,
    "output_tokens": 4,
    "cache_creation_input_tokens": 3,
    "cache_read_input_tokens": 2,
    "output_tokens_details": {"thinking_tokens": 5},
}


class TestRecordCallUsage:
    def test_the_sdk_result_usage_is_accumulated_into_the_session_totals(self):
        p = ClaudeCodeProvider()
        p._last_result_message = _ResultMessage(usage=_USAGE, total_cost_usd=0.25, session_id="s1", num_turns=3)
        p._record_call_usage("prompt", None, "answer")
        assert (p.total_prompt_tokens, p.total_completion_tokens, p.total_reasoning_tokens) == (10, 4, 5)
        assert (p.total_cache_creation_input_tokens, p.total_cache_read_input_tokens) == (3, 2)
        assert (p.last_cache_creation_input_tokens, p.last_cache_read_input_tokens) == (3, 2)
        assert (p.last_cost_usd, p.total_cost_usd) == (0.25, 0.25)
        assert (p.last_session_id, p.last_num_turns) == ("s1", 3)
        assert p._last_usage == {
            "input_tokens": 10,
            "output_tokens": 4,
            "reasoning_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }

    def test_two_calls_add_up_in_the_session_totals(self):
        p = ClaudeCodeProvider()
        p._last_result_message = _ResultMessage(usage=_USAGE, total_cost_usd=0.25, session_id="s1", num_turns=3)
        p._record_call_usage("prompt", None, "answer")
        p._record_call_usage("prompt", None, "answer")
        assert (p.total_prompt_tokens, p.total_completion_tokens) == (20, 8)
        assert p.total_cost_usd == pytest.approx(0.5)

    def test_a_non_string_session_id_and_non_int_turn_count_are_not_recorded(self):
        p = ClaudeCodeProvider()
        p._last_result_message = _ResultMessage(usage=_USAGE, total_cost_usd=0.0, session_id=5, num_turns="three")
        p._record_call_usage("prompt", None, "answer")
        assert p.last_session_id is None
        assert p.last_num_turns is None


class _Message:
    def __init__(self, content) -> None:
        self.content = content


class ToolUseBlock:
    def __init__(self, name=None) -> None:
        if name is not None:
            self.name = name


class ThinkingBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class TestAssistantMessageText:
    def test_text_blocks_are_joined_with_newlines(self):
        msg = _Message([prov.TextBlock(text="first"), prov.TextBlock(text="second")])
        assert ClaudeCodeProvider()._assistant_message_text(msg) == "first\nsecond"

    def test_a_message_without_content_gives_none(self):
        assert ClaudeCodeProvider()._assistant_message_text(_Message([])) is None
        assert ClaudeCodeProvider()._assistant_message_text(object()) is None

    def test_a_tool_use_block_raises_the_tripwire_naming_the_tool(self):
        with pytest.raises(prov.ClaudeCodeToolUseError, match="'Bash'"):
            ClaudeCodeProvider()._assistant_message_text(_Message([ToolUseBlock(name="Bash")]))

    def test_a_tool_use_block_without_a_name_is_reported_as_unknown(self):
        with pytest.raises(prov.ClaudeCodeToolUseError, match=r"'\?'"):
            ClaudeCodeProvider()._assistant_message_text(_Message([ToolUseBlock()]))

    def test_thinking_text_stays_out_of_the_answer_but_is_kept_for_the_caller(self):
        p = ClaudeCodeProvider()
        p._reasoning_fragments = []
        msg = _Message([ThinkingBlock("secret reasoning"), prov.TextBlock(text="answer")])
        assert p._assistant_message_text(msg) == "answer"
        assert p._reasoning_fragments


class TestFinalizeSdkText:
    def test_no_result_at_all_raises_and_is_logged_with_the_message_count(self, caplog):
        caplog.set_level(logging.ERROR)
        with pytest.raises(RuntimeError, match="produced no result"):
            ClaudeCodeProvider()._finalize_sdk_text("p", None, 3, ["assistant"])
        assert "SDK produced no result after 3 messages" in caplog.text

    def test_the_answer_is_stripped_of_surrounding_whitespace(self):
        assert ClaudeCodeProvider()._finalize_sdk_text("p", "  hello \n", 1, []) == "hello"

    def test_an_empty_answer_is_returned_empty_and_warned(self, caplog):
        caplog.set_level(logging.WARNING)
        assert ClaudeCodeProvider()._finalize_sdk_text("abc", "   ", 2, ["result"]) == ""
        assert "SDK returned empty response" in caplog.text


@pytest.fixture
def no_sleep(monkeypatch):
    """Record every asyncio.sleep the provider makes, with no real waiting and no jitter."""
    import asyncio

    slept: list = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(prov.random, "uniform", lambda a, b: 0.0)
    return slept


class TestRetryAfterTransientError:
    def test_a_permanent_error_is_re_raised_without_sleeping(self, no_sleep):
        import asyncio

        p = ClaudeCodeProvider()
        try:
            raise FileNotFoundError("no claude")
        except FileNotFoundError as e:
            with pytest.raises(FileNotFoundError):
                asyncio.run(p._retry_after_transient_error(e, 1, 0, 3))
        assert no_sleep == []

    def test_a_transient_error_sleeps_the_exponential_backoff_and_keeps_the_timeout_count(self, no_sleep):
        import asyncio

        p = ClaudeCodeProvider()
        result = asyncio.run(p._retry_after_transient_error(ConnectionResetError("reset"), 2, 0, 3))
        assert result == 0
        assert no_sleep == [10]

    def test_a_timeout_is_counted_and_retried_while_under_the_budget(self, no_sleep):
        import asyncio

        p = ClaudeCodeProvider()
        result = asyncio.run(p._retry_after_transient_error(prov.subprocess.TimeoutExpired("claude", 5), 1, 0, 3))
        assert result == 1
        assert no_sleep == [5]

    def test_the_last_timeout_in_the_budget_gives_up(self, no_sleep):
        import asyncio

        p = ClaudeCodeProvider()
        try:
            raise prov.subprocess.TimeoutExpired("claude", 5)
        except prov.subprocess.TimeoutExpired as e:
            with pytest.raises(prov.subprocess.TimeoutExpired):
                asyncio.run(p._retry_after_transient_error(e, 3, prov.MAX_TIMEOUT_RETRIES - 1, 3))
        assert no_sleep == []


class TestPauseForRateLimit:
    def test_an_unparsable_reset_waits_an_hour_and_says_so(self, no_sleep, caplog):
        import asyncio

        caplog.set_level(logging.WARNING)
        asyncio.run(ClaudeCodeProvider()._pause_for_rate_limit(RuntimeError("limit hit")))
        assert no_sleep == [3600]
        assert "couldn't parse reset time" in caplog.text

    def test_a_parsed_reset_waits_exactly_that_long(self, no_sleep, monkeypatch):
        import asyncio

        monkeypatch.setattr(prov, "_parse_reset_wait_seconds", lambda text: 120)
        asyncio.run(ClaudeCodeProvider()._pause_for_rate_limit(RuntimeError("resets 3pm")))
        assert no_sleep == [120]


class TestCostAndTokens:
    def test_a_subscription_call_is_never_priced(self):
        assert ClaudeCodeProvider().estimate_cost(10**6, 10**6) == 0.0

    def test_counting_tokens_without_an_api_key_falls_back_to_the_local_counter(self, monkeypatch):
        import asyncio

        from pyutilz.llm.token_counter import count_tokens

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        text = "the quick brown fox jumps over the lazy dog"
        assert asyncio.run(ClaudeCodeProvider().count_tokens(text)) == count_tokens(text)

    def test_the_account_credits_are_reported_as_not_fetchable(self):
        import asyncio

        with pytest.raises(NotImplementedError, match="Max subscription"):
            asyncio.run(ClaudeCodeProvider().get_account_credits())


class TestCleanup:
    def test_no_path_means_nothing_is_removed(self):
        assert prov._remove_system_prompt_file(None) is None

    def test_an_existing_prompt_file_is_removed(self, tmp_path):
        path = tmp_path / "system.txt"
        path.write_text("hello")
        prov._remove_system_prompt_file(str(path))
        assert not path.exists()

    def test_a_missing_prompt_file_is_tolerated_and_logged(self, tmp_path, caplog):
        caplog.set_level(logging.DEBUG)
        missing = tmp_path / "gone.txt"
        prov._remove_system_prompt_file(str(missing))
        assert f"Could not remove the temporary system-prompt file {missing}" in caplog.text

    def test_every_recorded_child_has_its_process_tree_killed(self, monkeypatch):
        killed = []
        monkeypatch.setattr(prov, "_kill_process_tree", lambda child: killed.append(child))
        prov._kill_proc_holder(["a", "b"])
        assert killed == ["a", "b"]


class TestUnsupportedParameterWarning:
    def test_an_unsupported_parameter_is_warned_with_its_name(self, caplog):
        caplog.set_level(logging.WARNING)
        ClaudeCodeProvider()._warn_unsupported_param_once("seed_for_pin_one")
        assert "ClaudeCodeProvider cannot forward 'seed_for_pin_one' to the Claude Code SDK" in caplog.text

    def test_the_same_parameter_is_warned_only_once(self, caplog):
        caplog.set_level(logging.WARNING)
        p = ClaudeCodeProvider()
        p._warn_unsupported_param_once("seed_for_pin_two")
        p._warn_unsupported_param_once("seed_for_pin_two")
        assert sum("seed_for_pin_two" in r.getMessage() for r in caplog.records) == 1
