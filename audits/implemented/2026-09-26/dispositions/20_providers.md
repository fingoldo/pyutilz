# Dispositions: 20_providers_coverage.md (PROV-*)

Worked 2026-09-26. Paths are relative to `src/pyutilz/llm/` unless stated; line numbers are the worktree after the fixes.
Regression tests live in `tests/llm/test_providers_coverage_20260926.py` (named `T::<class>` below) unless another file is given.

Vendor facts were re-fetched on 2026-09-26: platform.claude.com models overview, pricing, effort, extended-thinking,
stop-reasons and structured-outputs pages plus the Opus 4.5 / Sonnet 4.5 / Sonnet 4.6 / Opus 5 / Fable 5 model pages;
developers.openai.com pricing, models and reasoning guides; api-docs.deepseek.com pricing; docs.x.ai models, reasoning
and search-tools guides; ai.google.dev thinking and pricing pages. No Anthropic/OpenAI/DeepSeek/xAI/Gemini API key exists
in this environment, so no live API probe was made. The Claude Code CLI (2.1.263, subscription) WAS probed live, four
one-word calls: it confirmed `--effort` is accepted, `opus` resolves to `claude-opus-5` (not Opus 5.5), `sonnet` to
`claude-sonnet-5`, `haiku` to `claude-haiku-4-5-20251001`, `fable` to `claude-fable-5-1`, and that `usage` carries
`output_tokens_details.thinking_tokens` and `cache_creation.ephemeral_1h_input_tokens` (the field name PROV-7 flagged as unverified).

New shared module: `_claude_models.py`, one Claude model table (price, max output, context, cache-read multiplier, thinking
mode, effort and structured-output support) used by both the Anthropic API provider and the Claude Code provider, with no
SDK import. Unknown IDs warn once and use a documented fallback.

### PROV-1
**Disposition:** RESOLVED -- `anthropic_thinking_field` (anthropic_provider.py:81) now returns `{"type": "adaptive"}` for adaptive-thinking models and the budget form only for budget-only ones (per `_claude_models.py` table, `adaptive_thinking`); `anthropic_thinking_request` (anthropic_provider.py:127) adds `output_config.effort` (minimal->low, low/medium/high/xhigh/max passed through, `_thinking.py:53` `claude_effort`) where the model supports effort. The documented 400 (`"thinking.type.enabled" is not supported`, and the adaptive counterpart) is matched by `_thinking_mode_rejection` (:50) and learned per model in `_create_with_learned_repairs` (:426), the same way temperature rejections are. `THINKING_BUDGETS` gained `xhigh`/`max` (`_thinking.py:33`). Tests: `T::TestProv1AdaptiveThinking` (7 cases incl. learned flip and unrelated-400 passthrough); `tests/test_llm_anthropic_thinking.py` pinned to a budget model.

### PROV-2
**Disposition:** RESOLVED -- pricing/limits come from `_claude_models.py:51` `CLAUDE_MODELS` with real IDs (Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 5, Opus 4.8/4.7/4.6, Sonnet 5, Sonnet 4.6, Opus/Sonnet/Haiku 4.5, retired 4.x and 3.x under their real `claude-3-*` IDs). Resolution (`resolve_claude_model`, :99) accepts an exact ID or a snapshot suffix only (date, `-latest`, `@date`, `-v1:0`, `anthropic.` prefixes), so `claude-opus-4-9` is NOT priced as Opus 4. Unknown IDs warn once and use `UNKNOWN_CLAUDE_MODEL` (current Opus tier). `AnthropicProvider._get_pricing` (anthropic_provider.py:257) reads the table; `_PRICING` stays as a derived class attribute. Tests: `T::TestProv2Prov3ModelTable`; `tests/test_llm_provider_audit_fixes.py::TestAnthropicPricingFallback`, `tests/test_llm_providers.py::TestMaxOutputTokens` reframed to the documented figures.

### PROV-3
**Disposition:** RESOLVED -- `context_window` (anthropic_provider.py:268) reads the table: 1M for 4.6 and later, 200K for Haiku 4.5 / Opus 4.5 / Sonnet 4.5 and older. Test: `T::TestProv2Prov3ModelTable::test_real_ids_resolve`.

### PROV-4
**Disposition:** RESOLVED -- `_consume_response` joins every text block (anthropic_provider.py:527). Test: `T::TestProv4AllTextBlocks`.

### PROV-5
**Disposition:** RESOLVED -- `model_context_window_exceeded` raises `LLMTruncationError` with the partial text (:529); `refusal` raises `LLMRefusalError` carrying `stop_details` in `details` and on the new per-call `last_stop_details` (:541). Message Batches results map both to per-request errors (:771). Tests: `T::TestProv5StopReasons`.

### PROV-6
**Disposition:** RESOLVED -- `_account_usage` (:475) reads `usage.output_tokens_details.thinking_tokens` (field confirmed live on the CLI result); the chars//4 estimate is only the fallback when the field is absent, flagged by `last_thinking_tokens_estimated`. Tests: `T::TestProv6ReportedThinkingTokens` (reported wins; estimate only without the field).

### PROV-7
**Disposition:** RESOLVED -- `_cost_usd` (:670) uses the model's own cache-read multiplier (0.05 Opus 5.5, 0.025 Fable/Mythos 5.1, 0.10 otherwise) and bills 1-hour writes at 2x (tracked from `usage.cache_creation.ephemeral_1h_input_tokens` into `total_cache_creation_1h_input_tokens`). New `cache_ttl="5m"|"1h"` constructor option sends `cache_control.ttl` (:404). Tests: `T::TestProv7CacheRates` (3 multipliers, 1h vs 5m write cost, ttl sent and tracked).

### PROV-8
**Disposition:** RESOLVED -- `count_tokens` (:597) re-raises a permanent 4xx (not 408/409/429) and falls back to tiktoken only on transient failures, with a one-time WARNING instead of DEBUG. Tests: `T::TestProv8CountTokens`.

### PROV-9
**Disposition:** RESOLVED -- default model is `claude-sonnet-5` (:202); the default `max_tokens` stays `min(max_output, 21000)` because that is the SDK's non-streaming ceiling, and larger requests are now streamed (see NOT USED below). Test: `T::TestProv9DefaultModel`.

### NOT USED (Anthropic)
**Disposition:** RESOLVED (structured outputs, effort, streaming, Message Batches) -- structured outputs: `generate(json_schema=)` / `generate_json(json_schema=)` send `output_config.format` (:398), accepting a bare schema or OpenAI's `{"name","strict","schema"}` wrapper; `supports_json_schema` (:282) follows the documented model list. Effort: PROV-1. Streaming: `_create` (:459) streams via `messages.stream` above `_STREAMING_THRESHOLD_TOKENS` (21000), capturing headers from the stream response, so a caller can request the full 128K output. Message Batches: `generate_message_batch` (:714) submits, polls, collects results in request order, costs them at the documented 50% into `total_batch_cost_usd` (included in `get_session_cost`). Tests: `T::TestAnthropicNotUsedFeatures` (schema body, support table, streaming path, batch order + half price). Tool use, citations and `service_tier` remain unimplemented: this provider is single-turn text generation, and none of them has a caller in this package (WON'T FIX, out of scope).

### PROV-10
**Disposition:** RESOLVED -- `OpenAIProvider._openai_body` (openai_provider.py:260) renames `max_tokens` to `max_completion_tokens` on every request, applied in `_post_and_unwrap` (:275) and `_build_stream_body` (:279) overrides so both transports are covered without touching openai_compat.py. Test: `T::TestProv10Prov12OpenAIBody::test_reasoning_model_gets_max_completion_tokens_and_no_temperature` (drives `generate` through an `httpx.MockTransport`).

### PROV-11
**Disposition:** RESOLVED -- `_thinking_request_field` (:295) returns `{"reasoning_effort": ...}` for reasoning families (o-series, gpt-5+, gpt-6, not `gpt-5-chat`); `True` -> medium, off -> the family's lowest documented effort (`_LOWEST_EFFORT` :184: o-series/gpt-6-astra `low`, gpt-5 `minimal`, gpt-5.1+ `none`); a non-reasoning model warns and sends nothing. Test: `T::TestProv11OpenAIReasoningEffort` (8 cases).

### PROV-12
**Disposition:** RESOLVED -- not probed live (no key; OpenAI docs do not state it on the fetched pages), so both halves were implemented: reasoning models never get `temperature` (`_openai_body`), and any model answering a 400 about `temperature` (or about a `reasoning_effort` value) is repaired once through the base `_body_after_rejected_request` hook (:283), with the temperature rejection learned per model in `_MODELS_REJECTING_TEMPERATURE`. Tests: `T::TestProv10Prov12OpenAIBody` (reasoning model omits it, gpt-4o keeps it, 400 repaired and learned).

### PROV-13
**Disposition:** RESOLVED -- the always-raising `check_account_limits` override was deleted; OpenAI inherits the base snapshot of captured `x-ratelimit-*` headers. Tests: `T::TestProv13Prov14OpenAIAccounting::test_check_account_limits_returns_the_captured_headers`; `tests/test_llm_account_credits.py::test_openai_limits_mentions_response_headers` reframed.

### PROV-14
**Disposition:** RESOLVED -- `_known_row` (:337) recognises an exact ID or a dated/`-latest` snapshot of one; those price silently. Any other prefix match (e.g. `gpt-5-typo`) or a genuine miss still warns once, now naming the row actually used (`_resolve_pricing` :349, `_cache_hit_cost_per_1m` :393). Tables refreshed from developers.openai.com/api/docs/pricing (GPT-6 Astra/Sol/Luna, GPT-5.6 Sol/Terra/Luna, 5.4 family, 5.2, 5.1, o3-pro; -pro models carry no cache discount). Tests: `T::TestProv13Prov14OpenAIAccounting` (snapshot silent, genuine miss warns); `tests/test_llm_providers.py::TestOpenAIProvider` unchanged and passing.

### PROV-15
**Disposition:** NOT A DEFECT -- `num_tokens_from_messages` documents itself as the gpt-3.5-0613 framing estimate; its callers use it for budgeting only. The OpenAI "NOT USED" items: `max_completion_tokens`/`reasoning_effort` are now used (PROV-10/11); Responses API, Batch API, `service_tier` and `prompt_cache_key` are WON'T FIX here -- they need new request paths in openai_compat.py (not owned) and no caller in this package asks for them.

### PROV-16
**Disposition:** RESOLVED -- `deepseek-flash` added (384K output, 1M context), `deepseek-v4-flash` kept as the legacy alias billed at the Flash price, default model `deepseek-flash`, class defaults raised to the V4 values 384_000 / 1_000_000 (deepseek_provider.py:20, :70). Tests: `T::TestProv16Prov18DeepSeekFlash`; `tests/test_llm_deepseek.py` pricing/default tests reframed to the fetched table.

### PROV-17
**Disposition:** RESOLVED -- table is the fetched PEAK pricing (flash $0.30/$0.006/$1.20, v4-pro $1.32/$0.044/$3.96). `deepseek_price_multiplier` (:55) implements the documented window (01-04 and 06-10 UTC, Mon-Fri; Chinese public holidays are not modelled, so a holiday call is over- not under-estimated). `_track_provider_specific_usage` (:237) records each call's off-peak discount at completion time and `get_session_cost` (:257) subtracts it, reporting `offpeak_discount_usd`. Tests: `T::TestProv17DeepSeekOffPeak` (5 boundary times, off-peak half, peak full).

### PROV-18
**Disposition:** RESOLVED -- the pricing page states deepseek-flash "supports thinking mode (default) and non-thinking modes"; the toggle is now refused only for the fixed-mode legacy aliases `deepseek-chat`/`deepseek-reasoner` (`_LEGACY_FIXED_MODE_MODELS` :48, used at :139). Test: `T::TestProv16Prov18DeepSeekFlash::test_flash_takes_the_thinking_toggle`.

### PROV-19
**Disposition:** RESOLVED -- `_handle_special_status` (:100) now raises `LLMProviderError` on 402 (not an httpx error, so no retry predicate matches), unless the new `wait_on_insufficient_balance=True` constructor opt-in restores the warn-and-wait behaviour. The shared predicate in `_openai_compat_http.py` still classes 402 as retryable for other upstreams; unchanged. Tests: `T::TestProv19DeepSeek402`; `tests/test_llm_deepseek.py::test_handle_special_status_402_warns` now exercises the opt-in.

### PROV-20
**Disposition:** RESOLVED -- `search_parameters` is no longer sent. With `live_search` on, `XAIProvider.generate` (xai_provider.py:229) routes to `_generate_with_search` (:253), a `POST /responses` with `tools=[{"type":"web_search"},{"type":"x_search"}]` per docs.x.ai/docs/guides/tools/search-tools, retried with the shared HTTP predicate; `_unwrap_responses_output` (:283) maps usage into the chat-completions accounting, fills `last_citations`, and raises `LLMTruncationError` on `status=incomplete`. `live_search_max_sources` has no tools-API counterpart and is warned about. Implemented per docs, not probed live (no xAI key). Tests: `T::TestProv20XaiSearchTools` (path, tools, no search_parameters, citations, usage; chat path unchanged without live search).

### PROV-21
**Disposition:** RESOLVED (effort, models, long-context tier) / NOT VERIFIED (billed-output formula) -- `_thinking_request_field` (:188) sends `reasoning_effort` to grok-4.5/4.6/4.7 per the reasoning guide (off -> `low`, since reasoning is mandatory; grok-4.5 clamps `xhigh` to `high`; multi-agent never gets it because the field sets agent count there). Tables add grok-4.7/4.6/4.5/4.3, grok-4.20-0309 reasoning/non-reasoning/multi-agent, grok-build-0.1 with context windows. The documented >=200K-prompt tier (every rate doubles) is charged per call (`_track_provider_specific_usage` :207, `_LONG_CONTEXT_THRESHOLD` :102). `_compute_billed_output` (:178) stays completion+reasoning: the fetched guide states reasoning tokens are billed but not whether `completion_tokens` includes them, and no key was available to probe; the docstring records this. Tests: `T::TestProv21XaiEffortAndTier` (6 effort cases, new prices/context, tier on/off).

### PROV-22
**Disposition:** RESOLVED -- `gemini_thinking_config` (`_thinking.py:121`) sends `thinking_level` for Gemini 3+ (minimal/low/medium/high; xhigh/max clamp to high; "off" becomes the model's lowest documented level, `low` on 3-pro/3.1-pro/3.7-flash/3.8-flash which have no `minimal`) and keeps `thinking_budget` for 2.x. The "pro cannot disable" rule is now `startswith("gemini-2.5-pro")` (:161) instead of a substring. Wired at gemini_provider.py:357. Tests: `T::TestProv22GeminiThinkingLevel` (10 cases), `T::TestProv22Prov24GeminiRequest::test_gemini3_request_carries_thinking_level`.

### PROV-23
**Disposition:** RESOLVED -- the retry predicate includes `httpx.TransportError` (`_TRANSIENT_TRANSPORT_ERRORS`, gemini_provider.py:23), and the client is built with `HttpOptions(timeout=600_000 ms)` (:162). Tests: `T::TestProv23GeminiTransport`.

### PROV-24
**Disposition:** RESOLVED -- `generate(json_schema=)` / `generate_json(json_schema=)` send `response_json_schema` with `application/json` (:349); `supports_json_schema` returns True (:212). Test: `T::TestProv22Prov24GeminiRequest::test_json_schema_is_sent_as_response_json_schema`.

### PROV-25
**Disposition:** RESOLVED -- an empty-candidates response reads `prompt_feedback.block_reason` / `block_reason_message` into the `LLMSafetyBlockError` (:411); `PROHIBITED_CONTENT`, `RECITATION`, `SPII` (and `IMAGE_SAFETY`/`BLOCKLIST` via the existing substrings) are matched by `_BLOCKING_FINISH_REASONS` (:69). Also from this section's NOT USED list: the >200K tier of 2.5 Pro / 3.1 Pro is charged per call (`_add_long_context_surcharge` :230), Gemini 3.5/3.7/3.8 Flash were priced from the fetched page, and an unknown model now warns once before the default price (`_get_pricing` :216). Context-cache storage cost and Batch mode: WON'T FIX (storage is billed per hour of a cache resource this provider does not create; batch needs a separate job API with no caller here). Tests: `T::TestProv25GeminiBlockReasons` (prompt block reason, recitation, long-context cost).

### PROV-26
**Disposition:** RESOLVED -- `usage_int` (claude_code_cli.py:232) reads a usage field off a dict (the SDK's `ResultMessage.usage`) or an object; `generate` uses it (claude_code_provider.py:401-404) and takes the reasoning count from `output_tokens_details.thinking_tokens` when present (`usage_thinking_tokens`, cli :245; provider :425). Test: `T::TestProv26Prov27ClaudeCodeResult::test_dict_usage_from_the_sdk_is_read`.

### PROV-27
**Disposition:** RESOLVED -- CLI: a `result` event with `is_error: true` is an error regardless of `subtype` (claude_code_cli.py:198). SDK: `ResultMessage.is_error` raises `RuntimeError` with the result text (claude_code_provider.py:585), so a rate-limit notice still reaches `generate()`'s rate-limit wait via `_is_rate_limit_error`. Tests: `T::TestProv26Prov27ClaudeCodeResult::test_cli_is_error_result_is_an_error_not_an_answer`, `::test_sdk_is_error_result_raises`.

### PROV-28
**Disposition:** RESOLVED -- `_thinking_transport_kwargs` (claude_code_provider.py:495): on an adaptive model an effort string goes to `--effort` (CLI argv, :680) / `extra_args["effort"]` (SDK), mapped through `claude_code_effort` (`_thinking.py:67`); off still sends `MAX_THINKING_TOKENS=0`; a budget-thinking model (haiku) keeps the budget env. `--effort` acceptance verified live on CLI 2.1.263. Tests: `T::TestProv28Prov29ClaudeCodeThinkingAndLimits` (mapping, haiku budget, argv carries `--effort` with `--tools ""` still last); `tests/test_llm_thinking_reaches_every_provider.py::TestClaudeCode` extended to both paths.

### PROV-29
**Disposition:** RESOLVED -- `max_output_tokens`/`context_window` (claude_code_provider.py:282) resolve the model or CLI alias through the shared table (`CLAUDE_CODE_ALIASES`, `_claude_models.py:85`, pinned from the live probe: `opus` = Opus 5, 128K/1M; `haiku` 64K/200K). Tests: `T::TestProv28Prov29ClaudeCodeThinkingAndLimits::test_limits_follow_the_model`; `tests/test_llm_providers.py` claude-code limit tests reframed.

### PROV-30
**Disposition:** RESOLVED -- `generate_batch` forwards only the `temperature`/`max_tokens`/`thinking` keys a request carries (:847). Test: `T::TestProv30Prov31ClaudeCodeMisc::test_batch_does_not_invent_max_tokens`.

### PROV-31
**Disposition:** RESOLVED -- `count_tokens` uses `async with AsyncAnthropic()` (:883), closing the client, and resolves a CLI alias to its API model ID first (an alias like `opus` was always a 404 there). Tests: `T::TestProv30Prov31ClaudeCodeMisc::test_count_tokens_closes_its_client_and_resolves_the_alias`; `tests/test_llm_claude_code_generate.py::TestCountTokens` updated to the context-manager fake.

### PROV-32
**Disposition:** WON'T FIX -- porting the four `claude_code_sdk` monkey-patches to the renamed `claude-agent-sdk` cannot be verified here: neither SDK is installed, and installing one into the shared environment is out of bounds for this pass. The CLI backend, which is what actually runs, now carries every fix in this file (PROV-26/27/28). The `--json-schema` flag in the NOT USED list is not adopted because the schema is JSON in argv and the CLI path's argv guard (claude_code_provider.py, `_UNSAFE_ARGV`) refuses `"` on Windows `.cmd` shims by design; `--max-budget-usd`/`--fallback-model` have no caller.

### PROV-33
**Disposition:** NOT A DEFECT -- `count_tokens` is documented as an approximation used only for budgeting; the base class's proportional context reserve (`_CONTEXT_RESERVE_FRACTION = 0.30`) exists to absorb exactly this tokenizer error, and the providers with a native counter (Anthropic, Gemini) use it.

## Other changes made along the way

- `claude_code_provider.py` was at exactly 1000 lines; the rate-limit parsing helpers (`_RATE_LIMIT_PATTERN`, `_parse_reset_wait_seconds`, `_is_rate_limit_error`) moved to `claude_code_cli.py` (:450 on) and are re-exported, so existing imports keep working. The provider is now 954 lines.

## HANDOFF to CORE

1. `base.py` `_longest_prefix_pricing`: on a COMPLETE miss it returns `default` with no log line (only the prefix-match branch warns). Any provider relying on the base `_get_pricing` prices an unknown model silently. Spec: log one WARNING per (provider_label, model) before `return default`, same wording style as the prefix branch. Gemini now works around it in its own `_get_pricing` (gemini_provider.py:216); Anthropic no longer uses the base resolver.
2. `openai_compat.py` (current worktree, not by this pass) calls `self._raise_for_error_in_body(data, resp.request, ...)` unconditionally in `_post_and_unwrap`; `tests/test_llm_domain_extra.py::TestLLMTruncationErrorWiring::test_openai_compat_raises_on_length_finish_reason` builds a fake response without `.request` and now fails with `AttributeError`. Either read it with `getattr(resp, "request", None)` or give the test fake a `request` attribute.
3. Observed while running targeted tests, not caused by this pass: `tests/test_llm_supports_json_mode.py::test_provider_class_declares_own_override[...OpenAICompatibleProvider...]` fails (OpenAICompatibleProvider no longer declares `supports_json_mode` in the current openai_compat.py).
4. Observed once under `-n 4`, not caused by this pass (base.py `PerCallAttr`, untouched here): `tests/test_domain_db_web_cloud_llm_text_audit_20260903_llm.py::TestF10PerCallAttrThreadSafety::test_concurrent_first_touch_never_loses_a_write` failed.

## HANDOFF to OR

1. Observed, not caused by this pass: `tests/test_domain_db_web_cloud_llm_text_audit_20260903_llm.py::TestF37OpenRouterUnknownPricing` (2 tests) fail with `AttributeError: 'OpenRouterProvider' object has no attribute '_usage_by_model'` -- the tests build the provider with `__new__` and the new attribute is read without a `getattr` default.
5. REQUIRED by this pass: `base.py` `LLMProvider.generate_json` must accept `json_schema: dict[str, Any] | None = None` (last keyword, after `thinking`) and forward it to `_generate_json_via` only when not None, mirroring how `thinking` is forwarded. AnthropicProvider and GeminiProvider now declare it on their overrides (structured outputs, PROV-24 / NOT USED), so `tests/test_meta/test_code_audit_baseline.py` reports `override_signature_drift [P1] llm/base.py LLMProvider.generate_json`. ClaudeCodeProvider's `generate_json` would then take it too (ignore with a warning: the CLI's `--json-schema` is not usable, see PROV-32). The same baseline run also reports `async_primitive_reinit_per_call llm/_descriptors.py` (not from this pass).
