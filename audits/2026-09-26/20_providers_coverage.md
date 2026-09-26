# 20 - Native provider feature coverage and correctness (non-OpenRouter)

Scope: `src/pyutilz/llm/{anthropic_provider,openai_provider,openai_compat,openai_tokens,gemini_provider,deepseek_provider,xai_provider,claude_code_provider,claude_code_cli,_thinking,token_counter}.py`.
Read-only audit, 2026-09-26. Line numbers refer to the worktree at audit time.

Docs consulted (fetched 2026-09-26):
- D1 https://platform.claude.com/docs/en/about-claude/models/overview
- D2 https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- D3 https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
- D4 https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create (via search snippet; platform.openai.com returned 403)
- D5 https://api-docs.deepseek.com/quick_start/pricing
- D6 https://docs.x.ai/docs/guides/reasoning
- D7 https://docs.x.ai/docs/guides/live-search , https://github.com/langchain-ai/langchain/issues/33961 , https://github.com/openclaw/openclaw/issues/26355
- D8 https://ai.google.dev/gemini-api/docs/thinking
- D9 `claude --help` of installed CLI 2.1.263; `claude_code_sdk-0.0.25` wheel `types.py`

Items marked "unverified" were not confirmed against a fetched doc and need a live probe before fixing.

---

## Anthropic (`anthropic_provider.py`)

### PROV-1 (High) -- `thinking={"type":"enabled","budget_tokens":N}` is rejected with 400 by every current Claude model

**Disposition:** RESOLVED -- `anthropic_thinking_field` (anthropic_provider.py:81) now returns `{"type": "adaptive"}` for adaptive-thinking models and the budget form only for budget-only ones (per `_claude_models.py` table, `adaptive_thinking`); `anthropic_thinking_request` (anthropic_provider.py:127) adds `output_config.effort` (minimal->low, low/medium/high/xhigh/max passed through, `_thinking.py:53` `claude_effort`) where the model supports effort. The documented 400 (`"thinking.type.enabled" is not supported`, and the adaptive counterpart) is matched by `_thinking_mode_rejection` (:50) and learned per model in `_create_with_learned_repairs` (:426), the same way temperature rejections are. `THINKING_BUDGETS` gained `xhigh`/`max` (`_thinking.py:33`). Tests: `T::TestProv1AdaptiveThinking` (7 cases incl. learned flip and unrelated-400 passthrough); `tests/test_llm_anthropic_thinking.py` pinned to a budget model.
Evidence: `anthropic_thinking_field` always returns `{"type": "enabled", "budget_tokens": ...}` (anthropic_provider.py:75); `generate` sends it when `thinking` is truthy (:292-294). D2: "Claude 4.7 and later models do not support it and reject requests that use it, returning a 400 error"; deprecated on 4.6. D1 lists the current lineup (claude-fable-5-1, claude-opus-5-5, claude-sonnet-5) as "Adaptive". The 400 is not in the retry set, so the call fails outright.
Impact: any `thinking=True/"high"` call on a current model fails; on 4.6 it uses a deprecated path.
Fix: per-model mode: `{"type":"adaptive"}` + `output_config: {"effort": <effort>}` for 4.6+ (map minimal/low/medium/high onto effort), `budget_tokens` only for <=4.5. Learn it the same way `_MODELS_REJECTING_TEMPERATURE` is learned (retry on the documented `"thinking.type.enabled" is not supported` message) or key a table. Same for the Claude Code budgets (see PROV-22).

### PROV-2 (High) -- Pricing / limits tables miss every current model and use wrong legacy IDs

**Disposition:** RESOLVED -- pricing/limits come from `_claude_models.py:51` `CLAUDE_MODELS` with real IDs (Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 5, Opus 4.8/4.7/4.6, Sonnet 5, Sonnet 4.6, Opus/Sonnet/Haiku 4.5, retired 4.x and 3.x under their real `claude-3-*` IDs). Resolution (`resolve_claude_model`, :99) accepts an exact ID or a snapshot suffix only (date, `-latest`, `@date`, `-v1:0`, `anthropic.` prefixes), so `claude-opus-4-9` is NOT priced as Opus 4. Unknown IDs warn once and use `UNKNOWN_CLAUDE_MODEL` (current Opus tier). `AnthropicProvider._get_pricing` (anthropic_provider.py:257) reads the table; `_PRICING` stays as a derived class attribute. Tests: `T::TestProv2Prov3ModelTable`; `tests/test_llm_provider_audit_fixes.py::TestAnthropicPricingFallback`, `tests/test_llm_providers.py::TestMaxOutputTokens` reframed to the documented figures.
Evidence: `_PRICING` (:97-117) and `_MAX_OUTPUT_TOKENS` (:177-191) contain no `claude-opus-5-5`, `claude-sonnet-5`, `claude-fable-5-1`, `claude-opus-5`, `claude-opus-4-8`. Unknown models get `_DEFAULT_PRICING=(3,15)` (:118). D1: Opus 5.5 is $4/$20, Fable 5.1 $10/$50, Sonnet 5 $2/$10. Legacy keys use invented ID shapes: `claude-haiku-3-20240307`, `claude-haiku-3-5-20241022`, `claude-sonnet-3-7-20250219`, `claude-opus-3-20240229`; the real IDs are `claude-3-haiku-20240307` etc., which neither exact- nor prefix-match (`longest_prefix_lookup`, base.py:208-233), so Haiku 3 is priced at Sonnet $3/$15 (12x). Dates `claude-opus-4-6-20250610`, `claude-sonnet-4-6-20250610`, `claude-opus-4-5-20250414`, `claude-sonnet-4-5-20250414` do not match D1's statement that 4.6+ IDs are dateless; they only work through the trimmed-prefix fallback.
Impact: `get_session_cost` under-reports Fable 5.1 by 3.3x, over-reports Sonnet 5 by 1.5x; max_output 64000 default vs 128K actual caps outputs.
Fix: refresh from D1/pricing page with real IDs; log a warning on default-pricing fallback (the Anthropic path is silent, unlike OpenAI/xAI).

### PROV-3 (High) -- `context_window` hardcoded to 200_000

**Disposition:** RESOLVED -- `context_window` (anthropic_provider.py:268) reads the table: 1M for 4.6 and later, 200K for Haiku 4.5 / Opus 4.5 / Sonnet 4.5 and older. Test: `T::TestProv2Prov3ModelTable::test_real_ids_resolve`.
Evidence: anthropic_provider.py:208-210. D1: 1M for Fable 5.1 / Opus 5.5 / Sonnet 5; 200K only for Haiku 4.5.
Impact: `fit_max_tokens_to_context` clamps/refuses prompts >200K that fit.
Fix: per-model table (or Models API `max_input_tokens`, D1 "Using the Models API").

### PROV-4 (Med) -- Only the FIRST text block is returned

**Disposition:** RESOLVED -- `_consume_response` joins every text block (anthropic_provider.py:527). Test: `T::TestProv4AllTextBlocks`.
Evidence: :391-395 `break`s on the first `text` block. Responses with citations, or with text split around other blocks, carry multiple text blocks.
Impact: silently truncated answers.
Fix: `"".join(b.text for b in content if b.type == "text")`.

### PROV-5 (Med) -- `stop_reason` `refusal` and `model_context_window_exceeded` not handled

**Disposition:** RESOLVED -- `model_context_window_exceeded` raises `LLMTruncationError` with the partial text (:529); `refusal` raises `LLMRefusalError` carrying `stop_details` in `details` and on the new per-call `last_stop_details` (:541). Message Batches results map both to per-request errors (:771). Tests: `T::TestProv5StopReasons`.
Evidence: only `max_tokens` is checked (:403). D3 lists `refusal` ("read stop_details and retry on a fallback model") and `model_context_window_exceeded` ("treat the response as truncated"). `stop_details` is never read.
Impact: a refusal returns whatever partial text exists (or `LLMProviderError` "no text block"), not a typed refusal; a context-exceeded response is returned as complete.
Fix: raise `LLMTruncationError` for `model_context_window_exceeded`; raise a refusal-typed error (the package already has refusal semantics in base.py:289/exceptions.py:43) carrying `stop_details`.

### PROV-6 (Med) -- Thinking tokens estimated as chars//4 though the API reports them

**Disposition:** RESOLVED -- `_account_usage` (:475) reads `usage.output_tokens_details.thinking_tokens` (field confirmed live on the CLI result); the chars//4 estimate is only the fallback when the field is absent, flagged by `last_thinking_tokens_estimated`. Tests: `T::TestProv6ReportedThinkingTokens` (reported wins; estimate only without the field).
Evidence: :361-375 estimates from `block.thinking` text. D2: "monitor the `usage.output_tokens_details.thinking_tokens` field ... how many of the billed output tokens were internal reasoning". Thinking text is summarized (D2 example "Thinking summary"), so chars//4 undercounts real reasoning.
Fix: read `usage.output_tokens_details.thinking_tokens`, fall back to estimate only when absent. (Cost itself is fine: output_tokens already includes thinking.)

### PROV-7 (Med) -- Cache-read multiplier hardcoded 0.10; cache write always 1.25x

**Disposition:** RESOLVED -- `_cost_usd` (:670) uses the model's own cache-read multiplier (0.05 Opus 5.5, 0.025 Fable/Mythos 5.1, 0.10 otherwise) and bills 1-hour writes at 2x (tracked from `usage.cache_creation.ephemeral_1h_input_tokens` into `total_cache_creation_1h_input_tokens`). New `cache_ttl="5m"|"1h"` constructor option sends `cache_control.ttl` (:404). Tests: `T::TestProv7CacheRates` (3 multipliers, 1h vs 5m write cost, ttl sent and tracked).
Evidence: get_session_cost :525-529. D1 pricing note: cache reads "10% of the base input price (2.5% on Claude Fable 5.1 and Claude Mythos 5.1, 5% on Claude Opus 5.5)". The 1h-TTL write (2x) is not modeled and not requestable (cache_control hardcoded `{"type":"ephemeral"}` at :312, no `ttl`).
Fix: per-model cache-read multiplier; expose `cache_ttl` option and track `usage.cache_creation.ephemeral_1h_input_tokens` vs 5m separately (field name unverified against a fetched doc).

### PROV-8 (Low) -- `count_tokens` failure silently falls back to tiktoken

**Disposition:** RESOLVED -- `count_tokens` (:597) re-raises a permanent 4xx (not 408/409/429) and falls back to tiktoken only on transient failures, with a one-time WARNING instead of DEBUG. Tests: `T::TestProv8CountTokens`.
Evidence: :453-463 catches every exception at DEBUG. An auth failure or wrong model ID looks like a count. Fix: log WARNING once; re-raise non-transient 4xx.

### PROV-9 (Low) -- Default model `claude-sonnet-4-20250514` and `max_tokens` default `min(max_output, 21000)`

**Disposition:** RESOLVED -- default model is `claude-sonnet-5` (:202); the default `max_tokens` stays `min(max_output, 21000)` because that is the SDK's non-streaming ceiling, and larger requests are now streamed (see NOT USED below). Test: `T::TestProv9DefaultModel`.
Evidence: :140, :276. Not in D1 current/legacy list shown. Fix: default to a current model.

#### NOT USED (Anthropic) -- worth adding
- Structured outputs / `output_config` JSON schema: `supports_json_mode` returns False (:212-218) and JSON relies on prompt+extract. Worth adding (not re-verified in this audit which models accept it).
- Message Batches API (50% off, D1 pricing note; up to 300k output with beta header, D1 "Max output"): not used. Worth adding for bulk jobs.
- `effort` parameter (D1 "Default effort"): not used; natural mapping for the package's effort strings.
- Tool use, citations, `service_tier`, streaming: not implemented; streaming worth adding for long outputs (D2 advises batch/streaming for long thinking to avoid timeouts; `_request_timeout_seconds = 120` at :85 is short for 128K outputs).
- Thinking signature passthrough: N/A (single-turn API only).
USED correctly: 529/5xx/429 retry (:227-233), SDK retries disabled (:153), rate-limit header capture (:417-436), cache read/write token capture (:353-370), `input_tokens` excludes cache tokens in cost (:514-529), temperature forced to 1 under thinking (:295-306), learned temperature rejection (:330-346).

---

## OpenAI (`openai_provider.py` + `openai_compat.py`)

### PROV-10 (High) -- Sends `max_tokens`, which reasoning models do not accept

**Disposition:** RESOLVED -- `OpenAIProvider._openai_body` (openai_provider.py:260) renames `max_tokens` to `max_completion_tokens` on every request, applied in `_post_and_unwrap` (:275) and `_build_stream_body` (:279) overrides so both transports are covered without touching openai_compat.py. Test: `T::TestProv10Prov12OpenAIBody::test_reasoning_model_gets_max_completion_tokens_and_no_temperature` (drives `generate` through an `httpx.MockTransport`).
Evidence: request body uses `"max_tokens"` (openai_compat.py:767 buffered, :559 stream); `OpenAIProvider` does not override it and has no `_body_after_rejected_request` repair (grep: only the base stub at openai_compat.py:820). D4: "`max_tokens` ... is now deprecated in favor of `max_completion_tokens`, and is not compatible with o-series models." Default model is `gpt-5-mini` (openai_provider.py:136); table includes o1/o3/o4-mini/gpt-5*.
Impact: o-series calls are rejected (400 is non-retryable -> `LLMProviderError`). gpt-5 behaviour unverified; probe.
Fix: OpenAIProvider body hook sending `max_completion_tokens` (note `_timeouts.py:66` already reads that key).

### PROV-11 (High) -- `thinking=` silently ignored for OpenAI; `reasoning_effort` never sent

**Disposition:** RESOLVED -- `_thinking_request_field` (:295) returns `{"reasoning_effort": ...}` for reasoning families (o-series, gpt-5+, gpt-6, not `gpt-5-chat`); `True` -> medium, off -> the family's lowest documented effort (`_LOWEST_EFFORT` :184: o-series/gpt-6-astra `low`, gpt-5 `minimal`, gpt-5.1+ `none`); a non-reasoning model warns and sends nothing. Test: `T::TestProv11OpenAIReasoningEffort` (8 cases).
Evidence: `ThinkingControlMixin._thinking_request_field` returns None (_thinking.py:83-99 region, file lines 622-643 of dump); `OpenAIProvider` does not override it, although the base docstring claims "OpenAI `reasoning_effort`" (openai_compat.py:737-738).
Fix: override to return `{"reasoning_effort": effort}` for reasoning models; `False` -> lowest supported effort.

### PROV-12 (Med) -- Default `temperature=0.7` sent to reasoning models (unverified)

**Disposition:** RESOLVED -- not probed live (no key; OpenAI docs do not state it on the fetched pages), so both halves were implemented: reasoning models never get `temperature` (`_openai_body`), and any model answering a 400 about `temperature` (or about a `reasoning_effort` value) is repaired once through the base `_body_after_rejected_request` hook (:283), with the temperature rejection learned per model in `_MODELS_REJECTING_TEMPERATURE`. Tests: `T::TestProv10Prov12OpenAIBody` (reasoning model omits it, gpt-4o keeps it, 400 repaired and learned).
Evidence: openai_compat.py:712/766. OpenAI reasoning models are documented to reject non-default temperature (could not re-fetch: 403). No learned-rejection fallback like Anthropic's.
Fix: probe; then either omit for reasoning families or port the Anthropic learned-rejection pattern into `_body_after_rejected_request`.

### PROV-13 (Low) -- `check_account_limits` override discards captured headers

**Disposition:** RESOLVED -- the always-raising `check_account_limits` override was deleted; OpenAI inherits the base snapshot of captured `x-ratelimit-*` headers. Tests: `T::TestProv13Prov14OpenAIAccounting::test_check_account_limits_returns_the_captured_headers`; `tests/test_llm_account_credits.py::test_openai_limits_mentions_response_headers` reframed.
Evidence: openai_provider.py:178-186 always raises, although the base (openai_compat.py:210-232) returns the captured `x-ratelimit-*` snapshot. Fix: delete the override.

### PROV-14 (Low) -- Unknown-model warning fires even when prefix match succeeds

**Disposition:** RESOLVED -- `_known_row` (:337) recognises an exact ID or a dated/`-latest` snapshot of one; those price silently. Any other prefix match (e.g. `gpt-5-typo`) or a genuine miss still warns once, now naming the row actually used (`_resolve_pricing` :349, `_cache_hit_cost_per_1m` :393). Tables refreshed from developers.openai.com/api/docs/pricing (GPT-6 Astra/Sol/Luna, GPT-5.6 Sol/Terra/Luna, 5.4 family, 5.2, 5.1, o3-pro; -pro models carry no cache discount). Tests: `T::TestProv13Prov14OpenAIAccounting` (snapshot silent, genuine miss warns); `tests/test_llm_providers.py::TestOpenAIProvider` unchanged and passing.
Evidence: openai_provider.py:209-212 calls `_warn_unknown_model_once` before checking the prefix result; dated snapshots get a spurious "unknown; falling back to gpt-5-mini" warning while correctly priced. `_cache_hit_cost_per_1m` (:241-248) same.

### PROV-15 (Low) -- `num_tokens_from_messages` uses gpt-3.5-0613 framing

**Disposition:** NOT A DEFECT -- `num_tokens_from_messages` documents itself as the gpt-3.5-0613 framing estimate; its callers use it for budgeting only. The OpenAI "NOT USED" items: `max_completion_tokens`/`reasoning_effort` are now used (PROV-10/11); Responses API, Batch API, `service_tier` and `prompt_cache_key` are WON'T FIX here -- they need new request paths in openai_compat.py (not owned) and no caller in this package asks for them.
Evidence: openai_tokens.py:300-328; documented as estimate. OK as documented.
NOT USED (OpenAI): Responses API, Batch API (50% off), `service_tier` (flex/priority), `prompt_cache_key`. Worth adding: `service_tier="flex"` and Batch for bulk; Responses only if reasoning-item reuse is needed.
USED correctly: `prompt_tokens_details.cached_tokens` (openai_compat.py:656), `completion_tokens` already includes reasoning (openai_provider.py:159-165), strict `json_schema` response_format (openai_compat.py:300-320), `stream_options.include_usage` (:575).

---

## DeepSeek (`deepseek_provider.py`)

### PROV-16 (High) -- Current model `deepseek-flash` unknown: 8192 max output, 64K window, wrong price

**Disposition:** RESOLVED -- `deepseek-flash` added (384K output, 1M context), `deepseek-v4-flash` kept as the legacy alias billed at the Flash price, default model `deepseek-flash`, class defaults raised to the V4 values 384_000 / 1_000_000 (deepseek_provider.py:20, :70). Tests: `T::TestProv16Prov18DeepSeekFlash`; `tests/test_llm_deepseek.py` pricing/default tests reframed to the fetched table.
Evidence: D5 lists `deepseek-flash` (V4.1-Flash) and `deepseek-v4-pro`, both 1M context / 384K max output. Tables (deepseek_provider.py:348-369) have `deepseek-v4-flash` but not `deepseek-flash`; exact-match lookups (openai_compat.py:271, :283) give `_default_max_tokens=8192`, `_default_context_window=64_000`.
Fix: add `deepseek-flash`; raise defaults to V4 values.

### PROV-17 (Med) -- Pricing stale and off-peak not modeled

**Disposition:** RESOLVED -- table is the fetched PEAK pricing (flash $0.30/$0.006/$1.20, v4-pro $1.32/$0.044/$3.96). `deepseek_price_multiplier` (:55) implements the documented window (01-04 and 06-10 UTC, Mon-Fri; Chinese public holidays are not modelled, so a holiday call is over- not under-estimated). `_track_provider_specific_usage` (:237) records each call's off-peak discount at completion time and `get_session_cost` (:257) subtracts it, reporting `offpeak_discount_usd`. Tests: `T::TestProv17DeepSeekOffPeak` (5 boundary times, off-peak half, peak full).
Evidence: table `deepseek-v4-pro: (1.74, 0.0145, 3.48)` (:351). D5 peak: miss $1.32, hit $0.044, output $3.96; off-peak half ("Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday through Friday"). deepseek-flash peak $0.3/$0.006/$1.2.
Fix: refresh; compute cost per call with a time-of-day multiplier (record per-call cost at completion time rather than recomputing session cost from totals, since rate varies per call).

### PROV-18 (Low) -- Legacy alias check `startswith("deepseek-v4")` excludes `deepseek-flash`

**Disposition:** RESOLVED -- the pricing page states deepseek-flash "supports thinking mode (default) and non-thinking modes"; the toggle is now refused only for the fixed-mode legacy aliases `deepseek-chat`/`deepseek-reasoner` (`_LEGACY_FIXED_MODE_MODELS` :48, used at :139). Test: `T::TestProv16Prov18DeepSeekFlash::test_flash_takes_the_thinking_toggle`.
Evidence: :425. `thinking=` on deepseek-flash is warned "not supported" and dropped. Whether deepseek-flash takes the `thinking` field is unverified; check.

### PROV-19 (Low) -- HTTP 402 (no balance) retried forever

**Disposition:** RESOLVED -- `_handle_special_status` (:100) now raises `LLMProviderError` on 402 (not an httpx error, so no retry predicate matches), unless the new `wait_on_insufficient_balance=True` constructor opt-in restores the warn-and-wait behaviour. The shared predicate in `_openai_compat_http.py` still classes 402 as retryable for other upstreams; unchanged. Tests: `T::TestProv19DeepSeek402`; `tests/test_llm_deepseek.py::test_handle_special_status_402_warns` now exercises the opt-in.
Evidence: `_is_retryable_http_error` treats 402 as retryable (_openai_compat_http.py:61-65) with INFINITE_RETRY; the warning at :401-406 says so. Deliberate, but a batch job hangs silently until top-up. Consider a cap / opt-in.
USED correctly: `prompt_cache_hit_tokens` (openai_compat.py:656), `reasoning_content` via `_reasoning.from_message` (:892), `/user/balance` (:454-481).

---

## xAI (`xai_provider.py`)

### PROV-20 (High) -- Live Search `search_parameters` retired; requests now 410

**Disposition:** RESOLVED -- `search_parameters` is no longer sent. With `live_search` on, `XAIProvider.generate` (xai_provider.py:229) routes to `_generate_with_search` (:253), a `POST /responses` with `tools=[{"type":"web_search"},{"type":"x_search"}]` per docs.x.ai/docs/guides/tools/search-tools, retried with the shared HTTP predicate; `_unwrap_responses_output` (:283) maps usage into the chat-completions accounting, fills `last_citations`, and raises `LLMTruncationError` on `status=incomplete`. `live_search_max_sources` has no tools-API counterpart and is warned about. Implemented per docs, not probed live (no xAI key). Tests: `T::TestProv20XaiSearchTools` (path, tools, no search_parameters, citations, usage; chat path unchanged without live search).
Evidence: xai_provider.py:643-652 adds `search_parameters` when `live_search` is set. D7: Live Search "retired on January 12, 2026, after which requests return a 410 Gone ... 'Live search is deprecated. Please switch to the Agent Tools API'". 410 is non-retryable -> the whole call fails.
Fix: remove, or reimplement via Responses API `tools=[{"type":"web_search"}]` / `x_search`.

### PROV-21 (Med) -- `reasoning_effort` not sent; billed-output formula unverified

**Disposition:** RESOLVED (effort, models, long-context tier) / NOT VERIFIED (billed-output formula) -- `_thinking_request_field` (:188) sends `reasoning_effort` to grok-4.5/4.6/4.7 per the reasoning guide (off -> `low`, since reasoning is mandatory; grok-4.5 clamps `xhigh` to `high`; multi-agent never gets it because the field sets agent count there). Tables add grok-4.7/4.6/4.5/4.3, grok-4.20-0309 reasoning/non-reasoning/multi-agent, grok-build-0.1 with context windows. The documented >=200K-prompt tier (every rate doubles) is charged per call (`_track_provider_specific_usage` :207, `_LONG_CONTEXT_THRESHOLD` :102). `_compute_billed_output` (:178) stays completion+reasoning: the fetched guide states reasoning tokens are billed but not whether `completion_tokens` includes them, and no key was available to probe; the docstring records this. Tests: `T::TestProv21XaiEffortAndTier` (6 effort cases, new prices/context, tier on/off).
Evidence: no `_thinking_request_field` override, so `thinking=` is ignored. D6: grok-4.5/4.6/4.7 accept `reasoning_effort` low/medium/high/xhigh. Tables (:543-601) lack grok-4.5/4.6/4.7 (default pricing $0.20/$0.50 with warning, context 2M). `_compute_billed_output = completion + reasoning` (:665-667): D6 does not state whether `completion_tokens` excludes reasoning; if it includes it, cost double-counts. Probe a live response.
Fix: add effort mapping + new models; verify usage semantics.

---

## Gemini (`gemini_provider.py`, `_thinking.py`)

### PROV-22 (Med) -- Uses `thinking_budget`; Gemini 3 docs describe `thinking_level`

**Disposition:** RESOLVED -- `gemini_thinking_config` (`_thinking.py:121`) sends `thinking_level` for Gemini 3+ (minimal/low/medium/high; xhigh/max clamp to high; "off" becomes the model's lowest documented level, `low` on 3-pro/3.1-pro/3.7-flash/3.8-flash which have no `minimal`) and keeps `thinking_budget` for 2.x. The "pro cannot disable" rule is now `startswith("gemini-2.5-pro")` (:161) instead of a substring. Wired at gemini_provider.py:357. Tests: `T::TestProv22GeminiThinkingLevel` (10 cases), `T::TestProv22Prov24GeminiRequest::test_gemini3_request_carries_thinking_level`.
Evidence: gemini_provider.py:277-279 sends `ThinkingConfig(thinking_budget=...)`; `gemini_thinking_budget` (_thinking.py:597-616) maps effort to the Claude budgets 1024-8192 and sends 0 to disable. D8 documents `thinking_level` for Gemini 3 and says models other than 2.5-flash-lite "require explicitly setting thinking_level to 'minimal'" to reduce thinking; D8 does not mention budget 0. The "pro cannot disable" rule (:608) is a substring heuristic.
Fix: for `gemini-3*` send `thinking_level` (minimal/low/medium/high) directly from the effort string; keep budget for 2.5.

### PROV-23 (Med) -- httpx transport errors not retried; no explicit timeout

**Disposition:** RESOLVED -- the retry predicate includes `httpx.TransportError` (`_TRANSIENT_TRANSPORT_ERRORS`, gemini_provider.py:23), and the client is built with `HttpOptions(timeout=600_000 ms)` (:162). Tests: `T::TestProv23GeminiTransport`.
Evidence: retry predicate (:229) = `ConnectionError, TimeoutError, OSError` or genai `ServerError`/429. google-genai uses httpx, whose `httpx.TransportError` family (ConnectError, ReadTimeout) is not an `OSError` subclass, so network blips fail the call. `genai.Client(api_key=...)` (:125) sets no `http_options` timeout, unlike every other provider.
Fix: add `httpx.TransportError`; pass a timeout.

### PROV-24 (Low) -- Structured output not used

**Disposition:** RESOLVED -- `generate(json_schema=)` / `generate_json(json_schema=)` send `response_json_schema` with `application/json` (:349); `supports_json_schema` returns True (:212). Test: `T::TestProv22Prov24GeminiRequest::test_json_schema_is_sent_as_response_json_schema`.
Evidence: only `response_mime_type` (:268); `response_schema` / `response_json_schema` never sent, `supports_json_schema` not overridden. Worth adding (enum guarantees, as OpenAI path has).

### PROV-25 (Low) -- Prompt-level block reason not captured

**Disposition:** RESOLVED -- an empty-candidates response reads `prompt_feedback.block_reason` / `block_reason_message` into the `LLMSafetyBlockError` (:411); `PROHIBITED_CONTENT`, `RECITATION`, `SPII` (and `IMAGE_SAFETY`/`BLOCKLIST` via the existing substrings) are matched by `_BLOCKING_FINISH_REASONS` (:69). Also from this section's NOT USED list: the >200K tier of 2.5 Pro / 3.1 Pro is charged per call (`_add_long_context_surcharge` :230), Gemini 3.5/3.7/3.8 Flash were priced from the fetched page, and an unknown model now warns once before the default price (`_get_pricing` :216). Context-cache storage cost and Batch mode: WON'T FIX (storage is billed per hour of a cache resource this provider does not create; batch needs a separate job API with no caller here). Tests: `T::TestProv25GeminiBlockReasons` (prompt block reason, recitation, long-context cost).
Evidence: when `candidates` is empty, finish_reason is "unknown" (:291-294) and the error says "likely safety block"; `response.prompt_feedback.block_reason` is never read. `PROHIBITED_CONTENT`/`RECITATION` finish reasons are not matched by the "SAFETY"/"BLOCK" test (:321-322) and fall to the empty-text branch (still an LLMSafetyBlockError, but with less detail).
USED correctly: `thoughts_token_count` billed as output (:215; D8 "response pricing is the sum of output tokens and thinking tokens"), `cached_content_token_count` at cache rate (:207-214), explicit `cached_content` (:275-276), native `count_tokens` (:434-453), grounding/citation capture (:373-423).
NOT USED: long-context (>200K) price tier for 2.5-pro/3.x pro, context-cache storage cost, Batch mode.

---

## Claude Code (`claude_code_provider.py`, `claude_code_cli.py`)

### PROV-26 (High) -- SDK path records zero tokens and zero cost

**Disposition:** RESOLVED -- `usage_int` (claude_code_cli.py:232) reads a usage field off a dict (the SDK's `ResultMessage.usage`) or an object; `generate` uses it (claude_code_provider.py:401-404) and takes the reasoning count from `output_tokens_details.thinking_tokens` when present (`usage_thinking_tokens`, cli :245; provider :425). Test: `T::TestProv26Prov27ClaudeCodeResult::test_dict_usage_from_the_sdk_is_read`.
Evidence: `claude_code_sdk` `ResultMessage.usage: dict[str, Any] | None` (D9, types.py). `generate` reads it with `getattr(rm_usage, "input_tokens", 0)` (claude_code_provider.py:471-477 in dump), which returns 0 on a dict. The CLI path wraps usage in `SimpleNamespace` (claude_code_cli.py:236-241) and is fine.
Impact: whenever `claude_code_sdk` is installed (preferred backend), all token/cache accounting is 0 (cost is read correctly since `total_cost_usd` is an attribute).
Fix: normalise `rm.usage` with `dict.get` when it is a mapping.

### PROV-27 (Med) -- `is_error` ignored on both paths

**Disposition:** RESOLVED -- CLI: a `result` event with `is_error: true` is an error regardless of `subtype` (claude_code_cli.py:198). SDK: `ResultMessage.is_error` raises `RuntimeError` with the result text (claude_code_provider.py:585), so a rate-limit notice still reaches `generate()`'s rate-limit wait via `_is_rate_limit_error`. Tests: `T::TestProv26Prov27ClaudeCodeResult::test_cli_is_error_result_is_an_error_not_an_answer`, `::test_sdk_is_error_result_raises`.
Evidence: D9 `ResultMessage.is_error: bool`. CLI: `_consume_cli_stream` accepts any `subtype=="success"` with non-empty `result` (claude_code_cli.py:192-203); SDK: `msg.result` taken regardless (provider :638-644). An API error surfaced as result text would be returned as the model's answer. (Whether the CLI emits `subtype=success, is_error=true` for API errors is unverified; the field exists, so check it.)
Fix: treat `is_error` true as failure; route rate-limit text through `_is_rate_limit_error`.

### PROV-28 (Med) -- Thinking passed via MAX_THINKING_TOKENS budgets

**Disposition:** RESOLVED -- `_thinking_transport_kwargs` (claude_code_provider.py:495): on an adaptive model an effort string goes to `--effort` (CLI argv, :680) / `extra_args["effort"]` (SDK), mapped through `claude_code_effort` (`_thinking.py:67`); off still sends `MAX_THINKING_TOKENS=0`; a budget-thinking model (haiku) keeps the budget env. `--effort` acceptance verified live on CLI 2.1.263. Tests: `T::TestProv28Prov29ClaudeCodeThinkingAndLimits` (mapping, haiku budget, argv carries `--effort` with `--tools ""` still last); `tests/test_llm_thinking_reaches_every_provider.py::TestClaudeCode` extended to both paths.
Evidence: `claude_code_thinking_tokens` (_thinking.py:578-594) -> env `MAX_THINKING_TOKENS` (provider :583-584, :781-782). The CLI exposes `--effort <level>` (D9 `claude --help`: "Effort level for the current session"), and current models are adaptive-only (D1). Budget env semantics on adaptive models are unverified.
Fix: pass `--effort` from the effort string (`--effort` exists in 2.1.263).

### PROV-29 (Med) -- `max_output_tokens=32000`, `context_window=200_000` hardcoded

**Disposition:** RESOLVED -- `max_output_tokens`/`context_window` (claude_code_provider.py:282) resolve the model or CLI alias through the shared table (`CLAUDE_CODE_ALIASES`, `_claude_models.py:85`, pinned from the live probe: `opus` = Opus 5, 128K/1M; `haiku` 64K/200K). Tests: `T::TestProv28Prov29ClaudeCodeThinkingAndLimits::test_limits_follow_the_model`; `tests/test_llm_providers.py` claude-code limit tests reframed.
Evidence: provider :353-361. Default model "opus" resolves to a 128K-output / 1M-context model per D1.

### PROV-30 (Low) -- `generate_batch` hardcodes `max_tokens=1024` / temperature 0.7

**Disposition:** RESOLVED -- `generate_batch` forwards only the `temperature`/`max_tokens`/`thinking` keys a request carries (:847). Test: `T::TestProv30Prov31ClaudeCodeMisc::test_batch_does_not_invent_max_tokens`.
Evidence: provider :895-896; both are then warned-and-dropped. Harmless but produces a spurious one-time "max_tokens ignored" warning. Pass through only when present.

### PROV-31 (Low) -- `count_tokens` constructs a new `AsyncAnthropic()` per call, never closed

**Disposition:** RESOLVED -- `count_tokens` uses `async with AsyncAnthropic()` (:883), closing the client, and resolves a CLI alias to its API model ID first (an alias like `opus` was always a 404 there). Tests: `T::TestProv30Prov31ClaudeCodeMisc::test_count_tokens_closes_its_client_and_resolves_the_alias`; `tests/test_llm_claude_code_generate.py::TestCountTokens` updated to the context-manager fake.
Evidence: provider :931. Leaks an httpx client per call when a key is set. Reuse or `async with`.

### PROV-32 (Low) -- `claude_code_sdk` pinned to 0.0.25-0.0.30 and monkey-patched

**Disposition:** WON'T FIX -- porting the four `claude_code_sdk` monkey-patches to the renamed `claude-agent-sdk` cannot be verified here: neither SDK is installed, and installing one into the shared environment is out of bounds for this pass. The CLI backend, which is what actually runs, now carries every fix in this file (PROV-26/27/28). The `--json-schema` flag in the NOT USED list is not adopted because the schema is JSON in argv and the CLI path's argv guard (claude_code_provider.py, `_UNSAFE_ARGV`) refuses `"` on Windows `.cmd` shims by design; `--max-budget-usd`/`--fallback-model` have no caller.
Evidence: provider :55-169. The package was superseded by `claude-agent-sdk` (renamed; unverified date). Not installed in this env, so CLI path is what runs.
NOT USED (CLI, all present in `claude --help` 2.1.263): `--json-schema` (structured output; would let `supports_json_mode` be True), `--max-budget-usd`, `--fallback-model`, `--effort`.
USED correctly: `--restricted`, `--permission-prompts none`, `--strict-mcp-config`, `--tools ""` last, `--no-session-persistence`, `--system-prompt-file` (all present in `--help`); result-event usage/cost/session_id capture (claude_code_cli.py:224-245); tool-use tripwire (:126-140); process-tree kill (:248-280).

---

## token_counter.py
### PROV-33 (Low) -- Default cl100k for unknown models

**Disposition:** NOT A DEFECT -- `count_tokens` is documented as an approximation used only for budgeting; the base class's proportional context reserve (`_CONTEXT_RESERVE_FRACTION = 0.30`) exists to absorb exactly this tokenizer error, and the providers with a native counter (Anthropic, Gemini) use it.
Evidence: token_counter.py:699-721 (dump) resolves per model, default cl100k_base. Correctly documented as approximation. No finding beyond noting Gemini/DeepSeek/xAI all fall to cl100k (`OpenAICompatibleProvider.count_tokens`, openai_compat.py:981-986) where DeepSeek/xAI tokenizers differ; only used for budgeting.

---

## Counts
High: PROV-1,2,3,10,11,16,20,26 = 8
Med: PROV-4,5,6,7,12,17,21,22,23,27,28,29 = 12
Low: PROV-8,9,13,14,15,18,19,24,25,30,31,32,33 = 13
Total: 33
