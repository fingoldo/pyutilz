# 10 - OpenRouter API coverage and correctness (2026-09-26)

Scope: `src/pyutilz/llm/openrouter_provider/{_provider,_catalogue}.py`, `openrouter_batch.py`, `openrouter_decisions.py`,
`_openai_compat_http.py`, `_pricing.py`, `_reasoning.py`, `_stream_attempts.py`, and the shared paths in `openai_compat.py`
that OpenRouterProvider inherits. This was a read-only audit.

Docs consulted on 2026-09-26:
- Streaming: https://openrouter.ai/docs/api/reference/streaming
- Provider routing: https://openrouter.ai/docs/guides/routing/provider-selection
- Prompt caching: https://openrouter.ai/docs/guides/best-practices/prompt-caching
- Usage accounting: https://openrouter.ai/docs/guides/guides/usage-accounting
- Reasoning: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
- Errors: https://openrouter.ai/docs/api/reference/errors-and-debugging

## Feature matrix

| Feature | Status | Evidence |
|---|---|---|
| SSE streaming and keep-alive comments | USED correctly | `openai_compat.py:436` skips non-`data:` lines, so `: OPENROUTER PROCESSING` is ignored as the streaming doc allows |
| Mid-stream error events | **USED with bug** (OR-1) | `openai_compat.py:589` |
| Pre-stream HTTP errors | USED correctly | `openai_compat.py:421-423`, repair in `_stream_attempts.py:58-76`, retry in `openai_compat.py:505-523` |
| A 200 response with an `error` body and no choices | USED correctly | `openai_compat.py:578-600`, `:872` |
| provider.order / ignore / sort / allow_fallbacks | USED correctly | `_provider.py:400-411` |
| provider.only / require_parameters / data_collection / zdr / quantizations / max_price / preferred_* | NOT USED (OR-2, OR-3) | `_provider.py:396-420` |
| `models` fallback array | USED, but estimates ignore the resolved model (OR-7) | `_provider.py:412-413`, `:602-604` |
| Top-level `cache_control` (automatic caching) | USED correctly | `_provider.py:418-419`; the caching doc confirms top-level automatic caching for Anthropic |
| Per-block `cache_control` breakpoints | NOT USED (OR-8) | `_messages.py:67-69` |
| Cached and cache-write token accounting | USED, but the estimate misprices writes (OR-6) | `_provider.py:555-565`, `openai_compat.py:964-966` |
| reasoning.effort / exclude / enabled | USED correctly | `_provider.py:487-511` |
| reasoning.max_tokens and effort `"none"` | NOT USED (OR-9) | `_provider.py:487-492` |
| Reasoning capture from `reasoning`, `reasoning_content` and `reasoning_details` | USED correctly | `_reasoning.py:78-97` |
| reasoning_details passthrough across turns | N/A: the API is single-turn and has no tool-result loop | `openai_compat.py` `generate(prompt, system, ...)` |
| response_format json_schema / json_object | USED; `strict` is not enforced (OR-10) | `openai_compat.py:300-320`, `_provider.py:327-381` |
| Tool calling (tools, tool_choice, parallel) | PARTIAL: replies are parsed, but requests cannot send tools (OR-11) | `openai_compat.py:880`, `_openai_compat_http.py:145-173` |
| usage.cost, cost_details, cache_discount, is_byok | USED (per-call inconsistency, OR-5) | `_provider.py:541-579`, `:624-629` |
| `usage.include` / `stream_options.include_usage` | Deprecated field sent under a false comment (OR-12) | `openai_compat.py:572-575` |
| /generation, /key, /credits, /models, /endpoints, /parameters | USED | `_provider.py:652-843`, `_catalogue.py:52-98`; caller `dev/attempt_archive.py:312,364` |
| transforms (middle-out) | NOT USED (OR-13) | no grep hit |
| Web plugin | USED; citations are lost when streaming (OR-4) | `_provider.py:414-417`, `:611-621` |
| file-parser (PDF) plugin, `:online`, web plugin options | NOT USED (OR-14) | `_provider.py:417` |
| Image input | USED | `openai_compat.py` `_messages_for` / `_messages.py` |
| Presets, `user`, seed, stop, logprobs, verbosity | NOT USED (OR-15) | the `generate`/`generate_stream` signatures (`openai_compat.py:360-370`) |
| HTTP-Referer / X-Title | USED correctly | `_provider.py:156-159` |
| Retry-After on 429 | Streaming only (OR-16) | `openai_compat.py:517`; the gap is self-documented at `_openai_compat_http.py:113-118` |
| Empty completion handling | USED | `openai_compat.py:779-808` (re-issue without response_format, diagnostics) |
| Batch API | USED (OR-17, OR-18) | `openrouter_batch.py` |
| Decisions (alpha) | USED (OR-19) | `openrouter_decisions.py` |

## Findings

### OR-1 (High) -- Mid-stream errors are swallowed, so a failed stream returns partial text as a success

**Disposition:** RESOLVED. Added `LLMStreamInterruptedError(LLMProviderError)`, which carries `code`, `partial_text` and `retryable`, in `_openai_compat_http.py:58`. `raise_for_error_in_body` (`_openai_compat_http.py:113`) now raises whenever `error` is a dict. When choices are also present (the documented mid-stream event), it raises `LLMStreamInterruptedError` with the text streamed so far. The stream path passes its buffer through (`openai_compat.py:391`), and a bare `finish_reason == "error"` raises in `_raise_for_stream_finish` (`openai_compat.py:460`). The buffered path raises on an error body even when choices are present, and on `finish_reason == "error"` (`openai_compat.py:747`). There it is retryable through `_is_retryable_http_error`. A stream that has generated is still never re-opened. Tests: T::test_or1_documented_mid_stream_error_chunk_raises_with_partial_text, which uses the doc's verbatim chunk and also asserts a single stream open; plus test_or1_bare_finish_reason_error_in_stream_raises, test_or1_buffered_finish_reason_error_raises_retryable and test_or1_buffered_error_with_choices_is_no_longer_ignored.

**Evidence.** The streaming doc says an error that occurs after the headers are sent arrives as a 200 SSE event, after which the stream terminates:

`{"error": {"code": "server_error", "message": ...}, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}`

`_raise_for_error_in_body` returns early at `openai_compat.py:589` whenever `payload.get("choices")` is truthy. That chunk has choices, so nothing is raised. `_apply_stream_chunk` (`:602`) then stores `_last_finish_reason = "error"`. `generate_stream` only acts on `"length"` (`:478`), so it returns normally.

Even without the early return, the string code `"server_error"` would map to 0 at `:592`, raising a plain non-retryable `LLMProviderError` (`:600`). The buffered path never checks `finish_reason == "error"` either (`:875`).

**Impact.** When a provider disconnects mid-answer, the caller gets truncated text that looks like a normal completion and has been billed. This is a silent correctness bug.

**Fix.**
- In `_raise_for_error_in_body`, raise whenever `error` is a dict, whether or not choices are present.
- In both paths, treat `finish_reason == "error"` as a failure. Raise an `LLMProviderError` subclass that carries `partial_text`, and map `server_error`-class codes to retryable. The retry must happen only before generation starts: the existing `generated_chunks` guard already blocks re-opening a stream after generation.
- Add a regression test that feeds the exact documented chunk.

### OR-2 (Med) -- `provider.require_parameters` is never sent, although the code already works around the failure it prevents

**Disposition:** RESOLVED. Added the `provider_require_parameters: bool | None = None` constructor kwarg. `True` and `False` mean always and never. `None` is automatic, decided per request in `_finalize_request_body` (`openrouter_provider/_request.py:217`) via `auto_require_parameters` (`_request.py:132`). Automatic mode sends `require_parameters: true` when both of these hold: - the body carries `response_format` or asks the model to reason (an effort, a budget, or `enabled: true`; not the disable forms); - the catalogue lists every parameter in the body for this model (`json_schema` maps to `structured_outputs`). Justification under the accurate-default rule: the first condition protects the answer from an endpoint that silently drops the schema or the reasoning request. The second is required for safety. `require_parameters` filters on EVERY parameter, so with an unconditional True an o-series model whose catalogue lacks `temperature` would have no endpoint left and 404. For the same reason, an unknown model gets no requirement. A new hook, `_finalize_request_body`, was added to `_openai_compat_body.py`; the default is a no-op. Tests: T::test_or2_auto_require_parameters_on_json_schema, test_or2_auto_require_parameters_on_reasoning_effort_but_not_on_disable, test_or2_auto_mode_skips_when_catalogue_lacks_a_sent_parameter and test_or2_explicit_settings_win.

**Evidence.** `_extra_request_body` (`_provider.py:396-420`) never sends it. The routing doc defines `require_parameters` as: only route to providers that support all parameters in the request. Instead, the code repairs after the fact:
- endpoints that ignore `response_format`: `openai_compat.py:797-808`, the glm-4.7-flash incident where 58 of 60 answers were lost;
- endpoints that refuse `reasoning.enabled:false`: `_provider.py:494-511`.

**Impact.** A call that uses json_schema, json_mode or reasoning can land on an endpoint that drops the parameter. The result is either a second paid POST or a degraded answer.

**Fix.** Add a `provider_require_parameters` knob. Default it to True when a json_schema is sent, because that is when `last_json_schema_applied` claims a guarantee. It depends on the body, so merge it at body-build time.

### OR-3 (Med) -- Several routing knobs are missing: only, data_collection, zdr, quantizations, max_price, preferred_min_throughput / preferred_max_latency

**Disposition:** RESOLVED. Added these kwargs: - `provider_only`, `provider_data_collection` (validated `"allow"`/`"deny"` at construction), `provider_zdr` and `provider_quantizations`; - `provider_max_price`, `provider_preferred_min_throughput` and `provider_preferred_max_latency`, each taking a number, a mapping or a tuple of pairs. All of them are built once by `build_provider_field` (`_request.py:45`) and sent only when set. Tests: T::test_or3_all_routing_knobs_reach_the_provider_block (asserts the exact `provider` dict and that the kwargs stay hashable) and test_or3_bad_data_collection_fails_at_construction.

**Evidence.** The constructor (`_provider.py:132-149`) and `:399-411` do not handle them. All of them are listed in the provider-selection doc. `openrouter_batch.py:137` already sends `only`.

**Impact.** Callers cannot:
- pin a quantization (int4 versus fp8 changes output quality);
- cap the price per call;
- restrict private data to ZDR or no-training providers;
- allow-list providers without also fixing their order.

**Fix.** Add hashable kwargs (tuples, floats, and a sorted tuple of pairs for `max_price`) so the factory cache key keeps working, and forward them in `_extra_request_body`.

### OR-4 (Low) -- Web-search citations are lost on the streaming path

**Disposition:** RESOLVED. `OpenRouterRequestMixin._apply_stream_chunk` (`_request.py:231`) reads `delta.annotations` and accumulates the url_citations into `last_web_search_citations`. The buffered path shares the parser, `url_citations`. Test: T::test_or4_citations_from_stream_delta_annotations_accumulate.

**Evidence.** The OR hook reads `choices[0].message.annotations` (`_provider.py:611-621`). Stream chunks carry `delta`, not `message`. `_apply_stream_chunk` reads only the top-level `chunk["citations"]` (`openai_compat.py:616`).

**Impact.** A streamed call with `enable_web_search=True` leaves `last_web_search_citations` empty.

**Fix.** Also read `delta.annotations` and accumulate it across chunks.

### OR-5 (Low) -- Most per-call OR fields are overwritten, not summed, when one call makes a second POST

**Disposition:** RESOLVED. In `_track_provider_specific_usage` (`openrouter_provider/_provider.py:607`), these per-call fields are now summed and never reset by a POST that lacks them: `last_upstream_inference_cost_usd`, `last_cache_write_tokens`, `last_cache_hit_tokens`, `last_audio_tokens` and `last_cache_discount_usd`. Test: T::test_or5_second_post_adds_to_and_never_erases_per_call_fields. The stale assertion in tests/test_llm_openrouter.py::TestExtendedUsageCapture::test_upstream_inference_cost_accumulates was re-framed to the summed value.

**Evidence.** `last_actual_cost_usd` is summed (`_provider.py:541-545`), but these are assigned:
- `last_upstream_inference_cost_usd` (`:547-553`)
- `last_cache_write_tokens` (`:556-557`)
- `last_cache_hit_tokens` (`:565`)
- `last_audio_tokens` (`:567-568`)
- `last_cache_discount_usd` (`:574-579`)

A second POST that lacks a field resets that field to None or 0.

**Impact.** After a re-issue, the per-call breakdown disagrees with `last_actual_cost_usd` and with `_last_usage`, which is summed (`openai_compat.py:671-678`). Session totals are still correct.

**Fix.** Sum these the same way `last_actual_cost_usd` is summed, and never overwrite a stored value with None.

### OR-6 (Low) -- The session cost estimate prices cache writes at the plain input rate

**Disposition:** RESOLVED inside the OpenRouter files, with an optional HANDOFF to CORE. - The field name was verified against a live `/models` listing: `pricing.input_cache_write` (1.25x input) and `input_cache_write_1h` (2x). - Added `_cache_write_cost_per_1m_or_none` (`openrouter_provider/_catalogue.py:175`). It uses the 5-minute rate, which is the only TTL this package requests. - `OpenRouterAccountingMixin._estimate_by_served_model` (`openrouter_provider/_accounting.py:47`) carves cache writes out of the miss count and prices them at the write rate. `get_session_cost` uses it (`_provider.py:920`). - Test: T::test_or6_cache_writes_priced_at_the_write_rate, which expects $2.50, not $2.00, for 1M write tokens. HANDOFF to CORE (optional generalisation, not needed for OpenRouter): - Add `cache_write: float | None = None` as a fourth field of `Pricing` in `_pricing.py`. - In `OpenAICompatibleProvider.get_session_cost`, compute `miss = prompt - hit - total_cache_write_tokens` when the provider tracks writes (use `getattr(self, "total_cache_write_tokens", 0)`), and price writes at `pricing.cache_write`, falling back to the input rate. - After that, OpenRouter's `_resolve_pricing` can pass `_cache_write_cost_per_1m_or_none(model)`, and `_estimate_by_served_model` can read it from `Pricing`.

**Evidence.** `get_session_cost` (`openai_compat.py:964-966`) splits prompt tokens only into cache hits and misses. `_resolve_pricing` (`_provider.py:649-650`) and `Pricing` (`_pricing.py:22-28`) have no cache-write rate. The caching doc gives Anthropic's write price as 1.25x input for the 5-minute TTL and 2x for the 1-hour TTL.

**Impact.** The estimate `total_cost_usd` is too low for Anthropic caching workloads. `actual_cost_usd` is not affected.

**Fix.**
- Add an optional `cache_write` field to `Pricing`.
- Read the catalogue's `pricing.input_cache_write` field. This field name is inferred by analogy with the `input_cache_read` field the code already reads; verify it against a live `/models` entry.
- Price `total_cache_write_tokens` at that rate.

### OR-7 (Low) -- Pricing, limits and capability checks use the requested model, even when `models` fallback served another one

**Disposition:** RESOLVED. - **Pricing.** The buffered path now records response metadata before usage (`openai_compat.py`, `_post_and_unwrap`), so the served model is known when usage arrives. The stream path already tracked it at the first chunk. The usage hook tallies tokens per served model, and `get_session_cost` prices each tally at that model's catalogue rates. - **Unlisted snapshot ids.** A served id the catalogue does not list (a dated snapshot) is priced as the requested model (`_accounting.py` `_pricing_model`). - **Unattributed tokens.** Tokens outside the tally (counters set directly) are attributed to the requested model, so the totals still reconcile. - **Limits.** `max_output_tokens` and `context_window` now return the minimum known limit across `model` + `models_fallback` (`_provider.py:341`). - Tests: T::test_or7_usage_priced_by_the_model_that_served_it, test_or7_buffered_path_reads_the_served_model_before_recording_usage, test_or7_unlisted_snapshot_id_prices_as_the_requested_model and test_or7_limits_are_the_minimum_across_the_fallback_list. - Related fix: catalogue lookups strip the routing suffixes `:online`, `:nitro` and `:floor` (`catalogue_id`, `_request.py:28`). `:free` is a real catalogue id and is kept. Test: T::test_online_suffix_uses_the_base_catalogue_entry.

**Evidence.** These all use `self.model_name`: `_provider.py:273`, `:307`, `:349`, `:374`, `:649`. `last_upstream_model` is recorded (`:602-604`) but never used.

**Impact.**
- The estimate applies the wrong model's rates.
- `max_tokens` is sized to the primary model's cap. If a fallback model has a smaller cap, the request can fail with a 400.

**Fix.**
- Price usage by the resolved model.
- When a fallback list is set, size `max_tokens` to the smallest cap across the list.

### OR-8 (Low) -- Per-block `cache_control` breakpoints are not available

**Disposition:** RESOLVED. Added a `system_cache_control: bool = False` kwarg. `OpenRouterRequestMixin._build_messages` sends the system prompt as a text part carrying `cache_control: {"type": "ephemeral"}` (`system_with_cache_control`, `_request.py:160`). `_messages.py` is unchanged. Test: T::test_or8_system_prompt_as_cache_breakpoint.

**Evidence.** `_messages.py:67-69` sends content as a plain string. The only caching lever is the top-level flag (`_provider.py:418-419`). Per the caching doc, Gemini uses per-block breakpoints.

**Fix.** Add an option to send the system prompt as content parts with `cache_control: {"type": "ephemeral"}`.

### OR-9 (Low) -- `reasoning.max_tokens` is never sent

**Disposition:** RESOLVED in two parts. **1. `reasoning.max_tokens`.** A positive int `thinking` now sends `{"reasoning": {"max_tokens": n}}` (`_provider.py`, `_thinking_request_field`). The reasoning doc confirms that effort and max_tokens are exclusive. Test: T::test_or9_int_thinking_is_a_reasoning_max_tokens_budget. **2. `effort: "none"` vs `minimal` as the fallback.** Measured live on 2026-09-26 with user-approved paid calls. - **Setup:** prompt "Reply with the single word: yes", `max_tokens` 600. Raw results are in `audits/2026-09-26/or9_live_results.json` (24 calls) and `or9_live_results_repeat.json` (18 calls). Total spend was $0.0012. - **Models:** - five models the catalogue marks `reasoning.mandatory: true`: openai/gpt-oss-20b, openai/gpt-5-nano, z-ai/glm-5.3-flash, stepfun/step-3.5-flash and google/gemini-3.7-flash; - one control, deepseek/deepseek-v4.1-flash, which is `mandatory: false`. - **Results:** - `effort: "none"` AND `enabled: false` were refused with HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled." on all 5 mandatory models, 10 of 10 calls. So `"none"` is NOT a usable off switch or fallback. - `effort: "minimal"` returned 200 on all 5. Reasoning tokens were 0 to 121: gpt-5-nano 0, gemini 0/0/96/0, glm 0/0/7/0, gpt-oss-20b 5 to 23, step-3.5-flash 108 to 121. - Adding `exclude: true` made no measurable difference to billed reasoning tokens. Within one provider it was noise both ways; for example gemini on Google read 0/96/0 without and 89/0/70/0 with. - The control accepted all four forms. Both `enabled: false` and `effort: "none"` billed 0 reasoning tokens there. - **Conclusion:** the data supports the existing fallback (`effort: "minimal"`, with `exclude` kept only to hide the text), not `"none"`. **Implemented from that data:** - `_catalogue_says_reasoning_mandatory` (`_provider.py`). A model the catalogue marks mandatory now gets the minimal form UP FRONT. Before this, it paid one refused 400 per model per process. The live catalogue flags all five refusing models, and the refusal-keyed per-process set stays as the backstop for entries without the flag. - `_body_after_rejected_request` now also repairs a refused `effort: "none"`, for example one sent through `extra_body`, because it is refused with the identical message. Tests (using the recorded refusal text and catalogue shape): - T::test_or9_catalogue_mandatory_model_gets_minimal_effort_up_front, which also covers the control; - T::test_or9_refused_off_switch_is_repaired_to_minimal, parametrised over `enabled: false` and `effort: "none"`; - T::test_or9_recorded_live_results_support_the_fallback, which asserts against the saved JSON that exactly the two off forms were refused on every mandatory model.

**Evidence.** `_provider.py:487-492` only sends effort or `enabled:false`. The reasoning doc offers `max_tokens` as an alternative to effort; the two cannot be combined. The docstring at `_provider.py:478-480` records `medium` effort consuming the whole budget on deepseek-v4.1-flash, which is exactly the case a reasoning budget caps. The doc also lists `"none"` as an effort value.

**Fix.**
- When `thinking` is a positive int, send `{"reasoning": {"max_tokens": n}}`.
- Measure `effort: "none"` against `minimal` as the fallback for endpoints where reasoning cannot be disabled.

### OR-10 (Low) -- The json_schema wrapper does not force `strict`

**Disposition:** RESOLVED. `_response_format` (`_openai_compat_body.py:64`) sends a copy of the schema with `strict` defaulting to True. The caller's dict is not mutated. `last_json_schema_applied` is True only when `strict is True`, so an explicit `strict: False` is reported as not applied. Test: T::test_or10_strict_defaults_true_and_explicit_false_is_reported.

**Evidence.** `openai_compat.py:310-312` sends `{"type": "json_schema", "json_schema": json_schema}` as given and sets `last_json_schema_applied = True` whether or not `strict` is present.

**Impact.** A caller who omits `"strict": true` is told the schema was guaranteed when it was not enforced.

**Fix.** Call `setdefault("strict", True)` on a copy of the schema, or set the flag only when `strict` is present.

### OR-11 (Low) -- Tool calling is response-only

**Disposition:** RESOLVED. `generate()` and `generate_stream()` on every OpenAI-compatible provider accept `tools=` and `tool_choice=` (`_request_body`, `_openai_compat_body.py:117`). `parallel_tool_calls` goes through `extra_body`. The `reasoning_details` echo is N/A: the API is single-turn, with no tool-result loop. Test: T::test_or11_tools_are_sent_and_tool_calls_read_back, which covers the body sent, the `""` return and `last_tool_calls`.

**Evidence.** Tool calls in responses are parsed (`openai_compat.py:880`, `_openai_compat_http.py:145-173`), and streamed ones are reassembled. There is no parameter for `tools`, `tool_choice` or `parallel_tool_calls` (a grep of `src/pyutilz/llm` finds none), so that parsing never runs in normal use.

**Fix.** Either add passthrough for these fields (plus a `reasoning_details` echo for multi-turn), or document that tool calling is out of scope.

### OR-12 (Low) -- `stream_options.include_usage` is deprecated on OpenRouter, but a comment says it is required

**Disposition:** RESOLVED. The field is kept for the other OpenAI-compatible providers, and the comment now says OpenRouter documents it as deprecated with no effect (`_openai_compat_body.py:188`). Test: T::test_or12_stream_options_still_sent.

**Evidence.** The comment at `openai_compat.py:572-575` says the stream never publishes usage without this field. The usage-accounting doc says the field is "deprecated and have no effect", and that usage is always included in the final chunk.

**Fix.** Keep the field, since the other OpenAI-compatible providers still need it, and correct the comment for OpenRouter.

### OR-13 (Low) -- `transforms: ["middle-out"]` is not available

**Disposition:** RESOLVED. Added an opt-in `transforms: tuple[str, ...] | None` kwarg, for example `("middle-out",)`. The README and the code comment state that it is lossy. Test: T::test_or13_14_transforms_and_plugins.

**Evidence.** No grep hit. The code relies instead on a tokenizer clamp that has undercounted by about 12% (the comment in `openai_compat.py` `generate`).

**Fix.** Offer it as opt-in only; it discards content, so it is lossy.

### OR-14 (Low) -- The file-parser (PDF) plugin, `:online`, and web plugin options (`max_results`, `engine`) are not available

**Disposition:** RESOLVED. Added these kwargs: - `web_search_engine` and `web_search_max_results`, sent as options on the web plugin; - `pdf_engine` (`"mistral-ocr"`, `"cloudflare-ai"` or `"native"`), which attaches `{"id": "file-parser", "pdf": {"engine": ...}}`. Built by `build_plugins` (`_request.py:96`). `:online` works as a model suffix, and catalogue lookups now strip it (see OR-7). Test: T::test_or13_14_transforms_and_plugins.

**Evidence.** Only the bare `{"id": "web"}` plugin is sent (`_provider.py:417`).

**Fix.** Add these when a caller needs them.

### OR-15 (Low) -- seed, stop, logprobs/top_logprobs, verbosity, `user` and presets are not available

**Disposition:** RESOLVED. A generic `extra_body` is available in two places: - per call on `generate`/`generate_stream`; - on the OpenRouter constructor, as a mapping or a tuple of pairs. It is merged last by `merge_extra_body` (`_openai_compat_http.py:145`). A dict value is merged one level into a dict already in the body, so the constructor's `provider` block survives. This covers seed, stop, logprobs/top_logprobs, verbosity and user. Test: T::test_or15_extra_body_constructor_and_per_call_merge_last.

**Evidence.** The signature at `openai_compat.py:360-370` has none of these. `seed` matters for reproducible benchmarks, and `user` lets OpenRouter attribute usage per end user.

**Fix.** Add one generic `extra_body` passthrough, merged last, rather than a kwarg for each field.

### OR-16 (Low) -- The buffered path ignores Retry-After on 429

**Disposition:** RESOLVED. `wait_honoring_retry_after` (`_openai_compat_http.py:162`) wraps the shared wait in the buffered `generate` retry decorator (`openai_compat.py:577`). On a 429 only, the delay is `max(exponential, Retry-After)`. Delays are never shortened, and other statuses are untouched. `_retry.py` is not edited. Tests: T::test_or16_buffered_wait_takes_the_longer_retry_after_on_429_only and test_or16_generate_retry_decorator_uses_the_retry_after_wait.

**Evidence.** The gap is documented in the code at `_openai_compat_http.py:113-118`. The errors doc says clients should respect `Retry-After` on 429. Only the streaming path uses it (`openai_compat.py:517`).

**Fix.** Use a tenacity wait that takes the larger of `parse_retry_after(exc.response)` and the exponential wait, scoped to 429.

### OR-17 (Low) -- Batch recovery matches on `metadata.request_hash`, which submit never sends

**Disposition:** RESOLVED by removing the branch. The batch docs, checked 2026-09-26, list only `endpoint`, `model`, `requests`, `provider` and `completion_window`; there is no `metadata`. Sending it could get the submit rejected, so the `metadata.request_hash` exact-match branch was removed from `recover_submit`, and the docstring records why (`openrouter_batch.py:347`). Tests: - T::test_or17_listed_metadata_hash_is_not_trusted; - tests/llm/test_openrouter_batch_idempotence.py::test_a_listed_metadata_hash_does_not_decide, re-framed from the old test that trusted the hash.

**Evidence.** `openrouter_batch.py:353-358` reads `metadata.request_hash`, but `build_submit_payload` (`:135-147`) never sets `metadata`.

**Impact.** The exact-match branch never fires, so recovery always falls back to the time-window heuristic.

**Fix.** Send `metadata: {"request_hash": digest}`. This is not verified against the batch docs (https://openrouter.ai/docs/batch-quickstart): confirm that `metadata` is accepted before implementing. If it is not, remove the branch.

### OR-18 (Low) -- Batch result parsing reads only `message.reasoning` and treats truncated results as successful

**Disposition:** RESOLVED. - `parse_result_item` reads reasoning through `_reasoning.from_message` (`openrouter_batch.py:236`). - `BatchResult.truncated` was added (`openrouter_batch.py:99`), and `ok` is now False for a truncated result; the text is kept. Tests: T::test_or18_batch_reasoning_from_any_field_and_truncation_not_ok, plus the re-framed assertion in tests/test_llm_openrouter_batch.py::test_results_map_back_by_custom_id_not_position.

**Evidence.**
- `openrouter_batch.py:229` reads only `message.reasoning`. `_reasoning.from_message` also covers `reasoning_content` and `reasoning_details`.
- A result with `finish_reason == "length"` and partial text reports `ok == True` (`:97-100`, `:230-232`).

**Fix.** Use `_reasoning.from_message`, and mark truncated results.

### OR-19 (Low) -- The decisions client does not retry 408 or 504 and ignores Retry-After

**Disposition:** RESOLVED. `RETRYABLE_STATUSES` now includes 408 and 504 (`openrouter_decisions.py:49`). `OpenRouterDecisionsError.retry_after_s` is filled from the response headers. `_wait_honoring_retry_after` (`openrouter_decisions.py:326`) wraps the retry wait, whether the default or a caller-supplied one. Tests: T::test_or19_decisions_retry_408_and_504 and test_or19_decisions_wait_honours_retry_after.

**Evidence.** `openrouter_decisions.py:46` defines `RETRYABLE_STATUSES = {429, 500, 502, 503, 524, 529}`. The errors doc defines 408 as a request timeout, and 429 as a signal to respect `Retry-After`. The retry setup at `:362-368` uses a fixed wait.

**Fix.** Add 408 and 504 to the set, and honor Retry-After.

### OR-20 (Low) -- A 200 body with a non-numeric error code is never retried

**Disposition:** RESOLVED. `error_code_status` (`_openai_compat_http.py:94`) maps numeric codes and known transient string codes to their HTTP status: - `server_error` and similar go to 502; - overload codes go to 503; - `timeout` goes to 504; - rate-limit codes go to 429. A 200 body carrying such a code raises a retryable `httpx.HTTPStatusError`. Unknown strings still raise a plain, non-retryable `LLMProviderError`. Tests: T::test_or20_error_code_status (parametrised), test_or20_string_code_in_200_body_is_retryable_http_error and test_or20_unknown_string_code_is_not_retried.

**Evidence.** `openai_compat.py:592` maps a non-digit code to 0, which leads to a plain `LLMProviderError` (`:600`). `_is_retryable_http_error` (`_openai_compat_http.py:58-71`) does not retry that. The streaming doc shows OpenRouter using string codes such as `server_error`.

**Fix.** Map known transient string codes to 502 or 503, together with the OR-1 fix.

## Checked and correct

- The 402 grace window (`_provider.py:513-521`).
- The 404/405 routing retry is opt-in (`_provider.py:216-255`).
- Catalogue per-token prices are converted to per-1M by multiplying by 1e6 (`_catalogue.py:146-147`, `:170`).
- Cache hits are not double counted (`_provider.py:561-565` together with `openai_compat.py:656`).
- Usage is recorded once per stream (`openai_compat.py:475-477`).
- A stream is never re-opened after generation has started (`openai_compat.py:494-506`).
- Aborted stream attempts keep their generation id, so they can be reconciled through `/generation` (`_stream_attempts.py:42-55`).
- Batch submission is idempotent, and resuming a stored job checks the request hash.

## Counts

- High: 1 (OR-1)
- Med: 2 (OR-2, OR-3)
- Low: 17 (OR-4 to OR-20)
