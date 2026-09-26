# Dispositions: 10_openrouter_coverage.md (OR-1 to OR-20)

Tests: `tests/llm/test_openrouter_coverage_20260926.py` (T below) unless another file is named. Paths are relative to
`src/pyutilz/llm/`. OpenRouter docs re-checked on 2026-09-26 for provider selection, reasoning, batches, PDFs, the web
plugin and the `/models` pricing keys.

### OR-1
**Disposition:** RESOLVED. Added `LLMStreamInterruptedError(LLMProviderError)`, which carries `code`, `partial_text` and `retryable`, in `_openai_compat_http.py:58`. `raise_for_error_in_body` (`_openai_compat_http.py:113`) now raises whenever `error` is a dict. When choices are also present (the documented mid-stream event), it raises `LLMStreamInterruptedError` with the text streamed so far. The stream path passes its buffer through (`openai_compat.py:391`), and a bare `finish_reason == "error"` raises in `_raise_for_stream_finish` (`openai_compat.py:460`). The buffered path raises on an error body even when choices are present, and on `finish_reason == "error"` (`openai_compat.py:747`). There it is retryable through `_is_retryable_http_error`. A stream that has generated is still never re-opened. Tests: T::test_or1_documented_mid_stream_error_chunk_raises_with_partial_text, which uses the doc's verbatim chunk and also asserts a single stream open; plus test_or1_bare_finish_reason_error_in_stream_raises, test_or1_buffered_finish_reason_error_raises_retryable and test_or1_buffered_error_with_choices_is_no_longer_ignored.

### OR-2
**Disposition:** RESOLVED. Added the `provider_require_parameters: bool | None = None` constructor kwarg. `True` and `False` mean always and never. `None` is automatic, decided per request in `_finalize_request_body` (`openrouter_provider/_request.py:217`) via `auto_require_parameters` (`_request.py:132`). Automatic mode sends `require_parameters: true` when both of these hold:
- the body carries `response_format` or asks the model to reason (an effort, a budget, or `enabled: true`; not the disable forms);
- the catalogue lists every parameter in the body for this model (`json_schema` maps to `structured_outputs`).

Justification under the accurate-default rule: the first condition protects the answer from an endpoint that silently drops the schema or the reasoning request. The second is required for safety. `require_parameters` filters on EVERY parameter, so with an unconditional True an o-series model whose catalogue lacks `temperature` would have no endpoint left and 404. For the same reason, an unknown model gets no requirement. A new hook, `_finalize_request_body`, was added to `_openai_compat_body.py`; the default is a no-op. Tests: T::test_or2_auto_require_parameters_on_json_schema, test_or2_auto_require_parameters_on_reasoning_effort_but_not_on_disable, test_or2_auto_mode_skips_when_catalogue_lacks_a_sent_parameter and test_or2_explicit_settings_win.

### OR-3
**Disposition:** RESOLVED. Added these kwargs:
- `provider_only`, `provider_data_collection` (validated `"allow"`/`"deny"` at construction), `provider_zdr` and `provider_quantizations`;
- `provider_max_price`, `provider_preferred_min_throughput` and `provider_preferred_max_latency`, each taking a number, a mapping or a tuple of pairs.

All of them are built once by `build_provider_field` (`_request.py:45`) and sent only when set. Tests: T::test_or3_all_routing_knobs_reach_the_provider_block (asserts the exact `provider` dict and that the kwargs stay hashable) and test_or3_bad_data_collection_fails_at_construction.

### OR-4
**Disposition:** RESOLVED. `OpenRouterRequestMixin._apply_stream_chunk` (`_request.py:231`) reads `delta.annotations` and accumulates the url_citations into `last_web_search_citations`. The buffered path shares the parser, `url_citations`. Test: T::test_or4_citations_from_stream_delta_annotations_accumulate.

### OR-5
**Disposition:** RESOLVED. In `_track_provider_specific_usage` (`openrouter_provider/_provider.py:607`), these per-call fields are now summed and never reset by a POST that lacks them: `last_upstream_inference_cost_usd`, `last_cache_write_tokens`, `last_cache_hit_tokens`, `last_audio_tokens` and `last_cache_discount_usd`. Test: T::test_or5_second_post_adds_to_and_never_erases_per_call_fields. The stale assertion in tests/test_llm_openrouter.py::TestExtendedUsageCapture::test_upstream_inference_cost_accumulates was re-framed to the summed value.

### OR-6
**Disposition:** RESOLVED inside the OpenRouter files, with an optional HANDOFF to CORE.
- The field name was verified against a live `/models` listing: `pricing.input_cache_write` (1.25x input) and `input_cache_write_1h` (2x).
- Added `_cache_write_cost_per_1m_or_none` (`openrouter_provider/_catalogue.py:175`). It uses the 5-minute rate, which is the only TTL this package requests.
- `OpenRouterAccountingMixin._estimate_by_served_model` (`openrouter_provider/_accounting.py:47`) carves cache writes out of the miss count and prices them at the write rate. `get_session_cost` uses it (`_provider.py:920`).
- Test: T::test_or6_cache_writes_priced_at_the_write_rate, which expects $2.50, not $2.00, for 1M write tokens.

HANDOFF to CORE (optional generalisation, not needed for OpenRouter):
- Add `cache_write: float | None = None` as a fourth field of `Pricing` in `_pricing.py`.
- In `OpenAICompatibleProvider.get_session_cost`, compute `miss = prompt - hit - total_cache_write_tokens` when the provider tracks writes (use `getattr(self, "total_cache_write_tokens", 0)`), and price writes at `pricing.cache_write`, falling back to the input rate.
- After that, OpenRouter's `_resolve_pricing` can pass `_cache_write_cost_per_1m_or_none(model)`, and `_estimate_by_served_model` can read it from `Pricing`.

### OR-7
**Disposition:** RESOLVED.
- **Pricing.** The buffered path now records response metadata before usage (`openai_compat.py`, `_post_and_unwrap`), so the served model is known when usage arrives. The stream path already tracked it at the first chunk. The usage hook tallies tokens per served model, and `get_session_cost` prices each tally at that model's catalogue rates.
- **Unlisted snapshot ids.** A served id the catalogue does not list (a dated snapshot) is priced as the requested model (`_accounting.py` `_pricing_model`).
- **Unattributed tokens.** Tokens outside the tally (counters set directly) are attributed to the requested model, so the totals still reconcile.
- **Limits.** `max_output_tokens` and `context_window` now return the minimum known limit across `model` + `models_fallback` (`_provider.py:341`).
- Tests: T::test_or7_usage_priced_by_the_model_that_served_it, test_or7_buffered_path_reads_the_served_model_before_recording_usage, test_or7_unlisted_snapshot_id_prices_as_the_requested_model and test_or7_limits_are_the_minimum_across_the_fallback_list.
- Related fix: catalogue lookups strip the routing suffixes `:online`, `:nitro` and `:floor` (`catalogue_id`, `_request.py:28`). `:free` is a real catalogue id and is kept. Test: T::test_online_suffix_uses_the_base_catalogue_entry.

### OR-8
**Disposition:** RESOLVED. Added a `system_cache_control: bool = False` kwarg. `OpenRouterRequestMixin._build_messages` sends the system prompt as a text part carrying `cache_control: {"type": "ephemeral"}` (`system_with_cache_control`, `_request.py:160`). `_messages.py` is unchanged. Test: T::test_or8_system_prompt_as_cache_breakpoint.

### OR-9
**Disposition:** RESOLVED in two parts.

**1. `reasoning.max_tokens`.** A positive int `thinking` now sends `{"reasoning": {"max_tokens": n}}` (`_provider.py`, `_thinking_request_field`). The reasoning doc confirms that effort and max_tokens are exclusive. Test: T::test_or9_int_thinking_is_a_reasoning_max_tokens_budget.

**2. `effort: "none"` vs `minimal` as the fallback.** Measured live on 2026-09-26 with user-approved paid calls.
- **Setup:** prompt "Reply with the single word: yes", `max_tokens` 600. Raw results are in `audits/2026-09-26/or9_live_results.json` (24 calls) and `or9_live_results_repeat.json` (18 calls). Total spend was $0.0012.
- **Models:**
  - five models the catalogue marks `reasoning.mandatory: true`: openai/gpt-oss-20b, openai/gpt-5-nano, z-ai/glm-5.3-flash, stepfun/step-3.5-flash and google/gemini-3.7-flash;
  - one control, deepseek/deepseek-v4.1-flash, which is `mandatory: false`.
- **Results:**
  - `effort: "none"` AND `enabled: false` were refused with HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled." on all 5 mandatory models, 10 of 10 calls. So `"none"` is NOT a usable off switch or fallback.
  - `effort: "minimal"` returned 200 on all 5. Reasoning tokens were 0 to 121: gpt-5-nano 0, gemini 0/0/96/0, glm 0/0/7/0, gpt-oss-20b 5 to 23, step-3.5-flash 108 to 121.
  - Adding `exclude: true` made no measurable difference to billed reasoning tokens. Within one provider it was noise both ways; for example gemini on Google read 0/96/0 without and 89/0/70/0 with.
  - The control accepted all four forms. Both `enabled: false` and `effort: "none"` billed 0 reasoning tokens there.
- **Conclusion:** the data supports the existing fallback (`effort: "minimal"`, with `exclude` kept only to hide the text), not `"none"`.

**Implemented from that data:**
- `_catalogue_says_reasoning_mandatory` (`_provider.py`). A model the catalogue marks mandatory now gets the minimal form UP FRONT. Before this, it paid one refused 400 per model per process. The live catalogue flags all five refusing models, and the refusal-keyed per-process set stays as the backstop for entries without the flag.
- `_body_after_rejected_request` now also repairs a refused `effort: "none"`, for example one sent through `extra_body`, because it is refused with the identical message.

Tests (using the recorded refusal text and catalogue shape):
- T::test_or9_catalogue_mandatory_model_gets_minimal_effort_up_front, which also covers the control;
- T::test_or9_refused_off_switch_is_repaired_to_minimal, parametrised over `enabled: false` and `effort: "none"`;
- T::test_or9_recorded_live_results_support_the_fallback, which asserts against the saved JSON that exactly the two off forms were refused on every mandatory model.

### OR-10
**Disposition:** RESOLVED. `_response_format` (`_openai_compat_body.py:64`) sends a copy of the schema with `strict` defaulting to True. The caller's dict is not mutated. `last_json_schema_applied` is True only when `strict is True`, so an explicit `strict: False` is reported as not applied. Test: T::test_or10_strict_defaults_true_and_explicit_false_is_reported.

### OR-11
**Disposition:** RESOLVED. `generate()` and `generate_stream()` on every OpenAI-compatible provider accept `tools=` and `tool_choice=` (`_request_body`, `_openai_compat_body.py:117`). `parallel_tool_calls` goes through `extra_body`. The `reasoning_details` echo is N/A: the API is single-turn, with no tool-result loop. Test: T::test_or11_tools_are_sent_and_tool_calls_read_back, which covers the body sent, the `""` return and `last_tool_calls`.

### OR-12
**Disposition:** RESOLVED. The field is kept for the other OpenAI-compatible providers, and the comment now says OpenRouter documents it as deprecated with no effect (`_openai_compat_body.py:188`). Test: T::test_or12_stream_options_still_sent.

### OR-13
**Disposition:** RESOLVED. Added an opt-in `transforms: tuple[str, ...] | None` kwarg, for example `("middle-out",)`. The README and the code comment state that it is lossy. Test: T::test_or13_14_transforms_and_plugins.

### OR-14
**Disposition:** RESOLVED. Added these kwargs:
- `web_search_engine` and `web_search_max_results`, sent as options on the web plugin;
- `pdf_engine` (`"mistral-ocr"`, `"cloudflare-ai"` or `"native"`), which attaches `{"id": "file-parser", "pdf": {"engine": ...}}`. Built by `build_plugins` (`_request.py:96`).

`:online` works as a model suffix, and catalogue lookups now strip it (see OR-7). Test: T::test_or13_14_transforms_and_plugins.

### OR-15
**Disposition:** RESOLVED. A generic `extra_body` is available in two places:
- per call on `generate`/`generate_stream`;
- on the OpenRouter constructor, as a mapping or a tuple of pairs.

It is merged last by `merge_extra_body` (`_openai_compat_http.py:145`). A dict value is merged one level into a dict already in the body, so the constructor's `provider` block survives. This covers seed, stop, logprobs/top_logprobs, verbosity and user. Test: T::test_or15_extra_body_constructor_and_per_call_merge_last.

### OR-16
**Disposition:** RESOLVED. `wait_honoring_retry_after` (`_openai_compat_http.py:162`) wraps the shared wait in the buffered `generate` retry decorator (`openai_compat.py:577`). On a 429 only, the delay is `max(exponential, Retry-After)`. Delays are never shortened, and other statuses are untouched. `_retry.py` is not edited. Tests: T::test_or16_buffered_wait_takes_the_longer_retry_after_on_429_only and test_or16_generate_retry_decorator_uses_the_retry_after_wait.

### OR-17
**Disposition:** RESOLVED by removing the branch. The batch docs, checked 2026-09-26, list only `endpoint`, `model`, `requests`, `provider` and `completion_window`; there is no `metadata`. Sending it could get the submit rejected, so the `metadata.request_hash` exact-match branch was removed from `recover_submit`, and the docstring records why (`openrouter_batch.py:347`). Tests:
- T::test_or17_listed_metadata_hash_is_not_trusted;
- tests/llm/test_openrouter_batch_idempotence.py::test_a_listed_metadata_hash_does_not_decide, re-framed from the old test that trusted the hash.

### OR-18
**Disposition:** RESOLVED.
- `parse_result_item` reads reasoning through `_reasoning.from_message` (`openrouter_batch.py:236`).
- `BatchResult.truncated` was added (`openrouter_batch.py:99`), and `ok` is now False for a truncated result; the text is kept.

Tests: T::test_or18_batch_reasoning_from_any_field_and_truncation_not_ok, plus the re-framed assertion in tests/test_llm_openrouter_batch.py::test_results_map_back_by_custom_id_not_position.

### OR-19
**Disposition:** RESOLVED. `RETRYABLE_STATUSES` now includes 408 and 504 (`openrouter_decisions.py:49`). `OpenRouterDecisionsError.retry_after_s` is filled from the response headers. `_wait_honoring_retry_after` (`openrouter_decisions.py:326`) wraps the retry wait, whether the default or a caller-supplied one. Tests: T::test_or19_decisions_retry_408_and_504 and test_or19_decisions_wait_honours_retry_after.

### OR-20
**Disposition:** RESOLVED. `error_code_status` (`_openai_compat_http.py:94`) maps numeric codes and known transient string codes to their HTTP status:
- `server_error` and similar go to 502;
- overload codes go to 503;
- `timeout` goes to 504;
- rate-limit codes go to 429.

A 200 body carrying such a code raises a retryable `httpx.HTTPStatusError`. Unknown strings still raise a plain, non-retryable `LLMProviderError`. Tests: T::test_or20_error_code_status (parametrised), test_or20_string_code_in_200_body_is_retryable_http_error and test_or20_unknown_string_code_is_not_retried.

## Structural notes

- `openai_compat.py` had reached 1032 lines after these changes. The request-body methods moved, unchanged apart from the fixes above, into a new mixin, `_openai_compat_body.RequestBodyMixin`. The main module is now under 1000 lines.
- The OpenRouter overrides are in the new `openrouter_provider/_request.py` and `_accounting.py`. `_provider.py` is at 990 lines.

## Other HANDOFFs to CORE

- `LLMStreamInterruptedError` is importable from `pyutilz.llm.openai_compat` and `pyutilz.llm._openai_compat_http`. Consider re-exporting it from `pyutilz.llm.exceptions` / `pyutilz.llm.__init__`, which are not owned here.
