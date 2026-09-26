# Dispositions: 60_downstream_openrouter_adoption.md, autopsia part (DS-1 to DS-6)

Worktree: `scratchpad/wt-autopsia-ds`, detached at autopsia origin/master 09f5dea0. Not committed. Regression tests are in
`tests/test_openrouter_policy_and_interruptions.py` (T below). Every test there passes against the uncommitted pyutilz in
`wt-pyutilz-audit/src`, and the file also runs against the installed pre-change pyutilz, which lacks
`LLMStreamInterruptedError`, through a same-named stand-in subclass of `LLMProviderError`.

### DS-1
**Disposition:** RESOLVED. `complaint_parser.parse_complaint` and `complaint_parser_disambiguation` now catch
`(ValueError, OSError, LLMProviderError, httpx.HTTPError)`. The base `LLMProviderError` works on both pyutilz versions and
covers `LLMStreamInterruptedError`, "no choices" and a data-policy 404. `parse_complaint` also builds its provider inside
the guard now, matching the disambiguation call. Before this, a missing key or the DS-2 fail-closed path escaped as a
500. Tests: T::test_parse_complaint_degrades_a_non_value_error_call_failure_to_a_warning and
test_disambiguation_degrades_a_non_value_error_call_failure_to_unresolved, each parametrised over stream-interrupted,
provider-error and HTTP 404, plus test_parse_complaint_endpoint_answers_200_with_the_caveat_on_an_interrupted_stream
(a TestClient round trip).

### DS-2
**Disposition:** RESOLVED. Added `attempt_log.private_provider(model)`: `archived(get_llm_provider("openrouter",
model=..., provider_data_collection="deny", provider_zdr=True))`. Both Noema call sites use it. The helper fails
CLOSED on a pyutilz without these options: the constructor's TypeError is re-raised as a ValueError, so the call
degrades to a warning and the complaint text is never sent without the policy.

Live check on 2026-09-26: one call per model in `bench.run_tournament.CANDIDATES`, max_tokens=64, with deny+zdr. Models
that failed were retried with deny only. Total spend was about $0.0002.
- deny+zdr PASS on 11 models: deepseek/deepseek-v3.2, minimax/minimax-m2.5 (answered, then hit the 64-token cap),
  tencent/hy3, deepseek/deepseek-chat, qwen/qwen-2.5-72b-instruct, openai/gpt-4o-mini,
  meta-llama/llama-3.3-70b-instruct, mistralai/mistral-small-3.2-24b-instruct, google/gemini-2.5-flash-lite,
  deepseek/deepseek-v4-flash (the default and accuracy-validated model) and qwen/qwen3.5-9b.
- arcee-ai/trinity-large-thinking returned `404 No endpoints found matching your data policy (Zero data retention)` and
  passed with deny only. Decision: remove it from the Noema pool and keep the policy strict (no fallback to deny-only).
  It is listed in `attempt_log.NO_PRIVATE_ENDPOINT_MODELS`. `select_model()` skips it, and
  `ParseComplaintRequest._model_is_in_the_vetted_pool` rejects it as a model override. It stays in CANDIDATES for the
  bench, which sends public literature.
- inclusionai/ring-2.6-1t ("no longer available as a free model") and nex-agi/nex-n2-mini ("No endpoints found") return
  404 under EVERY policy. They are gone from the catalogue, so this is not a policy failure, and the live
  `screen_candidate_models` already drops them. Pruning them from CANDIDATES is a separate bench-maintenance item and was
  not done here.

Tests: T::test_private_provider_asks_for_deny_and_zdr_and_archives,
test_private_provider_fails_closed_on_a_pyutilz_without_the_policy_options,
test_parse_complaint_and_disambiguation_both_build_through_the_private_policy,
test_parse_complaint_degrades_when_the_policy_cannot_be_enforced,
test_select_model_never_picks_a_model_without_a_private_endpoint and
test_the_api_refuses_an_override_to_a_model_without_a_private_endpoint. The existing meta gate
`test_no_paid_call_bypasses_the_archive` still passes, because the helper wraps its construction directly.

noema_app privacy text: NOT edited here. This is a handoff with the exact text change. In
`lib/core/l10n/app_en.arb` `privacyWhatLeavesBody`, after "...to interpret it, including when a phrase is ambiguous;",
insert "the server lets OpenRouter route that text only to model providers that neither store nor train on it
(zero data retention);". In `lib/core/l10n/app_ru.arb` `privacyWhatLeavesBody`, after "...в том числе когда формулировка
неоднозначна;", insert "сервер разрешает OpenRouter передавать этот текст только тем поставщикам моделей, которые не
хранят его и не обучаются на нём (нулевое хранение данных);". Ship it only together with this backend change.

### DS-3
**Disposition:** RESOLVED. `pipeline._extract_or_record_truncation` now records a class named
`LLMStreamInterruptedError` (matched by name, like the truncation arm) as `status="failed_stream_interrupted"`. The
failure record carries `model`, `reason`, `code`, `retryable` and `partial_chars`, and no
`output_cap_is_the_models_own`. Test: T::test_an_interrupted_stream_during_ingest_is_a_named_outcome.

### DS-4
**Disposition:** RESOLVED. The generic retry arm in `ru_llm_translation._call_batch_with_retry` and
`ru_llm_translation_causes_validation._call_validation_batch_with_retry` breaks immediately when
`getattr(e, "retryable", True) is False`. Tests: T::test_translation_retries_only_what_the_provider_calls_retryable and
test_causes_validation_retries_only_what_the_provider_calls_retryable (non-retryable: 1 call; retryable: 3 calls).

### DS-5
**Disposition:** RESOLVED (both follow-ups).
(a) The live check showed that a 404 from OpenRouter surfaces as pyutilz `LLMProviderError`, which the sweep loop
already caught. The except in `track_b_bycatch_sweep.sweep_many` was still widened from `httpx.TransportError` to
`httpx.HTTPError`, so a bare non-retryable `HTTPStatusError` skips one article instead of ending the wave.
(b) `track_b_bycatch_model_arena._pin_routes` now keeps only endpoints that pass `_honours_json_schema`: their
`supported_parameters` lists `structured_outputs`, or the list is not reported at all.
The optional `provider_only` form was not adopted. Order plus no fallbacks is equivalent, and the pin override and
drift report are built around `provider_order`.
Tests: T::test_the_sweep_skips_one_article_on_an_http_status_error and
test_route_pinning_skips_an_endpoint_that_does_not_serve_structured_outputs.

### DS-6
**Disposition:** RESOLVED -- the autopsia half here; the pyutilz attempt_archive half landed in 30_core.md. `attempt_log.summary()` counts
`("error", "truncated", "interrupted")` as failed, so it is correct once pyutilz ships the outcome. Test:
T::test_attempt_summary_counts_an_interrupted_attempt_as_failed.

pyutilz spec for `src/pyutilz/dev/attempt_archive.py`. When this was written, items 1, 2 and 4 were ALREADY present as
uncommitted changes in the `wt-pyutilz-audit` worktree: `OUTCOMES` gained `"interrupted"`, `_Archiver.keep` gained
`_is_stream_interruption`, and `tests/test_dev_attempt_archive.py:141-163` covers them. Only item 3 is still open.
1. Add `"interrupted"` to `OUTCOMES`.
2. Classify `LLMStreamInterruptedError` ahead of the text-based truncated/error split, which is the shared `keep()`
   used by the buffered and the streaming wrappers. `LLMTruncationError` stays `"truncated"`.
3. OPEN: record `error_code=exc.code` and `retryable=exc.retryable` on the attempt, as optional fields that are absent
   for other outcomes. This lets `attempt_log.summary()` separate transient upstream faults from fatal ones.
4. Tests: an interruption with partial text, and one without, are both archived as `"interrupted"`.
5. glossum's consumers of `OUTCOMES`, if any, should also count `"interrupted"` as failed.
