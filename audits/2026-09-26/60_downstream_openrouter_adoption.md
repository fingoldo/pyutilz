# 60 - Downstream adoption of the 2026-09-26 OpenRouter options

Scope: consumers of `pyutilz.llm` OpenRouter after `dispositions/10_openrouter.md` (OR-1 to OR-20). Read-only survey on
2026-09-26. Repos checked at: autopsia `Redline/autopsia` 2c58ec76, llm_bench 1e816fb, glossum_backend_scripts 2bd2f2d9,
noema_app cada167. The `Redline/autopsia-wt-*` and `autopsia-push*` worktrees were not surveyed separately. noema-sources
has no OpenRouter code. noema_app (Flutter) talks only to the autopsia API; its only OpenRouter mention is the privacy
page text (`lib/core/l10n/app_en.arb:261`).

## How each repo calls OpenRouter

| Repo | Path to OpenRouter | pyutilz pin | Batch API | Own HTTP chat calls |
|---|---|---|---|---|
| autopsia (the Noema backend) | `pyutilz.llm.get_llm_provider("openrouter", model=...)`, wrapped by `attempt_log.archived` (`pyutilz.dev.attempt_archive`) | `pyutilz>=1.0` (`pyproject.toml:72,113`) | not used | none; `bench/screen_models.py:35` and `bench/track_b_bycatch_model_arena.py:750` read only `/models` and `/endpoints` |
| llm_bench | `get_llm_provider(cfg.provider_label, model=...)` or an injected `provider_factory` (`runner/round_runner.py:963-968`) | `pyutilz>=1.0` (`pyproject.toml:37`); the comment at `:28` claims `>=1.1` | not used | none |
| glossum_backend_scripts | its own `glossum.llm.providers.factory.get_llm_provider`, which wraps pyutilz, the ledger, the response cache and the streaming twin | `pyutilz @ git+https://github.com/fingoldo/pyutilz.git` (unpinned, `pyproject.toml:140`, resolved through uv.lock) | not used | two measurement scripts (DS-9) |

Batch changes (`ok=False` on truncation, `metadata.request_hash` removed): no consumer of `openrouter_batch`,
`BatchResult`, `recover_submit` or `request_hash` exists in any of the three repos. Not applicable.

Version floors: pyutilz's `pyproject.toml` still says `version = "1.0.0"`, so no consumer can state "needs the
2026-09-26 API". See DS-12.

## Findings

### DS-1 (High) -- autopsia: Noema complaint parsing lets LLMStreamInterruptedError escape as HTTP 500

**Disposition:** RESOLVED. `complaint_parser.parse_complaint` and `complaint_parser_disambiguation` now catch `(ValueError, OSError, LLMProviderError, httpx.HTTPError)`. The base `LLMProviderError` works on both pyutilz versions and covers `LLMStreamInterruptedError`, "no choices" and a data-policy 404. `parse_complaint` also builds its provider inside the guard now, matching the disambiguation call. Before this, a missing key or the DS-2 fail-closed path escaped as a 500. Tests: T::test_parse_complaint_degrades_a_non_value_error_call_failure_to_a_warning and test_disambiguation_degrades_a_non_value_error_call_failure_to_unresolved, each parametrised over stream-interrupted, provider-error and HTTP 404, plus test_parse_complaint_endpoint_answers_200_with_the_caveat_on_an_interrupted_stream (a TestClient round trip).
Evidence: `autopsia/ingest/complaint_parser.py:717-720` catches only `(ValueError, OSError)` around
`prov.generate_json(...)`, and so does `autopsia/ingest/complaint_parser_disambiguation.py:211-217`. The Noema workflow
reaches the second one through `reason/differential_disambiguation.py:24` and `api_noema_workflow.py:111,198`.
`LLMStreamInterruptedError(LLMProviderError)` is NOT a `ValueError`. The buffered path now raises it on
`finish_reason == "error"` or on an error body that also carries choices (`openai_compat.py:747`, after its retries).
Before this change, that response came back as a fragment, and `_coerce_json` or `JSONParsingError` (which is a
ValueError) degraded it into a warning. The disambiguation guard's own docstring says a failed call must degrade, never
return 500.
The same gap already existed for plain `LLMProviderError` ("returned no choices") and for non-retryable
`httpx.HTTPStatusError`. OR-2's automatic `require_parameters` makes the latter more likely, since a 404 "no endpoints"
now arrives where a silently dropped `response_format` used to.
Change, in both places:
```python
from pyutilz.llm.exceptions import LLMProviderError
import httpx
...
except (ValueError, OSError, LLMProviderError, httpx.HTTPError) as exc:
```
The existing warning text already names the class. Add a regression test that monkeypatches `generate_json` to raise
`LLMStreamInterruptedError("x", partial_text="{")` and asserts a 200 with the caveat.

### DS-2 (High) -- autopsia: Noema sends patient complaint text to OpenRouter with no data-collection policy

**Disposition:** RESOLVED. Added `attempt_log.private_provider(model)`: `archived(get_llm_provider("openrouter", model=..., provider_data_collection="deny", provider_zdr=True))`. Both Noema call sites use it. The helper fails CLOSED on a pyutilz without these options: the constructor's TypeError is re-raised as a ValueError, so the call degrades to a warning and the complaint text is never sent without the policy. Live check on 2026-09-26: one call per model in `bench.run_tournament.CANDIDATES`, max_tokens=64, with deny+zdr. Models that failed were retried with deny only. Total spend was about $0.0002. - deny+zdr PASS on 11 models: deepseek/deepseek-v3.2, minimax/minimax-m2.5 (answered, then hit the 64-token cap), tencent/hy3, deepseek/deepseek-chat, qwen/qwen-2.5-72b-instruct, openai/gpt-4o-mini, meta-llama/llama-3.3-70b-instruct, mistralai/mistral-small-3.2-24b-instruct, google/gemini-2.5-flash-lite, deepseek/deepseek-v4-flash (the default and accuracy-validated model) and qwen/qwen3.5-9b. - arcee-ai/trinity-large-thinking returned `404 No endpoints found matching your data policy (Zero data retention)` and passed with deny only. Decision: remove it from the Noema pool and keep the policy strict (no fallback to deny-only). It is listed in `attempt_log.NO_PRIVATE_ENDPOINT_MODELS`. `select_model()` skips it, and `ParseComplaintRequest._model_is_in_the_vetted_pool` rejects it as a model override. It stays in CANDIDATES for the bench, which sends public literature. - inclusionai/ring-2.6-1t ("no longer available as a free model") and nex-agi/nex-n2-mini ("No endpoints found") return 404 under EVERY policy. They are gone from the catalogue, so this is not a policy failure, and the live `screen_candidate_models` already drops them. Pruning them from CANDIDATES is a separate bench-maintenance item and was not done here. Tests: T::test_private_provider_asks_for_deny_and_zdr_and_archives, test_private_provider_fails_closed_on_a_pyutilz_without_the_policy_options, test_parse_complaint_and_disambiguation_both_build_through_the_private_policy, test_parse_complaint_degrades_when_the_policy_cannot_be_enforced, test_select_model_never_picks_a_model_without_a_private_endpoint and test_the_api_refuses_an_override_to_a_model_without_a_private_endpoint. The existing meta gate `test_no_paid_call_bypasses_the_archive` still passes, because the helper wraps its construction directly. noema_app privacy text: NOT edited here. This is a handoff with the exact text change. In `lib/core/l10n/app_en.arb` `privacyWhatLeavesBody`, after "...to interpret it, including when a phrase is ambiguous;", insert "the server lets OpenRouter route that text only to model providers that neither store nor train on it (zero data retention);". In `lib/core/l10n/app_ru.arb` `privacyWhatLeavesBody`, after "...в том числе когда формулировка неоднозначна;", insert "сервер разрешает OpenRouter передавать этот текст только тем поставщикам моделей, которые не хранят его и не обучаются на нём (нулевое хранение данных);". Ship it only together with this backend change.
Evidence: `complaint_parser.py:701` `archived(get_llm_provider("openrouter", model=chosen_model))` and
`complaint_parser_disambiguation.py:212`. Both send free-text health complaints typed by a user. The privacy page
(`noema_app/lib/core/l10n/app_en.arb:261`) tells the user that complaint text "may be sent to a language-model service
(OpenRouter)", but no routing constraint stops OpenRouter from choosing an endpoint that logs prompts or trains on them.
Change: pass `provider_data_collection="deny", provider_zdr=True` at both construction sites. It is simplest to add a
helper in `attempt_log.py`, for example `private_provider(model)`, and have both sites call it so they cannot drift apart.
Caveats that must be checked live before shipping:
- `select_model()` (`complaint_parser.py:~596-640`) picks from `bench.run_tournament.CANDIDATES`. Each candidate needs
  at least one ZDR/deny endpoint, or every call returns 404. Filter CANDIDATES on `/endpoints` for the policy, or fall
  back to `deny` without `zdr`.
- Update the privacy text (en + ru arb) to state the policy once it is enforced.

`parse.py:71-81` (Anthropic default) and the KB and bench pipelines, which send public literature, do not need this.

### DS-3 (Med) -- autopsia: the ingest pipeline treats a mid-stream failure as a crash, not as a recorded outcome

**Disposition:** RESOLVED. `pipeline._extract_or_record_truncation` now records a class named `LLMStreamInterruptedError` (matched by name, like the truncation arm) as `status="failed_stream_interrupted"`. The failure record carries `model`, `reason`, `code`, `retryable` and `partial_chars`, and no `output_cap_is_the_models_own`. Test: T::test_an_interrupted_stream_during_ingest_is_a_named_outcome.
Evidence: `autopsia/ingest/pipeline.py:330-335` `_extract_or_record_truncation` records only `LLMTruncationError` by
class name and re-raises everything else. A stream interrupted after the tokens were bought is the same "paid, no answer"
case. Today it propagates to the per-article `except Exception` at `:674`, which does not produce a named manifest
status.
Change: also accept `LLMStreamInterruptedError`. Set `base["status"] = "failed_stream_interrupted"` and record
`exc.code`, `exc.retryable` and `len(exc.partial_text)`. Do not set `output_cap_is_the_models_own`.

### DS-4 (Low) -- autopsia: KB translation retries a non-retryable interruption three times

**Disposition:** RESOLVED. The generic retry arm in `ru_llm_translation._call_batch_with_retry` and `ru_llm_translation_causes_validation._call_validation_batch_with_retry` breaks immediately when `getattr(e, "retryable", True) is False`. Tests: T::test_translation_retries_only_what_the_provider_calls_retryable and test_causes_validation_retries_only_what_the_provider_calls_retryable (non-retryable: 1 call; retryable: 3 calls).
Evidence: `autopsia/kb/ru_llm_translation.py:252-262`, and `ru_llm_translation_causes_validation.py:137-140` has the same
shape. Any exception other than parse, refusal or timeout is retried in place. `LLMStreamInterruptedError.retryable`
already states whether the upstream fault is transient.
Change: in the generic arm, `if getattr(e, "retryable", True) is False: break` before sleeping.

### DS-5 (Med) -- autopsia: Track A/B/C json_schema + thinking calls now get require_parameters automatically; pinned routes can 404

**Disposition:** RESOLVED (both follow-ups). (a) The live check showed that a 404 from OpenRouter surfaces as pyutilz `LLMProviderError`, which the sweep loop already caught. The except in `track_b_bycatch_sweep.sweep_many` was still widened from `httpx.TransportError` to `httpx.HTTPError`, so a bare non-retryable `HTTPStatusError` skips one article instead of ending the wave. (b) `track_b_bycatch_model_arena._pin_routes` now keeps only endpoints that pass `_honours_json_schema`: their `supported_parameters` lists `structured_outputs`, or the list is not reported at all. The optional `provider_only` form was not adopted. Order plus no fallbacks is equivalent, and the pin override and drift report are built around `provider_order`. Tests: T::test_the_sweep_skips_one_article_on_an_http_status_error and test_route_pinning_skips_an_endpoint_that_does_not_serve_structured_outputs.
Evidence: `bench/track_a_sourcing_pilot.py:587,712,769`, `track_b_sourcing_pilot.py:526,1365,1549,1772,2452`,
`track_b_bycatch_sweep.py:1333-1334`, `track_b_bycatch_validate.py:296-297` and `track_c_sourcing_pilot.py:468,590`
all pass `thinking=` with `json_schema=`. Under OR-2's auto mode these now send `require_parameters: true` whenever
the model-level catalogue lists the parameters. That is the fix for the degradation described at
`track_b_bycatch_sweep.py:264-266`.
The model arena and pinned capture pin one route with `provider_order=..., provider_allow_fallbacks=False`
(`track_b_bycatch_model_arena.py:842-848`, `track_b_pinned_capture.py:63`). If the pinned endpoint lacks
`structured_outputs` while the model as a whole lists it, the call is now refused (404) instead of silently degraded.
Change: none required. It is the desired behaviour, and the sweep loop's except (`track_b_bycatch_sweep.py:1465`)
catches `LLMProviderError`. Two follow-ups:
(a) Confirm that the 404 surfaces as `LLMProviderError` or `httpx.HTTPStatusError`. `httpx.HTTPStatusError` is not in
that tuple (only `httpx.TransportError` is), so widen it to `httpx.HTTPError` if needed.
(b) When `_pin_routes` chooses a route, filter candidate endpoints on `supported_parameters` containing
`structured_outputs`, so the pin never selects an endpoint the request will refuse.
Optional: `provider_only=tuple(pinned)` states the pin more directly than order plus no fallbacks.

### DS-6 (Low) -- autopsia: attempt archive records a mid-stream failure as "truncated"

**Disposition:** RESOLVED: - `AttemptRecord` gains optional `error_code` and `retryable` (`dev/attempt_archive.py`), filled from the `LLMStreamInterruptedError`'s `code` and `retryable` for an `interrupted` outcome only. - `to_dict` leaves both keys out for every other outcome, so existing JSONL rows keep their shape. - Tests: the interruption tests assert `("server_error", True)` and `(400, False)`, and `test_other_outcomes_carry_no_interruption_fields` checks that accepted and error rows carry neither key.
Evidence: pyutilz `dev/attempt_archive.py:392,463` reads `exc.partial_text` generically, so an interrupted call with
text is archived as `outcome="truncated"`. `attempt_log.summary()` (`attempt_log.py:~92`) counts it as failed, which is
correct, but it cannot be told apart from a length cut-off.
Change (pyutilz-side HANDOFF): add an `"interrupted"` outcome to `OUTCOMES` for `LLMStreamInterruptedError`, and count
it in `attempt_log.summary()`'s failed set.

### DS-7 (High) -- glossum: LLMStreamInterruptedError bypasses every `except LLMProviderError` in the pipeline

**Disposition:** RESOLVED -- Fixed (glossum). New `glossum.core.exceptions.LLMStreamInterruptedError(LLMProviderError)` carrying `code`, `partial_text`, `retryable`. `llm_client._as_glossum_provider_error` converts every non-truncation pyutilz `LLMProviderError` into glossum's class (stream interruption -> the new subclass, others -> `LLMProviderError` with `partial_text` kept), `raise ... from exc`; non-provider exceptions pass unchanged. The pyutilz class is looked up with `getattr(..., ())`, so an older pyutilz still imports. `retry_loop.retry_llm_with_budget` retries a retryable interruption once (sharing the one stall retry), records the attempt as `stream_interrupted` with its fragment, and raises a non-retryable one or a second one. Tests: `tests/test_llm/test_openrouter_adoption_ds7_ds9.py` (conversion, cause chain, non-conversion of other exceptions, retry-once, second/non-retryable raise).
Evidence: `glossum/llm/llm_client.py:655-668` converts only pyutilz truncations into glossum's own error types. Any other
pyutilz exception propagates unchanged. glossum's `LLMProviderError` (`core/exceptions.py:65`) is a different class from
pyutilz's, so `except LLMProviderError` misses it: `retry_loop.py:497`, `definition_generator.py:111`,
`example_generator.py:194,321`, `mwe_enrichment_generator.py:187,299` and `cli/main/_errors.py:88`. This is the same
defect class the truncation mapping at `:664` was written to close. The streaming twin (`providers/streaming.py:235-240`)
attaches `partial_text`, so the ledger keeps it, but the retry and fallback logic never sees the failure.
Change, in `llm_client.py` after the truncation arm:
```python
from pyutilz.llm.exceptions import LLMProviderError as _PyutilzProviderError
...
if isinstance(exc, _PyutilzProviderError):
    raise LLMProviderError(f"{type(exc).__name__}: {exc}") from exc
```
Carry `partial_text` and `retryable` through if glossum's class accepts them. Add a test with a fake provider that raises
`LLMStreamInterruptedError` and asserts that `retry_loop` retries it.

### DS-8 (Med) -- glossum: the response-cache route key misses the new routing attributes

**Disposition:** RESOLVED -- Fixed (glossum), pyutilz route_fingerprint() now landed (30_core.md). `response_cache.route_of(provider)` uses `type(provider).route_fingerprint(provider)` when the class defines it (class lookup, so a MagicMock's auto-attribute is not mistaken for one); otherwise the fallback `_ROUTE_ATTRIBUTES` now also covers `_provider_only`, `_provider_require_parameters`, `_provider_field`, `_plugins`, `_transforms`, `_system_cache_control`, `_default_extra_body`. Tests: real `OpenRouterProvider` with vs without `provider_quantizations=("fp4",)` gives different keys (same config gives the same key); each fallback attribute changes the key; fingerprint path used when present. pyutilz spec for `OpenRouterProvider.route_fingerprint()` is in the agent's final report.
Evidence: `glossum/llm/response_cache.py:49`
`_ROUTE_ATTRIBUTES = ("_provider_order", "_provider_ignore", "_provider_sort", "_provider_allow_fallbacks", "_models_fallback")`.
The new kwargs live in `_provider_only`, `_provider_require_parameters` and `_provider_field` (the built provider block,
which holds quantizations, data_collection, zdr, max_price and preferred_*), plus `_plugins`, `_transforms`,
`_system_cache_control` and `_default_extra_body` (pyutilz `openrouter_provider/_provider.py:214-236`). As soon as
glossum sets one of them, a call made with an fp4-only or price-capped route is served the cached answer of an
unconstrained one. That is the exact defect the tuple's comment exists to prevent.
Change: extend the tuple with `"_provider_only", "_provider_require_parameters", "_provider_field", "_plugins",
"_transforms", "_system_cache_control", "_default_extra_body"`. Better, have pyutilz expose one public
`route_fingerprint()` so the list cannot go stale again (pyutilz HANDOFF). Add a test that asserts two providers
differing only in `provider_quantizations` get different keys.

### DS-9 (Low) -- glossum: two measurement scripts call OpenRouter over raw HTTP and miss what the provider now does

**Disposition:** RESOLVED -- Fixed (glossum). `scripts/probe_word_family_verdicts.py` and `scripts/measurements/sv_f08/measure.py` now call `glossum.llm.providers.factory.get_llm_provider(...)` with `json_mode=True` (measure: `thinking="low"` on OpenRouter, none on `xai:` routes, which go to the `xai` provider). Raw httpx and the hand-rolled .env key readers are gone. A mid-stream error now raises and is stored as `{"error": ...}` (never scored as an answer); a truncation is stored with its partial text and `finish_reason: "length"` so `score()`'s retry logic still applies; cost comes from `last_actual_cost_usd`. Tests cover both scripts with a fake provider.
Evidence: `scripts/probe_word_family_verdicts.py:66-78` and `scripts/measurements/sv_f08/measure.py:115-139`.
- `measure.py` sends `reasoning.effort` together with `json_object` but not `require_parameters`, so an endpoint may
  silently drop the reasoning request, and that changes what the A/B measures.
- Neither script checks a 200 body for `error` or for `finish_reason == "error"` (OR-1). An interrupted fragment is
  scored as the model's answer (`measure.py:139` stores `content` as-is; `score()` only checks for `"error"` in the
  saved file).
- Cost is read from `usage.cost` only for printing.

Change: route both through `glossum.llm.providers.factory.get_llm_provider("openrouter", model=m)` and use
`generate(..., json_mode=True, thinking="low")`. That brings in auto require_parameters, the error checks, the ledger
and the cache. If raw HTTP stays, add `"provider": {"require_parameters": True}` and skip a choice whose
`finish_reason == "error"`.

### DS-10 (Med) -- llm_bench: no way to pass the new provider kwargs to the default factory, so benchmarks cannot hold quantization and route policy fixed

**Disposition:** RESOLVED -- Fixed (llm_bench). `provider_kwargs` added to `RoundConfig` and `Benchmark`; the default factory (round calls and preflight) calls `get_llm_provider(label, model=m, **provider_kwargs)`. The kwargs are folded into the recorded provider identity (`provider_identity()`: `openrouter{route:<sha12>}`), used for every row and for resume-cache lookups, so runs under different route policies are never pooled or served from each other's cache; empty kwargs or a custom `provider_factory` keep the plain label (no change for existing data). Example `job_app_cover_letter/run.py` live mode now uses the default factory with `provider_quantizations=("bf16","fp16","fp8","unknown")`. Tests: `tests/unit/test_provider_kwargs_and_stream_labels.py`.
Evidence: `src/llm_bench/runner/round_runner.py:963-968` calls `get_llm_provider(cfg.provider_label, model=model)` with
no constructor kwargs. The only route control is replacing the whole `provider_factory`. A benchmark ranking models
without `provider_quantizations` can grade one model on an fp4 endpoint and another on bf16. This is a known source of
ranking noise, and it is exactly what the new kwarg removes.
Change: add `provider_kwargs: Mapping[str, Any] = field(default_factory=dict)` to `RoundConfig` (`:130` area) and to
the `benchmark.py:97` config. Pass it as `get_llm_provider(cfg.provider_label, model=model, **cfg.provider_kwargs)`.
Recommend `provider_quantizations=("bf16","fp16","fp8","unknown")` in the examples (`examples/job_app_cover_letter/run.py`).
Also record the kwargs in the run's persisted config, so two runs with different route policies are never pooled.

### DS-11 (Low) -- llm_bench: failure classification has no label for a stream interruption, and a require_parameters 404 is labelled ModelNotFound

**Disposition:** RESOLVED -- Fixed (llm_bench and glossum twin). `classify_provider_error` maps `LLMStreamInterruptedError` to `StreamInterrupted` and "requested parameters"/"require_parameters" to `ParametersUnsupported`, both before the `ModelNotFound` arm. `StreamInterrupted` is in `TRANSIENT_ERROR_CLASSES` only (not DEAD, so the per-pipeline 3-strike breaker ignores it); `ParametersUnsupported` is in `DEAD_ERROR_CLASSES`, same verdict as `JsonModeUnsupported` for a JSON benchmark but no longer mislabelled as a removed model. Same split in glossum's `scripts/_run_experiment/_provider_discovery._classify_provider_error` and `glossum/llm/benchmark_halving/_liveness.py`. Tests in both repos.
Evidence: `src/llm_bench/runner/classify.py:12-45`. An `LLMStreamInterruptedError` falls through to the bare class name,
which is harmless but unlabelled. With OR-2's auto mode, a schema-bearing call whose model has no endpoint honouring
every parameter now gets a 404 "No endpoints found that can handle the requested parameters". The `"no endpoints found"`
test at `:31` maps that to `ModelNotFound`, and the alive-filter (`halving/alive_filter.py:41`) then treats the model as
gone rather than as unable to do strict JSON.
Change: before the ModelNotFound block, map `"requested parameters"` / `"require_parameters"` to a new
`ParametersUnsupported` label (a capability miss, not a removed model). Map `exc_name == "LLMStreamInterruptedError"` to
`StreamInterrupted`, which counts as transient and infra-class, never as a model-quality failure. glossum's
`benchmark_halving/_liveness.py:64` has the same mapping and needs the same split.

### DS-12 (Med) -- all repos: pins cannot express the new API

**Disposition:** OPEN -- pyutilz 1.1.0 is committed with this round; downstream floors move to pyutilz>=1.1 in the adoption commits, then this closes.
Evidence: pyutilz `pyproject.toml:17` is `version = "1.0.0"`. autopsia (`pyutilz>=1.0`), llm_bench (`pyutilz>=1.0`) and
glossum (git, unpinned) would all install a pyutilz without `LLMStreamInterruptedError`. DS-1, DS-3 and DS-7 import it,
and on an old pyutilz that import raises ImportError.
Change: bump pyutilz to 1.1.0 with this audit's commit, then set the floors to `pyutilz>=1.1` in autopsia's `llm` and
`test` extras and in llm_bench. For glossum, refresh uv.lock. Until the bump lands, downstream code should import the
class defensively (`getattr(pyutilz.llm.exceptions, "LLMStreamInterruptedError", ())`) or catch the base
`LLMProviderError`, which works on both versions. The DS-1 and DS-7 changes above already catch the base class.

## Per-option disposition matrix

| Option | autopsia | llm_bench | glossum |
|---|---|---|---|
| provider_only | optional for Track B pins (DS-5) | via DS-10 | N/A (pins use provider_order) |
| provider_quantizations | N/A | SHOULD (DS-10) | should, for refsuite/benchmark ranking (same argument as DS-10); goes in the cache key (DS-8) |
| provider_require_parameters (auto) | already effective on every json_schema+thinking call (DS-5); pinned-route caveat | auto; classification (DS-11) | auto; raw-HTTP scripts miss it (DS-9) |
| provider_data_collection / provider_zdr | SHOULD on Noema paths (DS-2) | N/A | N/A (public lexical data) |
| provider_max_price | optional: Noema's vetted pool already bounds spend (`api_models_clinical.py:645-656`) | could replace part of BudgetGate's per-call check; not required | N/A |
| provider_preferred_min_throughput / max_latency | worth trying on the interactive Noema path (latency-bound request) | N/A | N/A |
| web_search_engine / max_results | N/A (sourcing reads PMC full text) | N/A | N/A |
| pdf_engine | N/A (`ingest/vision.py` sends images; articles come as XML) | N/A | N/A |
| transforms | N/A | N/A | N/A |
| system_cache_control | could cut cost on the Track B sweep, where one long system prompt repeats per article; measure first | N/A | could help enrichment's repeated system prompts; measure first |
| extra_body / per-call tools, tool_choice, extra_body | N/A | pass-through already works via `generate_kwargs` | N/A |
| thinking=<int> | N/A (effort strings in use) | N/A | N/A |
| json_schema strict default | already relied on (`track_b_bycatch_validate.py:127-129` built its schema for strict mode) | stage `json_schema` passes through | N/A |
| LLMStreamInterruptedError | DS-1, DS-3, DS-4, DS-6 | DS-11 (the generic except already records it with telemetry, `round_runner.py:988-993`) | DS-7 |
| batch ok=False / request_hash | not used | not used | not used |
| cost accounting (OR-5 summed per-call fields, OR-6 cache-write pricing) | archive reads provider cost; no change | `_harvest_telemetry` prefers `last_call_summary()` (`round_runner.py:921-941`); no change | the ledger prefers `last_actual_cost_usd` (`call_ledger.py:360`); no change |
