# Dispositions: 60_downstream_openrouter_adoption.md, DS-7..DS-11 (glossum_backend_scripts, llm_bench)

Worktrees (uncommitted, detached from origin/main): `scratchpad/wt-glossum-ds`, `scratchpad/wt-llmbench-ds`.

### DS-7
**Disposition:** RESOLVED -- Fixed (glossum). New `glossum.core.exceptions.LLMStreamInterruptedError(LLMProviderError)` carrying
`code`, `partial_text`, `retryable`. `llm_client._as_glossum_provider_error` converts every non-truncation pyutilz
`LLMProviderError` into glossum's class (stream interruption -> the new subclass, others -> `LLMProviderError` with
`partial_text` kept), `raise ... from exc`; non-provider exceptions pass unchanged. The pyutilz class is looked up with
`getattr(..., ())`, so an older pyutilz still imports. `retry_loop.retry_llm_with_budget` retries a retryable
interruption once (sharing the one stall retry), records the attempt as `stream_interrupted` with its fragment, and
raises a non-retryable one or a second one. Tests: `tests/test_llm/test_openrouter_adoption_ds7_ds9.py` (conversion,
cause chain, non-conversion of other exceptions, retry-once, second/non-retryable raise).

### DS-8
**Disposition:** RESOLVED -- Fixed (glossum), pyutilz route_fingerprint() now landed (30_core.md). `response_cache.route_of(provider)` uses
`type(provider).route_fingerprint(provider)` when the class defines it (class lookup, so a MagicMock's auto-attribute is
not mistaken for one); otherwise the fallback `_ROUTE_ATTRIBUTES` now also covers `_provider_only`,
`_provider_require_parameters`, `_provider_field`, `_plugins`, `_transforms`, `_system_cache_control`,
`_default_extra_body`. Tests: real `OpenRouterProvider` with vs without `provider_quantizations=("fp4",)` gives different
keys (same config gives the same key); each fallback attribute changes the key; fingerprint path used when present.
pyutilz spec for `OpenRouterProvider.route_fingerprint()` is in the agent's final report.

### DS-9
**Disposition:** RESOLVED -- Fixed (glossum). `scripts/probe_word_family_verdicts.py` and `scripts/measurements/sv_f08/measure.py`
now call `glossum.llm.providers.factory.get_llm_provider(...)` with `json_mode=True` (measure: `thinking="low"` on
OpenRouter, none on `xai:` routes, which go to the `xai` provider). Raw httpx and the hand-rolled .env key readers are
gone. A mid-stream error now raises and is stored as `{"error": ...}` (never scored as an answer); a truncation is stored
with its partial text and `finish_reason: "length"` so `score()`'s retry logic still applies; cost comes from
`last_actual_cost_usd`. Tests cover both scripts with a fake provider.

### DS-10
**Disposition:** RESOLVED -- Fixed (llm_bench). `provider_kwargs` added to `RoundConfig` and `Benchmark`; the default factory (round
calls and preflight) calls `get_llm_provider(label, model=m, **provider_kwargs)`. The kwargs are folded into the
recorded provider identity (`provider_identity()`: `openrouter{route:<sha12>}`), used for every row and for resume-cache
lookups, so runs under different route policies are never pooled or served from each other's cache; empty kwargs or a
custom `provider_factory` keep the plain label (no change for existing data). Example `job_app_cover_letter/run.py` live
mode now uses the default factory with `provider_quantizations=("bf16","fp16","fp8","unknown")`. Tests:
`tests/unit/test_provider_kwargs_and_stream_labels.py`.

### DS-11
**Disposition:** RESOLVED -- Fixed (llm_bench and glossum twin). `classify_provider_error` maps `LLMStreamInterruptedError` to
`StreamInterrupted` and "requested parameters"/"require_parameters" to `ParametersUnsupported`, both before the
`ModelNotFound` arm. `StreamInterrupted` is in `TRANSIENT_ERROR_CLASSES` only (not DEAD, so the per-pipeline 3-strike
breaker ignores it); `ParametersUnsupported` is in `DEAD_ERROR_CLASSES`, same verdict as `JsonModeUnsupported` for a
JSON benchmark but no longer mislabelled as a removed model. Same split in glossum's
`scripts/_run_experiment/_provider_discovery._classify_provider_error` and `glossum/llm/benchmark_halving/_liveness.py`.
Tests in both repos.

### DS-12

**Disposition:** RESOLVED -- pyutilz is 1.1.0 (60c59c3; back-compat re-export 89eb1f6). autopsia requires pyutilz>=1.1 (c8575423), llm_bench requires pyutilz>=1.1 (57e79a7), glossum's uv.lock pins pyutilz 1.1.0 at 89eb1f6 (bc237a75; it installs from git, so the lock is its pin).
