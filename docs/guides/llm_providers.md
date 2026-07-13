# Unified LLM provider interface

## Why this exists

Every LLM vendor (Anthropic, OpenAI, Google Gemini, DeepSeek, xAI Grok, OpenRouter, and Claude Code as a local proxy) ships a different SDK, a different streaming protocol, a different way to report token usage, and a different (often absent) way to report cost and remaining credits. Code that calls providers directly ends up with one bespoke integration per vendor, and switching providers — for cost, latency, or rate-limit reasons — means rewriting call sites.

`pyutilz.llm` collapses all seven into one `LLMProvider` interface (`src/pyutilz/llm/base.py`) with a single factory, `get_llm_provider()` (`src/pyutilz/llm/factory.py`), that resolves a provider name (plus common aliases: `"claude"` → `anthropic`, `"gpt"` → `openai`, `"or"`/`"router"` → `openrouter`, etc.) to a lazily-imported provider class. Switching providers is a one-string change at the call site, not a rewrite.

## Quick example

```python
from pyutilz.llm import get_llm_provider

p = get_llm_provider("openrouter", model="anthropic/claude-sonnet-4.6")
text = await p.generate("Summarise this", system="You are concise.")

print(p.last_call_summary())
# {'generation_id': 'gen-...', 'upstream_provider': 'Anthropic',
#  'cost_usd': 0.0042, 'input_tokens': 1200, 'cache_hit_tokens': 800,
#  'native_finish_reason': 'end_turn', 'is_byok': False, ...}
```

Every provider exposes the same core surface:

- `generate(prompt, system=None, **kwargs)` — a single completion.
- `generate_json(...)` — schema-constrained JSON output where the upstream API supports it.
- `get_account_credits()` / `check_account_limits()` — works natively where the upstream exposes it (OpenRouter, DeepSeek); other providers fall back to capturing `anthropic-ratelimit-*` / `x-ratelimit-*` response headers automatically, so the caller gets *some* signal even when the vendor has no dedicated billing endpoint.

`generate_stream(...)` — token streaming with usage tracking preserved across the stream, not just on the final chunk — is available on the OpenAI-compatible providers (`openai`, `deepseek`, `xai`, `openrouter`). `anthropic`, `gemini`, and `claude-code` don't implement it yet.

## Instance caching

`get_llm_provider()` caches instances by `(canonical_name, kwargs_key)` (see `_provider_cache` in `factory.py`) so repeated calls with the same provider+model don't pay for client re-initialisation (SSL context setup, SDK client construction). Providers constructed with unhashable kwargs bypass the cache but are still tracked in a `WeakSet` so an `atexit` handler can close their HTTP clients — no leaked connections even for uncached instances.

## OpenRouter health-aware model selection

OpenRouter aggregates many upstream models behind one API, and upstream health varies. `list_openrouter_models()` does a two-stage lookup — an offline catalogue pass, then a concurrent live `/endpoints` health check — to drop degraded upstreams and rank by live latency:

```python
from pyutilz.llm import list_openrouter_models

rows = list_openrouter_models(
    name_contains="claude",
    max_input_per_1m=1.0,
    sort_by="uptime",
    min_uptime=0.99,
)
top = rows[0]
print(top["id"], top["health"]["best_uptime_30m"], top["health"]["best_latency_p50_ms"], "ms p50")
```

Stage 2 is auth-gated (needs an API key) but not billed — it queries endpoint metadata, not generation.

## Adding a new provider

Each provider is a `(module_path, class_name)` entry in `_PROVIDER_MODULES` (`factory.py`), lazily imported so a project that only uses one provider doesn't pay import cost for the other six's SDKs. A new provider implements the `LLMProvider` interface in `base.py` and registers itself the same way — see `anthropic_provider.py` as the reference implementation.
