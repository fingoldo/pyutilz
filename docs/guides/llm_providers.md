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
- `generate_json(...)` — JSON output. Passing `json_schema=` to `generate()`, `generate_stream()` or `generate_json()` sends `response_format={"type": "json_schema", ..., "strict": true}`, which CONSTRAINS generation (a closed enum becomes impossible to violate) rather than merely asking the model for valid JSON. `supports_json_schema()` advertises the capability: `False` on the base class, so Anthropic, Gemini and Claude Code report `False`; `True` for the OpenAI-compatible endpoints (`openai`, `deepseek`, `xai`); a per-model catalogue check on OpenRouter, which requires `structured_outputs` in the model's `supported_parameters` and returns `False` on an unknown model or a catalogue failure rather than claim a guarantee it cannot verify. A model without support degrades to plain JSON mode with a warning so a mixed-model sweep stays runnable — check `last_json_schema_applied` after the call to learn whether the guarantee actually held, since the degraded path still returns JSON that just is not schema-enforced.
- `get_account_credits()` — native only for OpenRouter and DeepSeek (both expose a real balance endpoint); every other provider raises `NotImplementedError`.
- `check_account_limits()` — native dedicated-endpoint support only for OpenRouter; Anthropic and DeepSeek fall back to captured `anthropic-ratelimit-*` / `x-ratelimit-*` response headers (populated after at least one `generate()` call); OpenAI, xAI, and Gemini raise `NotImplementedError` by design, even though OpenAI/xAI already capture the same rate-limit headers internally for other purposes; Claude Code shells out to the CLI and has no HTTP headers to capture.

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

## OpenRouter: a model id is not a route, and a ceiling is not a target

Everything below was measured on 2026-09-07/08 against a live account while benchmarking a fleet of 19
models. Each item cost a wave of captures to learn, and each one fails in a way that points at the wrong
place, so they are recorded with the symptom that misled first.

**One model id is served by many providers, and they are not interchangeable.** `z-ai/glm-5.3-flash` had 20
endpoints one hour and 24 the next, with advertised `max_completion_tokens` from 128,000 to 1,179,648.
OpenRouter picks per request, so two consecutive calls to the same id run under different ceilings and an
unpinned benchmark measures the route lottery while reporting it as a model difference.

**The advertised cap is not the enforced one.** A call asking for 131,072 tokens was routed to Reka, which
advertises 235,929, and stopped at exactly 15,000 with `finish_reason="length"`. Confirmed against
OpenRouter's own record: `GET /api/v1/generation?id=<generation_id>` returned `native_tokens_completion:
15000`, `provider_name: "Reka"`. Neither the model's catalogue entry nor its endpoint list predicts the
ceiling a given call actually gets — only the generation record says what happened.

**Pin with `provider_order` + `provider_allow_fallbacks=False`, and pin to SLUGS.** The order entries are
routing slugs — the part before the `/` in an endpoint's `tag` (`deepinfra/fp8` → `deepinfra`) — never the
`provider_name` display string (`"DeepInfra"`). A display name is not rejected as malformed; it matches no
route, and with fallbacks off the call returns **`ModelNotFound`**, which reads as if the model itself were
gone.

**Once pinned, the ROUTE's cap binds, not the model's.** `z-ai/glm-5.3-flash` advertises 131,072 as a
model while its `nextbit` endpoint advertises 128,000. Asking a pinned route for more than it serves leaves
OpenRouter with nothing matching the request, and it answers `404 No endpoints found for <model>` — again
reading as "this model does not exist" rather than "your ceiling and your pin disagree". Take the cap from
the same endpoint you took the slug from.

**One pinned route cannot carry a concurrent run.** With fallbacks off, a `429` has nowhere to go, so it
retries into the same wall until the attempts are spent: three 429s on a preflight ping alone, then a
timeout, and a whole wave produced zero captures while the same call issued on its own succeeded in 221 s.
Pin to a short ORDERED LIST (three routes that all clear the output you need, budget held to the smallest
of their caps) rather than to one host — the requirement is that every call run under a KNOWN ceiling, not
that every call reach the same machine. Record `last_upstream_provider` on every row regardless, so a pin
that has quietly stopped applying is visible in the results rather than assumed from the request.

**Choosing the pin by ADVERTISED cap is a heuristic; verify it from the rows you already keep.** The point
above is that advertised is not enforced, so picking the smallest advertised cap that clears your need
inherits exactly that unreliability — a pin that had landed on Reka would look right everywhere except in
the output. The verification is cheap: a row with `finish_reason == "length"` whose output sits well below
the cap its pinned route advertises is that route enforcing less than it claims. Checking WHICH provider
served (route drift) is a different check and does not imply this one; keep both.

**Do not pin a model served from ONE endpoint in total.** The pin cannot improve reproducibility — there
is nothing to choose between — while `allow_fallbacks=False` still switches off OpenRouter's own retry.
Measured: `qwen/qwen3.8-flash` has a single endpoint and emitted 50,466 and 70,783 tokens cleanly while
unpinned, then failed on `429` as soon as it was pinned to that same endpoint. Still hold the budget to that
endpoint's cap, which is the one that binds. The opposite case — one QUALIFYING endpoint among several —
must stay pinned: unpinned, the call can land on a route too small to hold the answer.

**Match a served row to its route by display name, not by lower-casing the slug.** `provider.order` takes
the slug (`sail-research`) while a served row records the display name (`Sail Research`), so comparing the
two by case-folding happens to work for `NextBit`/`nextbit` and reports drift for every provider whose name
carries a space or a hyphen. Record the slug → display-name pairs from the same `/endpoints` response the
pin came from.

**Never round a measured need to a comfortable number.** Keep the figure the captured rows show, per model.
Rounding 57,646 up to 64,000 feels harmless, and it is the same move as capping a timeout at 20 minutes
while the measured range in the very same comment reads 25,000-70,000 tokens — which killed every capture
from the model that needed 39. A round number is not evidence, and this is the failure that recurs most.

**Size the request timeout from what arrives, not from the ceiling.** `_timeout_for` derives a per-request
timeout from `max_tokens` at a pessimistic 30 tok/s, which is right in direction: a name-based heuristic
cannot see how much output was asked for, and a `z-ai/glm-5.3-flash` asked for 54,853 tokens once got the
240 s default and died in a ReadTimeout storm on every capture. But `max_tokens` is a ceiling, and deriving
a WAIT from it treats it as a prediction of how much the model will say. A 128,000-token request derives
71 minutes per stalled attempt, with the retry count behind it. Hence `_max_derived_timeout_s`, and hence
the way it is sized: the cap must clear the largest answer that actually ARRIVES. Set to 20 minutes on
first writing, it killed every capture from the one model that emits 70,783 tokens and needs 39. The
largest clean emission across the fleet is 85,694 (`finish_reason="stop"`), needing 2,856 s, so the cap is
3,000 s. Only the derived half is clamped — a slow-tier model keeps whatever `_get_timeout` grants it.

**A per-model figure, never one number for the fleet.** The same mistake recurs at three levels — the
output ceiling, the route pin, and the expected emission — and it fails identically each time. A single
fleet-wide "largest output this task needs" is either too small for the biggest arm (34,000 was, by 2.5x,
cutting the model that emits 85,694) or too demanding for the rest (requiring 85,694 of every route drops a
model whose own largest clean answer is 36,561 and whose endpoints cap at 64,000). Keep a measured figure
per model, defaulting an unmeasured one to the fleet maximum: admitting a route that then truncates costs
the capture, while demanding too much room only narrows the fleet, and that is the cheaper error.

**Some routes return no `finish_reason` at all.** A completed generation via Phala reported
`finish_reason: None`, `native_finish_reason: None` and `total_cost: 0` in OpenRouter's own generation
record, with real token counts. On such a route the usual "was this answer complete or cut off" test is
simply unavailable, so a short answer cannot be distinguished from a truncated one — worth knowing before
scoring a model low on a route that will not say.

**Telemetry to keep on failures too.** A truncated or errored call has already burned its input and
reasoning tokens and has already been billed. Discarding its telemetry records it at zero, so a wave that
fails a lot reads as CHEAPER than one that works: measured, 142 of 253 captures were failures and every one
carried cost 0, while one error message itself reported 5,741 reasoning tokens spent on the call it was
reporting. Any spend cap comparing accumulated cost against a limit goes quiet in exactly that case.

## Adding a new provider

Each provider is a `(module_path, class_name)` entry in `_PROVIDER_MODULES` (`factory.py`), lazily imported so a project that only uses one provider doesn't pay import cost for the other six's SDKs. A new provider implements the `LLMProvider` interface in `base.py` and registers itself the same way — see `anthropic_provider.py` as the reference implementation.
