# Provider pricing tiers: cache writes, long context, other tiers (2026-10-03)

Research input for the shared pricing mechanism a follow-up implements; no pricing structure was refactored here.
Each fact cites the page it was read from on 2026-10-03. LIVE = confirmed by a live call or listing (see
`provider_live_checks.json`); DOC = page only; UNVERIFIED = not on the page and not probed.

## Anthropic

Sources: https://platform.claude.com/docs/en/about-claude/pricing , `GET /v1/models` (LIVE listing).

- (a) Cache WRITE is priced per TTL as a multiplier of base input: 5-minute 1.25x, 1-hour 2x (e.g. Opus 5.5 $5 / $8
  against $4 input; Sonnet 5.5 $2.50 / $4 against $2). Cache READ is 0.1x, except 0.05x on Opus 5.5 and 0.025x on
  Fable 5.1 / Mythos 5.1. DOC. The `usage.cache_creation.ephemeral_{5m,1h}_input_tokens` split could NOT be read
  live: the API key's account has no credit (every Messages and count_tokens call returned 400 "credit balance is too
  low"). The 1h field name was seen live on the Claude Code CLI on 2026-09-26 only.
- (b) Long context: "Claude 4.6 and later models ... include the full 1M token context window at standard pricing"
  (a 900K request is billed at the same per-token rate as 9K). No tier for those models. DOC. Sonnet 4.5 is listed by
  `/v1/models` with `max_input_tokens: 1000000` (LIVE); whether a >200K Sonnet 4.5 prompt carries the old premium
  rate is not stated on the page: UNVERIFIED.
- (c) Batch API 50% off input and output (stacks with caching). Fast mode (Opus 5.5 $8 / $40; Opus 5 and 4.8 $10 /
  $50). `inference_geo: "us"` 1.1x on every category for 4.6+. Web search $10 / 1k searches. DOC.
- pyutilz today: `_cost_usd` bills writes 1.25x (5m) / 2x (1h) and per-model read multipliers: MATCHES the page.
  Batch 50%: implemented. GAPS: fast mode, `inference_geo` 1.1x, web search fees (none have a caller here).

## OpenAI

Sources: https://developers.openai.com/api/docs/pricing , https://developers.openai.com/api/docs/guides/prompt-caching ,
https://developers.openai.com/api/docs/models .

- (a) Cache WRITE: "For GPT-5.6 and later, cache writes cost 1.25x the standard, uncached input-token rate"; earlier
  models have no write charge. Reported as `usage.input_tokens_details.cache_write_tokens` (Responses naming; the
  chat-completions name was not stated). Pricing does NOT vary by retention (`prompt_cache_options.ttl: "30m"` on
  5.6+, `prompt_cache_retention` `in_memory` / `24h` earlier). Minimum cacheable prompt 1,024 tokens on 5.6+. DOC.
  Not verifiable live: the account has no credit (429 `insufficient_quota`).
- (b) Long context: "Short context: <=272K input tokens. Long context: >272K input tokens." Confirmed rows
  (short -> long, per 1M; input / cached input / cache write / output):

  | model | short | long |
  |---|---|---|
  | gpt-6-astra | 10.00 / 1.00 / 12.50 / 50.00 | 20.00 / 2.00 / 25.00 / 75.00 |
  | gpt-6.1-sol | 2.00 / 0.10 / 2.50 / 10.00 | 4.00 / 0.20 / 5.00 / 15.00 |
  | gpt-6-sol | 2.00 / 0.20 / 2.50 / 10.00 | 4.00 / 0.40 / 5.00 / 15.00 |
  | gpt-6-luna | 0.10 / 0.01 / 0.125 / 0.50 | 0.20 / 0.02 / 0.25 / 0.75 |
  | gpt-5.6-sol | 4.00 / 0.40 / 5.00 / 20.00 | 8.00 / 0.80 / 10.00 / 30.00 |
  | gpt-5.6-terra | 2.00 / 0.20 / 2.50 / 12.00 | 4.00 / 0.40 / 5.00 / 18.00 |
  | gpt-5.6-luna | 0.20 / 0.02 / 0.25 / 1.20 | 0.40 / 0.04 / 0.50 / 1.80 |
  | gpt-5.5 | 5.00 / 0.50 / n/a / 30.00 | 10.00 / 1.00 / n/a / 45.00 |
  | gpt-5.4 | 2.50 / 0.25 / n/a / 15.00 | 5.00 / 0.50 / n/a / 22.50 |

  So input, cached input and cache write double, and output is 1.5x. Whether the long rate applies to the WHOLE request
  once the prompt passes 272K, or only to the excess, is not stated in the text the page returned: UNVERIFIED (the
  column layout suggests whole-request, as at xAI and Gemini). DOC.
- (c) Batch, Flex and Priority tiers exist on the page; gpt-5.3-codex has a "Fast" column ($3.50 / $28). Their
  multipliers were not extracted: UNVERIFIED.
- pyutilz today: no cache-write accounting for OpenAI (`_record_usage` does not read `cache_write_tokens`; the base
  `get_session_cost` bills writes only when `total_cache_write_tokens` is set, which only OpenRouter does), no 272K
  tier, no batch/flex/priority. GAPS: all three. `gpt-6.1-sol` had no row at all (added in this pass).

## Google Gemini

Source: https://ai.google.dev/gemini-api/docs/pricing ; `GET /v1beta/models` (LIVE listing).

- (a) No per-write price for implicit caching. Explicit context caching bills cached reads (e.g. 2.5 Flash $0.03,
  3.8 Flash $0.075) plus STORAGE per 1M tokens per hour ($1.00 for most Flash models, $0.50 for 3.6/3.7/3.8 Flash
  until 2026-12-31, $4.50 for 2.5 Pro / 3.1 Pro). DOC.
- (b) Long context >200K prompt tokens only on 2.5 Pro (1.25 / 10 / 0.125 -> 2.50 / 15 / 0.25) and 3.1 Pro Preview
  (2 / 12 / 0.20 -> 4 / 18 / 0.40): input and cached input 2x, output 1.5x, applied to the whole request. No tier on
  the Flash models. DOC.
- (c) Batch and Flex 50% of Standard; Priority 1.8x Standard. Date-based change: 3.6 / 3.7 / 3.8 Flash go from
  $0.75 / $3.75 / $0.075 to $1.50 / $7.50 / $0.15 on 2027-01-01. DOC.
- pyutilz today: the 2.5 Pro / 3.1 Pro multipliers `(2.0, 1.5, 2.0)` and the 200K threshold in
  `GeminiProvider._LONG_CONTEXT_MULTIPLIERS` MATCH the page. GAPS: storage cost, batch/flex/priority, and the
  2027-01-01 price change (the table holds the 2026 rates only). 3.6 Flash, 3.5 Flash-Lite and GA 3.1 Flash-Lite rows
  were missing (added in this pass). The probe key is free tier: Gemini calls were billed $0.

## xAI

Sources: https://docs.x.ai/docs/pricing , https://docs.x.ai/docs/models , `GET /v1/language-models` (LIVE listing,
which carries machine-readable prices in 1e-4 USD per 1M tokens and a `long_context_threshold`).

- (a) No cache-write price ("cached input tokens are billed at reduced rates"; nothing for writes). DOC + LIVE listing.
- (b) Long context: threshold 200,000 prompt tokens for every listed model (LIVE); "requests whose prompt reaches the
  listed token threshold are billed at the higher rate for all tokens in the request" (whole request, DOC). Listed long
  rates are exactly 2x for input, cached input and output (e.g. grok-4.5 2.00 / 0.30 / 6.00 -> 4.00 / 0.60 / 12.00).
- (c) Batch API 20% off (grok-4.3 and grok-4.20 variants); Priority 2x. Server-side tools: web search $5 / 1k calls,
  code execution $5 / 1k, collections/file search $2.50 / 1k, X search $5 / 1k posts and $10 / 1k profiles. DOC; the
  web-search fee was confirmed LIVE ($0.0250 for 5 calls inside `cost_in_usd_ticks`).
- pyutilz today: `_LONG_CONTEXT_THRESHOLD = 200_000` with every rate doubled MATCHES. Tool fees: now implemented
  (this pass). GAPS: batch 20%, priority 2x.

## DeepSeek

Source: https://api-docs.deepseek.com/quick_start/pricing ; `GET /models` (LIVE).

- (a) No cache-write price; cache hit vs miss input only (flash peak $0.006 / $0.30, v4-pro $0.044 / $1.32). DOC + LIVE.
- (b) No long-context tier (1M context, one rate). DOC.
- (c) Off-peak: "Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday, excluding Chinese public
  holidays"; off-peak is 50% of every peak rate. DOC.
- pyutilz today: `_PEAK_HOURS_UTC = {1, 2, 3, 6, 7, 8, 9}`, weekdays only, 0.5 off-peak: MATCHES the window. Live
  cost matched at off-peak (Friday 19:00 UTC); the peak branch was not exercised live. GAP: public holidays not
  modelled (documented overestimate).
