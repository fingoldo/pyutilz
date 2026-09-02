# Preventive Meta-Tests and Static Checks - pyutilz (2026-09-02)

## Summary

This report reads the ten 2026-09-02 audit reports (223 findings, every one carrying a COMPLETED or REJECTED disposition; those ten reports now live under `audits/implemented/2026-09-02/`) together with the two 2026-07-21 waves (`_full-audit`, 13 reports; `_audit-round2`, 10 reports), and asks one question: **which of these would a machine have caught, and with what rule?**

Three facts drive everything below.

**Fact 1 - the highest-recurrence class is not a code defect, it is a gate that does not gate.** Every wave has found at least one check that existed, was declared blocking, and did not actually check: 2026-07-21 full-audit found `tests/test_meta/test_optional_deps_isolation.py` had a structural blind spot ("Guard meant to catch exactly this class of bug has a structural blind spot"); round 2 found `capture_signature` could not see default-parameter changes ("API-stability snapshot cannot detect default-parameter-value changes", `src/pyutilz/dev/meta_test_utils.py:180-205`) and that `publish.yml` never verified the tag against `[project].version`; 2026-09-02 found six more (`06/F01`, `06/F08`, `06/F17`, `04/F01`, `04/F08`, `03/F06`). No amount of new scanners helps if the new scanners inherit the same failure mode, so the first proposals below are about gates, not about code.

**Fact 2 - py-ci-shared already ships checks pyutilz never wired up.** `py_ci_shared` has 21 modules; pyutilz consumes exactly one of them for meta-testing (`code_audit_meta`, via `tests/test_meta/test_code_audit_baseline.py:28` and `test_code_audit_tests_baseline.py:53`) plus the two format wrappers in `.pre-commit-config.yaml`. `loc_budget`, `ci_workflow_timeout_gate`, `ci_workflow_gate`, `entry_points_resolvable`, `phantom_markdown_links`, `git_dependency_pins`, `ci_test_dir_reachability`, `changelog_promise_parity` and `content_hash_version_bump_gate` are all unused here, and at least four 2026-09-02 findings fall squarely inside their remit. The cheapest prevention available is not new code.

**Fact 3 - roughly a third of the corpus is not mechanically detectable and should not be chased.** Off-by-one arithmetic, wrong statistical denominators, a sliding window that misses its last position, an LRU that touches the payload but not the sidecar: these are semantic defects whose only automated detector is a test that already knows the right answer. The section "What is not mechanically detectable" names them plainly rather than proposing a lint with a bad false-positive rate into a repo that already runs about fifteen blocking hooks.

Ten proposals follow, ranked. The top three cover, between them, 16 findings directly and recur in all three waves.

## Root-cause taxonomy

Counts are over all 223 categorised 2026-09-02 findings (`- **Category**:` lines, exhaustively bucketed; no finding is unmapped and none is double-counted). "Recurs" is judged against the 2026-07-21 waves by reading their finding titles.

| # | Root-cause mechanism | 2026-09-02 count | Recurs in 07-21 wave 1 | Recurs in 07-21 wave 2 | Mechanically detectable? |
|---|---|---:|---|---|---|
| E | Boundary / degenerate / empty input, off-by-one, null-vs-NaN, numeric formula | 38 | yes (`generate_combinations_recursive_njit` on negative input; `weekofmonth` off-by-one at every 7-day boundary) | yes (`get_topk_indices` returns NaN as highest; `keys_changed_enough` wrong sign) | mostly NO - see final section |
| B | Gate / config / packaging integrity (a declared check that does not check, or checks a narrowed scope) | 31 | yes (optional-deps guard blind spot; CI never installs an extras group in isolation) | yes (publish.yml tag check absent; API snapshot blind to defaults) | YES - MT-1, MT-2 |
| A | Docs / CHANGELOG / prose vs reality drift | 29 | yes (TESTING.md claims 24 aliases, map has 27; CHANGELOG stale by ~10 commits; `pyutilz.stats` missing from both module docs) | yes (CHANGELOG meta-test count stale in the same session) | YES - MT-5, MT-6 |
| I | API-surface / package-boundary / annotation-vs-reality inconsistency | 22 | yes (three registry patterns with no shared abstraction; alias-path imports) | yes (four parameter names for one pandas DataFrame; `prefixize_columns` return shape flips) | PARTLY - MT-3, MT-7 |
| C | Silent swallow / silent `None` return / wrong exception class | 18 | yes (`unserialize()` fails OPEN and swallows tamper detection; `execute_alchemy` swallows after retries) | yes (`TomlLiveConfig.get()` substitutes a hardcoded 0) | PARTLY - existing scanners, blind spots below |
| L | Network / retry / rate-limit / cost-accounting policy | 17 | yes (`get_url()` retries POST; no timeout on urlopen; Gemini retry predicate names wrong hierarchy) | yes (unbounded Redis retry logs a traceback per attempt) | PARTLY - existing `network_timeout`, `test_retry_predicate_matches_sdk_hierarchy` |
| J | Test-suite quality (vacuous / conditional / skipped / source-inspecting assertions) | 15 | yes (tautological assertion; whole file permanently skipped; `inspect.getsource()` as primary guard) | yes (untested Windows-reserved-name matching; untested normality branches) | YES - already closed in-wave by `test_code_audit_tests_baseline.py` |
| G | Resource lifecycle (leak, reassign-without-close, transaction left open) | 14 | yes (`open_safe_shelve` never closes; `close_browser` calls `.close()` not `.quit()`) | yes (server-side cursor never closed; `requests.Session` replaced without closing) | PARTLY - existing `test_resource_*`, blind spots below |
| H | Cache correctness / per-call recomputation / measurement hygiene | 14 | yes (S3 backend partial-view lost updates; legacy-migration claim file) | yes (`compute_code_version` memoizes by identity; catalogue cache with no TTL) | PARTLY - MT-8, MT-10 |
| K | Security (SQL injection, SSRF, path traversal, secret leak) | 11 | yes (`bAddUpdatedAtTimestamp` spliced into raw SQL; proxy credentials logged cleartext) | no dedicated pass | PARTLY - bandit covers some; MT-4 for the SQL-builder subclass |
| F | Concurrency / shared mutable state | 11 | yes (`distributed.py` check-then-act race; locale mutation) | yes (module-global cursor across threads; per-call LLM metadata on shared instance) | PARTLY - MT-9 |
| D | Parameter accepted and never used | 3 by category (5 by reading titles) | yes (`wait_for_absense_of_tasks`'s `labels` accepted but ignored; `create_tabs` tabTooltip computed then discarded) | no | YES - MT-3, and it is currently uncovered by every gate |

Reading the table: classes B and A together are 60 of 223 (27%), both recur in all three waves, and both are almost entirely mechanical. Class E is the single largest (38) and is almost entirely not mechanical. That asymmetry is the whole design brief.

## Existing coverage and its blind spots

### `tests/test_meta/` - 40 test modules, 15 baseline JSON files

Covered and working: alias integrity (`test_module_alias_integrity.py`), API snapshot (`test_api_stability.py` plus the `test_capture_signature.py` regression added after round 2 found its defaults blind spot), import cycles, top-level side effects, optional-dep isolation, bare `except`, mutable defaults, lazy logger formatting, non-ASCII console output, encoding on `open()`, resource-handle `with`-safety, resource reassignment without close, thread daemon/lifecycle, lock-discipline consistency, `logger.exception()` inside a handler only, retry-predicate-vs-SDK-hierarchy, sibling API parity (pandaslib/polarslib), verbose-param consistency, naming convention, deprecation warnings actually raised, dict/lru cache bounded-or-invalidatable, TODO hygiene, version consistency, test-source parity, `_USER_DEFERRED_*` debt drift.

Blind spots the 2026-09-02 wave exposed:

- **`test_optional_deps_isolation.py` probed package `__init__.py` only.** Already repaired in-wave: the module's own comment block now records both the 2026-07-21 fix (`system`/`gpu` groups were missing from `_OPTIONAL_DEP_GROUPS` entirely) and the 2026-09-02 one ("the same structural blind spot documented for `system`/`gpu` just ..."), which is finding `03/F06`. Worth naming as the archetype: the guard's coverage is a hand-maintained dict, and a hand-maintained dict silently omits.
- **`_code_audit_baseline.json` is line-anchored.** Its 116 entries are `scanner::path:line` strings (for example `"default_via_or::core/serialization.py:222"`). Any reformat, any insertion above a finding, and every downstream entry drifts, forcing a wholesale refresh that also silently absorbs genuinely new findings. `_docstring_baseline.json` (19 entries) and `_annotation_baseline.json` (141 entries) are keyed `path::symbol` and do not have this problem, so the fix is known and local.
- **`test_public_docstrings.py` plus interrogate: `ignore-init-module = true`** (`pyproject.toml`, `[tool.interrogate]`). A module docstring on any `__init__.py` is unenforced, so when a monolith is split into a package the new `__init__.py` inherits no requirement - the same shape as `05/F16` ("Twelve shipped modules appear in no documentation at all").
- **`test_lock_discipline_consistency.py` requires an existing lock.** Its rule is "a name assigned at least once inside `with <lock>:` must always be". A per-call attribute written on a shared provider instance with *no* lock anywhere is invisible to it, which is why `09/F08` (OpenRouter per-call metadata race, round 2 of the same defect), `09/F09` (`_last_json_schema_applied` clobbered) and `09/F58` (`token_counter._encoding_cache` unlocked) all survived it.
- **`test_resource_handle_safety.py` checks acquisition sites, not lifecycle transitions.** `09/F52` (`LoginAndGetCookies()` restarts Selenium without quitting the old driver) and `09/F11` (LRU eviction closes an httpx client another caller still holds) are lifecycle, not acquisition.

### `src/pyutilz/dev/code_audit/` - 59 scanner modules

Substantial and genuinely good coverage of the classes it targets: `bare_except`, `broad_except`, `silent_escalation`, `skip_masking_except`, `unraised_exceptions`, `network_timeout`, `mutable_defaults`, `missed_await`, `closures`, `nan_equality`, `mutation_during_iteration`, `sql_lint`, `sql_migrations`, `docstring_args`, `measurement_hygiene`, `vacuous_assertions`, `duplicate_conditions`, `near_duplicate_function_body`, `uncurated_star_export`, `domain_boundary`, `import_cycles`, `unpicklable_resource_state`, `readonly_to_numpy_mutation`.

Blind spots:

- **The test-quality scanners were never run against `tests/`** - finding `04/F01` ("The repo's own test-quality scanners are never run against `tests/` - 9 unfixed P1 findings they already detect"). Repaired in-wave by `tests/test_meta/test_code_audit_tests_baseline.py`. This is the single clearest case in the corpus of a defect class that was already fully detectable and simply not wired.
- **`network_timeout.py` recognises `requests`/`grequests`/`httpx`/`urllib` call attrs only** (its `_NETWORK_MODULE_HINTS` frozenset). An SDK client constructed without an explicit timeout - `09/F35`, `AnthropicProvider` layering tenacity on the SDK's own retries with no explicit timeout - is out of scope by construction.
- **`docstring_args.py` only fires on functions that already have a Google-style `Args:` section** (documented deliberately in its own header). It cannot see `07/F34` (`ensure_valid_filename`'s entire docstring documents a differently-named function, `src/pyutilz/core/pythonlib.py:670`) or `07/F23` (`split_list_into_chunks_indices`'s docstring documents the wrong function), because neither has an `Args:` mismatch - both have a wrong subject.
- **`measurement_hygiene.py` checks numeric claims inside source docstrings and comments against adjacent assertions.** It does not reach prose files (README, TESTING.md, CHANGELOG), which is exactly where the recurring stale-count defects live.

### `py-ci-shared/src/py_ci_shared/` - 21 modules, 1 consumed by pyutilz's meta-suite

Consumed: `code_audit_meta` (both baseline meta-tests, plus `register_refresh_option` in `tests/conftest.py:138`), `black_filtered_apply`, `format_warn`, `advisory_warn` (manual stage), and the `mypy-full-manual` hook from the repo.

**Not consumed, though directly applicable here:** `loc_budget`, `ci_workflow_timeout_gate`, `ci_workflow_gate`, `entry_points_resolvable`, `phantom_markdown_links`, `git_dependency_pins`, `ci_test_dir_reachability`, `changelog_promise_parity`, `config_call_site_parity`, `content_hash_version_bump_gate`, `readme_env_var_parity`. See MT-1.

### `.pre-commit-config.yaml` - about 15 hooks, most blocking

detect-secrets, actionlint, shellcheck, codespell, `ruff --ignore C901`, the meta-test suite, black-filtered (blocking), mypy beachhead, mypy full (blocking), bandit `-ll`, vulture 80, interrogate 100, deptry, yamllint, zizmor. Manual stage: static-only meta-tests, advisory ruff/mccabe/pip-audit.

Blind spots exposed this wave:

- **`ruff` never selects `ARG`.** The shared `configs/ruff-base.toml` `select` list is `E W F I N UP B DTZ A T20 RUF G ISC PERF C90 NPY`. `flake8-unused-arguments` is absent, so "a parameter is accepted and never referenced" is uncovered by every gate in the repo. Measured today: `ruff check src/pyutilz --select ARG --isolated --statistics` reports **40 findings** (27 `ARG002`, 13 `ARG001`) - and that is *after* this wave's fixes landed.
- **`mypy src/pyutilz` is blocking but neither clean nor deterministic** - `06/F01`, still OPEN. It aborts with an `INTERNAL ERROR` inside `site-packages/transformers/models/sam_hq/processing_samhq.py:77` and exits 2, so the reported error set depends on traversal order. The gate asserts an exit code; it never asserts the run *completed*.
- **The blocking ruff gate runs `--ignore C901`** (`06/F17`), and both ruff and mypy `exclude` `tests/` (`06/F08`, 297 hidden ruff findings including about 22 in real-bug classes). Both narrowings are deliberate; neither is visible anywhere as a declared, reviewed decision a check could verify.
- **`B006` (mutable default) is in the shared base's `ignore` list** (`06/F13`) while `tests/test_meta/test_no_mutable_defaults.py` enforces the same class locally with its own baseline. Two mechanisms, one disabled, no cross-reference between them.

## Proposed checks (ranked)

Ranking metric: (findings prevented x recurrence across waves) / (implementation cost + expected false-positive cost).

---

### MT-1. `shared-checks-wired` - consume the py-ci-shared modules that already exist

- **Disposition**: COMPLETED - tests/test_meta/test_shared_checks_wired.py wires 7 py-ci-shared modules (loc_budget, ci_workflow_gate, ci_workflow_timeout_gate, entry_points_resolvable, phantom_markdown_links, git_dependency_pins, ci_test_dir_reachability) with a `_loc_over_1k_baseline.json` at {}; found and fixed a missing `timeout-minutes` on ci.yml's ci-required job, plus a real bug in the shared ci_test_dir_reachability (it only recognised a literal `pytest tests/`, so a pathless CI command reported every subdir unreachable). 0.69 s, blocking.
- **What it detects**: nothing new. It closes the gap between "a check for this class exists and is maintained cross-project" and "this repo runs it".
- **Detection rule**: add pre-commit hooks and/or `tests/test_meta/` shims invoking, at minimum: `py_ci_shared.loc_budget` (threshold 1000, per CLAUDE.md's module-size rule), `py_ci_shared.ci_workflow_timeout_gate`, `py_ci_shared.ci_workflow_gate`, `py_ci_shared.phantom_markdown_links`, `py_ci_shared.entry_points_resolvable`, `py_ci_shared.git_dependency_pins`, `py_ci_shared.ci_test_dir_reachability`.
- **Findings it would have caught**:
  - `01/F06` "Five modules exceed the project's 1000-LOC split threshold, two of them flat siblings of already-split packages" (`src/pyutilz/text/similarity.py:1`) - `loc_budget`, exactly its remit.
  - `03/F14` "`publish.yml` jobs have no `timeout-minutes`, unlike every other workflow in the repo" (`.github/workflows/publish.yml:16-17,76-79`) - `ci_workflow_timeout_gate`, exactly its remit.
  - `03/F01` "[dev] extra carries a direct VCS URL dependency - PyPI rejects such distributions, so `publish.yml` cannot actually publish" (`pyproject.toml`) - `git_dependency_pins` is adjacent (it enforces SHA-pinning of git URLs) and would have surfaced the dependency for review, though its verdict differs.
  - `04/F01` "The repo's own test-quality scanners are never run against `tests/`" (`tests/test_meta/`) - the general shape; `ci_test_dir_reachability` is the shared analogue for the CI half of it.
- **False-positive sources**: `loc_budget` fires on the five known modules on day one, so it must land with a frozen allowlist of exactly those five and block growth beyond them - the same posture every other baseline here takes. `phantom_markdown_links` can trip on relative doc links resolved by mkdocs; scope it to README/CONTRIBUTING/TESTING/CHANGELOG first.
- **Runtime cost**: sub-second each, pure file reads. No new dependency - py-ci-shared is already in `requirements-dev.txt`.
- **Home**: py-ci-shared (already there); the wiring is pyutilz's `.pre-commit-config.yaml` plus `ci.yml`.
- **Block or warn**: block, with the day-one allowlists above.

---

### MT-2. `gate-actually-gates` - assert completion and scope parity for every declared-blocking gate

- **Disposition**: COMPLETED - py_ci_shared/gate_integrity.py + mypy_gate.py assert a gate COMPLETED, not merely exited 0: the mypy hook now demands `Success: no issues found in N source files` with N above a declared floor, and 18 declared gates are checked for scope/threshold parity (venue::gate::flag keys, never path:line). 0.07 s, blocking. Two rules deviate with evidence: command/version parity is unreachable while CI's tool pins live in py-ci-shared's reusable workflows, and rule 4 became CI-vs-pyproject coverage-floor parity because measuring coverage cannot run per commit.
- **What it detects**: a gate that is declared blocking but (a) cannot fail, (b) can pass without having run to completion, (c) runs a narrower scope in one venue than in the other, or (d) has a threshold set below the value it is supposed to defend.
- **Detection rule**: a meta-test that parses `.pre-commit-config.yaml` and `.github/workflows/*.yml` and asserts, per gate:
  1. **Command parity** - the local hook `entry` and the CI `run:` for the same tool resolve to the same argv modulo path, including the pinned tool version.
  2. **Completion assertion** - for tools whose exit code alone is ambiguous, the invocation must be wrapped so the tool's own success terminator is required. For mypy that is the `Success: no issues found in N source files` line.
  3. **Declared narrowings** - every `--ignore`, `--select`, `exclude` and `continue-on-error` affecting a blocking gate must appear in an in-repo allowlist with a one-line reason. A new narrowing fails until it is declared.
  4. **Threshold vs measured** - a `--cov-fail-under` more than N points below the last measured coverage fails.
- **Findings it would have caught**:
  - `06/F01` [High, still OPEN] "`mypy src/pyutilz` - a declared-clean BLOCKING gate - aborts with an INTERNAL ERROR and exit code 2, making its findings nondeterministic" (`pyproject.toml:551`). Rule 2 catches it directly: no `Success: no issues found in 113 source files` line was ever printed.
  - `06/F17` "22 `C901` complexity findings are permanently advisory - the gate that would enforce them is invoked with `--ignore C901` in both CI and pre-commit". Rule 3.
  - `06/F08` "`tests/` is excluded from BOTH ruff and mypy, hiding 297 ruff findings including ~22 in real-bug classes" (`pyproject.toml:421`). Rule 3.
  - `06/F16` "The `tests.*` mypy override is dead config - mypy is never invoked on `tests/` from anywhere" (`pyproject.toml:563`). Rule 3, from the other direction.
  - `06/F07` "The local blocking mypy hook runs an unpinned mypy (`mypy>=1.0`, resolving to 1.8.0 here) while CI pins 2.1.0 - a MAJOR-version split". Rule 1.
  - `04/F08` "CI coverage gate is 20 points below actual coverage" (`.github/workflows/ci.yml:77`). Rule 4.
  - `03/F14` `publish.yml` timeouts (also MT-1).
  - Cross-wave: 2026-07-21 round 2's "`publish.yml` never verifies the pushed git tag matches `pyproject.toml`'s `[project].version` before publishing" is the same class; `tests/test_meta/test_publish_workflow_version_check.py` is the one-off regression written for it, where this would be the general rule.
- **False-positive sources**: legitimate venue differences (a manual-stage hook is *supposed* to be narrower than CI; a Windows-only step). Handled by rule 3's allowlist - the check does not judge whether a narrowing is right, only whether it was declared. That keeps the false-positive rate near zero at the price of a one-time declaration pass over about fifteen hooks.
- **Runtime cost**: YAML/TOML parsing, milliseconds. The completion assertion adds nothing - it inspects output already produced.
- **Home**: py-ci-shared. Every consumer repo has the identical hook/workflow duality, and `ci_workflow_gate.py` is already the seed for rule 3.
- **Block or warn**: block.

---

### MT-3. `unused-parameter-baseline` - enable ruff `ARG` behind a frozen snapshot

- **Disposition**: COMPLETED - tests/test_meta/test_unused_parameter_baseline.py freezes ruff ARG001/ARG002 keyed path::function::param, each entry requiring a written justification. Triage of the 40: 8 real bugs fixed (get_external_ip ignored all four proxy arguments and returned the machine's own IP; download_to_file ignored rewrite_existing; the Claude Code CLI transport silently dropped temperature/max_tokens; decodo summary ignored group_by), 32 baselined. 1.28 s, blocking. ARG was deliberately not added to the shared ruff select: mlframe reports 1245 findings today, so it would break both repos' blocking gate at once.
- **What it detects**: a parameter accepted in a signature (and usually documented) that is never referenced anywhere in the body.
- **Detection rule**: add `ARG` to `configs/ruff-base.toml`'s `select`, and in pyutilz freeze today's 40 findings into `tests/test_meta/_unused_param_baseline.json` keyed `path::function::param` - never `path:line`, per the `_code_audit_baseline.json` lesson above. Growth fails; shrinkage refreshes.
- **Findings it would have caught**:
  - `09/F06` [High] "`min_failed_idle_interval_minutes` is accepted, documented, threaded through three call sites, and never used" (`src/pyutilz/web/web.py:499, 542-545`). The finding's own Problem text is verbatim the `ARG001` rule: "the identifier does not appear anywhere in the function body after the docstring". A High, entirely uncovered, that would have been caught the day the parameter was added.
  - `09/F32` "`ClaudeCodeProvider` silently ignores `max_tokens` and `temperature`" (`src/pyutilz/llm/claude_code_provider.py:363, 365, 540-570`) - `ARG002` on the method parameters `_generate_sdk` drops.
  - `09/F19` "`auto_commit` is accepted by three public functions and does nothing" (`src/pyutilz/database/db/__init__.py:294, 327-328, 418, 423`) - caught at the pass-through wrappers where the name is never referenced; **not** at `basic_db_execute` itself, which does read it (`src/pyutilz/database/db/execution.py:72`). Partial credit, stated honestly.
  - Cross-wave, 2026-07-21 full audit: "`wait_for_absense_of_tasks`'s `labels` parameter is accepted but ignored" (`src/pyutilz/system/scheduling/prefect.py`) and "`create_tabs`: documented `tabTooltip` tuple element is computed then silently discarded" (`src/pyutilz/dev/dashlib.py`).
- **False-positive sources**: real and well-understood - interface/ABC conformance (an `LLMProvider` subclass accepting a parameter one backend cannot honour), callback signatures fixed by a third-party API, `**kwargs` forwarding shims, pytest fixtures. Ruff's `ARG` already skips `_`-prefixed names via `dummy-variable-rgx`; overridden abstract methods are the main residue. The 40-item day-one baseline is what makes this affordable: nothing is fixed under duress, only growth is stopped.
- **Runtime cost**: zero marginal - ruff already runs on every commit.
- **Home**: the `select` entry belongs in py-ci-shared (`configs/ruff-base.toml`, cross-project); the baseline file belongs in pyutilz's `tests/test_meta/`.
- **Block or warn**: block on growth.

---

### MT-4. `generated-sql-property-test` - parse every SQL string the builders emit

- **Disposition**: COMPLETED - tests/test_meta/test_generated_sql_property.py parses everything the builders emit with sqlglot (postgres dialect) across 1536 upsert x 512 db_command option combinations plus hostile identifiers, asserting no unbounded UPDATE/DELETE, no empty column list, ON CONFLICT targets present, placeholder/parameter parity and named refusals. Verified against the pre-fix sources: it catches 09/F01, 09/F16 and the 2026-07-21 db_command defect. Found one NEW live bug - the 09/F17 guard covered mode=insert only, so mode=update with empty set_fields still built invalid SQL - fixed and pinned. 4.96 s, blocking.
- **What it detects**: a query builder that emits syntactically invalid SQL, or valid-but-catastrophic SQL (an `UPDATE`/`DELETE` with no `WHERE`), for some reachable combination of its own parameters.
- **Detection rule**: a runtime property test over `pyutilz.database.db.upsert.build_upsert_query` and `pyutilz.database.db.sql_helpers.db_command`, driven by a bounded cartesian product of their boolean/optional parameters (present, absent, empty list, `None`) over a fixed toy schema. For each generated string assert: (1) it parses with `sqlglot.parse_one` (or `sqlparse` as a lighter fallback); (2) any `UPDATE` or `DELETE` produced has a `WHERE` clause, or is explicitly whitelisted; (3) the `%s` placeholder count matches the bound parameter tuple.
- **Findings it would have caught**:
  - `09/F01` [**Critical**] "`build_upsert_query` emits an `UPDATE` with no `WHERE` when `timestamp_update_fields` is used without `hash_fields`" (`src/pyutilz/database/db/upsert.py`). Assertion 2, exactly.
  - `09/F16` "`build_upsert_query` builds invalid SQL when `history_table_name` is given with empty `history_fields`" (`src/pyutilz/database/db/upsert.py`). Assertion 1.
  - `09/F17` "`db_command(mode="insert", set_fields=None)` crashes with an opaque `TypeError`" (`src/pyutilz/database/db/__init__.py:457-473`) and `09/F43` "`db_command(returning=None)` raises an opaque `TypeError`" (`:499, 510`) - the sweep reaches both combinations and turns an opaque crash into a named failing case.
  - Cross-wave, 2026-07-21 full audit: "`db_command(mode="update", ...)` always builds syntactically-invalid SQL" (`src/pyutilz/database/db/sql_helpers.py:87-90`), "`build_upsert_query` silently ignores `conflict_fields` (falls back to a bare, unscoped `on conflict do nothing`)", "`db_command` crashes or builds malformed SQL on empty/missing `where_fields`". Round 2: "`build_upsert_query`'s `timestamp_update_fields` / `custom_onconflict` / `fields_types` / `skip_fields` / `on_conflict_up...` [untested]".
  - Recurrence verdict: **the SQL-builder family has produced High-or-worse findings in all three waves.** It is the highest-recurrence single *code* surface in the corpus.
- **False-positive sources**: low. A parameter combination that is genuinely unsupported should raise a typed error, and the test can assert exactly that instead of a parse - which is itself the fix `09/F17` and `09/F43` asked for. The real cost is enumerating which combinations are legal, and that enumeration is the currently-missing documentation.
- **Runtime cost**: a few hundred string builds plus parses, well under a second. Adds `sqlglot` (or `sqlparse`) as a dev dependency.
- **Home**: pyutilz `tests/` - the builders are pyutilz-specific, though the harness shape generalises if another repo grows a query builder.
- **Block or warn**: block.

---

### MT-5. `prose-numeric-claim-parity` - every counted fact in prose is computed, not typed

- **Disposition**: COMPLETED - py_ci_shared/prose_numeric_claims.py + tests/test_meta/test_prose_numeric_claims.py compute four prose counts from their anchors instead of trusting the typed number, and fail when a claim loses its anchor (the failure mode a naive registry hides). Caught real drift within the hour: the mypy source-file count went stale as modules were added. 0.30 s, blocking. The date-qualifier half warns only - measured 8 findings at a 75% false-positive rate, and a rule people learn to ignore is worse than none.
- **What it detects**: a hardcoded number in README/TESTING/CONTRIBUTING/CHANGELOG/docs that names a countable repo fact (test count, alias count, meta-test file count, coverage percentage, module count, suite runtime) and no longer matches the fact.
- **Detection rule**: a registry mapping a regex anchor in a named file to a callable that computes the truth - for example `("README.md", r"(\d[\d,]*)\+? tests", collected_test_count)`, `("TESTING.md", r"(\d+) backward-compat aliases", lambda: len(pyutilz._MODULE_ALIASES))`, `("CHANGELOG.md", r"(\d+) meta-test", lambda: len(glob("tests/test_meta/test_*.py")))`. A mismatch fails, naming both numbers. A claim that cannot be computed must carry a date qualifier (`as of YYYY-MM-DD`), or the check fails on the undated claim itself.
- **Findings it would have caught**:
  - `05/F08` "\"1900+ tests\" is understated by ~1600 tests; the paired 79.6% coverage figure is undated and uncorroborated" (`README.md:274`) - both halves, the second by the date rule.
  - `04/F21` "TESTING.md's headline numbers are stale and its documented coverage command fails on Windows" (`TESTING.md:3`).
  - `05/F11` "Documented runtimes are ~6x optimistic for the meta-test suite (measured 187 s vs \"~30 s\")" (`TESTING.md:11`) - the same stale claim also sits in `.pre-commit-config.yaml`'s own header comment ("Total runtime: about 30 s").
  - Cross-wave, 2026-07-21 full audit: "TESTING.md and the meta-test's own docstring both claim \"24\" backward-compat aliases; the actual map has 27 entries" and "CHANGELOG.md meta-test file count is stale". Round 2: "CHANGELOG's own meta-test count is already stale as of the follow-up commit in the same session".
  - **Recurrence verdict: all three waves, every time.** The single most reliably recurring documentation defect in the corpus.
- **False-positive sources**: prose legitimately quoting a historical number ("as of v1.0.0 there were 24 aliases"). Handled by the same date qualifier the check already requires for uncomputable claims.
- **Runtime cost**: the collection-count callables are the expensive ones (`pytest --collect-only -q` is a few seconds). Put the cheap ones in pre-commit and the collection-dependent ones in CI only.
- **Home**: py-ci-shared. `measurement_hygiene.py` is the in-source analogue; this is its prose sibling, and mlframe has the same README-counts problem.
- **Block or warn**: block.

---

### MT-6. `docs-inventory-parity` - documented inventories are computed from the thing they document

- **Disposition**: COMPLETED - py_ci_shared/docs_inventory_parity.py + tests/test_meta/test_docs_inventory_parity.py compute documented extras, module inventories and doc-referenced paths/markers from the thing they document. Found four genuine live drifts nobody knew about (README's [web], [nlp] and [dev] bullets each omitted members, CONTRIBUTING named an unresolvable path, CHANGELOG linked a moved audit file) and caught a dependency another agent added minutes earlier. 0.09 s; inventory rules block, the undocumented-module rule warns.
- **What it detects**: three specific inventory drifts - (a) a README/docs bullet listing an extras group's members that disagrees with `[project.optional-dependencies]`; (b) a shipped module absent from every module-orientation doc; (c) a command, path, glob or pytest marker named in docs that does not exist.
- **Detection rule**: parse `pyproject.toml`'s `optional-dependencies` and `[tool.pytest.ini_options] markers`; parse the extras bullets out of README's install section by their `` `[name]` `` anchor; diff the sets. For (b) walk `src/pyutilz/**/*.py` and require each module path to appear in `README.md` or `docs/modules.md`. For (c) extract backtick-quoted paths and globs from the four prose files and `os.path.exists`/glob them, and extract `@pytest.mark.<name>` mentions and check them against the declared markers list.
- **Findings it would have caught**:
  - `05/F12` "`[system]` extras description lists three packages that are now core and omits the group's two heaviest members" (`README.md:28`); `05/F13` "`[llm]` extras description omits `pydantic-settings` and `tiktoken`" (`README.md:27`); `05/F14` "`[dev]` extras description omits six pytest plugins and `py-ci-shared`" (`README.md`); `05/F15` "The install block omits 8 of the 16 declared extras groups, including `[stats]` and `[docs]`" (`README.md:19`) - all four by rule (a).
  - `03/F05` "`[all]` silently omits `dash`, `prefect`, `tensorflow` (and `gpu`), so the README-recommended \"full install\" cannot import three shipped modules" and `05/F05` "`[all,dev]` is called the \"full install (recommended)\" but omits four extras groups" - rule (a), comparing the prose promise to the resolved group.
  - `03/F15` "CHANGELOG's 1.0.0 entry advertises a `pypiwin32` member of `[system]` that no longer exists anywhere" (`CHANGELOG.md:50`; `pyproject.toml:250`) - rule (a) extended to CHANGELOG.
  - `05/F16` "Twelve shipped modules appear in no documentation at all" (`README.md:48`); `05/F09` "`pyutilz.data.git_checkpoint_cache` - a CHANGELOG headline addition - is absent from both module-orientation docs" (`README.md:49`); `05/F17` "CHANGELOG's \"Module Categories\" reference section is stale by ~19 modules" (`CHANGELOG.md:82`) - rule (b).
  - `05/F04` "CONTRIBUTING tells contributors to use `@pytest.mark.integration`, which is a collection ERROR under this repo's `--strict-markers`" (`CONTRIBUTING.md`); `05/F10` "CONTRIBUTING points at `tests/benchmark_*.py`; benchmarks actually live in `_benchmarks/bench_*.py`" (`CONTRIBUTING.md:357`); `05/F02` "`pre-commit install` is instructed in three docs but `pre-commit` is in no extras group" (`README.md:38`) - rule (c).
  - Cross-wave, 2026-07-21 full audit: "`pyutilz.stats` subpackage missing from both module-orientation docs", "`pyutilz.dev.code_audit` - a CHANGELOG headline feature - missing from both module-orientation docs", "CONTRIBUTING's example test-run command references a test that doesn't exist" (`CONTRIBUTING.md:150`), "`pyproject.toml`'s own documentation comment for the `database` extras group cites a stale file path".
- **False-positive sources**: rule (b) is the risky one - a deliberately-undocumented private helper module would fire. Mitigate with an explicit `_UNDOCUMENTED_BY_DESIGN` set, the same convention as the existing `_USER_DEFERRED_*` whitelists already drift-tracked by `test_deferred_drift.py`. Rule (c) can trip on shell-fragment backticks; scope it to strings containing `/` or a known marker prefix.
- **Runtime cost**: file reads and one TOML parse. Milliseconds.
- **Home**: py-ci-shared - `readme_env_var_parity.py` is the identical pattern already implemented for env vars, so this is a sibling module, not a new concept.
- **Block or warn**: block for (a) and (c); warn for (b) until the twelve-module backlog is closed, then block.

---

### MT-7. `facade-and-exception-root-integrity`

- **Disposition**: COMPLETED - tests/test_meta/test_facade_and_exception_root_integrity.py asserts every subpackage is reachable and declares __all__, every provider in the factory registry resolves through the llm facade, each exceptions module has exactly one root every class reaches, and exported exceptions are actually raised. Proven against the reconstructed 01/F01, 01/F02, 01/F03, 01/F04 and 01/F09 shapes. 5.7 s, blocking.
- **What it detects**: (a) a subpackage under `src/pyutilz/` unreachable from the package facade; (b) a class registered in an in-repo registry but absent from its package's `__all__`; (c) an exception class in a package's `exceptions.py` that does not subclass that package's declared root; (d) an exception class defined and exported but never raised anywhere.
- **Detection rule**: pure AST plus one import of `pyutilz`. (a) the set of directories under `src/pyutilz/` minus the set of attributes reachable on the imported `pyutilz` module must be empty. (b) for `pyutilz.llm.factory._PROVIDER_MODULES`, every named class must appear in `pyutilz.llm.__all__`. (c) parse `**/exceptions.py`, take the class with no in-module base as the root, assert every other class transitively reaches it. (d) grep `raise <Name>` across `src/`.
- **Findings it would have caught**:
  - `01/F01` [High] "`OpenAIProvider` is missing from `pyutilz.llm`'s public surface while every sibling provider is exported" (`src/pyutilz/llm/__init__.py:6`) - rule (b).
  - `01/F02` [High] "Two of the six LLM exception types bypass the `LLMProviderError` root, so `except LLMProviderError` cannot catch all LLM errors" (`src/pyutilz/llm/exceptions.py:15`) - rule (c).
  - `01/F03` "`pyutilz.stats` and `pyutilz.performance` are invisible from the package facade - `pyutilz.stats` raises AttributeError" (`src/pyutilz/__init__.py:9`) and `01/F04` "`pyutilz.performance/__init__.py` is the only subpackage `__init__` with no `__all__` and no submodule binding" (`src/pyutilz/performance/__init__.py:1`) - rule (a).
  - `01/F09` "`database` and `web` ship typed-exception modules that neither package exports nor lists in `__all__`" (`src/pyutilz/database/__init__.py:3`) - rules (a) and (b).
  - Cross-wave, 2026-07-21 full audit: "`LLMTruncationError` is fully specified but never raised anywhere" (`src/pyutilz/llm/exceptions.py:45-53`) - rule (d). `code_audit/unraised_exceptions.py` already covers that half; that `01/F02` still landed says the *hierarchy* half is the gap.
- **False-positive sources**: rule (d) has a real one - an exception intended for downstream consumers to raise. One-line allowlist. Rules (a) to (c) are essentially FP-free because they compare two in-repo declarations against each other.
- **Runtime cost**: milliseconds; the package import is already performed by the meta-suite.
- **Home**: pyutilz `tests/test_meta/` - the facade shape and registry names are repo-specific. Rule (c) alone could move to py-ci-shared later.
- **Block or warn**: block.

---

### MT-8. `gpu-timing-requires-synchronize`

- **Disposition**: COMPLETED - py_ci_shared/gpu_timing_sync.py detects GPU work timed without a device synchronize, purely by AST, offline. It departs from the sketched rule where it matters: the Critical timed an opaque injected callable, so a callee-name rule would have missed it - enclosing-function parameters are carried into nested closures instead. Verified to fire on the pre-fix benchmark.py and to be silent on the fixed tree. 3.24 s, blocking; lives in py-ci-shared because mlframe has the same exposure.
- **What it detects**: a wall-clock measurement taken around a CUDA/cupy/numba.cuda call with no device synchronize between the launch and the second timestamp, so the "measurement" times the launch, not the work.
- **Detection rule**: AST. Inside any function, find a `time.perf_counter()`/`time.time()` assignment, a subsequent call whose callee resolves to a `cuda.`/`cupy.`/`cp.`/kernel-dispatch name, and a second timestamp read - with no `cuda.synchronize()`, `.synchronize()`, `cp.cuda.Stream.null.synchronize()` or `cp.cuda.runtime.deviceSynchronize()` call between them. Report P0.
- **Findings it would have caught**:
  - `10/F01` [**Critical**] "`time_backend` times CUDA kernels without a device synchronize - the kernel-tuning cache can be populated with phantom wins" (`src/pyutilz/performance/kernel_tuning/`). A Critical whose blast radius is persisted, cross-session and cross-project - the kernel-tuning cache is consumed by mlframe - which is what lifts it above its finding count of one.
  - Adjacent, 2026-07-21 round 2: "`benchmark_backends`/`time_backend` have zero per-variant exception isolation" (`src/pyutilz/performance/kernel_tuning/`) - same function, previous wave, so the surface is a repeat offender even though the specific defect differs.
- **False-positive sources**: a deliberately async launch-latency measurement. Rare enough for a `# noqa`-style allowlist entry, and such a measurement should be labelled anyway.
- **Runtime cost**: one AST pass over `src/`, tens of milliseconds, folded into the existing `code_audit` self-scan.
- **Home**: **py-ci-shared / `code_audit`** - the clearest cross-project win in the list. mlframe is GPU-heavy and this defect class is a standing entry in the maintainer's own reference notes ("cProfile inflates GPU-op tottime", "Synchronize GPU timings").
- **Block or warn**: block, as a P0 in the existing baseline mechanism.

---

### MT-9. `per-call-state-on-shared-instance`

- **Disposition**: COMPLETED - src/pyutilz/dev/code_audit/per_call_state_on_shared_instance.py flags per-call state written to a shared instance, deriving sharedness from module-level registries and lru_cache'd factories and propagating it to in-tree base classes (without which 09/F09's attribute on OpenAICompatibleProvider stays invisible). Reports 13 live findings, all in ClaudeCodeProvider.generate - the same defect class on the factory's default provider, left on the warn list per the proposal. 1.4 s, warn.
- **What it detects**: an instance attribute written during an in-flight `async def` (or from a method reachable from one) with no lock, on a class whose instances are cached and shared - per-call metadata masquerading as instance state.
- **Detection rule**: AST. For each class, collect attributes assigned inside `async def` bodies. Flag an attribute if (1) it is assigned in at least one `async def`, (2) it is read in a different method (a `last_*` or summary accessor), and (3) no `async with self._lock` or `threading.Lock` guards the assignment. Restrict to classes whose instances are returned from a cache (`functools.lru_cache`, or a module-level dict keyed by name) to keep the scope honest.
- **Findings it would have caught**:
  - `09/F08` [High] "OpenRouter's entire per-call metadata set is still the cross-request race round 2 described" (`src/pyutilz/llm/openrouter_provider/_provider.py`).
  - `09/F09` [High] "`_last_json_schema_applied` is clobbered across concurrent calls, so the strict-schema guarantee flag lies" (`src/pyutilz/llm/openai_compat.py`).
  - `09/F57` "`last_call_summary()` mixes real values with `PerCallAttr` defaults after a batch" (`src/pyutilz/llm/openrouter_provider/_provider.py`).
  - `09/F53` "`_reset_per_call_state` is a no-op for every provider except OpenRouter" (`src/pyutilz/llm/openai_compat.py:263-270`).
  - Cross-wave, 2026-07-21 round 2: "LLM providers' \"last_*\" per-call metadata is shared **instance** state, silently misattributed across concurrent requests" - **literally the same defect, one wave earlier, on the same surface**, which is why `09/F08`'s own title says "round 2".
  - `09/F58` "`token_counter._encoding_cache` is mutated without a lock" (`src/pyutilz/llm/token_counter.py:51-60`) is the module-global cousin; `test_lock_discipline_consistency.py` misses it because no lock exists anywhere in that module to anchor on.
- **False-positive sources**: moderate. A single-caller-by-contract provider, or an attribute that is genuinely append-only. Expect a handful of allowlist entries. This is the first proposal in the list where the FP rate is a real cost rather than a rounding error, which is why it sits at 9.
- **Runtime cost**: one AST pass, milliseconds.
- **Home**: pyutilz `code_audit` (the provider shape is pyutilz's; the rule generalises if it proves out).
- **Block or warn**: **warn** for one wave, then block once the allowlist stabilises.

---

### MT-10. `uncached-constant-cost-probe`

- **Disposition**: COMPLETED - src/pyutilz/dev/code_audit/uncached_constant_cost_probe.py flags parameterless functions that spawn a process, load a DLL, probe the filesystem or import by name with nothing memoizing the result. Proven against the reconstructed 10/F02, 10/F04, 10/F09 and 10/F11. Reports 14 live candidates (nvidia-smi, lscpu and power-plan probes among them) as a performance-triage list. 0.7 s, warn only.
- **What it detects**: a function performing an expensive, effectively-constant probe (subprocess shell-out, `ctypes.WinDLL`/`CDLL` construction, `os.makedirs`, a capability check) on every call, with no caching decorator and no module-level memo.
- **Detection rule**: AST. Flag a function that (1) takes no parameters or only defaulted config parameters, (2) contains a call to `subprocess.*`, `ctypes.WinDLL`/`CDLL`, `os.makedirs` or `importlib.import_module`, and (3) carries no `@lru_cache`/`@cache`/`@cached_property` and writes no module-level memo. Report P2.
- **Findings it would have caught**:
  - `10/F02` [High] "`gpu_capability_summary` / `occupancy_aware_block_size` re-shell-out to nvidia-smi on every call - 64 ms per dispatch decision".
  - `10/F04` [High] "`is_cuda_available()` is uncached and re-probes numba on every call - `dispatch_cpu_vs_gpu` costs 18-24 us per decision".
  - `10/F09` "`cache_dir()` / `host_cache_dir()` `makedirs(exist_ok=True)` on every call - 258 us and 0.5-1.6 ms per call for a directory that already exists" - a shape the maintainer's own reference note ("mkstemp != makedirs": `makedirs(exist_ok=True)` measured 2.89x slower than exists-then-skip) has already been burned by once.
  - `10/F11` "`_pid_alive` constructs a fresh `ctypes.WinDLL(\"kernel32\")` on every call" (`src/pyutilz/performance/kernel_tuning/cache/cache_base.py`).
- **False-positive sources**: high-ish by design - a probe that *must* be fresh (a liveness check, a config reload) looks identical to one that must not. `_pid_alive` itself must re-check the pid but need not rebuild the DLL handle, so the rule flags the right function for a partly wrong reason. Hence: triage list, not verdict.
- **Runtime cost**: one AST pass.
- **Home**: pyutilz `code_audit`.
- **Block or warn**: **warn only**, feeding a periodic performance triage rather than a commit gate.

---

## Rejected ideas

Considered and deliberately not proposed, with the reason:

- **A general empty/degenerate-input fuzzer over every public function.** Would nominally target the largest bucket (38 findings in class E). Rejected: the harness cannot distinguish "crashes on empty input, and should" from "crashes on empty input, and should not", so it either asserts nothing useful or demands a hand-written expectation per function - at which point it is just the test suite. A narrow version scoped to the dataframe-cleanup family may be worth revisiting after MT-4 proves the property-test pattern here.
- **A docstring-semantics-vs-implementation checker.** Would target `07/F34`, `07/F23`, `07/F35` ("`show_methods` documents \"non-dunder\" but excludes any name containing a double underscore anywhere", `src/pyutilz/core/pythonlib.py:88`) and the three `doc-behavior-mismatch` findings. Rejected as stated: matching prose meaning to code meaning is not a static analysis. The one mechanical sliver - "the docstring's first line names a function that is not this function" - is real but would have caught only `07/F34` and `07/F23`, and belongs as a five-line addition to `docstring_args.py`, not as a proposal of its own.
- **An off-by-one / boundary detector.** `09/F12` (sliding-window loop misses the last window), `07/F08` (`count_trailing_zeros` counts the integer part's zeros too), `07/F36` (`wait_for_absense_of_tasks` polls `max_retries + 1` times), `07/F09` (`str_to_class` slices off the closing parenthesis it documents as included). Every one is arithmetic that is only wrong relative to an intent no annotation records. Rejected outright.
- **A `[-n:]`-with-possibly-zero-n lint** (would catch `07/F24`, "`DeadLetterQueue(max_size=0)` grows without bound - the `[-0:]` trim is a whole-list slice", `src/pyutilz/system/resilience.py:327`). Rejected as a standalone check - too narrow to justify a hook - though it is a legitimate three-line rule inside an existing scanner if one is being touched anyway.
- **A "silent `return None` on multiple distinct paths" scanner.** Would target `07/F26`, `07/F27` ("`unserialize` returns `None` for success, for file-not-found, for wrong type, and for any exception - the four are indistinguishable") and `09/F47`. Rejected because `code_audit/silent_escalation.py` and `broad_except.py` already cover the swallow half, and the "N distinct outcomes collapse to one sentinel" half needs to know which outcomes are distinct - back to intent.
- **Enabling ruff `B006` repo-wide** (`06/F13`). Rejected: `tests/test_meta/test_no_mutable_defaults.py` already enforces this class with its own baseline, and the shared base's `ignore` entry documents the case-by-case decision. Turning it on duplicates an existing gate and churns. The genuine gap is that neither mechanism references the other - a comment, not a check.
- **A cross-repo `pyproject.toml` config-drift gate.** `py_ci_shared.config_drift_check` already exists and reports it; making it blocking would fail pyutilz's commits for mlframe's divergence. Leave it advisory.

## What is not mechanically detectable

Stated plainly, because proposing a check for these would be worse than proposing nothing:

- **Numeric and algorithmic correctness.** `08/F01` ("`np.allclose`'s default rtol=1e-5 makes the \"constant column\" test drop columns with real variation", `src/pyutilz/data/polarslib.py:1055`), `08/F12` ("`showcase_df_columns`'s uninformative-fraction divides by the full row count even when `dropna=True`"), `09/F13` ("`compute_entropy_stats` returns a large negative \"entropy\"", `src/pyutilz/text/strings/textentropy.py:108`), `09/F55` ("`_normalize_uptime` reads a genuine 1% uptime as 100%"), `07/F25` ("`benchmark_algos_by_runtime` logs the LAST repetition's duration while reporting and sorting on the MINIMUM"). Each is a correct-looking expression computing the wrong quantity. Only a test that independently knows the right quantity can see it.
- **Cache-coherence semantics.** `07/F01` ("`DiskCache` LRU eviction destroys the MOST-recently-used entry, because `get()` touches the payload but never its sidecar", `src/pyutilz/core/disk_cache.py`) is a two-file invariant no AST rule can express.
- **Intent-dependent parameter interactions.** `07/F37` ("`get_running_flows` silently ignores `allof_labels` whenever `anyof_labels` is also supplied", `src/pyutilz/system/scheduling/prefect.py:105-112`) - the code reads both parameters, so no unused-argument rule fires; whether the `elif` is a bug depends entirely on what the two filters are for.
- **Domain-knowledge thresholds.** `08/F07` ("`normality_verdict` reports \"too-few-samples\" for 8 <= n < 20 although Anderson-Darling is valid there"), `09/F41` ("A single-character prefix overlap scores 0.91+"). Correct code, wrong statistics.
- **Whether a documented promise is kept by a vendor API.** `09/F28` ("`GeminiProvider.generate_json()` never uses Gemini's native JSON mode"), `09/F25` ("`DecodoProvider.get_traffic()` fetches only page 1 and reports the truncated sum as the total"). Detecting these requires knowing what the vendor offers.
- **Security judgements about untrusted input.** `02/F03` ("Server-controlled `Retry-After` header drives an unbounded `time.sleep`", `src/pyutilz/web/cached_client.py:123`), `02/F04` ("Path traversal: `tag` is used as an unvalidated directory component of the cache path", `:104`). Bandit sees neither, because neither is a dangerous *call*; both are a trust boundary a human has to draw.

The honest bottom line: MT-1 through MT-8 would have prevented roughly thirty findings outright, and would have caught the recurring gate-integrity, docs-drift and SQL-builder classes at the moment each was introduced rather than one audit wave later. They would not have prevented the largest single bucket, and nothing reasonable would have.
