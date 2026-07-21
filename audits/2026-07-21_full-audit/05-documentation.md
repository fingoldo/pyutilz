# Documentation accuracy & quality audit — pyutilz

## Summary

Read in full: README.md, CONTRIBUTING.md, TESTING.md, CLAUDE.md, CHANGELOG.md, docs/index.md, docs/modules.md,
docs/guides/kernel_tuning_cache.md, docs/guides/llm_providers.md, docs/guides/safe_pickle.md. Cross-checked every
README/guide code snippet's imports, class names, and call signatures against the current `src/pyutilz` tree;
cross-checked CHANGELOG's "[Unreleased]" section against `git log`; sampled ~20 public docstrings across
core/data/dev/system/web/llm/database/text subpackages for Args/Returns accuracy, not just presence.

The three `docs/guides/*.md` files are in genuinely good shape — every API name, signature, default, and env-var
name I checked against `kernel_tuning/*`, `llm/*`, and `safe_pickle.py` matched the live code exactly (one very
minor wording nit in safe_pickle.md aside). The weaker spots are: (1) README.md's Modules table and docs/modules.md
are missing an entire real subpackage (`pyutilz.stats`) and a headline new subsystem (`pyutilz.dev.code_audit`);
(2) CHANGELOG.md's "Unreleased" section is stale by ~10 commits including a behaviorally significant one
(`py.typed` marker addition); (3) one of README's own flagship "Quick examples" snippets (parameterised-SQL
security example, duplicated verbatim in docs/modules.md) is not runnable as written; (4) CONTRIBUTING.md has
several internal-consistency problems beyond the already-known black/ruff one c(coverage bar vs. undisclosed
omit-list, version prerequisite vs. version-gated dev deps, a missing pre-commit setup step, and a copy-paste test
command that references a test that doesn't exist); (5) a couple of source docstrings (`matrix.py`,
`llm/base.py`) are stale/wrong relative to the code they describe.

Findings: 1 High, 7 Medium, 10 Low.

## Findings

### [High] README's flagship parameterised-SQL example is not runnable as written — README.md:254 (dup. docs/modules.md:15)
- **Category**: docs/correctness
- **Problem**: The "Parameterised SQL with identifier validation" snippet in the **Security**-adjacent Quick
  Examples section is:
  ```python
  table = validate_sql_identifier(user_input)             # raises on injection
  rows = safe_execute("SELECT * FROM {} WHERE id = %s", (table, user_id))
  ```
  `safe_execute(statement, data=None, ...)` (`src/pyutilz/database/db/__init__.py:322-323`) is a thin pass-through
  to `basic_db_execute`, which calls `cur.execute(statement, data)` verbatim (`src/pyutilz/database/db/__init__.py:261`)
  — a raw DB-API `cursor.execute(sql, params)` call. Nothing in the call chain ever calls `.format()` /
  `str.format` on `statement`. The literal string `"SELECT * FROM {} WHERE id = %s"` therefore reaches the driver
  with the `{}` placeholder UN-substituted (Python doesn't auto-format string literals), while `data=(table,
  user_id)` supplies **two** bind values for a string that contains exactly **one** `%s` placeholder.
- **Failure scenario**: Copy-pasting this exact snippet and calling it against a live connection raises a
  DB-API error at execute time (psycopg2: `TypeError: not all arguments converted during string formatting`,
  since 2 params are bound against 1 placeholder) — or, in adapters that error-check placeholder counts, an
  outright `ProgrammingError`/`IndexError`. The query never runs. Worse, this is the library's own worked example
  of "safe" identifier interpolation; a reader who "fixes" it by removing `validate_sql_identifier` and doing
  `statement.format(table, user_id)` on the WHOLE string (interpolating both the identifier and the value) would
  reintroduce the exact SQL-injection hole the example claims to close.
- **Suggested fix**: `rows = safe_execute("SELECT * FROM {} WHERE id = %s".format(table), (user_id,))` (table name
  formatted in after validation; only the value goes through the parameter placeholder) — and apply the same fix
  in docs/modules.md:15, which repeats the identical broken snippet verbatim.

### [Medium] `pyutilz.stats` subpackage missing from both module-orientation docs — README.md:46-57, docs/modules.md (whole file)
- **Category**: docs/completeness
- **Problem**: `src/pyutilz/stats/` (containing `normality.py`, a numba-jitted D'Agostino/Anderson-Darling normality
  test module) is a real, packaged, actively-typed subpackage — it's one of only two mypy strict-mode beachheads
  in the whole repo (`pyproject.toml:433`: `module = ["pyutilz.dev.code_audit.*", "pyutilz.stats.*"]`) — but it has
  zero rows in README.md's "Modules" table (lines 46-57 list core/data/database/web/cloud/system/performance/
  text/dev/llm — 10 rows, no `pyutilz.stats`) and zero paragraphs in docs/modules.md (which mirrors the same 9
  sub-package list, again omitting stats).
- **Failure scenario**: A user reading either doc to decide what pyutilz offers has no way to discover
  `dagostino_k2` / `anderson_darling_normal` / `normality_verdict` exist at all — the only two files with
  strict-mypy discipline in the repo are invisible to the "orientation" docs that are supposed to be the front
  door.
- **Suggested fix**: Add a `pyutilz.stats` row to README's Modules table and a corresponding paragraph to
  docs/modules.md, one sentence each, matching the style of the existing entries.

### [Medium] `pyutilz.dev.code_audit` — a CHANGELOG headline feature — missing from both module-orientation docs — README.md:56, docs/modules.md:33-35
- **Category**: docs/completeness
- **Problem**: `pyutilz.dev.code_audit` is an 18-file AST-based bug-class scanner with its own CLI
  (`python -m pyutilz.dev.code_audit <root>`), explicitly called out as a new capability in CHANGELOG.md's
  "[Unreleased] > Added" section (CHANGELOG.md:12: "`pyutilz.dev.code_audit` — AST-based scanner + CLI for four
  recurring bug classes...") and it's the *other* of the two mypy strict-mode beachheads (`pyproject.toml:433`).
  Yet README.md's `pyutilz.dev` table row (line 56: "Logging, benchmarking, dashboards, Jupyter helpers, meta-test
  utilities") and docs/modules.md's `pyutilz.dev` paragraph (lines 33-35: "Logging setup, benchmarking helpers,
  dashboards, Jupyter notebook helpers, and meta-test utilities...") both omit it entirely. `grep -rn
  "code_audit" README.md docs/modules.md docs/index.md` returns zero matches across all three orientation docs.
- **Failure scenario**: A user who reads CHANGELOG's Unreleased section, sees `code_audit` advertised, then goes
  to README's Modules table or docs/modules.md to learn more about it (the natural next step, since those docs are
  explicitly the "orientation" reference) finds no mention of it at all.
- **Suggested fix**: Add "AST-based scanner + CLI for common bug classes (`code_audit`)" to the `pyutilz.dev` row
  in both docs.

### [Medium] CHANGELOG "[Unreleased]" is stale by ~10 landed commits, including a behaviorally significant one — CHANGELOG.md:8-30
- **Category**: docs/process
- **Problem**: CHANGELOG.md was last edited 2026-07-06 (confirmed via `git log -1 --format=%cd -- CHANGELOG.md`).
  `git log` shows at least these commits landed since, none reflected anywhere in the Unreleased section:
  `39f423e` (2026-07-19, gate black dev-extra pin on python_version>=3.10), `fcbbc2a` (2026-07-19, bump
  black-filtered.yml pin to v1.2.1), `43aa4e6` (2026-07-18, sync py-ci-shared pins to v1.2.0), `445abce`
  (2026-07-17, perf fix in `pythonlib.get_parent_func_args`), `29c19d3` (2026-07-16, **add `py.typed` marker**),
  `0bd5062` (2026-07-16, clear python 3.8 CI leg bugs), `deea501` (2026-07-16, add code_audit redundant-test-fit
  scanner), `cee8480`/`9a0500d` (2026-07-15, mypy fixes), `31d5560` (2026-07-15, meta_test_utils sentinel fix for
  PEP 604 unions). The `py.typed` addition in particular is not cosmetic: per its own commit message, it's the
  difference between mypy silently treating every pyutilz import as `Any` (arity/type errors invisible across the
  package boundary) versus real type-checking for downstream consumers — exactly the kind of "notable change" a
  changelog exists to surface, and the CHANGELOG's own past entries (e.g. 1.0.0's "~30 hardware-detection
  functions migrated") show packaging-relevant additions are normally logged.
- **Failure scenario**: A downstream consumer (e.g. mlframe) reads CHANGELOG.md before upgrading and has no signal
  that pyutilz now ships `py.typed`, so they don't know to expect (and investigate) new mypy findings against
  their own call sites that a previous pyutilz version silently didn't catch.
- **Suggested fix**: Add an `### Added` / `### Fixed` entries for the `py.typed` marker, the `get_parent_func_args`
  perf fix, and the code_audit redundant-test-fit scanner at minimum before the next release.

### [Medium] `check_account_limits()` "native support" claim overstated for DeepSeek; OpenAI/xAI silently drop captured headers — docs/guides/llm_providers.md:27, README.md:105-108, CHANGELOG.md:15
- **Category**: docs/correctness
- **Problem**: Three docs assert the same shape of claim — "`get_account_credits()` / `check_account_limits()` —
  works natively where the upstream exposes it (OpenRouter, DeepSeek); other providers fall back to capturing
  ... headers" (llm_providers.md:27); "Account credits work where the upstream API exposes them (OpenRouter,
  DeepSeek); other providers fall back to capturing ... headers automatically" (README.md:106-108); "Unified
  `get_account_credits()` / `check_account_limits()` across every LLM provider (OpenRouter and DeepSeek implement
  them...)" (CHANGELOG.md:15). Checking every concrete provider (`grep -rn "def check_account_limits" src/pyutilz/llm`):
  only `openrouter_provider/_provider.py:593` has a real dedicated-endpoint implementation. `deepseek_provider.py`
  does **not** override `check_account_limits` at all — it inherits `openai_compat.py:252`'s generic
  header-capture fallback, i.e. exactly the same "fallback" behavior the docs attribute only to "other providers",
  not to DeepSeek. Meanwhile `openai_provider.py:170-178` and `xai_provider.py:152-156` **explicitly override**
  `check_account_limits` to always `raise NotImplementedError` — even though `openai_compat.py:442,608` DOES call
  `_capture_rate_limit_headers(resp.headers)` on every real call for these same providers, so the header data the
  docs describe as the universal fallback is captured into `self.last_rate_limits` and then deliberately never
  surfaced for OpenAI/xAI's `check_account_limits()`. So the real 3-way split is: OpenRouter = native endpoint;
  Anthropic + DeepSeek = header-fallback (not "native"); OpenAI + xAI = hard-raise despite having captured headers;
  Gemini = hard-raise (never captures headers at all, not an OpenAI-compat subclass); Claude Code = CLI shell-out.
  None of the three docs describe this correctly.
- **Failure scenario**: A caller reads "works natively... (OpenRouter, DeepSeek)" and expects
  `DeepSeekProvider().check_account_limits()` to behave like OpenRouter's dedicated introspection call; in fact it
  raises `NotImplementedError` until at least one `generate()` call has already succeeded (same caveat as every
  other OpenAI-compat provider). Separately, a caller who has already made an OpenAI/xAI `generate()` call
  (so `last_rate_limits` is populated) and calls `check_account_limits()` expecting the documented "fall back to
  captured headers" behavior instead unconditionally gets `NotImplementedError`, even though the exact same
  captured data would answer the question — the code choice to hard-raise anyway isn't a doc bug per se, but the
  docs claim behavior the code doesn't have.
- **Suggested fix**: Reword to: get_account_credits native for OpenRouter + DeepSeek only; check_account_limits
  native only for OpenRouter, header-fallback for Anthropic + DeepSeek, and hard-`NotImplementedError` (by design)
  for OpenAI/xAI/Gemini regardless of captured headers, CLI-based for Claude Code.

### [Medium] `get_sparse_memory_usage` docstring claims CSC support it doesn't have — src/pyutilz/core/matrix.py:86-99
- **Category**: docs/correctness
- **Problem**: Docstring: `"""Return mem usage of a csr or csc matrix"""` (line 88). The implementation only
  branches on `isinstance(mat, csr_matrix)` (line 91) and `isinstance(mat, coo_matrix)` (line 93) — `csc_matrix`
  is never checked, and there's no `import` of `csc_matrix` in the file at all (only `csr_matrix, coo_matrix` are
  imported at line 23). Any other type, including a real `scipy.sparse.csc_matrix`, falls into the `else` branch
  and silently returns `-1`.
- **Failure scenario**: `get_sparse_memory_usage(scipy.sparse.csc_matrix(...))` — a type the docstring explicitly
  claims support for — silently returns `-1` (not an exception, not a warning) instead of the actual byte count.
  A caller summing memory usage across a list of mixed csr/csc matrices for a budget check gets a systematically
  wrong (understated) total with no indication anything went wrong.
- **Suggested fix**: Either add a `csc_matrix` branch (`mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes`,
  same CSR-family layout) or fix the docstring to say "csr or coo" and make the `else` branch raise/log at
  warning level for unsupported types instead of returning a silently-plausible `-1`.

### [Medium] CONTRIBUTING.md repo-wide `black .` / `ruff check --fix .` conflicts with CLAUDE.md's rule (confirmed still present) — CONTRIBUTING.md:82,91,285
- **Category**: docs/consistency
- **Problem**: CONTRIBUTING.md's "Style Guide" section literally instructs `black .` (line 82) and
  `ruff check --fix .` (line 91) as the way to format/lint the whole project, and the PR Checklist repeats
  "Code formatted with black (`black .`)" (line 285). CLAUDE.md (this project's own binding convention doc)
  states as its top, CRITICAL-severity rule: "NEVER run a project-wide lint/format rewrite without explicit
  approval" and documents that raw `black`/`black --fix` must never be run in place — the project's actual
  formatter is `python -m py_ci_shared.black_filtered_apply --config pyproject.toml --write <files>`
  (CLAUDE.md:11), because stock Black's arg/collection-list explosion and blank-line insertion are deliberately
  excluded project-wide.
- **Failure scenario**: A new contributor follows CONTRIBUTING.md's literal PR checklist item, runs `black .` at
  the repo root, and reformats every file in the tree using behaviors (arg-list explosion, blank-line insertion)
  the project has explicitly decided to exclude — producing a PR with thousands of unrelated formatting-only diff
  lines that would need to be reverted, mirroring the exact incident CLAUDE.md documents having already happened
  once (2026-07-05).
- **Suggested fix**: Replace CONTRIBUTING.md's raw `black .` / `black --check .` instructions with the
  `py_ci_shared.black_filtered_apply` invocation, and point at CLAUDE.md's rule explicitly.

### [Medium] CONTRIBUTING's ">80% coverage for new code" bar is unenforceable for a large, undisclosed omit-list — CONTRIBUTING.md:164,275 vs pyproject.toml:229-239
- **Category**: docs/consistency
- **Problem**: CONTRIBUTING.md states "Coverage: Aim for >80% coverage for new code" (line 164) and "Add tests
  for new functionality (maintain >80% coverage)" (line 275) as PR requirements, with no caveats. The project's
  actual aggregate, quoted identically in README.md:270 and TESTING.md:3, is **79.6%** — already below the bar
  CONTRIBUTING sets for new code — and that 79.6% is itself measured against a shrunk denominator:
  `pyproject.toml:229-239`'s `[tool.coverage.run] omit` list excludes `*/scheduling/*`, `*/cloud/*`,
  `*/web/browser.py`, `*/dev/dashlib.py`, `*/dev/notebook_init.py`, and `*/text/tokenizers.py` from coverage
  measurement **entirely** — not "measured and low", but never counted at all. CONTRIBUTING.md never mentions
  this omit-list exists.
- **Failure scenario**: A contributor adds new functionality inside e.g. `pyutilz.web.browser` (an entirely
  omitted file) and follows the PR checklist's "maintain >80% coverage" literally — but `pytest --cov` never
  reports any coverage number for that file at all, so there is no tooling feedback telling them whether they hit
  0% or 100% on their own new code. The stated policy is silently inapplicable to contributions in six whole
  areas of the codebase, and CONTRIBUTING.md gives no indication of this.
- **Suggested fix**: Either note the omit-list and its rationale in CONTRIBUTING.md's Testing section, or (if
  those modules should eventually be covered) track that as separate debt rather than silently excluding them
  from the stated policy.

### [Low] CHANGELOG.md "Module Categories" reference section lists a module deleted at v1.0.0 — CHANGELOG.md:86
- **Category**: docs/staleness
- **Problem**: The freeform "Module Categories" section at the bottom of CHANGELOG.md, under "### Specialized",
  lists: "image, filemaker, com, openai" (line 86). `com` (a Windows COM-automation helper) was deliberately
  deleted during the src-layout restructuring: `git log --all --oneline -- "src/pyutilz/com.py"` finds it only in
  commit `57975e9` ("Restructure to src-layout with subpackages (v1.0.0)"), whose own message says "Cleaned up
  leftover files (com.py, ruff_errors.txt)". `find src/pyutilz -iname "com*.py"` today returns nothing.
- **Failure scenario**: Low-impact by itself (this section reads as a historical/freeform index, not an API
  promise), but a reader skimming it for "what modules exist" is told a `com` module exists when it was removed
  roughly 5 months earlier.
- **Suggested fix**: Drop `com` from the list, or note it was retired at v1.0.0.

### [Low] CHANGELOG.md meta-test file count is stale — CHANGELOG.md:30
- **Category**: docs/staleness
- **Problem**: "### Meta-test infrastructure — 45 tests across 18 files under `tests/test_meta/`..." The test
  *count* (45) is still exactly right (`grep -rc "^def test_" tests/test_meta/*.py` sums to 45), but the *file*
  count is stale: `find tests/test_meta -maxdepth 1 -type f -name "*.py"` currently returns 24 files (25 including
  `__init__.py`), not 18 — consistent with the several `test(meta): ...` / `feat(dev.code_audit): ...` commits
  that landed test_meta files after this line was written (e.g. `test_code_audit_baseline.py`).
- **Failure scenario**: Minor — a reader undercounts how much static-guard machinery exists.
- **Suggested fix**: Update to the current file count when next touching this section (or drop the specific file
  count and just say "under `tests/test_meta/`" to avoid re-staling it).

### [Low] `LLMProvider.check_account_limits` base docstring is stale — src/pyutilz/llm/base.py:388-390
- **Category**: docs/correctness
- **Problem**: "Default raises :class:`NotImplementedError`. Concrete providers override when they expose this
  (currently only OpenRouter)." In fact 6 of the 7 concrete provider classes define `check_account_limits`
  (`anthropic_provider.py:273`, `claude_code_provider.py:761`, `gemini_provider.py:348`, `openai_provider.py:170`,
  `xai_provider.py:152`, `openrouter_provider/_provider.py:593`) — only `deepseek_provider.py` relies on the
  inherited `openai_compat.py` fallback without its own override. This is the source-level root of the
  documentation-guide finding above (docs/guides/llm_providers.md over-claiming "currently only OpenRouter"-style
  scoping).
- **Failure scenario**: A future contributor reading this docstring while adding an 8th provider concludes (per
  the docstring) that "override only if you have a special case; otherwise the default `NotImplementedError` is
  what happens" — when in fact the established pattern in this codebase is that every provider DOES override it,
  each with its own provider-specific rationale/message.
- **Suggested fix**: Update the parenthetical to reflect the actual per-provider split (native/header-fallback/
  hard-raise/CLI), matching the corrected guide text above.

### [Low] `matrix.py` module docstring is duplicated; the more complete version is dead code — src/pyutilz/core/matrix.py:1-3
- **Category**: docs/correctness
- **Problem**:
  ```python
  """Incremental builders and utilities for scipy sparse (CSR/COO) matrices."""

  """Incremental builders and memory-usage helpers for scipy sparse (CSR/COO) matrices."""
  ```
  Only the first string literal in a module becomes `__doc__`; the second is a no-op expression statement that is
  parsed and immediately discarded. Verified: `python -c "import pyutilz.core.matrix as m; print(m.__doc__)"`
  prints the shorter first string, not the more complete second one that presumably was meant to replace it (it
  mentions "memory-usage helpers", which the file does provide via `get_sparse_memory_usage`, but the live
  `__doc__` doesn't mention that).
  - **Failure scenario**: Anything that introspects `pyutilz.core.matrix.__doc__` — `help()`, mkdocs'
  docstring-driven API pages, IDE tooltips — shows the stale, less-complete description; the edit that added the
  better description silently never took effect.
- **Suggested fix**: Delete line 1 (or line 3), keeping the more complete "...and memory-usage helpers..." wording
  as the actual module docstring.

### [Low] `sweep_backend_crossover` Args section omits 4 of its parameters — src/pyutilz/dev/benchmarking.py:116-164
- **Category**: docs/completeness
- **Problem**: The function signature (lines 116-131) has 13 parameters. The docstring's `Args:` block
  (lines 138-157) documents `variants`, `sizes`, `make_inputs`, `primary_axis`, `reference`, `extra_region_keys`,
  `repeats`, `synchronize_gpu`, `ranking` — 9 of them. `equiv_atol`, `equiv_rtol`, `decision_key`, and `verbose`
  are declared in the signature but never mentioned in `Args:`.
- **Failure scenario**: A caller reading the docstring to decide the equivalence tolerance for their sweep (a
  correctness-relevant knob — regions failing `equiv_atol`/`equiv_rtol` get silently dropped per the function's
  own equivalence-gating logic) has to read the source instead of the docstring to learn those params exist.
- **Suggested fix**: Add the four missing entries to the `Args:` block.

### [Low] `sweep_backend_grid` Args section omits 7 of its parameters — src/pyutilz/dev/benchmarking.py:316-370
- **Category**: docs/completeness
- **Problem**: Same pattern as above, more pronounced: signature has 13 parameters; `Args:` (lines 351-364)
  documents only `variants`, `axes`, `make_inputs`, `residencies`, `to_device`, `ranking` — 6 of 13. Missing:
  `reference`, `repeats`, `equiv_atol`, `equiv_rtol`, `synchronize_gpu`, `decision_key`, `verbose`.
- **Failure scenario**: Same as above — a caller can't learn from the docstring alone how to control the
  equivalence tolerance, repeat count, or GPU-sync behavior of a full-grid sweep; only the `Returns:` shape and a
  minority of the knobs are documented.
- **Suggested fix**: Add the missing entries (this function and `sweep_backend_crossover` share most of these
  parameter names, so the fix is largely copy-paste between the two).

### [Low] CONTRIBUTING's "Python 3.8 or higher" prerequisite hides that its own recommended `pip install -e .[all,dev]` silently skips two dev tools on 3.8/3.9 — CONTRIBUTING.md:46, 57 vs pyproject.toml:177-191
- **Category**: docs/consistency
- **Problem**: CONTRIBUTING.md's "Prerequisites" (line 46) says only "Python 3.8 or higher", and "Setup
  Instructions" (line 57) has the contributor run `pip install -e .[all,dev]` uniformly regardless of Python
  version. `pyproject.toml:182` pins `black==26.5.1 ; python_version >= '3.10'` and `pyproject.toml:191` pins
  `py-ci-shared @ git+... ; python_version >= '3.9'` — both with explicit markers, precisely so `pip install`
  doesn't hard-fail on 3.8/3.9 (per the inline comments at pyproject.toml:177-181, 188-190). CONTRIBUTING.md never
  mentions this: it presents one Python-version-blind setup path, then two paragraphs later (lines 78-92)
  instructs `black .` and (implicitly, via `.pre-commit-config.yaml`, never mentioned either — see next finding)
  the py-ci-shared-backed pre-commit hooks, both of which are silently absent from the environment a 3.8/3.9
  contributor was just told to set up.
- **Failure scenario**: A contributor on Python 3.8 runs the documented `pip install -e ".[all,dev]"`, gets no
  error, then tries `black .` per the Style Guide section and gets `command not found` (black was never
  installed) with no clue why, since CONTRIBUTING.md gave no indication dev tooling completeness depends on their
  interpreter version.
- **Suggested fix**: Note in Prerequisites that black/pre-commit tooling requires Python >=3.9/3.10 respectively,
  or recommend a specific interpreter version for contributors doing dev-tooling work.

### [Low] CONTRIBUTING.md never mentions `pre-commit install`, unlike README.md and TESTING.md — CONTRIBUTING.md (whole file)
- **Category**: docs/consistency
- **Problem**: `grep -n "pre-commit" CONTRIBUTING.md` returns zero matches. Yet README.md's own "For development"
  setup block explicitly includes it as step 3 of 4 (README.md:34-39: clone, `pip install -e ".[all,dev]"`,
  `pre-commit install`, `pytest`), and TESTING.md dedicates a whole section to it ("Pre-commit hook", explaining
  it "runs the meta-test suite on every commit"). CONTRIBUTING.md's own "Setup Instructions" (lines 49-61) stops
  at `pip install -e .[all,dev]` + `pytest`, omitting the pre-commit step entirely.
- **Failure scenario**: A contributor who onboards via CONTRIBUTING.md specifically (the file whose whole purpose
  is "how do I contribute") never installs the pre-commit hook, so their first several commits skip the ~30-60s
  meta-test gate locally and any static-guard violation (e.g. a new bare `except:`, a new mutable default) is only
  caught later in CI instead of at commit time — the exact local-feedback loop TESTING.md/CLAUDE.md describe as
  the point of the hook.
- **Suggested fix**: Add `pre-commit install` to CONTRIBUTING.md's Setup Instructions, matching README.md.

### [Low] CONTRIBUTING's example test-run command references a test that doesn't exist — CONTRIBUTING.md:150
- **Category**: docs/correctness
- **Problem**: "Run specific test" example: `pytest tests/test_pandaslib.py::test_optimize_dtypes`. There is no
  `test_optimize_dtypes` function in `tests/test_pandaslib.py` — `grep -n "^def test_" tests/test_pandaslib.py`
  lists only `test_list_length_check`. The closest real tests are `test_optimize_dtypes_does_not_truncate_
  fractional_object_column` and `test_optimize_dtypes_still_converts_genuine_integer_object_column`, both in a
  different file, `tests/test_dtypes_regression.py`.
- **Failure scenario**: A new contributor copy-pastes this exact command (it's presented as a literal, runnable
  example, distinct from the illustrative "Example test" code block later in the same doc) and gets
  `ERROR: not found: ... no tests ran` with no test named that in that file.
- **Suggested fix**: Point at a real test, e.g. `pytest tests/test_dtypes_regression.py::test_optimize_dtypes_still_converts_genuine_integer_object_column`.

### [Low] safe_pickle.md slightly understates the temp-file naming scheme — docs/guides/safe_pickle.md:14
- **Category**: docs/precision
- **Problem**: "`safe_dump` writes atomically (dump to a per-process temp file, fsync, `os.replace` onto the final
  path)..." The actual temp filename (`src/pyutilz/core/safe_pickle.py:212`) is
  `f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"` — per-**process-and-thread**, not merely per-process. The
  distinction is exactly what makes two threads in the same process concurrently `safe_dump`-ing the same path
  not collide on the same temp filename (the module's own docstring at safe_pickle.py:198-199 spells this out:
  "two concurrent same-key writers -- even two threads in the same process -- never interleave").
- **Failure scenario**: None functionally (the code is correct); purely a documentation-precision nit — a reader
  of the guide alone (without opening the source) would come away thinking two threads in one process writing the
  same path concurrently might race on the temp file, which is not the case.
- **Suggested fix**: Change "per-process temp file" to "per-process-and-thread temp file" in the guide.

## Things done well

- The three `docs/guides/*.md` files (kernel_tuning_cache, llm_providers, safe_pickle) are excellent — every
  function name, class name, default value, env-var name, and behavioral claim I checked against the live
  `kernel_tuning/*`, `llm/*`, and `safe_pickle.py` implementations matched exactly, including subtle things like
  the exact hw_fingerprint() format string, the `PYUTILZ_KERNEL_CACHE_DIR` override name, the instance-cache
  `(canonical_name, kwargs_key)` shape, and the WeakSet/atexit cleanup mechanism. This is a high bar and the docs
  clear it.
- safe_pickle.md's threat-model caveat is copied verbatim from the module's own docstring and is honest about what
  the sidecar does and doesn't protect against (corruption vs. tamper-resistance) — a real security-literacy
  strength, not just accuracy.
- README's Quick Examples generally cite real, current function/class names and correct call signatures — I
  verified `optimize_dtypes`, `benchmark_dataframe_compression`, `showcase_df_columns`, `get_llm_provider`,
  `list_openrouter_models`, `PortHealthTracker`, `strip_ai_patterns`/`introduce_typos`/`humanize`,
  `SentenceSimilarityIndex`, `get_max_affordable_workers_count`/`applyfunc_parallel`, `get_system_info`,
  `KernelTuningCache`/`hw_fingerprint`, `safe_dump`/`safe_load`, `timeout_wrapper`/`log_duration`, and
  `validate_sql_identifier` — all match current signatures/defaults exactly (only `safe_execute`'s call in the
  same section is broken, see High finding).
- The 24-entry backward-compat module alias map (`pyutilz.pandaslib` -> `pyutilz.data.pandaslib`, etc.) genuinely
  works — I initially suspected CONTRIBUTING.md's `from pyutilz.pandaslib import optimize_dtypes` example was a
  stale pre-restructuring import path, but it resolves correctly at runtime via the alias system.
- CI test-matrix and coverage-tooling claims ("Tested on 3.8 through 3.14", "1900+ tests", codecov badges, dual
  numba-disabled coverage run) all check out against `.github/workflows/ci.yml` and a static count of `def test_`
  across `tests/`.
- mkdocs.yml's nav correctly references all three guide files and both index.md/modules.md — no broken links in
  the docs site structure.

## Investigated, not an issue

- Suspected `CONTRIBUTING.md`'s `from pyutilz.pandaslib import optimize_dtypes` (line 171) was a stale
  pre-restructuring import path (the real module is `pyutilz.data.pandaslib`) — ruled out: `pyutilz.pandaslib` is
  a genuine, working backward-compat alias (`src/pyutilz/__init__.py:20`), confirmed by actually importing it.
- Suspected the README/docs `generate_stream()` "OpenAI-compatible providers only" scoping claim was inaccurate —
  checked `grep -rln "def generate_stream" src/pyutilz/llm/`: it's defined exactly once, in `openai_compat.py`,
  the shared base for openai/deepseek/xai/openrouter; anthropic/gemini/claude-code don't subclass it. Claim is
  accurate.
- Suspected the Deferred-work section's specific unimplemented-feature claims (DeepSeek FIM endpoint, xAI deferred
  chat completions, Gemini cachedContents full lifecycle) might already be implemented and just forgotten from the
  "Deferred" list — spot-checked all three against the actual provider files; none are implemented, list is
  accurate.
- Suspected "DeepSeek default model switched to deepseek-v4-flash" (CHANGELOG:22) might be stale — checked
  `deepseek_provider.py:57`: `model: str = "deepseek-v4-flash"` — still accurate.
- Suspected the "1900+ tests, 79.6% line coverage" figure itself (as opposed to the coverage omit-list issue
  raised above) might be inflated or stale — a static `grep -c "^def test_"` sum across `tests/*.py` +
  `tests/test_meta/*.py` + `tests/{performance,stats,system}/*.py` totals ~1909, consistent with "1900+" once
  parametrized-test expansion at collection time is accounted for (this only makes the real number larger, not
  smaller). Not flagged as an issue.
- Suspected "SQL/command-injection risks fixed (historical, 0.90)" in CHANGELOG's Unreleased/Fixed section was
  miscategorized (0.90 already has its own dedicated changelog section documenting the same fix) — this does read
  as a slightly odd duplicate mention, but it's clearly labeled "(historical, 0.90)" inline, so it's not
  misleading, just a mild redundancy; not worth a separate finding.
- Checked whether `pyutilz.core.openai.py` (listed under CHANGELOG's "Specialized" category alongside the removed
  `com` module) still exists and is current — it does (`src/pyutilz/core/openai.py`, 76 lines) — only `com` was
  stale in that list.
