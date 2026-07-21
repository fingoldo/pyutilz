# Backward-Compatibility & Versioning Risk Audit

## Summary

I read `CHANGELOG.md` and `pyproject.toml` in full, then verified their claims against actual git history: `git log`/`git show` on the commits behind the most consequential Unreleased entries (`5685867` "Remove 14 accidentally-public leaked imports", `445abce` "materialize FrameLocalsProxy locals", `a083702` "Implement all 204 audit findings", `206a69d` "Add the remaining 2 code_audit scanners"), plus `.github/workflows/publish.yml`, `CONTRIBUTING.md`, `tests/test_meta/test_version_consistency.py`, `tests/test_meta/test_api_stability.py`, `src/pyutilz/dev/meta_test_utils.py` (`capture_signature`/`capture_module_surface`), `src/pyutilz/__init__.py` (diffed against the v1.0.0-restructure commit `57975e9`), `src/pyutilz/system/__init__.py`, and `src/pyutilz/data/pandaslib/frames.py` in full. Several claims were empirically verified by running code against this checkout (`PYTHONPATH=src python -c ...`), not just read: the 14-symbol leaked-import count, the `capture_signature` blind spot to default-value changes, and the `pyutilz.system` lazy-attribute `dir()` behavior. I also confirmed `pyutilz` has never been published to PyPI (`pip index versions pyutilz` → no distribution found, network confirmed working via a control query against `requests`), which materially affects how much real-world exposure the "BREAKING" changes currently carry.

Total findings: 1 HIGH, 3 MEDIUM, 2 LOW.

## Findings

### [HIGH] API-stability snapshot cannot detect default-parameter-value changes — `src/pyutilz/dev/meta_test_utils.py:180-205`
- **Category**: versioning-safety-net gap
- **Problem**: `capture_signature()`, the function backing `tests/test_meta/test_api_stability.py`'s public-API drift detector (the test whose own docstring calls it "the highest-value defensive test for a backward-compat-oriented library"), captures only `name:kind:has_default` per parameter:
  ```python
  has_default = p.default is not inspect.Parameter.empty
  params.append(f"{name}:{kind_short}:{int(has_default)}")
  ```
  The actual default *value* is never captured or compared. Two functions `def f(verbose: bool = False)` and `def f(verbose: bool = True)` produce the byte-identical signature string `(verbose:any:1)`.
- **Failure scenario**: I verified this directly:
  ```
  >>> capture_signature(lambda verbose=False: None)
  '(verbose:any:1)'
  >>> capture_signature(lambda verbose=True: None)
  '(verbose:any:1)'
  >>> equal -> True
  ```
  A future PR that flips any existing public default (e.g. `dropna: bool = False` → `True`, `inplace: bool = False` → `True`, `use_uint: bool = True` → `False` — all real parameter names in `pandaslib`) — exactly the "default-parameter-value change that silently alters behavior for callers who don't pass it explicitly" this audit angle targets — sails through `test_public_api_matches_snapshot` with zero diff reported, on the exact test suite built specifically to catch API drift. Every other 26-file meta-test-suite invariant (renames, removals, docstring coverage, import cycles) is enforced; this one specific, high-value class silently isn't, and nothing in the test's docstring or the CHANGELOG's "Meta-test infrastructure" blurb (`CHANGELOG.md:31-32`) discloses the gap.
- **Suggested fix**: Extend `capture_signature` to also record `repr(p.default)` for defaults of "safe to serialize" types (bool/int/float/str/None/tuple-of-literals; skip objects/sentinels that would churn), and compare it in `test_api_stability.py`. Even a coarse `bool`/`None`/numeric-literal-only capture would catch the common, highest-risk case.

### [MEDIUM] `pyutilz.system` eager→lazy submodule conversion is a real behavior change, undocumented in CHANGELOG — `src/pyutilz/system/__init__.py`, `CHANGELOG.md`
- **Category**: undocumented behavior change
- **Problem**: Commit `a083702` (`git show a083702 -- src/pyutilz/system/__init__.py`) converted `parallel`, `monitoring`, `hardware_monitor`, and `UtilizationMonitor` from eager imports (`from . import parallel, monitoring, hardware_monitor, system`) to PEP 562 lazy `__getattr__` resolution. This is a deliberate, reasoned fix (avoids forcing `[system]`'s pandas/psutil/tqdm stack onto callers who only need `pyutilz.system.system`), but it changes real, observable behavior: `import pyutilz.system` no longer eagerly triggers (or fails on) `parallel`/`monitoring`/`hardware_monitor`'s own transitive imports, and those names are absent from `dir(pyutilz.system)` / the module `__dict__` until first attribute access.
- **Failure scenario**: Verified empirically:
  ```
  >>> import pyutilz.system as s
  >>> 'parallel' in dir(s)
  False
  >>> s.parallel   # first access
  >>> 'parallel' in dir(s)
  True
  ```
  Any downstream code that does `hasattr(pyutilz.system, "parallel")`, enumerates `pyutilz.system.__dict__`/`dir()` for plugin discovery, or relied on `import pyutilz.system` failing immediately with a clear `ModuleNotFoundError: pandas` at import time (now deferred to wherever `pyutilz.system.parallel.<anything>` is first touched, potentially far from the actual missing-dependency cause) sees different behavior than before. `CHANGELOG.md`'s Unreleased section documents unrelated smaller items (`tqdmu_lazy_start`, DeepSeek pricing, etc.) but has zero mention of this conversion — grepped for "lazy"/"eager"/"parallel"/"monitoring"/"hardware_monitor" and the only hits are unrelated (`tqdmu_lazy_start`, the Module Categories list).
- **Suggested fix**: Add a CHANGELOG "Changed" entry describing the lazy-loading conversion and its two user-visible effects (deferred `[system]`-extras `ImportError`, `dir()`/`hasattr` no longer reflecting these names pre-access).

### [MEDIUM] `remove_stale_columns` renamed with a docstring-only "Deprecated" notice — no `DeprecationWarning` ever raised — `src/pyutilz/data/pandaslib/frames.py:371-379`
- **Category**: silent deprecation
- **Problem**: The function was renamed to `get_non_stale_columns` (more accurate name — it doesn't mutate/remove anything) and the old name kept as a compat shim:
  ```python
  def remove_stale_columns(X: pd.DataFrame) -> list:
      """Deprecated alias for :func:`get_non_stale_columns` -- kept for backward compatibility.
      ...
      """
      return get_non_stale_columns(X)
  ```
  No `warnings.warn(..., DeprecationWarning)` call. A repo-wide grep confirms `DeprecationWarning` is never used anywhere in `src/pyutilz` (only referenced in two comments about *suppressing* Python's own coroutine-related warnings), and the two modules that `import warnings` at all only call `warnings.catch_warnings()`/`filterwarnings("ignore")`, never `warn()`.
- **Failure scenario**: A caller using `remove_stale_columns` (still exported from `pyutilz.data.pandaslib.__init__` and present in `_api_snapshot.json`) gets no signal — not at call time (no runtime warning), not via static tooling (no `@deprecated` decorator, no `.. deprecated::` Sphinx directive an IDE/linter would surface) — that they should migrate. The only way to discover this is reading the source. If/when `remove_stale_columns` is eventually actually removed, callers who never saw a warning get a hard break with zero advance notice, even though the codebase's own tooling (`pyutilz.dev.code_audit`) exists specifically to catch this class of "the fix is easy but nobody flags it" gap — deprecation-without-warning isn't one of the 26 scanners' checks.
- **Suggested fix**: Add `warnings.warn("remove_stale_columns is deprecated, use get_non_stale_columns instead", DeprecationWarning, stacklevel=2)` in the shim, and establish this as the project's standard pattern for any future rename-with-compat-alias (there's currently no such standard visible anywhere in the codebase).

### [MEDIUM] `publish.yml` never verifies the pushed git tag matches `pyproject.toml`'s `[project].version` before publishing — `.github/workflows/publish.yml`
- **Category**: release-process gap
- **Problem**: The publish workflow triggers on any `v*` tag push, verifies the tagged commit is an ancestor of `master` (a real, good check), then runs `python -m build` and uploads to PyPI — using whatever `version = "..."` is currently sitting in `pyproject.toml`. Nothing in the workflow (`grep -n "version" .github/workflows/publish.yml` → only the checkout-tag-reachability comment) compares `GITHUB_REF_NAME` (the pushed tag) against the built package's version. `tests/test_meta/test_version_consistency.py` only cross-checks `pyutilz.__version__` vs `pyutilz.version.__version__` vs `pyproject.toml`'s value against *each other* — it has no way to know what tag is about to be pushed, so it can't catch this class either. No `CONTRIBUTING.md`/`README.md`/`TESTING.md` documents a release checklist that would catch a forgotten version bump by hand.
- **Failure scenario**: Given the CHANGELOG's own "BREAKING" entry sitting under Unreleased with `pyproject.toml` still reading `version = "1.0.0"` (confirmed current), the natural next release requires both a CHANGELOG heading bump (e.g. to `2.0.0` per semver, since a public symbol removal is a breaking change) *and* a `pyproject.toml` version bump — two independent, manual, unenforced edits. If a maintainer bumps the CHANGELOG heading and tags `v2.0.0` but forgets the `pyproject.toml` edit (an easy slip — nothing before this workflow would catch it), the workflow silently builds and publishes a wheel whose actual metadata version is still `1.0.0`, permanently mismatching the git tag `v2.0.0` and the CHANGELOG section describing what `2.0.0` contains. Since PyPI is immutable per-version, this specific mismatch can only be fixed by picking a fresh, still-higher, still-manually-typed version — not asserted by any gate.
- **Suggested fix**: Add a step early in the `build` job: parse `GITHUB_REF_NAME` (strip the `v` prefix) and fail the workflow if it doesn't equal `[project].version` read from `pyproject.toml`.

### [LOW] CHANGELOG's own meta-test count is already stale as of the follow-up commit in the same session — `CHANGELOG.md:32`
- **Category**: documentation drift
- **Problem**: `CHANGELOG.md:32` states "46 tests across 24 files under `tests/test_meta/`". That count was accurate as of commit `a083702` (`git show a083702:CHANGELOG.md` shows the identical line). The very next commit, `206a69d` ("Add the remaining 2 code_audit scanners and 2 cross-cutting meta-tests from the audit follow-up"), added two new files (`test_retry_predicate_matches_sdk_hierarchy.py`, `test_sibling_api_parity.py`) without updating this count. Current actual state, verified by running the suite: `pytest tests/test_meta --collect-only -q` → **71 tests collected**; `ls tests/test_meta/test_*.py | wc -l` → **26 files** — both meaningfully higher than the CHANGELOG's claimed 46/24.
- **Failure scenario**: Low real impact (it's an internal-infrastructure count, not an API claim), but it's a concrete, dated instance of exactly the failure mode this audit angle is about: a same-day follow-up commit touched the thing a CHANGELOG entry describes and didn't update the entry — demonstrating the discipline lapses even on the commit immediately after the one that added the entry.
- **Suggested fix**: Either drop the specific counts from the prose (say "dozens of tests across `tests/test_meta/`") so it can't go stale, or compute them at CHANGELOG-generation time.

### [LOW] `CONTRIBUTING.md`'s documented `BREAKING CHANGE:` commit-footer convention has never been used and nothing parses it — `CONTRIBUTING.md:286-293`
- **Category**: dead process documentation
- **Problem**: `CONTRIBUTING.md` shows a worked example instructing contributors to append a `BREAKING CHANGE: ...` footer (Conventional-Commits style) to commit messages for breaking changes. `git log --all --grep="BREAKING CHANGE" --oneline` returns zero commits across all 447 commits in the repo's history, including the actual breaking-change commit (`5685867`, "Remove 14 accidentally-public leaked imports") — its commit message doesn't use the footer despite being the paradigm case the convention exists for. There is also no `release-please`/`semantic-release`/similar config anywhere in the repo that would ever consume such a footer even if it were used.
- **Failure scenario**: A contributor follows `CONTRIBUTING.md` literally, adds a `BREAKING CHANGE:` footer expecting it to feed some changelog/release automation (that's the entire point of the Conventional Commits convention) — nothing reads it; the only thing that actually gates a breaking change reaching users is a human remembering to hand-edit `CHANGELOG.md`, which the CI/pre-commit stack never verifies happened (`grep -i changelog .pre-commit-config.yaml .github/workflows/*.yml` finds only the codespell-paths list, no "PR touches src/pyutilz but not CHANGELOG.md" check).
- **Suggested fix**: Either wire up a lightweight commit-msg/CI check that actually parses `BREAKING CHANGE:` footers, or remove the worked example from `CONTRIBUTING.md` so it doesn't imply tooling that doesn't exist.

## Things done well

- The "14 accidentally-public leaked-import symbols" CHANGELOG claim (`CHANGELOG.md:27`) is **exactly** accurate — verified by counting removed entries in `tests/test_meta/_api_snapshot.json`'s diff at commit `5685867`: 6 from `image`, 6 from `logging` (present under two aliases), 1 from `numpylib`, 1 from `parallel` = 14 precisely, matching the named list in both the commit message and the CHANGELOG.
- `tests/test_api_stability.py` + `_api_snapshot.json` is a genuinely strong mechanism for the *shape* of the problem it targets (renames/removals of names, positions, kinds) — the gap found here (Finding 1) is narrow and specific, not evidence the whole mechanism is weak.
- The DeepSeek default-model change (`deepseek-v4-flash`) is a textbook example of doing this *correctly*: a real default-affecting change, explicitly called out under "Changed" in the CHANGELOG rather than silently shipped.
- `.github/workflows/publish.yml`'s tag-reachable-from-master check is a solid, real safeguard against publishing an unreviewed commit — the gap found here (Finding 4) is a distinct, narrower omission (tag-vs-version consistency), not a critique of that check.
- The kernel-tuning cache rewrite's "legacy caches migrate automatically" CHANGELOG claim checks out against the actual `_migrate_legacy()` implementation (idempotent, crash-safe claim marker, `.migrated` rename-aside) — not just asserted.

## Investigated, not an issue

- `thinking=` bool-vs-string backward compatibility (`CHANGELOG.md:17`): read `openai_compat.py`'s `_normalize_thinking()` — correctly treats `True`/`False`/`str` as distinct, backward-compatible cases; old boolean callers keep working.
- `pyutilz.web.browser` importing `pyutilz.web.pythonlib` (nonexistent) → `pyutilz.core.pythonlib`: confirmed fixed and current.
- `get_parent_func_args` perf fix (`445abce`): confirmed pure performance change (materializes `FrameLocalsProxy` once), same filtering logic, not a behavior change.
- Top-level `pyutilz/__init__.py` module-alias map, diffed against the v1.0.0-restructure commit (`57975e9`): no aliases were actually removed; the diff is dominated by a quote-style reformat plus two new aliases (`string`, `logging`) and an unrelated, already-fixed lazy-module-proxy bug (`tokenizers` global-namespace collision).
- `get_account_credits()`/`check_account_limits()` "unified across every provider" claim: spot-checked — every provider file (`anthropic_provider.py`, `claude_code_provider.py`, `deepseek_provider.py`, `gemini_provider.py`, `openai_provider.py`, `xai_provider.py`, `openrouter_provider/_provider.py`) does define at least one of the two methods, consistent with the CHANGELOG's stated support matrix.
- `pyutilz` PyPI publication status: confirmed via `pip index versions pyutilz` (network access itself verified against `requests`) that the package has never been published — so the "removes importable names that worked in 1.0.0" framing in the CHANGELOG, while a legitimate preemptive semver discipline, doesn't currently correspond to any real external installed base being broken.
