# Typing & Lint Hygiene Audit — pyutilz (2026-09-03)

## Summary

Read-only audit. No file in the repo was modified; no formatter, no `ruff --fix`, no `--unsafe-fixes`, no git write. The only file written is this report.

**Environment.** `D:/ProgramData/anaconda3/python.exe`; `mypy 2.3.1 (compiled: yes)` — matching the repo's `mypy==2.3.1` pin, `.github/workflows/mypy-full.yml`'s `mypy-version: "2.3.1"` and `.pre-commit-config.yaml`'s `rev: v2.3.1`. Installed `ruff 0.15.9`, which does **not** match the repo's own `"ruff==0.15.22"` pin (F06). `PY_CI_SHARED_DIR` was exported to `D:/Upd/Programming/PythonCodeRepository/py-ci-shared` for every ruff command, so `[tool.ruff] extend` resolved and the repo's real rule set was in force.

**Exact command lines run**

```
D:/ProgramData/anaconda3/python.exe -m mypy src/pyutilz
D:/ProgramData/anaconda3/python.exe -m mypy src/pyutilz --disallow-untyped-defs --no-incremental
D:/ProgramData/anaconda3/python.exe -m mypy src/pyutilz --disallow-any-generics --no-incremental
D:/ProgramData/anaconda3/python.exe -m mypy src/pyutilz --no-implicit-reexport --no-incremental
D:/ProgramData/anaconda3/python.exe -m mypy src/pyutilz --disallow-incomplete-defs --no-incremental
D:/ProgramData/anaconda3/python.exe -m ruff check src tests scripts --statistics
D:/ProgramData/anaconda3/python.exe -m ruff check src tests scripts --ignore C901
D:/ProgramData/anaconda3/python.exe -m ruff check _benchmarks --statistics --no-cache
D:/ProgramData/anaconda3/python.exe -m ruff check src tests scripts _benchmarks --select <PLE|ASYNC|FURB|SLOT|PYI|TRY302|RET50x|B905|A005> --isolated --statistics
D:/ProgramData/anaconda3/python.exe -m ruff check src tests scripts _benchmarks --select PLE,PYI034,RET503,FURB171 --isolated --output-format=concise
```

**Raw counts**

| Run | Result |
|---|---|
| `mypy src/pyutilz` (repo config, exactly as CI/pre-commit invoke it) | `Success: no issues found in 242 source files`, exit 0. The gate is genuinely clean, terminator printed. `warn_unreachable = true` is on and produces zero hits. |
| `ruff check src tests scripts` (full configured select) | **23 errors — all `C901`**, nothing else. Zero `Invalid # noqa directive` warnings on stderr. |
| `ruff check src tests scripts --ignore C901` (the actual blocking gate) | `All checks passed!` |
| `ruff check _benchmarks` (still excluded by config) | 76 errors: 66 `T201`, 8 `NPY002`, 1 `PERF203`, 1 `PERF401`. No `F`/`B`/`E9` real-bug class today (F09). |
| `mypy --disallow-untyped-defs` | **288 errors in 83 files** (242 checked) |
| `mypy --disallow-incomplete-defs` | **150 errors in 66 files** |
| `mypy --disallow-any-generics` | **493 errors in 98 files** |
| `mypy --no-implicit-reexport` | **219 errors in 27 files** |

**Non-selected ruff rule families measured** (each `--isolated --select <code>` over `src tests scripts _benchmarks`): `PLE` 2 · `RET501` 16 · `RET502` 6 · `RET503` 9 · `RET504` 16 · `FURB` 72 · `PYI` 3 · `B905` 27 · `ASYNC` 0 · `SLOT` 0 · `A005` 0 · `TRY302` 0.

**Independent AST scan over `src/pyutilz`** (242 files, 1528 function definitions):

- **0** implicit-`Optional` parameters (`param: T = None` with non-optional `T`) — CLAUDE.md rule 1 is holding across the whole package, now at 2x the file count of the 2026-09-02 scan.
- **0** bare `except:` handlers.
- **0** bare `# type: ignore` comments without an error code (was 4 on 2026-09-02; F10 of that wave is confirmed still fixed).
- **236** functions with no return annotation; **91** that take parameters and have no annotation on any parameter or the return.
- **124** `# type: ignore` comments, of which **54** carry one identical stamped justification string (F01).
- **16** parameters annotated `object`; 15 reviewed as genuinely duck-typed (unchanged verdict from 2026-09-02), 1 new one is a real rule-5 violation (F05).

**Prior wave re-verified.** All 17 findings in `audits/implemented/2026-09-02/06-typing-lint-hygiene.md` were checked against current state before writing anything here. **None has regressed**: `warn_unreachable = true` is set and clean; `mypy==2.3.1` is pinned identically in the dev extra, `mypy-full.yml` and `.pre-commit-config.yaml`; `tests` and `scripts` are out of `[tool.ruff] exclude` and out of the `ruff-real-bugs` hook's exclusion and both are at 0 findings; `extend-select = ["B006"]` is present; the `tests.*` mypy override is gone; `tests/test_meta/test_complexity_ratchet.py` and `_complexity_baseline.json` exist. Nothing from that wave is re-raised.

**Closed decisions respected.** `warn_unused_ignores` is deliberately off and no finding below proposes enabling it or removing any `# type: ignore` code — F01 and F10 are about the *underlying type* being fixable so the suppression is not needed, which is CLAUDE.md rule 4/7, a different action from trimming a code that merely looks unused locally. No repo-wide formatter or broad `--fix` proposed. Community-health files not raised. PEP 639 not raised.

**Direction premise corrected.** The task brief states `dev.code_audit.*` is "in the mypy untyped-defs exclusion list". It is the opposite: `pyutilz.dev.code_audit.*` is the first entry of the **strict-mode beachhead** (`[[tool.mypy.overrides]]` with `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`, `warn_return_any = true`, `no_implicit_optional = true`). Measured: the `--disallow-untyped-defs` run reports **0** errors anywhere under `code_audit`, across all 85 scanner modules. The ~40 fresh commits of scanner code are fully annotated and the beachhead is doing its job — one finding below (F02) is in that fresh code, and it is a duplication defect, not a typing one.

**Also measured and NOT reported.** `src/pyutilz/__init__.py:15`'s `PLE0604 Invalid object in __all__` is a ruff false positive on the `*_SUBPACKAGES` star-unpack (verified by reading lines 13-15; the tuple contains only strings). `system/hardware_monitor.py:236`'s `PYI034` is a style preference for `Self` over the hardcoded class name, not a defect. The 27 `B905` `zip(strict=)` hits cannot apply — `target-version = "py38"` and `strict=` needs 3.10. The 219 `no-implicit-reexport` errors are overwhelmingly this repo's intentional facade/`__init__` re-export architecture. `claude_code_provider.py`'s three `claude_code_sdk._internal` imports look alarming under `no-implicit-reexport` but are gated behind an explicit `_SUPPORTED_SDK_VERSIONS` allowlist that logs and skips all patching on an unknown version (lines 45-58) — correct handling, not a finding.

**Counts by severity:** 0 Critical, 0 High, 6 Medium, 6 Low. **12 findings total.**

## Findings

### F01. [Medium] 54 of the 124 `# type: ignore` comments in `src` carry one identical stamped justification, and for several of them that stated reason is demonstrably false — src/pyutilz/core/matrix.py:108
- **Disposition**: COMPLETED — the batch-applied string is gone from all 53 sites (0 remain). 22 had their CAUSE removed: int() at src/pyutilz/core/matrix.py:108,110, bool() at core/pythonlib/hardware.py:69, float() at data/polarslib/binning.py:60, text/strings/textentropy.py:98, system/system/memory.py:114 and dev/logginglib.py:231, np.asarray() at data/numpylib.py:138, typing.cast at data/pandaslib/frames.py:522,534,535, str() at database/db/connection.py:188, a `-> Set[Any]` return at core/pythonlib/objects.py:117, an annotated local at frames.py:565, an annotated `expr: pl.Expr` at data/polarslib/aggregations.py:323, and 8 genuinely dead ignores removed (none env-dependent). The remaining 31 keep the ignore with a per-site reason naming the actual untyped source. Regression: tests/test_typing_lint_hygiene_audit_20260903.py::test_f01_*
- **Category**: blanket-suppression
- **Problem**: Grouping every `# type: ignore` in `src` by the text following the code:
  ```
  54 [no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
  17 [attr-defined]
   8 [no-redef]
   3 [no-any-return]
  ```
  One string, applied verbatim at 54 sites across 33 files (6x `system/system/probing.py`, 4x `llm/openrouter_provider/_catalogue.py`, 4x `data/pandaslib/frames.py`, 3x `data/polarslib/aggregations.py`, ...). The reason is an unresolved OR across three unrelated causes ("json/external lib/dynamic attr") and is wrong at concrete sites I read:
  - `core/matrix.py:108` — `return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes`. numpy `.nbytes` is not json, not a dynamic attr; this is exactly the "chained numpy arithmetic loses the concrete type" case CLAUDE.md rule 4 names, whose prescribed fix is `int(...)`.
  - `core/matrix.py:110` — same, for the COO branch.
  - `core/pythonlib/hardware.py:69` — `return cuda.is_available()`, fixable with `bool(...)`.
  - `data/pandaslib/frames.py:440` and `:453` — `return df.columns.tolist()`, a pandas call whose result is a `list`; CLAUDE.md rule 7 explicitly names `cast` (zero runtime cost) as the correct treatment here rather than a redundant `list(...)` copy.
  - `core/pythonlib/objects.py:99` — `return res` where `res` is a local `set` built in the same function; nothing upstream and untyped is involved at all.
  - `data/pandaslib/benchmarks.py:145` — the ignore sits on `return pd.DataFrame(`, the FIRST line of a multi-line call, so it silences `no-any-return` for the whole constructed expression.
  `warn_return_any = true` is configured and is part of a blocking gate; these 54 lines are where it has been switched off in bulk. This is a distinct issue from the closed `warn_unused_ignores` decision: these ignores are all live and firing, and the proposal is to remove the *cause*, not the code.
- **Failure scenario**: `get_sparse_memory_usage` declares `-> int` and returns whatever `.nbytes` arithmetic produces. A future scipy/numpy change that makes one operand a 0-d array or `np.int64` propagates a non-`int` into every caller's memory accounting silently, because the one gate that would have flagged the boundary is suppressed by a comment asserting it was "verified correct at runtime" — an assertion made once, for 54 sites at a time, and never re-verified since. The stamped text also actively misleads the next reader: at `objects.py:99` it names an untyped upstream source that does not exist, so anyone triaging that line is sent looking for the wrong thing.
- **Suggested fix**: Triage the 54 into (a) sites fixable by CLAUDE.md rule 4's concrete wrap — `int(...)`, `bool(...)`, `float(...)` — which covers at least `matrix.py:108,110` and `hardware.py:69`; (b) sites fixable by annotating the local (`objects.py:99`); (c) sites fixable by `typing.cast` per rule 7 (`frames.py:440,453`); (d) a genuinely-untyped-third-party residue, where the ignore stays but the comment is rewritten to name which library and which call, per site. Move the multi-line one at `benchmarks.py:145` onto the narrowest line it can sit on. Do not batch-edit: the whole defect is that these were batched once already.

### F02. [Medium] Two separately-registered `code_audit` scanners implement the same defect class — src/pyutilz/dev/code_audit/patch_target_is_a_reexport.py:129
- **Disposition**: COMPLETED — kept deliberately as a pair, now stated at both sites: cross-referencing docstrings at src/pyutilz/dev/code_audit/patch_target_is_a_reexport.py:1 and reexport_patch_target.py:1 name each other and state the one thing that differs (the suppression rule); registry.py:316 already had reexport_patch_target in OPT_IN_ONLY so only one runs by default. The duplicate_function_body that flagged the pair was real and is fixed: the private `_dotted` copy is deleted and patch_target_is_a_reexport.py:8 now imports `_base.dotted_name`.
- **Category**: duplicate-code
- **Problem**: `scan_reexport_patch_target` (`reexport_patch_target.py:92`, 173 lines) and `scan_patch_target_is_a_reexport` (`patch_target_is_a_reexport.py:129`, 191 lines, added 2026-09-03 by commit `2399920 code_audit: a check for a patch aimed at the wrong copy of a name`) target the identical shape. Their own docstrings, side by side:
  > "A test that patches a name on the module which RE-EXPORTS it rather than the one that CALLS it. ... The shape this finds: a test patches `M.name`, and `M` itself obtained `name` from another module via `from OTHER import name`."

  > "Find a test that patches a name on a facade the real caller never looks through. `patch("facade.fetch")` rebinds the name in `facade`'s namespace ... When `facade` re-exports it from `_impl` and the caller lives in `_impl` calling `fetch()` directly, the patch rebinds a name nobody reads."

  Both are live: `registry.py:70` imports and `registry.py:216` calls `register_scanner("patch_target_is_a_reexport", ...)`, while `reexport_patch_target` is registered through `__init__.py:526`/`:620` and listed in the opt-in tuple at `__init__.py:681`. Both parse the same patch idioms (`setattr`/`patch`/`object`), both build the same `from X import name` re-export map, and they differ only in their false-positive suppression rule (the older one reports the ambiguity when it cannot see the call site's binding style; the newer one goes silent when the facade also calls the name). The repo ships `near_duplicate_function_body.py` for precisely this class and did not catch it because the two implementations are independently written, not copy-pasted.
- **Failure scenario**: A downstream repo running the full scanner set gets the same defect reported twice under two check names with different severities and different `detail` prose, so a triager cannot tell whether they are two findings or one. Worse for maintenance: a false-positive fix applied to one — and the commit log shows exactly this pattern, `4f4182b fix: two new code_audit checks fired on correct code in a downstream repo` — leaves the other still firing on the same correct code, so the same bug report arrives twice and gets half-fixed.
- **Suggested fix**: Pick one implementation, keep the union of the two suppression rules (both are sound and independently valuable), delete the other module, and keep the retired check name as an alias in `registry.py` so downstream configs naming it do not break. If both are deliberately kept, say so in each docstring and state what distinguishes them — right now neither file acknowledges the other exists.

### F03. [Medium] `check_if_pg_table_exists` returns `None` on the path its docstring says returns `False`, and `ensure_pg_table_exists` branches on the result — src/pyutilz/database/db/schema.py:25
- **Disposition**: COMPLETED — src/pyutilz/database/db/schema.py:27 is now `-> Optional[bool]` with an explicit terminal `return None` and a docstring stating the three-state contract; the caller at schema.py:69 raises RuntimeError on the unknown rather than proceeding to CREATE against a database it could not query.
- **Category**: implicit-return
- **Problem**: `ruff check --select RET503 --isolated`:
  ```
  src\pyutilz\database\db\schema.py:25:1: RET503 Missing explicit `return` at the end of function able to return non-`None` value
  ```
  The function is `def check_if_pg_table_exists(table_name: str, schema_name: Optional[str] = "public"):` — no return annotation — whose docstring reads "True when `table_name` exists in `schema_name` according to `information_schema.tables`". The body ends:
  ```python
  res = _facade.safe_execute(...)
  if res:
      return res[0][0]
  ```
  With no `else` and no trailing `return`, a falsy `res` falls off the end and yields `None`. The sole in-repo consumer is `schema.py:56`, `if not _facade.check_if_pg_table_exists(table):`, inside `ensure_pg_table_exists`. The missing return annotation is why mypy is silent: this module is not in the beachhead, so the declared-vs-actual mismatch has nothing to check against.
- **Failure scenario**: `safe_execute` returning `[]`/`None` — its behaviour on a swallowed error or a lost connection — makes `check_if_pg_table_exists` return `None`. `ensure_pg_table_exists` reads that as "table absent" and proceeds to its CREATE path against a database it could not actually query. The function cannot distinguish "the table is not there" from "I could not find out", and its two-state docstring promises it can.
- **Suggested fix**: Annotate `-> Optional[bool]` and add an explicit terminal `return None` if the three-state contract is intended (documenting the `None` case), or raise on a falsy `res` and annotate `-> bool` if the caller must never see an unknown. Either way the annotation must exist so a future mismatch is a gate failure and not invisible.

### F04. [Medium] `get_table_fields` falls off the end returning `None`, and its result is concatenated into SQL — src/pyutilz/database/db/execution.py:33
- **Disposition**: COMPLETED — src/pyutilz/database/db/execution.py:33 is fully annotated `-> str` (including `excluding: Union[str, Iterable[str]]`, whose rebinding is now a separate `excluded` local) and raises RuntimeError naming the table when the driver reports no cursor description, so neither None nor the literal text None can reach the SQL.
- **Category**: implicit-return
- **Problem**: `ruff check --select RET503 --isolated`:
  ```
  src\pyutilz\database\db\execution.py:33:1: RET503 Missing explicit `return` at the end of function able to return non-`None` value
  ```
  `def get_table_fields(table, alias, prefix="", suffix="", excluding=""):` — fully unannotated, one of the 91 the AST scan found — whose docstring says it "Returns a comma-separated `<alias>.<column> <prefix><column><suffix>` select list for a table". The body ends:
  ```python
  local_cur.execute("select * from " + table + " where 0=1")
  local_cur.fetchall()
  if local_cur.description is not None:
      return ",".join([...])
  ```
  DB-API sets `cursor.description` to `None` for any statement with no result set; the `if` acknowledges that case and then silently returns `None` for it. The function is a public export (`database/db/__init__.py:130` and `:200`).
- **Failure scenario**: The function exists to be interpolated into a larger SELECT. A caller writing `"select " + get_table_fields("orders", "o") + " from orders o"` gets `TypeError: can only concatenate str (not "NoneType") to str`, or, with an f-string, the literal text `None` inside the SQL, producing a Postgres syntax error at a site the docstring said returns a select list. Nothing in the type system flags either, because the function has no annotations at all.
- **Suggested fix**: Annotate `-> str` and make the empty case explicit — `return ""` after the `if`, or raise a clear error naming the table — and document which was chosen. Adding `table: str, alias: str, prefix: str = "", suffix: str = "", excluding: Union[str, Iterable[str]] = ""` at the same time costs nothing and covers the `excluding` parameter that is reassigned from `str` to `list` on line 41.

### F05. [Medium] `add_weighted_aggregates(columns_selector: object)` is a CLAUDE.md rule-5 violation, and it type-checks only because the other operand is `Any` — src/pyutilz/data/polarslib/aggregations.py:116
- **Disposition**: COMPLETED — src/pyutilz/data/polarslib/aggregations.py:116 is `columns_selector: cs.Selector`, the `: Any` escape on line 130 is gone, and the return tightened to `List[pl.Expr]`.
- **Category**: object-where-concrete-type-exists
- **Problem**: The signature is
  ```python
  def add_weighted_aggregates(
      columns_selector: object, weighting_columns: Iterable, fpref: str = "", fields_remap: Optional[dict] = None, nans_filler: float = 0.0
  ) -> list:
  ```
  and the body's first use is `aggregations.py:130`: `all_other_num_cols: Any = columns_selector - cs.by_name(wcol)`. `object` supports no `-` operator; the statement passes only because `cs.by_name(...)` is `Any`, which makes the whole binary expression `Any` and defeats the check. Every call site passes a polars column selector: `aggregations.py:370` passes `(cs.numeric() - cs.by_name(exclude_fields or []))`, and the tests pass `cs.numeric()` (`tests/test_polarslib_extra.py:253,257,261,266`). This is CLAUDE.md rule 5 verbatim — "give every parameter that only ever holds a concrete class a concrete type, not `object`". It is the one new `object` parameter since 2026-09-02; the other 15 were re-reviewed and remain genuinely duck-typed.
- **Failure scenario**: A caller passes a plain `list[str]` of column names — a natural reading of a parameter called `columns_selector` typed `object`, and the sort of thing the sibling polars helpers accept — and `list - Selector` raises `TypeError: unsupported operand` deep inside the aggregate builder rather than at the boundary. The type checker cannot help, because `object` accepts the list and the `Any` on the right hides the operator. The function also declares `-> list` while its docstring describes returning polars expressions, so neither end of the signature carries usable information.
- **Suggested fix**: `columns_selector: cs.Selector` (quoted, or under `TYPE_CHECKING`, if the import must stay lazy for the optional-polars gate). Then drop the `: Any` on line 130 and let the real expression type flow, which will also tighten `-> list` to `list[pl.Expr]`.

### F06. [Medium] The blocking ruff gate runs whatever ruff is installed, not the pinned one — this box is 13 minor versions behind the repo's own pin — .pre-commit-config.yaml:78
- **Disposition**: COMPLETED — scripts/check_pinned_tool_versions.py reads the exact pin out of pyproject.toml and fails when the interpreter's ruff differs; wired as the `pinned-tool-versions` pre-commit hook ahead of ruff-real-bugs (.pre-commit-config.yaml:81). This box was brought from 0.15.9 to the pinned 0.15.22, and the whole tree re-linted clean under it.
- **Category**: tool-version-drift
- **Problem**: `pyproject.toml:366` pins `"ruff==0.15.22"` with the comment "pinned exact, matching CI's uvx ruff==0.15.22 (ruff-blocking.yml) -- an open range here is exactly how mlframe's prior v0.8.6 pre-commit-vs-CI drift went unnoticed for a long time". The `ruff-real-bugs` hook that actually blocks commits is a `repo: local` hook with `entry: python -m ruff check --ignore C901` and `language: system`. `language: system` means pre-commit builds no isolated environment; `python -m ruff` resolves to whatever is in the interpreter. On this box that is `ruff 0.15.9`, against a pinned `0.15.22`, and nothing anywhere reports the mismatch. This is the same defect the 2026-09-02 wave fixed for mypy (that wave's F07) — except mypy is additionally protected by `py_ci_shared.mypy_gate --min-files 200`, which asserts completion. Ruff has no equivalent assertion, and it is the gate whose rule set changes most between releases.
  Consequence for this audit, stated plainly: every ruff number in the Summary was produced by 0.15.9, not by the version CI runs. `All checks passed!` here is not proof that `ruff-blocking.yml` is green.
- **Failure scenario**: Ruff 0.15.10-0.15.22 adds rules to already-selected families (`RUF`, `B`, `PERF` are all under active development). A contributor commits with 0.15.9 installed, the local blocking hook passes, and `ruff-blocking.yml` fails on CI with a rule the contributor's ruff had never heard of — the exact incident class the pin's own comment describes. In the other direction, a rule removed or downgraded upstream means a local pass is checking strictly less than CI believes.
- **Suggested fix**: Give the hook a version assertion the way the mypy hook got one — either move it off `language: system` to `repo: https://github.com/astral-sh/ruff-pre-commit` with `rev` matching the pin, or wrap it in a small `py_ci_shared` entry point that fails when `ruff.__version__` differs from the pin. The second keeps the "one resolved config, no `--select`" property the surrounding comment is careful about. Separately, this dev box's ruff should be brought to 0.15.22.

### F07. [Low] The strict-mode beachhead has not expanded since 2026-09-02, and its own documented next candidate is still stuck at exactly 33 errors — pyproject.toml:711
- **Disposition**: COMPLETED — beachhead grown by two subpackages after clearing their errors: `pyutilz.web.proxy.*` (2 return annotations on the @contextmanager session factories) and `pyutilz.llm.openrouter_provider.*` (3 functions), both listed at pyproject.toml:753-754. The kernel_tuning.cache note was re-measured and is recorded as unmoved (still 33 across the same 6 files) rather than reading as a fresh number.
- **Category**: annotation-coverage
- **Problem**: `disallow_untyped_defs = false` remains global. Re-measured today: `mypy src/pyutilz --disallow-untyped-defs` reports 288 errors in 83 files of 242, and `--disallow-incomplete-defs` reports 150 errors in 66 files. The AST scan puts 236 functions with no return annotation and 91 with no annotation on anything. The beachhead lists the same 5 entries added on 2026-09-02; the ~40 commits since added no sixth. The pyproject comment records `performance.kernel_tuning.cache` as the measured-and-rejected candidate at "33 `[no-untyped-def]` errors across 6 files"; today it is still exactly 33, still across 6 files (`_common.py` 8, `cache_hooks.py` 9, `cache_tuning.py` 9, `cache_sweeping.py` 3, `cache_base.py` 2, `cache_class.py` 2) — the recorded measurement has not moved at all, so it is functioning as a note rather than as a queue. Worst offenders overall: `text/similarity/_numba_kernels.py` 15, `system/system/misc.py` 14, `system/system/probing.py` 12, `data/pandaslib/benchmarks.py` 12, `dev/logginglib.py` 11, `database/db/execution.py` 11, `core/pythonlib/filesystem.py` 11.
  Two cohesive subpackages are far closer to admission than the recorded candidate and were not measured on 2026-09-02 because they did not exist in their current split form: `pyutilz.web.proxy` has 3 errors across 5 modules (all in `session.py`), and `pyutilz.llm.openrouter_provider` has 4 across 4 modules. Either is a one-sitting addition.
  This is Low, not higher, for the same reason as last wave: the write-time discipline is measurably holding (0 implicit-`Optional`, 0 bare `except:`, 0 bare `# type: ignore` across a file count that has doubled). It is a coverage gap, not a regression.
- **Failure scenario**: The package ships `py.typed`, so a downstream consumer type-checking against `get_table_fields` (F04) or `check_if_pg_table_exists` (F03) receives `Any` while believing it has a contract — and both of those findings exist because their modules are outside the beachhead, exactly as the previous wave's F04/F06 were caused by `monitored`'s unannotated `func`. Every wave that does not move the beachhead leaves the next wave the same discovery to make.
- **Suggested fix**: Add `pyutilz.web.proxy.*` and `pyutilz.llm.openrouter_provider.*` after closing their 3 and 4 errors respectively, and update the pyproject comment's recorded `kernel_tuning.cache` measurement with today's date so it is visible that the number has not moved in a month rather than looking freshly taken.

### F08. [Low] Seven further functions return a value on some paths and fall off the end on others, none of them annotated — src/pyutilz/core/pythonlib/filesystem.py:196
- **Disposition**: COMPLETED — RET503 is at zero across src/tests/scripts/_benchmarks and is now in `[tool.ruff.lint] extend-select` so it stays there. Fixed: cloud/cloud.py:75 (explicit `return None`, `-> _Optional[_Any]`), core/pythonlib/filesystem.py:183,204,229 (`-> bool` + `return False`), data/pandaslib/io_ops.py:94 (`-> Optional[pd.DataFrame]` + explicit `return None`), database/db/execution.py:86 (DuplicateTable now returns the documented `[]`; the exhausted-retry fall-off raises instead of an implicit None). dashlib.create_tabs and stackutils.lookup_in_stack were already fixed by the intervening wave.
- **Category**: implicit-return
- **Problem**: `RET503` is not in the shared base's select. An isolated run over `src tests scripts _benchmarks` reports 9 total; F03 and F04 cover the two with a caller that misreads the `None`. The other seven, each verified by reading the function:
  - `src/pyutilz/core/pythonlib/filesystem.py:196` — `ObjectsDumper._process_object`, returns `True` on the dump path, `None` otherwise. `process_objects` counts the results, so `None` works as falsy today, but the sibling override two definitions down has the same shape and neither declares `-> bool`.
  - `src/pyutilz/core/pythonlib/filesystem.py:221` — `ObjectsLoader._process_object`, same shape.
  - `src/pyutilz/cloud/cloud.py:75` — `connect_to_s3`; the docstring explicitly documents "Returns None (leaving `s3` unset) if credentials could not be read", so the behaviour is intended and only the annotation is missing.
  - `src/pyutilz/core/pythonlib/stackutils.py:27` — `lookup_in_stack`; docstring documents the `None`.
  - `src/pyutilz/data/pandaslib/io_ops.py:82` — `read_stats_from_multiple_files`, a 16-parameter function whose docstring does not mention a `None` result.
  - `src/pyutilz/database/db/execution.py:53` — `basic_db_execute`; the docstring says "a statement with no result set yields an empty list", which the fall-off path contradicts.
  - `src/pyutilz/dev/dashlib.py:287` — `create_tabs`.
  Ruff's `RET50x` family measures 47 hits in total (`RET501` 16, `RET502` 6, `RET503` 9, `RET504` 16); only `RET503` is a genuine correctness class, and the other three are stylistic and are not proposed for selection.
- **Failure scenario**: `basic_db_execute` is the module's central execution entry point and its docstring promises an empty list for a no-result statement. A caller doing `for row in basic_db_execute(...)` on a DDL statement gets `TypeError: 'NoneType' object is not iterable` instead of a zero-iteration loop, on a path the documentation says is handled.
- **Suggested fix**: Add the terminal `return` (or the honest `Optional[...]` annotation) at each of the seven, prioritising `basic_db_execute` where the docstring and the code disagree. Then add `"RET503"` to `[tool.ruff.lint] extend-select` alongside the existing `B006` entry, so the class stays closed; the other `RET5xx` codes should stay out.

### F09. [Low] `_benchmarks/` is the last directory with zero static analysis — pyproject.toml:495
- **Disposition**: COMPLETED — `_benchmarks` dropped from `[tool.ruff] exclude` (pyproject.toml:510) and from the ruff-real-bugs hook's exclusion, with a `"_benchmarks/**"` per-file-ignores block carrying one justification per code. Re-measured at removal: 76 findings, all in already-exempted idiomatic codes (66 T201, 8 NPY002, 1 PERF203, 1 PERF401); F/B/E9 clean, so the directory is at zero.
- **Category**: lint-config-gap
- **Problem**: `pyproject.toml:495` is now `exclude = ["_benchmarks", ".git", "__pycache__", "build", "dist"]` and the `ruff-real-bugs` hook mirrors it with `exclude: (^|/)_benchmarks/`. `tests` and `scripts` were brought in on 2026-09-02 and are at 0. `_benchmarks` (5 files) was left out and measures 76 findings: 66 `T201`, 8 `NPY002`, 1 `PERF203`, 1 `PERF401`. Every one is in a code the previous wave exempted per-file for `tests`/`scripts` as genuinely idiomatic, so there is no real-bug finding hidden there today — `F`, `B`, `E9` are all clean. The gap is structural, and it is the identical argument the previous wave accepted for `scripts/` (its F15, dispositioned COMPLETED): a directory nobody lints will never report the `F821` or `F811` it eventually acquires.
- **Failure scenario**: A benchmark script acquires an `F821` undefined name in a branch that only runs on the GPU leg, or an `F401`-shadowed import that changes which implementation is being timed. Neither is caught until someone runs the benchmark and reads a wrong number — and a benchmark producing a plausible wrong number is worse than one that crashes.
- **Suggested fix**: Drop `_benchmarks` from both exclusions and add `"_benchmarks/**" = ["T201", "NPY002", "PERF203", "PERF401"]` to `[tool.ruff.lint.per-file-ignores]` with the same one-line-per-code justifications the `tests/**` block already carries. That is a zero-finding change today and closes the last unlinted directory.

### F10. [Low] Seventeen `# type: ignore[attr-defined]` comments carry no reason, and at least one sits on a genuinely unsafe attribute access — src/pyutilz/dev/freevar_analysis.py:205
- **Disposition**: COMPLETED — the unsafe one is fixed at src/pyutilz/dev/freevar_analysis.py:255: the `lineno` fallback is now a guarded getattr, so a node type without it returns 0 instead of raising AttributeError mid-rewrite, and the ignore is gone because the error is. The code_audit group uses `typing.cast` at the point the guard already proved the type (locals_globals_output.py:69,87) or a narrower `dict[str, ast.expr]` declaration (tautological_guard.py:71); the 3.8-only ast.Index ones keep the ignore with the version named. The platform ones are unchanged per the closed decision, with the environment now named at the site.
- **Category**: unexplained-suppression
- **Problem**: Of the 124 `# type: ignore` comments in `src`, 17 are `[attr-defined]` with nothing after the code — no environment note, no reason:
  `dev/code_audit/locals_globals_output.py:74,92` · `dev/code_audit/mutable_defaults.py:150` · `dev/code_audit/silent_escalation.py:35` · `dev/code_audit/tautological_guard.py:89` · `dev/freevar_analysis.py:205` · `performance/kernel_tuning/cache/cache_base.py:315,335` · `system/system/memory.py:128,131,137,143,147` · `system/system/misc.py:445` · `system/system/probing.py:368` · `text/similarity/_numba_kernels.py:647,717`.
  The eight `ctypes.windll` / `ctypes.WinDLL` / `winsound` / `nb.prange` ones are the platform-conditional class CLAUDE.md's closed section covers and must stay — but they are the ones that most need the environment named in the comment, and they do not have it, while their siblings elsewhere in the repo do (`system/config.py:33`'s ignore states its reason). The `code_audit` ones are a different class: they stand in for a narrowing the checker cannot follow through a helper predicate, e.g. `locals_globals_output.py:74`'s `kw.value.func.id` is safe only because `_is_locals_or_globals_call` at line 39 already asserted `isinstance(node.func, ast.Name)`.
  `freevar_analysis.py:205` is the one that is not merely cosmetic:
  ```python
  def _end_line(node: ast.AST) -> int:
      end = getattr(node, "end_lineno", None)
      return int(end) if end is not None else int(node.lineno)  # type: ignore[attr-defined]
  ```
  The parameter is `ast.AST`, whose base class genuinely has no `lineno` — expression-context nodes (`ast.Load`, `ast.Store`), operator nodes (`ast.Add`), and `ast.Module` itself carry neither `end_lineno` nor `lineno`. The `getattr` guard covers `end_lineno` and the fallback then reaches for `lineno` with no guard at all. mypy is reporting the real hazard and the ignore is answering it with silence.
- **Failure scenario**: `_end_line` is called during module splitting — an operation that rewrites source files. Hand it a node type without `lineno` and it raises `AttributeError` mid-rewrite, the same class of tool the 2026-09-02 wave flagged for `scripts/auto_refactor.py`. For the `code_audit` group, the ignores also suppress every future `attr-defined` on those lines, so if a narrowing helper is later relaxed the line stops being safe with no signal.
- **Suggested fix**: `freevar_analysis.py:205` — replace the second reach with `int(getattr(node, "lineno", 0) or 0)`, or narrow the parameter to `ast.stmt`/`ast.expr`, which is what every call site actually passes; the ignore then disappears because the error does. For the `code_audit` five, use `typing.cast(ast.Name, ...)` at the point the helper already proved the type (zero runtime cost, CLAUDE.md rule 7) instead of a line-wide ignore. For the eight platform ones, keep the ignores exactly as they are and only append the environment note — "Windows-only; absent on the Linux CI runner" — so the closed rule is legible at the site rather than only in CLAUDE.md.

### F11. [Low] An invisible zero-width space sits inside a test docstring, and the ruff rule that finds it is not selected — tests/test_code_audit.py:6006
- **Disposition**: COMPLETED — the U+200B is gone from tests/code_audit/test_hardcoded_test_path.py:88 (it moved there from the old monolithic test file), replaced by plain text. PLE2510/2512/2513/2514/2515 added to extend-select; repo-wide at zero, and tests/test_typing_lint_hygiene_audit_20260903.py scans all four source dirs for five invisible code points.
- **Category**: disabled-real-bug-rule
- **Problem**: `ruff check --select PLE --isolated` over `src tests scripts _benchmarks` reports 2 findings; one is a false positive (see Summary), the other is real:
  ```
  tests\test_code_audit.py:6006:73: PLE2515 [*] Invalid unescaped character zero-width-space, use "​" instead
  ```
  `cat -A` on that line confirms it: `"""A file under a 'tests' directory is scanned even without a test_/M-bM-^@M-^K_test.py name."""` — a literal U+200B between `test_/` and `_test.py`. Nothing in the file comments on it and no adjacent test depends on it, so it reads as a paste artifact rather than a deliberate self-match guard. It cannot be seen in any editor. The `PLE` family (pylint errors) is not in the shared base's `select`, and `RUF001`/`RUF002`/`RUF003` — which would otherwise flag ambiguous unicode — are explicitly in the base's `ignore` list for good false-positive reasons on a codebase with intentional Cyrillic content. So there is currently no rule anywhere in this repo that can see an invisible character in source.
- **Failure scenario**: Docstrings are load-bearing in this repo: `interrogate` gates on them at `--fail-under=100`, `codespell` reads them, and `pyutilz.dev.code_audit` ships scanners (`docstring_args`, `comment_names_missing_symbol`) that parse docstring text and compare it to symbols. An invisible character inside one silently breaks any such string comparison, and the failure is undiagnosable by reading the file. The broader gap is that the exemption granted to `RUF001-003` for legitimate non-ASCII prose also exempted genuinely invisible characters, which are never legitimate.
- **Suggested fix**: Replace the character with a plain `/` (or `​` if it is deliberate, with a comment saying why). Then add `"PLE2510", "PLE2512", "PLE2513", "PLE2514", "PLE2515"` — the five invalid-invisible-character rules — to `[tool.ruff.lint] extend-select` next to `B006`. They are exactly the subset of the ambiguous-unicode problem that has no false positives, and they are currently at 1 finding repo-wide.

### F12. [Low] `get_attr` is annotated `obj: dict` / `-> object`, so its return is unusable without a cast and its own falsy-object guard contradicts the parameter type — src/pyutilz/core/pythonlib/objects.py:113
- **Disposition**: COMPLETED — src/pyutilz/core/pythonlib/objects.py:131 is now `obj: Optional[dict]` / `-> Any`, and the previously-unreachable-by-declaration None case is handled explicitly before `.get()` so a non-None `unwanted_value` no longer walks into AttributeError.
- **Category**: annotation-vs-reality
- **Problem**: `def get_attr(obj: dict, attr_name: str, default_value: object = _GET_ATTR_UNSET, unwanted_value=None, *, _unset: object = _GET_ATTR_UNSET) -> object:`. Two separate mismatches:
  - `-> object` on a public helper whose documented behaviour is "missing/unwanted values fall back to `[]`". `object` supports no indexing, no iteration and no `len()`, so every annotated caller must `cast` before using the result — the outcome CLAUDE.md rule 5 exists to prevent. The two `_unset` / `default_value` parameters are correctly `object` (they are genuine sentinels and may hold anything); the return is not the same case.
  - `obj: dict` versus the body's `if obj == unwanted_value: return default_value` with `unwanted_value=None` defaulting. Comparing a `dict` to `None` is a check the declared type says can never succeed; either the parameter is really `Optional[dict]` (the guard's evident purpose) or the guard is dead. `warn_unreachable` does not catch it because `==` against `None` is not a narrowing form mypy models as unreachable.
  This module is outside the beachhead, so neither is currently reported by any gate.
- **Failure scenario**: A caller in an annotated downstream module writes `for x in get_attr(cfg, "hosts"):` and gets a mypy error on correct, documented code, so they either add a `cast` (losing the check) or stop annotating the call site. Meanwhile a caller who passes `None` for `obj` — which the `unwanted_value` guard invites — is doing something the signature forbids, so no checker warns them, and whether it works depends on a comparison that reads as vestigial.
- **Suggested fix**: `-> Any` per CLAUDE.md's preference for an accurate annotation over a technically-true-but-unusable one, or the explicit `Union` of what the four paths return; `object` is the one option that is both true and useless. Change `obj: dict` to `obj: Optional[dict]` to match the guard, or delete the guard and say in the docstring that `obj` must be a dict.
