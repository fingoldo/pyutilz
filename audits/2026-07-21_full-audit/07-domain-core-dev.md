# Audit: pyutilz `core` + `dev` subsystem (incl. `code_audit` lint tool)

## Summary

Covered all 32 assigned files in full: `src/pyutilz/core/*` (filemaker, image, matrix, openai, pythonlib, safe_pickle, serialization), the stray `src/pyutilz/__init__.py.old`, `src/pyutilz/dev/*` (benchmarking, dashlib, logginglib, meta_test_utils, notebook_init), and the entire `code_audit` lint-tool subpackage (13 scanner modules + `_base`/`registry`/`cli`/`__init__`/`__main__`). Every finding below was verified either by reading the exact cited lines or, where a runtime scenario could confirm/refute a hypothesis, by actually executing the code (crafted repro scripts under a scratch directory, run against the real `pyutilz` installed in this environment) — results are reported alongside each finding. `safe_pickle.py` itself is clean from a correctness/efficiency standpoint (its atomic-replace-with-retry and per-path locking design is sound); I focused there on correctness/efficiency per the assignment and left the security posture to the dedicated security agent.

Two clusters dominate: (1) `pyutilz.core.pythonlib` and `pyutilz.dev.dashlib`/`logginglib` contain several small, previously-unexercised correctness bugs (a `TypeError` crash on a function's own documented default argument, a shelve-based "safe" concurrency primitive that never actually closes/syncs the database it wraps, an off-by-one week-bucketing helper, a sign bug in a digit-counting helper, a `NameError` reachable through the exact defensive path the code claims to tolerate) — none of these are caught by ruff/mypy/interrogate because they are runtime logic bugs, not static-analysis-shaped ones. (2) The `code_audit` scanners — while unusually well-engineered, with real false-positive lineage documented in most docstrings — still have several concrete, reproducible false-positive/false-negative gaps in exactly the "crafted example slips through" shape the assignment asked to hunt for, plus one bug in the CLI's own exit-code contract and one in the Markdown renderer that a scanner's own message text triggers.

Findings: 3 High, 13 Medium, 8 Low (listed below, most severe first; counted after, per instructions).

## Findings

### [HIGH] `open_safe_shelve` never closes/syncs the shelve DB it yields — src/pyutilz/core/pythonlib.py:872-884
- **Category**: correctness
- **Problem**: The context manager opens the lock file, `yield`s `shelve.open(db_path, ...)` directly (without binding it to a local variable), and after the `with`-block's body completes, only `fh.flush()`/`os.fsync(fh.fileno())` run — and `fh` is the **lock file** handle (`f"{db_path}.lock"`, opened `"wb"`, never written to), not the shelve database. The shelve/dbm object itself is never `.close()`d or `.sync()`d before the lock is released, even though the docstring claims "On exit, flushes the lock file and fsyncs it to disk before releasing the lock" (implying the *data* is durable at that point).
- **Failure scenario**: Verified by direct execution against the installed package (Python 3.14, default `dbm.sqlite3` backend):
  ```python
  from pyutilz.core.pythonlib import open_safe_shelve
  with open_safe_shelve(path) as db:      # default writeback=False
      db["x"] = 42
  with open_safe_shelve(path, flag="r") as db2:
      db2.get("x")
  ```
  raises `dbm.sqlite3.error: no such table: Dict` on the *second* `with`-block — the most basic, default-parameter, sequential usage pattern crashes outright, because the first writer's underlying sqlite3 connection/transaction was never committed/closed. The identical control case with an explicit `db.close()` after the write round-trips correctly (`[1, 2]` read back). With `writeback=True` (a parameter this very function exposes) the failure is even more fundamental: shelve caches all touched values in memory and only writes them back on `.sync()`/`.close()`, so mutating a stored container (`db["counter"].append(1)`) is silently lost entirely, on every backend, not just `dbm.sqlite3`. On Windows with Python <3.13 (no compiled `gdbm`/`ndbm`, confirmed on this machine), the fallback `dbm.dumb` backend additionally never commits its on-disk *directory* index except on `.close()`, so even non-writeback single-key writes can leave an orphaned data blob with an unreadable index.
- **Suggested fix**: Bind the shelve object to a local name, wrap the body in `try/finally`, and call `db.close()` (which internally commits/syncs) *before* flushing/fsyncing the lock file and releasing the lock — e.g. `db = shelve.open(...); try: yield db; finally: db.close(); fh.flush(); os.fsync(fh.fileno())`.

### [HIGH] `keys_changed_enough` crashes with `TypeError` on its own documented default argument — src/pyutilz/core/pythonlib.py:190-219
- **Category**: correctness / edge-case
- **Problem**: `key_contains: Optional[str] = None` is the declared default, but line 206's `if key_contains in key:` performs Python's substring `in` test, which requires the left operand to be a `str` — `None in "somekey"` raises `TypeError: 'in <string>' requires string as left operand, not NoneType`.
- **Failure scenario**: Verified: `keys_changed_enough(obj={'a':100,'b':180}, prev_obj={'a':100,'b':200})` (omitting `key_contains`, i.e. using the function's own default) raises `TypeError: 'in <string>' requires string as left operand, not NoneType` on the very first key of `prev_obj`. The function is effectively unusable unless every caller always passes `key_contains` explicitly, despite the signature advertising it as optional.
- **Suggested fix**: `if key_contains is None or key_contains in key:` (or default to `""`, since `"" in key` is always `True`).

### [HIGH] `create_tabs`: `NameError` on `user` exactly when its own defensive `except` fires — src/pyutilz/dev/dashlib.py:303-344
- **Category**: correctness / edge-case
- **Problem**: `user` is only assigned inside the `try:` block (`user = current_user`, line 306). If `flask_login` is not installed, or the import/attribute-access raises for any other reason, the `except Exception as e:` branch (line 309, explicitly documented as tolerating "flask_login may not be installed") logs a debug message and falls through — leaving `user` unbound. Later, line 344 evaluates `(tabUsers is None) or (user.role in tabUsers)`; `or` short-circuits only when `tabUsers is None`, so for any role-restricted tab (`tabUsers` set to a list of roles) `user.role` is evaluated and raises `NameError: name 'user' is not defined`.
- **Failure scenario**: Call `create_tabs(tabsName, tabsList, draw_fn)` where `tabsList` contains at least one tab with `allowed_user_roles` set (e.g. `("Admin", "admin", ["admin"])`) in an environment without `flask_login` installed (the exact scenario the surrounding `try/except` is documented to tolerate) → `NameError` crashes the whole tab-rendering call instead of gracefully degrading.
- **Suggested fix**: Initialize `user = None` before the `try`, and treat `user is None` as "not authenticated" (skip role-restricted tabs, or return early) instead of only guarding the `is_authenticated` check inside the `try`.

### [MEDIUM] `code_audit` CLI: `--min-severity` silently defeats its own documented CI-gate exit code — src/pyutilz/dev/code_audit/cli.py:91-99
- **Category**: correctness (tool-of-tools bug)
- **Problem**: `findings` is reassigned to the `--min-severity`-filtered list (line 91-94) *before* the final `return 1 if any(f.severity in {"P0","P1"} for f in findings) else 0` (line 99). The docstring promises exit code `1` "when any P0 or P1 finding is present", but that check now runs against the already-filtered list.
- **Failure scenario**: Verified: a file with exactly one P1 `missed_await` finding and zero P0 findings. `python -m pyutilz.dev.code_audit <dir>` (default `--min-severity Low`) → prints the finding, exit code `1`. The identical run with `--min-severity P0` → prints `[]` (nothing shown) **and exits `0`** — a real P1 issue is silently swallowed from the CI gate's perspective, purely because of a display-filtering flag that the `--min-severity` help text describes as "filter out findings below this severity" with no mention that it also weakens the gate.
- **Suggested fix**: Compute the exit code from the *unfiltered* `run_all(...)` result, and apply the `--min-severity` filter only to what gets rendered/printed.

### [MEDIUM] `scan_mutable_defaults`: scope-blind AST walk produces false-positive P0 across a shadowing nested function — src/pyutilz/dev/code_audit/mutable_defaults.py:38-56
- **Category**: correctness (scanner false positive)
- **Problem**: `_param_is_mutated` calls `ast.walk(func)`, which descends into *any* nested `def`/`lambda` inside `func`'s body — including a nested function that happens to declare its own, unrelated parameter with the same name, shadowing the outer default.
- **Failure scenario**: Verified by running the actual scanner against:
  ```python
  def outer(x=[]):
      def inner(x):
          x.append(1)      # mutates INNER's own x, not outer's default
          return x
      return inner([1, 2, 3])
  ```
  `python -m pyutilz.dev.code_audit <dir>` reports `mutable_default` **P0** — `"def outer(..., x=list()): MUTATED in body -> shared state leaks across callers"` — even though `outer`'s own `x=[]` default is never touched (only `inner`'s independently-scoped `x` parameter is mutated). A reviewer trusting the P0 severity would misdiagnose a non-bug as the project's highest-severity class.
- **Suggested fix**: When walking for mutation sites, stop descending into nested `FunctionDef`/`AsyncFunctionDef`/`Lambda` nodes that re-declare a parameter with the same name (shadow the outer binding) — same "rebound" idea `missed_await.py` already applies for its own analysis (see next finding for a related gap there).

### [MEDIUM] `scan_late_binding_closures`: comprehension-based closures are completely unscanned — src/pyutilz/dev/code_audit/closures.py (scan_late_binding_closures, only visits `ast.For`/`ast.AsyncFor`)
- **Category**: correctness (scanner false negative)
- **Problem**: The scanner's outer loop is `for node in ast.walk(tree): if not isinstance(node, (ast.For, ast.AsyncFor)): continue`. List/set/dict comprehensions and generator expressions (`ast.ListComp`/`ast.SetComp`/`ast.DictComp`/`ast.GeneratorExp`, each containing `ast.comprehension` nodes) are never visited, even though they share the exact same single-function-scope late-binding hazard as an explicit `for` loop.
- **Failure scenario**: Verified at runtime that the bug is real: `handlers = [lambda: x for x in range(3)]; [h() for h in handlers]` → `[2, 2, 2]` (classic late-binding: every lambda sees the final `x`). Running `python -m pyutilz.dev.code_audit` against a file containing exactly that comprehension returns **zero findings** — the scanner exists specifically to catch this bug class and misses it entirely for the comprehension spelling of the same pattern (only the `for`-statement spelling is covered).
- **Suggested fix**: Also walk `ast.comprehension` nodes reachable from `ListComp`/`SetComp`/`DictComp`/`GeneratorExp`, extracting loop-target names the same way `_loop_target_names` does for `ast.For`.

### [MEDIUM] `scan_missed_await`: nested-function shadowing isn't recognized as "rebound", unlike local-import shadowing — src/pyutilz/dev/code_audit/missed_await.py:55-63
- **Category**: correctness (scanner false positive)
- **Problem**: The `rebound` set is populated from `Import`/`ImportFrom` names, `Name` nodes in `Store` context, and function parameters (`ast.arg`) — but **not** from nested `FunctionDef`/`AsyncFunctionDef` names. The module's own docstring explicitly calls out that a local import legitimately "shadows a same-named async method" (restriction #3) — a nested `def` with the same name is the more common way to do exactly the same shadowing, and it isn't handled.
- **Failure scenario**: Verified by running the scanner against:
  ```python
  async def process(item):
      await item.save()

  def sync_wrapper(item):
      def process(x):      # local sync shadow of the module-level async `process`
          x.touch()
      process(item)         # calls the LOCAL sync process — legitimate, no missed await
  ```
  Output: `missed_await` **P1** on `process(item)` inside `sync_wrapper`, claiming it's "an async def in this module, called as a bare statement with the coroutine discarded" — false; it calls the nested synchronous shadow.
- **Suggested fix**: Add `FunctionDef`/`AsyncFunctionDef` names (nested defs) encountered while walking `func` to the same `rebound` set used for imports/params.

### [MEDIUM] `scan_dead_cli_flags`: dynamic `getattr(args, name)` access produces a false positive, contradicting the scanner's own documented guarantee — src/pyutilz/dev/code_audit/dead_cli_flags.py:54-59, 67-74
- **Category**: correctness (scanner false positive)
- **Problem**: `used_attrs` is built via the regex `r"\.([A-Za-z_][A-Za-z0-9_]*)\b"` over raw file text — it only recognizes literal `.attrname` dot-access syntax. Code that reads flags dynamically (`getattr(args, "resume")`, `vars(args)["resume"]`, a loop over flag names) leaves no `.resume` substring anywhere in the corpus. The module's docstring explicitly claims "a name collision with an unrelated `.name` attribute access elsewhere only produces a (safe) false negative, **never a false positive**" — that guarantee does not hold here.
- **Failure scenario**: Verified by running the scanner against:
  ```python
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--verbose", action="store_true")
  ...
  for name in ("resume", "verbose"):
      if getattr(args, name):
          print(f"{name} is set")
  ```
  Both flags are reported as `dead_cli_flag` (P2), "never referenced anywhere in the scanned tree" — false; both are read via `getattr` in a completely ordinary "loop over related flags" idiom.
- **Suggested fix**: Either broaden the corpus scan to also match `getattr(<name-like-args>, ["']<dest>["']` / `vars(<args>)\[["']<dest>["']\]` shapes, or soften the docstring's "never a false positive" claim to acknowledge dynamic-attribute-access as a known blind spot.

### [MEDIUM] `scan_sql_migration_idempotency`: custom PostgreSQL dollar-quote tags aren't recognized as a guard block — src/pyutilz/dev/code_audit/sql_migrations.py:37-46
- **Category**: correctness (scanner false positive)
- **Problem**: `_DO_BLOCK_RE`/`_DO_BLOCK_SPAN_RE` hard-code the bare `$$` delimiter (`r"\bDO\s*\$\$"` / `r"\bDO\s*\$\$(.*?)END\s*\$\$"`). PostgreSQL also allows (and many teams prefer, precisely to avoid ambiguity when the block body itself contains `$$`) custom dollar-quote tags like `$body$ ... END $body$`.
- **Failure scenario**: Verified by running the scanner against a genuinely idempotent, correctly-guarded migration:
  ```sql
  DO $body$
  BEGIN
      IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'pk_users'
      ) THEN
          ALTER TABLE users ADD PRIMARY KEY (id);
      END IF;
  END $body$;
  ```
  Output: `sql_migration_not_idempotent` **P1** — `"ADD PRIMARY KEY has no IF NOT EXISTS form ... and this file has no DO $$ ... END $$ guard block"` — false; the statement is inside a properly existence-guarded `DO` block, just spelled with a custom tag.
- **Suggested fix**: Generalize the regexes to `\bDO\s*(\$\w*\$)` and reuse the captured tag to find the matching `END\s*\1`.

### [MEDIUM] `Finding.as_md_row()` only escapes `|` in `snippet`, not in `detail` — and `default_via_or.py`'s own messages contain literal `|` — src/pyutilz/dev/code_audit/_base.py:36-39 / src/pyutilz/dev/code_audit/default_via_or.py:273-278
- **Category**: correctness (tool output corruption)
- **Problem**: `as_md_row()` (line 38) escapes pipes only in `self.snippet`; `self.detail` is interpolated raw into the Markdown table row. Several scanners write `detail` text containing an actual pipe character — most directly, `default_via_or.py`'s constructor-default message: `` "...LHS is almost certainly an `X | None` parameter..." ``.
- **Failure scenario**: Verified by running the actual scanner and Markdown renderer against `cfg = cfg or SomeConfig()`:
  ```
  | Low | default_via_or | ctor_default.py:2 | `cfg = cfg or SomeConfig()` | `or ClassName(...)`: constructor default -- LHS is almost certainly an `X | None` parameter and instances are always truthy, so only None triggers the fallback. Verify the class has no custom __bool__/__len__. |
  ```
  The unescaped `|` inside "an `X | None` parameter" splits the Detail cell into extra columns when rendered by any Markdown viewer (GitHub, mkdocs, VS Code preview) — a common, realistic finding shape (any `or ClassName()` default) corrupts the tool's primary human-facing report format.
- **Suggested fix**: Escape `|` in `detail` the same way it's escaped in `snippet` inside `as_md_row()`.

### [MEDIUM] `float_distinct_digits_percent`/`integer_digits`: silently wrong for negative numbers — src/pyutilz/core/pythonlib.py:335-380
- **Category**: correctness / edge-case
- **Problem**: `integer_digits` (line 336) is `while n > 0: ...` — for `n <= 0` it returns `(0, set())` immediately. `float_distinct_digits_percent` computes `int_part = int(number)` (which is negative for a negative `number`) and passes it to `integer_digits` **without `abs()`** (only the fractional-part computation applies `abs`), so the entire integer part's digit contribution is silently dropped for any negative input.
- **Failure scenario**: Verified: `float_distinct_digits_percent(11.882, precision=3)` → `0.6` (matches the function's own docstring example). `float_distinct_digits_percent(-11.882, precision=3)` → `0.6666666666666666` — a different, asymmetric result for the negated version of the exact same digits, because the `1`,`1` from the integer part are silently excluded from both the numerator and denominator only in the negative case.
- **Suggested fix**: Call `integer_digits(abs(int_part))` (mirroring the existing `abs(number - int_part)` used for the fractional part).

### [MEDIUM] `weekofmonth`: off-by-one at every 7-day boundary — src/pyutilz/core/pythonlib.py:525-527
- **Category**: correctness / edge-case
- **Problem**: `return date.day // 7 + 1` — for `day in 1..6` this yields week 1 (6 days), but `day == 7` already yields `7 // 7 + 1 == 2`, pushing day 7 into week 2 while days 1-6 stay in week 1. The docstring claims "7-day buckets starting at day 1", which would require `(date.day - 1) // 7 + 1`.
- **Failure scenario**: Verified: `[weekofmonth(date(2024,1,d)) for d in (1,6,7,8,13,14,15,20,21,28)]` → `[1, 1, 2, 2, 2, 3, 3, 3, 4, 5]`. Every multiple of 7 (7, 14, 21, 28) is bucketed into the *next* week rather than closing out the *current* 7-day bucket, contradicting the documented "7-day buckets starting at day 1" contract for 4 out of every ~28-31 days.
- **Suggested fix**: `return (date.day - 1) // 7 + 1`.

### [MEDIUM] `HashableDict.__hash__` crashes on mixed-type keys — src/pyutilz/core/pythonlib.py:855-859
- **Category**: correctness / edge-case
- **Problem**: `hash(tuple(sorted(self.items())))` — `sorted()` on `(key, value)` tuples compares element-wise starting with the keys; Python 3 keys of genuinely incomparable types (e.g. `str` vs `int`) raise `TypeError` on the first `<` comparison `sorted()` needs to perform.
- **Failure scenario**: Verified: `hash(HashableDict({1: 'a', 'b': 2}))` → `TypeError: '<' not supported between instances of 'str' and 'int'`. Any JSON-like dict with both string and integer keys (a very ordinary shape) makes this "hashable dict" recipe crash instead of hashing.
- **Suggested fix**: Sort by `str(k)` (or `(type(k).__name__, k)`) instead of raw `k`, or document the mixed-key-type limitation explicitly.

### [MEDIUM] `create_tabs`: documented "tabTooltip" tuple element is computed then silently discarded — src/pyutilz/dev/dashlib.py:324-334
- **Category**: correctness (dead code / silently dropped feature)
- **Problem**: The docstring documents the tab tuple format as `(label, tab_id, allowed_user_roles, tabClassName, labelClassName, tabTooltip)`. `_tabTooltip` is initialized (line 324) but the code that should populate it from the 6th element is a bare, un-assigned expression statement: `tabClassNames[2]` (line 334), immediately followed by a commented-out `# print('tabTooltip=%s' % tabTooltip)`. `_tabTooltip` is never read again anywhere in the function (no `dbc.Tooltip(...)` is ever built — that line, too, is commented out at line 342).
- **Failure scenario**: A caller passes `tabsList=[("Home", "home", None, "cls1", "lbl1", "This is a tooltip")]` expecting a tooltip per the documented tuple shape; the 6th element is computed and immediately thrown away — the tooltip feature is completely non-functional with no error or warning.
- **Suggested fix**: `_tabTooltip = tabClassNames[2]` and wire it into a `dbc.Tooltip(...)` as the adjacent commented-out code suggests was originally intended.

### [MEDIUM] `ensure_installed` shells out to bare `"pip"` instead of the running interpreter's pip — src/pyutilz/core/pythonlib.py:49-71
- **Category**: correctness / edge-case (environment-dependent)
- **Problem**: `subprocess.check_call(["pip", "install", pkg])` (line 69) resolves `"pip"` via `PATH`, which is not guaranteed to be the pip belonging to `sys.executable`. On a machine with multiple Python installs (this very repo's own CLAUDE.md-documented dev environment explicitly calls out multi-Python-version quirks on Windows), `"pip"` on `PATH` can point at a different interpreter than the one running `ensure_installed`.
- **Failure scenario**: A user running under `python3.14` (per this project's own Windows path notes) with a different, older Python's `pip.exe` earlier on `PATH` calls `ensure_installed("somepkg")`; the package installs into the *wrong* interpreter's site-packages, `importlib.util.find_spec` under the *running* interpreter still can't find it, and the calling code proceeds as if the dependency were satisfied (the failure is only logged at `debug` level per line 71, easy to miss) or silently continues to fail downstream.
- **Suggested fix**: `subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])`.

### [MEDIUM] `log_loaded_rows`: `lang` auto-detection reads this module's own globals, not the caller's — src/pyutilz/dev/logginglib.py:264-266
- **Category**: correctness / design smell
- **Problem**: `lang = globals().get("reports_language", "en")` — inside `logginglib.py`, `globals()` always resolves to `pyutilz.dev.logginglib`'s own module namespace, not the caller's. Since nothing in this module ever defines `reports_language`, this line is, for every real caller, a very roundabout way of writing `lang = "en"` unless the caller explicitly does `import pyutilz.dev.logginglib as logginglib; logginglib.reports_language = "ru"` (monkey-patching this specific module) — nowhere documented as the required mechanism.
- **Failure scenario**: An application defines its own `reports_language = "ru"` in its own module/config (the natural place a developer would expect a "current report language" global to live) and calls `log_loaded_rows(obj, source, lang=None)` expecting Russian messages; it silently always gets the English message instead, with no error or warning.
- **Suggested fix**: Either require `lang` to be passed explicitly (drop the `globals()` lookup), or look it up via `pyutilz.core.pythonlib.lookup_in_stack("reports_language")` (which this codebase already has, and which does search the *caller's* stack) instead of this module's own `globals()`.

### [LOW] `lookup_in_stack` / `ObjectsAndFilesProcessor.process_objects`: ~230x slower frame-globals lookup — src/pyutilz/core/pythonlib.py:591, 761
- **Category**: efficiency
- **Problem**: Both sites do `dict(inspect.getmembers(frame))["f_globals"]` to fetch a frame's globals, instead of the direct attribute access `frame.f_globals`. `inspect.getmembers` enumerates and evaluates *every* attribute of the frame object just to throw all but one away.
- **Failure scenario**: Measured on this machine: `dict(inspect.getmembers(frame))["f_globals"]` averaged 34.3 µs/call vs `frame.f_globals` at 0.15 µs/call — **~230x** slower. `lookup_in_stack` (line 582-594) performs this once *per frame of the entire call stack* it walks, so the cost scales with call-stack depth (e.g. inside a deeply nested framework call chain).
- **Suggested fix**: `caller_globals = frame[0].f_globals` (matches the same class of fix already applied elsewhere in this file for `get_parent_func_args`'s `FrameLocalsProxy`, per that function's own documented perf rationale).

### [LOW] `matrix.py`: duplicate module docstring — the richer one is dead code — src/pyutilz/core/matrix.py:1,3
- **Category**: correctness (dead code) / docs
- **Problem**: Two top-level string-literal statements open the file: line 1 ("Incremental builders and utilities for scipy sparse (CSR/COO) matrices.") and line 3 ("Incremental builders and **memory-usage helpers** for scipy sparse (CSR/COO) matrices.") — only the *first* is recognized by Python as `__doc__`; the second is a no-op expression statement.
- **Failure scenario**: Verified: `import pyutilz.core.matrix as m; m.__doc__` → the first (less complete) string. The file actually contains `get_sparse_memory_usage()`, so the *second*, discarded string is the more accurate description — any doc generator (mkdocs, `help()`) shows the stale text.
- **Suggested fix**: Delete line 1 (or line 3) so only the intended docstring remains.

### [LOW] `src/pyutilz/__init__.py.old`: stray, git-tracked, non-gitignored dead file with an orphaned `NotebookColors` enum
- **Category**: OSS-practice / repo hygiene
- **Problem**: This file is the pre-src-layout `__init__.py` (per `git log`, last touched in the "Restructure to src-layout with subpackages (v1.0.0)" commit) and is still tracked in git, sitting next to the real `src/pyutilz/__init__.py`. Its `NotebookColors(str, Enum)` (ANSI color codes for notebook output) is not referenced anywhere else in the repository (`grep -rn "NotebookColors"` across the whole tree returns nothing besides this file). Because the extension is `.py.old`, it is invisible to ruff/mypy/interrogate/pytest — it can drift indefinitely with zero tooling ever looking at it.
- **Failure scenario**: N/A (hygiene, not a runtime bug) — but a future contributor grepping for "how are notebook colors handled" finds a plausible-looking `NotebookColors` enum that is entirely unused and unreachable, wasting investigation time; or the intended notebook-coloring feature was silently dropped during the restructure and nobody noticed since nothing imports it.
- **Suggested fix**: Either delete the file (git history preserves it if ever needed), or, if `NotebookColors` is still wanted, migrate it into `pyutilz.dev.notebook_init` (which already owns Jupyter-facing helpers) and delete the `.old` file.

### [LOW] `code_audit` registry has no collision guard for external scanner registration — src/pyutilz/dev/code_audit/registry.py:25-40
- **Category**: architecture
- **Problem**: `SCANNERS` is a plain, mutable, module-level `dict[str, Callable]`. There is no `register_scanner(name, fn)` API and no duplicate-key protection. Any code (a downstream project's own conftest, a second import-time side effect) that does `SCANNERS["nan_equality"] = my_fn` silently replaces a built-in scanner for the rest of the process, with no warning, and `run_all`/`cli.main` would then silently run the substituted function under the original name.
- **Failure scenario**: Two independent test-suite fixtures (or a downstream project vendoring an "extra" scanner under a name that happens to collide with a future pyutilz-added built-in) each mutate `SCANNERS` directly; whichever import runs last wins silently, and CI output for that check name no longer matches what its docstring/CLI help describes.
- **Suggested fix**: Add a small `register_scanner(name, fn, *, allow_override=False)` helper that raises on an existing key unless explicitly told to override, and have `SCANNERS` remain the internal storage behind it.

### [LOW] `get_attr`: `default_value=None` is silently coerced to `[]`, so a caller can never get `None` back — src/pyutilz/core/pythonlib.py:174-187
- **Category**: correctness (API surprise)
- **Problem**: `default_value: object = None` is the signature's default, but line 179-180 (`if default_value is None: default_value = []`) means passing `default_value=None` explicitly (matching the declared default) behaves identically to not passing it at all — the caller can never actually get `None` back for a missing/unwanted key, only `[]`.
- **Failure scenario**: `get_attr({"a": 1}, "b", default_value=None)` returns `[]`, not `None`; code that does `if get_attr(d, "b", default_value=None) is None: ...` (an entirely natural way to read the signature) never takes that branch.
- **Suggested fix**: Use a private sentinel (`_UNSET = object()`) to distinguish "no default supplied" from "caller explicitly wants `None`", or document this coercion prominently in the docstring (currently only documents `unwanted_value`, not this).

### [LOW] `scan_duplicate_conditions`: structural (not semantic) equality misflags "retry with an impure call" idioms — src/pyutilz/dev/code_audit/duplicate_conditions.py:70-91
- **Category**: edge-case (scanner false positive)
- **Problem**: The BoolOp duplicate-operand check compares operands via `ast.dump(node)` (pure syntactic structure), with no notion of purity/determinism. Two syntactically identical but *impure* calls (e.g. drawing from a queue, retrying a flaky/random operation) are indistinguishable from a genuine copy-paste typo.
- **Failure scenario**: `success = attempt() or attempt()` (a real, if uncommon, "retry once independently" idiom for something non-deterministic) gets flagged `duplicate_condition` P2, "likely a copy-paste typo ... the intended second check is silently never performed" — which is wrong when `attempt()` has side effects/non-deterministic output.
- **Suggested fix**: Skip the flag when the repeated operand is a `Call` whose callee isn't a bare attribute/name known to be a pure predicate (heuristically hard; at minimum, note the limitation in the docstring the way `default_via_or.py` does for its own false-positive classes).

### [LOW] `scan_nan_equality`: attribute-name match ignores the base object/module — src/pyutilz/dev/code_audit/nan_equality.py:31-33
- **Category**: edge-case (scanner false positive)
- **Problem**: `isinstance(node, ast.Attribute) and node.attr == "nan"` matches *any* object's `.nan` attribute, regardless of whether the base is `np`/`math`/`numpy` or an unrelated object that happens to have a field literally named `nan`.
- **Failure scenario**: `if result.nan == expected.nan:` where `.nan` is some unrelated field on a custom class (not a floating-point NaN) would be flagged `nan_equality` P0 ("NaN does not equal anything ... result is always False") even though no actual float NaN is involved. Narrow (uncommon field name), but a real false-positive source with no base-module check.
- **Suggested fix**: Additionally require the attribute's `.value` to resolve (via a simple `Name`/`Attribute` chain) to `np`/`numpy`/`math`, at least as a heuristic narrowing.

### [LOW] `scan_mutation_during_iteration`: P1 detail text always describes dict/set add-key semantics, even for list receivers — src/pyutilz/dev/code_audit/mutation_during_iteration.py:106-130
- **Category**: docs / clarity (not a functional bug)
- **Problem**: The `d[...] = ...` branch's `detail` text (line ~123-129) always says "Reassigning an EXISTING key is size-preserving and safe (CPython); ADDING a new key changes size and raises RuntimeError on dict / set." This message is dict/set-specific; when the receiver is actually a `list` being iterated (`for x in mylist: mylist[i] = ...`), subscript assignment can never "add a new key" (it either succeeds on an in-range index or raises `IndexError`, never silently grows) — the message can mislead a triager into worrying about the wrong failure mode.
- **Failure scenario**: A reviewer sees a `mutation_during_iteration` P1 finding on a list subscript-assignment inside a `for` loop, reads "raises RuntimeError on dict/set", and either dismisses it as inapplicable (missing that in-place list reassignment during iteration, while safe re size, can still produce logically-wrong results if it's a *different* list being mutated) or spends time worrying about a dict/set failure mode that cannot occur for a list.
- **Suggested fix**: Track whether the iterated chain looks list-like (heuristically: not `.items()/.keys()/.values()`-derived) and tailor the message accordingly, or make the message backend-agnostic ("if this is a dict/set, adding a new key raises RuntimeError; if a list, verify the index is in-range and this doesn't observably change ordering").

## Things done well

- `safe_pickle.py`'s `safe_dump` is genuinely well engineered: atomic temp-file + `os.replace`, per-path threading locks to make the (payload, sidecar) pair atomic as a unit, and a documented, tested-for Windows `PermissionError`-retry loop for concurrent-rename races (`_replace_with_retry`) — the docstrings explain *why* each piece exists, including the specific race it closes.
- The `code_audit` scanners are unusually self-aware for a lint tool: most modules document the exact false positives that were found and fixed during real corpus triage (`default_via_or.py`'s `_DOCUMENTED_SAFE_LHS_FUNCS`/`_is_alias_key_fallback`/boolean-context exclusion; `missed_await.py`'s three numbered precision restrictions; `broad_except.py`'s best-effort-op and import-guard suppressions) — this is exactly the right way to keep a heuristic AST scanner's noise down, and it shows in how few *additional* false positives I was able to find versus how many I expected going in.
- `benchmarking.py`'s `sweep_backend_crossover`/`sweep_backend_grid`/`_rank_candidates` are thoughtfully designed around a real, cited incident (contended-GPU mis-ranking) and default to the more robust interleaved-min ranking while keeping the legacy mean mode available for A/B — good "measure before optimizing, keep the losing variant" discipline.
- `meta_test_utils.py`'s `sentinel_for_type` correctly handles the `types.UnionType` vs `typing.Union` split across Python 3.10-3.14 (including the fact that 3.14 unified the two representations, which would have masked the gap if only tested on 3.14) — careful, version-matrix-aware engineering.
- `dead_cli_flags.py` and `missed_await.py` both explicitly document their precision-vs-recall trade-offs (which direction a miss goes) rather than silently picking one — makes it easy to reason about what a "no findings" result actually guarantees.
- `registry.py`/`cli.py` cleanly separate "run scanners and return data" (`registry.run_all`) from "parse argv, render, decide exit code" (`cli.main`), and `main(argv=None)` returning an int rather than calling `sys.exit` directly is good testability practice.

## Investigated, not an issue

- **`safe_pickle.py`'s per-path lock dict (`_path_locks`) growing unbounded**: confirmed it's keyed by absolute path and never evicted, but this is a bounded, intentional trade-off (one small `threading.Lock` per distinct path ever dumped, for the process lifetime) explicitly scoped in the module's own docstring to "this process" — not a leak in any practical sense for the module's use case.
- **`serialize()`'s "Bug: this passed fname..." comment in `serialization.py:66-76`**: reads like a live bug at first glance, but the surrounding code already fixes it (passes `os.path.dirname(fname)`, guarded by `if dirname:`) — the comment is retrospective documentation of a *fixed* historical bug, not a currently-live one. Confirmed by reading the actual code path, not just the comment.
- **`initialize_function_log`/`load_object_params_into_func` frame-introspection helpers in `logginglib.py`/`pythonlib.py`**: considered flagging more `inspect.stack()`/`inspect.getargvalues()` call sites for the same `FrameLocalsProxy`-style inefficiency documented for `get_parent_func_args`, but these are one-shot calls (not iterated per stack frame like `lookup_in_stack`), so the absolute overhead is far smaller; not worth a separate finding beyond the two sites already flagged.
- **`notebook_init.py`'s `main()` → `init_notebook(inject_globals=True)` frame-walk targeting the wrong namespace when invoked via `%run -m`**: traced the frame chain (`init_notebook`'s `f_back` under `main()` resolves to the `__main__`/`notebook_init` module's own globals, not obviously "the notebook's namespace") and suspected a bug, but IPython's `%run` magic has its own post-execution namespace-merge behavior for scripts run as `__main__` that could plausibly make this work correctly in practice; couldn't construct a fully concrete, IPython-independent failure scenario, so left it out per the "no hand-waving" rule rather than report a guess.
- **`count_trailing_zeros`'s separator character class `",.e+-"` including `e`/`+`**: `format(number, f".{precision}f")` (the only formatting used) never actually produces scientific notation or a `+` sign, so those characters in the class are dead weight, not a correctness bug — confirmed the docstring's own example (`count_trailing_zeros(1.30e-6, precision=8) == 1`) still computes correctly; too cosmetic to list as a finding.
- **`duplicate_dict_key` scanner and Python's own `True`/`1` key-collision semantics**: verified `{1: "a", True: "b"}` collapses to one key in real Python dicts (`hash(True) == hash(1)`), and confirmed the scanner's `seen_keys` dict-membership check (`k.value in seen_keys`) correctly mirrors that collapsing behavior via the same hash/`==` semantics — no bug, a nice accidental correctness property.
