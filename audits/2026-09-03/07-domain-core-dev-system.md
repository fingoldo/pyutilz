# Core / Dev / System Domain Audit — pyutilz (2026-09-03)

## Summary

Read `CLAUDE.md` in full and respected every dispositioned decision there (no repo-wide reformat proposals, no community-health-file findings, no PEP 639 licence migration, no trimming of environment-specific `# type: ignore` codes). Read `audits/implemented/2026-09-02/07-domain-core-dev-system.md` in full and re-checked all 38 of its findings: **all 37 COMPLETED items are still fixed — no regressions** — and F36 stays REJECTED and closed. Nothing below re-raises a COMPLETED item, with one deliberate exception called out inline (F05: the 2026-09-02 report's summary line 11 rejected the cp1251 hypothesis for `ci_log_analyzer`; that rejection was re-measured on this box and does not hold, so the finding is raised with the reproduction attached).

Scope covered: `src/pyutilz/core/` (incl. `pythonlib/`), `src/pyutilz/system/` (incl. `system/` and `scheduling/`), `src/pyutilz/dev/` (incl. all 91 scanners of `dev/code_audit/` plus `_base.py`, `registry.py`, `cli.py`, `__init__.py`).

Work performed:

- **Python 3.8 hard-constraint sweep** over all 135 files in the three domains: an AST script flagging PEP 585 / PEP 604 constructs only in positions *evaluated at runtime* (function parameter/return annotations, class- and module-level `AnnAssign`) across the 32 files lacking `from __future__ import annotations` — clean; plus `vermin -t=3.8 --violations`. The two `vermin` hits (`system/config.py` `tomllib`, `system/distributed.py` `hashlib.md5(usedforsecurity=)`) were read and are correctly guarded (`try: import tomllib / except ImportError: import tomli`; explicit 3.9-vs-3.8 branch at `distributed.py:87-93`). Exactly one genuine violation survived (F06). `.github/workflows/ci.yml:37` confirms 3.8 is a live CI leg. Independently, `ast.parse(..., feature_version=(3,8))` passes over all 20 `system/` files.
- **The `ast.Index` class flagged on 2026-09-02 is closed.** `_base._subscript_index` (`_base.py:196-208`) unwraps by class *name*; the two direct `isinstance(..., ast.Index)` sites (`mutable_defaults.py:149`, `silent_escalation.py:35`) are guarded; `ast.Index` still exists and emits no warning on 3.14 (verified with `C:/Python314/python.exe`). No scanner reads `.slice` unwrapped (`provenance_flow.py:57,66`, `stats_key_coverage.py:82`, `unit_suffix_mismatch._subscript_key` all route through `_base`). Two *analogous* version-dependent degradations were found instead (F62, F63).
- **Registry/facade coherence**: enumerated every module-owned `scan_*` function against `get_scanners()` (89 registered, 91 defined) and against `__init__.__all__` (70 exported), and diffed both directions. Confirmed no duplicate registry ids, and that every `OPT_IN_ONLY` name resolves.
- **Live reproduction** of essentially every finding against minimal trees under `D:\Temp` using `D:/ProgramData/anaconda3/python.exe` (3.11) with the repo `src` on `sys.path`. Observed outputs quoted below are copied from real runs.
- Verified `run_all(parallel=True)` and `run_all(parallel=False)` produce byte-identical output on `src/pyutilz/core` (10 findings each) — the docstring's claim holds for built-in scanners, and fails only for runtime-registered ones (F04).

**Counts by severity: 4 Critical, 28 High, 73 Medium, 111 Low — 216 findings, all OPEN.**

## Findings

### F01. [Critical] `hash_array_summary` hashes only head/tail rows for every non-numeric dtype, so different bool / datetime64 / string arrays collide onto the same cache key — src/pyutilz/core/disk_cache.py:133
- **Disposition**: OPEN
- **Category**: silent-wrong-answer
- **Problem**: the column-statistics block is gated on `np.issubdtype(arr.dtype, np.number)`. `np.bool_`, `datetime64`, `timedelta64` and fixed-width string dtypes are not subtypes of `np.number`, so only shape, dtype and the first/last 64 rows enter the digest. The module docstring at disk_cache.py:66-69 promises "even a single-row change to a middle row is caught when the column-sum changes".
- **Failure scenario**: three pairs of 10000-row arrays differing only at row 2500/5000 — one `bool`, one `datetime64[s]`, one `<U2` — all produced identical keys. A cached result computed from the first array is returned for the second: a silent wrong answer, not a miss.
- **Suggested fix**: view bool/datetime64/timedelta64 through their integer counterparts and run the same reductions; refuse to summary-hash string/void/object dtypes (fall back to a full digest).

### F02. [Critical] `hash_array_summary` of an `object`-dtype array hashes raw heap pointers, giving a different key in every process — src/pyutilz/core/disk_cache.py:131
- **Disposition**: OPEN
- **Category**: unstable-cache-key
- **Problem**: `.tobytes()` is called unconditionally on the head/tail slices; for `dtype=object` that serialises `PyObject*` addresses, and the column-statistics block is skipped, so the addresses are the only content-bearing input.
- **Failure scenario**: identical input array, three fresh interpreters → keys `93ae5520…`, `81427668…`, `29086628…`. Any pandas frame carrying an `object` column gets a 0% cross-process cache-hit rate and one duplicate cache file per worker. This is the same class as the already-fixed F06 of 2026-09-02, one dtype away.
- **Suggested fix**: reject `arr.dtype.kind == "O"` from the summary path, or route it through `_feed(arr.tolist())`.

### F03. [Critical] `heartbeat_scraper` passes the `(sql, params)` tuple as the SQL statement, so no heartbeat can ever execute — src/pyutilz/system/distributed.py:201
- **Disposition**: OPEN
- **Category**: wrong-argument-shape
- **Problem**: `db.safe_execute(get_heartbeat_sql(status, ip))`, but `get_heartbeat_sql` returns `Tuple[str, Optional[tuple]]` (distributed.py:161) and `safe_execute(statement, data=None, ...)` forwards `statement` straight to `cur.execute(statement, data)`. The cursor receives a 2-tuple as query text and `None` as parameters.
- **Failure scenario**: every heartbeat fails at the driver. `register_scraper` calls this at distributed.py:157, so registration itself fails at its last step. `tests/test_distributed.py:268`'s `fake_safe_execute(sql_params)` enshrines the wrong convention and hides the defect.
- **Suggested fix**: `sql, params = get_heartbeat_sql(...); if sql: db.safe_execute(sql, params)`; correct the test fake's signature so it cannot re-hide this.

### F04. [Critical] `run_all(parallel=True)` — the default — raises `KeyError` for every scanner added through the public `register_scanner()` API — src/pyutilz/dev/code_audit/registry.py:269
- **Disposition**: OPEN
- **Category**: process-pool-state-not-propagated
- **Problem**: `_run_one` looks the scanner up by name in the worker's own `_SCANNERS`, which each `ProcessPoolExecutor` worker rebuilds by re-importing `registry` — so it holds only the 89 `register_scanner(...)` calls hard-coded at registry.py:109-224. A registration made by the parent never reaches the child. `run_all` defaults to `parallel=True` (registry.py:320) and takes the pool path whenever `len(selected) >= _MIN_SCANNERS_FOR_PARALLEL` (=4, registry.py:280) — i.e. on every realistic run. `register_scanner`'s own docstring (registry.py:97-103) explicitly anticipates "a downstream project's own scanner".
- **Failure scenario**: `register_scanner("my_scanner", my_scanner)` then `run_all(Path(tree))` → observed `parallel FAILED: KeyError KeyError('my_scanner')`. The same call with `parallel=False` returns `['my_scanner', 'mutable_default', 'unraised_exception_class']`.
- **Suggested fix**: send the callable through the pool when picklable (module-level functions are), or partition `selected` into names resolvable from `registry`'s module-level registrations (parallel) and the rest (sequential in the parent). At minimum detect unresolvable names before submitting and raise a message naming `register_scanner`.

### F05. [High] `subprocess.run(..., text=True)` decodes UTF-8 CI logs as cp1251 — mojibake, or the entire log silently becomes `None` — src/pyutilz/dev/ci_log_analyzer.py:106
- **Disposition**: OPEN
- **Category**: encoding
- **Problem**: `_gh_json` (line 106) and `_fetch_job_log` (line 128) both use `text=True`, which decodes with `locale.getpreferredencoding(False)`. On this box that is `cp1251`. **This contradicts the 2026-09-02 report's summary line 11, which rejected the cp1251 hypothesis; it was re-measured here and does not hold.**
- **Failure scenario**: a child emitting UTF-8 bytes including `0x98` (undefined in cp1251) gives `rc 0, stdout None` — the `UnicodeDecodeError` is raised inside subprocess's `_readerthread` and swallowed, so the whole log vanishes with no exception. `analyze_log_text(None, ...)` then raises `AttributeError: 'NoneType' object has no attribute 'splitlines'` (only `CalledProcessError` is caught); `_gh_json` reaches `json.loads(None)` → `TypeError`. Without the undefined byte it merely mojibakes: `'ok ✓ end'` → `'ok вњ“ end'`.
- **Suggested fix**: replace `text=True` with `encoding="utf-8", errors="replace"` on both calls.

### F06. [High] `str.removesuffix` is Python 3.9+, on a code path the 3.8 CI leg executes — src/pyutilz/dev/meta_test_utils.py:708
- **Disposition**: OPEN
- **Category**: py38-incompatibility
- **Problem**: `name = text.removesuffix("()")`. `str.removesuffix` was added in Python 3.9 (PEP 616); on 3.8 it raises `AttributeError`. `from __future__ import annotations` does not help — this is a runtime call. Grep-confirmed as the only occurrence in `src/`.
- **Failure scenario**: on Python 3.8, `unbacked_audit_dispositions` (meta_test_utils.py:713, public via `__all__` at :84) raises `AttributeError: 'str' object has no attribute 'removesuffix'` on the first bare qualified-name citation, turning `tests/test_meta_test_utils_roundtrip.py:241-268` red on the 3.8 CI leg.
- **Suggested fix**: `name = text[:-2] if text.endswith("()") else text`.

### F07. [High] `column_sum_min_max` reads out of bounds on a zero-row array, returning uninitialized memory instead of raising — src/pyutilz/core/array_summary.py:55
- **Disposition**: OPEN
- **Category**: out-of-bounds
- **Problem**: `_fused_col_reductions` seeds `lo = np.float64(a2[0, j])` before the row loop with no `n_rows == 0` guard, and numba compiles bounds-checking off. The public wrapper only guards `arr.ndim < 2` (array_summary.py:117).
- **Failure scenario**: `column_sum_min_max(np.empty((0,3)))` → mins/maxs of `[2.83545254e-310, -7.06364074e-251, 2.94469373e-310]`. The numpy reference path `_numpy_col_reductions` raises `ValueError` for the same input, so the two documented-interchangeable paths disagree.
- **Suggested fix**: guard `arr.shape[0] == 0` in the wrapper, or seed `nan` in the kernel.

### F08. [High] `analyze_range` misses a read-before-write when the write is on the same line and to the left — the exact trap the module exists to catch — src/pyutilz/dev/freevar_analysis.py:166
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: occurrences are ordered by `(lineno, col_offset)` and `first_is_store = occs[0][2]`. In `selected = [i for i in selected if i]` the Store is at col 0 and the Load at col ~11, so the name is classified as neither free nor needing an incoming value. The module docstring names this accumulator pattern as precisely what `find_names_needing_incoming_value` catches.
- **Failure scenario**: over `selected = [i for i in selected if i]` / `total = total + 1`, observed `free: []`, `needs_incoming: [('i', 3)]` — both genuinely-required names omitted. An extracted function built from that signature raises `UnboundLocalError`; the same analysis gates `split_out_module` (freevar_analysis.py:349).
- **Suggested fix**: compare against the enclosing `ast.Assign` statement's position, not the target `Name`'s column offset.

### F09. [High] A registered scanner emits a severity outside the project vocabulary, so all of its findings are silently dropped by the CLI at *every* `--min-severity` setting — src/pyutilz/dev/code_audit/unraised_exceptions.py:139
- **Disposition**: OPEN
- **Category**: wrong-severity-level
- **Problem**: the vocabulary is `P0/P1/P2/Low` (documented at `_base.py:31`, the only keys in `sev_order` at registry.py:357 and cli.py:93). unraised_exceptions.py:139 emits `severity="Medium"` — the sole occurrence in the package (47×`P2`, 33×`P1`, 16×`Low`, 7×`P0`, 1×`Medium`). `cli.py:94` filters with `sev_order.get(f.severity, 99) <= cutoff`, whose most permissive cutoff is `Low` = 3.
- **Failure scenario**: a tree with `class NeverRaisedError(Exception)` (never raised) plus `def g(items=[])`. `run_all` returns both but sorts the Medium one *below* the Low one: observed `'Low' mutable_default 4` then `'Medium' unraised_exception_class 1`. `main([tree])` at the default `--min-severity Low` prints only the `mutable_default` row — the `unraised_exception_class` finding is absent entirely and the exit code stays 0.
- **Suggested fix**: change unraised_exceptions.py:139 to `"P2"`, and make the unknown-severity default in cli.py:94 / registry.py:358 sort and pass as *most* severe (`-1`) so a future stray value is loud instead of invisible.

### F10. [High] Two scanners are exported as public API but never registered, so `run_all()` and `--check` can never run them — src/pyutilz/dev/code_audit/__init__.py:524
- **Disposition**: OPEN
- **Category**: dead-wiring
- **Problem**: `scan_assert_in_loop_reports_only_the_first` (imported at `__init__.py:524`, exported at `:618`) and `scan_reexport_patch_target` (imported at `:526`, exported at `:620`) have no `register_scanner` call in `registry.py`. Enumerating every module-owned `scan_*` function gives 91 defined against 89 registered; the difference is exactly these two. Both work when called directly. `tests/test_code_audit.py:2330` carries the comment "Every scanner in the registry is the facade-level attribute of the same name" but the loop under it only asserts `callable(fn)` — a vacuous assertion, which is why the drift went unnoticed.
- **Failure scenario**: a tree with two `for r in load_rows(): assert r > 0` sweeps. Direct call returns 2 findings (lines 6 and 8); `[f.check for f in run_all(tree, parallel=False) if 'assert_in_loop' in f.check]` returns `[]`. Likewise `scan_reexport_patch_target` returns a correct P2 finding on a `monkeypatch.setattr(facade, "fetch", ...)` tree while `run_all` reports nothing.
- **Suggested fix**: register both. `reexport_patch_target.py` and `patch_target_is_a_reexport.py` are near-duplicate checks and only the latter is registered — a half-finished rename; either register the old one or delete the module and its `__init__` exports. Then replace the vacuous loop at tests/test_code_audit.py:2331 with an assertion that the registry and the facade are in bijection.

### F11. [High] `import_cycles` discards every relative import made from a package `__init__.py`, hiding the most common real cycle shape — src/pyutilz/dev/code_audit/import_cycles.py:44
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_internal_imports` computes `base_parts = current_parts[: -node.level]`. For `pkg/__init__.py` the module name is the package itself, so `current_parts == ["pkg"]` and `base_parts` becomes `[]`; `from .a import f` resolves to `"a"`, whose first component is not `package_name`, and the guard at line 47 drops the edge.
- **Failure scenario**: `pkg/__init__.py` = `from .a import f`; `pkg/a.py` = `from pkg import f`. Observed graph `defaultdict(set, {'pkg.a': {'pkg'}, 'pkg': set()})` — the `pkg -> pkg.a` edge is missing — and `scan_import_cycles(pkg) == []`. Separately, `from . import other` in `pkg/mod.py` yields the edge `pkg.mod -> pkg` rather than `pkg.mod -> pkg.other`, which *fabricates* cycles: `pkg/__init__.py: from pkg.mod import a` + `pkg/mod.py: from . import other` reports a P1 `pkg -> pkg.mod -> pkg` that does not exist.
- **Suggested fix**: when the file is an `__init__.py`, `level=1` resolves to the module's own dotted name — do not strip a component. For `ImportFrom` with `module is None`, emit one edge per alias (`base + "." + alias.name`).

### F12. [High] `import_cycles` returns nothing at all when pointed at the root the CLI's own help text documents — src/pyutilz/dev/code_audit/import_cycles.py:148
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `pkg = package_name or root.name`, and neither `run_all` nor `_run_one` ever passes `package_name`, so the package name is always the scan root's directory name. `cli.py:62` documents `root` as "source-tree root to scan (e.g. ./src)", in which case `pkg == "src"`, no import target starts with `"src"`, the graph is empty, and lines 150-151 return `[]` with no warning.
- **Failure scenario**: `src/pkg/a.py` = `from pkg import b`; `src/pkg/b.py` = `from pkg import a`. `scan_import_cycles(Path("src"))` returns `[]`. The check is a silent no-op for the documented invocation.
- **Suggested fix**: derive package names from the top-level directories under `root` containing `__init__.py` and scan each; warn loudly when the built graph is empty.

### F13. [High] `raising_stub_swallowed` goes fully silent if the scan root sits under any directory named `tests` — src/pyutilz/dev/code_audit/raising_stub_swallowed.py:182
- **Disposition**: OPEN
- **Category**: absolute-vs-relative-path
- **Problem**: `"tests" in py.parts` tests the ABSOLUTE path (`_iter_py_files` yields absolute paths) — the exact bug `_base._is_excluded` was written to prevent, per its own docstring ("359 findings relative, 0 absolute, no error either way").
- **Failure scenario**: `prod.py` swallowing `probe()` in `except Exception` plus `test_x.py` patching it with a raising stub. Root `<tmp>/proj` → 1 finding; the identical tree at root `<tmp>/tests/proj` → no findings.
- **Suggested fix**: compute `rel = py.relative_to(root)` and test `rel.parts`.

### F14. [High] `source_text_assertions` has the same absolute-path bug, inverted, producing mass false positives — src/pyutilz/dev/code_audit/source_text_assertions.py:165
- **Disposition**: OPEN
- **Category**: absolute-vs-relative-path
- **Problem**: `"tests" in path.parts` on the absolute path decides whether a file is a test file.
- **Failure scenario**: `prod.py` with `src = inspect.getsource(g)` / `assert "x" in src`: root `<tmp>/x/` → no findings; the identical file at root `<tmp>/tests/proj/` → flagged at `prod.py:4`. Every production file in a repo checked out under a `tests` directory is treated as a test.
- **Suggested fix**: use the `rel` already computed at source_text_assertions.py:198.

### F15. [High] `unreachable_import_fallback` counts function-local lazy imports as unconditional, declaring live `except ImportError` handlers dead — src/pyutilz/dev/code_audit/unreachable_import_fallback.py:67
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: lines 67-75 collect imports from anywhere in the tree, not only direct children of `tree.body`.
- **Failure scenario**: `def load(): import numpy` above a `try: import numpy / except ImportError: numpy = None` → observed `('unreachable_import_fallback','Low','a.py',6,'import numpy','…so this except ImportError cannot fire.')`. The handler demonstrably does fire.
- **Suggested fix**: only collect imports that are direct children of `tree.body`.

### F16. [High] `additive_epsilon_denominator` never scans module level in any file that contains a function — src/pyutilz/dev/code_audit/additive_epsilon_denominator.py:81
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `scopes = functions if functions else [tree]` — module scope is examined only when the file has no functions at all.
- **Failure scenario**: `def f(): return 1` plus `RATIO = 5.0 / (SCALE + 1e-12)` → no findings. Delete the `def` and the identical division is reported P1.
- **Suggested fix**: always append `tree` to `scopes`; the existing `seen` set de-dupes.

### F17. [High] `effect_flag_outside_its_effect` treats `x = 1` and `x = 1.0` as a True flag — src/pyutilz/dev/code_audit/effect_flag_outside_its_effect.py:56
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `stmt.value.value in _RECORD_VALUES` uses `==`, and in Python `1 == True` and `1.0 == True`.
- **Failure scenario**: `if rows: write_parquet(rows)` then `counts['rows'] = 1` → observed `('effect_flag_outside_its_effect','P2','m.py',5,"counts['rows'] = 1",'a True flag recording `rows` …')`. A plain counter assignment is reported as a mis-placed boolean effect flag.
- **Suggested fix**: `stmt.value.value is True`.

### F18. [High] `nondiscriminating_test` flags the declarative `@pytest.mark.xfail` as an imperative `pytest.xfail()` — src/pyutilz/dev/code_audit/nondiscriminating_test.py:106
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `_own_nodes` starts at `ast.iter_child_nodes(func)`, which includes `decorator_list`.
- **Failure scenario**: `@pytest.mark.xfail(reason='known')\ndef test_thing(): assert 1+1==2` → P1 "imperative-xfail: `pytest.xfail(...)` discards the measurement just taken". The scanner flags the form it recommends.
- **Suggested fix**: skip `decorator_list` in `_own_nodes`, and require the call receiver to be `pytest`, not `pytest.mark`.

### F19. [High] `non_neutral_except_fallback` treats a nested function's `return` as the handler's substitution — src/pyutilz/dev/code_audit/non_neutral_except_fallback.py:286
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the handler body is walked without excluding nested `FunctionDef`/`Lambda` scopes.
- **Failure scenario**: handler body `def cb(): return 0.0` / `register(cb)` / `raise_later()` → observed `('non_neutral_except_fallback','P1','m.py',4,'except ValueError:','handler returns 0.0 …')`. The handler substitutes nothing and re-raises.
- **Suggested fix**: do not descend into nested `FunctionDef`/`Lambda`, mirroring `_own_nodes`.

### F20. [High] `patch_target_is_a_reexport` resolves relative imports one level off for non-`__init__` modules, so its main case never fires — src/pyutilz/dev/code_audit/patch_target_is_a_reexport.py:241
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `prefix = base[: len(base) - level + 1]` where `base` is the facade's own dotted *module* path. `pkg/facade.py` doing `from ._impl import fetch` resolves to `pkg.facade._impl`, which is absent from `modules` and dropped at line 340.
- **Failure scenario**: `pkg/facade.py` + `pkg/_impl.py` + `patch('pkg.facade.fetch')` → no findings. Spelled absolutely (`from pkg._impl import fetch`) it fires; with the facade as `pkg/__init__.py` it fires. The relative-import spelling — the normal one inside a package — is the blind spot.
- **Suggested fix**: for a non-`__init__` file use `base = package.split(".")[:-1]` before applying `level`.

### F21. [High] `sentinel_guard_mismatch` never detects the `-1` sentinel its own header cites as the motivating case — src/pyutilz/dev/code_audit/sentinel_guard_mismatch.py:29
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_FALSY_SENTINELS` contains `-1`, but matching (line 47) is against `ast.Constant` only; `return -1` parses as `UnaryOp(USub, Constant(1))`.
- **Failure scenario**: `get_count()` returning `-1` in its handler plus a caller guarding `if n is None:` → no findings. Change the sentinel to `0` and it fires: `('sentinel_guard_mismatch','P1','a.py',10,…)`.
- **Suggested fix**: unwrap `UnaryOp(USub, Constant)` before comparison.

### F22. [High] `provenance_flow` reports a plain `d["k"]` read as "read with a default", contradicting its own header — src/pyutilz/dev/code_audit/provenance_flow.py:66
- **Disposition**: OPEN
- **Category**: wrong-diagnosis
- **Problem**: the module header states verbatim that a read *without* a default is not reported, but the subscript-Load branch at line 66 reports one.
- **Failure scenario**: writer `{"modality_source": 1}`, reader `return d["modality_sources"]` → observed `('field_read_never_written','P1','a.py',6,…,"'modality_sources' is read with a default but nothing … writes it -- the default wins on every run…")`. That expression raises `KeyError`; there is no default and nothing "wins on every run". A reader acting on the stated diagnosis will look for a wrong default instead of a missing key.
- **Suggested fix**: drop the subscript-Load branch, or give it its own correctly-worded detail.

### F23. [High] `stats_key_coverage` silently disables itself when the reset uses an annotated assignment — src/pyutilz/dev/code_audit/stats_key_coverage.py:51
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `if not isinstance(stmt, ast.Assign): continue` (lines 51/55) skips `ast.AnnAssign`.
- **Failure scenario**: `self.stats: dict = {"a": 0}` plus `self.stats["zz"] += 1` → no findings. Remove the annotation and the same code yields `('stats_key_coverage','P2','a.py',6,…)`. Annotated code — the style `CLAUDE.md` mandates — is exempt.
- **Suggested fix**: handle `ast.AnnAssign` alongside `ast.Assign`.

### F24. [High] `async_primitive_reinit` false-positives on tuple-assigned primitives in `__init__` — src/pyutilz/dev/code_audit/async_primitive_reinit.py:40
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `_is_persistent_target` has no `Tuple`/`List` case, so a tuple-unpacked assignment to `self.x` is not recognised as persistent.
- **Failure scenario**: `class C: def __init__(self): self.a, self.b = asyncio.Lock(), asyncio.Event()` → **two** P1 findings at `e.py:5` ("every call gets its own private instance"). This is exactly the fix the scanner recommends elsewhere.
- **Suggested fix**: recurse into `Tuple`/`List` elements in `_is_persistent_target`.

### F25. [High] `accumulator_helper_bypassed`: `_SETUP_NAMES` substring matching exempts ordinary methods — src/pyutilz/dev/code_audit/accumulator_helper_bypassed.py:37
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `any(hint in lowered ...)` with hints including `"load"`, `"copy"`, `"init"` matches any method whose name merely contains them.
- **Failure scenario**: the same bypass body reported once as `handle_dup` produced no findings when renamed to any of `upload_batch`, `download_page`, `reload_rows`, `payload_scan`, `recopy`.
- **Suggested fix**: match on word/segment boundaries after splitting the name on `_`.

### F26. [High] `accumulator_helper_bypassed`: the "under a lock" exemption fires on `block` and `clock` — src/pyutilz/dev/code_audit/accumulator_helper_bypassed.py:145
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `"lock" in name.lower()` is tested over every name in the whole `with` subtree.
- **Failure scenario**: `with self._block_reader:`, `with blocking_section():`, `with unblocked_ctx:` and `with clock_timer:` each silence a genuine accumulator bypass.
- **Suggested fix**: regex word boundary, and restrict the scan to `node.items[*].context_expr`.

### F27. [High] `column_no_write_path` reports the wrong line whenever the `.sql` file has a block comment — src/pyutilz/dev/code_audit/column_no_write_path.py:52
- **Disposition**: OPEN
- **Category**: off-by-n
- **Problem**: `_strip_comments` deletes `/*…*/` including its newlines (`re.S`), and `_declared_columns` then counts newlines in the shortened text.
- **Failure scenario**: a 4-line leading block comment followed by `CREATE TABLE t (…)` → observed `('column_no_write_path','P1','sql/a.sql',4,'spanning lines */',…)` — off by 3, with a comment fragment as the snippet.
- **Suggested fix**: `_BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)`.

### F28. [High] `docstring_numbers_moved_to_config`'s `_NAMES_A_SOURCE_RE` matches "e.g." / "i.e." / `run.py` and discards the whole line — src/pyutilz/dev/code_audit/docstring_numbers_moved_to_config.py:52
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the pattern `[a-z_]+\.[a-z_]+` matches ordinary prose abbreviations; the match drives a `continue` at line 68 that discards the line.
- **Failure scenario**: control `"""Prunes at a limit of 10 hits, 5 for rare sources."""` is flagged; `"…10 hits, e.g. rare sources."`, `"…i.e. aggressively."` and `"…per run.py invocation."` each produce no findings.
- **Suggested fix**: `\b[a-z_]{2,}\.[a-z_]{2,}\b` plus an explicit `e.g.`/`i.e.` allowlist.

### F29. [High] The credential-name regex is blind to every snake_case compound name — src/pyutilz/dev/code_audit/credential_logging.py:19
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_CREDENTIAL_NAME_RE` wraps each term in `\b`, but `_` is a word character, so `\bpassword\b` does not match inside `db_password`. Verified: `password`/`proxy`/`api_key`/`token` match; `db_password`, `proxy_url`, `user_api_key`, `auth_token`, `access_token` do not. The docstring's own claimed false-positive example, `token_type`, also does not match — the stated behaviour is wrong in both directions.
- **Failure scenario**: `log.info("connecting %s %s %s", db_password, proxy_url, auth_token)` → scanner returns `[]`. The scanner effectively fires only on bare single-word identifiers, the minority naming for credentials.
- **Suggested fix**: match on segment boundaries, e.g. `(?:^|[^A-Za-z0-9])(proxy|password|...)(?:[^A-Za-z0-9]|$)`, or split the identifier on `_` and test the parts.

### F30. [High] `scan_sql_migration_idempotency` matches `exclude_dirs` against the ABSOLUTE path, so an ancestor directory silences the whole scan — src/pyutilz/dev/code_audit/sql_migrations.py:95
- **Disposition**: OPEN
- **Category**: absolute-vs-relative-path
- **Problem**: `any(part in excluded for part in sql_path.parts)`. This scanner walks `root.glob`, not `_iter_py_files`, so it never gets `_base._is_excluded`'s relative-path fix.
- **Failure scenario**: `D:/Temp/audit_verify/build/proj/001.sql` containing `ALTER TABLE t DROP COLUMN c;` scanned with `root=.../build/proj` → `[]`; the identical file under `.../ok/proj` → 1 finding. Any repo checked out below a path component named `build`, `dist`, `env`, `.git`, `node_modules`, … reports zero SQL findings, silently.
- **Suggested fix**: compare `sql_path.relative_to(root).parts`, mirroring `_is_excluded`.

### F31. [High] `dead_wiring` seeds any function merely NAMED in another audited file as live, making the reachability pass vacuous — src/pyutilz/dev/code_audit/dead_wiring.py:98
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `elif any(name in names for other, names in external.items() if other != path)`. `external` is built at line 90 from **all** trees, audited ones included, so a callee mentioned anywhere in a sibling audited file is seeded live before propagation runs. This defeats the module's stated purpose (header comment line 20: "a public function called only by another dead public function is dead too").
- **Failure scenario**: `m1.py: def dead_leaf(): return 1` and `m2.py: def also_dead(): return dead_leaf()`, nothing else. Both are unreachable; the scanner reports only `also_dead`.
- **Suggested fix**: build the seed's cross-file mention set from `consumer_roots` files only (plus module-scope/decorator/entry-point seeds from audited files), and let the propagation loop mark in-tree callees live.

### F32. [High] `tautological_guard` collapses a target to its root object, so a threshold on one attribute plus an equality on a DIFFERENT attribute is flagged — src/pyutilz/dev/code_audit/tautological_guard.py:42
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `_root_of` (used at lines 81 and 86) reduces both `item.score` and `item.label` to root `item`, so the scanner concludes the pin fixes the thresholded value. Pinning `item.label` constrains nothing about `item.score`.
- **Failure scenario**: `if item.score > 0.5 and item.label == "ok":` → P1 finding "combines a threshold on 'item' with a pin of 'item' to one value". One of the most common guard shapes in ordinary code, emitted at P1.
- **Suggested fix**: require `pinned_key == root_of(thresholded_key)` or `thresholded_key.startswith(pinned_key + ".")` before collapsing.

### F33. [Medium] `remove_nas()`'s return value is discarded at both call sites, so nvidia-smi output is never cleaned — src/pyutilz/system/system/probing.py:665
- **Disposition**: OPEN
- **Category**: discarded-return
- **Problem**: `remove_nas` (`system/system/_common.py:78-100`) is pure and mutates nothing; probing.py:665 and :670 call it as bare statements.
- **Failure scenario**: input dict unchanged; the discarded return was `{'b': 3.5, 'c': {}}`. Every `"N/A"` survives into `get_nvidia_smi_info()`, then `to_float("N/A")` raises inside `_collect_sample`, so **every** sample is lost on a card reporting `power_draw: N/A`.
- **Suggested fix**: assign the results at both sites.

### F34. [Medium] `SingleFlightCache.__getstate__` drops the lock but keeps `self._loop`, so any *used* instance is unpicklable — src/pyutilz/system/single_flight_cache.py:192
- **Disposition**: OPEN
- **Category**: pickle-getstate-gap
- **Problem**: `__setstate__` at :202 sets `_loop = None`, showing the intent; `__getstate__` does not remove it.
- **Failure scenario**: after one `asyncio.run`, `pickle.dumps(c)` → `AttributeError: Can't pickle local object 'WeakSet.__init__.<locals>._remove'`. It pickles fine before first use and fails only after real work — the worst failure timing for a joblib/multiprocessing hand-off.
- **Suggested fix**: `state["_loop"] = None` in `__getstate__`.

### F35. [Medium] `UtilizationMonitor.stop()` raises `RuntimeError` when the monitor was never started — src/pyutilz/system/hardware_monitor.py:233
- **Disposition**: OPEN
- **Category**: boundary
- **Problem**: the guard tests `thread is not None`, but `__init__` (:86) already assigns an unstarted `Thread`.
- **Failure scenario**: `UtilizationMonitor().stop()` → `RuntimeError: cannot join thread before it is started`. Any `try/finally: monitor.stop()` masks the primary error with this one.
- **Suggested fix**: `if self.thread is not None and self.thread.is_alive():`.

### F36. [Medium] `applyfunc_parallel(return_dataframe=False)` returns a `tqdm` object, not the documented `list` — src/pyutilz/system/parallel.py:266
- **Disposition**: OPEN
- **Category**: wrong-return-type
- **Problem**: `res = tqdmu(pool.starmap(...))` at :251/:259 wraps an already-complete list, so the bar can never progress and is never `close()`d.
- **Failure scenario**: observed return type `tqdm`, printing `0%| | 0/2` twice. A caller doing `len(result)` or `result[0]` gets a `TypeError`.
- **Suggested fix**: `list(pool.starmap(...))`, and drop the meaningless progress bar.

### F37. [Medium] `count_app_instances(cmdline=...)` does exact-argument matching while documenting substring matching — src/pyutilz/system/system/misc.py:223
- **Disposition**: OPEN
- **Category**: name-behaviour-mismatch
- **Problem**: `cmdline not in proc.cmdline()` tests list-element equality; a live `cmdline()` is e.g. `['...python.exe', '-c', '...']`.
- **Failure scenario**: `count_app_instances(cmdline="scraper.py")` returns 0 for a running `python worker/scraper.py --id 3`, so a duplicate-instance guard silently allows unlimited duplicates.
- **Suggested fix**: test against `" ".join(proc.cmdline())`.

### F38. [Medium] `report_large_objects` scans its own module's globals, never the caller's — src/pyutilz/system/system/misc.py:247
- **Disposition**: OPEN
- **Category**: wrong-scope
- **Problem**: the docstring promises "any module-level global object whose deep size exceeds `min_size_mb`", but `globals()` here is `misc`'s own namespace. `nbig` stays 0, so the tracemalloc comparison guarded by `if nbig > 0` is skipped too.
- **Failure scenario**: the diagnostic reports "no large objects" on the exact run it was added to debug.
- **Suggested fix**: take a `namespace` parameter, or use `sys._getframe(1).f_globals`.

### F39. [Medium] `get_locale_settings` permanently changes the process-wide C locale — src/pyutilz/system/system/misc.py:426
- **Disposition**: OPEN
- **Category**: unrestored-global
- **Problem**: `locale.setlocale(LC_ALL, locale_name)` is never reverted; the docstring's WARNING covers thread-safety only.
- **Failure scenario**: a diagnostic *read* leaves `LC_ALL` changed for every subsequent `strftime` and `LC_NUMERIC` consumer in the process — including float formatting, which changes the decimal separator.
- **Suggested fix**: capture and restore in a `finally` inside the existing lock.

### F40. [Medium] `parse_dmidecode_info` runs `sudo dmidecode` with no timeout, and the hang happens while a lock is held — src/pyutilz/system/system/probing.py:181
- **Disposition**: OPEN
- **Category**: subprocess-hang
- **Problem**: `capture_output=True` leaves stdin attached, so a password-needing `sudo` waits forever. The `nvidia-smi` call at :596 does carry a timeout.
- **Failure scenario**: `get_system_info(return_hardware_details=True)` hangs, and `register_scraper` calls it **while holding `_identity_lock`**, deadlocking every other thread's heartbeat.
- **Suggested fix**: `timeout=10, stdin=subprocess.DEVNULL`, and `sudo -n`.

### F41. [Medium] `get_system_info` swallows every exception and returns a partial dict, voiding `register_scraper`'s "propagates any error" contract — src/pyutilz/system/system/sysinfo.py:326
- **Disposition**: OPEN
- **Category**: silent-error-swallow
- **Problem**: a blanket handler wraps the whole builder; `distributed.py:117-124` is built on the opposite assumption.
- **Failure scenario**: if `socket.gethostname()` (sysinfo.py:184) raises, the returned dict lacks `host_name`/`os_serial`, and `register_scraper` runs its `where_fields="host_name,os_machine_guid,os_serial"` query against a source missing those keys — wrong node identity, no error anywhere.
- **Suggested fix**: narrow the handler to the optional-probe sections only.

### F42. [Medium] `prefect.connect` logs the API key in cleartext at INFO — src/pyutilz/system/scheduling/prefect.py:59
- **Disposition**: OPEN
- **Category**: credential-logging
- **Problem**: `logger.info("prefect_key=%s", prefect_key)` on every `connect()`.
- **Failure scenario**: under `setup_cli_logging()` (INFO by default) the live key lands in stdout, in log files and in CI job logs.
- **Suggested fix**: log `bool(prefect_key)` or a masked prefix.

### F43. [Medium] `float_distinct_digits_percent` mis-counts fractional digits — binary-float truncation plus lost leading zeros — src/pyutilz/core/pythonlib/numerics.py:69
- **Disposition**: OPEN
- **Category**: silent-numeric-error
- **Problem**: `int(frac_part * 10**precision)` truncates (`0.05063*100000 = 5062.999…` → `5062`), and `integer_digits` cannot see leading zeros (`0.005` at precision 3 counts as one digit). Both numerator and denominator are wrong.
- **Failure scenario**: **2557 of 20000 (12.8%)** random 5-decimal values disagree with the string ground truth; `x=25.05063` → `0.8333` against a true `0.7143`. `f(0.0)` returns `1.0`.
- **Suggested fix**: operate on `format(abs(number), f".{precision}f").replace(".", "")`.

### F44. [Medium] `flatten_keys_to_set` drops `stringify` / `verbose` / `max_chars` in every recursive call — src/pyutilz/core/pythonlib/objects.py:72
- **Disposition**: OPEN
- **Category**: parameter-not-forwarded
- **Problem**: lines 72 and 88 forward only `dict_merge_symbol`. Since the function *is* a recursive walk, the other three flags are effectively dead for nested input.
- **Failure scenario**: `flatten_keys_to_set({"a": [object()]}, stringify=True)` → `set()`, silently discarding the value, while the equivalent top-level form works.
- **Suggested fix**: forward all four arguments at both recursion sites.

### F45. [Medium] `flatten_keys_to_set` silently loses the KEY for every string-valued entry — src/pyutilz/core/pythonlib/objects.py:70
- **Disposition**: OPEN
- **Category**: silent-data-loss
- **Problem**: `isinstance(value, (dict, Iterable))` matches `str`, so string values recurse and the `key + ":" + value` merge at :75 is reachable only for non-iterable scalars.
- **Failure scenario**: `{"a":"b"}` → `{'b'}` while `{"a":1}` → `{'a:1'}`. A caller diffing configs sees `{"host":"prod"}` and `{"region":"prod"}` as identical.
- **Suggested fix**: exclude `str`/`bytes` from the iterable test.

### F46. [Medium] `read_timezoned_ts` corrupts any timestamp carrying no timezone offset — src/pyutilz/core/pythonlib/datetimes.py:61
- **Disposition**: OPEN
- **Category**: silent-corruption
- **Problem**: the function searches for `+` then `-` anywhere in the string; an ISO date always contains `-`, so line 69 rejoins the parts and strips **every colon** out of the time.
- **Failure scenario**: `read_timezoned_ts('2020-02-20T11:54:00')` → `'2020-02-20T115400'`, unparseable by any format in the codebase, with no log line.
- **Suggested fix**: anchor on `r'([+-]\d{2}):(\d{2})$'`.

### F47. [Medium] `suppress_stdout_stderr` restores a stale stream when two blocks overlap, leaving `sys.stdout` a permanently closed file — src/pyutilz/core/pythonlib/filesystem.py:285
- **Disposition**: OPEN
- **Category**: race
- **Problem**: the saved-stream snapshot is per-invocation over process-global state.
- **Failure scenario**: two overlapping threads → afterwards `sys.stdout.closed is True` and the next `print` raises `ValueError: I/O operation on closed file`; stdout is lost for the process. Also reachable single-threaded via interleaved generators or `ExitStack`.
- **Suggested fix**: only restore if `sys.stdout is devnull`, guarded by a module-level lock; document non-reentrancy.

### F48. [Medium] `get_partitioned_filepath("")` returns a bare separator, so `os.path.join` resolves to the filesystem root — src/pyutilz/core/pythonlib/filesystem.py:48
- **Disposition**: OPEN
- **Category**: boundary
- **Problem**: `sep.join([]) + sep` is `"\\"`, an absolute prefix that discards everything before it in `os.path.join`.
- **Failure scenario**: `os.path.join("C:\\data", get_partitioned_filepath("") + "x.pckl")` → `'C:\\x.pckl'`. Triggered by any blank slug or id.
- **Suggested fix**: return `""` for an empty name, or pad short names to `depth`.

### F49. [Medium] `get_attr` raises `ValueError` on any numpy-array value — src/pyutilz/core/pythonlib/objects.py:131
- **Disposition**: OPEN
- **Category**: wrong-comparison-operator
- **Problem**: lines 128 and 131 compare with `==` against `unwanted_value=None`.
- **Failure scenario**: `get_attr({"a": np.array([1,2])}, "a")` → `ValueError: The truth value of an array with more than one element is ambiguous`. Any results dict holding a feature vector or Series crashes on lookup.
- **Suggested fix**: use `is` when the sentinel is `None`.

### F50. [Medium] `serialize()` silently rejects `NamedTemporaryFile` and every other non-`io.IOBase` file object — src/pyutilz/core/serialization.py:86
- **Disposition**: OPEN
- **Category**: silent-error-swallow
- **Problem**: the `raise TypeError` at :89 sits inside the function's own `try` and is swallowed by the blanket `except Exception: return None` at :93-95. `tempfile._TemporaryFileWrapper` is not an `io.IOBase` subclass.
- **Failure scenario**: `serialize(obj, NamedTemporaryFile())` returned `None` and wrote nothing; the caller sees a file it believes was written.
- **Suggested fix**: duck-type on `hasattr(fname, "write")`, and move the dispatch above the `try`.

### F51. [Medium] `verify_sidecar` raises `TypeError` out of a function documented to return only True/False — src/pyutilz/core/safe_pickle.py:210
- **Disposition**: OPEN
- **Category**: undocumented-exception
- **Problem**: `expected` is unvalidated on-disk content passed to `hmac.compare_digest`, which rejects non-ASCII `str`. The `except (OSError, UnicodeDecodeError, IndexError)` at :206 covers only the read.
- **Failure scenario**: a sidecar of 64 `é` characters → `TypeError: comparing strings with non-ASCII characters is not supported` escaping both `verify_sidecar` and `safe_load`. `DiskCache.get` catches `PickleVerificationError`/`UnpicklingError`/`EOFError`/`OSError` — not `TypeError` — so it escapes there too.
- **Suggested fix**: validate the sidecar is 64 lowercase hex characters before comparing; otherwise return `False`.

### F52. [Medium] `verify_sidecar` / `safe_load` raise `FileNotFoundError` when the sidecar exists but the payload does not — src/pyutilz/core/safe_pickle.py:209
- **Disposition**: OPEN
- **Category**: undocumented-exception
- **Problem**: `_sha256_of_file(path)` sits outside any handler and the payload's existence is never checked. The codebase produces this state itself: `DiskCache._evict_if_needed` unlinks payload then sidecar with `continue`/`pass` on failure (disk_cache.py:464, :468).
- **Failure scenario**: orphaned-sidecar state → `FileNotFoundError` instead of the documented `False`.
- **Suggested fix**: `if not isfile(path): return False` at the top.

### F53. [Medium] `DiskCache._key_locks` grows without bound — one `threading.Lock` per key ever written — src/pyutilz/core/disk_cache.py:289
- **Disposition**: OPEN
- **Category**: resource-leak
- **Problem**: nothing removes entries, and keys are content-address digests, effectively unique per call. The docstring at :247 claims it "mirrors `safe_pickle.safe_dump`'s per-path lock", but safe_pickle.py:129-135 deliberately refcounts and deletes, with a comment warning against exactly this. `__getstate__` (:270-276) drops the dict, hiding the leak from pickle-based probes.
- **Failure scenario**: 60 puts against a 3000-byte cap → 1 file on disk, 59 evictions, **60 `_key_locks` entries** still resident.
- **Suggested fix**: reuse the refcounted `_PathLockEntry` pattern from `safe_pickle` verbatim.

### F54. [Medium] `get_session_token` returns `None` on total failure and `init()` discards it — src/pyutilz/core/filemaker.py:41
- **Disposition**: OPEN
- **Category**: discarded-return
- **Problem**: `init()` calls `get_session_token()` as a bare statement and returns `None` unconditionally, despite documenting "Must be called once before any other function".
- **Failure scenario**: against an unreachable host `init()` blocks ~100 s, logs ten warnings, then returns normally; `web.connect` keeps the Basic-auth headers from filemaker.py:34-39, so every `post_filemaker_record` 401s and raises a misleading `ValueError` per record at :129.
- **Suggested fix**: capture the token and `raise RuntimeError` when it is `None`.

### F55. [Medium] `create_tabs` seeds the active-tab session key from the tab LABEL, not the tab id — src/pyutilz/dev/dashlib.py:330
- **Disposition**: OPEN
- **Category**: wrong-key
- **Problem**: `tabsList[0][0]` is the label; ids are built from `[1]` at :364.
- **Failure scenario**: `session["tabsMainActiveTab"] = 'tabOverview'` while the real ids are `['tabov','tabdt']`. No tab renders selected, `activeLabelClassName` never applies, and the content callback is invoked with a nonexistent id.
- **Suggested fix**: `prefix + (tabsList[0][1] or tabsList[0][0])`.

### F56. [Medium] `sweep_backend_grid` / `sweep_backend_crossover` silently truncate non-integer axis values to `int` — src/pyutilz/dev/benchmarking.py:483
- **Disposition**: OPEN
- **Category**: silent-numeric-coercion
- **Problem**: benchmarking.py:483 (and :249) cast axis values with `int()`; `axes` is documented as `{dim: [values]}` with no integrality requirement.
- **Failure scenario**: `axes={"density":[0.25,0.5]}` emits two measured cells both bounded `density_max: 0`, so the `kernel_tuning_cache` matcher can never distinguish them and every real density falls to the catch-all entry.
- **Suggested fix**: preserve the axis value's type; warn when `int(v) != v`.

### F57. [Medium] `@logged` re-raises a logging/DB failure from its `finally`, replacing the wrapped function's real exception — src/pyutilz/dev/logginglib.py:411
- **Disposition**: OPEN
- **Category**: exception-masking
- **Problem**: `finalize_function_log` is unguarded in the `finally` and reaches `safe_execute_values` (:233). An exception raised from `finally` supersedes whatever was propagating, and on the success path it also discards the return value.
- **Failure scenario**: a function raising `ValueError("REAL ERROR")` with a failing finalizer surfaced as `RuntimeError: DB DOWN`. A transient DB outage breaks every decorated call site and hides every real error.
- **Suggested fix**: wrap the finalize call in its own `try/except Exception: logger.exception(...)`.

### F58. [Medium] `sentinel_for_type` returns `True` for `bool`, which is also the commonest default — src/pyutilz/dev/meta_test_utils.py:533
- **Disposition**: OPEN
- **Category**: nondiscriminating-sentinel
- **Problem**: lines 533-534 pick `True` as the bool sentinel; the comparison at :568 is `actual != expected`, so it cannot distinguish "parser populated True" from "parser dropped it and the default is True" — unlike the improbable `str`/`int`/`float` sentinels.
- **Failure scenario**: a parser dropping **every** field reports only `['name']` as mismatched; `enabled` passes as intact. That is precisely the "declared field never extracted" bug class the harness exists to catch.
- **Suggested fix**: build the probe from a non-default baseline, or use `not f.default` as the bool sentinel.

### F59. [Medium] `mutable_defaults` misses the canonical P0 shape whenever the mutable default is constructed with arguments — src/pyutilz/dev/code_audit/mutable_defaults.py:33
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the call-form detector requires `default.func.id in {"list","dict","set"} and not default.args and not default.keywords`; any argument disqualifies it.
- **Failure scenario**: `def m(x=dict(a=1)):` / `x.update({"b": 2})` / `return x` → `scan_mutable_defaults` returns `[]`. This is exactly the defect the scanner exists for. The bare `def g(items=[])` form in the same file is caught, so the scanner looks like it is working.
- **Suggested fix**: drop the no-arguments condition for `list`/`dict`/`set`, and add `collections.defaultdict`/`OrderedDict`/`Counter`.

### F60. [Medium] `retry_loops`' "a break in a NESTED loop doesn't count" filter does not filter anything — src/pyutilz/dev/code_audit/retry_loops.py:30
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_loop_has_break` (:28-37) and `_loop_has_bounding_raise` (:40-55) iterate `ast.walk(loop)` and `continue` on a nested `For`/`While`. `continue` skips only the nested loop *node*; `ast.walk` has already queued its children, so the nested `break`/`raise` is still found. The comments at :34 and :53 state the opposite.
- **Failure scenario**: `while True:` whose `try` body contains `for x in range(3): if x: break`, with `time.sleep(1)` in the `except` → observed `scan_retry_loops(tree) == []`. That `while True` has no break of its own and is a genuine unbounded retry loop.
- **Suggested fix**: use an explicit stack that does not descend into nested `For`/`While` subtrees, as `mutable_defaults._walk_skipping_shadowed_scopes` already does for scopes.

### F61. [Medium] Every snippet after a form feed is attributed to the wrong line, because `str.splitlines()` splits on characters the Python tokenizer does not — src/pyutilz/dev/code_audit/_base.py:137
- **Disposition**: OPEN
- **Category**: off-by-n
- **Problem**: `_line_text` indexes `src_lines[lineno - 1]`, and ~90 scanner call sites build `src_lines` with `read_text(...).splitlines()`. `str.splitlines()` breaks on `\x0b`, `\x0c` (form feed — a conventional section separator in Python source), `\x1c`-`\x1e`, `\x85`, `\u2028` and `\u2029`; CPython's tokenizer counts only `\n`/`\r\n` for `lineno`. `Finding.line` stays right while `Finding.snippet` drifts by one per such character.
- **Failure scenario**: `x = 1\n\x0c\ndef f():\n    try:\n        pass\n    except:\n        pass\n` → observed `bare ff.py 6 'pass'`. Line 6 is `    except:`; the snippet is the following line.
- **Suggested fix**: add a shared `_read_src_lines(path)` in `_base` doing `src.replace("\r\n", "\n").split("\n")`, and route every scanner through it.

### F62. [Medium] `assert_in_loop` collapses every finding in a file to one identical `detail` on Python 3.8, defeating the per-site identity its own comment says a baseline needs — src/pyutilz/dev/code_audit/assert_in_loop.py:100
- **Disposition**: OPEN
- **Category**: py38-degradation
- **Problem**: lines 100-101 fall back to the literals `"<item>"` / `"<source>"` when `ast.unparse` is absent (3.8). The comment at :92-95 states the loop target and source go into `detail` *precisely* because "a consumer keying a baseline on (check, file, detail) … would otherwise silence the whole FILE for this check on the first entry" — which is what the 3.8 fallback produces.
- **Failure scenario**: a file with two distinct sweeps in one function. On 3.11 the findings carry `` `for r in load_rows()` `` and `` `for q in load_rows()` ``; with `ast.unparse` removed (simulating 3.8) both read `` `for <item> in <source>` `` — byte-identical `detail`, so one baseline entry silences both sites and every sweep added later in that file.
- **Suggested fix**: build the fallback from `ast.dump(node.target)` / `ast.dump(node.iter)`, the technique `effect_flag_outside_its_effect.py:84` already chose for the same reason; or fall back to the source line text.

### F63. [Medium] `scan_redundant_test_fit_calls` is a guaranteed no-op on Python 3.8 — src/pyutilz/dev/code_audit/redundant_test_fit.py:109
- **Disposition**: OPEN
- **Category**: py38-degradation
- **Problem**: `_call_signature` returns `None` whenever `ast.unparse` is unavailable, and the whole scanner keys on that signature. On 3.8 — a supported, CI-exercised interpreter — the registered scanner returns `[]` for every input with no warning. Documenting it as "degrade to a no-op scan on 3.8" does not make it correct under a hard 3.8 support constraint: the same code is green on one CI leg and red on another.
- **Failure scenario**: any file with two `test_*` functions calling the same `_build_data(100, seed=0)` — reported on 3.9+, silently clean on the 3.8 leg.
- **Suggested fix**: key the signature on `ast.dump` of the arguments; the signature is only used for equality, never displayed verbatim, so a normalised dump is a drop-in.

### F64. [Medium] `getattr_literal_on_known_dataclass` ignores base classes and methods, producing P1 findings on correct code — src/pyutilz/dev/code_audit/getattr_literal_on_known_dataclass.py:19
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `_dataclass_fields` collects only `AnnAssign` nodes in the class's own body — `node.bases` is never read and methods/properties are ignored. Class names are also keyed globally, so same-named dataclasses in different modules collide.
- **Failure scenario**: `@dataclass class Base: shared: int = 0` / `@dataclass class Child(Base): own: int = 0` with a `helper` method / `getattr(c, "shared", None)` and `getattr(c, "helper", None)` → two P1 findings, both wrong: *"'Child' has no field named 'shared' (['own'] are its real fields)"*.
- **Suggested fix**: union in in-tree base-class fields transitively, add the class body's `FunctionDef`/property names, and key `known` by `(module, class)`.

### F65. [Medium] `broad_except` flags a handler that returns a structured error — the very pattern its detail text recommends — src/pyutilz/dev/code_audit/broad_except.py:63
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the silent-body test accepts `if isinstance(s, ast.Return): continue` for a `Return` carrying *any* value; the finding text and the package docs both describe the defect as a **bare** `return`.
- **Failure scenario**: `try: g()` / `except Exception as e: return {"ok": False, "error": str(e)}` → observed one P1 `broad_except_swallow` at line 5, detail *"except Exception with silent body (pass/continue/return, no …"*.
- **Suggested fix**: treat only `Return` with `value is None` (or a constant `None`/`False`) as silent; a returned expression mentioning the bound exception name is an escalation.

### F66. [Medium] `bare_except` flags `except: raise`, which its own docstring promises not to flag — src/pyutilz/dev/code_audit/bare_except.py:12
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the bare-re-raise exemption is applied only inside the `except BaseException:` branch; when `handler.type is None` the function returns `True` before reaching it. bare_except.py:41 states: *"A handler that immediately re-raises (bare `raise`) is not flagged."*
- **Failure scenario**: `try:\n    pass\nexcept:\n    raise` → one P1 finding at line 3, detail *"bare `except:` (or `except BaseException:`) swallo…"*. Nothing is swallowed.
- **Suggested fix**: hoist the bare-re-raise scan above the `handler.type is None` early return.

### F67. [Medium] Substring "is this variable guarded?" matching silences real hits in `vacuous_matching` and `partial_fix` — src/pyutilz/dev/code_audit/vacuous_matching.py:20
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_GUARD_TEMPLATES` entries such as `"if {v}"` are tested with `in` against the raw module source, so any different identifier starting with the variable's name satisfies the guard. `partial_fix.py:38` uses the same technique.
- **Failure scenario**: `def f(x, xs):\n    if xs:\n        pass\n    return all(i in 'abc' for i in x)` → observed `scan_vacuous_empty_pattern_match(tree) == []`. `x` is never guarded; `"if x"` matched inside `"if xs:"`.
- **Suggested fix**: word-boundary regexes (`rf"\bif\s+{re.escape(v)}\b"`), or decide guardedness from the AST.

### F68. [Medium] `unpicklable_resource_state` matches constructors by their last dotted component, flagging picklable domain objects — src/pyutilz/dev/code_audit/unpicklable_resource_state.py:14
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `Pool`, `Event`, `Queue`, `Condition`, `Barrier` and `Semaphore` are matched by last component (`_last_component`, used at :53-63), whereas `_UNPICKLABLE_DOTTED_CTORS` already requires a known base for the ambiguous `Stream`/`Event` spellings — the same discipline is not applied to the bare-name list.
- **Failure scenario**: `self.train_pool = catboost.Pool(X, y)` — picklable, and an extremely common assignment in this codebase's ML domain — yields a P2 `unpicklable_resource_state` finding; a domain `Event(...)` dataclass does the same.
- **Suggested fix**: require a `threading`/`multiprocessing`/`asyncio`/`subprocess`/`queue` base (or an import trace) for every ambiguous bare name.

### F69. [Medium] `undeclared_imports._domain_for` matches domains by bare string prefix, ignoring path boundaries — src/pyutilz/dev/code_audit/undeclared_imports.py:113
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: domain assignment is `rel_path.startswith(prefix)`.
- **Failure scenario**: `webhooks.py` is attributed to the `web` domain, `developer_notes.py` to `dev`, `database_helpers/…` to `database`; each is then checked against the wrong extras group, so a genuinely undeclared import is reported as declared or vice versa.
- **Suggested fix**: compare on `PurePosixPath(rel_path).parts`.

### F70. [Medium] `count_then_fetch_same_table` emits duplicate findings, one per nested function — src/pyutilz/dev/code_audit/count_then_fetch_same_table.py:81
- **Disposition**: OPEN
- **Category**: duplicate-finding
- **Problem**: no `reported` dedup set, unlike `effect_flag_outside_its_effect.py:165`.
- **Failure scenario**: a nested `inner(cur)` with a COUNT/fetch pair emits two byte-identical findings at `m.py:3`.
- **Suggested fix**: add a `(file, line, detail)` dedup set.

### F71. [Medium] `count_then_fetch_same_table` never fires when the fetch precedes the COUNT — src/pyutilz/dev/code_audit/count_then_fetch_same_table.py:91
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the pairing loop only searches `queries[index + 1:]`.
- **Failure scenario**: fetch on line 2, COUNT on line 3 → no findings; reversed → fires. The redundancy and the TOCTOU race are identical in both orders.
- **Suggested fix**: search the whole query list for a partner, not only the suffix.

### F72. [Medium] `guard_decidable_from_constants` treats every keyword-argument name in the tree as an external write — src/pyutilz/dev/code_audit/guard_decidable_from_constants.py:96
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the `for kw in node.keywords` block (lines 96-98) sits under the plain `elif isinstance(node, ast.Call)`, outside the `setattr`/`setdefault`/`update`/`monkeypatch` guard.
- **Failure scenario**: `_ENABLED = False` plus `helper(_ENABLED=1)` anywhere in the tree plus `if _ENABLED: recover()` → no findings; delete the `helper(...)` call and it fires at line 4.
- **Suggested fix**: move the keyword loop inside the `if name in {...}` branch.

### F73. [Medium] `lazy_log_assertion` never harvests `logger.log(LEVEL, fmt, …)` format strings, producing a false positive — src/pyutilz/dev/code_audit/lazy_log_assertion.py:169
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: lines 169-171 take `node.args[0]`, which for `.log()` is the level, not the format string.
- **Failure scenario**: `log.log(logging.WARNING, 'Retried 3 times for %s', x)` plus `assert 'Retried 3 times for' in str(log.log.call_args)` → P2 claiming "no logger format string in this package contains it". The same code spelled `log.warning(...)` is correctly silent.
- **Suggested fix**: use `args[1]` when the attribute is `log`.

### F74. [Medium] `nondiscriminating_test` flags `except Exception: pytest.fail(...)` as swallowing AssertionError — src/pyutilz/dev/code_audit/nondiscriminating_test.py:80
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: only `ast.Raise` and `ast.Assert` count as escape hatches (lines 80-84).
- **Failure scenario**: `try: assert compute() == 3 / except Exception as e: pytest.fail(str(e))` → observed `('nondiscriminating_test','P1','test_c.py',3,…,'swallows-assertionerror: …')`. That handler fails *harder*, not softer.
- **Suggested fix**: treat a call to `pytest.fail`/`self.fail` as an escape hatch.

### F75. [Medium] `non_neutral_except_fallback` misses `AnnAssign` and tuple-unpacking fallbacks, including its own motivating example — src/pyutilz/dev/code_audit/non_neutral_except_fallback.py:291
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: only plain `ast.Assign` with a single `Name` target is recognised.
- **Failure scenario**: three handlers in one file — `v: float = 0.0` (line 5), `a, b = 0.0, 0.0` (line 12), `x = 0.0` (line 18) — produce exactly **one** finding, at line 18. The `_max_err: float = 0.0` shape the module docstring names as its motivating example is the missed one.
- **Suggested fix**: handle `AnnAssign` and `Tuple`/`List` targets.

### F76. [Medium] `asymmetric_except_siblings` reports a single method as its own sibling — src/pyutilz/dev/code_audit/asymmetric_except_siblings.py:111
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the bucket key is (exception set, call name) only, so two handlers inside one method pass the `len(sites) < 2` test.
- **Failure scenario**: one class, one method `run` with two `try/except OSError` blocks → "`Db.run` calls `rollback` bare inside `except OSError`, while its sibling `run` wraps the identical call." There is no sibling.
- **Suggested fix**: require at least two distinct method names in the bucket.

### F77. [Medium] `asymmetric_except_siblings` misses methods nested in a class-body `if`/`try` — src/pyutilz/dev/code_audit/asymmetric_except_siblings.py:104
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `for method in cls.body` walks only direct children of the class body.
- **Failure scenario**: a correctly-reported asymmetric pair wrapped in `if True:` inside the class body → no findings.
- **Suggested fix**: walk the class body recursively for `FunctionDef`/`AsyncFunctionDef`.

### F78. [Medium] `additive_epsilon_denominator` misses `denom: float = d + 1e-12` and chained assignment — src/pyutilz/dev/code_audit/additive_epsilon_denominator.py:49
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: only `ast.Assign` is handled and only `node.targets[0]` is recorded.
- **Failure scenario**: the annotated form produces no findings while the unannotated form is reported at line 3; `a = denom = d + 1e-12` records only `a`.
- **Suggested fix**: handle `AnnAssign` and every element of `targets`.

### F79. [Medium] `additive_epsilon_denominator` keys `seen` on line number, dropping a second padded division on the same line — src/pyutilz/dev/code_audit/additive_epsilon_denominator.py:86
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the de-dup set holds bare line numbers.
- **Failure scenario**: `return (x / (d + 1e-12), x / (e + 1e-12))` → exactly one finding for two distinct defects.
- **Suggested fix**: key on `(lineno, col_offset)`.

### F80. [Medium] `async_primitive_reinit` false-positives on a default-argument primitive, with inverted wording — src/pyutilz/dev/code_audit/async_primitive_reinit.py:230
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: defaults live under `arguments` and are reached by `ast.walk(func)`, but a default is evaluated once at `def` time and therefore *is* shared.
- **Failure scenario**: `async def handler(x, lock=asyncio.Lock())` → P1 "every call gets its own private instance, so concurrent callers never actually coordinate through it" — the exact opposite of the semantics. A reader following the advice would introduce a real bug.
- **Suggested fix**: exclude `func.args` defaults from the walk (they warrant their own, oppositely-worded check).

### F81. [Medium] `async_primitive_reinit` never sees `from asyncio import Lock` or `import asyncio as aio` — src/pyutilz/dev/code_audit/async_primitive_reinit.py:19
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the matcher hard-requires `Attribute(value=Name(id="asyncio"))`.
- **Failure scenario**: `from asyncio import Lock` plus `lk = Lock()` inside an `async def` → no findings; the `asyncio.Lock()` control fires at line 4.
- **Suggested fix**: resolve the module alias and the `from`-import bindings from the file's import nodes.

### F82. [Medium] `sql_selects_unread_column` attributes any same-arity tuple unpacking in the function to the query — src/pyutilz/dev/code_audit/sql_selects_unread_column.py:128
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: lines 128-131 match an unpacking by arity alone, with no link to the cursor.
- **Failure scenario**: `cur.execute("SELECT id, name FROM t")` followed later by an unrelated `lo, hi = compute()` → P2 "this query fetches `name` (bound to `hi`)…". This contradicts the module header's "it has never yet produced a false positive".
- **Suggested fix**: require the unpacked value to derive from the cursor (`fetchone`/`fetchall`/iteration over it).

### F83. [Medium] `sentinel_cached_as_answer` never scans the documented else/fallback arm — src/pyutilz/dev/code_audit/sentinel_cached_as_answer.py:100
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the header at :21-31 promises "or in the `else`/fallback arm right after one", but lines 100-102 visit only `ExceptHandler`.
- **Failure scenario**: `try/except → v = None` then `if v is None: cache[k] = None` → no findings; move the write into the handler and it fires P1 at line 6. The missed form is the shape the header names as the original production defect.
- **Suggested fix**: also visit the `orelse` of an `If` whose test compares against the sentinel.

### F84. [Medium] `stats_key_coverage` reports a read-only `.get(k, 0)` as a write — src/pyutilz/dev/code_audit/stats_key_coverage.py:85
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `.get(...)` on the stats dict is counted as a write.
- **Failure scenario**: `return self.stats.get("never_written_key", 0)` → `('stats_key_coverage','P2','a.py',6,…,"`C` writes `self.stats['never_written_key']`…")`. This contradicts the docstring's "It cannot produce a false positive."
- **Suggested fix**: count only `Store`-context subscripts, `__setitem__`, `setdefault` and augmented assignment as writes.

### F85. [Medium] `source_text_assertions`: sibling-nested-function name leak — the false positive `_scopes` exists to prevent — src/pyutilz/dev/code_audit/source_text_assertions.py:144
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `_scopes` (lines 144-155) yields outer-before-inner, and the `seen` set at :199, :203-205 locks each assert to the first scope that claims it, so an inner assert is judged against the outer scope's bindings.
- **Failure scenario**: two sibling inner functions — `check_source` using `getsource`, `check_behaviour` using `build()` — result in BOTH line 5 and line 8 being reported; line 8 is the docstring's own counter-example.
- **Suggested fix**: yield scopes innermost-first.

### F86. [Medium] `source_text_assertions`: the documented `pytest.fail` guards were never implemented — src/pyutilz/dev/code_audit/source_text_assertions.py:175
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the docstring at :175 describes handling `pytest.fail` guards, but :203 short-circuits with `if not isinstance(node, ast.Assert)`.
- **Failure scenario**: `if "x" not in src: pytest.fail("missing")` → no findings, though it is the same source-text assertion in a different spelling.
- **Suggested fix**: also treat an `If` whose body calls `pytest.fail`/`self.fail` as an assertion.

### F87. [Medium] `test_asserts_against_production_constant`: the path-construction escape only checks bare string Constants — src/pyutilz/dev/code_audit/test_asserts_against_production_constant.py:74
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the escape hatch recognises only a `Constant` string operand.
- **Failure scenario**: `assert p == CHECKPOINT_DIR / f"{name}.jsonl"` and `assert p == CHECKPOINT_DIR / sub` are both flagged P2, though the module's own comment names that shape as must-be-silent.
- **Suggested fix**: treat any `BinOp(Div)` whose left operand is a path-like constant as path construction.

### F88. [Medium] `test_asserts_against_production_constant`: `pytest.approx(...)` suppresses the check the module exists to make — src/pyutilz/dev/code_audit/test_asserts_against_production_constant.py:66
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `if any(isinstance(sub, ast.Call) …): return set()` bails on any call in the compared expression. `_ASSERT_HELPERS` at :27 already lists `"approx"`, so the intent exists but is neutralised.
- **Failure scenario**: `assert f() == pytest.approx(BASE_DELAY * 2)` → no findings; without `approx` it is flagged. It fires only by accident on `pytest.approx(X, 0.1)`, where the tolerance is mis-paired as "the other side".
- **Suggested fix**: unwrap a `pytest.approx(...)` call to its first argument before the re-derivation check.

### F89. [Medium] `uncached_constant_cost_probe`: the `Path.mkdir` probe entry is dead — src/pyutilz/dev/code_audit/uncached_constant_cost_probe.py:27
- **Disposition**: OPEN
- **Category**: dead-rule
- **Problem**: `_dotted_name` (:43-45) yields `".mkdir"` for a receiver it cannot name, so the matcher at :90-93 never matches the registered `Path.mkdir` entry.
- **Failure scenario**: `p.mkdir(parents=True, exist_ok=True)` → no findings; `Path("/tmp/x").mkdir(exist_ok=True)` → no findings. Only the never-written spelling `Path.mkdir(p)` fires. The `os.makedirs` sibling works, so the rule covers half the family it advertises — and `CLAUDE.md`-adjacent guidance records `makedirs(exist_ok=True)` as measurably slower than `exists`+skip, making this the half that matters.
- **Suggested fix**: match on the attribute name `mkdir` when the receiver is `Path`-shaped or unknown.

### F90. [Medium] `uncached_constant_cost_probe` emits a duplicate finding per enclosing function — src/pyutilz/dev/code_audit/uncached_constant_cost_probe.py:129
- **Disposition**: OPEN
- **Category**: duplicate-finding
- **Problem**: lines 129-160 `break` after one finding per function, but a nested `def` is visited both as its own function and as part of its parent.
- **Failure scenario**: `def outer(): def inner(): return subprocess.run([...])` → two findings, both at `b.py:4`.
- **Suggested fix**: skip descent into nested function definitions.

### F91. [Medium] `unit_suffix_mismatch` does not scan annotated or augmented assignments — src/pyutilz/dev/code_audit/unit_suffix_mismatch.py:142
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: lines 142-146 handle `ast.Assign` and call keywords only.
- **Failure scenario**: `work_s: float = totals["minutes"]` → no findings; the unannotated form is reported P2. `work_s += totals["minutes"]` → no findings. Typed code — where unit suffixes are most used, and which `CLAUDE.md` mandates — is exempt.
- **Suggested fix**: handle `AnnAssign` and `AugAssign`.

### F92. [Medium] `unit_suffix_mismatch` reads `_min` as minutes — src/pyutilz/dev/code_audit/unit_suffix_mismatch.py:76
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `_UNIT_FAMILIES` contains the bare token `min`.
- **Failure scenario**: `window_min = bucket_hours` → P2 "`window_min` declares minutes and is assigned `bucket_hours`, which declares hours" — its highest severity here, on a statistical minimum.
- **Suggested fix**: drop bare `min` from the family, keeping `mins`/`minute`/`minutes`.

### F93. [Medium] `unreachable_import_fallback` counts `if`/`else`-branch imports as unconditional — src/pyutilz/dev/code_audit/unreachable_import_fallback.py:67
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: same root cause as F15 — branch bodies are not distinguished from module-body statements.
- **Failure scenario**: `if sys.platform == "win32": import winreg` above a `try/except ImportError` → flagged at line 5 (and again at line 7 for an `else:` branch). A platform-conditional import is the archetype of a conditional import.
- **Suggested fix**: restrict collection to direct children of `tree.body`.

### F94. [Medium] `hardcoded_test_path._is_test_file` tests the absolute path's parts — src/pyutilz/dev/code_audit/hardcoded_test_path.py:20
- **Disposition**: OPEN
- **Category**: absolute-vs-relative-path
- **Problem**: `"tests" in path.parts` where `path` comes from `root.rglob`, so ancestors above `root` count.
- **Failure scenario**: `D:/Temp/audit_verify/tests/myproj/prod.py` containing `DATA = "C:/Users/alice/data.csv"` scanned at `root=.../tests/myproj` → 1 finding on a production file; the identical file under `.../plain/myproj` → 0.
- **Suggested fix**: use `path.relative_to(root).parts`.

### F95. [Medium] `scan_thresholds_below_documented_result` compares the docstring claim against EVERY `>`/`>=` in the function — src/pyutilz/dev/code_audit/measurement_hygiene.py:115
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the `ast.walk(fn)` loop at :115-121 applies no filter tying the comparison to an assertion or to the documented quantity.
- **Failure scenario**: a test whose docstring says "Recovers 7 of 8 demonstration cards", whose real assertion `assert len(values) >= 7` is correct, but which also contains a loop guard `if i > 0:` → a finding at the `if i > 0` line: "documents 7 but asserts >= 0". A pure false positive pointing at the wrong line.
- **Suggested fix**: restrict to `Compare` nodes inside an `ast.Assert`, and report only the minimum such bound.

### F96. [Medium] `readonly_to_numpy_mutation` re-walks nested functions, producing duplicates and cross-scope false positives — src/pyutilz/dev/code_audit/readonly_to_numpy_mutation.py:70
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the outer `ast.walk(tree)` yields both an outer `def` and its nested `def`, and each inner `ast.walk(func_node)` descends into nested defs (lines 70-79).
- **Failure scenario**: (a) a `to_numpy()`→`np.fill_diagonal` pair inside a nested `inner()` is reported **twice** at the same file:line. (b) with the `to_numpy()` assignment inside a nested `helper()` and an unrelated outer `C = 1; np.fill_diagonal(C, 0.0)`, the outer site is flagged — the risky name leaked out of the nested scope, contradicting the docstring's "only same-function, same-name tracking".
- **Suggested fix**: collect risky names and mutator calls per scope without descending into nested `FunctionDef`/`AsyncFunctionDef`/`Lambda`, as `return_annotation._OwnReturnFinder` already does.

### F97. [Medium] `uncurated_star_export` ignores `ImportFrom.level`, resolving `from ..x import *` against the wrong directory — src/pyutilz/dev/code_audit/uncurated_star_export.py:38
- **Disposition**: OPEN
- **Category**: wrong-resolution
- **Problem**: `_resolve_relative_module_path(py, dotted)` (call site :84) receives only the dotted name; `node.level` is checked for `>= 1` at :32 and then discarded, so every relative import resolves as if `level == 1`.
- **Failure scenario**: `outer/inner/__init__.py` doing `from ..shared import *`, where the real target `outer/shared.py` has no `__all__` (should be flagged) but a decoy `outer/inner/shared.py` does → scanner returns `[]`. The mirror case yields a false positive against a module the code never imports.
- **Suggested fix**: pass `node.level` through and walk up `level - 1` parents before joining.

### F98. [Medium] `table_drift` compares every `writerow` in a function against the LAST `DictWriter` header seen — src/pyutilz/dev/code_audit/table_drift.py:74
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the collection loop at :74-80 overwrites `declared`/`declared_line` on each match, then :82-89 checks all `writerow` calls against that single header.
- **Failure scenario**: one function creating `w1 = csv.DictWriter(f1, fieldnames=["a","b"]); w1.writerow({"a":1,"b":2})` and `w2 = csv.DictWriter(f2, fieldnames=["x","y"]); w2.writerow({"x":1,"y":2})` — both internally consistent — emits a **P1** "header-only ['x','y'], row-only ['a','b']".
- **Suggested fix**: key the header by the writer variable assigned from the `DictWriter(...)` call and match `writerow` by receiver.

### F99. [Medium] `spy_arity` produces false positives from cross-class short-name collisions, contradicting its own stated failure mode — src/pyutilz/dev/code_audit/spy_arity.py:48
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: `_collect_prod_call_max` (lines 48-62) keys on `Name.id` **or** `Attribute.attr`, and the lookup at :98-99 uses `_short_name(target)`. The docstring at :141-143 promises "false negatives are the safe failure mode here, not false positives".
- **Failure scenario**: `prod.build_rows(a)` (1 param) patched with a correct 1-arg spy, plus an unrelated `class Other: def build_rows(self, a, b, c)` called as `o.build_rows(1, 2, 3)` → **P1** "spy … accepts at most 1 positional arg(s), but a real call site for 'build_rows' passes 3".
- **Suggested fix**: record bare-`Name` call sites separately from attribute calls; use the bare-`Name` maximum only when the patch target resolves to a module-level `def`, and discount `self` for method call sites.

### F100. [Medium] `sql_migrations` flags commented-out SQL — src/pyutilz/dev/code_audit/sql_migrations.py:105
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the per-line regex scan at :105-109 does no `--` or `/* */` stripping.
- **Failure scenario**: a file whose only content is `-- DROP TABLE legacy_users;  (removed in v2)` and `SELECT 1;` → P1 "DROP TABLE without IF EXISTS", snippet being the comment itself. A changelog note inside a migration becomes a P1.
- **Suggested fix**: strip unquoted `--`-to-EOL and `/* … */` spans before matching.

### F101. [Medium] `sql_lint` never sees f-string SQL, though `_base._sql_text` exists for exactly this — src/pyutilz/dev/code_audit/sql_lint.py:130
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_string_constants` (lines 130-141) collects `ast.Constant` nodes only. An f-string is a `JoinedStr` whose literal pieces are split at every interpolation, so `SELECT` and `LIMIT` usually land in different Constant nodes and no single node passes both gates.
- **Failure scenario**: `sql = f"SELECT id, name FROM users WHERE owner = {user_id} LIMIT 50"` → `scan_sql_limit_without_order_by` returns `[]`; the same query as a plain literal is reported. This blinds all three SQL-lint scanners to interpolated queries, the common shape.
- **Suggested fix**: also feed `_base._sql_text(node, _module_sql_constants(tree))` for `JoinedStr`/`BinOp`/`Name` nodes — it already renders interpolations as `?` so the recovered text stays parseable.

### F102. [Medium] `constructor_param_overwritten` ignores positional-only `__init__` params, valid on the 3.8 floor — src/pyutilz/dev/code_audit/constructor_param_overwritten.py:70
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `params` is built from `args.args | kwonlyargs` only; `_base._arg_names` already handles posonly/vararg/kwarg and is not used. PEP 570 positional-only parameters are valid Python 3.8 syntax, so this is a live gap on the supported floor.
- **Failure scenario**: `def __init__(self, rate, /)` → no findings; drop the `, /` and the same body yields a Low finding at `a.py:6`.
- **Suggested fix**: use `_base._arg_names(func)`.

### F103. [Medium] `per_call_state_on_shared_instance`: any capitalized identifier-shaped string in ANY collection literal marks a class shared — src/pyutilz/dev/code_audit/per_call_state_on_shared_instance.py:102
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: the docstring says "module-level container"; lines 102-105 walk the whole tree.
- **Failure scenario**: a local `labels = ["Worker", "Other"]` in an unrelated function → `('per_call_state_on_shared_instance','P2','a.py',4,'self.last_usage = 1',…)`.
- **Suggested fix**: restrict the scan to module-level assignments.

### F104. [Medium] `per_call_state_on_shared_instance`: substring lock hints suppress real findings — src/pyutilz/dev/code_audit/per_call_state_on_shared_instance.py:15
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_LOCK_HINTS` are matched against `ast.dump(...).lower()` (lines 22-25), so `blocking`, `blocklist`, `unlocked` and `clock` all count as a lock.
- **Failure scenario**: `with self.blocking_io: self.last_usage = 1` → no findings; remove the `with` and it fires.
- **Suggested fix**: word-boundary matching against the context-expression names only.

### F105. [Medium] `per_call_state_on_shared_instance` counts an annotation-only `self.x: T` as a store — src/pyutilz/dev/code_audit/per_call_state_on_shared_instance.py:43
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: lines 43-50 take `child.target` without checking `child.value is not None`.
- **Failure scenario**: `self.last_usage: int` → P2 "…is written during the in-flight async def run()". Nothing is stored; this is the class-scope attribute declaration `CLAUDE.md` explicitly asks for.
- **Suggested fix**: require `child.value is not None`.

### F106. [Low] `DeadLetterQueue.get_recent(0)` returns the entire queue — src/pyutilz/system/resilience.py:358
- **Disposition**: OPEN
- **Category**: boundary
- **Problem**: `queue[-0:]` is a whole-list slice — the same `[-0:]` bug fixed for `max_size` at :316-319 but left here.
- **Failure scenario**: `get_recent(0)` → 3 items, `get_recent(-1)` → 2 items.
- **Suggested fix**: `if n <= 0: return []`.

### F107. [Low] A negative `chunk_size` makes both chunkers yield nothing, silently — src/pyutilz/system/parallel.py:63
- **Disposition**: OPEN
- **Category**: boundary
- **Problem**: parallel.py:63 and :84 clamp only `chunk_size == 0`. The sibling `split_list_into_nchunks_indices` raises for `<= 0` (:103).
- **Failure scenario**: `split_list_into_chunks(range(10), -3)` → `[]`; the caller processes nothing and sees no error.
- **Suggested fix**: `raise ValueError` for `chunk_size < 1`.

### F108. [Low] `count_trailing_zeros(x, precision=0)` counts the integer part's zeros — the boundary the F08 fix does not cover — src/pyutilz/core/pythonlib/numerics.py:87
- **Disposition**: OPEN
- **Category**: boundary
- **Problem**: `format(x, ".0f")` has no decimal separator, so the `break` added by 2026-09-02's F08 never fires.
- **Failure scenario**: `(100.0, 0)` → `2` and `(1000, 0)` → `3`; correct is `0`. (F08's own case `(100.0, 5) == 5` was re-verified as still fixed.)
- **Suggested fix**: `if precision <= 0: return 0`.

### F109. [Low] `benchmark_algos_by_runtime(n_reps=0)` returns the `1e20` sentinel as a real timing — src/pyutilz/dev/benchmarking.py:76
- **Disposition**: OPEN
- **Category**: boundary
- **Problem**: the min-tracking sentinel is returned unchanged when the repetition loop never runs.
- **Failure scenario**: observed `([a, b], [1e+20, 1e+20])`. A dispatcher persists both backends at 1e20 s with arbitrary argsort order.
- **Suggested fix**: `raise ValueError` for `n_reps < 1`.

### F110. [Low] `get_session_token` sleeps a full interval after its last failed attempt — src/pyutilz/core/filemaker.py:85
- **Disposition**: OPEN
- **Category**: retry-logic
- **Problem**: N attempts cost N sleeps rather than N-1.
- **Failure scenario**: the defaults add 10 s of dead time to every failed boot.
- **Suggested fix**: skip the sleep on the final iteration.

### F111. [Low] `safe_repr`'s truncation notice reports the wrong dropped-character count for an odd `max_arg_size` — src/pyutilz/system/monitoring.py:266
- **Disposition**: OPEN
- **Category**: off-by-one
- **Problem**: the code keeps `2*half` characters but reports `len - max_size` as dropped.
- **Failure scenario**: with an odd `max_arg_size` the notice under-reports by one character.
- **Suggested fix**: report `len - 2*half`.

### F112. [Low] `n_samples` is double-counted when a sample fails after the GPU query — src/pyutilz/system/hardware_monitor.py:154
- **Disposition**: OPEN
- **Category**: off-by-n
- **Problem**: `n_samples` is incremented at :154, before the parse loop at :169-202 that can raise and increment `n_sampling_errors` (:118).
- **Failure scenario**: 100 consecutively-failing samples log "100 of 200 failed".
- **Suggested fix**: increment `n_samples` at the end of `_collect_sample`.

### F113. [Low] `create_tabs` raises `IndexError` on an empty `tabsList` and returns `None` on three undocumented paths — src/pyutilz/dev/dashlib.py:330
- **Disposition**: OPEN
- **Category**: boundary
- **Problem**: `tabsList[0]` is unguarded; :321 and :372 return `None` implicitly, and the caller cannot distinguish the reasons. No return value is documented.
- **Failure scenario**: an empty tab list raises `IndexError` at import/render time instead of rendering an empty tab bar.
- **Suggested fix**: guard the empty case and document each return.

### F114. [Low] `to_float` / `is_float` silently turn a comma-decimal into a 10x-larger number — src/pyutilz/core/pythonlib/numerics.py:25
- **Disposition**: OPEN
- **Category**: silent-numeric-coercion
- **Problem**: commas are stripped unconditionally as thousands separators.
- **Failure scenario**: `to_float("1,5")` → `15.0`; this propagates so that `keys_changed_enough({"b":"1,5"}, {"b":"1.5"})` → `True`, a 900% "change" between identical quantities written in two locales.
- **Suggested fix**: strip commas only in genuine thousands positions.

### F115. [Low] `serialize()` reports every failure as a bare `None` its docstring never mentions — src/pyutilz/core/serialization.py:93
- **Disposition**: OPEN
- **Category**: silent-error-swallow
- **Problem**: its round-trip partner `unserialize` was deliberately reframed to raise by 2026-09-02's F27, so the two halves now have opposite error contracts.
- **Failure scenario**: `serialize(threading.Lock(), None)` → `None`; a caller doing `redis.set(k, blob)` stores a `None`.
- **Suggested fix**: re-raise after logging, matching `unserialize`.

### F116. [Low] `keys_changed_enough` reports "no change" when a key disappears entirely — src/pyutilz/core/pythonlib/objects.py:155
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `obj.get(key)` yields `None`, `is_float(None)` is False, and the key is skipped with no log.
- **Failure scenario**: `keys_changed_enough({}, {"b":100})` → `False`, so a change-triggered alert never fires for a vanished metric.
- **Suggested fix**: treat a key present on one side only as a change.

### F117. [Low] `ensure_installed` raises `ModuleNotFoundError` for a dotted name despite documenting "failures are logged, not raised" — src/pyutilz/core/pythonlib/packages.py:24
- **Disposition**: OPEN
- **Category**: undocumented-exception
- **Problem**: `find_spec` sits outside the try.
- **Failure scenario**: `ensure_installed(["numpy","requests","nosuch.child"])` raises at the check stage, so **none** of the earlier packages installs.
- **Suggested fix**: wrap the per-package `find_spec` call.

### F118. [Low] `is_cuda_available`'s `except (ImportError, Exception)` reads narrower than it is, and the result is cached forever — src/pyutilz/core/pythonlib/hardware.py:70
- **Disposition**: OPEN
- **Category**: silent-error-swallow
- **Problem**: 2026-09-02's F38 removed this exact redundancy from `check_cpu_flag`, but the construct survived the module split here. Combined with `@lru_cache(maxsize=1)` (:52), one transient probe failure pins `False` for the process, logged only at DEBUG.
- **Failure scenario**: a driver hiccup at import time silently disables every CUDA fast path for the run.
- **Suggested fix**: plain `except Exception`, WARNING for anything that is not `ImportError`.

### F119. [Low] `get_gpuutil_gpu_info`'s `assert "id" in attrs` precondition disappears under `python -O` — src/pyutilz/system/system/probing.py:755
- **Disposition**: OPEN
- **Category**: assert-as-validation
- **Problem**: `gpu_dispatch.py:125` and `:283` index `g["id"]` unconditionally.
- **Failure scenario**: under `-O` a shape change in GPUtil produces a `KeyError` deep in dispatch instead of a named failure at the probe.
- **Suggested fix**: `raise ValueError`.

### F120. [Low] `init_notebook`'s global-injection handler discards the reason — src/pyutilz/dev/notebook_init.py:174
- **Disposition**: OPEN
- **Category**: silent-error-swallow
- **Problem**: `except Exception:` with no `as e`, unlike every other handler in the file.
- **Failure scenario**: an injection failure produces a generic message with no way to tell what failed.
- **Suggested fix**: bind and log the exception.

### F121. [Low] `ipython.magic()` is removed in IPython 9.0, and the failure is reported as success — src/pyutilz/dev/notebook_init.py:71
- **Disposition**: OPEN
- **Category**: silent-degradation
- **Problem**: notebook_init.py:71, :78, :133, :134 call `ipython.magic()`; each is wrapped in an `except Exception` that only appends to `failed`.
- **Failure scenario**: on IPython 9.0 `init_notebook` still prints `[OK]Notebook initialization complete!` while autoreload is silently off.
- **Suggested fix**: use `run_line_magic`, and surface a non-empty `failed` list at WARNING.

### F122. [Low] `SingleFlightCache.clear()` strands any in-flight waiter forever — src/pyutilz/system/single_flight_cache.py:296
- **Disposition**: OPEN
- **Category**: race
- **Problem**: `_inflight.clear()` discards Events without setting them; the fetcher's `finally` (:285) then pops nothing.
- **Failure scenario**: a `clear()` racing an in-flight fetch leaves that waiter awaiting permanently.
- **Suggested fix**: `evt.set()` for each entry before clearing.

### F123. [Low] `ensure_idle_devices` can wait forever — src/pyutilz/system/system/misc.py:385
- **Disposition**: OPEN
- **Category**: unbounded-wait
- **Problem**: `while True` with no bound, emitting only DEBUG messages.
- **Failure scenario**: a permanently busy device hangs the caller with no visible reason at default log levels.
- **Suggested fix**: add a `max_wait_seconds` parameter and log at WARNING while waiting.

### F124. [Low] `lookup_in_stack` / `get_parent_func_args` retain frame objects and read source per frame — src/pyutilz/core/pythonlib/stackutils.py:33
- **Disposition**: OPEN
- **Category**: resource-retention
- **Problem**: `lookup_in_stack` returns early at :39, but `inspect.stack()` has already built the whole list (including source context) and the retained frames pin large locals.
- **Failure scenario**: a lookup from deep inside a training loop pins every enclosing frame's locals — arrays included — until the returned objects are dropped.
- **Suggested fix**: walk `f_back` manually; `del` frame references in a `finally`.

### F125. [Low] `register_scraper` walks the whole call stack just to identify the caller's module — src/pyutilz/system/distributed.py:81
- **Disposition**: OPEN
- **Category**: efficiency
- **Problem**: `inspect.stack()[1]` materialises `FrameInfo` for every frame and pins the caller's locals for the rest of the call.
- **Failure scenario**: on a deep stack this is both slow and memory-retaining, for a single attribute.
- **Suggested fix**: `sys._getframe(1)`.

### F126. [Low] `_repo_filenames` / `_repo_symbols` are `lru_cache`d on `Path` and never invalidated — src/pyutilz/dev/meta_test_utils.py:245
- **Disposition**: OPEN
- **Category**: stale-cache
- **Problem**: meta_test_utils.py:245 and :252 cache on the root path alone.
- **Failure scenario**: in a pytest session where one test writes a file and a later test scans the same root, the later test sees a stale set — so the result depends on test ordering.
- **Suggested fix**: document and expose `cache_clear`, or key on the root's mtime.

### F127. [Low] Windows subprocess probes decode child output as UTF-8, hiding the real error behind a generic one — src/pyutilz/system/system/sysinfo.py:124
- **Disposition**: OPEN
- **Category**: encoding
- **Problem**: sysinfo.py:124, :134, :139, :147, :263 all do `check_output(...).decode()` against `wmic`/`cat`/`getprop`/`nvcc`, which emit the OEM codepage.
- **Failure scenario**: a localized non-ASCII error message raises `UnicodeDecodeError`, is caught by the broad handler at :127, and is reported as the generic "Could not extract Windows serial!" — the machine falls back to the MAC-derived GUID and the real cause is invisible.
- **Suggested fix**: `encoding="oem", errors="replace"`.

### F128. [Low] `init_logging` attaches a `StreamHandler` with no encoding while the same module formats Russian messages — src/pyutilz/dev/logginglib.py:103
- **Disposition**: OPEN
- **Category**: encoding
- **Problem**: the file handlers are explicitly `encoding="utf-8"` (:97, :100) but the stream handler is not; `log_loaded_rows` emits Cyrillic at :308 when `lang == "ru"`.
- **Failure scenario**: on a cp1251/cp437 console the record yields `--- Logging error ---` instead of the message.
- **Suggested fix**: wrap `sys.stderr.buffer` in a `TextIOWrapper` with `errors="replace"`.

### F129. [Low] `half_open_max_calls` never limits calls — it is only a success count — src/pyutilz/system/resilience.py:238
- **Disposition**: OPEN
- **Category**: name-behaviour-mismatch
- **Problem**: the value is used solely as the number of consecutive successes needed to close the circuit; nothing caps concurrent admissions.
- **Failure scenario**: 200 threads arriving at the recovery boundary are all admitted to the still-recovering service.
- **Suggested fix**: rename to `half_open_successes_to_close`, or add a real admission counter.

### F130. [Low] `monitored` skips the heartbeat entirely when `should_have_data` and the result is falsy — src/pyutilz/system/monitoring.py:157
- **Disposition**: OPEN
- **Category**: undocumented-behaviour
- **Problem**: the early return at :157 precedes `job_completed` (:171) and is not documented.
- **Failure scenario**: a nightly job legitimately returning `{}` triggers a false dead-man's-switch alert.
- **Suggested fix**: document it, or send a heartbeat with a non-zero status code.

### F131. [Low] `occupancy_aware_block_size` bails to the floor width when shared memory does not bind at all — src/pyutilz/system/gpu_dispatch.py:513
- **Disposition**: OPEN
- **Category**: boundary
- **Problem**: `shared_per_sm` is required even when `bytes_per_thread == 0`, which the code itself defines as "does not bind".
- **Failure scenario**: with `max_shared_mem_per_sm: 0` it returns `(32, 0)` — one warp — instead of the 1024 the other limits fully determine, silently crippling occupancy.
- **Suggested fix**: require `shared_per_sm` only when `bytes_per_thread > 0`.

### F132. [Low] `load_object_params_into_func` can never restore anything through `locals()` — src/pyutilz/core/pythonlib/stackutils.py:83
- **Disposition**: OPEN
- **Category**: impossible-mechanism
- **Problem**: writes to an optimized frame's `locals()` snapshot never reach fast locals, in every CPython version.
- **Failure scenario**: the target parameter stays `None`; the function returns `None` either way, so the caller gets no signal that nothing happened.
- **Suggested fix**: return the collected dict for the caller to unpack, and rewrite the docstring.

### F133. [Low] `simplify_types` mutates the caller's dict in place while presenting itself as a transform — src/pyutilz/core/filemaker.py:98
- **Disposition**: OPEN
- **Category**: name-behaviour-mismatch
- **Problem**: the `.copy()` at :92 copies only the iteration view.
- **Failure scenario**: the caller's original dict loses its list fields and its `None`-valued keys after the call.
- **Suggested fix**: build and return a new dict, or rename to `simplify_types_inplace`.

### F134. [Low] `skip_empty_exif` does not skip empty exif — it skips tags Pillow's `TAGS` cannot name — src/pyutilz/core/image.py:115
- **Disposition**: OPEN
- **Category**: name-behaviour-mismatch
- **Problem**: with `skip_empty_exif=False`, raw integer keys and undecoded `bytes` are injected into a dict the function promises is JSON-serializable.
- **Failure scenario**: `json.dumps(get_image_properties(p, skip_empty_exif=False))` raises `TypeError: keys must be str…`.
- **Suggested fix**: rename to `skip_unknown_exif_tags`, keeping a deprecated alias.

### F135. [Low] `count_user_deferred_entries` records `0` for ANY call-expression whitelist, inverting the drift tracker — src/pyutilz/dev/meta_test_utils.py:432
- **Disposition**: OPEN
- **Category**: inverted-metric
- **Problem**: the comment at :432-434 says "empty constructor", but the branch also matches `set(_LEGACY_ENTRIES)`.
- **Failure scenario**: 40 entries report as `0`, so the tracked debt appears to shrink in exactly the case where it grew.
- **Suggested fix**: match only no-argument calls.

### F136. [Low] `snake_case_variants_of` uses an unanchored `replace("_config","")` — the same class as the fixed F33 — src/pyutilz/dev/meta_test_utils.py:460
- **Disposition**: OPEN
- **Category**: unanchored-replace
- **Problem**: the replace strips every occurrence, not the trailing one.
- **Failure scenario**: `"MyConfigManagerConfig"` → `"my_manager"`, so the plausible real binding `my_config_manager` is never generated and a valid symbol is reported as missing.
- **Suggested fix**: an `endswith`-guarded slice.

### F137. [Low] `setup_polars_config` sets `POLARS_MAX_THREADS` after polars may already be imported, and reports success either way — src/pyutilz/dev/notebook_init.py:39
- **Disposition**: OPEN
- **Category**: ineffective-configuration
- **Problem**: polars reads the variable once at import; lines 39-40 may run afterwards.
- **Failure scenario**: prints `Using 8 polars threads` while polars keeps its default pool.
- **Suggested fix**: check `"polars" in sys.modules` first and warn when it is too late.

### F138. [Low] `split_out_module`'s byte-identity check does not cover decorators — src/pyutilz/dev/freevar_analysis.py:238
- **Disposition**: OPEN
- **Category**: incomplete-verification
- **Problem**: `_top_level_bodies` slices from `node.lineno` (the `def`), while `_top_level_span` (:212) deliberately includes decorators in the moved range.
- **Failure scenario**: a move that drops an `@lru_cache` still passes the "every body verified byte-identical" claim printed at :397 — a silent behaviour change presented as verified-safe.
- **Suggested fix**: start the slice at the minimum decorator lineno.

### F139. [Low] `DiskCache.get` cannot represent a cached `None`, and the hit counter contradicts the observed behaviour — src/pyutilz/core/disk_cache.py:315
- **Disposition**: OPEN
- **Category**: sentinel-collision
- **Problem**: `None` is both the miss signal and a legal cached value.
- **Failure scenario**: `put("k", None)` then `get("k")` → `None` with `hits: 1, misses: 0`. The caller recomputes forever while the stats say every lookup hit.
- **Suggested fix**: return a `_MISS` sentinel, as `safe_pickle` and `single_flight_cache` already do.

### F140. [Low] `DiskCache._key_path` admits a key containing a path separator, producing a permanently unwritable key — src/pyutilz/core/disk_cache.py:304
- **Disposition**: OPEN
- **Category**: input-validation
- **Problem**: `"sub/deep"` passes the `parents` traversal guard (`../evil` is correctly rejected) but `put` never creates intermediate directories.
- **Failure scenario**: every `put` for that key fails and every `get` misses, forever, with only the F30-added WARNING.
- **Suggested fix**: reject any key containing a path separator.

### F141. [Low] `get_image_properties` overwrites a caller-supplied `filesize` and drops a genuine 0 — src/pyutilz/core/image.py:72
- **Disposition**: OPEN
- **Category**: parameter-ignored
- **Problem**: :72 overwrites `filesize` whenever `img` is a path, and the emit guard at :96 is `if filesize:`.
- **Failure scenario**: a caller passing a known size has it silently replaced; a genuinely zero-byte file omits the key entirely.
- **Suggested fix**: honour a supplied value and use `if filesize is not None:`.

### F142. [Low] `get_average_utilization` returns `nan` rather than `None` for the three RAM metrics when the series are empty — src/pyutilz/system/hardware_monitor.py:266
- **Disposition**: OPEN
- **Category**: inconsistent-sentinel
- **Problem**: lines 266-268 differ from every sibling field, which returns `None`.
- **Failure scenario**: a consumer doing `if v is None: skip` silently averages `nan` into a report.
- **Suggested fix**: return `None` for consistency.

### F143. [Low] `show_biggest_session_objects` formats a possibly-`None` reading with `%.2f` — src/pyutilz/system/system/memory.py:204
- **Disposition**: OPEN
- **Category**: unguarded-format
- **Problem**: the value can be `None` when the probe fails.
- **Failure scenario**: `TypeError: must be real number, not NoneType` out of a diagnostic helper.
- **Suggested fix**: guard the format.

### F144. [Low] `EXTERNAL_IP` is never assigned, so `include_node_ip=True` always records `{"ip": None}` — src/pyutilz/dev/logginglib.py:37
- **Disposition**: OPEN
- **Category**: dead-feature
- **Problem**: logginglib.py:37 declares it; :83 and :385 read it; the only assignment is commented out.
- **Failure scenario**: the default-on node-IP field is `None` in every log row ever written.
- **Suggested fix**: populate it or remove the parameter.

### F145. [Low] `create_tabs` emits the literal CSS class `"None "` on tabs lacking a `labelClassName` — src/pyutilz/dev/dashlib.py:366
- **Disposition**: OPEN
- **Category**: string-formatting
- **Problem**: an absent value is interpolated rather than defaulted.
- **Failure scenario**: reproduced `labelClassName == 'None '` in the rendered attribute.
- **Suggested fix**: default to `""`.

### F146. [Low] `create_tabs`'s `user = None` fallback does not apply when `is_authenticated` itself raises — src/pyutilz/dev/dashlib.py:315
- **Disposition**: OPEN
- **Category**: incomplete-guard
- **Problem**: `user` is bound to the proxy first (lines 315-325), so an exception raised by the proxy's attribute access escapes the fallback.
- **Failure scenario**: a request outside an app context raises instead of rendering the anonymous view.
- **Suggested fix**: bind inside the `try`.

### F147. [Low] Jobs whose log fetch fails are listed by raw numeric id — src/pyutilz/dev/ci_log_analyzer.py:190
- **Disposition**: OPEN
- **Category**: poor-diagnostic
- **Problem**: `job_names` is populated only on the success path (lines 190-192).
- **Failure scenario**: the failure report names `48213771234` instead of `tests (windows-latest, 3.8)` — the least useful id for exactly the leg that failed.
- **Suggested fix**: populate `job_names` from the job listing before fetching logs.

### F148. [Low] Four `pythonlib` modules contain `>>>foo(...)` with no space, a doctest collection error that hides the valid F34 doctests — src/pyutilz/core/pythonlib/numerics.py:49
- **Disposition**: OPEN
- **Category**: doctest-syntax
- **Problem**: same class as 2026-09-02's F34, fixed only for `ensure_valid_filename`. Sites: `numerics.py:49`, `objects.py:139-148`, `datetimes.py:102`, `filesystem.py:185` and `:210`.
- **Failure scenario**: `--doctest-modules` aborts collection of the whole module, so the correct doctests in it — including the ones F34 added — never run.
- **Suggested fix**: insert the space after `>>>` at all six sites.

### F149. [Low] `load_file` imports pandas unconditionally though only the `.pckl` branch uses it — src/pyutilz/core/pythonlib/filesystem.py:79
- **Disposition**: OPEN
- **Category**: unnecessary-hard-dependency
- **Problem**: the adjacent comment documents exactly this lazy-import reasoning for catboost, but pandas was left behind.
- **Failure scenario**: `load_file("m.joblib")` fails with `ImportError` in an environment that has joblib but not pandas.
- **Suggested fix**: move the pandas import into the `.pckl` branch.

### F150. [Low] `analyze_range` reports a function's own parameters as free names — src/pyutilz/dev/freevar_analysis.py:144
- **Disposition**: OPEN
- **Category**: undocumented-limitation
- **Problem**: parameters are `ast.arg`, not `ast.Name`, so they are never seen as bound (lines 144-149). Acknowledged in a code comment but absent from the docstring's "Known limitation" list.
- **Failure scenario**: `split_out_module` refuses a valid move because the extracted function's own parameters look like free variables.
- **Suggested fix**: seed the bound set from the enclosing function's `arg` nodes, and document the limitation until then.

### F151. [Low] `accumulator_helper_bypassed`'s owner suppression compares bare function names across files — src/pyutilz/dev/code_audit/accumulator_helper_bypassed.py:195
- **Disposition**: OPEN
- **Category**: cross-file-collision
- **Problem**: `_file` is ignored in the comparison.
- **Failure scenario**: a same-named helper in an unrelated module suppresses a genuine bypass here.
- **Suggested fix**: compare `(file, name)`.

### F152. [Low] Ungrammatical detail text — src/pyutilz/dev/code_audit/accumulator_helper_bypassed.py:209
- **Disposition**: OPEN
- **Category**: message-quality
- **Problem**: the rendered detail reads `` `_inc_stat` (a.py) own it ``.
- **Failure scenario**: the finding text is unreadable for the multi-owner case.
- **Suggested fix**: pluralise the verb from the owner count.

### F153. [Low] Stray comma in the detail text — src/pyutilz/dev/code_audit/additive_epsilon_denominator.py:90
- **Disposition**: OPEN
- **Category**: message-quality
- **Problem**: renders as `…an epsilon-padded sum,. Adding a constant…`.
- **Failure scenario**: cosmetic, but the finding is quoted verbatim into audit reports.
- **Suggested fix**: remove the stray comma from the f-string.

### F154. [Low] `asymmetric_except_siblings` points at the `except` line, not the unguarded call its detail names — src/pyutilz/dev/code_audit/asymmetric_except_siblings.py:126
- **Disposition**: OPEN
- **Category**: wrong-location
- **Problem**: `bare_line = node.lineno` takes the handler's line.
- **Failure scenario**: the detail says "calls `rollback` bare" while the line/snippet show `except OSError:`, so a reader jumps to the wrong statement.
- **Suggested fix**: report the offending call's lineno.

### F155. [Low] `asymmetric_except_siblings` counts an inner `try` as "wrapped" regardless of what it catches — src/pyutilz/dev/code_audit/asymmetric_except_siblings.py:58
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: any enclosing `Try` satisfies the guard test.
- **Failure scenario**: `try: rollback() / except ValueError: pass` silences a genuine `OSError` asymmetry.
- **Suggested fix**: require the inner handler to catch a compatible exception.

### F156. [Low] `comment_names_missing_symbol`'s `_NOT_LOCAL` list is dead, which also disables the empty-tree bail-out — src/pyutilz/dev/code_audit/comment_names_missing_symbol.py:39
- **Disposition**: OPEN
- **Category**: dead-rule
- **Problem**: `_BACKTICKED_CALL` captures only `_`-prefixed names, and none of `_NOT_LOCAL`'s ten entries starts with `_` (used at :100). Consequently `if not known: return findings` (:101) is unreachable.
- **Failure scenario**: the intended allowlist never applies, and an empty symbol table produces findings instead of bailing out.
- **Suggested fix**: widen `_BACKTICKED_CALL` or align `_NOT_LOCAL` with it.

### F157. [Low] `comment_names_missing_symbol` reports docstring findings at the `def`/`class` line — src/pyutilz/dev/code_audit/comment_names_missing_symbol.py:80
- **Disposition**: OPEN
- **Category**: wrong-location
- **Problem**: the docstring node's owner lineno is used rather than the citing line.
- **Failure scenario**: a module docstring's stale citation always reports line 1, regardless of where in the docstring it appears.
- **Suggested fix**: offset by the citation's line within the docstring.

### F158. [Low] `_LINE_CITATION` requires a 2-to-5-digit line number — src/pyutilz/dev/code_audit/comment_names_missing_symbol.py:36
- **Disposition**: OPEN
- **Category**: over-narrow-pattern
- **Problem**: `(\d{2,5})` excludes single-digit and ≥6-digit line numbers.
- **Failure scenario**: `# see line 7 of foo` → no findings, while `# see line 42` is reported.
- **Suggested fix**: `(\d+)`.

### F159. [Low] Only the first line citation per comment is checked — src/pyutilz/dev/code_audit/comment_names_missing_symbol.py:148
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `search` is used where `finditer` is needed.
- **Failure scenario**: `# see line 42 and line 99` produces one finding, for 42 only.
- **Suggested fix**: iterate all matches.

### F160. [Low] `constructor_param_overwritten` emits duplicate findings, one per config-reading method — src/pyutilz/dev/code_audit/constructor_param_overwritten.py:294
- **Disposition**: OPEN
- **Category**: duplicate-finding
- **Problem**: no dedup on `(file, line)`.
- **Failure scenario**: a class with three config-reading methods reports the same site three times.
- **Suggested fix**: dedup before returning.

### F161. [Low] `_reads_config` matches any attribute named `config`/`settings`/`environ` anywhere in the method — src/pyutilz/dev/code_audit/constructor_param_overwritten.py:218
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: there is no data-flow link between the matched attribute and the assignment being judged.
- **Failure scenario**: a method that logs `self.config` and separately assigns an unrelated parameter is reported as overwriting a constructor parameter from config.
- **Suggested fix**: require the config read to reach the assignment's value expression.

### F162. [Low] `_PAGINATED` is a whole-text regex, so a `LIMIT` inside a subquery silences the rule — src/pyutilz/dev/code_audit/count_then_fetch_same_table.py:94
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the pagination test is not scoped to the outer query.
- **Failure scenario**: `SELECT * FROM (SELECT id FROM t LIMIT 10) x` suppresses the count-then-fetch finding for the outer statement.
- **Suggested fix**: test only the outermost query's tail.

### F163. [Low] `effect_flag_outside_its_effect` misses `AnnAssign` records — src/pyutilz/dev/code_audit/effect_flag_outside_its_effect.py:56
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: only `ast.Assign` is examined.
- **Failure scenario**: `ok['rows']: bool = True` → no findings; `ok['rows'] = True` fires.
- **Suggested fix**: handle `AnnAssign`.

### F164. [Low] Any string-key subscript store anywhere in the package suppresses a same-named constant everywhere — src/pyutilz/dev/code_audit/guard_decidable_from_constants.py:85
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: lines 85-88 collect subscript-store keys globally with no scoping.
- **Failure scenario**: `d['_ENABLED'] = 1` in an unrelated file silences `if _ENABLED:` in `m.py`.
- **Suggested fix**: scope the suppression to the file, or to a resolvable alias of the constant.

### F165. [Low] `guard_decidable_from_constants` reports an `IfExp` at the body expression's line — src/pyutilz/dev/code_audit/guard_decidable_from_constants.py:194
- **Disposition**: OPEN
- **Category**: wrong-location
- **Problem**: `node.lineno` for an `IfExp` is the start of the value expression, not the test.
- **Failure scenario**: a multi-line ternary reports line 5 with snippet `'a'` while `if _FLAG` sits on line 6.
- **Suggested fix**: use `node.test.lineno`.

### F166. [Low] `lazy_log_assertion` recognises only the bare `assert` statement — src/pyutilz/dev/code_audit/lazy_log_assertion.py:251
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `unittest`-style assertions are not scanned.
- **Failure scenario**: `self.assertIn('reached only 0/3', str(log.warning.call_args))` → no findings.
- **Suggested fix**: also handle `assertIn`/`assertTrue` call forms.

### F167. [Low] `non_neutral_except_fallback`'s detail names a substitution that is not the first in source order — src/pyutilz/dev/code_audit/non_neutral_except_fallback.py:286
- **Disposition**: OPEN
- **Category**: message-quality
- **Problem**: `ast.walk` is breadth-first.
- **Failure scenario**: with several fallbacks in one handler the reported one is arbitrary, so the line and the quoted value can disagree.
- **Suggested fix**: sort candidates by `(lineno, col_offset)` before choosing.

### F168. [Low] `nondiscriminating_test` never scans `*_test.py` files — src/pyutilz/dev/code_audit/nondiscriminating_test.py:145
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `py.name.startswith(test_prefix)` only. The sibling `patch_target_is_a_reexport.py:321` accepts both conventions.
- **Failure scenario**: a project using the `*_test.py` convention gets zero findings from this scanner.
- **Suggested fix**: share one `_is_test_file` helper in `_base`.

### F169. [Low] `_PATCHERS` is dead and `patch.object(...)` is never matched — src/pyutilz/dev/code_audit/patch_target_is_a_reexport.py:211
- **Disposition**: OPEN
- **Category**: dead-rule
- **Problem**: `_PATCHERS = frozenset({"patch", "patch.object"})` at :211 has no other reference, and the matcher at :221-223 sees the callee attribute as `object`.
- **Failure scenario**: `patch.object(mod, "name")` — a standard spelling — is never examined.
- **Suggested fix**: match on the attribute chain, and delete or use `_PATCHERS`.

### F170. [Low] `provenance_flow` reports dict-literal writes at the `ast.Dict` node's line — src/pyutilz/dev/code_audit/provenance_flow.py:53
- **Disposition**: OPEN
- **Category**: wrong-location
- **Problem**: the key's own lineno is available but not used.
- **Failure scenario**: every key in a multi-line literal reports line 3 with snippet `return {`.
- **Suggested fix**: use `k.lineno`.

### F171. [Low] `_is_field_like`'s docstring says "no leading underscore" but the body does not check it — src/pyutilz/dev/code_audit/provenance_flow.py:34
- **Disposition**: OPEN
- **Category**: doc-behaviour-mismatch
- **Problem**: verified `_is_field_like('_secret') → True`.
- **Failure scenario**: private keys enter the provenance graph and are reported as unwritten fields.
- **Suggested fix**: implement the documented check, or amend the docstring.

### F172. [Low] `_PATCH_FUNCS` is dead and disagrees with the inline set actually used — src/pyutilz/dev/code_audit/raising_stub_swallowed.py:28
- **Disposition**: OPEN
- **Category**: dead-rule
- **Problem**: `_PATCH_FUNCS = {"patch","setattr","patch.object"}` has no reference; `_patched_targets` (:136) uses its own `{"patch","setattr","object"}`.
- **Failure scenario**: a future edit to the visible constant changes nothing, which is how a rule silently stops matching.
- **Suggested fix**: delete the constant or make it the single source.

### F173. [Low] `_asserts_on_a_raise` implements one of the two exclusions its docstring claims — src/pyutilz/dev/code_audit/raising_stub_swallowed.py:64
- **Disposition**: OPEN
- **Category**: doc-behaviour-mismatch
- **Problem**: only the `raises` case is handled (lines 64-77); the return-value case lives elsewhere.
- **Failure scenario**: a reader trusting the docstring assumes a test is exempt when it is not.
- **Suggested fix**: correct the docstring to describe the split.

### F174. [Low] A cache write via a method call is only seen as a bare `ast.Expr` statement — src/pyutilz/dev/code_audit/sentinel_cached_as_answer.py:72
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the write detector requires a statement-level expression.
- **Failure scenario**: `return cache.set(k, None)` → no findings.
- **Suggested fix**: also match the call in a `Return`/assignment value.

### F175. [Low] `src: str = inspect.getsource(g)` is not treated as a source binding — src/pyutilz/dev/code_audit/source_text_assertions.py:129
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: lines 129-131 handle `ast.Assign` only.
- **Failure scenario**: the annotated form produces no findings while the unannotated form is flagged.
- **Suggested fix**: handle `AnnAssign`.

### F176. [Low] `dis` is recognised only under its literal name — src/pyutilz/dev/code_audit/source_text_assertions.py:67
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: import aliases are not resolved.
- **Failure scenario**: `import dis as d` / `d.dis(f)` → no findings.
- **Suggested fix**: resolve module aliases from the file's imports.

### F177. [Low] The helper-call arm is not restricted to test functions or `self`/`cls` receivers — src/pyutilz/dev/code_audit/test_asserts_against_production_constant.py:96
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: lines 96-100 accept any `assertEqual`-shaped call anywhere.
- **Failure scenario**: `recorder.assertEqual(BASE_DELAY * 2, x)` inside a non-test helper is flagged.
- **Suggested fix**: require a test-function context and a `self`/`cls` receiver.

### F178. [Low] Unary-negated re-derivation is missed — src/pyutilz/dev/code_audit/test_asserts_against_production_constant.py:71
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `UnaryOp` is not unwrapped (lines 71-72).
- **Failure scenario**: `assert f() == -BASE_DELAY` → no findings.
- **Suggested fix**: unwrap `UnaryOp` before matching.

### F179. [Low] A multi-line helper call reports the opening-paren line and snippet — src/pyutilz/dev/code_audit/test_asserts_against_production_constant.py:100
- **Disposition**: OPEN
- **Category**: wrong-location
- **Problem**: the call node's lineno is used.
- **Failure scenario**: snippet reads `'self.assertEqual('` rather than the compared expression.
- **Suggested fix**: report the offending argument's position.

### F180. [Low] `_is_cached` substring-matches `ast.dump` text, so `@app.route("/cache")` silences the rule — src/pyutilz/dev/code_audit/uncached_constant_cost_probe.py:51
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the decorator test is a substring search over the dumped tree (lines 51-52).
- **Failure scenario**: any decorator whose arguments contain the text `cache` exempts the function.
- **Suggested fix**: match decorator names structurally.

### F181. [Low] Bare single-segment names match the cost-probe table — src/pyutilz/dev/code_audit/uncached_constant_cost_probe.py:92
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: a one-segment call name is compared against dotted probe entries by last component.
- **Failure scenario**: a locally defined `def run()` called from `def probe()` is flagged "spawns a process", in a file with no imports at all.
- **Suggested fix**: require a resolvable module qualifier for single-segment names.

### F182. [Low] Any `global`/`nonlocal` in the body suppresses the finding — src/pyutilz/dev/code_audit/uncached_constant_cost_probe.py:55
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_has_module_level_memo` (lines 55-58) treats any such statement as a memo.
- **Failure scenario**: an unrelated `global counter` exempts a genuinely uncached probe.
- **Suggested fix**: require the declared name to be assigned from the probe's result.

### F183. [Low] One-letter unit tokens produce noise findings — src/pyutilz/dev/code_audit/unit_suffix_mismatch.py:76
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: single-character unit tokens such as `s` are in the family table.
- **Failure scenario**: `ax.scatter(x, y, s=sizes_pct)` → Low "`s` declares seconds and is assigned `sizes_pct`".
- **Suggested fix**: require at least two characters, or a `_`-separated suffix position.

### F184. [Low] Keyword-argument findings use the call's lineno — src/pyutilz/dev/code_audit/unit_suffix_mismatch.py:146
- **Disposition**: OPEN
- **Category**: wrong-location
- **Problem**: the keyword node's own position is available but unused.
- **Failure scenario**: a multi-line `schedule(..., timeout_s=cfg["ms"], )` reports line 1 with snippet `'schedule('`.
- **Suggested fix**: use `kw.value.lineno`.

### F185. [Low] Relative imports never participate in `unreachable_import_fallback` in either direction — src/pyutilz/dev/code_audit/unreachable_import_fallback.py:73
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `node.level == 0` at :73 and `stmt.level == 0` at :119 exclude them.
- **Failure scenario**: a genuinely dead handler around `from . import util` is invisible.
- **Suggested fix**: resolve relative imports to their absolute names first.

### F186. [Low] Only direct children of the `try` body are examined — src/pyutilz/dev/code_audit/unreachable_import_fallback.py:115
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: nested statements inside the guarded block are skipped.
- **Failure scenario**: `try: if True: import numpy` is missed.
- **Suggested fix**: walk the try body.

### F187. [Low] `_reads_configuration` iterates only `ast.Call`, missing subscript access to `environ` — src/pyutilz/dev/code_audit/docstring_numbers_moved_to_config.py:92
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `environ` is in `_CONFIG_READERS` (:45), where subscript access is the normal spelling.
- **Failure scenario**: `os.environ["PRUNE"]` → no findings; `os.environ.get("PRUNE")` → flagged.
- **Suggested fix**: also match `Subscript` on a `_CONFIG_READERS` receiver.

### F188. [Low] `_CACHING_DECORATORS` contains a redundant entry under substring matching — src/pyutilz/dev/code_audit/uncached_constant_cost_probe.py:13
- **Disposition**: OPEN
- **Category**: dead-entry
- **Problem**: `"lru_cache"` is subsumed by `"cache"`.
- **Failure scenario**: harmless today, but it hides the fact that the matching is substring-based (see F180).
- **Suggested fix**: remove the entry when the matching is made structural.

### F189. [Low] `skip_masking_except` only scans `test_*.py`, missing the `*_test.py` convention — src/pyutilz/dev/code_audit/skip_masking_except.py:58
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `if not py.name.startswith("test_")`. `spy_arity._is_test_path` and `hardcoded_test_path._is_test_file` both accept `*_test.py`.
- **Failure scenario**: `widget_test.py` containing `try: result = do_work(1) / except TypeError: pytest.skip(...)` → `[]`. `measurement_hygiene.py:101` has the same narrow check.
- **Suggested fix**: share one `_is_test_file` helper in `_base`.

### F190. [Low] `log_throttle` bumps loop depth for the loop's own `target`/`iter`/`test`, contradicting its docstring — src/pyutilz/dev/code_audit/log_throttle.py:168
- **Disposition**: OPEN
- **Category**: doc-behaviour-mismatch
- **Problem**: :168 and :177 visit `target`/`iter`/`test` at `loop_depth + 1`; the docstring at :147-148 states loops "bump depth for their body/orelse but **not** their own target/iter/test expressions".
- **Failure scenario**: `for x in (logger.error("boom") or items):` → "log.error(...) inside a loop (depth 1)". The iter expression evaluates once, not per iteration.
- **Suggested fix**: visit `target`/`iter`/`test` at `loop_depth`.

### F191. [Low] A cue that normalises to the empty string compiles to `\b\b` and matches every record — src/pyutilz/dev/code_audit/field_text_agreement.py:66
- **Disposition**: OPEN
- **Category**: input-validation
- **Problem**: `parts` can be empty after `normalise_text`, yielding `re.compile(r"\b" + "" + r"\b")`. The `if p` guards at :184-185 reject empty *input* strings, not strings that normalise to empty.
- **Failure scenario**: `normalise_text("_")` → `''`, `_cue_pattern("_").pattern` → `\b\b`; with `cues={"postmortem": ["_"]}`, `cues_in_text(rule, "vital hanging")` → `{'postmortem': '_'}`. As an anti-cue the same input cancels every cue instead.
- **Suggested fix**: return a never-matching pattern (or raise) when `parts` is empty.

### F192. [Low] `unraised_exceptions` keys classes by bare name, so same-named exception classes in different files collapse — src/pyutilz/dev/code_audit/unraised_exceptions.py:58
- **Disposition**: OPEN
- **Category**: cross-file-collision
- **Problem**: `all_classes: dict[str, ...]` at :58 with an overwriting assignment at :90.
- **Failure scenario**: `e1.py` and `e2.py` each defining `class DupError(Exception)` with no raise anywhere → exactly one finding, on `e2.py`. Same-named per-module error classes are ordinary in the provider/plugin layouts the header comment itself cites.
- **Suggested fix**: key on `(rel, name)`, keeping name-keyed sets only for the raise/subclass lookups.

### F193. [Low] `_is_zero_literal` returns True for `False` — src/pyutilz/dev/code_audit/llm_max_tokens_cap.py:16
- **Disposition**: OPEN
- **Category**: bool-int-conflation
- **Problem**: `node.value == 0` is True for `False` and `0.0`, though the docstring says "the bare integer literal `0`".
- **Failure scenario**: `p.generate("prompt", max_tokens=False)` → flagged as having no explicit cap.
- **Suggested fix**: `isinstance(node.value, int) and not isinstance(node.value, bool) and node.value == 0`.

### F194. [Low] `network_timeout` cannot see a directly-imported `urlopen` — src/pyutilz/dev/code_audit/network_timeout.py:70
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the matcher requires `isinstance(node.func, ast.Attribute)`.
- **Failure scenario**: in one file, `urlopen(u).read()` (line 5, from `from urllib.request import urlopen` — the more common spelling than the `urllib.request.urlopen(u)` named in the docstring) is missed while `requests.get(u).text` (line 6) is caught.
- **Suggested fix**: also handle bare `ast.Name` callees resolved through the file's `ImportFrom` nodes.

### F195. [Low] `asymmetric_resource_guard` emits P0 on in-memory containers — src/pyutilz/dev/code_audit/asymmetric_resource_guard.py:29
- **Disposition**: OPEN
- **Category**: wrong-severity-level
- **Problem**: `update`, `write`, `delete`, `insert`, `query` (lines 29-32) are generic method names, and the scanner requires no evidence the receiver is a DB handle.
- **Failure scenario**: a class whose `a()` does `with self._lock: self._cache.update({"k": 1})` and whose `b()` does `self._cache.update({"j": 2})` → **P0** "operation `self._cache.update` is guard-wrapped in ['a'] but NOT in ['b']". A plain dict under a lock is a normal, deliberate asymmetry, and P0 is the package's crash-level tier.
- **Suggested fix**: emit P2/Low unless the receiver resolves to a connection/cursor-shaped name, or drop the generic container verbs from the default suffix set.

### F196. [Low] `domain_boundary` accepts a `severity` parameter but hardcodes `"P1"` for `boundary_symbol_missing` — src/pyutilz/dev/code_audit/domain_boundary.py:136
- **Disposition**: OPEN
- **Category**: parameter-ignored
- **Problem**: the parameter is documented at :114 as "severity tag for emitted findings", unqualified.
- **Failure scenario**: a caller lowering the severity for an advisory boundary still gets P1 rows, which gate CI via cli.py:99.
- **Suggested fix**: apply `severity`, or document that this one id is fixed at P1 and why.

### F197. [Low] `check_record` picks the alphabetically-first cued value when the text cues several — src/pyutilz/dev/code_audit/field_text_agreement.py:211
- **Disposition**: OPEN
- **Category**: arbitrary-tiebreak
- **Problem**: `next(iter(sorted(hits.items())), ("", ""))`. When the declared value is not among the hits, which of several supported values becomes the reported `supported` — and therefore whether `opposes()` fires and which partition is compared — depends on alphabetical order, not evidence. Nothing in the docstring says so.
- **Failure scenario**: the same record yields a different verdict purely because a value was renamed.
- **Suggested fix**: prefer the value whose winning cue is longest or earliest in the text, and record the discarded alternatives in the verdict.

### F198. [Low] `credential_logging` only recognises loggers literally named `logger`/`log`/`logging` — src/pyutilz/dev/code_audit/credential_logging.py:62
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `isinstance(node.func.value, ast.Name) and node.func.value.id in ("logger", "log", "logging")` skips `self.logger.info(...)`, `self._log.warning(...)` and `LOGGER.info(...)`.
- **Failure scenario**: `self_logger.info("password is %s", password)` → not reported. The sibling `log_throttle._is_log_call` (:117-135) already handles attribute receivers and the `*log`/`*logger` suffix; this security-adjacent scanner is the stricter one.
- **Suggested fix**: reuse `log_throttle`'s receiver test.

### F199. [Low] Nine emitted `check` ids do not exist in the registry, so a reader cannot re-run the check they were given — src/pyutilz/dev/code_audit/cli.py:66
- **Disposition**: OPEN
- **Category**: id-drift
- **Problem**: `--check` accepts `choices=sorted(get_scanners())`, i.e. registry keys, but scanners emit `Finding.check` strings that differ in nine cases: `assert_in_loop_first_failure_only`, `boundary_symbol_missing`, `busy_retry_loop`, `duplicate_dict_key`, `duplicate_function_body_subset`, `field_read_never_written`, `field_written_never_read`, `reexport_patch_target`, `unbounded_retry_loop`. E.g. `retry_loops.py:141,152` emit `busy_retry_loop`/`unbounded_retry_loop` while the scanner is registered as `retry_loop`.
- **Failure scenario**: a user reads `unbounded_retry_loop` in the report and runs `--check unbounded_retry_loop` → argparse rejects it as an invalid choice; a downstream baseline keyed on `check` cannot be mapped back to a scanner.
- **Suggested fix**: register one key per emitted id (or accept emitted ids as aliases), plus a meta-test asserting every emitted `check` resolves.

### F200. [Low] `--min-severity` silently deletes findings whose severity it does not recognise, even at its most permissive setting — src/pyutilz/dev/code_audit/cli.py:94
- **Disposition**: OPEN
- **Category**: silent-drop
- **Problem**: `sev_order.get(f.severity, 99) <= cutoff`, whose largest cutoff is `Low` = 3. The exit-code line at :99 also recognises only `P0`/`P1`, so such a finding gates nothing either.
- **Failure scenario**: reproduced under F09 — the `unraised_exception_class` row is absent from `main([tree])`'s output while `run_all` returned it.
- **Suggested fix**: map an unknown severity to `-1` so it always renders and always gates, and log a warning naming the value.

### F201. [Low] `default_via_or` skips every or-chain with more than two operands — src/pyutilz/dev/code_audit/default_via_or.py:409
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `if len(node.values) != 2: continue`, undocumented in the scanner's docstring.
- **Failure scenario**: `x = arg or fallback or 5` carries exactly the flagged trap (a falsy-but-valid `arg` such as `0` or `""` silently becomes `fallback`) and is never reported, while `x = arg or 5` in the same file is.
- **Suggested fix**: examine adjacent operand pairs in a chain of any length, or document the restriction.

### F202. [Low] `import_cycles` names a file that does not exist for a package-level cycle member — src/pyutilz/dev/code_audit/import_cycles.py:173
- **Disposition**: OPEN
- **Category**: wrong-location
- **Problem**: `file=comp[0].replace(f"{pkg}.", "", 1)…` — when the representative is the package itself the prefix does not match, so the replace is a no-op.
- **Failure scenario**: the fabricated-cycle tree from F11 reports `file="pkg.py"`; the real location is `pkg/__init__.py`. A file-keyed baseline entry never matches.
- **Suggested fix**: emit `<name>/__init__.py` when the dotted name resolves to a package directory.

### F203. [Low] `vacuous_assertions`' `test_glob` parameter is not a glob and inverts its own filter — src/pyutilz/dev/code_audit/vacuous_assertions.py:152
- **Disposition**: OPEN
- **Category**: name-behaviour-mismatch
- **Problem**: `if not py.name.startswith("test_") and test_glob == "test_*.py": continue` — the pattern is compared for equality against the default, never used as a glob.
- **Failure scenario**: `scan_vacuous_assertions(root, test_glob="check_*.py")` scans every `.py` file in the tree, production modules included.
- **Suggested fix**: `if not fnmatch(py.name, test_glob): continue`.

### F204. [Low] `resource_handle_safety` misses every non-builtin acquisition form, including `Path.open()` — src/pyutilz/dev/code_audit/resource_handle_safety.py:20
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: the acquisition set matches a bare `Name` `open` plus the attributes `NamedTemporaryFile`/`TemporaryFile`/`SpooledTemporaryFile`/`Popen`; `func.attr == "open"` is never tested.
- **Failure scenario**: `f = Path(p).open()` outside a `with` (likewise `io.open`, `gzip.open`, `codecs.open`, `socket.socket()`) is not reported, though it leaks a handle identically to the `f = open(p)` the scanner does catch — and `Path(...).open()` is the spelling this codebase prefers.
- **Suggested fix**: add `attr == "open"` and the `socket.socket` constructor.

### F205. [Low] `console_unicode` inspects only the first positional argument — src/pyutilz/dev/code_audit/console_unicode.py:26
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: only `node.args[0]` is scanned for non-ASCII content.
- **Failure scenario**: `print("done", "→")` and `logger.info("x %s", "→")` are not flagged, though on a cp1251 console they raise `UnicodeEncodeError` exactly as `print("→")` does — and cp1251 console breakage is a standing hazard in this project.
- **Suggested fix**: scan every positional argument and each `JoinedStr` part.

### F206. [Low] `missed_await` never examines module-level statements — src/pyutilz/dev/code_audit/missed_await.py:52
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: only nodes inside a `FunctionDef`/`AsyncFunctionDef` are considered; the blind spot is undocumented.
- **Failure scenario**: a discarded coroutine at module scope — a script body or an `if __name__ == "__main__":` block calling an `async def main()` — produces no finding, and the program exits having done nothing.
- **Suggested fix**: also walk `tree.body` at module scope.

### F207. [Low] 21 registered scanners are unreachable from the package's public surface — src/pyutilz/dev/code_audit/__init__.py:570
- **Disposition**: OPEN
- **Category**: dead-wiring
- **Problem**: `__all__` lists 70 `scan_*` names against 89 registered scanners — the exact inverse of F10. Verified unreachable: `scan_accumulator_helper_bypassed`, `scan_asymmetric_except_siblings`, `scan_column_no_write_path`, `scan_comment_cites_absolute_line`, `scan_comment_names_missing_symbol`, `scan_constructor_param_overwritten`, `scan_count_then_fetch_same_table`, `scan_docstring_numbers_moved_to_config`, `scan_effect_flag_outside_its_effect`, `scan_guard_decidable_from_constants`, `scan_lazy_log_assertion`, `scan_patch_target_is_a_reexport`, `scan_raising_stub_swallowed`, `scan_sentinel_cached_as_answer`, `scan_sentinel_guard_mismatch`, `scan_source_text_assertions`, `scan_sql_selects_unread_column`, `scan_stats_key_coverage`, `scan_test_asserts_against_production_constant`, `scan_unit_suffix_mismatch`, `scan_unreachable_import_fallback`. The cleanup loop at :670-692 `globals().pop`s the submodules, so they are not reachable by submodule attribute either.
- **Failure scenario**: `from pyutilz.dev.code_audit import scan_unit_suffix_mismatch` → `ImportError: cannot import name 'scan_unit_suffix_mismatch' from 'pyutilz.dev.code_audit'`, and `hasattr(code_audit, "scan_unit_suffix_mismatch")` is `False`. A caller wanting one scanner in isolation must reach into the private `registry._SCANNERS`.
- **Suggested fix**: add the 21 names to the imports and `__all__`, guarded by the bijection meta-test proposed in F10.

### F208. [Low] `_PARSE_CACHE` is an unbounded process-global that is never cleared — src/pyutilz/dev/code_audit/_base.py:18
- **Disposition**: OPEN
- **Category**: resource-growth
- **Problem**: keyed on `(path, mtime_ns, size)` with no eviction and no public `clear()`; every distinct version of every file ever parsed is retained as a full `ast.Module`.
- **Failure scenario**: a long-lived process that scans repeatedly — a watch loop, or a test session rewriting tmp fixtures between assertions, the exact scenario the key design at :12-17 calls out — accumulates one retained AST per edit, none ever released.
- **Suggested fix**: cap the cache (an `OrderedDict` LRU) or expose a `clear_parse_cache()` that `run_all` calls when it owns the process.

### F209. [Low] `_is_excluded` calls `Path.resolve()` on both operands for every candidate file — src/pyutilz/dev/code_audit/_base.py:73
- **Disposition**: OPEN
- **Category**: efficiency
- **Problem**: `path.resolve().relative_to(root.resolve())` re-resolves the constant root once per file, and `resolve()` is syscall-bound on Windows. `_iter_py_files` calls it for every `root.rglob("*")` entry.
- **Failure scenario**: on a 1500-file tree with 74 default scanners this is roughly 111k redundant `root.resolve()` calls per process.
- **Suggested fix**: resolve `root` once in `_iter_py_files` and pass it down.

### F210. [Low] The `Finding` severity contract is documented in a docstring but enforced nowhere — src/pyutilz/dev/code_audit/_base.py:31
- **Disposition**: OPEN
- **Category**: unenforced-invariant
- **Problem**: `Finding` is a frozen dataclass with `severity: str` and a docstring stating the P0/P1/P2/Low convention, but there is no `__post_init__` validation and no meta-test over the registry's output. F09 is the direct consequence: one stray literal in one scanner made that scanner's entire output invisible and nothing failed.
- **Failure scenario**: reproduced in F09 — `severity="Medium"` constructs successfully, sorts to position 99, and is filtered out of the CLI at every setting.
- **Suggested fix**: add `__post_init__` raising `ValueError` outside the four allowed values, plus a meta-test running every registered scanner over a fixture corpus and asserting each emitted severity is in the set.

### F211. [Low] `column_no_write_path`'s header and docstring state the inverted rule — src/pyutilz/dev/code_audit/column_no_write_path.py:27
- **Disposition**: OPEN
- **Category**: doc-behaviour-mismatch
- **Problem**: the header at :27 and the docstring at :98 list "a column named anywhere in a SELECT" as *excluded*, while :134 reads `if column in written or column not in read_names: continue` — being read is what makes a column reportable. Both directions were verified; the code and the `detail` string are right, both docstrings are wrong.
- **Failure scenario**: a reader triaging a finding concludes the scanner has a bug and dismisses a true positive.
- **Suggested fix**: rewrite both docstrings to match the implemented rule.

### F212. [Low] `column_no_write_path` never matches single-line `CREATE TABLE … );` — src/pyutilz/dev/code_audit/column_no_write_path.py:40
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: `_CREATE_TABLE` requires a literal newline before the closing paren.
- **Failure scenario**: single-line DDL → no findings; the multi-line spelling of the same table fires.
- **Suggested fix**: make the newline optional in the pattern.

### F213. [Low] `import_cycles` fabricates a cycle from `from . import X` — src/pyutilz/dev/code_audit/import_cycles.py:45
- **Disposition**: OPEN
- **Category**: false-positive
- **Problem**: with `node.module is None` the edge points at the base package rather than at each imported submodule.
- **Failure scenario**: `pkg/__init__.py: from pkg.mod import a`, `pkg/mod.py: from . import other`, `pkg/other.py: b = 2` → a P1 `pkg -> pkg.mod -> pkg` import cycle that does not exist.
- **Suggested fix**: emit one edge per alias (`base + "." + alias.name`); tracked together with F11.

### F214. [Low] `mojibake` reads files with `errors` unset, so an undecodable file is skipped rather than reported — src/pyutilz/dev/code_audit/mojibake.py:62
- **Disposition**: OPEN
- **Category**: silent-skip
- **Problem**: the read is `encoding="utf-8"` with the `UnicodeDecodeError` caught and the file dropped — but a file that is not valid UTF-8 is the strongest possible mojibake signal.
- **Failure scenario**: a cp1251-saved source file, the exact artefact the scanner exists to find, is silently skipped.
- **Suggested fix**: report an undecodable file as its own finding.

### F215. [Low] `_module_sql_constants` ignores constants bound inside a class or function body — src/pyutilz/dev/code_audit/_base.py:162
- **Disposition**: OPEN
- **Category**: false-negative
- **Problem**: only `tree.body` is scanned.
- **Failure scenario**: a repository that keeps its SQL in a `class Queries:` body gets no SQL findings from any of the scanners that resolve constants through this helper.
- **Suggested fix**: also collect class-body assignments, qualified by class name.

### F216. [Low] `run_all`'s parallel path silently drops the per-scanner failure boundary — src/pyutilz/dev/code_audit/registry.py:351
- **Disposition**: OPEN
- **Category**: error-handling
- **Problem**: `pool.map` re-raises the first worker exception, aborting the whole run; the sequential path at :354-355 has the same all-or-nothing behaviour, so one scanner raising on one pathological file loses all 88 others' findings.
- **Failure scenario**: reproduced as the visible symptom of F04 — a single `KeyError` in one worker discarded the entire run's output.
- **Suggested fix**: wrap each scanner call in `try/except Exception`, log at WARNING naming the scanner, and continue.
