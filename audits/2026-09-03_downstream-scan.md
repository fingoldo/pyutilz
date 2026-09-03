# Downstream scan: py-ci-shared and mlframe

Date: 2026-09-03. Scanner suite: `pyutilz.dev.code_audit` (95 registered checks; default set = every non-OPT-IN check).
First run of either repo after the two scanner-repair waves.

## What was run

```python
from pathlib import Path
from pyutilz.dev.code_audit import run_all
run_all(Path(<tree>), parallel=True)   # under an if __name__ == "__main__" guard
```

Four trees, default check selection (OPT-IN checks excluded), default exclude dirs:

| Tree | .py files | Findings | Wall |
|---|---|---|---|
| `py-ci-shared/src` | 43 | 26 | 5s |
| `py-ci-shared/tests` | 36 | 73 | 5s |
| `mlframe/src` | 1728 | 2885 | 225s |
| `mlframe/tests` | 2580 | 5591 | 318s |
| **total** | **4387** | **8575** | |

Python: `D:/ProgramData/anaconda3/python.exe` (3.11). No 3.8 leg is available on this box, so the
five "blind on 3.8" repairs are not exercised by this run; the `redundant_test_fit_call` 3.8
`ast.dump` fallback was read but never executed.

## Counts by check and severity

### py-ci-shared/src (43 files) -- 26 findings

| Sev | Check | N |
|---|---|---|
| Low | default_via_or | 7 |
| P2 | default_via_or | 4 |
| P2 | duplicate_function_body | 3 |
| P1 | getattr_unknown_attribute | 3 |
| Low | assert_in_loop_first_failure_only | 2 |
| Low | docstring_args_incomplete | 2 |
| P1 | non_neutral_except_fallback | 2 |
| P2 | duplicate_credential_regex | 1 |
| Low | duplicate_function_body_subset | 1 |
| Low | todo_hygiene | 1 |

### py-ci-shared/tests (36 files) -- 73 findings

| Sev | Check | N |
|---|---|---|
| Low | redundant_test_fit_call | 48 |
| P2 | duplicate_function_body | 6 |
| Low | near_duplicate_function_body | 6 |
| P2 | hardcoded_absolute_path_in_test | 4 |
| P1 | nondiscriminating_test | 4 |
| P2 | source_text_assertion | 2 |
| P2 | tautological_is_not_none_only_test | 2 |
| Low | comment_names_missing_symbol | 1 |

### mlframe/src (1728 files) -- 2885 findings

| Sev | Check | N |
|---|---|---|
| P1 | broad_except_swallow | 580 |
| P1 | non_neutral_except_fallback | 446 |
| P2 | duplicate_function_body | 384 |
| Low | possibly_dead_import | 324 |
| Low | default_via_or | 304 |
| P2 | unthrottled_hot_loop_log | 240 |
| P1 | additive_epsilon_denominator | 165 |
| P1 | getattr_unknown_attribute | 125 |
| P2 | default_via_or | 79 |
| P2 | mojibake | 57 |
| Low | docstring_args_incomplete | 34 |
| P1 | default_via_or | 23 |
| P2 | resource_handle_safety | 23 |
| P2 | return_annotation_mismatch | 16 |
| Low | unreachable_import_fallback | 14 |
| P1 | import_cycle | 12 |
| P0 | parameter_aliasing_mutation | 8 |
| P2 | unpicklable_resource_state | 8 |
| P2 | effect_flag_outside_its_effect | 7 |
| Low | near_duplicate_function_body | 7 |
| P1 | locals_get_fragile_lookup | 6 |
| Low | locals_globals_as_output | 4 |
| P2 | log_only_except | 3 |
| P1 | mutation_during_iteration | 3 |
| P1 | sentinel_cached_as_answer | 3 |
| Low | comment_names_missing_symbol | 2 |
| P1 | sentinel_guard_mismatch | 2 |
| Low | assert_in_loop_first_failure_only | 1 |
| Low | constructor_param_overwritten | 1 |
| P2 | credential_shaped_log_arg | 1 |
| P1 | getattr_literal_on_known_dataclass | 1 |
| P2 | readonly_to_numpy_mutation | 1 |
| Low | redundant_test_fit_call | 1 |

### mlframe/tests (2580 files) -- 5591 findings

| Sev | Check | N |
|---|---|---|
| Low | redundant_test_fit_call | 2422 |
| P1 | nondiscriminating_test | 1030 |
| P2 | duplicate_function_body | 525 |
| Low | possibly_dead_import | 462 |
| Low | assert_in_loop_first_failure_only | 185 |
| P1 | getattr_unknown_attribute | 175 |
| P2 | tautological_is_not_none_only_test | 163 |
| P1 | broad_except_swallow | 145 |
| P1 | except_skip_masks_call_under_test | 74 |
| P1 | non_neutral_except_fallback | 66 |
| P2 | source_text_assertion | 53 |
| P2 | resource_handle_safety | 48 |
| P1 | additive_epsilon_denominator | 45 |
| Low | default_via_or | 44 |
| Low | duplicate_function_body_subset | 40 |
| P2 | default_via_or | 39 |
| P2 | vacuous_assertion | 19 |
| P2 | hardcoded_absolute_path_in_test | 13 |
| Low | near_duplicate_function_body | 12 |
| P1 | default_via_or | 9 |
| Low | comment_names_missing_symbol | 8 |
| P1 | bare_except | 4 |
| P0 | parameter_aliasing_mutation | 3 |
| Low | unreachable_import_fallback | 3 |
| P2 | test_asserts_against_production_constant | 2 |
| P2 | effect_flag_outside_its_effect | 1 |
| P2 | return_annotation_mismatch | 1 |

## Triage: py-ci-shared

99 findings, every one triaged individually (no sampling).

### Real defects

| Where | Check | Why it matters |
|---|---|---|
| `src/py_ci_shared/loc_budget.py:85` | non_neutral_except_fallback (P1) | `_loc()` returns `0` on `OSError`. The gate is `if _loc(p) > limit`, so any file the walker cannot open is silently scored as 0 LOC and exempted from the LOC budget. A gate that fails open with no log line is exactly what this check exists for. |
| `src/py_ci_shared/loc_budget.py:121` | found by hand while verifying the row above | `{... : _loc(p) for p in files if _loc(p) > limit}` calls `_loc` twice per file, so every production file is read line-by-line twice on every test run. |
| `src/py_ci_shared/version_tag_currency.py:45` | non_neutral_except_fallback (P1) | `_git()` returns `""` both for "git said nothing" and for "git is missing / not a repo". The currency gate then reports "no tags" and passes. Same fail-open shape, smaller blast radius. |
| `src/py_ci_shared/edge_function_hygiene.py:60,74` vs `src/py_ci_shared/dart_scanners.py:57,72` | duplicate_function_body (P2) | `_line_of` and `_balanced` / `_balanced_body` are byte-identical brace-matching and line-counting primitives living in two modules. Every Dart/TS scanner in those files is built on them; a fix to one silently misses the other. |
| `src/py_ci_shared/loc_budget.py:75` vs `src/py_ci_shared/content_hash_version_bump_gate.py:77` | duplicate_function_body (P2) | `_refresh_requested()` duplicated verbatim. They read different `REFRESH_FLAG` constants, so the bodies converge legitimately, but "how do we detect a refresh request" now has two homes. |
| `src/py_ci_shared/gpu_timing_sync.py:229` vs `:160` | duplicate_function_body_subset (Low) | `_own_stmt_blocks` contains 99% of `_iter_stmt_blocks` inline. Real inlining, worth collapsing. |
| `src/py_ci_shared/docs_inventory_parity.py:141,213` | docstring_args_incomplete (Low) | Two `Args:` sections omitted parameters. **Fixed** - see below. |
| `tests/test_code_audit_meta.py:165`, `tests/test_content_hash_version_bump_gate.py:168`, `tests/test_loc_budget.py:152`, `tests/test_readme_env_var_parity.py:253` | nondiscriminating_test (P1) | Four `test_double_registration_is_a_noop*` tests contain no assertion at all. They stay green whatever the second registration does, including starting to raise a different exception or ceasing to be a no-op. Cheap fix: assert the registry length is unchanged. |

### Reviewed false positives

| Check | N | FP | Why it misfired |
|---|---|---|---|
| `redundant_test_fit_call` | 48 | 40 (83%) | The scanner keys on the unparsed call signature, so `_write(tmp_path, "a.py", ...)` in two tests looks like one repeated deterministic call. `tmp_path` is a function-scoped pytest fixture - a different directory in every test. Nothing is recomputed, and the recommended remedy (`@cache`, a shared fixture) would make the tests share one directory. `_call_signature()` has no fixture-name exclusion, and `_is_literal_data_factory()` does not exempt write-to-disk helpers. |
| `getattr_unknown_attribute` | 3 | 3 (100%) | All three receivers are objects this tree does not define: pyutilz's `Finding` (`getattr(f, "snippet", "")`), stdlib `ast` nodes (`getattr(s, "end_lineno", None)`), and a module (`getattr(sys, "stdlib_module_names", ())`, a deliberate 3.9 compat shim). The premise "an attribute of no class in this tree" only holds in a tree that owns every receiver. |
| `hardcoded_absolute_path_in_test` | 4 | 4 (100%) | `"C:\\Users\\Admin\\repo\\lib\\a.dart#0"` and `"/home/ci/repo/lib/a.dart"` are test inputs to the scanner that detects absolute paths, not paths anything opens. The rule flags every absolute-path string constant with no dataflow check that the literal reaches a filesystem call or a skip gate. |
| `default_via_or` | 11 | 10 (91%) | Fires on plain boolean disjunction (`return is_environ_get or is_getenv`, `any((root / target).exists() or any(root.glob(target)) ...)`), on regex-alternation idiom (`m.group("bc") or m.group("test") or m.group("arith")`), and on the universal `alias.asname or alias.name` import idiom. Only `baseline_trend.py:138` (`args.directories or list(DEFAULT_DIRECTORIES)`) is a genuine "confirm the semantics" case, and there argparse never produces an empty list. |
| `tautological_is_not_none_only_test` | 2 | 2 (100%) | `check_mypy_output(...)` returns an error message or `None`, so `is not None` *is* its pass/fail contract and `assert check_mypy_output("", 0) is not None` is the discriminating assertion. The rule treats `is not None` as vacuous without asking whether the callee's return contract is Optional-as-verdict. |
| `source_text_assertion` | 2 | 2 (100%) | `test_loc_budget.py:137` asserts on a JSON baseline the code under test wrote; `test_safe_precommit.py:93` asserts on a `.py` file the hook under test rewrote. Both are output assertions. The rule triggers on "reads a `.py`/`.sql` file" without asking who produced it. |
| `duplicate_credential_regex` | 1 | 1 | `edge_function_hygiene.py:51` is a detector pattern in a scanner repo, not a scrubber that drifted. Not applicable to this repo class. |
| `todo_hygiene` | 1 | 1 | `ci_workflow_gate.py:54` - the word TODO appears inside a backticked example of the YAML the regex has to match. Comment prose and real markers are not distinguished. |
| `assert_in_loop_first_failure_only` | 2 | 2 | Both are internal invariants over pydantic `model_fields`; failing fast on the first malformed schema field is the intent, not a batching bug. |
| `comment_names_missing_symbol` | 1 | 1 | `_gpu_sync()` in a test docstring names the shape a consuming project should adopt, not a symbol of this tree. |
| `duplicate_function_body` (test helpers) | 6 | 4 | Four are per-file `_write(tmp_path, name, text)` fixtures - normal, and unifying them would couple unrelated test modules. The other two are the assertion-free `test_double_registration...` bodies, i.e. the real defect already listed above. |
| `near_duplicate_function_body` | 6 | 6 | All parallel test bodies differing by one literal. `Low` severity is doing its job, but nothing here is actionable. |

Reviewed-false rate across py-ci-shared: 74 of 99.

### Changes made in py-ci-shared

One file edited, `src/py_ci_shared/docs_inventory_parity.py` (CRLF preserved; file re-parsed clean afterwards):

- `find_aggregate_group_drift`, `Args:` block at line 150 - added `pyproject_path` and `doc_path`.
- `find_phantom_doc_paths`, `Args:` block at line ~232 - added `doc_paths`, `repo_root` and `ignore`.

Nothing else was touched. No git operation was run in any repo.

## Triage: mlframe (report only, nothing modified)

8476 findings over 4308 files. Every low-volume check read in full; the high-volume checks
sampled 5-10 findings each and, where the misfire was mechanical, measured over the whole
population programmatically.

### Real defects worth acting on

| Where | Check | Why it matters |
|---|---|---|
| `mlframe/feature_engineering/_numerical_numba.py` (11 lines, e.g. :70, :73, :76); `mlframe/training/targets/target_temporal_audit.py` (24); `mlframe/training/targets/_target_temporal_changepoint.py` (18); `mlframe/feature_selection/filters/cat_interactions.py:133`; `mlframe/models/ensembling/process_method.py:446`; `mlframe/training/core/_phase_helpers_fit_pipeline.py:417`; `mlframe/training/neural/_recurrent_torch_model.py:607` | mojibake (P2) x57 | Verified by re-running the predicate against the real bytes: the flagged runs are `U+0420 U+00A4 U+0421 U+0453 ...` ("R-junk", UTF-8 read as CP1251) and `U+0432 U+0402 U+201D` (a corrupted em dash). Real, already-committed encoding damage in 7 files, sitting alongside correctly-encoded em dashes in the same files. 57/57 true positives. |
| `mlframe/feature_selection/shap_proxied_fs/_shap_proxy_prefilter_univariate.py:95` | sentinel_guard_mismatch (P1) | `if ram is not None:` guards a value from `_available_ram_bytes`, which returns `-1` from its except handler (`filters/_fe_cpu_batch.py:28`) and never returns `None`. The failure path reads downstream as a live RAM figure of -1 bytes. |
| `mlframe/training/composite/hpo.py:385` | sentinel_guard_mismatch (P1) | `if inner_spaces is None:` cannot see the empty-container fallback `_default_inner_spaces` returns at `hpo.py:194`. HPO then searches nothing instead of taking the default-space branch. |
| `mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py:6438`; `mlframe/training/core/_phase_train_one_target_polars_fastpath.py:187`; `mlframe/training/reporting/_reporting.py:67` | sentinel_cached_as_answer (P1) x3 | A failed lookup inside an except handler is written into the cache (`{}` / `None`). One transient error pins that key for the life of the process and is served to every later caller as the answer. |
| `mlframe/feature_selection/shap_proxied_fs/_shap_proxy_preflight.py:119` | readonly_to_numpy_mutation (P2) | `np.fill_diagonal(C, 0.0)` on an uncopied `.to_numpy()`. Under pandas Copy-on-Write this raises `ValueError: underlying array is read-only`. |
| `mlframe/training/core/_phase_finalize.py:898` | getattr_literal_on_known_dataclass (P1) | `getattr(ctx, "configs", None)` where `ctx` is a `TrainingContext` this tree defines and which has no `configs` field. Always the default, i.e. a config root that is silently always absent. |
| `mlframe/training/phases.py:40`; `training/feature_handling/cache_backend.py:103`; `training/composite/cache.py:639`; `training/suite_artefact_cache.py:191`; `training/feature_handling/cache.py:132`; `feature_selection/shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_loss.py:187` | unpicklable_resource_state (P2) x6 | Six cache/registry classes hold a `threading.Lock`/`RLock` with no `__getstate__`. Any of them crossing a joblib/loky boundary raises at pickle time. (The two `bench_fe_peak_memory.py` hits are a benchmark-local poller, not shipped.) |
| `mlframe/feature_selection/wrappers/_knockoffs.py:121`; `training/core/_phase_train_one_target_body.py:942,946`; `training/honest_diagnostics.py:202,210,211` | locals_get_fragile_lookup (P1) x6 | Six sites read conditionally-bound names out of `locals()`/`globals()`. A rename or an extract-helper refactor turns each into "always the default" with no error. `honest_diagnostics.py:202` is the sharpest - the resampler factory silently disappears. |
| 12 cycles, e.g. `feature_selection/filters/_gpu_resident_basis.py` (6 modules), `feature_selection/filters/_feature_engineering_pairs/` (5), `training/reporting/_reporting.py` (5), `training/targets/_target_distribution_analyzer.py` (4) | import_cycle (P1) x12 | All are the monolith-split-via-re-export shape; they resolve today only because of import order. Worth knowing about, low urgency. |
| `mlframe/feature_engineering/spatial.py:288`; `feature_selection/filters/_internals.py:240,242` | mutation_during_iteration (P1) x3 | Assignment into the dict being iterated. Safe today (existing keys only), one added key away from `RuntimeError`. Both sites deserve an explicit `for k in list(d)`. |
| `mlframe/training/composite/discovery/__init__.py:283` | constructor_param_overwritten (Low) | `self.config` is reassigned by `fit_with_stability_check`, so the constructor argument is advisory. A test that sets it in the constructor is testing nothing. |
| `tests/feature_selection/biz_val/test_biz_value_fe_rejection_ledger.py:349`; `tests/feature_selection/contracts/test_evaluation.py:290` | test_asserts_against_production_constant (P2) x2 | `assert len(records) == FE_REJECTION_LEDGER_CAP + 1` - the expectation moves with the constant it is supposed to pin. |
| 761 of the 1030 `nondiscriminating_test` hits | nondiscriminating_test (P1) | The no-assert subclass is a real, large problem here: `tests/feature_engineering/transformer/test_biz_val_real_datasets.py:5673` (`test_iter61_mammography`) and roughly 700 siblings run a full fit and assert nothing. Also real: 161 "skips instead of failing", 82 `except AssertionError: pass`, 23 imperative `pytest.xfail`. Only about 9 are pytest-benchmark tests where the `benchmark` fixture is the measurement. A genuine suite-wide finding, not scanner noise. |
| 2378 of the 2422 `redundant_test_fit_call` hits | redundant_test_fit_call (Low) | Unlike py-ci-shared, only 44 of 2422 (1.8%) involve a pytest fixture argument. Samples such as `_make_clf_data(n=300, f=8)` re-run by 3 tests and `_make_mtr_data()` by 4 are exactly the documented 7-14x wall-clock waste. Real, and the largest single lever on suite runtime. |

Individually-actionable defects outside the two suite-wide classes: about 40 of 8476.

### High-volume checks judged mostly noise on mlframe

| Check | N (src+tests) | Sampled FP | Why it misfired |
|---|---|---|---|
| `getattr_unknown_attribute` | 300 | 10/10 | Same root cause as in py-ci-shared, at scale. Receivers are third-party objects (`_cusolver.CUSOLVERError`, a polars `dtype.time_unit`, sklearn's `self._iso.X_max_`), foreign model objects tagged by mlframe (`model._mlframe_nan_scaler`), and config attributes injected by `setattr` (`self.fe_additive_fusion_ols_r_margin_sd`, `self.stability_n_jobs`). The rule needs the receiver's class to be defined in the tree - which is precisely what its sibling `getattr_literal_on_known_dataclass` already requires, and that sibling produced its single finding with no false positive. |
| `parameter_aliasing_mutation` | 11 (P0) | 11/11 | The suite's only P0 hits, and none survive triage. Three are `i = l` / `i = path_index` followed by `i += 1` on an integer parameter inside an `@njit` kernel (`core/arrays.py:119,124`, `_shap_proxy_treeshap.py:258`) - `+=` on a Name rebinds, it does not mutate. Four are `final_transformed_vals = final_transformed_vals_shared` (`_pairs_score.py:477,483,654,668`), a deliberately reused preallocated buffer whose columns are fully overwritten; the aliasing is the point. One (`integrations/mlflow.py:107`) aliases in the `if run_tags is None:` arm and mutates in the `else:` arm - mutually exclusive branches the scanner does not separate. The three test hits are the same preallocated-`out`-buffer shape. |
| `additive_epsilon_denominator` | 210 | 8/8 not actionable at P1 | Every hit is the `x / (y + 1e-12)` idiom, standard in numerical code. One sampled hit (`_binned_numeric_agg_fe.py:125`) is already wrapped in `np.where(std > 1e-9, ...)`, i.e. genuinely guarded and still flagged. The underlying observation is legitimate, but as an unconditional P1 across an ML library it is a policy opinion, not a defect report. |
| `non_neutral_except_fallback` | 512 | 4/6 | The optional-dependency probe (`except Exception: _HAS_NUMBA = False`, `_CUDA_AVAIL = False`) is the canonical correct spelling of that pattern and accounts for a large share of hits. The remainder (a numeric guard returning `False`, `variant = 'Categorical'`) are worth reading. On py-ci-shared, which has no optional-dep probes, both hits were real - so this check's precision is a function of repo shape. |
| `broad_except_swallow` | 725 | mixed | Pattern-true by construction; several sampled sites already carry `# noqa: BLE001`, i.e. a reviewer has been there. At 725 findings it is a codebase-wide policy readout, not a triage queue. |
| `unthrottled_hot_loop_log` | 240 | mostly per-target/per-fold loops of small N | Loop depth alone does not distinguish a 5-iteration outer loop from a per-row inner one. |
| `possibly_dead_import` | 786 | unverified per site | `import gc` / `import pandas as pd` bound and never referenced. Plausible, but this class also covers imports kept for side effects or for `TYPE_CHECKING`-adjacent reasons; not verified here. |
| `duplicate_function_body` | 909 | mixed | Some are real (`_occupied_k` across two benchmark files, `param_groups` across two optimizer wrappers); many are protocol-shaped setters and one-line accessors. |
| `tautological_is_not_none_only_test` | 163 | see the py-ci-shared row | Same Optional-as-verdict blind spot; the mlframe share was not measured site by site. |
| `source_text_assertion` | 53 | see the py-ci-shared row | Same "who wrote the file being read" blind spot. |
| `hardcoded_absolute_path_in_test` | 13 | see the py-ci-shared row | Same no-dataflow blind spot. |

## Verdict on the scanners

### Ready to point at another repo as-is

- `mojibake` - 57/57 verified true on mlframe, zero findings on the clean repo. Highest-precision check in the suite; the `_MIN_RUN_LENGTH = 3` repair is holding.
- `unpicklable_resource_state` - 8/8 real, exactly the pickle-breaking shape, no misfires.
- `sentinel_guard_mismatch` and `sentinel_cached_as_answer` - 5 findings, 5 real, each a genuine silent-failure bug with a cross-file proof in the detail text. Best signal per finding in the whole run.
- `getattr_literal_on_known_dataclass` - 1 finding, real. Its "receiver must be a class this tree defines" precondition is exactly what its noisy sibling lacks.
- `import_cycle`, `mutation_during_iteration`, `locals_get_fragile_lookup`, `readonly_to_numpy_mutation`, `constructor_param_overwritten`, `test_asserts_against_production_constant` - low volume, high hit rate, all worth reading in full.
- `nondiscriminating_test` - 1034 findings but a real 74% no-assert core and roughly 1% benchmark-fixture noise. Earned its keep loudly.
- `duplicate_function_body` - earned its keep on the small repo (found the `_line_of` / `_balanced` / `_refresh_requested` triples immediately); on a 1700-file tree it needs a body-size floor to stay readable.
- `non_neutral_except_fallback` - both py-ci-shared hits were real fail-open gates. Add an optional-dependency-probe exemption before pointing it at a repo with GPU/numba fallbacks.

### Need work before the next repo

1. **`getattr_unknown_attribute` - 303 findings, 13 of 13 sampled false, no true positive observed in either repo.** The premise ("an attribute of no class in this tree") cannot hold for code touching stdlib, third-party or dynamically-configured objects, which is all real code. It should require what `getattr_literal_on_known_dataclass` requires: a receiver whose class is defined in the scanned tree. As shipped it is the largest noise source among the P1s and should probably be OPT-IN until fixed.
2. **`parameter_aliasing_mutation` - 11 of 11 false, and they are the suite's only P0s.** Three separate gaps: `x += 1` on a scalar counted as in-place mutation; no branch-exclusivity analysis; no notion of an intentionally shared preallocated buffer. A P0 with a 100% false-positive rate is worse than no check, because it burns the severity.
3. **`redundant_test_fit_call` - 83% false on a fixture-heavy repo (40 of 48), 1.8% false on mlframe (44 of 2422).** One fix covers it: exclude calls whose arguments name a function-scoped pytest fixture (`tmp_path`, `tmp_path_factory`, `tmpdir`, `monkeypatch`, `capsys`, `caplog`, `request`). Cheap, and it turns the check from unusable to excellent on small repos.
4. **`default_via_or` - 10 of 11 false on py-ci-shared, 498 findings on mlframe.** It currently fires on any `BoolOp(Or)`. It needs to exclude pure-boolean operands, regex `.group()` alternation, and the `alias.asname or alias.name` family, or be demoted to OPT-IN.
5. **`hardcoded_absolute_path_in_test`, `source_text_assertion`, `tautological_is_not_none_only_test`** - all three flag a syntactic shape with no check on what the value is for. Respectively: does the literal reach a filesystem call or skip gate; did the code under test write the file being read; is the callee's contract Optional-as-verdict. Each needs one dataflow question answered before emitting.
6. **`additive_epsilon_denominator` at P1** - 210 findings on a numerical library, all the same idiom, at least one already guarded. Either demote to Low/OPT-IN, or teach it to skip a division already inside an `np.where`/`if` guard on the same denominator.
7. **`todo_hygiene` and `comment_names_missing_symbol`** - both flagged prose that quotes an example rather than makes a claim. A backticked-example exclusion would clear both of their only findings here.
8. **`broad_except_swallow` (725) and `unthrottled_hot_loop_log` (240)** - not wrong, but unrankable at that volume. Both would benefit from suppressing sites already annotated (`# noqa: BLE001`, an explicit comment) and, for the loop one, from a bound on the loop's plausible iteration count.

### Cross-cutting observation

The clean repo's 99 findings split roughly 25 real to 74 reviewed-false. mlframe's 8476 reduce to
about 40 individually-actionable defects plus two genuinely suite-wide classes (57 corrupted
lines, roughly 1000 non-discriminating tests, roughly 2400 redundant fixture calls). The checks
that survived triage in both repos are the ones asserting a cross-file fact - "this sentinel
cannot reach that guard", "this class is defined here and lacks that field", "these bytes
round-trip". The ones that failed match a local syntactic shape and then guess at intent. That is
the line to hold when adding check number 96.
