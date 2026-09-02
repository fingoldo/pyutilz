# Performance Audit — pyutilz (2026-09-02)

## Summary

Scope: `src/pyutilz/performance/kernel_tuning/**` (registry, benchmark, cache_base, cache_class, region_matching), `src/pyutilz/system/gpu_dispatch.py` + its probing backends in `src/pyutilz/system/system/probing.py`, plus a sweep of hot paths elsewhere in the library (`core/pythonlib.py`, `core/disk_cache.py`, `data/numpylib.py`, `data/pandaslib/frames.py`, `text/strings/jsonutils.py`, `text/tokenizers.py`, `text/similarity.py`).

Every finding below is **measured** with `D:/ProgramData/anaconda3/python.exe`, `time.perf_counter`, 2-3 repeats of N-iteration loops; scratch scripts live under `D:\Temp\perfaudit\` (none in the repo). No repo file was modified.

**Machine caveats (important):** this is a Windows 10 box that is NOT quiet — timings jitter by 20-50% between repeats (visible in the numbers below, e.g. `is_cuda_available` at 16.5 / 17.7 / 31.5 us across three identical loops). Ratios inside a single interleaved A/B loop are trustworthy; absolute milliseconds are not. A CUDA GPU is present and `is_cuda_available()` returns True, so the GPU findings are measured on real hardware rather than reasoned about. Where a repeat range is wide the whole range is quoted, not the best number.

**Not re-raised (verified fixed since 2026-07-21):** the HIGH `O(word_len^3)` morpheme loop in `text/tokenizers.py` now has a `MAX_WORD_LENGTH` skip guard (`tokenizers.py:146-153`); both MEDIUM redundant-full-column-rescan findings in `data/pandaslib/frames.py` (pandas and polars branches) are fixed and carry explanatory comments. No `bench-attempt-rejected` markers exist anywhere in the repo (grepped), so nothing below re-proposes an already-measured-and-rejected change.

Counts by severity: **1 Critical, 3 High, 3 Medium, 4 Low** (11 total).

## Findings

### F01. [Critical] `time_backend` times CUDA kernels without a device synchronize — the kernel-tuning cache can be populated with phantom wins — src/pyutilz/performance/kernel_tuning/benchmark.py:199-202

- **Disposition**: OPEN
- **Category**: gpu-timing-correctness
- **Problem**: `_run` does `t0 = timer(); fn(*args); local.append(timer() - t0)` with no GPU synchronize anywhere in the module. Grepping the whole package, `synchronize` appears **only** in `src/pyutilz/dev/benchmarking.py` (`synchronize_gpu_if_available`, default `synchronize_gpu=True`, whose own docstring at line 37 spells out the exact hazard: "the variant that didn't synchronize actually LOST"). The kernel-tuning module — the one whose numbers are persisted into the on-disk tuning cache as `wall_ms` and used to pick the production backend — has no equivalent. CUDA launches are asynchronous, so the timer stops at *launch*, not at completion.
  Measured on this box with a cupy 4000x4000 float32 matmul through the real `time_backend`:
  - `time_backend(gpu_matmul, ..., n_iters=10, warmup=2)` -> **0.0366 ms**
  - identical call where the timed fn ends in `cp.cuda.Stream.null.synchronize()` -> **69.42 ms**
  - understatement factor: **~1894x**.
  The warmup loop (`benchmark.py:182-183`) is also unsynchronized, so warmup can return before the first launch has even executed.
- **Failure scenario**: any project using `benchmark_backends` / `time_backend` to choose between a CPU and a GPU variant, or between two GPU variants, records a GPU time that is essentially "Python launch overhead" instead of wall time. The GPU backend wins every sweep unconditionally; the region is written into the immutable per-kernel tuning file with a bogus `wall_ms`, and every later `get_or_tune` on that host returns it from cache without ever re-measuring. A GPU variant that is genuinely 10x slower than the CPU one will be selected and stay selected. The recorded `max_abs_diff` equivalence gate (`_apply_equiv_gate`) does not catch this, because the numerics are fine — only the timing is wrong.
- **Suggested fix**: reuse the existing primitive rather than writing a second one — import `synchronize_gpu_if_available` from `pyutilz.dev.benchmarking` and call it (a) once after the warmup loop and (b) immediately before `local.append(timer() - t0)` in `_run`, behind a `synchronize_gpu: bool = True` kwarg on `time_backend` mirroring `dev/benchmarking.py`'s signature. Default ON. The sync must be inside the timed region's close, not after it. Under `concurrency > 1` a null-stream sync serializes threads — for the concurrent path the honest primitive is a per-thread `cp.cuda.Event` pair rather than a global stream sync.

### F02. [High] `gpu_capability_summary` / `occupancy_aware_block_size` re-shell-out to nvidia-smi on every call — 64 ms per dispatch decision — src/pyutilz/system/gpu_dispatch.py:538-601, 437 (via 593)

- **Disposition**: OPEN
- **Category**: missing-caching
- **Problem**: `gpu_capability_summary()` is uncached and calls `get_gpuutil_gpu_info(attrs="id,name,memoryFree,memoryTotal")` (`gpu_dispatch.py:593`), which goes through GPUtil to an `nvidia-smi` **subprocess**. `occupancy_aware_block_size()` calls `gpu_capability_summary` whenever `caps` is None (its default). Measured, device 0:
  - `gpu_capability_summary(0)`: **64.21 / 66.26 ms** per call
  - `occupancy_aware_block_size(16)`: **63.47 ms** per call
  - component split: `get_gpuutil_gpu_info(...)` **66.24 ms**; `get_gpu_cuda_capabilities(0)` **4.57 ms**; `query_cuda_device_attribute(...)` 0.021 ms
  - an `lru_cache(maxsize=8)` wrapper around the same function: **0.137 us/call** (~470,000x on repeat calls)
  `cache_base.py:69` already knows this and wraps it in `_gpu_summary_cached` for the kernel-tuning cache's own use — but the public `gpu_dispatch` API that every other caller reaches for has no such guard, so the memoization is a private detail of one consumer instead of a property of the function.
- **Failure scenario**: a dispatcher that asks for a block size per kernel launch (which is what `occupancy_aware_block_size` is named for) pays a 64 ms `nvidia-smi` subprocess spawn per launch. On a loop of 1,000 launches that is 64 seconds of pure process-spawn, dwarfing the kernels themselves. Worse, `nvidia-smi` under GPU contention can stall for seconds, so this also injects unbounded latency into a supposedly-pure sizing helper. Also relevant to `dev/benchmarking.py` sweeps that size blocks per candidate.
- **Suggested fix**: split static from live. The `cc_*`, `sm_count`, `max_threads_*`, `max_shared_mem_*`, `max_blocks_per_sm`, `warp_size`, `reserved_shared_mem_per_block` and `name` fields are hardware-invariant — put those behind `@lru_cache(maxsize=16)` keyed on `device_id` (matching `_gpu_summary_cached`'s existing per-device keying rationale) and have the existing `reset_cache()` (`gpu_dispatch.py:149`) clear it. `free_vram_gb` is genuinely live: either move it to a separate `free_vram_gb(device_id)` helper, source it from the far cheaper `_free_bytes_via_cupy` (`gpu_dispatch.py:237`, no subprocess) instead of GPUtil, or give it a short TTL. `occupancy_aware_block_size` should not need live VRAM at all, so it should call the static-caps path.

### F03. [High] `json_pg_dumps` does a full recursive Python rebuild plus a stdlib re-parse that orjson makes unnecessary — 25-32x slower than needed — src/pyutilz/text/strings/jsonutils.py:253-274

- **Disposition**: OPEN
- **Category**: redundant-serialization
- **Problem**: on the orjson path the function does three things it does not need to do:
  1. `_normalize_nonfinite_floats(obj)` (line 266) walks and **rebuilds every dict/list in the document** in pure Python to turn NaN/Inf into None. orjson already emits `null` for both natively — verified: `orjson.dumps({"x": nan, "y": inf})` -> `b'{"x":null,"y":null}'`. The docstring justifies the pass as making orjson and stdlib output identical, but it only needs to run on the **stdlib fallback** branch to achieve that.
  2. `json.loads(raw...)` (line 274) re-parses the just-serialized document with **stdlib** json, discarding the orjson win the comment on line 262 explicitly claims. Measured: `json.loads` 1.203-1.268 ms vs `orjson.loads` 0.554-0.887 ms on the same string (1.4-2.2x).
  3. the parsed dict is then handed to `psycopg2.extras.Json`, which serializes it a **third** time at adapt time. dumps -> loads -> dumps.
  Measured on a 500-row nested payload, interleaved A/B, 200 reps x3:
  - current `json_pg_dumps`: **7.27 / 7.70 / 9.01 ms**
  - drop the reparse only (`Json(None, dumps=lambda _: raw)`): 3.30 / 3.96 / 6.44 ms (**1.18-2.40x**)
  - drop the normalize pass too: **0.238 / 0.286 / 0.290 ms** (**25.4-32.4x**)
  - `_normalize_nonfinite_floats` alone: **2.98 ms** of the 7.4 ms
  Output equivalence checked in both variants: the adapted JSON text compares equal after `json.loads` on both sides, including with NaN values seeded into the payload.
  Two smaller costs in the same function: `from psycopg2.extras import Json` and `import orjson` execute on every call (~1 us each of `sys.modules` lookup plus frame work), and `Json(...)`'s default `dumps` is stdlib.
- **Failure scenario**: this is the jsonb insert path. A bulk load writing 10,000 rows pays roughly 74 s where roughly 3 s would do, and the pure-Python `_normalize_nonfinite_floats` rebuild also allocates a full second copy of every document — a real memory spike on large payloads. The 3 ms normalize pass scales linearly with document node count, so a wide feature-vector row is disproportionately hit.
- **Suggested fix**: hoist both imports to module scope with a `try/except ImportError` orjson probe resolved once. On the orjson branch, skip `_normalize_nonfinite_floats` entirely (orjson's null emission already satisfies the stated invariant) and skip the round-trip: `raw = orjson.dumps(obj, default=json_serial, option=opts).decode("utf-8").replace("\\u0000", "")`, then hand the finished text to psycopg2 as `Json(None, dumps=lambda _obj, _r=raw: _r)` so it is never re-serialized. Keep `_normalize_nonfinite_floats` on the stdlib fallback branch only, where it is genuinely load-bearing (stdlib emits the non-standard `NaN`/`Infinity` tokens postgres rejects). Add a test asserting the two branches produce byte-identical output for a NaN/Inf/NUL payload — that is the invariant the current unconditional pass was protecting.

### F04. [High] `is_cuda_available()` is uncached and re-probes numba on every call — `dispatch_cpu_vs_gpu` costs 18-24 us per decision — src/pyutilz/core/pythonlib.py:956-969; src/pyutilz/system/gpu_dispatch.py:605-641

- **Disposition**: OPEN
- **Category**: missing-caching
- **Problem**: `is_cuda_available()` calls `_ensure_cuda_home_from_pip()` then `numba.cuda.is_available()` on every invocation. `dispatch_cpu_vs_gpu` — documented as *the* CPU-vs-GPU dispatcher, i.e. a per-call decision helper — calls it unguarded at line 639. Measured (20,000-iteration loops, x3):
  - `is_cuda_available()`: **16.50 / 17.73 / 31.46 us** per call
  - `dispatch_cpu_vs_gpu(1_000_000)`: **18.01 / 24.19 / 19.74 us** per call
  - the same value behind `@lru_cache(maxsize=1)`: **0.105 / 0.108 / 0.109 us** per call (**~160-290x**)
  - component split: `numba.cuda.is_available()` alone **30.36 / 31.38 us**; `_ensure_cuda_home_from_pip()` **6.21 / 7.54 us** *even on its early-return path* with `CUDA_HOME` already set — that cost is the per-call `import os as _os` (line 940) plus two `os.environ.get`s, all re-executed to reach a `return` that does nothing.
  CUDA availability cannot change within a process (the module's own comment at `pythonlib.py:938` notes "numba caches it"), and `_ensure_cuda_home_from_pip` is explicitly a run-once-before-first-probe operation.
- **Failure scenario**: any hot loop that dispatches per chunk/batch/row-group pays 20 us per iteration to re-answer a question with a constant answer. At 100,000 dispatch decisions that is roughly 2 s of pure overhead, and for small work items (`n_work` just under `gpu_min_work`) the *decision* costs more than the work being dispatched. It also makes `dispatch_cpu_vs_gpu` unusable inside an inner loop, which is exactly where a dispatcher belongs.
- **Suggested fix**: `@lru_cache(maxsize=1)` on `is_cuda_available` (and on `_ensure_cuda_home_from_pip`, which is idempotent by construction), with the existing `gpu_dispatch.reset_cache()` clearing both so tests that mock CUDA availability still work — `reset_cache` at `gpu_dispatch.py:149` is already the established reset seam. Separately, hoist `import os as _os` in `_ensure_cuda_home_from_pip` to module scope; `os` is already imported there.

### F05. [Medium] `hash_array_summary` claims a "sub-O(N) summary" but runs three separate full-array numpy reductions; a fused njit pass is 27-33x faster — src/pyutilz/core/disk_cache.py:87-133

- **Disposition**: OPEN
- **Category**: njit-opportunity
- **Problem**: the docstring says "Stable content hash of an ndarray from a sub-O(N) summary", but lines 119-121 compute `arr.sum(axis=col_axis, dtype=np.float64)`, `arr.min(axis=col_axis)` and `arr.max(axis=col_axis)` as three independent full passes over the array (and along `axis=0` on a C-contiguous array, which is the strided/cache-hostile direction for numpy). Measured on a `(2_000_000, 4)` float64 C-contiguous array, interleaved A/B, 10 reps x3:
  - numpy 3-pass (`sum` + `min` + `max`, as written): **154.16 / 151.19 / 149.86 ms**
  - `@njit(cache=True, parallel=True)` single fused pass computing all three per column: **4.84 / 4.57 / 5.53 ms**
  - speedup: **27.1x / 33.1x / 31.9x**; results verified with `np.allclose` on all three outputs.
  Secondary: `arr = np.ascontiguousarray(arr)` at line 102 materializes a **full copy** of a non-contiguous input (e.g. a strided column view) before anything is hashed, even though only the head/tail row slices need contiguity and those are individually `np.ascontiguousarray`-wrapped again at lines 117-118. Measured on a non-contiguous `(2_000_000, 4)` view: 183-206 ms as written vs 172-175 ms without the up-front copy (roughly 1.15x, and one whole-array allocation avoided).
- **Failure scenario**: `hash_array_summary` is the cache-key function for `DiskCache` and is reached from `hash_object` for any ndarray nested in a params dict. Hashing a 64 MB array costs roughly 150 ms *per cache probe* — on a cache **hit**, where the entire point is to avoid recomputation, the key computation alone can exceed the cost of the work being cached. The non-contiguous copy additionally doubles peak memory at exactly the moment a large array is in flight.
- **Suggested fix**: move the three column reductions into one `@njit(cache=True, parallel=True)` kernel over a 2-D view (`arr.reshape(-1, arr.shape[-1])`), keeping the existing numpy code as the fallback when numba is unavailable or the dtype is unsupported — matching how `data/numbalib.py` already structures its njit kernels. Drop the up-front `np.ascontiguousarray(arr)` and rely on the per-slice wrapping already present at lines 117-118. **Both changes alter the digest** (fused sequential accumulation differs from numpy's pairwise summation in the last ulp; the contiguity change was also observed to shift the digest on a strided view), so bump a `_HASH_VERSION` constant into the hash prefix so existing on-disk entries miss cleanly rather than silently colliding — this is a cache key, not a persisted value, so a one-time full miss is the correct migration.

### F06. [Medium] `KernelTuningCache.lookup` re-derives its constraint-key set and re-builds three f-string keys per region per call — 3-5x slower than a precompiled plan — src/pyutilz/performance/kernel_tuning/cache/cache_class.py:587-611; src/pyutilz/performance/kernel_tuning/cache/region_matching.py:10-28

- **Disposition**: OPEN
- **Category**: per-call-recomputation
- **Problem**: on every `lookup()` the function rebuilds `constraint_keys = {f"{ax}{suf}" for ax in axes for suf in _AXIS_SUFFIXES}` (line 606), and `_region_matches` then formats **three more f-strings per axis per region** (`f"{axis_name}_max"`, `_min`, `_eq`) and does three `dict.get`s against a region dict whose contents never change between calls. Nothing here depends on `dims` except the comparisons themselves. Measured with a 3-region single-axis kernel, in-memory cache, 200,000-iteration interleaved A/B x3:
  - current `lookup`: **4.73 / 5.14 / 6.25 us** per call (a separate run: 4.81 / 4.87 / 6.08 us)
  - prototype with a per-kernel precompiled `[(constraints, stripped_payload)]` plan: **0.87 / 1.73 / 1.95 us**
  - speedup: **5.43x / 2.98x / 3.21x**; return values verified identical on both a first-region hit and a last-region hit
  - component costs inside the current call: `constraint_keys` rebuild **0.86 / 1.38 us** (roughly 20-25% of the call), `_ensure_loaded()`'s lock acquire plus dict walk **0.72 / 0.87 us** (roughly 15%)
  Related, same path: `get_or_tune` on a pure cache hit measures **5.61 / 6.05 / 7.07 us** against `lookup`'s 2.97-3.41 us in the same loop. The gap is (a) constructing the `_fb` closure (`cache_class.py:746`) on every call even though it is only consulted on a miss, and (b) `_code_version_stale` (line 789) doing a *second* `_ensure_loaded()` — a second lock acquire and dict walk for the entry `lookup` is about to fetch again.
- **Failure scenario**: `lookup`/`get_or_tune` is the per-launch dispatch decision the whole kernel-tuning subsystem exists to make cheap. At 6 us a hit, a kernel launched 1,000,000 times in a training loop spends 6 s just asking the cache which variant to use. The cost also scales with region count — a kernel tuned across 20 regions pays the f-string formatting 60 times per miss-to-last-region lookup.
- **Suggested fix**: compile the plan once per `(kernel_name, entry identity)` and memoize it (in a dict keyed by kernel name, invalidated in `update()`, `evict()` and `reset()`). Per region store a pre-split `[(axis_name, op_code, value)]` list plus the already-stripped payload dict, so a hit becomes a tuple walk and a dict return with zero string formatting and zero comprehension. Keep `_region_match_reason` (`region_matching.py:31`) on the current string-building path — it is only used by `lookup_explain`, a diagnostic, and is not hot. Separately in `get_or_tune`: make `_fb` a lazily-constructed local (or a bound method taking the needed args) so the hit path never builds the closure, and pass the already-loaded entry into `_code_version_stale` instead of having it re-enter `_ensure_loaded`. Any memoized plan must be excluded from `__getstate__` (`cache_class.py:135`) so the cache stays picklable.

### F07. [Medium] `get_topk_indices` has no `k == 1` fast path — argpartition plus argsort where argmax would do, 2.4-2.9x — src/pyutilz/data/numpylib.py:25-72

- **Disposition**: OPEN
- **Category**: missing-fast-path
- **Problem**: for `k == 1` the function still runs the general pipeline: a full `np.where(np.isnan(arr), ±inf, arr)` copy, `np.argpartition`, `np.take`, `np.take_along_axis`, `np.argsort`, `np.flip`, and a final `np.take_along_axis` — seven array ops to find one index. `np.nanargmax`/`np.nanargmin` compute exactly the same answer (NaN-safe by definition, which is what the `np.where` substitution is emulating). Measured, interleaved A/B:
  - `n = 1_000_000` float32: current **6.30 / 8.30 ms** vs `nanargmax` **2.16 / 3.49 ms** -> **2.9x / 2.4x**
  - `n = 10_000` float32: current 0.066 / 0.069 ms vs 0.268 / 0.033 ms — the first repeat is dominated by numpy warm-up noise; the second repeat gives 2.1x. The small-n number is unreliable on this box; the 1M result is the stable one.
  Separately, `np.where(np.isnan(arr), -np.inf, arr)` allocates and writes a **full copy of the array on every call for any floating dtype**, whether or not a NaN is present. Measured for `k = 5` at `n = 10_000`, with an `np.isnan(arr).any()` guard that skips the copy when clean: 0.243 / 0.191 ms current vs 0.071 / 0.118 ms guarded (**3.4x / 1.6x**); at `n = 1_000_000` the same guard gives 15.38 / 17.20 ms vs 14.30 / 13.62 ms (**1.08x / 1.26x**) — the win shrinks as the `isnan` scan itself becomes the cost, so this second change is clearly worth it for small-to-medium arrays and roughly neutral for very large ones.
- **Failure scenario**: `k=1` is the single most common top-k request (best model, best split, argmax of a score vector) and it is the case the current code handles worst. A selection loop calling this per candidate on 1M-element score arrays pays roughly 3x more than necessary; combined with the unconditional NaN copy it also allocates a full extra array per call, which on a 1M float32 vector is 4 MB of churn per call and shows up as allocator pressure rather than as a clean hotspot.
- **Suggested fix**: add an early `if k == 1` branch returning `np.expand_dims(np.nanargmax(arr, axis=axis), axis)` (or `nanargmin` when `highest=False`) for floating dtypes and `argmax`/`argmin` for integer ones, preserving the current output shape and `int64` dtype. Guard the NaN substitution with a cheap `np.isnan(arr).any()` so a clean array skips the copy. Both are behavior-preserving; keep the existing doctests plus a new one asserting `k=1` on an all-NaN input matches the current output.

### F08. [Low] `_build_provenance()` is recomputed once per kernel directory inside the cache load loop — src/pyutilz/performance/kernel_tuning/cache/cache_class.py:382, 291

- **Disposition**: OPEN
- **Category**: per-call-recomputation
- **Problem**: `_read_kernel_dir_by_path` (line 382) and `_read_kernel_newest` (line 291) each call `live_prov = _build_provenance()` on entry. `_load` invokes `_read_kernel_dir_by_path` once per kernel directory, so a host cache holding N kernels rebuilds the identical provenance snapshot N times. Measured: `_build_provenance()` = **9.9 / 9.4 / 7.6 us** per call (its component `_safe_version` = 0.9 us). The result depends only on installed package versions and the CUDA driver — invariant for the process lifetime, and the GPU part of it is *already* memoized via `_gpu_summary_cached`.
- **Failure scenario**: minor but pure waste — a 100-kernel cache pays roughly 1 ms of redundant version probing at process startup; a 1,000-kernel cache roughly 10 ms. Small in absolute terms, which is why this is Low, but it is on the cold-start path of every process that touches the tuning cache.
- **Suggested fix**: wrap `_build_provenance` in `@lru_cache(maxsize=1)` (returning a copy, or freezing the dict, so a caller cannot mutate the shared snapshot), or hoist the call to `_load` and thread `live_prov` in as a parameter. Add a `cache_clear()` to whatever reset seam the kernel-tuning tests already use for `hw_fingerprint` / `_gpu_summary_cached`.

### F09. [Low] `cache_dir()` / `host_cache_dir()` `makedirs(exist_ok=True)` on every call — 258 us and 0.5-1.6 ms per call for a directory that already exists — src/pyutilz/performance/kernel_tuning/cache/cache_base.py:230, 252

- **Disposition**: OPEN
- **Category**: syscall-churn
- **Problem**: both helpers unconditionally `os.makedirs(path, exist_ok=True)` before returning, even though after the first call the directory provably exists for the process lifetime. Measured, 5,000-iteration loops:
  - `cache_dir()`: **257.68 / 259.12 us** per call
  - `host_cache_dir()`: **1597.48 / 497.73 us** per call (two `makedirs` plus `hw_fingerprint()`)
  - `os.makedirs(path, exist_ok=True)` alone on the same existing path: **238.77 us**
  - `os.path.isdir(path)`: **47.52 us** (roughly 5x cheaper)
  This is consistent with the repo's own documented measurement in `system/system/fsutils.py:44-49` ("the (measured, ~2.9x) syscall overhead of `makedirs(exist_ok=True)` vs. exists-then-skip"); on the kernel-cache path the observed ratio is roughly 5x.
- **Failure scenario**: call-site analysis (grepped: `cache_base.py:127,154,242,251` and `cache_class.py:111`) shows these are reached from `KernelTuningCache.__init__` and from the lru-cached `hw_fingerprint`, so in practice this is process-startup cost, not per-lookup cost — roughly 0.5-2 ms per process, on Windows where directory syscalls are expensive. It becomes real if any future caller puts `cache_path()` or `host_cache_dir()` inside a loop, which nothing in the current signatures discourages.
- **Suggested fix**: memoize the creation, not the path — `@lru_cache(maxsize=4)` on a private `_ensure_cache_dir(path)` that does the `makedirs` once per distinct path, with `cache_clear()` wired into whatever the tests use when they repoint `PYUTILZ_KERNEL_CACHE_DIR`. The `exist_ok=True` semantics (and its race-tolerance, which is why it was chosen over exists-then-skip) are preserved for the first call; subsequent calls skip the syscall entirely.

### F10. [Low] Polars branch of `showcase_df_columns` materializes both value-count columns to Python lists twice — src/pyutilz/data/pandaslib/frames.py:243-244, 259-260

- **Disposition**: OPEN
- **Category**: redundant-materialization
- **Problem**: the display block calls `vc.get_column(var).to_list()` and `vc.get_column("count").to_list()` (lines 243-244), then the rare/uninformative block immediately calls the identical two `.to_list()` conversions again on the same unchanged frame as `rare_vals` / `rare_mask` (lines 259-260). Each `.to_list()` is a full Arrow-to-Python-object conversion of the whole column. This is the same *class* of issue as the two 2026-07-21 MEDIUM findings in this function (redundant recomputation of an already-available value) — those were fixed for `n_unique`; this pair was left. **Unmeasured**: the duplication is gated by `n_unique <= max_cat_uniq_qty` (default small), so the absolute cost is bounded by the cardinality cap, and no benchmark was constructed that would produce an honest number for a realistic cap. Flagged on code-reading plus the fixed sibling findings, not on a measurement.
- **Failure scenario**: bounded — at most one extra Arrow-to-Python conversion of `max_cat_uniq_qty` values per column. On a wide frame (hundreds of columns) with a generous `max_cat_uniq_qty` it is a visible constant factor on a display/profiling helper, and it is a latent trap if the cap is ever raised.
- **Suggested fix**: hoist `vals = vc.get_column(var).to_list()` and `counts = vc.get_column("count").to_list()` above the display branch and reuse them in the rare/uninformative block (they are the same values), mirroring the `n_unique = vc.height` reuse comment already in place directly above at lines 252-255. Better still, do the rare filter in polars (`vc.filter(pl.col("count") <= rare_threshold)`) and convert only the survivors.

### F11. [Low] `_pid_alive` constructs a fresh `ctypes.WinDLL("kernel32")` on every call — src/pyutilz/performance/kernel_tuning/cache/cache_base.py:284-286

- **Disposition**: OPEN
- **Category**: per-call-reinitialization
- **Problem**: the Windows branch does `kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)` inside the function body, so every liveness probe re-runs library loading and function-pointer setup. Measured, 2,000-iteration loops: `_pid_alive(os.getpid())` = **51.65 us** per call, of which the `ctypes.WinDLL("kernel32", use_last_error=True)` construction alone is **16.25 us** (roughly 31%). The `use_last_error=True` handle is required (the comment at lines 279-283 documents exactly why the shared `ctypes.windll.kernel32` is wrong here) — but that handle is a process-lifetime constant, not a per-call one.
- **Failure scenario**: bounded — `_pid_alive` is only reached from the stale-sweep-marker steal path, which runs at most a handful of times per process. This is Low on frequency, not on ratio. It matters if a future caller ever polls liveness in a loop.
- **Suggested fix**: build the handle once, lazily, at module scope (a module-level `_KERNEL32 = None` plus a small accessor, so import on non-Windows never touches `WinDLL`) and reuse it. Keep `use_last_error=True` and the per-call `ctypes.get_last_error()` read — the thread-local last-error is read per call and is unaffected by sharing the library handle.
