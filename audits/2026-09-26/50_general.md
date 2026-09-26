# 2026-09-26 general audit (pyutilz outside src/pyutilz/llm)

Scope: the modules mlframe imports most (from a `git grep` of `from pyutilz` in mlframe): system 53, core.pythonlib 45,
performance.kernel_tuning.cache 44, kernel_tuning.registry 36, dev.benchmarking 30, parallel 11, gpu_dispatch 6,
safe_pickle 3. Also covered: core.disk_cache and system.single_flight_cache. I checked the prior rounds (audits/2026-07-21_*,
audits/implemented/2026-09-02, audits/implemented/2026-09-03) and do not repeat items fixed there.
"Reproduced" means I confirmed the behaviour by running the code with PYTHONPATH=src.

### GEN-1 (High) -- `hash_array_summary` gives the same hash when numeric data changes in the middle, or anywhere in a column that contains a NaN

**Disposition:** RESOLVED -- `hash_array_summary` now hashes shape, dtype and EVERY byte of the array (object dtype: element `repr`, as before); the head/tail + sum/min/max summary is gone (`src/pyutilz/core/disk_cache.py:175`). Large buffers are hashed as a blake2b leaf tree with fixed 8 MiB element-aligned leaves (`_feed_full_buffer`, `disk_cache.py:138`), leaves hashed on up to 8 threads (hashlib releases the GIL); the leaf size is a constant, so the digest does not depend on the core count. Strided input is hashed leaf by leaf through `arr.flat[i:j]` rather than by copying the whole array. `_buffer` now returns a 1-D byte view (a 2-D memoryview slices rows, not bytes). Measured on 20M float64 (160 MB): old summary 0.05 s, serial blake2b 0.83 s, leaf tree 0.25-0.8 s on this (loaded) box. xxhash3 measured 0.15 s but is not a declared dependency, and an optional backend would make the key depend on the environment, so it was not used. `_HASH_VERSION` 3 -> 4 (`disk_cache.py:101`) and the pinned digests in `tests/test_disk_cache_digests_are_a_compatibility_contract.py` were re-captured at v4. `n_summary_rows` is still accepted and ignored. Tests: `TestGen1FullContentHash` (the four audited collision pairs, last-leaf change on the multi-leaf path, strided == contiguous, thread-count independence).
- Evidence: src/pyutilz/core/disk_cache.py:174-202. For a numeric array the key is built from the first and last 64 rows plus the
  per-column sum, min and max. Reproduced: each of these pairs of arrays hashes to the same value:
  - swap elements 500 and 501 of `np.arange(1000.)`;
  - swap rows 400 and 401 of a (1000,3) array;
  - set `b[500,0]=123` in a column whose row 0 is NaN;
  - set `b[500]=7` in a 1-D array with a NaN at index 0.
  In the two NaN cases the sum, min and max are all NaN, so changes to the other values have no effect on them.
- Impact: `DiskCache` silently returns a stale result that was computed for different data. The 2026-09-03 fixes (F01/F02)
  covered only non-numeric dtypes. The "sub-O(N)" argument for a summary no longer holds, because the sum/min/max pass already
  reads every element. Measured on 20M float64: the summary took 0.31 s and a full blake2b of the buffer took 0.85 s.
- Fix: hash the full contiguous buffer with blake2b over `_buffer(arr)` (use xxhash3 if that is too slow). If a summary must
  stay, add a position-weighted checksum to the fused numba pass (for example sum(i*x[i]) plus a NaN count and NaN-position
  hash). Bump `_HASH_VERSION` either way. mlframe already has a full-content `hash_array_content` in `mlframe.utils.disk_cache`.
  Moving that into pyutilz and removing the summary also removes a duplicated helper.

### GEN-2 (Med) -- Different kernel names can map to the same cache directory, and one of the two kernels is then lost

**Disposition:** RESOLVED -- `_kernel_dir` is now `<readable>-<blake2b(kernel_name)[:16 hex]>` with no CPU/GPU/`@` stripping and no case folding of the identity (`src/pyutilz/performance/kernel_tuning/cache/cache_base.py:295`). Compat: the old lossy slug is kept as `_legacy_kernel_dir` (`cache_base.py:288`); `_load` groups records from ALL directories by the `kernel_name` stored in each record and keeps the newest per kernel (`cache_persistence.py:165`, `_read_kernel_dir_all` at `:215`), so existing on-disk tunings in old directories are still found (no re-tune) and a newer record in the new directory wins. `_delete_kernel_files` (evict) also removes this kernel's records, and only this kernel's, from the legacy directory (`cache_persistence.py:380`). Tests: `TestGen2KernelDirs` (every audited colliding pair gets distinct dirs, two colliding kernels survive a reload, a legacy dir holding two kernels yields both, new beats legacy, evict leaves the neighbour).
- Evidence: `_slug` (cache_base.py:39-44) removes `\bCPU\b`, `\bGPU\b` and everything from `@` on, drops punctuation, lowercases
  and truncates. `_kernel_dir` (:279-281) names the directory `_slug(kernel_name, 80)`. Reproduced:
  - `hist.cpu`, `hist.gpu` and `hist cpu` all become `hist`;
  - `a@b` and `a@c` both become `a`;
  - `Mm` and `mm` both become `mm`;
  - `x(1)` and `x1` both become `x1`.

  `_read_kernel_dir_by_path` (cache_persistence.py:241-277) returns only one (name, entry) pair per directory, and
  `_gc_kernel_dir` (:345) keeps 4 files in the directory counted across both kernels.
- Impact: the other kernel is missing from `_load()`. It is re-tuned in every process, and GC deletes its files to make room for
  the one that shares its directory. No current mlframe kernel name triggers this (they all use only lowercase letters, digits
  and underscores), so it is latent. A name ending in `.cpu` or `.gpu`, containing `@`, or longer than 80 characters would
  trigger it.
- Fix: name the directory `<slug>-<blake2b(kernel_name)[:8]>`, or group files by the `kernel_name` stored inside them in
  `_read_kernel_dir_by_path` and return every kernel found. Stop removing CPU/GPU from kernel names: that rule was written for
  hardware brand strings.

### GEN-3 (Med) -- `_region_matches` matches when a dim is NaN or misspelled, and never matches `_eq` tuples after a reload

**Disposition:** RESOLVED -- `_region_match_reason` is now the single implementation and `_region_matches` wraps it (`region_matching.py:38`): a NaN dim fails any `_max`/`_min` constraint, and `_eq` compares with tuples normalised to lists on both sides (`_normalize_eq`, `:22`). The compiled hot path in `lookup` uses `not v <= cap` / `not v >= lo` so NaN fails there too, and normalises `_eq` bounds at plan-compile time (`cache_class.py:265`). A dim that is neither a declared axis nor constrained by any region logs one WARNING per (kernel, dim) per process (`_warn_unconstrained_dim`, `cache_class.py:276`); it is not treated as an error, since catch-all-only kernels are legitimate. Tests: `TestGen3RegionMatching`.
- Evidence: performance/kernel_tuning/cache/region_matching.py:20-38. Reproduced:
  - `_region_matches({'n_max':10},{'n':nan})` returns True.
  - `_region_matches({'n_max':10},{'rows':10**9})` returns True. A dim with no matching constraint is ignored, so a misspelled
    dim name matches the first region.
  - `_region_matches({'shape_eq':[2,3]},{'shape':(2,3)})` returns False. JSON storage turns tuples into lists, so an `_eq`
    region on a tuple value can never match once it has been reloaded from disk.
- Impact: the lookup silently picks the wrong tuned backend or config. In the tuple case every lookup falls back to the
  catch-all region or re-tunes.
- Fix: treat a NaN dim as an error or as no match. In `lookup`, warn once per (kernel, dim name) when a dim is not constrained by
  any region of that kernel. Convert tuples to lists on both sides before the `_eq` comparison. `_region_match_reason`
  (:41-54) repeats the same logic as `_region_matches`, so the two can drift apart. Merge them into one function.

### GEN-4 (Med) -- `synchronize_gpu_if_available` only waits on cupy's null stream and never on numba.cuda, although its docstring says it does

**Disposition:** RESOLVED -- `synchronize_gpu_if_available` (`src/pyutilz/dev/benchmarking.py:53`) now calls `cupy.cuda.runtime.deviceSynchronize()` (all streams, including per-thread default streams) when cupy is already imported and the thread has a CUDA context, and `numba.cuda.synchronize()` when numba.cuda is already imported and its driver initialised. It imports neither library and creates no context. The cupy context probe calls `getDeviceCount()` before `ctxGetCurrent()`, because `cuCtxGetCurrent` before driver init segfaulted in a reproduction on this box. Gate: the `gpu_timing_sync` gate lives in py_ci_shared (not editable here), so pyutilz's `tests/test_meta/test_gpu_timing_synchronize.py` gains `test_no_null_stream_only_sync_in_pyutilz` (AST: no `*.null.synchronize()` anywhere in src, with a can-fail fixture) and `test_the_timing_helper_is_a_device_barrier` (behavioural). Tests: `TestGen4DeviceWideSync`.
- Evidence: dev/benchmarking.py:28-52. Line 48 is `_cp.cuda.Stream.null.synchronize()`, but the docstring says "(cupy / numba.cuda)".
  If cupy is not installed the function does nothing. It also does not wait for work on other cupy streams, or when
  `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1` is set.
- Impact: `benchmark_algos_by_runtime`, `sweep_backend_crossover`, `_rank_candidates` and `sweep_backend_grid` measure only the
  kernel launch for those backends, and the kernel-tuning cache stores those times. This is the same failure that SCHEMA_VERSION
  4 was bumped to invalidate. The py_ci_shared `gpu_timing_sync` gate (tests/test_meta/test_gpu_timing_synchronize.py) only
  checks that a sync call is present, so it cannot tell that this helper's sync is too weak.
- Fix: wait on the whole device. Call `cp.cuda.runtime.deviceSynchronize()` when cupy is present, and `numba.cuda.synchronize()`
  when numba.cuda is available and already initialised. Check initialisation first so CPU-only runs do not create a CUDA context.

### GEN-5 (Med) -- `sweep_backend_crossover`: a NaN in the reference output rejects every other backend, and tuple or ragged outputs crash the sweep

**Disposition:** RESOLVED -- new `_output_vector` flattens host/device arrays and tuple/list/dict (including ragged) outputs into one float vector; `_output_scale` takes the largest finite |value| (1.0 if none or 0); `_max_abs_diff` requires non-finite values at the same positions with equal values and takes the max over finite positions (`benchmarking.py:160-215`). In both `sweep_backend_crossover` and `sweep_backend_grid` the scale is computed inside the reference `try`, so a non-numeric reference skips the size or cell instead of aborting the sweep. The grid gate is now `not (diff <= tol)` (a NaN diff used to pass `diff > tol`). `_to_host` no longer calls `dict.get()` and recurses into tuple/list/dict (`benchmarking.py:357`). Tests: `TestGen5SweepEquivalence` (a NaN-bearing reference lets a faster matching candidate win, ragged tuple outputs, a dict-returning reference in the grid sweep).
- Evidence:
  - dev/benchmarking.py:220-221: `float(np.abs(...).max() or 1.0)` stays NaN, because NaN is truthy and `or` never substitutes
    1.0.
  - :236: `diff <= atol + rtol*nan` is then False for every candidate.
  - `_max_abs_diff` (:136) also returns NaN when both outputs have a NaN in the same position.
  - The `ref_scale` line is outside any try block. If the reference returns a dict, or a tuple of arrays with different shapes,
    `np.asarray(..., float64)` raises and stops the whole sweep.
  - `sweep_backend_grid` has the same code at :474.
- Impact: the reference backend wins every size band and is stored as the tuned choice. The only trace is an INFO log line,
  written only when `verbose` is on.
- Fix: compute the scale with `np.nanmax`. Compare NaN positions first (`np.array_equal(isnan(a), isnan(b))`), then take nanmax
  of the difference. Move the scale computation into the try that skips a failing size. Flatten tuple and list outputs into one
  float vector before comparing.

### GEN-6 (Med) -- `cuda_memory_guard` raises MemoryError when cupy's pool has the memory, and empties the pool on every exit

**Disposition:** RESOLVED -- `cuda_memory_guard` counts driver free memory PLUS cupy's default-pool free bytes for that device (`_pool_free_bytes`, `src/pyutilz/system/gpu_dispatch.py:366`). New `release_pool: Optional[bool] = None` (`gpu_dispatch.py:406`): by default the pool is flushed on exit only when the driver's free memory alone was below the threshold. True always flushes, False never flushes. `_free_bytes_via_cupy` gained `include_pool=False`; the default stays driver-only because numba allocations cannot use cupy's pool, so `free_vram_gb` / `gpu_capability_summary` are unchanged. Tests: `TestGen6MemoryGuard`.
- Evidence: system/gpu_dispatch.py:257 and :259 read the driver's `memGetInfo`, which counts memory cached in cupy's default pool
  as used. Line 344 calls `cp.get_default_memory_pool().free_all_blocks()` in `finally` on every exit.
- Impact:
  - The guard raises MemoryError even when cupy could serve the allocation from its own pool. Callers then fall back to CPU,
    which gives correct results but runs much slower.
  - Using the guard in a loop empties the process-wide pool each time, including memory held for other threads, so every later
    allocation goes back to cudaMalloc and cudaFree.
- Fix: treat `free + pool.free_bytes()` as available. Make the pool flush on exit opt-in (a `release_pool=False` default), or
  flush only when the guard itself found memory tight.

### GEN-7 (Med) -- nvidia-smi (GPUtil) device ids are used as CUDA device ids

**Disposition:** RESOLVED -- GPUtil (nvidia-smi) entries are re-keyed to CUDA ordinals by UUID before use (`_gputil_to_cuda_ids`, `gpu_dispatch.py:147`; uuid map from cupy `getDeviceProperties(i)['uuid']`, falling back to numba `cuda.gpus[i].uuid`, cached per process and cleared by `reset_cache`, `_probe_cuda_ordinals_by_uuid` at `:117`). GPUs CUDA cannot see are dropped. This applies in `select_best_gpu`, `_free_bytes_via_gputil`, the static capability summary and `free_vram_gb`. Without cupy or numba the ids pass through unchanged, and a WARNING is logged when that is ambiguous (more than one GPU, or `CUDA_VISIBLE_DEVICES` set). Existing mocked tests pin the uuid map explicitly. Verified on this box: map `{gpu-31e313d0-...: 0}`, `select_best_gpu() == 0`. Tests: `TestGen7CudaOrdinals`.
- Evidence: system/gpu_dispatch.py:96-125. `select_best_gpu` returns `int(best["id"])` from GPUtil, and `_cc_tuple` passes that
  same id to CUDA. `_free_bytes_via_gputil` (:275-289) also compares `device_id` with nvidia-smi ids. No code under src/pyutilz
  reads `CUDA_VISIBLE_DEVICES` or `CUDA_DEVICE_ORDER`; the only related grep hit is a PCI_BUS_ID attribute removal at
  system/system/probing.py:765.
- Impact: nvidia-smi numbers GPUs in PCI bus order, while CUDA's default order is FASTEST_FIRST, and `CUDA_VISIBLE_DEVICES`
  renumbers the visible devices. On a multi-GPU host, or whenever `CUDA_VISIBLE_DEVICES` is set, the returned id can refer to a
  different GPU or to one CUDA cannot see. The result is either a silent wrong-device choice or an invalid-device error.
- Fix: map GPUtil entries to CUDA ordinals by UUID, which is already requested, using
  `cupy.cuda.runtime.getDeviceProperties(i)['uuid']` or numba's `get_current_device().uuid`. Drop any GPU that CUDA cannot see.

### GEN-8 (Med) -- `safe_load` checks the hash of one read and unpickles a second read; `write_sidecar` rewrites the sidecar in place

**Disposition:** RESOLVED -- `safe_load` reads the payload once, verifies THOSE bytes against the sidecar (`_verify_sidecar(..., payload=...)`), then calls `pickle.loads` on the same bytes (`src/pyutilz/core/safe_pickle.py:280`). `write_sidecar` and `safe_dump` write the sidecar through a unique temp file and `_replace_with_retry` (`_write_sidecar_digest`, `:263`). `safe_dump` hashes while dumping through `_HashingWriter` (`:343`), so it no longer reads the payload back. The public `verify_sidecar` signature is unchanged. Tests: `TestGen8SafePickle` (load works with `_sha256_of_file` patched to raise, a tampered payload is still rejected, the dump digest equals sha256 of the file, the sidecar goes through `_replace_with_retry` and leaves no temp).
- Evidence: in core/safe_pickle.py, `verify_sidecar` hashes the file at :238 (`_sha256_of_file`), then :259-260 open the file
  again for `pickle.load`. `write_sidecar` (:243) truncates the sidecar and writes it in place.
- Impact:
  - If another process `os.replace`s the payload between the two opens, the bytes that get unpickled were never verified. That
    defeats the only guarantee this module gives, and the per-path lock only works within one process.
  - Every load reads the whole file twice, and this sits on the DiskCache hot path.
  - A reader in another process can see an empty or half-written sidecar and raise PickleVerificationError on a valid entry.
    DiskCache.get then deletes that valid entry (disk_cache.py:406-412).
- Fix: read the bytes once, hash them, then call `pickle.loads(buf)`. Write the sidecar to a temp file and `os.replace` it into
  place. Optionally hash while dumping, with a writer that also feeds the hasher, so `safe_dump` does not read the file back.

### GEN-9 (Low) -- `hw_fingerprint` writes its disk cache through a fixed temp file name

**Disposition:** RESOLVED -- `_write_hw_fingerprint_to_disk` uses a per-writer temp name `<path>.<pid>.<thread>.<uuid4>.tmp` and removes it on failure (`cache_base.py:169`). Test: `TestGen9FingerprintTempName` (4 concurrent writers -> 4 distinct temp names, none equal to `path + ".tmp"`, no leftovers). The shared `atomic_write_staging` gate is now adopted too (GEN-15).
- Evidence: performance/kernel_tuning/cache/cache_base.py:163, `tmp = path + ".tmp"`.
- Impact: when two processes start at once, which is normal for per-target training scripts, both truncate and write the same
  temp file, and one can `os.replace` it while the other is still writing. The reader tolerates bad JSON, so the cost is a
  wasted ~2.7 s re-probe or a Windows PermissionError logged at DEBUG. `_atomic_write_json` (cache_persistence.py:61) already
  does this correctly.
- Fix: use `_CachePersistenceMixin._atomic_write_json`, or a temp name that includes the pid and a random suffix.

### GEN-10 (Low) -- `_read_kernel_newest` is unused and duplicates `_read_kernel_dir_by_path`

**Disposition:** RESOLVED -- `_read_kernel_newest` (no callers) deleted. One reader, `_read_kernel_dir_all`, remains; `_read_kernel_dir_by_path` is a thin wrapper over it, and the readers and `_gc_kernel_dir` share one recency key, `_record_sort_key` (`cache_persistence.py:205`). Tests: `TestGen10NoDuplicateReader`.
- Evidence: cache_persistence.py:144-184 has no callers; grep over src finds only the definition and mentions in docstrings. Its
  body is a line-for-line copy of `_read_kernel_dir_by_path` (:241-277).
- Fix: delete it, or keep one implementation and make the other a thin wrapper. The GC code (:345) has a third copy of the same
  sort key and should use the shared one too.

### GEN-11 (Low) -- `SingleFlightCache.get_or_fetch` counts a hit as a miss, and a second cancellation can leave waiters blocked forever

**Disposition:** RESOLVED -- `misses` is incremented only after the locked re-check misses; a re-check hit counts as a hit (`src/pyutilz/system/single_flight_cache.py:153`). The fetcher's `finally` pops and sets the event synchronously, without awaiting the lock (`:188`); the class is single-loop, so this is atomic. Tests: `TestGen11SingleFlight` (a re-check hit gives hits=1, misses=0; a cancelled fetcher wakes its waiter while the in-flight lock is held elsewhere).
- Evidence: system/single_flight_cache.py:145 adds to `misses` before the re-check at :151-153. When the re-check finds the
  value, it returns without counting a hit. At :183 the `finally` block does `async with lock`. If the task is cancelled again
  while waiting for that lock, `evt.set()` never runs and every waiter on that key blocks forever.
- Fix: count a hit when the re-check succeeds. The class runs on a single event loop, so the `finally` block does not need the
  lock: pop the event and set it synchronously, without awaiting.

### GEN-12 (Low) -- `_get_path_lock` does not normalise path case on Windows

**Disposition:** RESOLVED -- `_get_path_lock` keys on `os.path.normcase(os.path.abspath(path))` (`safe_pickle.py:124`). Test: `TestGen12PathLockCase` (on Windows, the upper-cased path maps to the same key).
- Evidence: core/safe_pickle.py:122 builds the lock key with `os.path.abspath` only.
- Impact: `C:\x\a.pkl` and `c:\X\A.pkl` are the same file on Windows but get two different locks. Two threads can then
  interleave the payload and sidecar writes, which is what the lock exists to prevent.
- Fix: `os.path.normcase(os.path.abspath(path))`.

### GEN-13 (Low) -- `distribute_work` balances load in input order instead of largest-first

**Disposition:** RESOLVED -- `distribute_work` assigns items largest-first (a stable sort, so equal sizes keep input order) and then returns each worker's items in input order, the same shape as before (`src/pyutilz/system/parallel.py:126`). Test: `TestGen13Lpt` (`[1,1,1,1,4]` over 2 workers: max load 4, was 6).
- Evidence: system/parallel.py:173-176.
- Impact: adding each item to the least-loaded worker in whatever order the items arrive can leave the slowest worker with up to
  about twice the ideal load. Sorting the items largest-first costs nothing extra and guarantees at most 4/3 of the ideal, and
  the result still maps back through `workload_indices_per_worker`.
- Fix: iterate `sorted(enumerate(workload), key=lambda t: -t[1])`.

### GEN-14 (Low) -- Four near-identical chunking helpers

**Disposition:** RESOLVED -- `split_list_into_chunks` and `split_array` are built on `split_list_into_chunks_indices` (`parallel.py:58-124`); `split_array` raises `ValueError` for `step < 1` (it was an `assert`, which `-O` strips, leaving an infinite loop for step 0; `tests/test_parallel_extra.py` updated from AssertionError). `batch` (`src/pyutilz/core/pythonlib/objects.py:262`) raises `ValueError` for `n < 1` and keeps its own loop, because importing `system.parallel` from `core.pythonlib` would invert the layering. Tests: `TestGen14Chunking`.
- Evidence: in system/parallel.py, `split_list_into_chunks` (:58) and `split_list_into_chunks_indices` (:81) differ only in what
  they yield, and `split_array` (:125) is a third version of the same thing. `core/pythonlib/objects.py:262 batch` is a fourth.
  `batch` silently yields nothing for a negative n, while the others raise on bad input.
- Fix: build the other three on `split_list_into_chunks_indices` and give all four the same `n >= 1` check.

### GEN-15 (Low) -- Local code_audit scanners duplicate py_ci_shared gates, and relevant shared gates are not adopted

**Disposition:** RESOLVED (gate adoption) / WON'T FIX (scanner retirement) -- Adopted in `tests/test_meta/test_shared_gates_adopted.py:98-120`: `atomic_write_staging`, `hash_key_determinism`, `hash_fed_by_array_copy` and `naive_utcnow`. All four exist at the pinned py-ci-shared SHA 6a8e382 (v1.17.0) with the same signatures. The only finding, `read_json_with_checkpoint_fallback` rewriting its cache in place, was fixed to go through `_atomic_write_bytes` (`src/pyutilz/data/git_checkpoint_cache.py:74`; test `TestGen15CheckpointRestoreIsAtomic`), so all four gates run at zero findings with no baseline. `import_cycles` and `stale_source_citations` were already adopted in that file. The local `pyutilz.dev.code_audit.import_cycles` / `stale_source_citations` are NOT replaced by re-exports. They are public scanners of the runtime `pyutilz.dev.code_audit` registry and return `pyutilz` `Finding` objects through `scan_*` functions, while the shared modules expose `assert_*`/`find_*` over their own Finding type (a different shape). py-ci-shared is also a dev-only git dependency (`requirements-dev.txt`, python>=3.9, unpublishable as a runtime requirement), so a re-export would break `pyutilz.dev.code_audit` for every installed user and on the 3.8 legs. The registry already maps both to their shared counterparts (`registry.py` `SHARED_GATE_COUNTERPARTS`).
- Evidence: src/pyutilz/dev/code_audit/import_cycles.py (257 LOC) and stale_source_citations.py have the same names as
  py_ci_shared/import_cycles.py (413 LOC) and py_ci_shared/stale_source_citations.py. pyutilz uses about 10 py_ci_shared
  modules. In .pre-commit-config.yaml: pinned_tool_versions, format_warn, black_filtered_apply, advisory_warn, mypy_gate. In
  tests: audit_round_format, gpu_timing_sync, swallowed_exceptions. Several shared gates that cover bug types found above are not
  used: hash_fed_by_array_copy, hash_key_determinism, module_cache_thread_safety, atomic_write_staging, naive_utcnow.
- Fix: replace the two same-name scanners with thin re-exports of the py_ci_shared versions. Adopt `atomic_write_staging` and
  `hash_key_determinism` in tests/test_meta/test_shared_gates_adopted.py; `atomic_write_staging` is the kind of gate that
  targets GEN-8's in-place sidecar write and GEN-9.

## Counts
High 1, Med 7, Low 7 (total 15).

Checked, no findings: `open()` calls outside llm all pass an explicit encoding; the numerics and datetimes helpers in
core.pythonlib; the min-over-repeats timing logic in `benchmark_algos_by_runtime`.
