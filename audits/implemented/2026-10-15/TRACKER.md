# 2026-10-15 audit round: tracker

One row per finding. Statuses: RESOLVED (fixed; regression test where a test can show it), WON'T FIX (reason in the report), DEFERRED (waits on the owner decision named in the report). The disposition sits under each report's 'Disposition (round close)' section.

| Status | Sev | ID | Title |
|---|---|---|---|
| **RESOLVED** | P2 | `C-01` | The mtime recorded after a reload is stat'ed after the file is read, so an edit landing in between is recorded as already seen and never reloaded. |
| **RESOLVED** | P1 | `C-02` | Any single thread's transient `OperationalError` closes the whole shared pool, including connections other threads curre |
| **RESOLVED** | P3 | `C-03` | Lock-free fast path reads `_pool` and `_pool_dsn` as two separate globals that the writers update in two separate steps. |
| **RESOLVED** | P2 | `C-04` | Check-then-unlink on a shared filename is not atomic, and unlinking a file that is still memory-mapped fails on Windows. |
| **RESOLVED** | P3 | `C-05` | Bounded-FIFO eviction is a check-then-act on a shared dict with no lock. |
| **RESOLVED** | P3 | `C-06` | Freshness check and value read are two separate reads of a global that other threads set to `None`. |
| **RESOLVED** | P3 | `C-07` | `get` does not take the per-key lock that `put` holds, so a reader can see a half-written pair and delete a valid new en |
| **RESOLVED** | P3 | `C-08` | `get()` reads `self._data` without the lock, and the `data` setter stores the caller's dict by reference. |
| **RESOLVED** | P3 | `C-09` | The breaker's `state` is a public, mutable `CircuitState` object that callers can read or mutate without the lock, and ` |
| **RESOLVED** | P2 | `R-01` | `connect_to_db()` overwrites `_facade.conn` with a new psycopg2 connection without closing the previous one. |
| **RESOLVED** | P3 | `R-02` | Each (re)connect builds a new SQLAlchemy engine and rebinds `_facade.conn_alchemy` without `dispose()` on the old engine |
| **RESOLVED** | P3 | `R-03` | A named (server-side, WITH HOLD) cursor created for the attempt is not closed on the `OperationalError`/`InterfaceError` |
| **RESOLVED** | P3 | `R-04` | The temp directory from `mkdtemp` is removed only on the straight-line success path. The benchmark calls are wrapped, but a BaseException such as KeyboardInterrupt skips the cleanup. |
| **RESOLVED** | P3 | `R-05` | When `webdriver.Chrome(**kwargs)` raises, the proxy-auth zip (credential-bearing, mode 0600) created by `_configure_prox |
| **WON'T FIX** | P3 | `R-06` | Gate coverage gap. The gate checks only the `connect`/`close` pair, and it exempts the three modules where the release l |
| **RESOLVED** | P1 | `S-01` | A scanner that raises is converted to a single WARNING and `[]`, and `run_all` merges that empty list with the other sca |
| **RESOLVED** | P1 | `S-02` | `subprocess.check_call` failure is caught and logged at DEBUG only; the function then returns `None`. |
| **RESOLVED** | P1 | `S-03` | Any exception from the concat-and-write step is logged at DEBUG and the function returns `None`. The docstring says `Non |
| **RESOLVED** | P2 | `S-04` | Any failure of `cpuinfo.get_cpu_info()` is logged at DEBUG and reported as "flag unsupported". |
| **RESOLVED** | P2 | `S-05` | `except Exception: out[name] = float("inf")` has no log call at all. |
| **RESOLVED** | P2 | `S-06` | Category-conversion failure logs only when `verbose=True`, then the column is left unconverted. |
| **RESOLVED** | P3 | `S-07` | An `exists()` OS error is logged at DEBUG and treated as "not found". |
| **RESOLVED** | P3 | `S-08` | Failure to delete a source file after a successful concat is logged at DEBUG. |
| **RESOLVED** | P3 | `S-09` | A non-404 fetch failure returns `{}`, the same value as "stats not requested". |
| **RESOLVED** | P3 | `S-10` | A write failure increments `put_failures` and logs WARNING, then returns `None`. |
| **RESOLVED** | P3 | `S-11` | Any unexpected probe failure returns `True` ("assume alive") at DEBUG. |
| **RESOLVED** | P3 | `S-12` | A `TypeError` in the per-column fallback skips the column with no log. |
| **RESOLVED** | P3 | `S-13` | A persistent download error is logged at ERROR and retried after a sleep, with no retry limit. |
| **RESOLVED** | P1 | `S-14` | The scanner returns "not silent" for any handler whose only log call is `logger.debug`, including handlers that return a |
| **RESOLVED** | P2 | `X-01` | `_attempt` logs `str(e)` at WARNING; for healthchecks.io and cronitor.io the API/ping key is embedded in the URL path, a |
| **RESOLVED** | P2 | `X-02` | The "fixed" CDP config still exposes the DevTools endpoint on 127.0.0.1 with an OS-chosen port. The comment itself says  |
| **RESOLVED** | P2 | `X-03` | `unserialize(path)` defaults to `verify_sidecar=False`, so a file path goes straight to `pickle.loads` with no integrity |
| **WON'T FIX** | P2 | `X-04` | The comment says a payload planted in the cache dir is "never unpickled silently". The sidecar is integrity-only, as saf |
| **RESOLVED** | P3 | `X-05` | `load_file` unpickles `.joblib`/`.pckl` with no sidecar check by default. The `nosec B301` reason ("caller-supplied, sam |
| **RESOLVED** | P3 | `X-06` | `load_tokens` unpickles a caller-named gzip file. The "same trust as the writer" claim is unchecked: the writer's output |
| **RESOLVED** | P3 | `X-07` | `returning` is concatenated into SQL with no validation (comment at :418-419 calls it an "accepted raw fragment"). Ident |
| **DEFERRED** | P3 | `X-08` | `alternate_fields_values` is raw SQL inside the insert, unvalidated. The comment at :240 says so, and the docstring has  |
| **WON'T FIX** | P3 | `X-09` | `custom_onconflict` is returned verbatim into the statement with no validation, and `on_conflict_update_values` entries  |
| **WON'T FIX** | P3 | `X-10` | `condition` is spliced verbatim into `SELECT`. The WARNING docstring (:660-663) says so, so this is a documented escape  |
| **WON'T FIX** | P3 | `X-11` | `read_table_into_dict` and `read_table_into_dict_reversed` splice `condition` into SELECT. The docstrings (:106-109, :13 |
| **RESOLVED** | P3 | `X-12` | The lock file is `tempfile.gettempdir()/sha256(path).lock`, a predictable name in a shared directory, with no per-user s |
| **RESOLVED** | P3 | `X-13` | The "xor" mode is base64 only. With the default `encryption="xor"`, values written with `write_config_file` (for example |
| **RESOLVED** | P3 | `X-14` | Credentials are sent via Basic auth to `filemaker_url` with no check that the scheme is https. `init()` (line 32-53) doe |
| **WON'T FIX** | P3 | `X-15` | Public-IP lookups use plain http. The response is trusted to be an IP or geo record, so a MITM can feed a wrong IP into  |
| **RESOLVED** | P3 | `X-16` | The download helpers accept any URL and follow redirects with no scheme or host check. `url_guard.py:6-9` says every cal |
| **RESOLVED** | P3 | `X-17` | `get_hash` defaults to MD5. No security caller was found in `src` (the only other MD5 use, `system/distributed.py:61-63` |
| **RESOLVED** | P3 | `X-18` | `alias`, `prefix`, `suffix` are concatenated into the returned SQL select list with no validation. Only `table` is valid |
| **RESOLVED** | P2 | `N-01` | When `clean_features=False` and `clean_targets=False`, no clipping runs, so a column whose min is -inf gets `bin_width = |
| **RESOLVED** | P2 | `N-02` | `ensure_dataframe_float32_convertability` and `convert_float64_to_float32` cast int32/int64/uint32/uint64 to float32 unc |
| **RESOLVED** | P3 | `N-03` | The overflow check uses `pl.col(c).abs().max()`. For Int64 this is `abs(-2**63)`, which wraps back to a negative number, |
| **RESOLVED** | P3 | `N-04` | `rel_total_by = total_by / sum(total_by).over(groups)` has no guard for a zero group total, so the result is NaN or inf  |
| **WON'T FIX** | P3 | `N-05` | The weighted mean divides by `sum(weights)` and then relies on the caller's `nans_filler` to repair the result. Zero-sum |
| **RESOLVED** | P3 | `N-06` | `entropy()` calls `math.log2(proba / normalization_factor)` for every count, so a zero count (`Counter` can hold explici |
| **RESOLVED** | P3 | `N-07` | The kernels are `@njit(fastmath=True)`, which assumes no NaN or inf. The NaN filter lives only in `normality_verdict` (` |
| **RESOLVED** | P3 | `N-08` | The Decimal to Float64 cast is justified by a comment about polars versions ("older polars refuses ..."). The result dep |
| **RESOLVED** | P2 | `P-01` | `_trim_all` rebuilds every port's outcome list on every `report_success`, `report_error` and pick, under the pool-wide l |
| **RESOLVED** | P1 | `P-02` | Sampling appends to unbounded Python lists for the monitor's whole lifetime (about 10 lists, one entry per sample). |
| **RESOLVED** | P2 | `P-03` | `_collect_frames` keeps every loaded frame in `lst` and calls `get_df_memory_consumption(deep=True)` on each one. Deep m |
| **RESOLVED** | P3 | `P-04` | `re.search(r"```(?:json)?\s*\n?(\{.*?\}\/\[.*?\])\s*\n?```", text, re.DOTALL)` lazily scans the full response. On long L |
| **RESOLVED** | P3 | `P-05` | The dead-letter queue, once full, rebuilds its list with a slice on every `add`, so each insert costs O(max_size). |
| **RESOLVED** | P2 | `A-01` | `[speedups]` is documented as "orjson -- drop-in accelerator, every use site falls back to stdlib json without it", but  |
| **RESOLVED** | P2 | `A-02` | Core dependency list and count omit `orjson`: the docs say "nine hard dependencies" and list eight names (plus tomli), b |
| **RESOLVED** | P2 | `A-03` | `[llm]` is documented as "google-genai (>=3.9 only)", but pyproject declares `google-genai>=1.51 ; python_version >= '3. |
| **RESOLVED** | P2 | `A-04` | Docstring says `lock_timeout` is "How many seconds to wait for the lock before skipping", but on timeout the code logs a |
| **RESOLVED** | P2 | `A-05` | On local paths the function raises `ImportError` unless the `filelock` package is installed, but the docstring is silent |
| **RESOLVED** | P3 | `A-06` | The `[dev]` line lists the dev extras but omits `pytest-xdist`, which README.md:39 and pyproject `dev` (pytest-xdist>=3. |
| **RESOLVED** | P3 | `A-07` | The package `__all__` omits public helpers that the `system.system` subpackage docstring advertises as its contents (`re |
| **RESOLVED** | P2 | `T-01` | Absolute wall-clock bound `min(elapsed) < 2.0` on a list of timings; the 1s timeout is also checked only through a 10s s |
| **RESOLVED** | P3 | `T-02` | `min(elapsed) < 1.0` as the only proof that a fast function did not wait for its 5s timeout. |
| **RESOLVED** | P2 | `T-03` | `test_closed_default_dynamic` has a body identical to `test_group_by_dynamic_basic` (line 389): same call, same `isinsta |
| **RESOLVED** | P2 | `T-04` | `test_benchmarker_picks_faster_equivalent` asserts the winner from a 5ms sleep difference measured over 3 repeats, so th |
| **RESOLVED** | P3 | `T-05` | Ranking tests rely on a 20ms `time.sleep` in the reference to make the candidate win, so the expected winner is decided  |
| **RESOLVED** | P3 | `T-06` | `assert slow > fast` compares two real `time_backend` medians with a 10ms sleep and no injected timer. |
| **RESOLVED** | P3 | `T-07` | Fixture teardown removes `carvepkg*` modules with `del sys.modules[name]`, a form the reload gate does not inspect. The  |
| **RESOLVED** | P2 | `T-08` | The whitelist reason says `config.py` "defines no sentinel objects (verified via grep for '= object()')", but `src/pyuti |
| **RESOLVED** | P3 | `T-09` | `pytest.skip("py-cpuinfo reports no flags on this host")` makes the only check of `check_cpu_flag` depend on host hardwa |
| **RESOLVED** | P3 | `T-10` | Two "allowed" tests have no assertion of their own; they pass only if `assert_only_target_changed` does not raise (`src/ |
| **RESOLVED** | P3 | `T-11` | A live-provider timeout becomes `pytest.skip(...)`, so a provider outage turns the live test green. The reason is stated |
