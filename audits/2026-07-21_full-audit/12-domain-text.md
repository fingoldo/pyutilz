# Domain audit: text (strings, similarity, humanizer, tokenizers)

## Summary

Covered `src/pyutilz/text/__init__.py`, `humanizer.py`, `similarity.py`, `tokenizers.py`, and the `strings/` subpackage (`__init__.py`, `_logproxy.py`, `basics.py`, `configfiles.py`, `jsonutils.py`, `textentropy.py`, `webtext.py`) in full, plus enough of `pyutilz.database`/`pyutilz.database.db` to verify one cross-module claim. The numba-accelerated `similarity.py` kernels are, overall, solid: a 218-case differential fuzz test (empty/single-char/Unicode/Cyrillic/astral-emoji/very-different-length inputs) against the pure-Python reference found **zero** mismatches, real measured speedups (2.7x-10.8x) were reproduced, and the sort-based large-N greedy matcher was verified bit-exact against the plain scan at its threshold boundary (N=560). All findings below were confirmed by actually running the code (not inferred), per the zero-hallucination rule. Two High-severity issues were found (an orjson-vs-stdlib-json silent NaN/Infinity divergence in `jsonutils.py`, and an unconditional `psycopg2`/`sqlalchemy` import chain in `tokenizers.py` that contradicts its own comment and is not covered by the existing masked-import test suite), five Medium, and three Low.

## Findings

### [High] `json_pg_dumps` silently changes NaN/Infinity semantics depending on whether orjson is installed — src/pyutilz/text/strings/jsonutils.py:228-246
- **Category**: correctness / portability
- **Problem**: `json_pg_dumps` tries `orjson.dumps(...)` and falls back to stdlib `json.dumps(...)` only on `ImportError`. `orjson` does **not** raise on `NaN`/`Infinity`/`-Infinity` floats — it silently serializes them as JSON `null` (verified below). Stdlib `json.dumps` (used when orjson isn't installed) serializes them as the literal (non-standard-JSON) tokens `NaN`/`Infinity`/`-Infinity`, which `json.loads` on the next line happily parses back into real Python `float('nan')`/`float('inf')` values. So the exact same call produces **semantically different payloads** depending purely on which packages happen to be installed in the environment — with orjson, the real NaN/Infinity information is silently discarded (turned into `null`, indistinguishable from "value absent"); without orjson, it survives as a non-standard JSON token that Postgres's strict `json`/`jsonb` parser will typically reject at `INSERT` time (Postgres documents that its json/jsonb types reject `NaN`/`Infinity` since the JSON standard disallows them — not independently re-verified against a live DB in this sandbox, flagged as it is standard, well-documented Postgres behavior).
- **Failure scenario**: Reproduced directly:
  ```
  json_pg_dumps({"score": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "ok": 1.5})
  ```
  With orjson installed (this repo's dev env, orjson 3.11.9): the returned `Json` wrapper adapts to `{"score": null, "inf": null, "ninf": null, "ok": 1.5}` — a "score" that was NaN is now indistinguishable from a missing/null score once stored in the jsonb column.
  With orjson blocked (simulated via a masked `__import__`, i.e. what a `pip install pyutilz` without orjson looks like): the returned `Json` wrapper adapts to `{"score": NaN, "inf": Infinity, "ninf": -Infinity, "ok": 1.5}` — invalid JSON tokens that a real Postgres server will reject at insert time with a syntax error, rather than the silent-null behavior above.
  Either behavior might be "intended" in isolation, but having them differ **silently based on installed packages** is the real bug: the same application code behaves differently in dev (orjson installed) vs. a minimal prod image (orjson absent), or vice versa.
- **Suggested fix**: Decide on one canonical NaN/Infinity policy (e.g. always coerce to `None` before serializing, or always raise) and apply it explicitly before dispatching to either backend, rather than letting the two libraries' incompatible default behaviors leak through un-normalized.

### [High] `tokenizers.py`'s unconditional `from pyutilz.database import db` breaks import when only the `nlp` extra is installed, contradicting its own comment and untested by the existing isolation-test suite — src/pyutilz/text/tokenizers.py:22-25
- **Category**: architecture / packaging / testing-gap
- **Problem**: Lines 22-24 read: *"psycopg2 + pyutilz.database are deferred to call site - tokenizers.py is loaded by mlframe via the pyutilz.text re-export chain; not every consumer needs the database stack just to use string utilities below."* The very next line, however, is a plain top-level `from pyutilz.database import db` — **not** deferred at all. `pyutilz.database.db.__init__` (`src/pyutilz/database/db/__init__.py:32-42`) unconditionally does `import sqlalchemy`, `import psycopg2`, `from psycopg2 import sql`, etc. at module scope. `pyproject.toml`'s `nlp` extra (lines 110-122) — the extra you'd install to actually use `AdvancedTokenizer`/spacy/nltk — does **not** include `sqlalchemy`/`psycopg2`/`pymysql` (those live only in the separate `database` extra, lines 61-71). So `pip install pyutilz[nlp]` alone leaves `pyutilz.text.tokenizers` unimportable.
  This exact scenario is not covered by the existing meta-test suite: `tests/test_meta/test_optional_deps_isolation.py::test_tokenizers_module_imports_without_nlp_group_deps` imports `pyutilz.text.tokenizers` but only masks the **`nlp`** dep group (spacy/nltk/jellyfish/tiktoken) — it never masks the **`database`** group while testing this specific import, so the gap slips through CI green.
- **Failure scenario**: Reproduced directly by mirroring the project's own masking-test technique (`sys.meta_path` blocking `sqlalchemy`/`psycopg2`/`pymysql`, matching `pyproject.toml`'s `database` extra exactly) and running `import pyutilz.text.tokenizers`:
  ```
  IMPORT FAILED: ImportError (masked) sqlalchemy
  ```
  A real user who does `pip install pyutilz[nlp]` (the extra whose whole purpose is enabling `AdvancedTokenizer`) and then `from pyutilz.text.tokenizers import AdvancedTokenizer` gets `ModuleNotFoundError: No module named 'sqlalchemy'` before ever touching a database, even for the purely in-memory `tokenize()` method that has nothing to do with SQL.
- **Suggested fix**: Move `from pyutilz.database import db` (and the `psycopg2.extras` import already correctly deferred inside `tokenize_db_reviews`) to be lazy/deferred inside the one method that actually needs it (`tokenize_db_reviews`), matching what the comment already claims is the design; then extend `test_tokenizers_module_imports_without_nlp_group_deps` (or add a sibling test) to also mask the `database` group.

### [Medium] `read_config_file` silently drops values that collide across multiple INI sections — src/pyutilz/text/strings/configfiles.py:32-38, 59-62
- **Category**: correctness
- **Problem**: `prepend_section_names` is hardcoded to `False` in **both** branches of `if isinstance(section, str): ... prepend_section_names = False / elif section is None: ... prepend_section_names = False`. There is no code path that ever sets it `True`, so the `if prepend_section_names: object[next_section.lower() + "_" + var] = val` branch (line 60) is dead code. The intent (visible from the branch's existence and the `else: object[var] = val` fallback) was clearly to disambiguate same-named variables across different sections when reading *all* sections (`section=None`, the function's own default) — but that disambiguation never fires, so when the same variable name appears in more than one section, each section's read silently **overwrites** the previous section's value in `object`.
- **Failure scenario**: Reproduced directly with a two-section INI file:
  ```ini
  [SectionA]
  timeout = 30

  [SectionB]
  timeout = 60
  ```
  ```python
  obj = {}
  read_config_file(path, obj, section=None)   # section=None is the function's own default
  # obj == {'timeout': 60}   -- SectionA's timeout=30 is silently lost, no warning, no error
  ```
  No current in-repo caller triggers this (`pyutilz/system/scheduling/prefect.py` and `pyutilz/cloud/cloud.py` both pass an explicit `section=`), but this is a public API function (tracked in `tests/test_meta/_api_snapshot.json`) whose own default argument (`section: Optional[str] = None`) is the trigger, and multi-environment `[dev]`/`[prod]`-style INI files with repeated key names across sections are a very common real-world pattern.
- **Suggested fix**: Set `prepend_section_names = True` in the `elif section is None:` branch (reading multiple sections) — that's the one place a key collision can actually occur — or at minimum warn/collect-into-list instead of silently last-write-wins. Also note: since `section`'s only checked possibilities are `str`/`None` yet its declared type is `Optional[str]`, this is exhaustive as typed, but if the type is ever loosened, `sections`/`prepend_section_names` become unbound — worth an explicit `else: raise`/assert for robustness.

### [Medium] `sentences_similarity_numba` crashes on lone/unpaired surrogate codepoints where the pure-Python reference succeeds — src/pyutilz/text/similarity.py:577-603, 656-658
- **Category**: correctness / edge-case
- **Problem**: `_pack_words` (used by `sentences_similarity_numba`, `pack_sentence`, `sentences_similarity_numba_batch`, and `SentenceSimilarityIndex`) packs every word via `w.encode("utf-32-le")` **unconditionally and eagerly**, before any comparison logic runs. A Python `str` containing a lone/unpaired UTF-16 surrogate codepoint (e.g. `"\ud800"` — producible via malformed-encoding recovery, `surrogateescape`, or hand-built strings) cannot be UTF-32-encoded and raises `UnicodeEncodeError`. The pure-Python `sentences_similarity`, by contrast, never encodes anything — it compares Python strings directly and only calls `jellyfish.levenshtein_distance` (which *would* also raise on the same input) when an exact-match/prefix-match shortcut doesn't already resolve the pair. So for inputs where every actual comparison is resolved by the exact-match fast path, the numba variant crashes while the "same algorithm, same results, just faster" pure-Python reference succeeds.
- **Failure scenario**: Reproduced directly:
  ```python
  sentences_similarity(["\ud800"], ["\ud800"])          # -> 1.0  (a == b, exact-match shortcut, no encoding needed)
  sentences_similarity_numba(["\ud800"], ["\ud800"])     # -> UnicodeEncodeError: 'utf-32-le' codec can't encode
                                                          #    character '\ud800' in position 0: surrogates not allowed
  ```
  Confirmed this is specific to lone surrogates, not Unicode in general: a 218-case differential fuzz run (including Cyrillic text and astral-plane emoji like `"😀😀😀"` vs `"😀😀😁"`) found zero mismatches and zero crashes outside this surrogate case.
- **Suggested fix**: Either wrap `_pack_words`'s encode step to catch `UnicodeEncodeError` and fall back to the pure-Python path for the offending call (mirroring the existing `HAS_NUMBA` fallback pattern), or document the limitation explicitly in `sentences_similarity_numba`'s docstring (it currently claims "Same algorithm... Same result" with no caveat).

### [Medium] `_FacadeLoggerProxy` doesn't forward `repr()`/`str()`/`isinstance()` — src/pyutilz/text/strings/_logproxy.py:16-31
- **Category**: correctness / edge-case
- **Problem**: `_FacadeLoggerProxy.__getattr__` correctly forwards ordinary attribute/method access (`.warning(...)`, `.debug(...)`, `.name`, `.level`, ...) to the live-resolved real logger — this is the documented design goal and it works. But Python resolves *implicit special-method protocols* (`repr()`, `str()`, `isinstance()` via the type's MRO, not `__getattr__`) directly against the object's **type**, bypassing `__getattr__` entirely. Since `_FacadeLoggerProxy` defines no `__repr__`/`__str__` and doesn't subclass `logging.Logger`, these all silently fall through to `object`'s defaults instead of reflecting the real logger.
- **Failure scenario**: Verified directly on the live proxy object (`pyutilz.text.strings._logproxy.logger`):
  ```
  repr(logger)  -> '<pyutilz.text.strings._logproxy._FacadeLoggerProxy object at 0x0000...>'
                   (a real logging.Logger would repr as '<Logger pyutilz.text.strings (WARNING)>')
  str(logger)   -> same generic object string, not the resolved logger's identity
  isinstance(logger, logging.Logger) -> False
  ```
  Pickling, by contrast, works fine (verified: `pickle.loads(pickle.dumps(logger))` round-trips to a functionally-equivalent fresh proxy, since the class is stateless). The blast radius today is limited — the public facade `pyutilz.text.strings.logger` is a *real* `logging.Logger` defined directly in `strings/__init__.py`; the proxy is only reachable by importing a specific submodule directly, e.g. `pyutilz.text.strings.basics.logger` or `pyutilz.text.strings.webtext.logger` — but those submodule names are not underscore-prefixed and are perfectly reasonable to import directly.
- **Suggested fix**: Add `__repr__`/`__str__` (and optionally `__eq__`) that delegate to `self._resolve()`, and/or make `_FacadeLoggerProxy` inherit from `logging.Logger` (or register it via `logging.Logger.register(_FacadeLoggerProxy)` if using an ABC) so `isinstance` checks against `logging.Logger` succeed for code that validates logger type before use.

### [Medium] Re-exported `nlp`/`inflect_engine`/`ascii_emojies`/`unicode_emojies` in the `strings` facade go stale immediately after import — src/pyutilz/text/strings/__init__.py:103-111, src/pyutilz/text/strings/webtext.py:13-15,22-26,365,407-412
- **Category**: correctness / edge-case / architecture
- **Problem**: `strings/__init__.py:111` does `from .webtext import inflect_engine, nlp, ascii_emojies, unicode_emojies` — a **one-time snapshot** taken at package-import time (all `None`, since `webtext.py` initializes them lazily). `webtext.py`'s `spacy_sent_tokenize`, `suffixize`, and `sentencize_text` later mutate `webtext.py`'s *own* module-level globals via `global nlp` / `global inflect_engine` / `global ascii_emojies, unicode_emojies` — but nothing ever updates the separate copies bound into `strings/__init__.py`'s namespace. This is explicitly acknowledged in a comment ("The authoritative, mutated copies live in webtext; read those for live state") but the pre-split flat `strings.py` had a single truly-live global, so this is a genuine, if documented, behavioral regression for anyone who imports these names through the facade.
- **Failure scenario**: Reproduced directly:
  ```python
  import pyutilz.text.strings as strings_pkg
  import pyutilz.text.strings.webtext as webtext_mod
  strings_pkg.inflect_engine                       # None
  webtext_mod.suffixize("job", 2)                   # "jobs"  (lazily creates webtext_mod.inflect_engine)
  webtext_mod.inflect_engine                        # <inflect.engine object ...>   -- now populated
  strings_pkg.inflect_engine                        # still None -- the facade's copy never updated
  strings_pkg.inflect_engine is webtext_mod.inflect_engine   # False
  ```
  Any code that does `from pyutilz.text.strings import inflect_engine` (or `nlp`, `ascii_emojies`, `unicode_emojies`) expecting to observe the lazily-cached object after calling `suffixize`/`spacy_sent_tokenize`/`sentencize_text` will always see `None`/the pre-population placeholder, no matter how many times those functions were called.
- **Suggested fix**: Apply the same forwarding-proxy pattern already built for `logger` (`_logproxy.py`) to these four names, or simply drop them from the facade's re-export and document that live state must be read from `pyutilz.text.strings.webtext` directly.

### [Medium] `humanize()`'s `protected_spans` silently loses accuracy across its own pipeline, defeating the documented compliance-phrase use case — src/pyutilz/text/humanizer.py:203-209, 325-394
- **Category**: correctness / edge-case
- **Problem**: The module docstring's headline example is protecting "attention-check compliance phrases in cover letters" via `protected_spans`. `humanize()` runs `strip_ai_patterns` -> `fix_dashes` -> `strip_emojis` -> `introduce_typos(..., protected_spans=...)` in sequence; the first three stages can delete/insert text (e.g. `strip_ai_patterns` removes whole filler phrases like `"It's worth noting that "`), shifting every downstream offset. The docstring for `humanize()` admits: *"the caller should compute them against the cleaned text or pass raw offsets and accept minor drift."* But `_is_protected`'s overlap check (`humanizer.py:203-209`) is exact-offset-based with no fuzziness/anchoring — "accept minor drift" for a feature whose entire purpose is exact preservation of a compliance-verification string means the drifted span can end up (a) protecting the wrong substring (typos still land inside the real compliance phrase) or (b) missing the real phrase's now-shifted location, i.e. the safety feature can silently fail exactly when it matters, without raising or logging anything.
- **Failure scenario**: `humanize("It's worth noting that PASSPHRASE-42 must appear verbatim.", protected_spans=[(23, 36)])` where `(23, 36)` was computed against the *original* string (covering `"PASSPHRASE-42"`) — after `strip_ai_patterns` removes the 24-character leading filler phrase, `"PASSPHRASE-42"` now starts at offset 0, but `_is_protected` is still checking against the stale `[23, 36)` window (now pointing at unrelated trailing text), so `introduce_typos` is free to corrupt `"PASSPHRASE-42"` while "protecting" text that no longer needs it.
- **Suggested fix**: Either have `humanize()` track and return the cumulative offset delta from each stripping stage (so callers can remap original-text spans automatically), or expose the individual pipeline stages plus a documented "compute your spans against this already-cleaned string" helper so the caller never has to guess.

### [Low] `pbar.update(nitems)` passes a cumulative total where tqdm expects a per-call increment — src/pyutilz/text/tokenizers.py:239-243
- **Category**: efficiency / correctness (cosmetic)
- **Problem**: `tqdm.update(n)` adds `n` to the bar's current position; it is not "set position to n". The loop computes `nitems = nitems + chunksize` (a running cumulative total) and then calls `pbar.update(nitems)` — so each call adds the *entire cumulative count so far* on top of the previous (already-inflated) position, rather than just the new chunk's `chunksize`.
- **Failure scenario**: With `chunk_size=1000` and 3 chunks processed: after chunk 1, `nitems=1000`, `pbar.update(1000)` -> bar at 1000 (correct by coincidence, since it started at 0). After chunk 2, `nitems=2000`, `pbar.update(2000)` -> bar jumps to `1000+2000=3000` (should be 2000). After chunk 3, `nitems=3000`, `pbar.update(3000)` -> bar jumps to `3000+3000=6000` (should be 3000). The displayed progress diverges further from reality with every chunk; it never reflects real work done. Purely cosmetic — doesn't affect the accumulated corpus statistics themselves.
- **Suggested fix**: `pbar.update(chunksize)` (the per-chunk increment), not `pbar.update(nitems)`.

### [Low] Codepoint packing assumes 4-byte native `int` and little-endian byte order — src/pyutilz/text/similarity.py:577-603
- **Category**: efficiency / edge-case (portability)
- **Problem**: `_pack_words` builds `array.array("i", w.encode("utf-32-le"))` then does `np.frombuffer(a, dtype=np.int32)`. `array`'s `"i"` typecode is the platform's native C `int`, which per the `array` module docs is only guaranteed to be *at least* 2 bytes — in practice 4 bytes on essentially every mainstream platform today, but not a documented guarantee — and `array.array(typecode, bytes)` interprets the raw bytes in **native byte order**, while the source bytes were explicitly produced as **little-endian** (`"utf-32-le"`). On a big-endian platform (rare today, but real: some POWER/older-ARM/embedded/mainframe configurations still run big-endian Python) this would silently byte-swap every codepoint above `0xFF`, corrupting non-ASCII comparisons rather than raising.
- **Failure scenario**: Not reproducible on this (little-endian x64) sandbox; flagged as a latent, currently-dormant correctness gap rather than something independently exercised here. On a hypothetical big-endian host, `sentences_similarity_numba(["café"], ["café"])` would compare byte-swapped codepoints for `é` (U+00E9) and non-ASCII input would silently mismatch rather than compare correctly, while the pure-Python fallback would be unaffected.
- **Suggested fix**: Build the int32 buffer with an explicit, endianness-independent method (e.g. `np.frombuffer(w.encode("utf-32-le"), dtype="<u4").astype(np.int32)`, forcing the little-endian dtype rather than relying on `array`'s native-order reinterpretation) or document the little-endian-platform assumption.

### [Low] `naive_entropy_rate("")` returns `-0.0` silently instead of raising or being documented — src/pyutilz/text/strings/textentropy.py:114-123
- **Category**: edge-case / docs
- **Problem**: For an empty string, `np.unique(np.array([]))` returns an empty counts array; `p = cnt / np.sum(cnt)` divides an empty array by `0` — numpy does not raise/warn here (there are zero elements to actually divide), so `-np.sum(p * np.log2(p))` silently evaluates to `-0.0`. Verified with `warnings.simplefilter("error")` active — no warning fires either.
- **Failure scenario**: `naive_entropy_rate("")` returns `-0.0` rather than `0.0`, `None`, or a raised error — "entropy of nothing" is a genuinely undefined quantity, and a caller aggregating/comparing entropy scores across many short strings (some possibly empty after upstream cleaning) would silently mix in this sentinel-like value with no signal that the input was degenerate.
- **Suggested fix**: Add an explicit `if not a: return 0.0` (or raise) with a one-line docstring note, so the behavior is a deliberate choice rather than an accident of numpy's zero-length-array semantics.

## Things done well

- **Extensive, honest, self-critical benchmarking documented in comments.** `similarity.py`'s comments around `_SORTED_MATCH_THRESHOLD` include a full measured timing table (N=5..1000, scan vs. sort, with the crossover point) and openly document that an *earlier* attempt at the sort-based greedy matcher was WRONG (a quantized composite key silently collapsed real float64 ULP differences into false ties on 8/480 random trials) before landing on the current exact-bit-pattern-reinterpretation fix. This is exactly the kind of adversarial self-testing this audit is looking for.
- **UTF-32 (not UTF-16) codepoint packing** in `_pack_words` correctly treats astral-plane characters (e.g. emoji outside the BMP) as a single codepoint, matching CPython's `len()` semantics and avoiding the classic surrogate-pair-splitting bug a naive UTF-16-based packer would have. Verified empirically: mixed emoji comparisons (`"😀😀😀"` vs `"😀😀😁"`) match exactly between the pure-Python and numba paths.
- **The `_logproxy` forwarding pattern correctly achieves its one stated goal**: `mock.patch.object(pyutilz.text.strings, "logger", fake)`-style monkeypatching transparently affects every split-out submodule's `logger.warning(...)` calls, verified directly.
- **Empty/degenerate-input handling across the numba batch API is thorough and correct**: empty candidate lists, candidates containing empty-word sublists, empty queries, and `parallel=True` vs `parallel=False` vs `SentenceSimilarityIndex` were all verified to agree with each other and with the pure-Python reference across dozens of randomized cases with zero mismatches.
- **Measured, reproducible speedups** matching the documented "5-20x" claim: 2.7x (N=5) up to 10.8x (N=10) and 7.6x on a deliberately length-skewed pair, with results matching bit-for-bit against the pure-Python reference in every case tested.

## Investigated, not an issue

- **`from pyutilz.text import *` despite `text/__init__.py` never explicitly importing `strings`/`tokenizers`/`similarity`/`humanizer`.** Suspected this would raise `AttributeError` since `__init__.py`'s body is just `__all__ = [...]` with no corresponding imports. Verified this actually works: Python's star-import machinery falls back to importing `__all__` entries as submodules when they aren't already bound attributes. A plain `import pyutilz.text` does *not* trigger this (confirmed separately), only `from pyutilz.text import *` does — consistent with documented Python behavior.
- **Pickling `_FacadeLoggerProxy`.** Initially suspected this might fail (no `__getstate__`/`__reduce__`, `__slots__ = ()`), but verified `pickle.dumps`/`pickle.loads` round-trips cleanly to a functionally-equivalent fresh proxy, since the class carries no per-instance state and dunder methods used by the pickle protocol (`__reduce_ex__` etc.) are resolved via the type/MRO, not `__getattr__`.
- **Astral-plane Unicode (general, non-surrogate) correctness in the numba similarity kernels.** Fully correct in all tested cases; only lone/unpaired surrogate codepoints (a much narrower, genuinely pathological case — see the Medium finding above) trigger a crash.
- **The large-N sort-based greedy matcher's correctness at its threshold boundary.** Ran a direct differential test at N=560 (just above the 550×550 `_SORTED_MATCH_THRESHOLD`) between `sentences_similarity_numba` (which dispatches to the sorted path here) and the pure-Python `sentences_similarity` — results matched to `0.0` absolute difference.
