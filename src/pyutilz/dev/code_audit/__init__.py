"""AST-based code-audit scanners for recurring Python production-bug classes.

Library + CLI for the kinds of static checks we kept rebuilding from
throwaway scripts inside subagent runs. Promoted into pyutilz so they
stop living in D:/Temp and can run against any project (mlframe,
pyutilz itself, downstream applications) with a single import or
``python -m pyutilz.dev.code_audit ./src``.

The catalogue of record is GENERATED from the registry, not this docstring:
``python -m pyutilz.dev.code_audit --list-checks`` (or ``describe_scanners()``)
lists every registered check with its one-line summary and marks the ones
``run_all`` skips unless asked for explicitly. Prefer it -- the prose below is
background on the older checks and is not, and cannot be, complete.

Checks described below (each a top-level scanner function returning a
list[Finding]):

- ``scan_mutable_defaults``: ``def f(x=[])`` / ``=={}`` / ``=set()`` /
  ``=dict()`` / ``=list()``. Flagged at ``severity="P0"`` only when the
  parameter is observably mutated in the body (``append``, ``extend``,
  ``setitem``, ``add``, ``update``, ``pop``, ``clear``); otherwise
  ``Low`` (idiomatic-but-questionable). Catches the classic
  "options dict default accumulates state across calls" bug.

- ``scan_late_binding_closures``: ``lambda`` / nested ``def`` inside a
  ``for`` loop that references the loop variable AND escapes the
  iteration (stored in a list/dict, passed as callback, registered).
  Classic "all closures see the FINAL loop value" bug.

- ``scan_default_via_or_trap``: ``x = arg or DEFAULT`` shape where the
  RHS default is a non-trivial literal that would silently clobber
  ``0`` / ``""`` / ``[]`` / ``{}`` if the caller passed one of those
  as a legitimate sentinel. Detected via AST ``BoolOp(Or, ...)`` whose right operand looks like a
  DEFAULT VALUE -- a constant, a call, or an ALL-CAPS constant name. A lowercase name, attribute
  or subscript on the right is a read of another SOURCE (``alias.asname or alias.name``), not a
  default, and is not reported; nor is regex capture-group alternation, the same key read off a
  second mapping, or an ``or`` whose operands are all boolean-valued. Heuristic; not 100% recall.

- ``scan_broad_except_swallows``: ``except Exception:`` / ``except:``
  followed by a body of ``pass`` / ``continue`` / a bare ``return``
  WITHOUT a ``logger.warning``/``logger.error`` call. Catches the
  "silent data drop" pattern (a column failed to encode, the row got
  skipped, the operator sees fewer features and no log line).

- ``scan_log_only_except``: complements the above -- an ``except``
  handler that DOES log but never appends to a caller-visible
  ``errors``/``validation_errors``-style collection and never re-raises.
  The operator sees a log line; a caller reading only the return value
  (or a persisted "was this successful" flag derived from it) has no
  way to learn anything failed. Only applies within files that use that
  collect-and-report naming convention somewhere; the attribute-name set
  is configurable via the ``escalation_attrs`` kwarg.

- ``scan_sql_limit_without_order_by``: a SQL ``SELECT ... LIMIT`` string
  literal with no ``ORDER BY`` -- which rows survive the cap is
  arbitrary DB physical order, not reproducible across runs. ``LIMIT 1``
  is exempted (the ubiquitous "fetch a single row" idiom).

- ``scan_sql_offset_pagination``: a SQL literal combining ``LIMIT`` and
  ``OFFSET``. Advisory -- flags the pattern so a reviewer can confirm
  the filtered result set is stable across batches; a loop that both
  SELECTs via ``WHERE flag IS NULL`` and UPDATEs that flag between
  batches silently skips rows under OFFSET (keyset pagination on a
  stable id column is immune).

- ``scan_dead_cli_flags``: an ``argparse`` ``add_argument(...)`` whose
  bound attribute (``args.<name>``) is never referenced anywhere in the
  scanned tree -- either dead code or a flag that silently does nothing.

- ``scan_sql_migration_idempotency``: raw ``.sql`` file scan (not
  Python) for statements that fail on a second run instead of no-op'ing
  -- ``DROP CONSTRAINT``/``DROP COLUMN``/``DROP TABLE`` without
  ``IF EXISTS``, ``ADD COLUMN`` without ``IF NOT EXISTS``, and
  ``ADD PRIMARY KEY`` with no ``DO $$ ... END $$`` guard block. A
  migration re-run (deploy retry, multi-environment rollout, manual
  re-apply after a partial failure) is a routine operational event, not
  an edge case.

- ``scan_duplicate_function_body``: two or more functions/methods anywhere in the tree
  with a STRUCTURALLY IDENTICAL body (name, file, docstring, and decorators ignored --
  matching is on the AST-normalized statement sequence only; dunder methods like
  ``__init__``/``__getstate__``/``__eq__`` are always exempt, since their bodies
  legitimately converge on the same shape by protocol design). A helper's exact logic
  copy-pasted into N files instead of living once in a shared module is a drift risk:
  a later bug fix or edge-case addition updates one copy and silently misses the rest.
  ``Finding.check`` is ``"duplicate_function_body"``.

- ``scan_near_duplicate_function_body``: two or more functions/methods with a
  NEARLY (not exactly) identical body -- ``difflib.SequenceMatcher`` ratio
  ``>= 0.99`` over the same structural AST normalization
  ``scan_duplicate_function_body`` uses, with exact matches excluded (that's
  the other scanner's job). Catches copy-paste that then drifted by one line,
  one changed constant, or one extra guard clause, which a byte-exact check
  cannot see. ``Finding.check`` is ``"near_duplicate_function_body"``,
  severity ``Low`` -- a human should judge unify-vs-independent before acting.
  Default-on (2026-08-02): confirmed useful in the wild against a real
  ~2000-function corpus (found and drove real fixes, not just noise), and
  severity ``Low`` already signals "review before acting" without needing
  an opt-in gate on top of that.

- ``scan_duplicate_conditions``: three copy-paste-typo shapes, all
  emitted under this one scanner (distinct ``Finding.check`` values --
  ``"duplicate_condition"`` for the first two, ``"duplicate_dict_key"``
  for the third):
  (1) the same operand repeated inside one ``and``/``or`` expression
  (``x.endswith('a') or x.endswith('a')``);
  (2) an ``elif`` whose test is identical to a preceding branch's test
  (the later branch is dead);
  (3) a dict literal with the same constant key twice -- Python keeps
  only the LAST value, silently discarding whatever the earlier entry
  encoded. All three were confirmed in the wild during a large-scale
  triage, including a correction-table dict that silently lost a rule.

- ``scan_missed_await``: a statement-level call to a same-module
  ``async def`` whose coroutine is discarded -- the function body never
  runs, and Python only emits an easily-missed RuntimeWarning. Precision
  restrictions (statement-level only, same-module plain-name callees,
  no locally-rebound names) were each derived from a concrete false
  positive during corpus validation; the shipped heuristic had zero
  false positives across three repos.

- ``scan_redundant_test_fit_calls``: within a ``test_*.py`` file, an
  identical call to an underscore-prefixed local helper (``_build_x(seed)``,
  ``_fit_y(X, y, seed)``) made from 2+ different ``test_*`` functions --
  since the call is deterministic, every occurrence after the first
  recomputes the same result. Confirmed in the wild as a 7-14x wall-clock
  cost on real MRMR biz_value test suites (mlframe, 2026-07-16), where
  several sibling test functions each independently re-ran the same
  expensive model fit to check a different assertion on the identical
  result. A helper already decorated with ``@cache``/``@lru_cache`` is
  skipped (already fixed).

- ``scan_undeclared_imports``: a module-top-level third-party import whose
  package belongs to a DIFFERENT pyproject.toml extras group than the
  importing file's own domain -- catches an import reachable only because
  another feature's extras group happens to also be installed, so a
  minimal install of just this file's own domain breaks at import time.

- ``scan_vacuous_assertions``: a tautological ``assert`` in a ``test_*.py``
  file -- a bare ``assert True``, or an ``assert (X is None or X == {} or
  ... )`` whose every OR-branch just restates "any value of X is fine",
  so the assertion can never fail regardless of what the code under test
  actually returns.

- ``scan_additive_epsilon_denominator``: a division guarded by ADDING a
  small constant to its denominator. The pad is harmless only while the
  denominator's natural scale stays far above it -- an assumption about the
  data, not the code -- and it does not guard the division, it replaces the
  answer with a plausible finite value for every input.

- ``scan_non_neutral_except_fallback``: an ``except`` handler that returns or
  assigns a non-neutral literal without logging above debug level. The
  substituted value is indistinguishable downstream from a real result, and
  is often the exact value that DISABLES the check consuming it (``0.0`` for
  a max error, ``True`` for a permission guard, ``-inf`` for a minimised
  score). Complements ``scan_log_only_except``, which asks instead whether a
  handler escalates what it logged.

- ``scan_nondiscriminating_test_functions``: a ``test_*`` function that cannot
  fail for the reason it names -- no assertion at all, every assertion behind
  an ``if``, ``except AssertionError: pass``, an ``if`` body of ``pass``, or an
  imperative ``pytest.xfail(...)`` discarding the measurement just taken.
  Where ``scan_vacuous_assertions`` judges the assertion EXPRESSION, this one
  judges structure: whether any assertion is reachable and unconditional.

- ``scan_locals_globals_as_output``: ``locals()``/``globals()`` passed as
  an argument whose name suggests it's meant to be an output/mutation
  channel (``object=``, ``out=``, ``target=``, ...) -- both are
  disconnected snapshots inside a normal function, so the intended
  mutation never reaches the caller.

- ``scan_missing_network_timeout``: an HTTP call (``requests``/``httpx``/
  ``urllib``/...) with no ``timeout=`` and no pre-configured
  session/client timeout -- a hung remote peer blocks the call forever.

- ``scan_parameter_aliasing_mutation``: ``local = param`` (a bare rebind,
  no ``.copy()``) followed by an in-place mutation of ``local`` in the
  same function -- the mutation silently reaches the CALLER's object
  through the un-copied parameter reference. Not reported: a scalar loop counter (``i += 1`` on a
  name used as an index or a loop bound), an alias and a mutation in mutually exclusive branch
  arms, and a parameter NAMED as a caller-owned output buffer (``out``, ``*_shared``,
  ``*_buffer``), where writing through to the caller is the contract.

- ``scan_sync_blocking_in_async``: a synchronous blocking call
  (``requests.*``, bare ``httpx.get/post/...``, ``time.sleep``,
  ``subprocess.run/call/...``) inside an ``async def`` body with no
  ``await``/``asyncio.to_thread`` wrapper -- stalls the WHOLE event loop,
  not just the current task.

- ``scan_retry_loops``: two shapes of risky retry loop -- a ``while
  True:`` retry whose ``except`` handler has no ``sleep()`` (a busy-loop
  burning 100% CPU on a persistent failure), and a sleep-backed ``while
  True:`` retry with no visible ``break`` (advisory: confirm "retry
  forever" is deliberate).

- ``scan_duplicate_module_docstring``: two consecutive bare string-literal
  statements at the top of a module -- only the first becomes
  ``__doc__``; the second is silently discarded at runtime.

- ``scan_unraised_exceptions``: a custom exception class (repo-wide,
  cross-file) that is never ``raise``d anywhere in the scanned tree --
  its documented error-signaling contract silently never fires.

- ``scan_credential_shaped_log_args``: a ``logger.<level>(...)`` call
  whose arguments include a credential-shaped name (``proxy``,
  ``password``, ``token``, ``secret``, ``api_key``, ``cookie``, ...) with
  no redaction hint nearby -- a real, if noisy, security-adjacent signal.

- ``scan_docstring_args_completeness``: a function whose docstring HAS a
  Google-style ``Args:`` section that omits one or more of the function's
  actual parameters -- a caller reading the docstring has no idea an
  undocumented parameter exists.

- ``scan_return_annotation_mismatch``: a function declared with a concrete
  scalar return annotation (``-> float``/``-> int``/...) that has a
  ``return`` statement returning a container literal or a bare
  ``return``/``return None`` -- the declared type doesn't match what the
  function actually hands back on that path.

- ``scan_sql_aggregate_before_cast``: a SQL ``MIN``/``MAX`` aggregate
  whose argument contains a JSON text extraction operator with no
  ``::type`` cast anywhere in the argument -- Postgres compares the
  extracted value as TEXT, lexicographically rather than numerically,
  so aggregating an un-cast numeric-looking string silently picks the
  wrong row (e.g. treats ``"9"`` as greater than ``"10"``).

- ``scan_getattr_unknown_attribute``: ``getattr(obj, "name", default)`` where ``name`` is an
  attribute of no class in the tree, so the call can only ever return the default. The RECEIVER
  must be an object the tree DEFINES (``self`` in an in-tree class, a name annotated with one, a
  variable assigned from one, or a self-referencing module): the premise says nothing about a
  stdlib, third-party or dynamically-configured object. Pass ``require_known_receiver=False`` for
  the wider, much noisier pre-2026-09-03 behaviour.
- ``scan_getattr_literal_on_known_dataclass``: ``getattr(obj, "name", default)`` where ``obj`` is
  locally inferable as an instance of a SPECIFIC ``@dataclass`` in the tree and ``name`` is not one
  of that class's own fields -- narrower and stronger than ``scan_getattr_unknown_attribute``, which
  only catches a name that is an attribute of NOTHING anywhere; this one catches a name that is a
  real field of some OTHER class but not the one actually being read.
- ``scan_locals_get_fragile_lookup``: ``locals().get(...)``/
  ``globals().get(...)`` as a stand-in for a direct name reference --
  fragile to a rename (the target is a magic string, not a symbol a
  refactor tool or grep-based rename can find) and, for ``locals()``,
  frequently wrong regardless (a nested function's ``locals()`` doesn't
  see an enclosing scope's later-defined name the way a closure would).

- ``scan_shielded_resource_release_race``: ``asyncio.shield(closure())``
  where the shielded closure reads a name the ENCLOSING function's own
  ``finally:`` also hands to a release-shaped call, outside the shielded
  closure -- ``asyncio.shield`` protects the shielded task from the
  caller's cancellation, not the caller's local variables, so a resource
  release in the outer ``finally:`` can race a still-running shielded
  task on the exact same pooled resource.

- ``scan_duplicate_credential_regex``: a ``re.compile(...)`` call whose
  pattern matches a credential-shaped keyword (password/token/secret/
  api_key/credential/authorization/bearer), defined outside the project's
  designated canonical scrubber module(s) (``canonical_module_rel_paths``)
  -- the same "redact a secret before logging it" problem solved
  independently, non-identically, in multiple places, with coverage that
  silently drifts between the copies.

- ``scan_asymmetric_resource_guard``: within one class, an operation-shape
  (a dotted call like ``conn.cursor`` or ``self._db.execute``) that's
  wrapped in a guarding ``with``/``async with`` block
  (``.transaction()``/``.atomic()``/``.begin()``, or a bare
  ``self._lock``-shaped context manager) in at least one method but
  performed UNGUARDED in a SIBLING method of the same class -- the class's
  own code already demonstrates the correct pattern in one place and
  omits it in another, the strongest signal available without
  understanding what the operation actually does.
- ``scan_stale_test_spy_arity``: a test's hand-written spy/recorder
  function, passed as ``patch(..., side_effect=<local def>)`` in place of
  a real production function, whose own positional-arg arity has fallen
  behind that production function's real call sites -- the production
  function grew a new parameter (safe for real callers, since it has a
  default) but the spy's fixed ``def spy(a, b, c): ...`` didn't grow with
  it, so it raises ``TypeError`` the moment the new argument is actually
  passed. ``MagicMock`` tolerates any arity; a hand-written spy doesn't.

- ``scan_unthrottled_hot_loop_log``: a ``log.warning``/``log.error`` call
  inside a ``for``/``while`` loop with no apparent rate-limiting guard (no
  enclosing ``if`` that calls a throttle-shaped helper or uses a modulo
  counter idiom) -- fine in isolation, but floods logs the moment every
  item in a large batch hits the same condition (e.g. a systemic upstream
  outage), exactly when an operator most needs signal, not noise.

- ``scan_possibly_dead_import``: a module-level import never referenced
  as a bare name in its own file AND never accessed as ``<anything>.name``
  anywhere in the whole scanned corpus -- a genuinely unused import, not
  a facade re-export consumed indirectly elsewhere. The whole-corpus
  attribute-access cross-check exists because a naive same-file-only scan
  has a real false-positive class: 9 of 12 audit-suspected "dead"
  re-exports in one real codebase turned out to be live, consumed via
  ``import module as m; m.<name>`` in test files.

- ``scan_unpicklable_resource_state``: a class's ``__init__`` assigns a
  live, process-bound, unpicklable resource (``threading.Lock``/``RLock``/
  ``Event``/..., a ``Thread``/``Process``/``Pool``, an ``open()`` file
  handle, or a CUDA stream/event) directly to ``self.<attr>`` with no
  ``__getstate__`` defined on the class to drop it before a pickle
  round-trip -- ``pickle.dumps(instance)`` raises ``TypeError`` the first
  time anything upstream caches, checkpoints, or ships the instance
  across a process boundary. Generalizes a bug pattern fixed by hand
  twice independently in the same downstream codebase before this
  scanner existed (mlframe's ``FeatureCache._lock`` and
  ``training/neural/ranker.py``'s trainer_/CUDA-tensor exclusion).

- ``scan_bare_except``: ``except:`` (bare) or ``except BaseException:`` --
  swallows ``KeyboardInterrupt``/``SystemExit``/``MemoryError`` and masks
  real bugs. A bare re-raise inside the handler is not flagged.

- ``scan_console_unicode``: ``print(...)``/``logger.X(...)`` first-arg
  string literal contains a non-ASCII character -- crashes with
  ``UnicodeEncodeError`` on Windows cp1251/cp1252 stdout.

- ``scan_mojibake``: a source line contains a round-trip-detectable
  mojibake run (UTF-8 bytes misread as CP1251 and re-saved -- a lossy
  clipboard/terminal round-trip is the usual cause). Detected via a
  double round-trip (encode cp1251, decode utf-8) succeeding into
  different, non-empty text -- low false-positive by construction.

- ``scan_resource_handle_safety``: ``open()``/``tempfile.NamedTemporaryFile()``/
  ``tempfile.TemporaryFile()``/``tempfile.SpooledTemporaryFile()``/
  ``subprocess.Popen()`` used outside a ``with`` block -- handle
  close/cleanup on exception is not guaranteed.

- ``scan_todo_hygiene``: an un-attributed ``TODO``/``FIXME``/``XXX``/``HACK``
  comment (no assignee in parens, ISO date, or ``@mention``). Thin wrapper
  over ``pyutilz.dev.meta_test_utils.scan_todo_markers``.

- ``scan_import_cycles``: a multi-node cycle in the package's internal
  top-level import graph (Tarjan SCC over an AST-built dependency graph,
  lazy/function-body imports excluded). Accepts optional ``package_name``
  (defaults to ``root.name``) and ``deferred_cycles`` (a set of reviewed,
  runtime-safe cycles to skip) keyword arguments.

- ``scan_readonly_to_numpy_mutation``: ``np.fill_diagonal(X, ...)`` /
  ``np.copyto(X, ...)`` where ``X`` was assigned, earlier in the same
  function, from an uncopied ``<pandas obj>.to_numpy()`` call. Under
  pandas Copy-on-Write (default from pandas 3.0, opt-in in 2.x),
  ``to_numpy()`` can return a read-only view -- the in-place mutation then
  raises ``ValueError: underlying array is read-only`` at runtime, silent
  on a non-CoW install and real on a CoW one. Fix: ``.to_numpy(copy=True)``.
  Generalizes a real bug (2026-07-31, mlframe's ``dataset_diagnostics``).

- ``scan_tautological_is_not_none_only_tests``: a ``test_*`` function whose
  ONLY unconditional assertion(s) are a bare ``X is not None``/``X != None``
  check -- confirms something came back, never that it's the RIGHT thing.
  Narrower than a raw ``is not None`` grep: a function with a stronger
  assertion alongside the weak one is not flagged.

- ``scan_except_skip_masks_call_under_test``: a ``try:`` block in a test
  file that does more than import a module (calls the function/class under
  test, or other real work) whose ``except`` handler calls
  ``pytest.skip(...)`` -- a genuine API break gets silently reclassified as
  an environment-limitation skip. The legitimate optional-dependency-guard
  shape (try body is ONLY import statements) is exempted.

- ``scan_uncurated_star_exports``: an ``__init__.py`` doing
  ``from .submodule import *`` where neither it nor the target submodule
  defines ``__all__`` -- every public name in the submodule silently joins
  the package's public surface, and same-named helpers from two
  star-imported submodules can silently shadow one another.

- ``scan_per_call_state_on_shared_instance`` (OPT_IN_ONLY, warn): an
  attribute written on ``self`` during an in-flight ``async def`` of a
  class whose instances are shared between callers, with no lock held and
  a different method reading it back (or a ``last_*``/``current_*`` name).
  Two concurrent calls interleave and the value is attributed to the WRONG
  request -- silently, since nothing raises. Sharedness is read off the
  tree (instance stored in a module-level container, built by an
  lru_cache'd factory, or the class name resolved from a registry) and
  propagated to in-tree base classes, because the attributes usually live
  on the base while only the concrete subclass is registered.

- ``scan_uncached_constant_cost_probe`` (OPT_IN_ONLY, warn): a function
  taking no parameters (or only defaulted ones) that shells out via
  ``subprocess``, builds a ``ctypes.WinDLL``/``CDLL``, calls
  ``os.makedirs``, ``shutil.which`` or ``importlib.import_module`` on
  EVERY call, with no caching decorator and no module-level memo. Its
  answer cannot change within a process, so a hot dispatch path pays a
  constant cost per decision. Warn-only on purpose: a probe that must be
  re-taken (a liveness check, a config reload) is indistinguishable from
  one that must not, so this is a triage list rather than a verdict.

The next block came out of a two-round adversarial audit of a research
codebase, where a handful of shapes accounted for most of the findings.
All of them are in ``OPT_IN_ONLY``: several need project configuration to
say anything, and the rest report accumulated design debt, which is a
decision a project takes rather than inherits on upgrade. Name them in
``checks=`` to run them.

- ``scan_dead_public_callables``: a public callable nothing outside the
  test suite can reach, computed by reachability rather than grep --
  a function called only by another dead function is dead too, and a
  callable behind a live module IMPORT is not thereby live. Takes
  ``consumer_roots`` for trees that count as call sites without being
  audited (demos, a separate bench tree).

- ``scan_vacuous_empty_pattern_match``: ``all(... for x in NAME)`` where
  nothing establishes ``NAME`` is non-empty -- an empty pattern then
  matches every input. ``any()`` is not flagged; it fails closed.

- ``scan_tautological_guards``: an ``and`` combining a threshold on some
  target with an identity/equality pin of that same target to one value,
  so only that value can reach the threshold and the guard reduces to
  the pin.

- ``scan_table_header_row_drift``: a ``csv.DictWriter`` whose literal
  ``fieldnames`` and literal written row disagree (the missing key
  becomes ``restval`` silently), and a function formatting a wide table
  from two unpaired f-strings of differing field counts.

- ``scan_record_field_flow``: a record field with one side of its
  contract. ``field_read_never_written`` (P1) is a ``.get(k, default)``
  or ``getattr(o, k, default)`` for a key nothing writes -- the default
  wins on every run. Restricted to NEAR MISSES by default, because
  "this dict came from outside" is not statically decidable.
  ``field_written_never_read`` (P2) is off by default; it measured 732
  hits on the source project, which is an allowlist longer than the rule.

- ``scan_unenforced_docstring_invariants``: a docstring stating an
  absolute invariant (``never``, ``must not``, ``deliberately not``)
  whose symbol no test in ``test_roots`` names. Deliberately a weak bar:
  the target is the claim NOTHING references.

- ``scan_partial_guard_across_siblings``: a family of ``prefix_*``
  functions over the same first parameter where a majority guard it and
  one member does not -- a fix that stopped one call short.

- ``scan_inconsistent_filter``: consumers of a collection that skip an
  exclusion rule its other consumers apply. Needs ``filter_pairs``;
  silent until configured.

- ``scan_domain_vocabulary_leak``: a symbol the project DECLARES domain-neutral
  (a candidate for a later extraction into a shared library) whose source span
  names a term from the current domain's vocabulary -- in an identifier, a
  string, a docstring or a comment. Boundary and vocabulary are both caller
  supplied, so the check is silent until configured. Symbol granularity, not
  module: before the split, the neutral concept and the domain concept live in
  the same file by definition, and a module-level rule would report every such
  file and say nothing. A companion ``boundary_symbol_missing`` finding fires
  when a declared symbol no longer exists, so a rename cannot silently empty
  the boundary and leave the check passing by vacuity.

- ``scan_regex_integer_parse``: a bare digit-class regex over free text converted
  to a number -- "4.10" arrives as 4 and "2,054" as 2, silently.

- ``scan_thresholds_below_documented_result``: a ``test_*`` function
  whose docstring states "7 of 8" while its assertion accepts ``>= 6``.

- ``scan_docstring_numbers_moved_to_config`` (OPT-IN): a docstring still
  spelling out threshold values the body no longer contains, because they
  were made configurable and the prose was not updated. The prose is then
  the only surviving record of a number that does not exist. Opt-in: on
  four repos it produced three hits and all three were false, so it is a
  tool to point at a suspect file, not a ratchet.

One member of this package is deliberately NOT a scanner and NOT in
``get_scanners()``: ``field_text_agreement``. It checks a defect shape the other
rules cannot see from source alone - a structured field that duplicates
information also present in free text, where the two are never compared -
and it does that over RECORDS at runtime, not over ``.py`` files. Every
registered scanner honours ``(root: Path, exclude_dirs) -> list[Finding]``,
and a record-level checker cannot; registering it would break ``run_all``
rather than extend it. It lives here because the mechanism is general and
this is where the shared audit rules are, it renders its hits as
``Finding`` objects so a caller can report them beside source findings, and
because it is unregistered it cannot alter any project's committed baseline
on upgrade - which is the reason ``OPT_IN_ONLY`` exists in the first place.
The CUE VOCABULARIES are domain knowledge and stay with the caller.

Each scanner is a pure function: ``(root_path: Path) -> list[Finding]``.
The CLI ``__main__`` block wraps them with argparse and emits markdown
or JSON.

Severity tags follow the rest of the audit suite:
    P0 - silent data corruption / cross-call leak at top-level boundary
    P1 - silent leak within a suite / session / per-target loop
    P2 - degraded diagnostics / metrics-reporting bias
    Low - cosmetic / idiomatic-but-questionable

Per ``feedback_no_padding_parametric_pins``: scanners never manufacture
findings. If the codebase is clean, the result list is empty.

Per ``feedback_save_useful_scripts_in_package``: this file lives here
(in the package) precisely so subagent runs stop dropping ad-hoc AST
scanners in D:/Temp. The CLI exposes the same checks the agents would
otherwise hand-write.
"""

from __future__ import annotations

# Preserve the original flat-module import surface. These names were
# importable as ``from pyutilz.dev.code_audit import ast`` etc. before the
# split into a subpackage; keep them re-exported so no caller breaks.
import ast  # re-exported facade name (`pyutilz.dev.code_audit.ast`), part of the historic flat-module public surface
import json
import sys  # re-exported facade name (`pyutilz.dev.code_audit.sys`), part of the historic flat-module public surface
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ._base import Finding
from .mutable_defaults import scan_mutable_defaults, scan_parameter_aliasing_mutation
from .closures import scan_late_binding_closures
from .default_via_or import scan_default_via_or_trap
from .broad_except import scan_broad_except_swallows
from .nan_equality import scan_nan_equality
from .mutation_during_iteration import scan_mutation_during_iteration
from .sql_lint import scan_sql_limit_without_order_by, scan_sql_offset_pagination, scan_sql_aggregate_before_cast
from .dead_cli_flags import scan_dead_cli_flags
from .silent_escalation import scan_log_only_except, DEFAULT_ESCALATION_ATTRS
from .sql_migrations import scan_sql_migration_idempotency
from .duplicate_conditions import scan_duplicate_conditions
from .duplicate_function_body import scan_duplicate_function_body
from .near_duplicate_function_body import scan_near_duplicate_function_body
from .missed_await import scan_missed_await, scan_sync_blocking_in_async
from .redundant_test_fit import scan_redundant_test_fit_calls
from .undeclared_imports import scan_undeclared_imports
from .vacuous_assertions import scan_vacuous_assertions, scan_tautological_is_not_none_only_tests
from .additive_epsilon_denominator import scan_additive_epsilon_denominator
from .non_neutral_except_fallback import scan_non_neutral_except_fallback
from .nondiscriminating_test import scan_nondiscriminating_test_functions
from .locals_globals_output import scan_locals_globals_as_output
from .network_timeout import scan_missing_network_timeout
from .retry_loops import scan_retry_loops
from .module_docstring import scan_duplicate_module_docstring
from .unraised_exceptions import scan_unraised_exceptions
from .credential_logging import scan_credential_shaped_log_args
from .docstring_args import scan_docstring_args_completeness
from .return_annotation import scan_return_annotation_mismatch
from .getattr_unknown_attribute import scan_getattr_unknown_attribute
from .getattr_literal_on_known_dataclass import scan_getattr_literal_on_known_dataclass
from .locals_get import scan_locals_get_fragile_lookup
from .shielded_resource_release import scan_shielded_resource_release_race
from .duplicate_credential_regex import scan_duplicate_credential_regex
from .asymmetric_resource_guard import scan_asymmetric_resource_guard
from .spy_arity import scan_stale_test_spy_arity
from .log_throttle import scan_unthrottled_hot_loop_log
from .assert_in_loop import scan_assert_in_loop_reports_only_the_first
from .dead_import import scan_possibly_dead_import
from .reexport_patch_target import scan_reexport_patch_target
from .unpicklable_resource_state import scan_unpicklable_resource_state
from .skip_masking_except import scan_except_skip_masks_call_under_test
from .uncurated_star_export import scan_uncurated_star_exports
from .dead_wiring import scan_dead_public_callables
from .vacuous_matching import scan_vacuous_empty_pattern_match
from .tautological_guard import scan_tautological_guards
from .table_drift import scan_table_header_row_drift
from .provenance_flow import scan_record_field_flow
from .claimed_invariants import scan_unenforced_docstring_invariants
from .partial_fix import scan_partial_guard_across_siblings, scan_inconsistent_filter
from .measurement_hygiene import scan_regex_integer_parse, scan_thresholds_below_documented_result
from .domain_boundary import BoundarySymbol, scan_domain_vocabulary_leak
from .readonly_to_numpy_mutation import scan_readonly_to_numpy_mutation
from .bare_except import scan_bare_except
from .console_unicode import scan_console_unicode
from .mojibake import scan_mojibake
from .resource_handle_safety import scan_resource_handle_safety
from .todo_hygiene import scan_todo_hygiene
from .import_cycles import scan_import_cycles
from .hardcoded_test_path import scan_hardcoded_absolute_path_in_test
from .async_primitive_reinit import scan_async_primitive_reinit_per_call
from .llm_max_tokens_cap import scan_llm_call_missing_max_tokens_cap
from .per_call_state_on_shared_instance import scan_per_call_state_on_shared_instance
from .uncached_constant_cost_probe import scan_uncached_constant_cost_probe
from .field_text_agreement import (
    AGREE,
    CONTRADICT,
    KIND_OPPOSED,
    KIND_UNFILLED,
    UNCHECKABLE,
    FieldTextReport,
    FieldTextRule,
    FieldTextVerdict,
    check_all,
    check_record,
    check_records,
    cues_in_text,
)
# Every registered scanner is importable from the package: the registry is the single source of
# truth for what exists, and a meta-test pins the registry and this facade in bijection.
from .accumulator_helper_bypassed import scan_accumulator_helper_bypassed
from .asymmetric_except_siblings import scan_asymmetric_except_siblings
from .column_no_write_path import scan_column_no_write_path
from .comment_names_missing_symbol import scan_comment_names_missing_symbol, scan_comment_cites_absolute_line
from .constructor_param_overwritten import scan_constructor_param_overwritten
from .count_then_fetch_same_table import scan_count_then_fetch_same_table
from .docstring_names_a_caller_that_does_not_call import scan_docstring_names_a_caller_that_does_not_call
from .docstring_numbers_moved_to_config import scan_docstring_numbers_moved_to_config
from .effect_flag_outside_its_effect import scan_effect_flag_outside_its_effect
from .guard_decidable_from_constants import scan_guard_decidable_from_constants
from .lazy_log_assertion import scan_lazy_log_assertion
from .patch_target_is_a_reexport import scan_patch_target_is_a_reexport
from .raising_stub_swallowed import scan_raising_stub_swallowed
from .sentinel_cached_as_answer import scan_sentinel_cached_as_answer
from .sentinel_guard_mismatch import scan_sentinel_guard_mismatch
from .sibling_guard_missing import scan_sibling_guard_missing
from .source_text_assertions import scan_source_text_assertions
from .sql_selects_unread_column import scan_sql_selects_unread_column
from .sql_sibling_missing_time_bound import scan_sql_sibling_missing_time_bound
from .stats_key_coverage import scan_stats_key_coverage
from .test_asserts_against_production_constant import scan_test_asserts_against_production_constant
from .unit_suffix_mismatch import scan_unit_suffix_mismatch
from .vacuous_loop_assertion import scan_vacuous_loop_assertion
from .unreachable_import_fallback import scan_unreachable_import_fallback
# SCANNERS is deliberately NOT re-exported: the registry is private, read via get_scanners()
# and written via register_scanner() (whose duplicate-name guard direct assignment bypassed).
from .registry import OPT_IN_ONLY, describe_scanners, render_check_catalogue, run_all, register_scanner, get_scanners
from .cli import main

__all__ = [
    "Finding",
    "OPT_IN_ONLY",
    "run_all",
    "register_scanner",
    "get_scanners",
    "describe_scanners",
    "render_check_catalogue",
    "main",
    "scan_mutable_defaults",
    "scan_late_binding_closures",
    "scan_default_via_or_trap",
    "scan_broad_except_swallows",
    "scan_nan_equality",
    "scan_mutation_during_iteration",
    "scan_sql_limit_without_order_by",
    "scan_sql_offset_pagination",
    "scan_dead_cli_flags",
    "scan_log_only_except",
    "DEFAULT_ESCALATION_ATTRS",
    "scan_sql_migration_idempotency",
    "scan_duplicate_conditions",
    "scan_duplicate_function_body",
    "scan_near_duplicate_function_body",
    "scan_missed_await",
    "scan_redundant_test_fit_calls",
    "scan_undeclared_imports",
    "scan_vacuous_assertions",
    "scan_additive_epsilon_denominator",
    "scan_non_neutral_except_fallback",
    "scan_nondiscriminating_test_functions",
    "scan_locals_globals_as_output",
    "scan_missing_network_timeout",
    "scan_parameter_aliasing_mutation",
    "scan_sync_blocking_in_async",
    "scan_retry_loops",
    "scan_duplicate_module_docstring",
    "scan_unraised_exceptions",
    "scan_credential_shaped_log_args",
    "scan_docstring_args_completeness",
    "scan_return_annotation_mismatch",
    "scan_sql_aggregate_before_cast",
    "scan_getattr_unknown_attribute",
    "scan_getattr_literal_on_known_dataclass",
    "scan_locals_get_fragile_lookup",
    "scan_shielded_resource_release_race",
    "scan_duplicate_credential_regex",
    "scan_asymmetric_resource_guard",
    "scan_stale_test_spy_arity",
    "scan_unthrottled_hot_loop_log",
    "scan_assert_in_loop_reports_only_the_first",
    "scan_possibly_dead_import",
    "scan_reexport_patch_target",
    "scan_unpicklable_resource_state",
    "scan_tautological_is_not_none_only_tests",
    "scan_except_skip_masks_call_under_test",
    "scan_uncurated_star_exports",
    "scan_dead_public_callables",
    "scan_vacuous_empty_pattern_match",
    "scan_tautological_guards",
    "scan_table_header_row_drift",
    "scan_record_field_flow",
    "scan_unenforced_docstring_invariants",
    "scan_partial_guard_across_siblings",
    "scan_inconsistent_filter",
    "scan_regex_integer_parse",
    "scan_thresholds_below_documented_result",
    "scan_domain_vocabulary_leak",
    "BoundarySymbol",
    "scan_readonly_to_numpy_mutation",
    "scan_bare_except",
    "scan_console_unicode",
    "scan_mojibake",
    "scan_resource_handle_safety",
    "scan_todo_hygiene",
    "scan_import_cycles",
    "scan_hardcoded_absolute_path_in_test",
    "scan_async_primitive_reinit_per_call",
    "scan_llm_call_missing_max_tokens_cap",
    "scan_per_call_state_on_shared_instance",
    "scan_uncached_constant_cost_probe",
    # field/text cross-check: a runtime record checker, not a source scanner - see the note above.
    "AGREE",
    "CONTRADICT",
    "UNCHECKABLE",
    "KIND_UNFILLED",
    "KIND_OPPOSED",
    "FieldTextRule",
    "FieldTextVerdict",
    "FieldTextReport",
    "cues_in_text",
    "check_record",
    "check_records",
    "check_all",
    # Registered scanners that had no public import; kept in bijection with the registry by
    # tests/test_code_audit.py::test_registry_and_facade_are_in_bijection.
    "scan_accumulator_helper_bypassed",
    "scan_asymmetric_except_siblings",
    "scan_column_no_write_path",
    "scan_comment_cites_absolute_line",
    "scan_comment_names_missing_symbol",
    "scan_constructor_param_overwritten",
    "scan_count_then_fetch_same_table",
    "scan_docstring_names_a_caller_that_does_not_call",
    "scan_docstring_numbers_moved_to_config",
    "scan_effect_flag_outside_its_effect",
    "scan_guard_decidable_from_constants",
    "scan_lazy_log_assertion",
    "scan_patch_target_is_a_reexport",
    "scan_raising_stub_swallowed",
    "scan_sentinel_cached_as_answer",
    "scan_sentinel_guard_mismatch",
    "scan_sibling_guard_missing",
    "scan_source_text_assertions",
    "scan_sql_selects_unread_column",
    "scan_sql_sibling_missing_time_bound",
    "scan_stats_key_coverage",
    "scan_test_asserts_against_production_constant",
    "scan_unit_suffix_mismatch",
    "scan_unreachable_import_fallback",
    "scan_vacuous_loop_assertion",
]

import types as _types

# Keep the public attribute surface identical to the pre-split flat module: drop the submodule
# names that ``from .X import ...`` bound into this namespace as a side effect. They stay importable
# on demand as ``code_audit.<submodule>`` (Python re-registers them in sys.modules); this only trims
# ``dir()``.
#
# COMPUTED, not a hand-maintained tuple: the tuple this replaces had fallen thirty submodules behind
# the package, so the invariant it asserts was simply false and the surface was arbitrary -- one
# newer scanner module visible, its neighbour not. Deliberate stdlib re-exports (``ast``, ``json``,
# ``sys``) are unaffected: they are not submodules of this package.
for _submod_name, _submod_value in list(globals().items()):
    if isinstance(_submod_value, _types.ModuleType) and _submod_value.__name__.startswith(__name__ + "."):
        globals().pop(_submod_name, None)
del _submod_name, _submod_value, _types
