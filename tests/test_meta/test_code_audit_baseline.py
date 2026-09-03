"""Meta-test: run pyutilz.dev.code_audit's own scanners against pyutilz's
own source (self-scan), baseline-driven per this directory's snapshot-style
meta-test convention (see test_no_bare_except.py / test_no_mutable_defaults.py).

Findings are baselined together (keyed by ``check::file:line``) so
pre-existing debt doesn't block adoption -- only a NEW finding fails the
test. Refresh with ``--refresh-code-audit-baseline`` after a deliberate
change, or add a narrow, commented exclusion in the ``exclude_dirs``
passed below for a confirmed false positive.

2026-09-03, five-direction audit wave (packaging / docs / core-dev-system / data-stats /
db-web-cloud-llm-text). The baseline is keyed ``check::file:line``, so the bulk of the delta is
line shift from the batch. Diffed per ``(check, file)`` pair instead of per key: 41 pairs where
the COUNT actually rose, reviewed individually below. Five scanners are NEW in this wave and had
no baseline entries at all (``credential_shaped_log_arg``, ``import_cycle``,
``sql_limit_without_order_by``, ``undeclared_import``, ``unpicklable_resource_state``), so every
one of their hits is a first sighting rather than a regression.

THREE were real and were FIXED in the code, not baselined -- all ``duplicate_function_body``:
``spy_arity._dotted_module``/``_module_aliases`` were verbatim copies of
``patch_target_is_a_reexport._module_name``/``_module_aliases``, and
``uncached_constant_cost_probe._own_nodes`` was a verbatim copy of
``readonly_to_numpy_mutation._own_nodes``. All three moved into ``_base`` as
``_dotted_module_path``/``_module_aliases``/``_own_nodes`` and are imported from there now.

The rest are reviewed false positives:

* ``credential_shaped_log_arg`` x6 -- none of the logged values is a secret. Two log the integer
  constant ``SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD``; ``web/browser.py`` logs
  ``password_input_name``, the NAME of a form field, in the "could not locate the field" error;
  the proxy ones log ``proxy_server``/``proxy_port``/``local_proxy_server``, a host and a port.
  The rule keys on the identifier's shape, which cannot tell a field name from a field value.

* ``default_via_or`` (the largest group) -- the canonical ``alias.asname or alias.name`` ast
  idiom in the wave's new scanners (``asname`` is ``str | None``, never the empty string), plus
  ordinary boolean ``or`` chains (``guarded or _guard_looks_throttled(...)``,
  ``use_print or not HAS_IPYTHON``) and string/int fallbacks where the falsy value and the default
  mean the same thing (``usage.get("prompt_cache_hit_tokens") or ... or 0``,
  ``payload.get("data") or payload.get("models") or []``, ``top.get("context_length") or
  entry.get("context_length")``, ``event.get("result") or event.get("error") or subtype``). The
  ast-alias case is now on its fifth review; every module that walks imports writes that line.

* ``default_via_or [P2] llm/openai_compat.py:158`` and ``:674`` -- ``body.get("max_tokens") or
  body.get("max_completion_tokens") or 0`` and the cache-hit equivalent: every falsy branch resolves
  to the same ``0``, so a caller-supplied 0 cannot be rewritten into anything else.

* ``non_neutral_except_fallback [P1] dev/code_audit/registry.py:421`` -- ``_is_picklable`` exists to
  answer exactly this question, so ``False`` in the handler IS the answer rather than a substituted
  measurement, and the caller acts on it by running the scanner in-process. Debug level is deliberate:
  every lambda scanner takes this path on every run.

* ``default_via_or [Low] dev/freevar_analysis.py`` x2 -- ``self.stmt_lineno or lineno`` (line 0 is
  not a valid line number, so the falsy case cannot be a real value) and ``getattr(node, "targets",
  None) or [node.target]`` (AnnAssign carries ``target``, not ``targets``): both are the ast idiom,
  not a caller-supplied default.

* ``non_neutral_except_fallback [P1] dev/freevar_analysis.py:485`` -- the handler is a CLI ``main()``
  error path: it prints the message to stderr and returns exit code 1. A process exit status is not a
  substituted measurement, and the caller is the shell.

* ``default_via_or [Low] database/db/schema.py:62`` (``or 'public'``) -- reviewed by an earlier
  agent in this wave: an empty schema name means UNSET, so rewriting it to the Postgres default
  is the intended behaviour, not a trap.

* ``duplicate_function_body`` x2 on ``llm/openai_provider.py``'s one-line pricing accessors --
  reviewed by an earlier agent in this wave: identical by coincidence. The tuples they index
  differ in arity and in meaning (input vs output vs cache-hit price), so there is nothing to
  share; merging them would couple three unrelated price tables through one accessor.

* ``getattr_unknown_attribute`` x2 -- ``'reconfigure'`` is on ``io.TextIOWrapper`` and
  ``'run_line_magic'`` on IPython's ``InteractiveShell``; both are classes defined OUTSIDE this
  tree, which is exactly the case the rule's own message says to pass via ``extra_known``. Same
  class as the 20 such entries already baselined.

* ``import_cycle`` x4 (pandaslib, polarslib, db, web.web) -- every one closes through
  ``import <parent> as _facade``, this project's documented re-export-package idiom (see the
  comment block at the top of each submodule and ``test_reexport_package_idiom.py``, which
  enforces it mechanically). Plain ``import x`` binds the partially-initialised ``sys.modules``
  entry and defers attribute lookup to call time, so it survives the cycle by construction; the
  forbidden ``from <parent> import <name>`` spelling is what the idiom exists to prevent, and no
  submodule uses it. The scanner cannot distinguish the two spellings' runtime consequences.

* ``non_neutral_except_fallback`` x2 -- both are PREDICATES, where the handler's "no" IS the
  answer: ``packages.py`` sets ``found = False`` when ``find_spec`` raises (which means "not
  installed", and errs toward installing rather than skipping), and ``hardcoded_test_path.py``
  returns ``False`` when ``relative_to`` raises (the path is outside the scan root, so it is not
  test code). Same rationale as the already-baselined ``registry._is_picklable``.

* ``resource_handle_safety`` x7 -- each has explicit, exception-safe cleanup the rule cannot see:
  ``core/image.py`` tracks ``opened_here`` and closes in the exit path, ``filesystem.py`` is a
  ``@contextlib.contextmanager`` whose ``finally`` closes the shelve, ``serialization.py`` and
  ``cache_sweeping.py`` open raw fds and close them in ``finally`` (with a documented
  fd-adoption flag for the ``os.fdopen`` failure window), and ``web/url_guard.py`` RETURNS the
  handle as its public API -- the exact case the rule's own message names as intentional.

* ``sql_limit_without_order_by`` on ``database/db/execution.py:550`` -- a generic
  ``SELECT * FROM <table> <condition> LIMIT n`` escape hatch. There is no column it could order
  by: the table is a parameter, so any ORDER BY would have to be supplied by the caller, which is
  what the ``condition`` parameter already allows. Not fixable without changing the public
  signature, and arbitrary order is the documented contract of a "give me n rows" helper.

* ``undeclared_import`` x5 -- ``dev/dashlib.py`` needs the ``[dash]`` extra and
  ``system/scheduling/prefect.py`` needs ``[prefect]``. The rule infers the required extras group
  from the file's top-level DIRECTORY, which is a layout convention this repo does not follow:
  neither module is imported by its package ``__init__`` (both are name-only ``__all__``
  entries), so ``pip install pyutilz[dev]`` imports fine and only an explicit
  ``import pyutilz.dev.dashlib`` needs dash -- which is what the ``[dash]`` extra is for.

* ``unpicklable_resource_state`` on ``core/disk_cache.py:64`` -- ``_KeyLockEntry`` is an internal
  value in ``DiskCache._key_locks``, and ``DiskCache.__getstate__`` deletes ``_key_locks`` and
  ``_key_locks_guard`` (``__setstate__`` rebuilds them). The entry is never pickled; the rule
  looks for ``__getstate__`` on the class holding the lock, not on the class that owns it.

* ``unthrottled_hot_loop_log`` x5 on ``database/db/execution.py`` -- reviewed by an earlier agent
  in this wave: a bounded retry loop, not a hot path. Each iteration is a failed DB call already
  paced by its own backoff, so a warning per attempt is the intended signal.

2026-09-03, scanner-precision wave (three checks repaired, baseline 229 -> 164). Driven by
``audits/2026-09-03_downstream-scan.md``, which measured per-check false-positive rates on two
fresh repos. Nothing was baselined to hide a finding and no severity was demoted; the drop is
entirely three rules that stopped firing on shapes proven false there.

* ``getattr_unknown_attribute`` 22 -> 0. The rule now requires a RECEIVER the scanned tree
  defines (``self`` in an in-tree class, a name annotated with one, a variable assigned from one,
  or a self-referencing module), which is what its high-precision sibling
  ``getattr_literal_on_known_dataclass`` has always required. Its premise -- "that name is an
  attribute of no class in this tree" -- says nothing about an ``ast`` node, a pandas dtype or a
  foreign model object, and 13 of 13 sampled findings downstream were exactly that, with no true
  positive observed in either repo. All 22 entries here were of that class (``'reconfigure'`` on
  ``io.TextIOWrapper``, ``'run_line_magic'`` on IPython's ``InteractiveShell``, and 20 siblings).
  Classes that bind their fields via a bulk ``setattr(self, k, v)`` loop are skipped too. The old
  behaviour stays reachable as ``require_known_receiver=False`` and is still unit-tested.

* ``default_via_or`` 90 -> 46. Four narrowings, each one a measured false-positive class: the
  right operand must look like a DEFAULT rather than a read of another source (a lowercase name,
  attribute or subscript is "try this other place" -- this is the canonical
  ``alias.asname or alias.name`` idiom, on its fifth review in this file, and it retires that
  entire class); regex capture-group alternation; the same key read off a second mapping; and
  boolean operands the scanner previously could not see through -- ``Path.exists``/``is_dir``/
  ``str.isdigit``-family methods, and bare names that are provably flags (a parameter declared
  ``bool``, or a local every assignment of which is boolean-valued). ``x or DEFAULT_TIMEOUT``
  stays flagged: the constant-naming convention is evidence of a real default.

* ``parameter_aliasing_mutation``, the suite's only P0, produced no entries here but was 11 of 11
  false downstream. Three gaps closed: ``i += 1`` on a scalar loop counter (a numeric literal
  plus index/loop-bound use, which a numpy array cannot have), an alias and a mutation in
  mutually exclusive ``if``/``try`` arms, and a parameter NAMED as a caller-owned output buffer
  (``out``, ``*_shared``, ``*_buffer``), where writing through is the contract.

Every remaining entry was re-read. No real defect was found among them: the surviving classes are
the ones already dispositioned below (``non_neutral_except_fallback`` on optional-dependency
probes and predicate handlers, ``unthrottled_hot_loop_log`` on per-column/per-setting warnings in
small-N loops, the documented resource/extras/import-cycle classes), plus the 46
``default_via_or`` survivors, which are the empty-string-means-unset (``effort or "medium"``,
``lang or "en"``), the divide-by-zero-guard (``abs(x).max() or 1.0``, ``min(n, total) or 1`` --
both carry the reasoning in-code at the site) and the resolved-credential
(``api_key or (settings.x.get_secret_value() if ... else None)``) families.

ONE genuinely new entry: ``non_neutral_except_fallback`` on ``dev/code_audit/default_via_or.py``
gains a third hit, from the ``except (ValueError, RecursionError): return False`` guard around
``ast.unparse`` in the new same-key helper. Identical in shape and rationale to the two already
reviewed on that file -- ``ast.unparse`` is absent on 3.8 and can recurse out on a pathological
expression, and "cannot decide" is the helper's answer, not a substituted measurement.

2026-09-03, code_audit infrastructure batch (audit 07 F04/F10/F199/F207/F210): three entries.

* `assert_in_loop_first_failure_only` x2, in `data/polarslib/binning.py` and
  `database/db/upsert.py`. The scanner existed, was exported and was unit-tested but had never
  been registered, so run_all() could not reach it; registering it surfaced two real pre-existing
  asserts-inside-a-loop. Baselined as pre-existing debt in modules this batch does not own.

* `non_neutral_except_fallback` on `registry.py`'s `_is_picklable`: the handler's "no" IS the
  predicate's answer, so there is no neutral value to substitute. Narrowing the except to the four
  pickling exceptions and assigning a flag instead of returning from the handler does not change
  the verdict; the rule cannot tell a predicate from a value-producing fallback.

2026-09-03 baseline refresh, reviewed individually rather than bulk-accepted. Eleven new
scanners were registered in this commit. They produced nine entries here; two of the nine were
real weaknesses in a new scanner and were fixed in it rather than baselined, leaving seven.

* `effect_flag_outside_its_effect` x2 -- FIXED, not baselined. `self._ready = True` after an
  unrelated `if self._process.stdout:` was reported purely because both statements mention
  `self`, and `res.add(str(obj))` after `if verbose: logger.info(...)` was reported because a
  logging-only guard was mistaken for the work being recorded. The scanner now drops `self`/`cls`
  as linking tokens and skips guards whose body only logs; both narrowings have a test that
  fails when the narrowing is removed.

* `comment_names_missing_symbol` x2, in `duplicate_function_body.py` and `vacuous_matching.py`.
  Both cite a helper (`_need_cuda()`, `_covers()`) from the codebase whose defect the comment
  DESCRIBES, not from this tree -- illustrative prose, not a rotted pointer. That distinction is
  not statically decidable, and it is a class this package will keep producing precisely because
  documenting defects found elsewhere is what these modules are for. Baselined rather than
  weakening the rule for every consumer, and rather than rewriting correct documentation to
  satisfy a linter.

2026-09-03, log_throttle extraction: one drain and one re-add,  on
dev/logginglib.py moving 294 -> 296. Pure line shift from the two imports the helper needs;
the code it names is unchanged.

2026-09-03, sixth batch (docstring_names_a_caller_that_does_not_call): four entries, all the
scanner reporting its own prose. `comment_names_missing_symbol` cites `_flush_rows()`, the
worked example from the codebase this rule was written for, which is the same illustrative-prose
class reviewed twice below -- documenting a defect found elsewhere is what these modules are for.
`default_via_or` is on `match.group("quoted") or match.group("bare")`, where exactly one of the
two alternatives can have matched, so the `or` is a selection and not a default.

2026-09-03, fifth batch (patch_target_is_a_reexport): one entry, `default_via_or` on
`alias.asname or alias.name` again -- the canonical ast-alias idiom, reviewed three times below
now, where `asname` is `str | None` and never the empty string. Every module that walks imports
writes this line, so this package will keep producing it.

2026-09-03, fourth batch (test_asserts_against_production_constant): one entry, `default_via_or`
on `alias.asname or alias.name` -- the same canonical ast-alias idiom reviewed twice below, where
`asname` is `str | None` and never the empty string. The scanner's own hits against real code were
not baselined: six were false and were fixed in the rule, and the one true positive was fixed in
the codebase it found (a scraper test comparing a variable with its own definition).

2026-09-03, second batch (guard_decidable_from_constants, sql_selects_unread_column,
count_then_fetch_same_table): one new entry, `default_via_or` on `alias.asname or alias.name`
-- the same canonical ast-alias idiom already reviewed below, where `asname` is `str | None` and
never the empty string. The batch's other self-scan hit was NOT baselined: this package's own
duplicate_function_body check caught the two SQL scanners carrying copied helpers, so they were
extracted into `_base` (`_module_sql_constants`, `_sql_text`) and the finding went away.

* `default_via_or` x3 in the new scanners. All three are correct idioms, not traps:
  `calls.get(name, False) or <bool>` is a boolean OR on booleans, and `a.asname or a.name` is
  the canonical ast-alias idiom where `asname` is `str | None` and never the empty string.

* `unthrottled_hot_loop_log` x2 in `web/browser.py` -- the same two entries already baselined,
  moved by the ten comment lines this commit adds above them. Line shift, not a new finding.

This is the same baseline-driven wiring rolled out to every downstream
consumer of dev.code_audit (glossum_backend_scripts, llm_bench,
realtime_applications, production_scrapers, mlframe, algopacksimple) --
pyutilz eats its own dog food, and (like every other consumer) uses the
shared harness in py_ci_shared.code_audit_meta rather than a hand-rolled
copy. See ``pyutilz/src/pyutilz/dev/code_audit/__init__.py`` for what
each check catches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# py-ci-shared requires python>=3.9 (pyproject.toml dev-dependency marker), so it's not
# installed on the 3.8 CI leg -- skip cleanly there instead of erroring at collection.
py_ci_shared_code_audit_meta = pytest.importorskip("py_ci_shared.code_audit_meta")
assert_no_new_code_audit_findings = py_ci_shared_code_audit_meta.assert_no_new_code_audit_findings

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_code_audit_baseline.json"


def test_no_new_code_audit_findings(request):
    assert_no_new_code_audit_findings(
        root=PYUTILZ_DIR,
        baseline_path=_BASELINE_PATH,
        request=request,
    )
