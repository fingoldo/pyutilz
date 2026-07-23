"""AST-based code-audit scanners for recurring Python production-bug classes.

Library + CLI for the kinds of static checks we kept rebuilding from
throwaway scripts inside subagent runs. Promoted into pyutilz so they
stop living in D:/Temp and can run against any project (mlframe,
pyutilz itself, downstream applications) with a single import or
``python -m pyutilz.dev.code_audit ./src``.

Implemented checks (each a top-level scanner function returning a
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
  as a legitimate sentinel. Detected via AST ``BoolOp(Or, ...)`` with
  a constant/call/name on the right. Heuristic; not 100% recall.

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
  through the un-copied parameter reference.

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
import ast
import json
import sys
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
from .missed_await import scan_missed_await, scan_sync_blocking_in_async
from .redundant_test_fit import scan_redundant_test_fit_calls
from .undeclared_imports import scan_undeclared_imports
from .vacuous_assertions import scan_vacuous_assertions
from .locals_globals_output import scan_locals_globals_as_output
from .network_timeout import scan_missing_network_timeout
from .retry_loops import scan_retry_loops
from .module_docstring import scan_duplicate_module_docstring
from .unraised_exceptions import scan_unraised_exceptions
from .credential_logging import scan_credential_shaped_log_args
from .docstring_args import scan_docstring_args_completeness
from .return_annotation import scan_return_annotation_mismatch
from .locals_get import scan_locals_get_fragile_lookup
from .shielded_resource_release import scan_shielded_resource_release_race
from .duplicate_credential_regex import scan_duplicate_credential_regex
from .asymmetric_resource_guard import scan_asymmetric_resource_guard
from .spy_arity import scan_stale_test_spy_arity
from .log_throttle import scan_unthrottled_hot_loop_log
from .registry import SCANNERS, run_all, register_scanner, get_scanners
from .cli import main

__all__ = [
    "Finding",
    "SCANNERS",
    "run_all",
    "register_scanner",
    "get_scanners",
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
    "scan_missed_await",
    "scan_redundant_test_fit_calls",
    "scan_undeclared_imports",
    "scan_vacuous_assertions",
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
    "scan_locals_get_fragile_lookup",
    "scan_shielded_resource_release_race",
    "scan_duplicate_credential_regex",
    "scan_asymmetric_resource_guard",
    "scan_stale_test_spy_arity",
    "scan_unthrottled_hot_loop_log",
]

# Keep the public attribute surface identical to the pre-split flat module:
# drop the submodule names that ``from .X import ...`` bound into this
# namespace. They stay importable on demand as ``code_audit.<submodule>``
# (Python re-registers them in sys.modules); this only trims ``dir()``.
for _submod in (
    "_base", "mutable_defaults", "closures", "default_via_or",
    "broad_except", "nan_equality", "mutation_during_iteration",
    "sql_lint", "dead_cli_flags", "silent_escalation", "sql_migrations",
    "duplicate_conditions", "missed_await", "redundant_test_fit",
    "undeclared_imports", "vacuous_assertions", "locals_globals_output",
    "network_timeout", "retry_loops", "module_docstring",
    "unraised_exceptions", "credential_logging",
    "docstring_args", "return_annotation", "locals_get",
    "shielded_resource_release", "duplicate_credential_regex",
    "asymmetric_resource_guard",
    "spy_arity", "log_throttle",
    "registry", "cli",
):
    globals().pop(_submod, None)
del _submod
