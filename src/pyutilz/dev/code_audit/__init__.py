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
from .mutable_defaults import scan_mutable_defaults
from .closures import scan_late_binding_closures
from .default_via_or import scan_default_via_or_trap
from .broad_except import scan_broad_except_swallows
from .nan_equality import scan_nan_equality
from .mutation_during_iteration import scan_mutation_during_iteration
from .sql_lint import scan_sql_limit_without_order_by, scan_sql_offset_pagination
from .dead_cli_flags import scan_dead_cli_flags
from .silent_escalation import scan_log_only_except, DEFAULT_ESCALATION_ATTRS
from .sql_migrations import scan_sql_migration_idempotency
from .duplicate_conditions import scan_duplicate_conditions
from .registry import SCANNERS, run_all
from .cli import main

__all__ = [
    "Finding",
    "SCANNERS",
    "run_all",
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
]

# Keep the public attribute surface identical to the pre-split flat module:
# drop the submodule names that ``from .X import ...`` bound into this
# namespace. They stay importable on demand as ``code_audit.<submodule>``
# (Python re-registers them in sys.modules); this only trims ``dir()``.
for _submod in (
    "_base", "mutable_defaults", "closures", "default_via_or",
    "broad_except", "nan_equality", "mutation_during_iteration",
    "sql_lint", "dead_cli_flags", "silent_escalation", "sql_migrations",
    "duplicate_conditions",
    "registry", "cli",
):
    globals().pop(_submod, None)
del _submod
