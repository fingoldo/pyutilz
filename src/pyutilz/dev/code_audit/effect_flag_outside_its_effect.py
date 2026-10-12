"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from typing import Iterator
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

_EXIT_STMTS = (ast.Continue, ast.Return, ast.Raise, ast.Break)

# --- a success record set outside the conditional work it records ------------------------------
#
# A flag, set membership or counter says "X happened", and X itself sits inside an `if` that may
# not run. The record then outlives its referent, and every consumer downstream believes work was
# done that was not.
#
# Confirmed instances in one audited codebase:
#
# * `_kind_ok["pq"] = True` on a path where the parquet write was `if table.num_rows:` and the flag
#   assignment was not -- so an empty crawl advertised a file it had not written.
# * `reached_uids.add(job_uid)` on a branch whose own comment says "we never actually asked the
#   API", making circuit-breaker skips indistinguishable from successful reaches.
#
# The link between record and work is a SHARED name or key: the flag mentions the same string or
# variable the guarded call does. Without that, the rule would flag every assignment that follows
# an `if`, which is most of them.

_RECORD_VALUES = (True,)

# `self` and `cls` are mentioned by almost every statement in a method, so accepting either as
# the shared token between a guard and a record makes the link vacuous -- `self._ready = True`
# after an unrelated `if self._process.stdout:` was reported on exactly that basis.
_VACUOUS_TOKENS = frozenset({"self", "cls"})

# A guard whose body only logs guards no work: the statement after it is the work, and it is
# unconditional by design. `if verbose: logger.info(...)` followed by `res.add(str(obj))` is
# the canonical shape, and reporting it inverts the rule.
_LOGGING_RECEIVERS = frozenset({"log", "logger", "logging", "_log", "_logger"})


def _shared_tokens(node: ast.AST) -> set[str]:
    """Names and string literals mentioned anywhere in this node."""
    tokens: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            tokens.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            tokens.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            tokens.add(sub.value)
    return tokens


def _is_a_success_record(stmt: ast.stmt) -> tuple[str, set[str]] | None:
    """(what it records, tokens it mentions) if this statement records that work happened."""
    # `is True`, never `== True`: `1 == True` and `1.0 == True` in Python, so an ordinary counter
    # assignment (`counts['rows'] = 1`) would otherwise be read as a boolean success flag.
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.Constant) and any(stmt.value.value is v for v in _RECORD_VALUES):
        target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
        return "a True flag", _shared_tokens(target)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        # `.add` only, never `.append`. Appending to a list is ordinary accumulation and
        # produced 44 of this rule's 50 first hits with not one success record among them;
        # `set.add(id)` is the shape the evidence has -- membership standing for "this one was
        # reached".
        if isinstance(call.func, ast.Attribute) and call.func.attr == "add":
            return "a set membership", _shared_tokens(call)
    return None


def _tests_membership_of(test: ast.expr, record: ast.stmt) -> bool:
    """Does the guard test membership in the very container the record writes to?

    ``if sid in seen: report_duplicate(...)`` followed by ``seen.add(sid)`` is the
    seen-set idiom: the set tracks everything encountered, and the `if` REPORTS a
    repeat rather than gating the record. Moving the record inside -- the remedy
    this rule would otherwise suggest -- deletes the duplicate detection entirely.

    Keyed on the container identity rather than on what the guard's body does,
    because a body that only appends to an error list is ALSO the shape of the
    real defect (record success even though the branch failed). What separates
    them is that here the guard interrogates the same object the record mutates.
    """
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.In, ast.NotIn))):
        return False
    # ast.dump rather than ast.unparse: unparse needs python>=3.9, and all this comparison needs is
    # "is it the same expression", which the position-free dump answers identically on every version.
    container = ast.dump(test.comparators[0])
    if isinstance(record, ast.Expr) and isinstance(record.value, ast.Call):
        func = record.value.func
        if isinstance(func, ast.Attribute):
            return ast.dump(func.value) == container
    return False


def _only_initialises(block: list[ast.stmt]) -> bool:
    """Does this block do nothing but create an empty container?

    ``if key not in d: d[key] = set()`` is a lazy-initialisation guard, not the
    work a following ``d[key].add(...)`` records -- the record belongs to a
    DIFFERENT condition (typically an ``if already_seen: continue`` right after).
    Pairing the record with this guard reports the canonical seen-set idiom as a
    defect, and its suggested remedy -- move the record inside -- would break it.
    """
    if not block:
        return False
    for stmt in block:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            return False
        value = stmt.value
        if isinstance(value, ast.Dict) and not value.keys:
            continue
        if isinstance(value, (ast.List, ast.Set, ast.Tuple)) and not value.elts:
            continue
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)                 and value.func.id in {"set", "list", "dict", "Counter", "defaultdict"} and not value.args:
            continue
        return False
    return True


def _only_logs(block: list[ast.stmt]) -> bool:
    """Does this block do nothing but log? Then it is not the work a following record records."""
    calls = [sub for stmt in block for sub in ast.walk(stmt) if isinstance(sub, ast.Call)]
    if not calls:
        return False
    for call in calls:
        func = call.func
        receiver = func.value if isinstance(func, ast.Attribute) else None
        if isinstance(receiver, ast.Name) and receiver.id in _LOGGING_RECEIVERS:
            continue
        if isinstance(func, ast.Name) and func.id in {"print", "str", "len", "type", "repr"}:
            continue
        return False
    return True


def _calls_in(node: ast.AST) -> bool:
    """Does this subtree contain a call? A guard with no call guards no work."""
    return any(isinstance(sub, ast.Call) for sub in ast.walk(node))


def _blocks(func: ast.AST) -> Iterator[list[ast.stmt]]:
    """Every statement list in this function, so the rule sees siblings rather than the whole tree."""
    for node in ast.walk(func):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block:
                yield block


def scan_effect_flag_outside_its_effect(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a success record set beside, rather than inside, the conditional work it records.

    An `if` guards a call; the statement after it records that the call happened. When the
    condition is false the record is written anyway, and every consumer believes work was done.
    That is how an empty crawl advertised a parquet file it had never written.

    The record and the work must share a name or a string key -- otherwise this would flag every
    assignment following an `if`. The record must also not appear INSIDE the guarded block, which
    is the correct form.
    """
    findings: list[Finding] = []
    # `_blocks` yields nested statement lists, so one site can be visited more than once.
    reported: set[tuple[str, int]] = set()
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                findings.extend(_function_findings(func, rel, src_lines, reported))
    return findings


def _function_findings(func: ast.AST, rel: str, src_lines: list[str], reported: set[tuple[str, int]]) -> list[Finding]:
    """Findings for every guard ``if`` in ``func`` whose following sibling records work it guards."""
    out: list[Finding] = []
    for block in _blocks(func):
        for index, stmt in enumerate(block[:-1]):
            if not isinstance(stmt, ast.If) or _is_exempt_guard(stmt):
                continue
            guarded_tokens = _shared_tokens(ast.Module(body=stmt.body, type_ignores=[]))
            finding = _first_unguarded_record(stmt, block[index + 1 :], guarded_tokens, rel, src_lines, reported)
            if finding is not None:
                out.append(finding)
    return out


def _is_exempt_guard(stmt: ast.If) -> bool:
    """True when this guard must not be paired with a following record (no call, log-only, lazy init, or record inside)."""
    if stmt.orelse or not _calls_in(stmt):
        return True
    if _only_logs(stmt.body) or _only_initialises(stmt.body):
        return True
    # The correct form records INSIDE the guarded block; if it does, say nothing.
    if any(_is_a_success_record(inner) for inner in stmt.body):
        return True
    # A guarded block containing any early exit (continue/return/raise/break), at any depth, means the
    # record that follows is on the OTHER path by construction. Conservative on purpose: see _has_exit.
    return _has_exit(stmt)


def _has_exit(node: ast.AST) -> bool:
    """Does this subtree contain an early exit? A guard or follower with one is not reached on the success path."""
    return any(isinstance(sub, _EXIT_STMTS) for sub in ast.walk(node))


def _first_unguarded_record(
    stmt: ast.If,
    followers: list[ast.stmt],
    guarded_tokens: set[str],
    rel: str,
    src_lines: list[str],
    reported: set[tuple[str, int]],
) -> Finding | None:
    """The first sibling after ``stmt`` that records the guarded work outside its guard, or None.

    Scanning stops at the first follower containing an early exit: an unconditional terminator between the
    guard and the record puts the record on a different path entirely.
    """
    for follower in followers:
        if _has_exit(follower):
            return None
        hit = _unguarded_record(stmt, follower, guarded_tokens)
        if hit is None or (rel, follower.lineno) in reported:
            continue
        reported.add((rel, follower.lineno))
        return _effect_finding(stmt, follower, hit[0], hit[1], rel, src_lines)
    return None


def _unguarded_record(stmt: ast.If, follower: ast.stmt, guarded_tokens: set[str]) -> tuple[str, list[str]] | None:
    """``(what, shared_tokens)`` when ``follower`` records a token the guard's work also names, else None."""
    record = _is_a_success_record(follower)
    if record is None:
        return None
    what, record_tokens = record
    shared = sorted(t for t in record_tokens & guarded_tokens if len(t) > 1 and t not in _VACUOUS_TOKENS)
    if not shared or _tests_membership_of(stmt.test, follower):
        return None
    return what, shared


def _effect_finding(stmt: ast.If, follower: ast.stmt, what: str, shared: list[str], rel: str, src_lines: list[str]) -> Finding:
    """Build the P2 finding for a record set outside the conditional work it records."""
    return Finding(
        check="effect_flag_outside_its_effect",
        severity="P2",
        file=rel,
        line=follower.lineno,
        snippet=_line_text(src_lines, follower.lineno),
        detail=(
            f"{what} recording `{shared[0]}` is set here, outside the `if` at "
            f"line {stmt.lineno} that guards the work it records. When that "
            "condition is false the record is written anyway, and every "
            "consumer downstream believes work was done that was not."
        ),
    )
