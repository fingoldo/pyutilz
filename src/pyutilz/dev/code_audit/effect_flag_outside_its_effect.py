"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

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
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and stmt.value.value in _RECORD_VALUES:
        return "a True flag", _shared_tokens(stmt.targets[0])
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        # `.add` only, never `.append`. Appending to a list is ordinary accumulation and
        # produced 44 of this rule's 50 first hits with not one success record among them;
        # `set.add(id)` is the shape the evidence has -- membership standing for "this one was
        # reached".
        if isinstance(call.func, ast.Attribute) and call.func.attr == "add":
            return "a set membership", _shared_tokens(call)
    return None


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
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for block in _blocks(func):
                for index, stmt in enumerate(block[:-1]):
                    if not isinstance(stmt, ast.If) or stmt.orelse or not _calls_in(stmt):
                        continue
                    if _only_logs(stmt.body):
                        continue
                    guarded_tokens = _shared_tokens(ast.Module(body=stmt.body, type_ignores=[]))

                    # The correct form records INSIDE the guarded block; if it does, say nothing.
                    if any(_is_a_success_record(inner) for inner in stmt.body):
                        continue

                    # A guarded block ending in `continue`/`return`/`raise`/`break` never reaches
                    # the statement after it, so the record is on the OTHER path by construction.
                    # That is the shape the audited codebase adopted when it fixed this defect --
                    # the breaker-skip branch ends in `continue` and `reached_uids.add()` follows
                    # it -- and without modelling that, the rule reports the FIX as the bug.
                    if isinstance(stmt.body[-1], (ast.Continue, ast.Return, ast.Raise, ast.Break)):
                        continue

                    for follower in block[index + 1 :]:
                        # An unconditional terminator between the guard and the record means the
                        # record is on a different path entirely. Both of this rule's remaining
                        # hits on real code were this -- and both were the FIXED form of the very
                        # defect it looks for, where the failure branch ends in `continue` and the
                        # success record follows it.
                        if isinstance(follower, (ast.Continue, ast.Return, ast.Raise, ast.Break)):
                            break
                        record = _is_a_success_record(follower)
                        if record is None:
                            continue
                        what, record_tokens = record
                        shared = sorted(t for t in record_tokens & guarded_tokens if len(t) > 1 and t not in _VACUOUS_TOKENS)
                        if not shared or (rel, follower.lineno) in reported:
                            continue
                        reported.add((rel, follower.lineno))
                        findings.append(
                            Finding(
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
                        )
                        break
    return findings
