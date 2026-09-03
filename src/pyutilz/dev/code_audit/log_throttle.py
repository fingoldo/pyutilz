"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

_LOG_METHODS = ("warning", "error", "exception", "critical")
_THROTTLE_HINTS = ("throttle", "rate_limit", "ratelimit", "debounce")
_BOUNDED_RETRY_HINTS = ("retr", "attempt")


def _call_name(node: ast.AST) -> str | None:
    """Return a Call's callable name (bare ``Name.id`` or ``Attribute.attr``), else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _guard_looks_throttled(test: ast.expr) -> bool:
    """True if an ``if`` test guarding a log call plausibly already throttles it: calls a
    helper whose name contains a throttle-shaped hint, or uses a modulo/interval-counter idiom
    (``if i % N == 0``, a common cheap throttle)."""
    for node in ast.walk(test):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name and any(h in name.lower() for h in _THROTTLE_HINTS):
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return True
    return False


def _looks_like_retry_bound_name(node: ast.expr | None) -> bool:
    """True if ``node`` is a Name/Attribute whose identifier contains a retry/attempt-shaped
    hint (``max_retries``, ``num_attempts``, ``retry_count``, ...)."""
    if node is None:
        return False
    name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
    return name is not None and any(h in name.lower() for h in _BOUNDED_RETRY_HINTS)


def _loop_looks_bounded_retry(test: ast.expr) -> bool:
    """True if a ``while`` loop's test is the common ``n < max_retries`` / ``attempt <=
    max_attempts`` bounded-retry idiom: a Compare where either side is a Name/Attribute whose
    identifier contains a retry/attempt-shaped hint. A retry loop capped at a small,
    caller-configured count (typically single digits) logging once per attempt is standard
    practice, not a "hot loop that compounds into spam under load" the way a per-item loop over
    an unbounded batch/dataset is -- the whole point of the retry cap is that it CAN'T iterate
    more than a handful of times."""
    if not isinstance(test, ast.Compare):
        return False
    return any(_looks_like_retry_bound_name(side) for side in (test.left, *test.comparators))


def _for_loop_looks_bounded_retry(iter_node: ast.expr) -> bool:
    """True for the ``for _ in range(max_retries)`` / ``for attempt in range(num_attempts)``
    bounded-retry idiom -- same reasoning as ``_loop_looks_bounded_retry``, just the For-loop
    spelling of a bounded retry count instead of a While-loop comparison."""
    if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == "range"):
        return False
    return any(_looks_like_retry_bound_name(arg) for arg in iter_node.args)


def _while_test_is_unconditionally_true(test: ast.expr) -> bool:
    """True for the ``while True:`` / ``while 1:`` spelling of an intentionally-unbounded-by-its-
    OWN-test loop (the retry cap, if any, lives in an internal ``break``/``raise`` instead)."""
    if isinstance(test, ast.Constant):
        return test.value is True or test.value == 1
    return False


def _loop_body_has_bounded_retry_break(stmts: list[ast.stmt]) -> bool:
    """True if ``stmts`` (walked recursively) contains an ``if <retry-hint comparison>: break``/
    ``raise`` -- the ``while True: ... if attempts >= max_retries: break`` idiom for a bounded
    retry loop, just spelled with the bound as an internal early-exit instead of the loop's own
    test expression (which ``_loop_looks_bounded_retry`` already recognizes).

    Confirmed false-positive pattern found dogfooding this scanner on pyutilz's own source
    (2026-08-04, ``dev/logginglib.py``'s ``debugged()`` decorator): a `while True:` retry loop
    whose only exit is `if not interactive or attempts >= max_retries: raise` inside the except
    block is EXACTLY as bounded as `while attempts < max_retries:` -- arguably the more common
    spelling for a "try body must run at least once before the first bound check" retry loop --
    but the bound lives in the break/raise condition, not the (constant, uninformative) `while
    True:` test itself, so it was invisible to the existing check.
    """
    for stmt in stmts:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.If):
                continue
            # The retry-bound comparison may be combined with other conditions via `and`/`or`
            # (e.g. `if not interactive or attempts >= max_retries: raise`), not just a bare
            # Compare -- walk the whole test expression for a qualifying Compare anywhere in it.
            has_retry_bound = any(isinstance(sub, ast.Compare) and _loop_looks_bounded_retry(sub) for sub in ast.walk(node.test))
            if has_retry_bound and any(isinstance(s, (ast.Break, ast.Raise)) for s in ast.walk(node)):
                return True
    return False


def _loop_body_has_meaningful_sleep(stmts: list[ast.stmt]) -> bool:
    """True if any statement in ``stmts`` (walked recursively, including inside nested
    if/try/for/while) calls ``sleep(...)``/``time.sleep(...)``/``asyncio.sleep(...)`` with at
    least one argument -- a polling loop that sleeps between iterations (``while not
    condition(): ...; sleep(10)``) is naturally rate-limited to at most one log line per sleep
    interval, the same "already throttled" signal as an explicit rate-limit guard, just
    expressed as pacing rather than a counter/modulo check.

    ``<event>.wait(<timeout>)`` counts too: a monitor thread spelled ``while not
    stop_flag.is_set(): ...; stop_flag.wait(interval)`` paces itself exactly as ``sleep(interval)``
    would, and only uses the Event form so ``stop()`` returns immediately instead of blocking for a
    full interval. Measured false positive: ``system/hardware_monitor.py``'s sampling thread."""
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and _call_name(node.func) in {"sleep", "wait"} and node.args:
                return True
    return False


#: A loop iterating a literal collection this small cannot "compound into spam under load" -- its
#: iteration count is written in the source and does not grow with the data. 16 comfortably covers the
#: measured shapes (``for func in (a, b, c)``, ``for candidate in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)``,
#: ``for col in ("minr", "maxr")``) while staying well under any plausible per-item batch.
_STATIC_ITERATION_BOUND = 16


def _iterable_is_small_literal(iter_node: ast.expr) -> bool:
    """True when the loop's iterable is a source-visible collection of at most
    ``_STATIC_ITERATION_BOUND`` elements: a tuple/list/set display, a dict display (or its
    ``.items()``/``.keys()``/``.values()``), or ``range(<small int literal>)``.

    This is the iteration bound the 2026-09-03 downstream scan asked for. It is a STATIC fact, not
    a heuristic: the count is spelled out in the source, so no amount of load can raise it."""
    node = iter_node
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"items", "keys", "values"} and not node.args:
        node = node.func.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return not any(isinstance(e, ast.Starred) for e in node.elts) and len(node.elts) <= _STATIC_ITERATION_BOUND
    if isinstance(node, ast.Dict):
        return None not in node.keys and len(node.keys) <= _STATIC_ITERATION_BOUND
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and len(node.args) == 1:
        arg = node.args[0]
        return isinstance(arg, ast.Constant) and isinstance(arg.value, int) and not isinstance(arg.value, bool) and arg.value <= _STATIC_ITERATION_BOUND
    return False


def _collect_log_calls(stmt: ast.stmt) -> "list[ast.Call]":
    """Every log Call inside ``stmt`` that is NOT under a nested loop or a nested function scope.

    A call under a nested loop runs per-inner-iteration, so a sibling ``break``/``return`` after the
    OUTER statement says nothing about how often it fires."""
    out: list[ast.Call] = []
    todo: list[ast.AST] = [stmt]
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if _is_log_call(node):
            assert isinstance(node, ast.Call)
            out.append(node)
        todo.extend(ast.iter_child_nodes(node))
    return out


def _single_shot_log_calls(tree: ast.AST) -> "dict[int, str]":
    """Map ``id(log_call) -> 'hard' | 'break'`` for log calls that CANNOT repeat within their loop.

    A log statement followed, among its own siblings, by an unconditional ``return``/``raise``
    ("hard": leaves the function outright) or ``break`` ("break": leaves the innermost loop) fires at
    most once per entry to that loop -- the very thing "compounds into spam under load" denies. This
    was the single largest measured false-positive shape: ``logger.warning(...)`` immediately
    followed by ``break``/``return`` inside a scanning loop.
    """
    marks: dict[int, str] = {}
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            _mark_block(getattr(node, field, None), marks)
        for handler in getattr(node, "handlers", []) or []:
            _mark_block(handler.body, marks)
    return marks


def _mark_block(stmts: "list[ast.stmt] | None", marks: "dict[int, str]") -> None:
    """Mark the log calls of each statement in ``stmts`` that a LATER sibling unconditionally exits past."""
    # `Lambda.body` and `IfExp.body` are single EXPRESSIONS, not statement lists -- getattr cannot tell.
    if not stmts or not isinstance(stmts, list):
        return
    kind: str | None = None
    for stmt in reversed(stmts):
        if kind is not None:
            for call in _collect_log_calls(stmt):
                # A "hard" exit already recorded for this call wins over a mere `break`.
                if marks.get(id(call)) != "hard":
                    marks[id(call)] = kind
        if isinstance(stmt, (ast.Return, ast.Raise)):
            kind = "hard"
        elif isinstance(stmt, ast.Break):
            kind = "break" if kind != "hard" else "hard"


def _is_log_call(node: ast.AST) -> str | None:
    """Return the log method name ('warning'/'error'/...) if this Call node is
    ``<something ending in 'log'/'logger'>.<method>(...)``, else None."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr not in _LOG_METHODS:
        return None
    receiver = node.func.value
    receiver_name = None
    if isinstance(receiver, ast.Name):
        receiver_name = receiver.id
    elif isinstance(receiver, ast.Attribute):
        receiver_name = receiver.attr
    if receiver_name is None:
        return None
    low = receiver_name.lower()
    if low.endswith("log") or low.endswith("logger") or low == "_log":
        return node.func.attr
    return None


def _visit_if_aware(
    node: ast.AST,
    loop_depth: int,
    guarded: bool,
    findings: list[Finding],
    rel: str,
    src_lines: list[str],
    single_shot: "dict[int, str] | None" = None,
) -> None:
    """Manual recursive descent (not ast.walk) so If nodes can pass the "throttle-guarded" flag
    to their `body` only (not `orelse`), and loop nodes bump depth for their BODY (a loop's own
    `orelse` runs once per loop entry, not per iteration, and keeps the enclosing depth). A `for`
    loop's own target/iter expressions evaluate ONCE and keep the enclosing depth; a `while` loop's
    test is re-evaluated every iteration and is treated as part of the loop. Takes findings/rel/src_lines as explicit params
    (not closure captures) so this can be a plain module-level function reused across files."""
    single_shot = {} if single_shot is None else single_shot
    method = _is_log_call(node)
    # A log call a sibling `return`/`raise` exits past cannot repeat at all; one a sibling `break`
    # exits past cannot repeat within its innermost loop, which settles the question only when that
    # loop is the ONLY one enclosing it (at depth 2+ the outer loop still re-enters it per item).
    shot = single_shot.get(id(node))
    single_shot_here = shot == "hard" or (shot == "break" and loop_depth == 1)
    if method is not None and loop_depth > 0 and not guarded and not single_shot_here:
        assert isinstance(node, ast.Call)  # guaranteed by _is_log_call returning non-None
        findings.append(Finding(
            check="unthrottled_hot_loop_log",
            severity="P2",
            file=rel,
            line=node.lineno,
            snippet=_line_text(src_lines, node.lineno),
            detail=(
                f"log.{method}(...) inside a loop (depth {loop_depth}) with no apparent throttle guard "
                f"-- every iteration that hits this line logs, compounding into spam under load. "
                f"Guard with a throttle helper (e.g. this project's own rate-limit debounce pattern) "
                f"or demote to log.debug()."
            ),
        ))
    if isinstance(node, (ast.For, ast.AsyncFor)):
        already_throttled = _for_loop_looks_bounded_retry(node.iter) or _loop_body_has_meaningful_sleep(node.body) or _iterable_is_small_literal(node.iter)
        # `target`/`iter` evaluate ONCE, outside the iteration, so they keep the enclosing depth --
        # as the docstring above says. Only body/orelse run per iteration.
        for once_evaluated in (node.target, node.iter):
            if once_evaluated is not None:
                _visit_if_aware(once_evaluated, loop_depth, guarded or already_throttled, findings, rel, src_lines, single_shot)
        for stmt in node.body:
            _visit_if_aware(stmt, loop_depth + 1, guarded or already_throttled, findings, rel, src_lines, single_shot)
        # A loop's `orelse` runs at most ONCE per loop entry (after the iterable is exhausted), not
        # per iteration, so it keeps the ENCLOSING depth -- flagging it as a hot-loop log was simply
        # wrong. Measured shape: `for ...: ... else: log.warning("stopped after N pages")`.
        for stmt in node.orelse:
            _visit_if_aware(stmt, loop_depth, guarded, findings, rel, src_lines, single_shot)
    elif isinstance(node, ast.While):
        already_throttled = (
            _loop_looks_bounded_retry(node.test)
            or _loop_body_has_meaningful_sleep(node.body)
            or (_while_test_is_unconditionally_true(node.test) and _loop_body_has_bounded_retry_break(node.body))
        )
        # A `while` test, unlike a `for` target/iter, IS re-evaluated every iteration, so a log
        # call in it is per-iteration spam and keeps the bumped depth.
        _visit_if_aware(node.test, loop_depth + 1, guarded or already_throttled, findings, rel, src_lines, single_shot)
        for stmt in node.body:
            _visit_if_aware(stmt, loop_depth + 1, guarded or already_throttled, findings, rel, src_lines, single_shot)
        for stmt in node.orelse:  # once per loop entry, not per iteration -- see the For branch above
            _visit_if_aware(stmt, loop_depth, guarded, findings, rel, src_lines, single_shot)
    elif isinstance(node, ast.If):
        child_guarded = guarded or _guard_looks_throttled(node.test)
        for stmt in node.body:
            _visit_if_aware(stmt, loop_depth, child_guarded, findings, rel, src_lines, single_shot)
        for stmt in node.orelse:
            _visit_if_aware(stmt, loop_depth, guarded, findings, rel, src_lines, single_shot)
        _visit_if_aware(node.test, loop_depth, guarded, findings, rel, src_lines, single_shot)
    else:
        for child in ast.iter_child_nodes(node):
            _visit_if_aware(child, loop_depth, guarded, findings, rel, src_lines, single_shot)


def scan_unthrottled_hot_loop_log(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a ``log.warning``/``log.error`` (etc.) call inside a ``for``/``while`` loop with no
    apparent rate-limiting guard -- a per-item log line that compounds into log-flood spam the
    moment every item in a large batch hits the same condition (e.g. a systemic upstream outage).

    Each individual call looks reasonable in isolation ("log the error when it happens"); the
    compounding-under-load behavior only shows up during an actual incident, which code review
    rarely simulates. Real example: ``job_details_shared.py``'s two ``ErrorCat.OTHER`` warning
    sites fired unconditionally per-item inside a batch loop, right next to an established
    ``_log_throttle`` debounce helper already used elsewhere in the same codebase for the
    structurally identical ``scan_loop`` case (2026-07-21 full-audit, observability-Low, closed
    in the gap-pass).

    A log call is considered guarded (not flagged) if its nearest enclosing ``if`` test either
    calls a function whose name contains a throttle-shaped hint (``throttle``, ``rate_limit``,
    ``debounce``, case-insensitive -- matches this project's own ``_log_throttle`` and similarly
    named helpers elsewhere without per-project configuration) or uses a modulo expression
    (``i % N``), a common cheap "every Nth iteration" throttle idiom. A ``while``/``for`` loop is
    also exempt when its test/iter is the common ``n < max_retries`` / ``for _ in
    range(max_retries)`` bounded-retry idiom (logs at most a handful of times per call, unable to
    "compound into spam under load"), or when the loop body calls ``sleep(...)`` with an
    argument -- a polling loop that sleeps between iterations is naturally rate-limited to at
    most one log line per sleep interval, the same signal as an explicit throttle guard.

    Three further exemptions came out of measuring this check against pyutilz, py-ci-shared and
    mlframe on 2026-09-03 -- each one a site that CANNOT compound under load, so exempting it costs
    no true positive. A loop's ``else`` clause runs at most once per loop ENTRY, after the iterable
    is exhausted, and keeps the enclosing depth. A log statement that a LATER SIBLING statement
    unconditionally exits past -- ``return``/``raise`` at any depth, ``break`` when that is the only
    enclosing loop -- fires at most once per entry to that loop. And a loop over a source-visible
    collection (``for func in (a, b, c)``, ``range(<small int literal>)``) has its iteration count
    spelled out in the source, so no amount of data can raise it.

    Severity: P2 (usually a hygiene/observability issue, not a correctness bug -- but can degrade
    into real operational pain: log-volume-driven disk fill, alerting fatigue, or a downstream
    log-aggregation cost spike during exactly the outage an operator most needs signal, not noise).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        _visit_if_aware(tree, 0, False, findings, rel, src_lines, _single_shot_log_calls(tree))
    return findings
