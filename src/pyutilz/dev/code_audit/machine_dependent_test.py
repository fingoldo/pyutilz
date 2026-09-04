"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from ._base import Finding, is_test_file, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

_FunctionDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]

# --- tests whose PASS/FAIL depends on the machine rather than on the behaviour under test ----------
#
# Four separate rounds of CI-only failures came from this class while the whole suite was green on the
# dev box, so the shapes below are taken from those rounds rather than from a general list of things
# that could in principle differ between machines. Each one is a shape whose truth value is a property
# of the runner - how fast it is, how many samples a thread got through, which attributes this
# interpreter happens to have - and not a property of the code the test names.
#
# The rounds, and what each shape covers:
#
# 1. A test wrote a 20000-term expression and asserted the parser returned None. How deep an
#    expression a parser survives varies by interpreter version and platform: it blew the stack on
#    the dev box and parsed cleanly on twelve of the matrix's legs. NOT COVERED HERE - see the
#    package docs; the shape was measured and rejected because it cannot be separated from the
#    legitimate "oversized input hits a cap the test itself configures" test at a usable rate.
# 2. A monitor test slept a wall-clock 1.5s and then asserted a count that a sampling thread had
#    produced meanwhile; on a GPU-less runner the real sampler raised every interval, so the count
#    was whatever had elapsed (32, 41, 54 on different legs) instead of the one injected failure.
#    Covered by ``scan_sleep_then_assert``.
# 3. A test read ``ast.unparse`` unconditionally and then deleted it, to exercise the 3.8 fallback -
#    so it raised AttributeError on 3.8, the one interpreter whose fallback it exists to cover.
#    Covered by ``scan_deleted_attribute_read_unconditionally``.
# 4. Assertions on an elapsed wall-clock duration. Not one of the four rounds, but the same class,
#    and the one that pays for itself as a ratchet. Covered by ``scan_wall_clock_assertion``.

# Reading either of these is a wall-clock (or CPU-clock) sample; a difference of two of them is a
# duration, and an assertion on a duration is an assertion about the runner.
_TIMER_FUNCS = frozenset({"perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns", "process_time", "process_time_ns", "time", "time_ns"})

# A sleep BELOW this is a scheduling yield - it hands the GIL over so another thread can run, and the
# assertion that follows is normally about what that thread did, not about how long it had. A sleep AT
# OR ABOVE it is the test paying real wall-clock time in the hope that enough happened meanwhile,
# which is exactly the failing shape. Measured over this repo's ~4900 tests: the unrestricted rule
# fires on 7 tests of which 3 are real, and this threshold selects those 3 and nothing else.
_WALL_CLOCK_SLEEP_SECONDS = 0.1


def _is_timer_call(node: ast.AST) -> bool:
    """Whether ``node`` is a ``<module>.perf_counter()`` / ``.time()`` / ... clock reading."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _TIMER_FUNCS


def _is_timer_difference(node: ast.AST) -> bool:
    """Whether ``node`` is ``<clock>() - t0`` or ``t1 - <clock>()`` - i.e. an elapsed duration."""
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) and (_is_timer_call(node.left) or _is_timer_call(node.right))


def _is_duration(node: ast.AST, elapsed_names: Set[str]) -> bool:
    """Whether ``node`` is an elapsed duration - written inline, or held by one of ``elapsed_names``."""
    return _is_timer_difference(node) or (isinstance(node, ast.Name) and node.id in elapsed_names)


def _own_body_nodes(fn: ast.AST) -> List[ast.AST]:
    """Every node lexically inside ``fn`` but NOT inside a nested function, lambda or class.

    A ``time.sleep(0.1)`` inside a nested ``def task(): ...`` is the test's stand-in for a slow
    workload - the test is about the wrapper around it, and the sleep is an input, not a wait. Only
    the sleeps the test itself performs are waits, and only those are of interest.
    """
    out: List[ast.AST] = []
    stack: List[ast.AST] = [fn]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            out.append(child)
            stack.append(child)
    return out


def _test_functions(tree: ast.Module) -> List[_FunctionDef]:
    """Every ``test_*`` function in the module, methods of ``Test*`` classes included."""
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")]


def scan_wall_clock_assertion(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> List[Finding]:
    """Find a ``test_*`` function that asserts on an elapsed wall-clock duration.

    Matches an ``assert`` comparing a clock difference - written inline as ``time.perf_counter() - t0``
    or through the usual ``elapsed = time.time() - start`` variable - against anything. Both bound
    directions are reported: an upper bound fails when the runner is slower than the author's box
    (a cold CI container, a loaded shared runner, an emulated architecture), and a lower bound
    (``assert elapsed >= 0.01``) fails when the clock is coarser than the interval being timed, which
    on Windows' ``time.time()`` is a 15.6ms tick.

    The rule is deliberately blunt because the class is: there is no expression of "this took less
    than N seconds" that is a statement about the code rather than about the machine. A deliberate,
    generously-bounded performance smoke test is a legitimate use and is expected to be BASELINED
    with its reason written down - the value here is the ratchet, which stops the next one arriving
    unexamined, not a demand that the existing ones be deleted.

    Severity: P2. The test is real and its subject is real; only its verdict is the runner's.
    """
    findings: List[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not is_test_file(py, root):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for fn in _test_functions(tree):
            body = list(ast.walk(fn))
            elapsed_names: Set[str] = set()
            for node in body:
                if isinstance(node, ast.Assign) and _is_timer_difference(node.value):
                    elapsed_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
                elif isinstance(node, ast.AnnAssign) and node.value is not None and _is_timer_difference(node.value):
                    if isinstance(node.target, ast.Name):
                        elapsed_names.add(node.target.id)

            for node in body:
                if not isinstance(node, ast.Assert):
                    continue
                for cmp_node in ast.walk(node.test):
                    if not isinstance(cmp_node, ast.Compare):
                        continue
                    if not (_is_duration(cmp_node.left, elapsed_names) or any(_is_duration(c, elapsed_names) for c in cmp_node.comparators)):
                        continue
                    findings.append(
                        Finding(
                            check="wall_clock_assertion",
                            severity="P2",
                            file=rel,
                            line=node.lineno,
                            snippet=_line_text(src_lines, node.lineno),
                            detail=(
                                f"{fn.name}() asserts on an elapsed wall-clock duration, so its verdict is a property of the runner's speed and clock resolution, not of the code under test. Assert the observable outcome, or a ratio against a baseline measured on the same box."
                            ),
                        )
                    )
                    break
    return findings


def scan_sleep_then_assert(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> List[Finding]:
    """Find a ``test_*`` function that sleeps a real interval and then asserts on what happened meanwhile.

    Fires only when the sleep is (a) in the test's OWN body rather than in a nested workload stub,
    (b) a literal of at least ``_WALL_CLOCK_SLEEP_SECONDS``, and (c) followed by an assertion. That
    combination is a test waiting out a background worker: how many samples, retries or callbacks it
    observes is set by how much of the interval the runner actually gave that worker, so the number
    the assertion sees is the elapsed time in disguise. A monitor test of exactly this shape counted
    32, 41 and 54 sampling errors on three CI legs while asserting 1.

    The fix is always the same shape: wait on the event the test is really waiting for
    (``assert done.wait(30)``), which is fast when things work and diagnostic when they do not.

    Severity: P2. It can fail on a slow runner and, worse, pass vacuously on a fast one.
    """
    findings: List[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not is_test_file(py, root):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for fn in _test_functions(tree):
            own = _own_body_nodes(fn)
            sleeps: List[ast.Call] = []
            for node in own:
                if not (isinstance(node, ast.Call) and len(node.args) == 1):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else "")
                if name != "sleep":
                    continue
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and not isinstance(arg.value, bool) and arg.value >= _WALL_CLOCK_SLEEP_SECONDS:
                    sleeps.append(node)
            if not sleeps:
                continue
            first_sleep = min(sleeps, key=lambda c: c.lineno)
            if not any(isinstance(node, ast.Assert) and node.lineno > first_sleep.lineno for node in own):
                continue
            findings.append(
                Finding(
                    check="sleep_then_assert",
                    severity="P2",
                    file=rel,
                    line=first_sleep.lineno,
                    snippet=_line_text(src_lines, first_sleep.lineno),
                    detail=(
                        f"{fn.name}() sleeps a real interval and then asserts on what a background worker managed to do in it, so the asserted quantity is the runner's speed in disguise. Wait on the event itself (an Event/Condition with a generous timeout) instead."
                    ),
                )
            )
    return findings


def _deleted_attributes(node: ast.AST) -> List[Tuple[ast.AST, str, int]]:
    """``(owner_expression, attribute_name, lineno)`` for each attribute this node removes.

    Two spellings remove an attribute: a bare ``del mod.attr``, and ``monkeypatch.delattr(mod, "attr")``.
    A ``delattr`` carrying ``raising=False`` is EXCLUDED: that keyword is precisely the statement
    "this attribute may legitimately not exist here", which is the missing precondition the rule is
    about, so a test that already declares it is correct by construction.
    """
    if isinstance(node, ast.Delete):
        return [(t.value, t.attr, node.lineno) for t in node.targets if isinstance(t, ast.Attribute)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "delattr":
        if any(kw.arg == "raising" and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in node.keywords):
            return []
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            return [(node.args[0], node.args[1].value, node.lineno)]
    return []


def _guarded_lines(fn: ast.AST) -> Set[int]:
    """Line numbers inside a ``try`` or an ``if``/``else`` within ``fn`` - i.e. conditionally executed."""
    guarded: Set[int] = set()
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Try, ast.If)):
            continue
        for inner in ast.walk(node):
            lineno = getattr(inner, "lineno", None)
            if lineno is not None:
                guarded.add(lineno)
    return guarded


def scan_deleted_attribute_read_unconditionally(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> List[Finding]:
    """Find a test that saves an attribute unconditionally and then deletes it to exercise its absence.

    ``unparse = ast.unparse`` followed by ``del ast.unparse`` reads as symmetric bookkeeping, but the
    save is an unguarded attribute access: on the interpreter where the attribute is genuinely
    missing - the one the fallback under test exists FOR - the test raises ``AttributeError`` before
    reaching the code it covers. The dev box, where the attribute exists, is the one environment that
    cannot show this. Exactly this cost a 3.8 leg on a test written to cover 3.8's fallback.

    The correct form states the precondition instead: ``saved = getattr(mod, "attr", None)``, delete
    only when it is there, restore only what was saved - or hand the whole dance to
    ``monkeypatch.delattr(mod, "attr", raising=False)``, which is why that spelling is not flagged.

    Only an UNGUARDED read is reported: a read already inside a ``try`` or an ``if`` has made the
    absence a case it handles.

    Severity: P1. The test errors, rather than failing, on the one environment it was written for.
    """
    findings: List[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not is_test_file(py, root):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = list(ast.walk(fn))
            guarded = _guarded_lines(fn)
            reads: Dict[Tuple[str, str], int] = {}
            for node in body:
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.lineno not in guarded:
                    key = (ast.dump(node.value), node.attr)
                    reads[key] = min(reads.get(key, node.lineno), node.lineno)
            for node in body:
                for owner, attr, lineno in _deleted_attributes(node):
                    first_read: Optional[int] = reads.get((ast.dump(owner), attr))
                    if first_read is None or first_read >= lineno:
                        continue
                    findings.append(
                        Finding(
                            check="deleted_attribute_read_unconditionally",
                            severity="P1",
                            file=rel,
                            line=lineno,
                            snippet=_line_text(src_lines, lineno),
                            detail=(
                                f"{fn.name}() reads .{attr} unguarded at line {first_read} and deletes it here, so on the interpreter or install where .{attr} is genuinely absent - the very one this fallback covers - the test errors before reaching it. Use getattr(..., None), or monkeypatch.delattr(..., raising=False)."
                            ),
                        )
                    )
    return findings
