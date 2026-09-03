"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- tests that assert on SOURCE TEXT rather than on behaviour -----------
#
# A test that reads the source of the code under test and asserts a substring appears in it passes
# whether or not that code ever runs. This is not a stylistic objection; it has shipped defects
# twice in the projects this package audits:
#
# * A dashboard fix asserted via `inspect.getsource(...)` that a query carried a new WHERE clause.
#   The clause was there. The function was unreachable. The test was green for a full audit round
#   while the fix did nothing, and the next round found it only by measuring the live result.
# * A scraper fix asserted a SQL constant contained `AT TIME ZONE 'utc'`. It did. The call site
#   passed a different constant.
#
# Both tests were written specifically to prove a fix had landed, and both proved only that a
# string existed in a file. What makes the shape worth a mechanical check rather than a review
# habit is that it LOOKS like a strong test: it names the exact thing that was changed.
#
# The honest version calls the function and asserts on what comes back -- or, when the thing under
# test really is a string (a generated migration, a rendered template), asserts on the string the
# PRODUCTION path produced, not on the source that defines it.

# Callables whose return value is program text about the program itself.
_SOURCE_READERS = {
    "getsource",
    "getsourcelines",
    "getsourcefile",
    "getclosurevars",
    "unparse",  # ast.unparse(...) round-trips source just as readably
}

# `dis` output is source text by another name: an assertion on it pins bytecode spelling, not
# behaviour, and breaks or passes for reasons unrelated to what the code does.
_DISASSEMBLERS = {"dis", "get_instructions", "code_info"}

# Attributes that hand back a function's own text/bytecode.
#
# `__code__` and `__wrapped__` are deliberately NOT here. Reaching through them is how a test gets
# at the real callable behind a decorator -- `wrapped.__code__.co_freevars.index("func")` to pull a
# closure cell, and then CALL what it finds. That is behavioural testing wearing introspective
# clothes, and including `__code__` mislabelled exactly such a test as a source-text claim. Only
# the attributes that hand back program CONTENT belong here.
_SOURCE_ATTRS = {"co_code", "co_consts", "co_names"}


def _dis_aliases(tree: ast.AST) -> set[str]:
    """Local names that refer to the ``dis`` module, including ``import dis as d``."""
    names = {"dis"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dis":
                    names.add(alias.asname or "dis")
    return names


def _reads_source(node: ast.AST, dis_aliases: "set[str] | None" = None) -> str | None:
    """Name the source-reading construct inside *node*, or None.

    Matches `inspect.getsource(f)`, a bare `getsource(f)` (it is imported that way often enough
    to matter), `ast.unparse(...)`, the `dis` family, and `f.__code__`.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _SOURCE_READERS:
                return f"{name}()"
            if name in _DISASSEMBLERS and isinstance(func, ast.Attribute):
                mod = func.value
                if isinstance(mod, ast.Name) and mod.id in (dis_aliases or {"dis"}):
                    return f"dis.{name}()"
        if isinstance(sub, ast.Attribute) and sub.attr in _SOURCE_ATTRS:
            return sub.attr
    return None


def _reads_a_python_file(node: ast.AST) -> bool:
    """`Path("...py").read_text()` / `open("...py").read()` -- reading source the long way round.

    Requires a literal `.py`/`.sql` path somewhere in the expression, so reading a fixture, a JSON
    blob, or a log file is not caught.
    """
    has_read = False
    has_code_path = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in ("read_text", "read", "readlines", "read_bytes", "open"):
                has_read = True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            lowered = sub.value.lower()
            if lowered.endswith(".py") or lowered.endswith(".sql"):
                has_code_path = True
    return has_read and has_code_path


def _is_membership_or_match(test: ast.AST) -> bool:
    """Is this assertion about text CONTENT -- `in`, `.count()`, a regex search, `==` on a string?

    An assertion that merely calls a source reader and checks it is not None says nothing about
    content, and is usually a smoke test for the reader itself; that is not the defect.
    """
    for sub in ast.walk(test):
        if isinstance(sub, ast.Compare):
            for op in sub.ops:
                if isinstance(op, (ast.In, ast.NotIn, ast.Eq, ast.NotEq)):  # codespell:ignore
                    return True
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in ("count", "search", "match", "findall", "index", "startswith", "endswith"):
                return True
    return False


def _source_bound_names(tree: ast.AST, dis_aliases: "set[str] | None" = None) -> dict[str, str]:
    """Map local names bound from a source reader to the construct that produced them.

    This is the spelling that actually occurs. Almost nobody writes the reader inline inside the
    assertion; they write::

        src = inspect.getsource(cc._load_all_ct01)
        assert "sys.stderr" in src

    The first version of this scanner examined only the assert's own expression and therefore found
    ZERO offences in a repo that had several, including one asserting a fix was present in a
    function's text. A check that misses the common spelling of its own defect is worse than none,
    because its silence reads as a clean bill of health.
    """
    bound: dict[str, str] = {}
    # Aliases come from the MODULE, not from this scope: `import dis as d` is a module-level statement.
    aliases = dis_aliases if dis_aliases is not None else _dis_aliases(tree)
    for node in ast.walk(tree):
        # AnnAssign too: `src: str = inspect.getsource(g)` is the same binding, annotated.
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        reader = _reads_source(node.value, aliases)
        if reader is None and _reads_a_python_file(node.value):
            reader = "reading a .py/.sql file"
        if reader is None:
            continue
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    bound[name.id] = reader
    return bound


def _scopes(tree: ast.Module) -> Iterator[ast.AST]:
    """Yield each function body as its own scope, plus the module itself.

    Binding must be scoped, not file-wide. `src` and `fn` are ordinary local names that recur in
    every other test in a file; matching them file-wide made a plainly behavioural assertion
    (`assert fn("tabMarket") == "body:tabMarket"`) look like a source-text claim because some
    unrelated test three hundred lines away had bound `fn` from a reader.
    """
    # INNERMOST FIRST. `ast.walk` is breadth-first, so it yields an outer function before the
    # functions nested in it; combined with the caller's `seen` set that locked every assert to
    # the OUTERMOST scope claiming it, so a sibling inner function's bindings leaked across --
    # exactly the false positive this scoping exists to prevent. Reversing the BFS order puts the
    # narrowest enclosing scope first.
    funcs = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    yield from reversed(funcs)
    yield tree


def _looks_like_a_test_file(path: Path, root: Path) -> bool:
    """Whether this file is test code.

    Only test files are scanned: a build script or a code generator manipulates source text as
    its actual job, and flagging it would be noise.

    The ``tests`` directory test is made against the path RELATIVE to the scan root: ``path.parts``
    is absolute, so a checkout under any directory named ``tests`` classified every production
    file in the repository as a test.
    """
    name = path.name
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in rel_parts


def _fails_in_body(node: ast.If) -> bool:
    """Does this ``if``'s body do nothing but call ``pytest.fail`` / ``self.fail``? Then it is an assertion."""
    for stmt in node.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "fail":
                return True
    return False


def scan_source_text_assertions(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find test assertions made against the SOURCE TEXT of the code under test.

    Flags an ``assert`` (or ``pytest.fail`` guard) whose condition both reads program text --
    ``inspect.getsource``, ``ast.unparse``, ``dis``, ``f.__code__``, or reading a ``.py``/``.sql``
    file -- and tests that text's CONTENT with ``in`` / ``==`` / a regex.

    Such a test passes whether or not the code it inspects ever executes, so it certifies a fix
    that shipped unreachable. Both times this happened in the audited projects, the test had been
    written for the express purpose of proving the fix had landed.

    Not flagged: reading source without asserting on its content (a smoke test for the reader), and
    non-test files -- a build script or a code generator legitimately manipulates source text.

    Some assertions on source text ARE the point: a meta-test that bans a pattern across the
    codebase (including this scanner's own tests) must read source to do its job. Those belong in
    a reasoned allowlist at the project level, which is why this scanner reports rather than
    forbids -- the baseline mechanism is where an accepted instance is recorded.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not _looks_like_a_test_file(py, root):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        seen: set[int] = set()
        aliases = _dis_aliases(tree)
        for scope in _scopes(tree):
            bound_here = _source_bound_names(scope, aliases)
            for node in ast.walk(scope):
                # `if "x" not in src: pytest.fail(...)` is the same source-text assertion in a
                # different spelling; the docstring has always claimed to handle it.
                if not (isinstance(node, ast.Assert) or (isinstance(node, ast.If) and _fails_in_body(node))) or id(node) in seen:
                    continue
                seen.add(id(node))
                reader = _reads_source(node.test, aliases)
                if reader is None and _reads_a_python_file(node.test):
                    reader = "reading a .py/.sql file"
                if reader is None:
                    # The far more common spelling: read into a variable, assert on the variable.
                    #
                    # A name that is CALLED in the assertion is exempt: calling it exercises
                    # behaviour, which is the thing this check wants tests to do.
                    called = {c.func.id for c in ast.walk(node.test) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
                    for used in ast.walk(node.test):
                        if isinstance(used, ast.Name) and used.id in bound_here and used.id not in called:
                            reader = f"{bound_here[used.id]} via `{used.id}`"
                            break
                if reader is None:
                    continue
                if not _is_membership_or_match(node.test):
                    continue
                how = reader
                findings.append(Finding(
                    check="source_text_assertion",
                    severity="P2",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"assertion is made against SOURCE TEXT (via {how}) rather than against "
                        "behaviour -- it passes whether or not the inspected code ever runs, which is "
                        "how a fix has twice shipped unreachable and stayed green. Call the code and "
                        "assert on what it returns or does; if the artefact under test really is a "
                        "string, assert on the string the PRODUCTION path produced."
                    ),
                ))
    return findings
