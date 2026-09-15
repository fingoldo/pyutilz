"""pyutilz supports python 3.8, and two ``ast`` differences there fail only on the 3.8 legs.

* ``ast.unparse`` exists only from 3.9: an unguarded call raises AttributeError on 3.8.
* Up to 3.8 the parser wraps a subscript's contents in ``ast.Index``; reading ``Subscript.slice`` and expecting
  the inner expression matches nothing there (a rule silently goes dead) or loops (a helper that returns its
  input unchanged, then recurses on it).

Both reached CI in the code_audit scanners. The fixes are ``getattr(ast, "unparse", ast.dump)`` and
``_base._subscript_index``; this test keeps new code on them, on every leg rather than only on 3.8.
"""
from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "pyutilz"


def _mentions(node: ast.AST, needle: str) -> bool:
    """True if *needle* appears as a name, attribute or string anywhere under *node*."""
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Name) and sub.id == needle) or (isinstance(sub, ast.Attribute) and sub.attr == needle):
            return True
        if isinstance(sub, ast.Constant) and sub.value == needle:
            return True
    return False


def _violations(src: str) -> "list[tuple[int, str]]":
    """``(line, what)`` for each 3.8-unsafe ``ast.unparse`` call or ``Subscript.slice`` read in *src*."""
    tree = ast.parse(src)
    parents: "dict[ast.AST, ast.AST]" = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def enclosing(node: ast.AST, kinds: tuple) -> "list[ast.AST]":
        chain = []
        while node in parents:
            node = parents[node]
            if isinstance(node, kinds):
                chain.append(node)
        return chain

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr == "unparse" and isinstance(node.value, ast.Name) and node.value.id == "ast":
            # Guarded when an enclosing conditional tests hasattr(ast, "unparse").
            tests = [n.test for n in enclosing(node, (ast.If, ast.IfExp))]
            if not any(_mentions(t, "hasattr") and _mentions(t, "unparse") for t in tests):
                found.append((node.lineno, "ast.unparse"))
        elif node.attr == "slice" and isinstance(node.ctx, ast.Load):
            # Guarded when the enclosing function handles the 3.8 wrapper itself (or IS the shared helper).
            funcs = enclosing(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if not (funcs and _mentions(funcs[0], "Index")):
                found.append((node.lineno, ".slice"))
    return sorted(found)  # ast.walk is breadth-first, not source order


def test_the_check_tells_guarded_from_unguarded():
    unsafe = '''
def label(node):
    return ast.unparse(node)

def key(sub):
    return sub.slice
'''
    safe = '''
def label(node):
    return getattr(ast, "unparse", ast.dump)(node)

def label2(node):
    return ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)

def key(sub):
    index = sub.slice
    return index.value if isinstance(index, ast.Index) else index
'''
    assert [what for _, what in _violations(unsafe)] == ["ast.unparse", ".slice"]
    assert _violations(safe) == []


def test_no_py38_unsafe_ast_use_in_the_package():
    offenders = []
    for py in sorted(_SRC.rglob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        if "unparse" not in text and ".slice" not in text:
            continue
        offenders += [f"{py.relative_to(_SRC).as_posix()}:{line} {what}" for line, what in _violations(text)]
    assert offenders == [], (
        "python 3.8 is supported: use getattr(ast, 'unparse', ast.dump)(node) instead of ast.unparse, and "
        "_base._subscript_index(node) instead of reading Subscript.slice directly:\n  " + "\n  ".join(offenders)
    )
