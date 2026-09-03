"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, is_test_file, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- asserting a FORMATTED string against a lazily-formatted log record ------------------------
#
# Production logs `log.warning("%s: reached only %d/%d", label, reached, total)`. A test replaces
# `log` with a MagicMock and asserts `"reached only 0/3" in str(log.warning.call_args)`. That can
# never match: `%`-formatting is deferred until a handler emits the record, so the numbers live in
# `call_args.args[2:]` and never appear in the format string.
#
# The test then passes for the only reason a never-matching `in` can pass -- it doesn't, unless the
# assertion is `not in`, or the list comprehension it feeds comes back empty and the test asserts
# on emptiness. Either way it stops observing what it names.
#
# The discriminator is cheap and almost noise-free: a digit. Lazy `%`-formatting means digits
# essentially never appear in a format STRING -- they appear in the arguments. So an asserted
# literal that contains a digit, matched against a mock logger's call record, is either this bug or
# a format string that genuinely carries a digit ("HTTP 429 from %s"), and the second case is
# excluded by checking the package's actual format strings.

_LOG_LEVELS = {"debug", "info", "warning", "error", "exception", "critical", "log"}

# `call_args`, `call_args_list`, `mock_calls` -- the record of what a mock was called with.
_CALL_RECORD_ATTRS = {"call_args", "call_args_list", "mock_calls", "call_count"}

_HAS_DIGIT = re.compile(r"\d")


def _format_strings(root: Path, exclude_dirs: frozenset[str]) -> tuple[set[str], list[list[str]]]:
    """Logger message formats, split by WHEN they are formatted.

    Returns (lazy `%`-style formats, eager f-string literal-part lists).

    The distinction is the whole rule. `log.warning("%s: only %d", a, b)` defers formatting,
    so the values never reach the record. `log.warning(f"{a}: only {b}")` formats EAGERLY,
    so they do -- and an assertion on the rendered text is correct there. Lumping the two
    together reported two honest f-string assertions as bugs on the first real run.
    """
    formats: set[str] = set()
    eager: list[list[str]] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in _LOG_LEVELS or not node.args:
                continue
            # `logger.log(LEVEL, fmt, ...)` puts the level first: the format string is args[1].
            # Harvesting args[0] there collected the level object as a "format", so the real format
            # was never known and every assertion against it was reported.
            index = 1 if node.func.attr == "log" else 0
            if len(node.args) <= index:
                continue
            first = node.args[index]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                formats.add(first.value)
            elif isinstance(first, ast.JoinedStr):
                eager.append([p.value for p in first.values if isinstance(p, ast.Constant) and isinstance(p.value, str)])
    return formats, eager


def _could_be_an_eager_rendering(literal: str, eager: list[list[str]]) -> bool:
    """Could *literal* appear in some f-string's rendered output?

    An f-string renders its literal parts with values spliced between them, so an asserted
    string that spans an interpolation -- "Found 2 on-disk checkpoint" against
    f"Found {n} on-disk checkpoint(s)" -- matches no single part while being perfectly
    correct. Fragments of the literal, in order, are what has to line up.
    """
    fragments = [f for f in re.split(r"\d+", literal) if len(f.strip()) > 2]
    if not fragments:
        return False
    for parts in eager:
        joined = "\x00".join(parts)
        cursor = 0
        for fragment in fragments:
            found = joined.find(fragment, cursor)
            if found < 0:
                break
            cursor = found + len(fragment)
        else:
            return True
    return False


def _touches_a_log_record(node: ast.AST) -> str | None:
    """Name the mock-logger call record this expression reaches, or None.

    Requires BOTH a logger level and a call-record attribute in the same expression, so
    `log.warning.call_args` matches and a bare `mock.call_args` on some unrelated mock does not.
    """
    attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    record = attrs & _CALL_RECORD_ATTRS
    if record and (attrs & _LOG_LEVELS):
        return sorted(record)[0]
    return None


_UNITTEST_ASSERTIONS = frozenset({"assertIn", "assertNotIn", "assertTrue", "assertFalse", "assertEqual", "assertNotEqual"})


def _assertion_expression(node: ast.AST) -> ast.expr | None:
    """The expression an assertion tests, for both the `assert` statement and unittest's methods.

    A suite written with `self.assertIn('reached only 0/3', str(log.warning.call_args))` carries
    exactly the defect this rule names, and reading only `ast.Assert` never saw it.
    """
    if isinstance(node, ast.Assert):
        return node.test
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _UNITTEST_ASSERTIONS and node.args:
        return node
    return None


def _asserted_literals(test: ast.AST) -> list[str]:
    """String literals this assertion compares against something."""
    out: list[str] = []
    for node in ast.walk(test):
        if isinstance(node, ast.Compare):
            out.extend(side.value for side in [node.left, *node.comparators] if isinstance(side, ast.Constant) and isinstance(side.value, str))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _UNITTEST_ASSERTIONS:
            out.extend(arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    return out


def scan_lazy_log_assertion(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find assertions matching a formatted string against a lazily-formatted log record.

    `log.warning("%s: reached only %d/%d", label, r, t)` puts the numbers in `call_args.args[2:]`,
    never in the format string, so `"reached only 0/3" in str(log.warning.call_args)` cannot match.

    Reported when the asserted literal contains a DIGIT -- lazy `%`-formatting means digits live in
    the arguments, not the format -- and no logger format string in the package contains that
    literal. The second clause excludes the honest case of a format that really does carry a digit,
    such as `"HTTP 429 from %s"`.

    The fix is to read the arguments: `c.args[0]` for the format, `c.args[2]` for the value.
    """
    findings: list[Finding] = []
    formats, eager = _format_strings(root, exclude_dirs)

    for py in _iter_py_files(root, exclude_dirs):
        if not is_test_file(py, root):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()

        for node in ast.walk(tree):
            # `assert x in y` and the unittest spelling `self.assertIn(x, y)` state the same thing.
            expression = _assertion_expression(node)
            if expression is None:
                continue
            assertion_line: int = getattr(node, "lineno", expression.lineno)
            record = _touches_a_log_record(expression)
            if record is None:
                continue
            for literal in _asserted_literals(expression):
                if not _HAS_DIGIT.search(literal):
                    continue
                # The literal must carry MESSAGE text, not just a value. `"j1"` is a job id the
                # test supplied, and production logs it through an f-string, so it genuinely is
                # in `args[0]`. Only a literal that also contains static prose can be the "I
                # expected the format to be rendered" mistake this rule is about.
                if not any(len(f.strip()) > 2 for f in re.split(r"\d+", literal)):
                    continue
                if any(literal in fmt for fmt in formats):
                    continue  # a lazy format really does carry this text, digit and all
                if _could_be_an_eager_rendering(literal, eager):
                    continue  # an f-string renders its values, so the assertion can match
                findings.append(
                    Finding(
                        check="lazy_log_assertion",
                        severity="P2",
                        file=rel,
                        line=assertion_line,
                        snippet=_line_text(src_lines, assertion_line),
                        detail=(
                            f"asserts the literal {literal!r} against `{record}`, but no logger "
                            "format string in this package contains it and it carries a digit -- "
                            "`%`-formatting is deferred, so the values live in the call's ARGS and "
                            "never in the format. Read `c.args[0]` for the format and `c.args[N]` "
                            "for the value instead."
                        ),
                    )
                )
    return findings
