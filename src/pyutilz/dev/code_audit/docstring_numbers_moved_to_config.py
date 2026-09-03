"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- a docstring still describing constants the code no longer has -----------------------------
#
# Making a hard-coded threshold configurable is a routine, welcome change. Updating the docstring
# that spelled the old number out is the half that gets forgotten, and the result is worse than an
# undocumented knob: the prose confidently states a value that no longer exists anywhere, and the
# reader has no reason to doubt it. Every operator who reads "prunes at 10 hits, 5 for rare
# sources" and goes looking for those numbers is looking for something that was deleted.
#
# Confirmed instance (upwork scrapers, 2026-09-01): a prune helper's docstring described its
# hard-coded 10/5 tiers for a full day after the tiers had been replaced by config lookups -- and
# it was written during a round whose entire subject was this class of defect.
#
# The rule is deliberately narrow, because docstrings cite numbers for many legitimate reasons
# (measured timings, row counts, dates, percentages, cost figures). It fires only when BOTH hold:
#
#   * the docstring cites numbers in threshold-shaped prose, and
#   * the function body contains no numeric literals at all, while it DOES read configuration.
#
# That conjunction is what makes it specific: the numbers did not merely move, they moved somewhere
# a reader cannot see, and the prose is the only remaining record of a value that is no longer real.

# Prose that presents a number as a tunable rather than as a measurement or a date.
_THRESHOLD_PROSE_RE = re.compile(
    r"\b(threshold|thresholds|tier|tiers|limit|limits|cap|capped|at least|at most|more than|"
    r"fewer than|after|every|per|retries|attempts|batch size|interval|timeout|window)\b",
    re.IGNORECASE,
)

# Numbers that are never tunables, however they are phrased.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")

# How configuration is read. Deliberately a small, conventional set: a false NEGATIVE here is a
# missed finding, whereas widening it to any `.get(...)` would make every dict lookup qualify.
_CONFIG_READERS = {"cfg", "config", "getenv", "get_config", "settings", "get_setting", "environ"}


# A docstring line that NAMES the constant or config key holding the value is doing the right
# thing -- it states where the number lives, and the number beside it is an illustration rather
# than a claim. Both of the first version's only two hits across three repos were this shape, e.g.
# "``MIN_WH_RESCAN_FREQ_DAYS`` = 14". Flagging the good pattern is worse than flagging nothing.
# `e.g.`/`i.e.` and a bare `run.py` are prose, not a named source: the dotted-name alternative
# used to match them and discard the whole line, silencing the rule on any sentence containing
# one. Both sides need >= 2 characters and the two abbreviations are excluded outright.
_ABBREVIATION_RE = re.compile(r"\b(?:e\.g|i\.e|etc|vs|cf)\.", re.IGNORECASE)
# A FILE name is not a named source either -- "per run.py invocation" says nothing about
# where the number lives.
_FILENAME_RE = re.compile(r"\b[\w-]+\.(?:py|pyi|sql|json|ya?ml|toml|ini|cfg|txt|md|csv|log)\b", re.IGNORECASE)
_NAMES_A_SOURCE_RE = re.compile(r"[A-Z][A-Z0-9_]{3,}|\b[a-z_]{2,}\.[a-z_]{2,}\b|``[^`]+``")

# "audit 04.1", "round 13", "wave 20" -- references to a document, not a tunable. They sit in prose
# that routinely also contains words like "after" and "every", which is how one got through.
_REFERENCE_RE = re.compile(r"\b(audit|round|wave|phase|item|issue|ticket|section|step|figure)\s+[\d.]+", re.IGNORECASE)


def _docstring_numbers(doc: str) -> list[str]:
    """Numbers cited in threshold-shaped prose, excluding years, references and sourced values.

    Works line by line so that a docstring mixing a measurement paragraph with a threshold
    paragraph contributes only the latter, and requires the number to sit CLOSE to the threshold
    word -- sharing a long line with the word "after" is not evidence of anything.
    """
    found: list[str] = []
    for line in doc.splitlines():
        if _NAMES_A_SOURCE_RE.search(_FILENAME_RE.sub(" ", _ABBREVIATION_RE.sub(" ", line))):
            continue
        cleaned = _REFERENCE_RE.sub(" ", _YEAR_RE.sub(" ", line))
        for keyword in _THRESHOLD_PROSE_RE.finditer(cleaned):
            window = cleaned[max(0, keyword.start() - 30) : keyword.end() + 30]
            for match in _NUMBER_RE.finditer(window):
                value = match.group()
                if value in ("0", "1"):
                    continue  # too ordinary to carry meaning ("every 1 of", "0 rows")
                found.append(value)
    return found


def _body_numbers(func: ast.AST) -> set[str]:
    """Every numeric literal in the function body, as text, so "5" matches 5 and 5.0 matches 5.0."""
    numbers: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            numbers.add(str(node.value))
            if isinstance(node.value, float) and node.value.is_integer():
                numbers.add(str(int(node.value)))
    return numbers


def _reads_configuration(func: ast.AST) -> bool:
    """Does the body fetch a value from configuration or the environment?"""
    for node in ast.walk(func):
        # `os.environ["PRUNE"]` is the normal spelling for `environ`, which is in _CONFIG_READERS;
        # only matching Call nodes saw the `.get(...)` form and nothing else.
        if isinstance(node, ast.Subscript):
            receiver = node.value
            if isinstance(receiver, ast.Attribute) and receiver.attr in _CONFIG_READERS:
                return True
            if isinstance(receiver, ast.Name) and receiver.id in _CONFIG_READERS:
                return True
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        names = set()
        while isinstance(target, ast.Attribute):
            names.add(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            names.add(target.id)
        if isinstance(target, ast.Call) and isinstance(target.func, ast.Name):
            names.add(target.func.id)  # the `cfg().get(...)` spelling
        if names & _CONFIG_READERS:
            return True
    return False


def scan_docstring_numbers_moved_to_config(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find docstrings that still spell out constants the function now reads from configuration.

    Fires only when the docstring cites numbers in threshold-shaped prose AND the body holds no
    numeric literals at all while reading configuration -- the signature of a knob that was made
    configurable without updating the prose describing its old value.

    The prose is then the only surviving record of a number that no longer exists, which is worse
    than no documentation: a reader has no reason to distrust it, and an operator goes looking in
    the code for a value that was deleted.

    Not flagged: a docstring citing measurements, dates, or row counts (no threshold phrasing), and
    a function that still contains numeric literals -- if the numbers are in the body, the prose can
    be checked against them by reading.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            cited = _docstring_numbers(doc)
            if not cited:
                continue
            if _body_numbers(node):
                continue  # numbers are still visible in the code; prose can be checked by reading
            if not _reads_configuration(node):
                continue
            findings.append(
                Finding(
                    check="docstring_numbers_moved_to_config",
                    severity="Low",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"`{node.name}`'s docstring states threshold value(s) {sorted(set(cited))}, but the "
                        "body contains no numeric literals and reads its values from configuration -- the "
                        "prose is the only remaining record of numbers that no longer exist. State the "
                        "config keys instead of the values, so the documentation cannot go stale again."
                    ),
                )
            )
    return findings
