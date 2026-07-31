"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse

# --- round-trip-detectable mojibake ----------------------------------------

# Maximal runs of consecutive non-ASCII characters shorter than this are skipped: a lone accented
# character (rare but not impossible in a name/URL) round-tripping by coincidence is far more
# plausible at length 1 than a multi-character run doing so.
_MIN_RUN_LENGTH = 2

_NON_ASCII_RUN_RE = re.compile(r"[^\x00-\x7F]+")


def _is_roundtrip_mojibake(run: str) -> bool:
    """True if ``run`` round-trips cleanly through cp1251-encode -> utf-8-decode into DIFFERENT, non-empty text."""
    if len(run) < _MIN_RUN_LENGTH:
        return False
    try:
        recovered = run.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    return bool(recovered) and recovered != run


def scan_mojibake(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find source lines containing round-trip-detectable mojibake (UTF-8 bytes misread as
    CP1251/similar and re-saved -- a lossy clipboard/terminal round-trip is the usual cause).

    Detection principle: genuine mojibake of this kind is round-trip DETECTABLE. The corrupted
    text is itself valid Unicode (e.g. Cyrillic-block characters), so ``run.encode("cp1251")``
    succeeds; because those bytes are the ORIGINAL character's real UTF-8 encoding misread one
    byte at a time, decoding them back as UTF-8 (``.decode("utf-8")``) ALSO succeeds and recovers
    legible text different from the input. This double round-trip succeeding is a strong,
    low-false-positive signal: deliberately-authored non-ASCII text (e.g. a genuine Russian
    comment) also survives ``.encode("cp1251")`` but its bytes essentially never form valid UTF-8
    when decoded back, so it is correctly NOT flagged; a single already-correct special character
    with no CP1251 mapping (e.g. ``>=`` written as U+2265) fails at the encode step and is also
    correctly NOT flagged.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        # _safe_parse also validates the file is readable/parseable Python; reuse it as the gate,
        # then re-read the raw text for line-by-line scanning (this check is text-based, not AST).
        if _safe_parse(py) is None:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = py.relative_to(root).as_posix()
        for lineno, line in enumerate(text.split("\n"), start=1):
            for match in _NON_ASCII_RUN_RE.finditer(line):
                if _is_roundtrip_mojibake(match.group()):
                    findings.append(
                        Finding(
                            check="mojibake",
                            severity="P2",
                            file=rel,
                            line=lineno,
                            snippet=line.strip(),
                            detail=("line contains round-trip-detectable mojibake (UTF-8 misread as CP1251 " "and re-saved). Re-save the file as clean UTF-8."),
                        )
                    )
                    break  # one flag per line is enough to locate it
    return findings
