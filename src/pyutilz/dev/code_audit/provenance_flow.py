"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import difflib
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a record field with only one side of the contract --------------------------------------------
#
# Two mirrored defects, confirmed together (Autopsia, 2026-07-26):
#
# - READ, NEVER WRITTEN. `triple.get("mechanism_modality", "")` reads a key no producer writes, so it
#   is always the default and the branch behind it never fires. `getattr(p, "quote_proves_relation",
#   True)` reads an attribute the dataclass does not declare, so the permissive default always wins
#   and a verification flag that exists elsewhere in the pipeline silently stops gating anything.
#   This direction is a bug on every run and is reported at P1.
#
# - WRITTEN, NEVER READ. `char_span`, `quote_tier`, `verbatim_rate`, `arguments_swapped`,
#   `relation_raw`, `shape_repairs`, `field_validation`, `modality_source`, `modality_model`,
#   `source_citation` -- all populated on every row, none consumed. Provenance that nothing reads is
#   provenance nobody can rely on; the cost is paid and the guarantee is not delivered. P2, OFF by
#   default: measured over that project it returned 732 hits, because a serialised envelope
#   legitimately carries fields for a consumer outside the scanned tree and this scan cannot see one.
#
# The default-bearing read is what makes the first direction dangerous, so a `.get(k)` or
# `getattr(o, k)` WITHOUT a default is not reported: it raises, which is a loud failure, not a silent one.

_MIN_KEY_LEN = 3  # single letters and two-letter keys are loop variables and axis labels, not record fields


def _is_field_like(value: object) -> bool:
    """Whether the name looks like a record field rather than a local: no leading underscore, snake case."""
    return isinstance(value, str) and len(value) >= _MIN_KEY_LEN and value.replace("_", "").isalnum() and not value[0].isdigit()


def _collect(tree: ast.Module) -> tuple[dict[str, int], dict[str, int]]:
    """(writes, defaulted_reads) as key -> first line number."""
    writes: dict[str, int] = {}
    reads: dict[str, int] = {}

    def note(store: dict[str, int], key: object, line: int) -> None:
        """Record one read-without-writer site, keyed so the same field is not reported twice."""
        if _is_field_like(key) and key not in store:
            store[str(key)] = line

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant):
                    note(writes, k.value, node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    note(writes, target.slice.value, node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "setdefault" and node.args and isinstance(node.args[0], ast.Constant):
                note(writes, node.args[0].value, node.lineno)
            elif node.func.attr == "get" and len(node.args) == 2 and isinstance(node.args[0], ast.Constant):
                note(reads, node.args[0].value, node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) == 3:
            if isinstance(node.args[1], ast.Constant):
                note(reads, node.args[1].value, node.lineno)
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.ctx, ast.Load):
            note(reads, node.slice.value, node.lineno)
    return writes, reads


def scan_record_field_flow(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    known_external_fields: frozenset[str] = frozenset(),
    near_miss_only: bool = True,
    report_written_never_read: bool = False,
) -> list[Finding]:
    """Find record/envelope fields with a writer and no reader, or a defaulted reader and no writer.

    ``known_external_fields`` exempts keys that belong to somebody else's schema. Keep it short: a
    long exemption list means the scan is pointed at the wrong tree, not that the rule is wrong.

    ``near_miss_only`` (default True) restricts the read-never-written direction to keys that closely
    resemble a key the tree DOES write -- ``mechanism_modality`` beside ``modality_source``. Measured
    on a real project, the unrestricted form returned 93 hits of which the overwhelming majority were
    keys of somebody else's JSON (an NCBI response, an LLM payload, an environment variable), because
    "this dict came from outside" is not statically decidable. The near-miss restriction keeps the
    shape that is unambiguously a defect - you meant the other field - and discards the rest. Pass
    False for the exhaustive form when auditing a tree with no external boundaries.

    ``report_written_never_read`` (default False) enables the mirror direction. It is off by default
    because on that same project it returned 732 hits: every serialised artefact field, every
    argparse namespace key, every payload assembled for an outbound call. A rule needing an allowlist
    longer than itself does not survive contact with a team.

    Severity: P1 for a defaulted read of a key nothing writes (the default wins on every run, so the
    code behind it is unreachable), P2 for a written key nothing reads.
    """
    writes: dict[str, tuple[str, int]] = {}
    reads: dict[str, tuple[str, int]] = {}
    lines_by_file: dict[str, list[str]] = {}

    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        rel = py.relative_to(root).as_posix()
        lines_by_file[rel] = py.read_text(encoding="utf-8", errors="replace").splitlines()
        file_writes, file_reads = _collect(tree)
        for key, line in file_writes.items():
            writes.setdefault(key, (rel, line))
        for key, line in file_reads.items():
            reads.setdefault(key, (rel, line))

    findings: list[Finding] = []
    written_names = sorted(set(writes))
    for key in sorted(set(reads) - set(writes) - known_external_fields):
        near = difflib.get_close_matches(key, written_names, n=1, cutoff=0.85)
        if near_miss_only and not near:
            continue
        rel, line = reads[key]
        near_note = f" Did you mean {near[0]!r}?" if near else ""
        findings.append(
            Finding(
                check="field_read_never_written",
                severity="P1",
                file=rel,
                line=line,
                snippet=_line_text(lines_by_file.get(rel, []), line),
                detail=(f"{key!r} is read with a default but nothing in this tree writes it -- the default wins on every run, so whatever depends on it never fires.{near_note}"),
            )
        )
    if not report_written_never_read:
        return findings
    for key in sorted(set(writes) - set(reads) - known_external_fields):
        rel, line = writes[key]
        findings.append(
            Finding(
                check="field_written_never_read",
                severity="P2",
                file=rel,
                line=line,
                snippet=_line_text(lines_by_file.get(rel, []), line),
                detail=f"{key!r} is written but never read -- provenance nobody consumes costs storage and delivers no guarantee.",
            )
        )
    return findings
