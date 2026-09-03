"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from ._base import SEVERITIES, Finding, UNKNOWN_SEVERITY_RANK, _DEFAULT_EXCLUDE_DIRS, severity_rank
from .registry import get_check_aliases, get_scanners, render_check_catalogue, run_all

# --- CLI ----------------------------------------------------------------


def _render_markdown(findings: list[Finding]) -> str:
    """Renders findings as a Markdown table (severity/check/location/snippet/detail), or a "no findings" note if empty."""
    if not findings:
        return "_No findings._\n"
    lines = [
        "| Sev | Check | File:Line | Snippet | Detail |",
        "|---|---|---|---|---|",
    ]
    lines.extend(f.as_md_row() for f in findings)
    return "\n".join(lines) + "\n"


def _render_json(findings: list[Finding]) -> str:
    """Renders findings as a sorted-keys, indented JSON array of dataclass dicts."""
    return json.dumps([asdict(f) for f in findings], indent=2, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``python -m pyutilz.dev.code_audit``.

    Parses argv (uses ``sys.argv[1:]`` when ``None``) into a ``root``
    directory, an optional list of ``--check`` scanner names, an output
    format (markdown or JSON), and a minimum severity threshold. Runs
    the selected scanners against ``root``, prints the rendered findings
    to stdout, and returns an exit code: ``1`` when any P0 or P1 finding
    is present (so CI can gate on the result), ``0`` otherwise.

    Returns the exit code rather than calling ``sys.exit`` directly so
    the function is testable from a process.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pyutilz.dev.code_audit",
        description=(
            "AST audit: mutable defaults, late-binding closures, "
            "default-via-or trap, silent broad-except swallows, "
            "logged-but-not-escalated excepts, SQL LIMIT-without-ORDER-BY, "
            "OFFSET-pagination advisories, dead CLI flags, "
            "non-idempotent SQL migrations, duplicate conditions "
            "(copy-paste typos in and/or chains, elif branches, dict "
            "keys), and discarded coroutines (missed await). "
            "Designed to be run against any Python source tree, not "
            "just pyutilz."
        ),
    )
    parser.add_argument("root", type=Path, nargs="?", help="source-tree root to scan (e.g. ./src)")
    # The catalogue is generated from the registry, so it can never fall behind the checks that
    # actually run -- which the package docstring, hand-maintained, repeatedly did.
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="print every registered check with its one-line summary (opt-in ones marked) and exit.",
    )
    # The emitted ``Finding.check`` ids are accepted too, so a check id read off a report row is
    # always runnable -- several scanners emit an id that differs from their registry key.
    selectable = sorted(set(get_scanners()) | set(get_check_aliases()))
    parser.add_argument(
        "--check",
        action="append",
        choices=selectable,
        help=("scanner(s) to run; repeat for multiple. Default: run all. " "Available: " + ", ".join(selectable)),
    )
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown",
        help="output format (default markdown).",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=None,
        help=("directory name to exclude (matched against any path part). " "Repeat. Adds to the default set of build/cache/venv dirs."),
    )
    parser.add_argument(
        "--min-severity", choices=SEVERITIES, default="Low",
        help="filter out findings below this severity (default Low: show all).",
    )
    args = parser.parse_args(argv)

    if args.list_checks:
        sys.stdout.write(render_check_catalogue())
        return 0
    if args.root is None:
        parser.error("root is required (or pass --list-checks)")

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"root must be a directory: {root}")

    exclude_dirs = frozenset(_DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir or ()))

    all_findings = run_all(root, checks=args.check, exclude_dirs=exclude_dirs)
    cutoff = severity_rank(args.min_severity)
    # An unrecognised severity ranks -1, ABOVE P0: it renders at every --min-severity setting and
    # gates the exit code, rather than sorting last and being filtered out of every report unseen.
    unknown = sorted({f.severity for f in all_findings if severity_rank(f.severity) == UNKNOWN_SEVERITY_RANK})
    if unknown:
        sys.stderr.write(f"warning: findings carry severities outside {list(SEVERITIES)}: {unknown}\n")
    findings = [f for f in all_findings if severity_rank(f.severity) <= cutoff]

    out = _render_json(findings) if args.format == "json" else _render_markdown(findings)
    sys.stdout.write(out)
    # exit code computed from the UNFILTERED findings: --min-severity only controls what's
    # rendered, it must never weaken the CI gate.
    return 1 if any(severity_rank(f.severity) <= severity_rank("P1") for f in all_findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
