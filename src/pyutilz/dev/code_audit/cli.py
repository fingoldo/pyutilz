"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS
from .registry import SCANNERS, run_all

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
            "default-via-or trap, silent broad-except swallows. "
            "Designed to be run against any Python source tree, "
            "not just pyutilz."
        ),
    )
    parser.add_argument("root", type=Path, help="source-tree root to scan (e.g. ./src)")
    parser.add_argument(
        "--check",
        action="append",
        choices=sorted(SCANNERS),
        help=("scanner(s) to run; repeat for multiple. Default: run all. " "Available: " + ", ".join(sorted(SCANNERS))),
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
        "--min-severity", choices=("P0", "P1", "P2", "Low"), default="Low",
        help="filter out findings below this severity (default Low: show all).",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"root must be a directory: {root}")

    exclude_dirs = frozenset(_DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir or ()))

    findings = run_all(root, checks=args.check, exclude_dirs=exclude_dirs)
    sev_order = {"P0": 0, "P1": 1, "P2": 2, "Low": 3}
    cutoff = sev_order[args.min_severity]
    findings = [f for f in findings if sev_order.get(f.severity, 99) <= cutoff]

    out = _render_json(findings) if args.format == "json" else _render_markdown(findings)
    sys.stdout.write(out)
    # exit code: non-zero only when P0/P1 found, so CI can gate on it.
    return 1 if any(f.severity in {"P0", "P1"} for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
