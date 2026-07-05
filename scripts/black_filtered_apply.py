"""Apply Black's reformatting to a file, EXCEPT for opcodes matching excluded
classes: blank-line insertion (pure or amid other lines), and arg/import-list
EXPLOSION (packed multi-item line -> one-item-per-line). Everything else
Black wants to change (collapses, quote style, blank-line removal, docstring
reflow, redundant-paren removal, whitespace reflow) is applied normally.

Usage:
    python black_filtered_apply.py --config pyproject.toml [--write] <file.py> [file2.py ...]
    python black_filtered_apply.py --config pyproject.toml --check <root_dir> [root_dir2 ...]

--config is required (never rely on directory walk-up -- a file processed
from a scratch/CI checkout dir must not silently fall back to Black's
88-col default).

Without --write and without --check: prints a unified diff of what WOULD
change for each listed FILE and exits 0.
With --write: rewrites each listed FILE in place.
With --check <dirs>: recursively discovers *.py under the given directories
(skipping .git/.venv/.pytest_cache/build/dist/__pycache__/legacy), reports
which files still have non-excluded-class Black findings, and exits 1 if any
do (0 if the tree is fully filtered-Black-clean) -- the CI-friendly mode.
"""
import sys
import os
import shlex
import re
import subprocess
import difflib
import pathlib

EXCLUDED_DIR_NAMES = {".git", ".venv", ".pytest_cache", "build", "dist", "__pycache__", "legacy"}

# Override how `black` is invoked via the BLACK_CMD env var (space-separated),
# e.g. BLACK_CMD="uvx black" for projects that run Black through uv's tool
# cache rather than a pip-installed environment. Defaults to `python -m black`.
_BLACK_CMD = shlex.split(os.environ["BLACK_CMD"]) if os.environ.get("BLACK_CMD") else [sys.executable, "-m", "black"]


def discover_py_files(roots):
    files = []
    for root in roots:
        for p in pathlib.Path(root).rglob("*.py"):
            if any(part in EXCLUDED_DIR_NAMES for part in p.parts):
                continue
            files.append(str(p))
    return sorted(files)


def run_black_stdin(src: str, config_path: str) -> str:
    proc = subprocess.run(
        [*_BLACK_CMD, "-q", "--config", config_path, "-"],
        input=src.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode not in (0,):
        raise RuntimeError(f"black failed: {proc.stderr.decode('utf-8', 'replace')}")
    return proc.stdout.decode("utf-8")


def _swap_single_to_double_quotes(s: str) -> str:
    """Best-effort normalize simple 'x' string literals to "x" so quote-style
    changes (which Black applies independently of explosion/collapse) don't
    mask an explosion/collapse comparison. Skips triple-quoted strings and
    literals that already contain a double quote (Black wouldn't requote
    those either, so leaving them alone is correct)."""
    def repl(m):
        inner = m.group(1)
        if '"' in inner:
            return m.group(0)
        return '"' + inner + '"'

    return re.sub(r"'((?:[^'\\]|\\.)*)'", repl, s)


def norm(s: str) -> str:
    return re.sub(r"[\s,()]+", "", _swap_single_to_double_quotes(s))


def is_all_blank(lines):
    return all(line.strip() == "" for line in lines) and len(lines) > 0


def looks_like_import_or_call_list(old_block, new_block):
    """True if old_block and new_block are the SAME tokens (ignoring whitespace/
    commas/parens) but spread over a DIFFERENT number of lines -- the arg/import
    explosion (or collapse) pattern. Returns 'explode' / 'collapse' / None.
    """
    old_txt = norm("".join(old_block))
    new_txt = norm("".join(new_block))
    if old_txt != new_txt or old_txt == "":
        return None
    if len(new_block) > len(old_block):
        return "explode"
    if len(new_block) < len(old_block):
        return "collapse"
    return None


def filtered_apply(orig: str, formatted: str) -> str:
    orig_lines = orig.splitlines(keepends=True)
    fmt_lines = formatted.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, orig_lines, fmt_lines, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old_block = orig_lines[i1:i2]
        new_block = fmt_lines[j1:j2]
        if tag == "equal":
            out.extend(new_block)
            continue
        if tag == "insert":
            # pure blank-line insertion -> reject (keep nothing, i.e. don't insert)
            if is_all_blank(new_block):
                continue
            out.extend(new_block)
            continue
        if tag == "delete":
            # pure blank-line removal -> accept (i.e. actually remove: emit nothing)
            if is_all_blank(old_block):
                continue
            # non-blank deletion with nothing replacing it: accept Black's change
            continue
        if tag == "replace":
            kind = looks_like_import_or_call_list(old_block, new_block)
            if kind == "explode":
                out.extend(old_block)  # reject explosion, keep original packed form
                continue
            # mixed replace containing a blank-line insertion bundled with other content:
            # re-diff the sub-blocks at a finer grain so we only reject the blank part.
            sub_sm = difflib.SequenceMatcher(None, old_block, new_block, autojunk=False)
            sub_out = []
            for stag, si1, si2, sj1, sj2 in sub_sm.get_opcodes():
                s_old = old_block[si1:si2]
                s_new = new_block[sj1:sj2]
                if stag == "equal":
                    sub_out.extend(s_new)
                elif stag == "insert" and is_all_blank(s_new):
                    continue
                elif stag == "delete" and is_all_blank(s_old):
                    continue
                else:
                    sub_kind = looks_like_import_or_call_list(s_old, s_new)
                    if sub_kind == "explode":
                        sub_out.extend(s_old)
                    else:
                        sub_out.extend(s_new)
            out.extend(sub_out)
            continue
    return "".join(out)


def process_one(path, config_path):
    """Returns (changed: bool, orig: str, result: str)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        orig = f.read()
    formatted = run_black_stdin(orig, config_path)
    result = filtered_apply(orig, formatted)
    return result != orig, orig, result


def main():
    args = sys.argv[1:]
    write = "--write" in args
    check = "--check" in args
    args = [a for a in args if a not in ("--write", "--check")]
    config_path = None
    if "--config" in args:
        idx = args.index("--config")
        config_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]
    if not config_path:
        raise SystemExit("--config <path-to-pyproject.toml> is required")

    if check:
        files = discover_py_files(args)
        changed_files = []
        for path in files:
            try:
                changed, _, _ = process_one(path, config_path)
            except RuntimeError as e:
                print(f"ERROR: {path}: {e}", file=sys.stderr)
                changed = True
            if changed:
                changed_files.append(path)
        if changed_files:
            print(f"{len(changed_files)}/{len(files)} files have non-excluded-class Black findings:")
            for p in changed_files:
                print(f"  {p}")
            raise SystemExit(1)
        print(f"All {len(files)} files are filtered-Black-clean.")
        return

    files = args
    for path in files:
        changed, orig, result = process_one(path, config_path)
        if not changed:
            print(f"UNCHANGED: {path}")
            continue
        if write:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(result)
            print(f"WROTE: {path}")
        else:
            diff = difflib.unified_diff(
                orig.splitlines(keepends=True),
                result.splitlines(keepends=True),
                fromfile=path,
                tofile=path,
            )
            sys.stdout.writelines(diff)


if __name__ == "__main__":
    main()
