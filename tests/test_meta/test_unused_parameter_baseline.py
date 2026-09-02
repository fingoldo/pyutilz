"""Meta-test: a parameter accepted in a signature and never referenced in the body.

Ruff already implements the rule (``flake8-unused-arguments``, ``ARG001``/``ARG002``), but it is
not in the repo's blocking ruff gate and cannot be: ruff has no baseline mechanism, so switching
``ARG`` on in the shared ``configs/ruff-base.toml`` select would turn every pre-existing instance
into a hard failure in every consuming repo at once. This test is that missing baseline layer --
it invokes ruff for this ONE rule family and compares the result against a frozen snapshot, so a
NEWLY-introduced unused parameter fails while the audited existing ones stay quiet.

Why the class is worth a gate of its own: an accepted-and-ignored parameter is routinely a real
defect, not a style nit. The parameter is documented, threaded through call sites, and does
nothing -- callers get silence where they asked for behaviour. Every finding this baseline holds
was read individually and either FIXED in the source or recorded here with the reason it is
intentional (see ``_unused_param_baseline.json``: each entry carries its own justification
string, so the snapshot is a set of reviewed decisions rather than an undifferentiated debt pile).

Keys are ``path::function::param`` -- deliberately NOT ``path:line``. A line-anchored baseline
drifts on any insertion above a finding and forces a wholesale refresh that silently absorbs
genuinely new findings; the symbol-anchored form survives reformatting and unrelated edits.

Refresh with ``--refresh-unused-param-baseline`` after a deliberate change, and write a real
justification for every entry the refresh adds.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess  # nosec B404 - runs the repo's own pinned linter on the repo's own source, no external input
import sys
from pathlib import Path
from typing import Optional

import pytest

import pyutilz

SRC_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_unused_param_baseline.json"


def _enclosing_functions(path: Path) -> dict[tuple[int, int], str]:
    """Map each parameter's (line, 1-based column) to its enclosing dotted function name.

    Ruff reports the location of the offending ``arg`` node but not which function owns it, and a
    line number alone is not a stable key (see the module docstring). Re-parsing the file to name
    the owner is what makes the baseline survive reformatting.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}

    out: dict[tuple[int, int], str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                a = child.args
                for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg):
                    if arg is not None:
                        out[(arg.lineno, arg.col_offset + 1)] = qualified
                walk(child, qualified)
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}.{child.name}" if prefix else child.name)
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def collect_unused_parameters(root: Path) -> dict[str, str]:
    """Return ``{"path::function::param": "ARG00N <message>"}`` for every unused parameter under ``root``.

    ``--isolated`` (rather than the repo's resolved ruff config) is deliberate and safe HERE, unlike in a
    blocking whole-repo gate: this invocation selects exactly one rule family, so there is no resolved
    ignore-list for ``--select`` to drop and no RUF100 determination for it to skew. It also makes the
    snapshot independent of unrelated changes to the shared config.
    """
    proc = subprocess.run(  # nosec B603 - fixed argv, no shell, arguments are repo paths
        [sys.executable, "-m", "ruff", "check", str(root), "--select", "ARG", "--isolated", "--output-format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"ruff failed (exit {proc.returncode}): {proc.stderr.strip()[:2000]}")

    findings: dict[str, str] = {}
    by_file: dict[str, dict[tuple[int, int], str]] = {}
    for item in json.loads(proc.stdout or "[]"):
        filename = Path(item["filename"]).resolve()
        rel = filename.relative_to(root).as_posix()
        owners = by_file.setdefault(str(filename), _enclosing_functions(filename))
        loc = (item["location"]["row"], item["location"]["column"])
        func = owners.get(loc, "<unknown>")
        # The parameter name is the only backticked token in ruff's message ("Unused method argument: `x`").
        message = item["message"]
        param = message.rsplit("`", 2)[-2] if "`" in message else message
        findings[f"{rel}::{func}::{param}"] = f"{item['code']} {message}"
    return findings


def _load_baseline() -> dict[str, str]:
    if not _BASELINE_PATH.exists():
        return {}
    with _BASELINE_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return dict(data.get("entries", data))


def _write_baseline(findings: dict[str, str], previous: dict[str, str]) -> None:
    """Rewrite the snapshot, carrying every still-present justification forward verbatim."""
    entries = {key: previous.get(key, "TODO: justify this entry or fix the parameter") for key in sorted(findings)}
    payload = {
        "_comment": (
            "Frozen ruff ARG001/ARG002 snapshot; see test_unused_parameter_baseline.py. "
            "Key: path::function::param. Value: why this parameter is intentionally unused."
        ),
        "entries": entries,
    }
    with _BASELINE_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
        fh.write("\n")


@pytest.fixture(scope="module")
def unused_parameters() -> dict[str, str]:
    """Today's findings, or a clean skip when ruff is absent.

    The availability check is a separate import probe rather than a try/except around the scan
    itself: wrapping the scan would turn a genuine regression in it into a silent skip.
    """
    if importlib.util.find_spec("ruff") is None:  # pragma: no cover - ruff absent on a minimal CI leg
        pytest.skip("ruff is not installed")
    return collect_unused_parameters(SRC_DIR)


def test_no_new_unused_parameters(unused_parameters: dict[str, str], request: pytest.FixtureRequest) -> None:
    """A parameter accepted and never used must be fixed, or added to the baseline WITH a justification."""
    baseline = _load_baseline()
    if request.config.getoption("--refresh-unused-param-baseline"):
        _write_baseline(unused_parameters, baseline)
        pytest.skip(f"baseline refreshed with {len(unused_parameters)} entries")

    new = sorted(set(unused_parameters) - set(baseline))
    assert not new, (
        "New unused parameter(s) -- a parameter accepted and never referenced is usually a real defect "
        "(the caller's value is silently discarded). Fix it, or add it to "
        f"{_BASELINE_PATH.name} with a justification (interface conformance, callback signature, "
        "**kwargs passthrough, cache-key-only argument):\n  " + "\n  ".join(f"{k}  [{unused_parameters[k]}]" for k in new)
    )


def test_baseline_has_no_stale_entries(unused_parameters: dict[str, str]) -> None:
    """A baselined entry whose parameter is now used (or gone) must be removed, or the snapshot silently
    licenses a NEW finding that happens to reuse the same path::function::param key."""
    stale = sorted(set(_load_baseline()) - set(unused_parameters))
    assert not stale, "Baseline entries no longer reported by ruff -- refresh with --refresh-unused-param-baseline:\n  " + "\n  ".join(stale)


def test_every_baseline_entry_is_justified() -> None:
    """The snapshot is a set of reviewed decisions, not a debt pile: an entry with no real reason is a bug
    waiting to be re-found by the next audit, so an empty or placeholder justification fails."""
    unjustified: list[str] = []
    for key, reason in _load_baseline().items():
        text: Optional[str] = reason if isinstance(reason, str) else None
        if not text or len(text.strip()) < 20 or text.strip().upper().startswith("TODO"):
            unjustified.append(key)
    assert not unjustified, "Baseline entries lacking a written justification:\n  " + "\n  ".join(unjustified)


def test_collector_catches_reconstructed_unused_parameters(tmp_path: Path) -> None:
    """Proof against the findings this check exists to prevent, reconstructed rather than re-broken.

    ``min_failed_idle_interval_minutes`` (accepted, documented, threaded through three call sites,
    never used), the Claude Code CLI path silently dropping ``temperature``/``max_tokens``, an
    ``auto_commit`` pass-through wrapper that never forwards it, and ``labels`` accepted but ignored
    by ``wait_for_absense_of_tasks`` - all four in one tmp tree, plus the fixed forms that must be
    silent.
    """
    if importlib.util.find_spec("ruff") is None:  # pragma: no cover
        pytest.skip("ruff is not installed")

    defective = {
        "web.py": (
            "def get_new_smartproxy(proxy_user, failed_dict=None, min_failed_idle_interval_minutes=1440):\n"
            '    """min_failed_idle_interval_minutes: how long a failed proxy stays out of rotation."""\n'
            "    return (proxy_user, failed_dict)\n"
        ),
        "claude_code_provider.py": (
            "class ClaudeCodeProvider:\n" "    async def _generate_cli(self, prompt, temperature=0.7, max_tokens=1024):\n" "        return prompt\n"
        ),
        "db.py": (
            "def basic_db_execute(sql, auto_commit=True):\n"
            "    return (sql, auto_commit)\n\n\n"
            "def db_execute(sql, auto_commit=True):\n"
            "    return basic_db_execute(sql)\n"
        ),
        "prefect.py": "def wait_for_absense_of_tasks(flow_name, labels=None):\n    return bool(flow_name)\n",
    }
    for name, body in defective.items():
        (tmp_path / name).write_text(body, encoding="utf-8")

    found = set(collect_unused_parameters(tmp_path))
    assert found == {
        "web.py::get_new_smartproxy::min_failed_idle_interval_minutes",
        "claude_code_provider.py::ClaudeCodeProvider._generate_cli::temperature",
        "claude_code_provider.py::ClaudeCodeProvider._generate_cli::max_tokens",
        "db.py::db_execute::auto_commit",
        "prefect.py::wait_for_absense_of_tasks::labels",
    }

    fixed = {
        "web.py": (
            "def get_new_smartproxy(proxy_user, failed_dict=None, min_failed_idle_interval_minutes=1440):\n"
            "    if min_failed_idle_interval_minutes and failed_dict:\n"
            "        failed_dict.clear()\n"
            "    return proxy_user\n"
        ),
        "claude_code_provider.py": "class ClaudeCodeProvider:\n    async def _generate_cli(self, prompt):\n        return prompt\n",
        "db.py": (
            "def basic_db_execute(sql, auto_commit=True):\n"
            "    return (sql, auto_commit)\n\n\n"
            "def db_execute(sql, auto_commit=True):\n"
            "    return basic_db_execute(sql, auto_commit=auto_commit)\n"
        ),
        "prefect.py": "def wait_for_absense_of_tasks(flow_name, labels=None):\n    return bool(flow_name) and (labels is None or bool(labels))\n",
    }
    for name, body in fixed.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    assert collect_unused_parameters(tmp_path) == {}
