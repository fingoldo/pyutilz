"""Scanner tests for registry_and_cli, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    Finding,
    run_all,
    scan_dead_public_callables,
)

from ._helpers import _temporal_rule, _write

# ---- run_all + ordering -------------------------------------------------


def test_run_all_returns_sorted_by_severity(tmp_path: Path):
    _write(tmp_path, "mixed.py", """
def bad_mutable(items=[]):
    items.append(1)

def bad_or(n=None):
    return n or 4
""")
    findings = run_all(tmp_path)
    # P0 (mutable_default mutated) should come before P1 (default_via_or).
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: {"P0": 0, "P1": 1, "P2": 2, "Low": 3}[s])
    assert "P0" in severities
    assert "P1" in severities


def test_run_all_empty_on_clean_tree(tmp_path: Path):
    _write(tmp_path, "clean.py", """
def f(x=None):
    if x is None:
        x = []
    return x
""")
    findings = run_all(tmp_path)
    assert findings == []


def test_run_all_parallel_matches_sequential(tmp_path: Path):
    """run_all(parallel=True) (the default) must return the EXACT same findings as
    parallel=False -- scanners are independent (each only appends to its own list), so
    distributing them across a ProcessPoolExecutor must be a pure wall-clock optimization,
    never a behavior change."""
    _write(tmp_path, "mixed.py", """
def bad_mutable(items=[]):
    items.append(1)

def bad_or(n=None):
    return n or 4

try:
    risky()
except Exception:
    pass
""")
    parallel = run_all(tmp_path, parallel=True)
    sequential = run_all(tmp_path, parallel=False)
    assert parallel == sequential
    assert len(parallel) > 0


class TestWorkerPoolSizing:
    """Each worker pays a fixed spawn + import + full-corpus-re-parse cost that does NOT
    shrink as workers are added, so the pool must be sized off PHYSICAL cores and a minimum
    batch of scanners per worker -- not off ``os.cpu_count()`` (logical) or the raw scanner
    count. Sizing it off the logical count measurably made the scan SLOWER (see
    ``_MIN_SCANNERS_PER_WORKER``'s sweep); these pin the fix so it cannot silently regress."""

    def _workers_for(self, n_scanners: int) -> int:
        from pyutilz.dev.code_audit.registry import _MIN_SCANNERS_PER_WORKER, _physical_cpu_count

        return max(2, min(_physical_cpu_count(), n_scanners // _MIN_SCANNERS_PER_WORKER))

    def test_physical_count_does_not_exceed_logical(self):
        import os

        from pyutilz.dev.code_audit.registry import _physical_cpu_count

        physical = _physical_cpu_count()
        assert physical >= 1
        assert physical <= (os.cpu_count() or 1), "physical cores cannot exceed logical CPUs"

    def test_worker_count_never_reaches_one_per_scanner(self):
        """The pre-fix formula was min(len(selected), os.cpu_count()), which on a big machine
        spawned an interpreter per scanner. Every worker beyond the batch threshold adds a
        whole corpus re-parse for a shrinking slice of scan work."""
        for n in (20, 49, 200):
            assert self._workers_for(n) <= n // 5, f"{n} scanners spawned too many workers"

    def test_worker_count_is_capped_by_physical_cores(self):
        from pyutilz.dev.code_audit.registry import _physical_cpu_count

        assert self._workers_for(10_000) == _physical_cpu_count()

    def test_small_scanner_sets_still_get_at_least_two_workers(self):
        """The floor keeps a small-but-parallel run (>= _MIN_SCANNERS_FOR_PARALLEL scanners)
        from degenerating into a single-worker pool, which would be strictly worse than the
        sequential path it already opted out of."""
        assert self._workers_for(4) == 2


def test_excluded_dir_ignored(tmp_path: Path):
    bad = tmp_path / "build" / "bad.py"
    bad.parent.mkdir()
    bad.write_text("def f(x=[]): x.append(1)\n", encoding="utf-8")
    findings = run_all(tmp_path)
    assert findings == [], "build/ should be excluded by default"


def test_finding_md_row_format():
    f = Finding(
        check="x", severity="P0", file="src/a.py", line=42,
        snippet="def f(x=[])", detail="bad",
    )
    row = f.as_md_row()
    assert row.startswith("| P0 | x | src/a.py:42 |")
    assert "`def f(x=[])`" in row


# ---- CLI surface --------------------------------------------------------


def test_cli_exits_nonzero_on_p1(tmp_path: Path, capsys):
    _write(tmp_path, "bad.py", "def f(items=[]):\n    items.append(1)\n")
    from pyutilz.dev.code_audit import main as cli_main
    rc = cli_main([str(tmp_path), "--format", "markdown"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "mutable_default" in out
    assert "P0" in out


def test_cli_exits_zero_on_clean(tmp_path: Path, capsys):
    _write(tmp_path, "ok.py", "def f(x=None):\n    return x\n")
    from pyutilz.dev.code_audit import main as cli_main
    rc = cli_main([str(tmp_path)])
    assert rc == 0


# ---- __main__ module-execution guard ---------------------------------------


def test_dunder_main_module_execution_delegates_to_cli_main(tmp_path: Path):
    """``python -m pyutilz.dev.code_audit <root>`` runs the ``if __name__ == '__main__'`` guard in
    __main__.py, which only executes under real module execution (never when pytest imports the
    module normally) -- exercised here via a real subprocess."""
    import subprocess
    import sys

    _write(tmp_path, "ok.py", "def f(x=None):\n    return x\n")
    result = subprocess.run(  # nosec B603 -- fixed local argv, no shell, no untrusted input
        [sys.executable, "-m", "pyutilz.dev.code_audit", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


# ---- exclude_dirs must judge the path BELOW the root, never the root's own ancestors --------


def test_an_excluded_name_ABOVE_the_scan_root_does_not_silence_the_scan(tmp_path: Path):
    """A scan root living inside a directory named in ``exclude_dirs`` must still be scanned.

    The check used to match ``exclude_dirs`` against the ABSOLUTE path's components, so every file
    under a checkout at ``.../.claude/worktrees/<agent>/`` matched on the ancestor ``.claude`` and
    the scan returned nothing at all - for a whole package, silently, with every audit built on it
    passing vacuously. That is where Claude Code agent worktrees live, and the same trap sits one
    directory away for any project under ``build/``, ``dist/``, ``env/`` or ``venv/``.
    """
    buried = tmp_path / ".claude" / "worktrees" / "agent-1" / "pkg"
    buried.mkdir(parents=True)
    (buried / "mod.py").write_text("def obviously_dead_helper(x):\n    return x\n", encoding="utf-8")

    findings = scan_dead_public_callables(buried)
    assert [f.detail for f in findings if "obviously_dead_helper" in f.detail], "a root under a `.claude` ancestor was scanned as if empty"


def test_an_excluded_name_BELOW_the_scan_root_is_still_skipped(tmp_path: Path):
    """The negative control: the exclusion itself must keep working for real build/cache dirs."""
    inner = tmp_path / "__pycache__"
    inner.mkdir()
    (inner / "mod.py").write_text("def obviously_dead_helper(x):\n    return x\n", encoding="utf-8")

    assert not [f for f in scan_dead_public_callables(tmp_path) if "obviously_dead_helper" in f.detail]


def test_every_tree_walking_scanner_agrees_between_a_relative_and_absolute_root(tmp_path: Path):
    """A scan must not depend on how its root was spelled.

    `_iter_py_files` was fixed first, and two scanners turned out to carry their OWN copy of the
    exclude check against the absolute path - so the same tree scanned as `Path("tests")` and as
    `Path("tests").resolve()` gave different answers whenever the checkout sat under an excluded
    ancestor. Both are routed through `_is_excluded` now; this pins that they cannot drift apart again.
    """
    import os

    from pyutilz.dev.code_audit import scan_import_cycles, scan_redundant_test_fit_calls

    pkg = tmp_path / ".claude" / "worktrees" / "agent-1" / "proj"
    (pkg / "tests").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tests" / "test_thing.py").write_text(
        "from functools import lru_cache\n\n\n"
        "@lru_cache\ndef fit(a):\n    return a\n\n\n"
        "def test_one():\n    assert fit(1) == 1\n\n\n"
        "def test_two():\n    assert fit(1) == 1\n",
        encoding="utf-8",
    )

    cwd = os.getcwd()
    try:
        os.chdir(pkg)
        for scan, args in ((scan_redundant_test_fit_calls, (Path("tests"),)), (scan_import_cycles, (Path("."), "proj"))):
            relative = len(list(scan(*args)))
            absolute = len(list(scan(args[0].resolve(), *args[1:])))
            assert relative == absolute, f"{scan.__name__} saw {relative} findings via a relative root and {absolute} via an absolute one"
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# check_all / normalise_text / get_scanners -- all three are in dev.code_audit's __all__
# and were previously never mentioned anywhere under tests/ (audit F20, 2026-09-02).
# get_scanners and the registry it copies are part of the shared meta-test harness pyutilz
# exports to its downstream consumers, so a regression here breaks six other repos.
# ---------------------------------------------------------------------------


def test_normalise_text_deletes_intra_word_hyphens_so_both_spellings_are_one_token():
    from pyutilz.dev.code_audit.field_text_agreement import normalise_text

    assert normalise_text("Ante-Mortem") == normalise_text("antemortem") == "antemortem"
    assert normalise_text("POST-mortem") == "postmortem"


def test_normalise_text_leaves_edge_and_digit_adjacent_hyphens_alone():
    """Only a hyphen BETWEEN two letters joins a compound; a leading/trailing one or a
    letter-digit boundary is meaningful punctuation, not a spelling variant."""
    from pyutilz.dev.code_audit.field_text_agreement import normalise_text

    assert normalise_text("co-2") == "co-2"
    assert normalise_text("-lead") == "-lead"
    assert normalise_text("trail-") == "trail-"


def test_normalise_text_collapses_whitespace_and_underscores():
    from pyutilz.dev.code_audit.field_text_agreement import normalise_text

    assert normalise_text("  Multiple   spaces\tand\nnewlines  ") == "multiple spaces and newlines"
    assert normalise_text("snake_case_name") == "snake case name"


def test_normalise_text_handles_none_and_non_strings():
    from pyutilz.dev.code_audit.field_text_agreement import normalise_text

    assert normalise_text(None) == ""
    assert normalise_text("") == ""
    assert normalise_text(42) == "42"


def test_check_all_returns_one_report_per_rule_keyed_by_name():
    from pyutilz.dev.code_audit import FieldTextRule, check_all

    rows = [
        {"subject": "haemorrhage", "object": "vital hanging", "temporal_class": "antemortem"},
        {"subject": "", "object": "putrefaction", "temporal_class": "postmortem"},
    ]
    manner = FieldTextRule(name="manner", field="manner", text_fields=("object",))
    reports = check_all([_temporal_rule(), manner], rows)

    assert set(reports) == {"temporal_class", "manner"}
    assert reports["temporal_class"].agree == 2
    assert reports["temporal_class"].contradict == 0
    assert reports["temporal_class"].n_records == 2


def test_check_all_includes_vocabulary_less_rules_instead_of_dropping_them():
    """A rule with no cues must appear as 100% uncheckable, not vanish from the output -- an
    unmeasured pair silently missing from a report reads as a pair that passed."""
    from pyutilz.dev.code_audit import FieldTextRule, check_all

    rows = [{"object": "anything", "manner": "blunt"}]
    reports = check_all([FieldTextRule(name="manner", field="manner", text_fields=("object",))], rows)

    assert "manner" in reports
    assert reports["manner"].has_vocabulary is False
    assert reports["manner"].uncheckable == 1
    assert reports["manner"].coverage == 0.0


def test_check_all_accepts_explicit_record_ids():
    from pyutilz.dev.code_audit import check_all

    rows = [{"subject": "", "object": "vital hanging", "temporal_class": "postmortem"}]
    reports = check_all([_temporal_rule()], rows, ids=["record-7"])

    assert reports["temporal_class"].contradict == 1
    assert reports["temporal_class"].contradictions[0].record_id == "record-7"


def test_get_scanners_returns_a_populated_registry_of_callables():
    from pyutilz.dev.code_audit import get_scanners

    scanners = get_scanners()
    assert isinstance(scanners, dict)
    assert scanners, "the scanner registry must not be empty"
    assert all(callable(fn) for fn in scanners.values())
    # A few names every consumer's baseline depends on.
    for expected in ("vacuous_assertion", "except_skip_masks_call_under_test", "bare_except"):
        assert expected in scanners, f"{expected!r} missing from the registry: {sorted(scanners)}"


def test_get_scanners_returns_a_copy_so_callers_cannot_corrupt_the_shared_registry():
    """The documented reason this accessor exists at all: ``get_scanners().pop(...)`` must not
    silently disarm a scanner for every subsequent run_all() in the same process."""
    from pyutilz.dev.code_audit import get_scanners

    first = get_scanners()
    victim = next(iter(first))
    first.pop(victim)
    first["definitely_not_a_real_scanner"] = lambda *a, **kw: []

    second = get_scanners()
    assert victim in second
    assert "definitely_not_a_real_scanner" not in second
