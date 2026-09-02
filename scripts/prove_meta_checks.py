"""Proof harness: for each new check, build the DEFECTIVE input reproducing a cited
2026-09-02 finding and show the check fires on it, while passing on the real tree.

Never mutates the real repo: every defective artefact is written into a temp copy.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(r"D:\Upd\Programming\PythonCodeRepository\pyutilz")
results: list[tuple[str, str, bool, str]] = []


def record(check: str, finding: str, fired: bool, detail: str) -> None:
    results.append((check, finding, fired, detail))


def copy_repo_shell(tmp: Path) -> Path:
    """A minimal but REAL copy of the repo's config/prose surface plus src/."""
    root = tmp / "repo"
    root.mkdir()
    for name in ("pyproject.toml", "README.md", "TESTING.md", "CONTRIBUTING.md", "CHANGELOG.md", ".pre-commit-config.yaml", "requirements-dev.txt"):
        shutil.copy2(REPO / name, root / name)
    shutil.copytree(REPO / ".github" / "workflows", root / ".github" / "workflows")
    shutil.copytree(REPO / "docs", root / "docs")
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "_benchmarks", root / "_benchmarks", ignore=shutil.ignore_patterns("__pycache__"))
    # Empty stand-ins for the real test files, so a documented `tests/<name>.py` still resolves
    # without copying the whole suite.
    (root / "tests").mkdir()
    for test_file in (REPO / "tests").rglob("*.py"):
        stub = root / test_file.relative_to(REPO)
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("", encoding="utf-8")
    return root


def main() -> int:
    from py_ci_shared import ci_test_dir_reachability, ci_workflow_gate, ci_workflow_timeout_gate, docs_inventory_parity, gate_integrity, git_dependency_pins, loc_budget, phantom_markdown_links, prose_numeric_claims
    from py_ci_shared.mypy_gate import check_mypy_output

    with tempfile.TemporaryDirectory(dir=r"D:\Temp") as raw_tmp:
        tmp = Path(raw_tmp)
        root = copy_repo_shell(tmp)
        workflows = root / ".github" / "workflows"

        # ---- MT-1 / loc_budget : 01/F06 "Five modules exceed the 1000-LOC split threshold"
        oversized = root / "src" / "pyutilz" / "text" / "similarity_monolith.py"
        oversized.write_text("# padding\n" * 1200, encoding="utf-8")
        over = {p.name: loc_budget._loc(p) for p in (root / "src").rglob("*.py") if loc_budget._loc(p) > 1000}
        record("MT-1 loc_budget", "01/F06 module over the 1000-LOC split threshold", bool(over), str(over))
        oversized.unlink()

        # ---- MT-1 / ci_workflow_timeout_gate : 03/F14 "publish.yml jobs have no timeout-minutes"
        publish = workflows / "publish.yml"
        publish.write_text("\n".join(line for line in publish.read_text(encoding="utf-8").splitlines() if "timeout-minutes" not in line), encoding="utf-8")
        missing = ci_workflow_timeout_gate.find_jobs_missing_timeout(publish)
        record("MT-1 ci_workflow_timeout_gate", "03/F14 publish.yml jobs have no timeout-minutes", bool(missing), f"jobs missing a timeout: {missing}")

        # ---- MT-1 / ci_workflow_gate : a blocking gate step turned advisory
        ci = workflows / "ci.yml"
        ci.write_text(ci.read_text(encoding="utf-8").replace("    - name: Run tests with coverage\n", "    - name: Run tests with coverage\n      continue-on-error: true\n"), encoding="utf-8")
        advisory = ci_workflow_gate.find_continue_on_error_steps(ci)
        record("MT-1 ci_workflow_gate", "gate-defeating continue-on-error on a blocking step", bool(advisory), f"advisory steps: {advisory}")

        # ---- MT-1 / git_dependency_pins : 03/F01 "[dev] extra carries a direct VCS URL dependency"
        pyproject = root / "pyproject.toml"
        original_pyproject = pyproject.read_text(encoding="utf-8")
        pyproject.write_text(original_pyproject.replace('dev = [\n', 'dev = [\n    "py-ci-shared @ git+https://github.com/fingoldo/py-ci-shared.git",\n', 1), encoding="utf-8")
        unpinned = git_dependency_pins.find_unpinned_git_dependencies(pyproject)
        record("MT-1 git_dependency_pins", "03/F01 [dev] extra carries a direct VCS URL", bool(unpinned), f"unpinned git refs in project metadata: {unpinned}")
        pyproject.write_text(original_pyproject, encoding="utf-8")

        # ---- MT-1 / phantom_markdown_links : a dead markdown link target
        readme = root / "README.md"
        original_readme = readme.read_text(encoding="utf-8")
        readme.write_text(original_readme + "\nSee [the architecture doc](docs/architecture.md).\n", encoding="utf-8")
        phantom = phantom_markdown_links.find_phantom_markdown_links([readme], root)
        record("MT-1 phantom_markdown_links", "dead markdown-link target", bool(phantom), str(phantom))
        readme.write_text(original_readme, encoding="utf-8")

        # ---- MT-1 / ci_test_dir_reachability : 04/F01 shape, a test dir no CI job collects
        (root / "tests" / "orphan_tier").mkdir()
        # Every workflow's pathless `pytest` (which collects the whole tree) is given an explicit
        # path, so the fixture actually models "no job collects this subdir".
        for wf in workflows.glob("*.yml"):
            wf.write_text(wf.read_text(encoding="utf-8").replace('pytest -m "not gpu"', 'pytest -m "not gpu" tests/test_meta').replace("pytest --no-cov -q", "pytest --no-cov -q tests/test_meta"), encoding="utf-8")
        unreachable = ci_test_dir_reachability.find_unreachable_test_subdirs(root, workflows)
        record("MT-1 ci_test_dir_reachability", "04/F01 shape: a tests/ subdir no CI job collects", "orphan_tier" in unreachable, f"unreachable: {unreachable}")

        # ---- MT-2 rule 2 : 06/F01 mypy INTERNAL ERROR / narrowed scope
        internal_error = "src/pyutilz/x.py: error: INTERNAL ERROR --\nversion: 1.8.0\nTraceback (most recent call last):\n"
        record("MT-2 completion", "06/F01 mypy INTERNAL ERROR passes an exit-code-only gate", check_mypy_output(internal_error, 0) is not None, str(check_mypy_output(internal_error, 0)))
        record("MT-2 completion", "06/F01 sibling: a silently narrowed mypy scope", check_mypy_output("Success: no issues found in 3 source files\n", 0, min_files=200) is not None, str(check_mypy_output("Success: no issues found in 3 source files\n", 0, min_files=200)))
        record("MT-2 completion", "a completed clean run is accepted", check_mypy_output("Success: no issues found in 216 source files\n", 0, min_files=200) is None, "no problem reported")

        # ---- MT-2 rule 2 (static) : the hook invoking mypy directly again
        precommit = root / ".pre-commit-config.yaml"
        original_precommit = precommit.read_text(encoding="utf-8")
        precommit.write_text(original_precommit.replace("python -m py_ci_shared.mypy_gate --min-files 200 src/pyutilz", "python -m mypy src/pyutilz"), encoding="utf-8")
        direct = gate_integrity.find_gates_without_completion_assertion(precommit, {"python -m mypy": "py_ci_shared.mypy_gate"})
        record("MT-2 completion (static)", "a blocking hook reverting to a bare `python -m mypy`", bool(direct), direct[0][:110] if direct else "")
        precommit.write_text(original_precommit, encoding="utf-8")

        # ---- MT-2 rule 3 : 06/F17 --ignore C901, and 06/F08 tests/ excluded from ruff
        sys.path.insert(0, str(REPO / "tests" / "test_meta"))
        import test_gate_integrity as pyutilz_gate_test  # the real declaration map

        declared = dict(pyutilz_gate_test._DECLARED_NARROWINGS)
        undeclared, _ = gate_integrity.find_undeclared_narrowings(precommit, workflows, {k: v for k, v in declared.items() if "C901" not in k}, pyproject, pyutilz_gate_test._GATE_CONFIG_TABLES)
        record("MT-2 rule 3", "06/F17 `--ignore C901` present with no declared reason", any("C901" in u for u in undeclared), f"{len(undeclared)} undeclared: {[u.split('   <-')[0] for u in undeclared]}")

        pyproject.write_text(original_pyproject.replace('exclude = ["_benchmarks", ".git"', 'exclude = ["tests", "_benchmarks", ".git"', 1), encoding="utf-8")
        present = gate_integrity.find_narrowings(precommit, workflows, pyproject, pyutilz_gate_test._GATE_CONFIG_TABLES)
        excluded_value = present.get("pyproject::[tool.ruff]::exclude", "")
        record("MT-2 rule 3", "06/F08 tests/ excluded from ruff (value change re-opens the declaration)", "tests" in excluded_value and excluded_value != f"exclude = {['_benchmarks', '.git', '__pycache__', 'build', 'dist']!r}", excluded_value)
        pyproject.write_text(original_pyproject, encoding="utf-8")

        # ---- MT-2 rule 4 : 04/F08 CI coverage gate below the project floor
        ci.write_text(ci.read_text(encoding="utf-8").replace("--cov-fail-under=82", "--cov-fail-under=62"), encoding="utf-8")
        mismatches = gate_integrity.find_coverage_gate_mismatches(pyproject, workflows)
        record("MT-2 rule 4", "04/F08 CI coverage gate below the declared floor", bool(mismatches), str(mismatches))

        # ---- MT-5 : 05/F08 / the recurring stale-count shape
        testing = root / "TESTING.md"
        original_testing = testing.read_text(encoding="utf-8")
        testing.write_text(original_testing.replace("The 27-entry backward-compat", "The 24-entry backward-compat"), encoding="utf-8")
        claim = prose_numeric_claims.NumericClaim(testing, r"The (\d+)-entry backward-compat module alias map", lambda: 27, "alias count")
        stale = prose_numeric_claims.find_stale_claims([claim])
        record("MT-5", "cross-wave: TESTING.md claims 24 aliases, the map holds 27", bool(stale), stale[0] if stale else "")

        testing.write_text(original_testing.replace("The 27-entry backward-compat module alias map", "The backward-compat module alias map"), encoding="utf-8")
        unanchored = prose_numeric_claims.find_stale_claims([claim])
        record("MT-5", "the prose is reworded so the claim silently stops being checked", bool(unanchored) and "no longer matches" in unanchored[0], unanchored[0][:120] if unanchored else "")
        testing.write_text(original_testing, encoding="utf-8")

        readme.write_text(original_readme.replace("across 7 providers", "across 5 providers"), encoding="utf-8")
        provider_claim = prose_numeric_claims.NumericClaim(readme, r"Unified LLM interface across (\d+) providers", lambda: 7, "provider count")
        record("MT-5", "05/F08 shape: a hand-typed README count drifts from the registry", bool(prose_numeric_claims.find_stale_claims([provider_claim])), str(prose_numeric_claims.find_stale_claims([provider_claim])))

        record(
            "MT-5 (warn)",
            "05/F11 an undated measured runtime claim",
            bool(prose_numeric_claims.find_undated_volatile_claims([_write(tmp / "T.md", "The meta-test suite runs in ~30 s.\n")])),
            "undated volatile figure reported",
        )
        readme.write_text(original_readme, encoding="utf-8")

        # ---- MT-6 rule (a) : 05/F12 / 05/F13 / 05/F14 / 05/F15
        bullet = pyutilz_docs_bullet()
        readme.write_text(original_readme.replace("pip install pyutilz[system]           # scipy + Pillow + py-cpuinfo + GPUtil + xmltodict + jellyfish", "pip install pyutilz[system]           # scipy + Pillow + tqdm"), encoding="utf-8")
        drift = docs_inventory_parity.find_extras_documentation_drift(pyproject, readme, bullet, ignore_packages=["python-dateutil"], undocumented_groups=["pandas", "polars", "all"])
        record("MT-6 rule (a)", "05/F12 [system] bullet omits members and names non-members", bool(drift), "; ".join(drift))

        readme.write_text("\n".join(line for line in original_readme.splitlines() if "pyutilz[llm]" not in line and "pyutilz[docs]" not in line), encoding="utf-8")
        drift = docs_inventory_parity.find_extras_documentation_drift(pyproject, readme, bullet, ignore_packages=["python-dateutil"], undocumented_groups=["pandas", "polars", "all"])
        record("MT-6 rule (a)", "05/F15 install block omits declared extras groups", len(drift) == 2, "; ".join(drift))

        readme.write_text(original_readme.replace("`[all]` = `pandas,polars,database,web,cloud,nlp,llm,system,stats`", "`[all]` = `pandas,polars,database,web,cloud,nlp,llm,system,stats,dash,prefect,tensorflow`"), encoding="utf-8")
        aggregate = docs_inventory_parity.find_aggregate_group_drift(pyproject, readme, r"`\[(all)\]` = `([\w,]+)`")
        record("MT-6 rule (a)", "03/F05 / 05/F05 the aggregate [all] composition sentence drifts", bool(aggregate), "; ".join(aggregate))
        readme.write_text(original_readme, encoding="utf-8")

        # ---- MT-6 rule (c) : 05/F04 undeclared marker, 05/F10 moved benchmark path
        contributing = root / "CONTRIBUTING.md"
        original_contributing = contributing.read_text(encoding="utf-8")
        contributing.write_text(original_contributing + "\nTag integration tests with @pytest.mark.integration.\nBenchmarks live in `tests/benchmark_similarity.py`.\n", encoding="utf-8")
        markers = docs_inventory_parity.find_undeclared_markers([contributing], pyproject)
        paths = docs_inventory_parity.find_phantom_doc_paths([contributing], root, search_roots=[root / "src", root / "src" / "pyutilz"])
        record("MT-6 rule (c)", "05/F04 documented @pytest.mark.integration is undeclared", bool(markers), "; ".join(markers))
        record("MT-6 rule (c)", "05/F10 documented benchmark path does not exist", bool(paths), "; ".join(paths))
        contributing.write_text(original_contributing, encoding="utf-8")

        # ---- MT-6 rule (b) : 05/F16 twelve shipped modules documented nowhere
        (root / "src" / "pyutilz" / "brandnew").mkdir()
        (root / "src" / "pyutilz" / "brandnew" / "__init__.py").write_text("", encoding="utf-8")
        (root / "src" / "pyutilz" / "brandnew" / "engine.py").write_text("", encoding="utf-8")
        undocumented = docs_inventory_parity.find_undocumented_modules(root / "src" / "pyutilz", [readme, root / "docs" / "modules.md", root / "docs" / "index.md", root / "CHANGELOG.md"])
        record("MT-6 rule (b, warn)", "05/F16 a shipped subpackage appears in no documentation", "pyutilz.brandnew" in undocumented, str(undocumented))

    width = max(len(check) for check, _, _, _ in results)
    failures = 0
    for check, finding, fired, detail in results:
        status = "CAUGHT" if fired else "MISSED"
        failures += 0 if fired else 1
        print(f"[{status}] {check.ljust(width)}  {finding}")
        if detail:
            print(f"           -> {detail[:200]}")
    print(f"\n{len(results) - failures}/{len(results)} defective inputs detected")
    return 1 if failures else 0


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def pyutilz_docs_bullet() -> str:
    return r"pip install pyutilz\[(\w+)\]\s*#\s*(.*)"


if __name__ == "__main__":
    raise SystemExit(main())
