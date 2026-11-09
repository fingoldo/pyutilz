"""Mutation sweep over one whole module: mutation_sweep_module.py <module rel path> <tests,comma> <checkpoint.json>.

Paths are relative to the repository root. Test files that do not exist are skipped.
"""

import sys
from pathlib import Path

from py_ci_shared.mutation_teeth import find_surviving_mutants

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    mod = Path(sys.argv[1])
    tests = [Path(t) for t in sys.argv[2].split(",") if t]
    checkpoint = Path(sys.argv[3])
    n = sum(1 for _ in (REPO / mod).read_text(encoding="utf-8").splitlines())
    run = find_surviving_mutants(
        mod,
        [t for t in tests if (REPO / t).exists()],
        str(REPO),
        lines=[range(1, n + 1)],
        timeout=1800,
        use_cache=False,
        limit=600,
        checkpoint_path=checkpoint,
    )
    print(f"RUN mutants={run.mutants_run} killed={run.killed} survived={len(run.survivors)} truncated={run.truncated} candidates={run.candidates_total}")
    for s in run.survivors:
        print(f"SURVIVOR {s}")


if __name__ == "__main__":
    main()
