"""Targeted mutation re-run: re-check only the survivor lines of a log, against a given test set.

Usage: python scripts/mutation_rerun_lines.py <module.py> <tests,comma,separated> <survivor_log> <notes.json> [min_line max_line]

The checkpoint is notes.json with the .ckpt.jsonl suffix, so each notes file gets its own fresh verdict cache.
Survivors are only written to tests/test_meta/mutation_survivors.json when the notes file explains them.
"""

import json
import re
import sys
from pathlib import Path

from py_ci_shared.mutation_teeth import find_surviving_mutants

REPO = Path(__file__).resolve().parents[1]
SURVIVORS = REPO / "tests" / "test_meta" / "mutation_survivors.json"


def main() -> None:
    module, tests, log_path, notes_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    lo, hi = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (1, 10**9)
    notes: dict[str, str] = json.loads(notes_path.read_text(encoding="utf-8"))
    name = module.name
    pattern = re.compile(rf"^(?:SURVIVOR|mutation survivor: ).*{re.escape(name)}:(\d+):", re.M)
    lines = sorted({int(m.group(1)) for m in pattern.finditer(log_path.read_text(encoding="utf-8")) if lo <= int(m.group(1)) <= hi})
    print(f"LINES {len(lines)}", flush=True)
    if not lines:
        return
    checkpoint = notes_path.with_suffix(".ckpt.jsonl")
    run = find_surviving_mutants(
        module,
        [Path(t) for t in tests.split(",")],
        str(REPO),
        lines=[range(n, n + 1) for n in lines],
        timeout=600,
        use_cache=False,
        limit=600,
        checkpoint_path=checkpoint,
    )
    print(f"RUN mutants={run.mutants_run} killed={run.killed} survived={len(run.survivors)} truncated={run.truncated}", flush=True)
    accepted: dict[str, str] = {}
    for m in run.survivors:
        text = str(m)
        loc = text.split(":", 1)[1].split()[0] if ":" in text else ""
        print("SURVIVOR", text, flush=True)
        if loc in notes:
            accepted[m.key] = notes[loc]
        else:
            print("UNCLASSIFIED", text, flush=True)
    existing: dict[str, str] = json.loads(SURVIVORS.read_text(encoding="utf-8"))
    existing.update(accepted)
    SURVIVORS.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {len(accepted)} new of {len(run.survivors)} survivors; total entries {len(existing)}", flush=True)


if __name__ == "__main__":
    main()
