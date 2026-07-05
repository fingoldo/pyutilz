# pyutilz — project conventions

## NEVER run a project-wide lint/format rewrite without explicit approval FIRST (CRITICAL, 2026-07-05)

Any auto-fixer or formatter run across the WHOLE repo (or a huge fraction of it) — `black .`, `ruff format .`, `ruff check --fix` beyond a narrow just-edited file set, `isort .`, `autopep8`, etc. — is EXPLICITLY OUT OF SCOPE unless the user approved that exact repo-wide run in this conversation. This holds even when the user asked to "fix all errors" / "fix everything" — a broad reformat is a DIFFERENT class of action (thousands of lines, every file touched) from fixing the errors a linter's real-bugs gate reports, and "fix all" does NOT imply consent for it.

**Why:** incident 2026-07-05 — asked to sync the ruff pre-commit gate with CI and fix the errors it surfaced (scoped, semantic fixes). When CI's `black --check` step then failed too, ran `black .` across the WHOLE repo unasked — 183 of 215 files, ~4000 lines changed — while the user was away. Reverted in full once caught, but it should never have run.

**How to apply:** when a project-wide formatter would fix a CI/lint failure, do NOT run it. Instead: report the scope (e.g. "N of M files need reformatting, that's a repo-wide rewrite"), and ask whether to (a) run it now, (b) turn the gate advisory/non-blocking, or (c) leave it for a deliberate separate pass. Small, scoped fixes to the specific files already being edited in the current task remain fine without asking — the line is "does this touch files/lines beyond what I'm already deliberately changing for a diagnosed reason".

**Black repo-wide reformat — resolved (2026-07-05):** user decided to exclude exactly two Black behaviors from any run, project-wide (mirrored in mlframe): (1) arg/collection-list explosion (multi-item packed line -> one-item-per-line, including `from x import (...)` blocks) and (2) blank-line insertion. Neither is configurable via a stock Black flag. `scripts/black_filtered_apply.py` applies everything else Black wants while mechanically rejecting those two, validated via AST-equivalence + compile checks. Use `python scripts/black_filtered_apply.py --config pyproject.toml --write <files>` (or `--check .` for CI/dry-run) — never raw `black`/`black --fix` in place.

## Ruff `--fix` scope (mirrors mlframe)

A broad / repo-wide / dir-wide pass is `ruff check` ONLY — no `--fix`, zero auto-edits. Read findings and fix anything worthwhile by hand. `--fix` is reserved for a narrow set of files just edited, and only for genuinely mechanical/safe rules (e.g. F541 empty f-strings) — never for rules that can change behavior or delete a re-export (F401).
