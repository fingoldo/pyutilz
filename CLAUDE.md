# pyutilz — project conventions

## NEVER run a project-wide lint/format rewrite without explicit approval FIRST (CRITICAL, 2026-07-05)

Any auto-fixer or formatter run across the WHOLE repo (or a huge fraction of it) — `black .`, `ruff format .`, `ruff check --fix` beyond a narrow just-edited file set, `isort .`, `autopep8`, etc. — is EXPLICITLY OUT OF SCOPE unless the user approved that exact repo-wide run in this conversation. This holds even when the user asked to "fix all errors" / "fix everything" — a broad reformat is a DIFFERENT class of action (thousands of lines, every file touched) from fixing the errors a linter's real-bugs gate reports, and "fix all" does NOT imply consent for it.

**Why:** incident 2026-07-05 — asked to sync the ruff pre-commit gate with CI and fix the errors it surfaced (scoped, semantic fixes). When CI's `black --check` step then failed too, ran `black .` across the WHOLE repo unasked — 183 of 215 files, ~4000 lines changed — while the user was away. Reverted in full once caught, but it should never have run.

**How to apply:** when a project-wide formatter would fix a CI/lint failure, do NOT run it. Instead: report the scope (e.g. "N of M files need reformatting, that's a repo-wide rewrite"), and ask whether to (a) run it now, (b) turn the gate advisory/non-blocking, or (c) leave it for a deliberate separate pass. Small, scoped fixes to the specific files already being edited in the current task remain fine without asking — the line is "does this touch files/lines beyond what I'm already deliberately changing for a diagnosed reason".

**Black repo-wide reformat — resolved (2026-07-05):** user decided to exclude exactly two Black behaviors from any run, project-wide (mirrored in mlframe): (1) arg/collection-list explosion (multi-item packed line -> one-item-per-line, including `from x import (...)` blocks) and (2) blank-line insertion. Neither is configurable via a stock Black flag. Originally `scripts/black_filtered_apply.py`; extracted 2026-07-09 into the shared `py_ci_shared` package (https://github.com/fingoldo/py-ci-shared, installed via `pip install -e ".[dev]"`) so pyutilz and mlframe consume the identical script instead of drifting copies. Applies everything else Black wants while mechanically rejecting those two, validated via AST-equivalence + compile checks. Use `python -m py_ci_shared.black_filtered_apply --config pyproject.toml --write <files>` (or `--check .` for CI/dry-run) — never raw `black`/`black --fix` in place.

## Ruff `--fix` scope (mirrors mlframe)

A broad / repo-wide / dir-wide pass is `ruff check` ONLY — no `--fix`, zero auto-edits. Read findings and fix anything worthwhile by hand. `--fix` is reserved for a narrow set of files just edited, and only for genuinely mechanical/safe rules (e.g. F541 empty f-strings) — never for rules that can change behavior or delete a re-export (F401).

## Write mypy-clean, fully-annotated code from the start (CRITICAL, 2026-07-08)

New or edited code must type-check under strict mypy the first time, not after a later cleanup pass. A multi-thousand-error backlog (mlframe: 3979 -> 0 over one long sweep) exists ONLY because code was routinely written without this discipline; do not add to a fresh backlog in either repo.

**Why:** the sweep's own error log is the evidence. The dominant categories were all self-inflicted at write-time, not inherent to the domain: implicit-`Optional` defaults (`x: T = None` instead of `Optional[T] = None`, PEP 484 — by far the single largest category), `no-any-return` from returning a numpy/pandas arithmetic chain without a final concrete-type wrap, `dtype: object` on params that always receive a dtype class, dict/list literals with no annotation whose later use needed a wider type than mypy inferred from the first assignment, and return-type annotations that didn't match the actual `return` statements (`-> float` on a function that returns a tuple). None of these require design work to avoid — they are pure write-time discipline, and each one found downstream cost far more to diagnose and fix (plus review overhead) than getting it right the first time would have.

**How to apply, when writing or editing a function:**
1. **Never write `param: T = None`.** If `None` is a valid default, the annotation is `Optional[T] = None` (or `T | None = None`). No exceptions — this was the single biggest error class in the sweep.
2. **Match the return annotation to what the function actually returns**, checked by re-reading every `return` statement, not assumed from the function name or a stale docstring. If a helper returns a tuple, annotate `-> tuple[...]`, never leave a lazy `-> None` or `-> float` that doesn't match.
3. **Annotate dict/list/set literals explicitly when their later use needs a wider type than the first assignment implies** — e.g. `results: dict[str, Any] = {}` up front if a later branch stores a different value type than the first, rather than letting mypy infer a narrow type from line 1 and fail three lines down. This applies doubly to any collection later passed across a function boundary.
4. **When a function returns a numpy/pandas arithmetic expression**, wrap the final returned value in the concrete constructor matching the declared return type (`np.asarray(...)`, `float(...)`, `bool(...)`) rather than trusting the stub to infer through arithmetic — numpy/pandas stubs lose concrete types through chained operations and helper calls extremely easily.
5. **Give every parameter that only ever holds a concrete class a concrete type**, not `object` — e.g. a `dtype` param that always receives `np.int64`/`np.float32` is `dtype: type`, never `dtype: object`.
6. **For attributes set dynamically** (via a params-to-self helper, an externally-bound method, or a mixin whose attribute is set by a sibling class in the MRO), declare the attribute's type at class scope even though no `__init__` assignment triggers mypy's normal inference — this documents the contract and is useful independent of mypy.
7. **Never silence a mypy error with the laziest fix that also changes behavior** (e.g. wrapping an already-`list`-typed `.tolist()` result in a second, pointless `list(...)` just to make a return-type complaint go away) — understand what the type actually is at runtime and either annotate correctly or use `typing.cast` (zero runtime cost) instead of a redundant copy.
8. **Run `mypy` on any file you touch before considering the change done**, the same way you'd run the relevant tests. A clean mypy run on the file(s) you just wrote/edited is part of "finished," not a follow-up chore for someone else.

This is a general engineering-discipline rule, not a project-specific one — it applies to every new function and every edit to an existing one, in both `mlframe` and `pyutilz`.

## Community-health files: intentionally absent (2026-07-21)

`SECURITY.md`, `CODE_OF_CONDUCT.md`, `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`, and `CITATION.cff` are deliberately NOT present in this repo. This is a decision, not a gap — do not add any of them, and do not flag their absence as a finding in a future audit.

(`CODEOWNERS` was a separate, later Low finding in the same audit batch — not covered by the user's explicit won't-fix decision above; it was added as a minimal file rather than declared won't-fix, see repo root/`.github/CODEOWNERS`.)

**Why:** raised during the 2026-07-21 full-repo audit, which flagged their absence as a Medium/Low finding per standard OSS-repo-hygiene conventions. The user explicitly said this is intentional behavior for this project and asked that it be documented so future audits/sessions stop proposing them.

**How to apply:** if an audit, linter, or template-completeness check surfaces "missing SECURITY.md" / "missing CODE_OF_CONDUCT.md" / "no issue or PR templates" / "no CITATION.cff" for this repo, treat it as already dispositioned (won't-fix) — do not create the file, and do not re-raise it as an open item.
