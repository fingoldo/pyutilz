# 40 - CI optimality and README badges (pyutilz)

Scope: `.github/workflows/*`, `README.md:3-12`, compared against `C:\Users\Admin\Machine learning\mlframe` (read only).
Data: `gh run list --repo fingoldo/pyutilz --limit 50`, `gh run view` job timings, badge SVGs fetched with curl on 2026-09-26.
Billing-blocked runs are not counted as defects.

## Measured runtime (last 50 runs)

| Workflow | Recent conclusions | Wall time |
|---|---|---|
| CI | failure x7, cancelled x1 | 20-110 min wall; last green run 35695001623: 21 test legs at 4-10 min each (~150 runner-min), ubuntu 4-8, windows 7-10, macOS 5-9 |
| Black | failure on every run | 2-13 min (most of it queue time) |
| MyPy | success | 1.3-26 min |
| docs | success | 0.8-12 min |
| numba-coverage-nightly | schedule success (job itself 1.5 min); every workflow_run leg skipped | 1.5 min job |
| codecov-full | schedule success; workflow_run legs skipped | 0.5 min |
| Dependabot auto-merge | failure x2 | <1 min |

## Findings

### CI-1 (High) -- Black gate red on every run: one real formatting finding

**Disposition:** RESOLVED -- the Black finding was not a formatting difference: the committed blob of `src/pyutilz/dev/code_audit/constructor_param_overwritten.py` used bare CR line terminators (324 CR, 18 CRLF). Line endings normalised to LF; `uvx black==26.5.1` leaves it unchanged and `python -m py_ci_shared.black_filtered_apply --config pyproject.toml --check src/pyutilz/dev/code_audit` reports "All 100 files are filtered-Black-clean". The next Black run on master has to confirm green.
Evidence: `.github/workflows/black-filtered.yml:20`; run 36227232589 log: `01 files have non-excluded-class Black findings: src/pyutilz/dev/code_audit/constructor_param_overwritten.py`. Every Black run in the last 50 failed.
Impact: the README Black badge (`README.md:5`) shows "failing". A gate that is always red stops being read, so a new formatting regression lands unnoticed.
Fix: run `uvx black==26.5.1` on that one file (same pin as the reusable workflow), commit it, and confirm the next run is green.

### CI-2 (High) -- Dependabot auto-merge cannot work: repo setting disabled

**Disposition:** RESOLVED -- `gh api -X PATCH repos/fingoldo/pyutilz -F allow_auto_merge=true` returned `allow_auto_merge: true` (gh authed as fingoldo, admin). HANDOFF: branch protection on master requires `mypy-full / mypy-full`, a context from the retired reusable-workflow era; the job now reports as `mypy (whole project, blocking, completion-asserted)`, so the required check never arrives and a Dependabot PR will still never auto-merge. Replace that required context in the repo settings (not changed here: only the auto-merge PATCH was authorised).
Evidence: `.github/workflows/dependabot-auto-merge.yml:40`; run 36176438242: `GraphQL: Auto merge is not allowed for this repository (enablePullRequestAutoMerge)`. `gh api repos/fingoldo/pyutilz` -> `allow_auto_merge: false`.
Impact: every Dependabot PR (e.g. #17) gets a red check, and none of them merge on their own. The workflow's header comment (lines 3-14) says they do.
Fix: turn on "Allow auto-merge" in the repo settings (`gh api -X PATCH repos/fingoldo/pyutilz -F allow_auto_merge=true`) and check that branch protection requires "CI required checks". Or, if auto-merge should stay off, delete the workflow. Nothing in the code needs to change.

### CI-3 (High) -- Every push runs the full 21-leg OS x Python matrix

**Disposition:** RESOLVED -- new `matrix-setup` job builds the matrix from `github.event_name` (`.github/workflows/ci.yml:45`); push runs ubuntu x {3.8, 3.11, 3.14} + windows 3.11 + macos 3.11 (5 legs); pull_request, the new weekly `schedule` (Sun 03:17 UTC, `ci.yml:19`) and `workflow_dispatch` with `full_matrix=true` run all 21 legs. `test` consumes it via `fromJSON` (`ci.yml:86`).
Evidence: `ci.yml:3-7` (push and PR, no filter), `ci.yml:36-50` (3 OS x 7 Python, including 2 macos-15-intel legs), `ci.yml:15` (push runs are never cancelled). In mlframe, `ci.yml` pushes run only the representative leg, and a weekly `schedule` (Sunday 03:17 UTC) runs the full OS/Python sweep. Its comment at mlframe `ci.yml:5-20` explains the ~20-concurrent-job budget shared across the account.
Impact: about 150 runner-minutes per push. macOS minutes are billed at 10x and Windows at 2x, and 9 legs are macOS. This repo pushes to master many times a day, so this is the main reason CI takes 60-110 min of wall time (the 110 min run was queued behind the concurrent-job cap) and the main load on the billing limit.
Fix: copy mlframe's pattern. On `push`, run ubuntu x {3.8, 3.11, 3.14} plus windows-latest x 3.11 plus macos-latest x 3.11. That covers the floor, the coverage leg, the ceiling, and both non-linux OS branches. Run the full 21-leg matrix on `pull_request` and on a weekly `schedule` plus `workflow_dispatch`. Build the matrix from `github.event_name` with a `fromJSON` expression in a small setup job, like mlframe's.

### CI-4 (Med) -- No paths-ignore on push: docs/audit-only commits run the full CI

**Disposition:** RESOLVED -- push-only `paths-ignore` (`**.md`, `docs/**`, `audits/**`, `LICENSE`) in `ci.yml:9`, `mypy-full.yml:27`, `black-filtered.yml:7`. Not on pull_request, since the required checks would then stay pending.
Evidence: `ci.yml:4-5`, `mypy-full.yml:24-25`, `black-filtered.yml:4-5` have no path filter. mlframe `ci.yml:11-16` uses `paths-ignore: ["**.md", "docs/**", "audits/**", "LICENSE"]` on push only, and explains why it stays off `pull_request` (the required check would stay pending forever).
Impact: every audit round commit to `audits/` (like this report) triggers 21 test legs, mypy and black.
Fix: add the same push-only `paths-ignore` block to ci.yml, mypy-full.yml and black-filtered.yml.

### CI-5 (Med) -- Tests run serially: no pytest-xdist

**Disposition:** RESOLVED -- `pytest-xdist>=3.5` added to [dev] (`pyproject.toml:362`); the test step runs `-n auto --dist=loadgroup` (`ci.yml:156`). The suite already runs under pytest-randomly, so order dependence is exercised. No existing test was found needing an `xdist_group` pin in the targeted meta run. The first CI run is the full-suite check, and any shared-state collision it shows gets an `xdist_group` pin.
Evidence: `ci.yml:121` `pytest -m "not gpu" --cov=... --durations=200` has no `-n`. `pyproject.toml:354-369` [dev] does not include pytest-xdist. `pyproject.toml:426` addopts has no `-n`. mlframe's shards use `-n auto --dist=loadgroup` (mlframe `ci.yml:324-352`).
Impact: GitHub runners have 4 vCPU (3 on macOS arm). The test step is most of each 4-10 min leg, so CI takes about 2-3x longer than it needs to.
Fix: add `pytest-xdist>=3.5` to [dev] and use `-n auto --dist=loadgroup` in ci.yml. pytest-cov combines worker data by itself, and pytest-randomly works with xdist. Before turning it on, pin any tests that write to shared temp or global state with `xdist_group`.

### CI-6 (Med) -- numba-disabled flag badge shows "unknown": no codecov.yml, no carryforward

**Disposition:** RESOLVED -- new `codecov.yml` with `flag_management.default_rules.carryforward: true` plus explicit `unit`/`numba-disabled`/`combined`, `coverage.status.project: off` and informational patch status (target 85%, the ratchet). ci.yml's upload now sends `flags: unit` (`ci.yml:164`). The numba/full badges will read "unknown" until the next upload of each flag after this lands.
Evidence: `README.md:7` -> shields returns `codecov-numba: unknown`. pyutilz has no `codecov.yml` (mlframe has one at its repo root). The numba upload does succeed: run 36118611749 went to commit a7fad26 with `flags: numba-disabled`, and the codecov CLI logged `No config file could be found`. The flag is uploaded only from the nightly on one commit, and each later push to master creates a HEAD commit without it. shields reads the HEAD commit, so the badge goes to unknown. The `combined` flag (`README.md:8`) currently reads 87% only because codecov-full uploaded more recently.
Impact: the numba badge is blank most of the day, and the combined badge will also go blank after every push.
Fix: add `codecov.yml` with `flag_management: default_rules: carryforward: true` (or `individual_flags` listing `numba-disabled` and `combined` with `carryforward: true`), plus `coverage.status.project: off` and informational patch status, as in mlframe's codecov.yml. Also give ci.yml's upload an explicit `flags: unit` (`ci.yml:127-130` sends no flag), so the three series can be told apart.

### CI-7 (Med) -- Codecov CLI regenerates coverage.xml and hits the whole-suite fail_under in the numba job

**Disposition:** RESOLVED -- the numba upload step sets `COVERAGE_RCFILE: .coveragerc-numba` (`numba-coverage.yml:154`), so the codecov CLI's coverage plugin regenerates the XML with the same gate-free rcfile (no fail_under, same omit). HANDOFF (py-ci-shared): the upload-codecov action has no `plugins`/`disable_search` input; adding `plugins: noop` there is still the cleaner fix for both repos.
Evidence: run 36118611749: `Generating coverage.xml report ... Coverage failure: total of 7.05 is less than fail-under=85.00`. The workflow carefully writes `.coveragerc-numba` without `fail_under` (`numba-coverage.yml:77-100`), but the codecov action's own coverage.xml plugin runs `coverage xml` with pyproject's `[tool.coverage.report] fail_under = 85` (`pyproject.toml:468`).
Impact: currently harmless (`fail-ci-if-error: false`), but the upload's coverage.xml may be regenerated without the numba rcfile's `omit` settings, and the log shows a misleading failure.
Fix: pass `plugins: noop` (or `disable_search: true` plus explicit `files`) through py-ci-shared's upload-codecov action so the CLI uploads the prepared XML as-is. Do this once in py-ci-shared so both repos get it.

### CI-8 (Med) -- Chained coverage workflows almost never fire because CI is never green

**Disposition:** RESOLVED -- per-job logs of run 36222503752 (jobs API; `gh run view --log-failed` stream-errored) showed that none of the failures were platform-specific: (a) 17-24 LLM tests `TypeError: ... not _Unset` on every leg, (b) `test_no_stale_todos` refusing the shallow clone on every leg -- both already fixed in e547038; (c) every 3.8 leg (ubuntu/windows/macos-intel) failing collection of `tests/test_llm_openrouter_decisions.py` with `'ABCMeta' object is not subscriptable`, which dragged coverage to 19.68%: `src/pyutilz/llm/openrouter_decisions.py:130` runtime alias subscripted `collections.abc.Mapping`. It now uses string forward references (verified on a 3.8 interpreter). With the 5-leg push matrix (CI-3), the workflow_run-chained coverage jobs can fire again.
Evidence: `numba-coverage.yml:18-21,44` and `codecov-full.yml:36-39` chain through `workflow_run` and require `conclusion == 'success'`. Every workflow_run leg in the last 50 runs was `skipped`, and only the daily crons ran. CI is red on test failures in windows/macOS legs (run 36222503752: windows 3.11/3.12 and macOS 3.12/3.13/3.14 fail in "Run tests with coverage").
Impact: the "after every push" freshness described at `numba-coverage.yml:14-17` does not happen. The coverage badges lag by up to 24 h. The red matrix is also a real problem of its own.
Fix: repair the failing windows/macOS test legs (out of scope for this audit; triage separately). Once CI-3 lands, the push matrix is smaller and should be green much more often.

### CI-9 (Low) -- fetch-depth: 0 on all 21 test legs

**Disposition:** RESOLVED -- `fetch-depth` is 0 only on ubuntu 3.11 (`ci.yml:100`). Other legs set `PYUTILZ_CI_SHALLOW_CLONE=1` (`ci.yml:154`). `test_no_stale_todos` (`tests/test_meta/test_shared_checks_wired.py:203`) skips only when that variable is set AND git reports a shallow repository, so a local shallow clone still fails loudly.
Evidence: `ci.yml:58`. It is needed only by `test_no_stale_todos`, which uses git blame.
Impact: full-history clone on every leg, about 5-20 s each on Windows and macOS.
Fix: keep `fetch-depth: 0` only on the ubuntu 3.11 leg (`${{ (matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11') && 0 || 1 }}`). On the other legs, have the test skip explicitly on a shallow clone. Or leave it as is: the cost is small.

### CI-10 (Low) -- pip rather than uv for the heavy install

**Disposition:** RESOLVED -- `astral-sh/setup-uv@11f9893` (v8.3.2, mlframe's pin) with `enable-cache` keyed on `pyproject.toml` + `requirements-dev.txt`, and `uv pip install` with `UV_SYSTEM_PYTHON=1`, in `ci.yml:110`, `mypy-full.yml:58`, `numba-coverage.yml:60`. pip-audit.yml gets `cache: pip` (`pip-audit.yml:41`); it stays on pip because pip-audit audits the pip environment.
Evidence: `ci.yml:74-83`, `mypy-full.yml:56-59`, `numba-coverage.yml`, `pip-audit.yml:46-48` (pip-audit also has no `cache: pip`, lines 38-40). mlframe installs with `astral-sh/setup-uv@11f9893...` + `uv pip install --system` (mlframe `ci.yml:245-275`). The Black reusable workflow in pyutilz already uses uv.
Impact: the `[pandas,polars,database,system,llm,nlp,cloud,speedups,dev]` resolve plus install takes a noticeable part of each leg.
Fix: switch to `astral-sh/setup-uv` with `enable-cache: true` and `uv pip install --system`, keyed on `pyproject.toml` and `requirements-dev.txt`. The pip cache key currently ignores `requirements-dev.txt` (`ci.yml:65`), so that should be added too.

### CI-11 (Low) -- Missing gates that mlframe has: dependency-review and CodeQL

**Disposition:** RESOLVED -- `.github/workflows/dependency-review.yml` and `.github/workflows/codeql.yml` ported from mlframe (branches `[main, master]`). pip-audit installs `.[all]` (`pip-audit.yml:51`), declared in `tests/test_meta/test_gate_integrity.py` `_DECLARED_DIFFERENT_EXTRAS`.
Evidence: mlframe has `dependency-review.yml` (PR-only, `fail-on-severity: high`) and `codeql.yml`. pyutilz has neither. Its `pip-audit.yml:14-17` is a weekly cron that audits only the base install (`pip install -e .`, line 47), not the extras.
Fix: copy mlframe's `dependency-review.yml` as-is. Optionally add CodeQL for python. Make pip-audit install `.[all]` so the extras' dependency closure gets audited too.

### CI-12 (Low) -- ubuntu-latest will move to Ubuntu 26 on 2026-10-19; Node-20 actions inside the reusable workflows

**Disposition:** WON'T FIX (here) -- HANDOFF to py-ci-shared: bump the Node-20 pins in its reusable workflows (`actions/checkout@34e1148`, `actions/setup-python@a26af69`, `astral-sh/setup-uv@d4b2f3b`) and add a `plugins` input to upload-codecov (CI-7), cut a release, then move the `@6a8e382` refs here (`black-filtered.yml`, `ci.yml` ruff/lint/upload steps, `numba-coverage.yml`, `codecov-full.yml`, `requirements-dev.txt`). The ubuntu-26 image switch on 2026-10-19 needs watching on the first runs after it.
Evidence: annotations on runs 36227232589 and 36176438242. The Black reusable workflow (py-ci-shared v1.17.0) still pins `actions/checkout@34e1148`, `actions/setup-python@a26af69` and `astral-sh/setup-uv@d4b2f3b`, which are Node-20 builds. pyutilz's own pins (`checkout@3d3c42e` v7.0.1, `setup-python@5fda3b9` v7.0.0) are already current.
Fix: bump those pins in py-ci-shared and then update the `@6a8e382` refs here (`black-filtered.yml:20-22`, `ci.yml:127,145,152,162`, `numba-coverage.yml:140`, `codecov-full.yml:197`). Watch the first runs after 10-19 for image changes (for example the spaCy model download and system libraries).

### CI-13 (Low) -- publish.yml reruns the full suite on tag push with no cache

**Disposition:** WON'T FIX -- adding `cache: pip` to publish.yml makes zizmor (run blocking in lint-blocking) raise `cache-poisoning` (high) on the tag-triggered release job, so it was reverted. The suite rerun stays: releases are rare, and it is the last check before an immutable upload.
Evidence: `publish.yml:16-22,56-58`: the full test suite with `[all,dev]`, 30 min timeout, and `setup-python` without `cache:`.
Impact: releases are rare, so the cost is minor. The tagged commit has already passed CI on master, and the ancestor check is at `publish.yml:29`.
Fix: add `cache: pip`. Optionally replace the suite rerun with a check that the tagged SHA has a green CI run: `gh run list --commit $GITHUB_SHA --workflow ci.yml --status success`.

#### Checked and fine
- Permissions: every workflow sets top-level `contents: read`. Write scopes are job-level: publish uses `id-token: write` on the publish job only; docs has `pages`/`id-token`, and the reason is documented at `docs.yml:19-25`; the auto-merge workflow has `contents/pull-requests: write` gated by `if: dependabot[bot]`.
- Action pinning: every third-party action is pinned by SHA with a version comment. py-ci-shared is on one SHA everywhere (`6a8e382`, v1.17.0).
- Timeouts: set on every job (CI 60, mypy 45, numba 30, codecov-full 15, pip-audit 15, publish 30/10, build 10, auto-merge 5).
- Concurrency: set on all push/PR workflows. CI's PR-only cancel (`ci.yml:15`) is a deliberate, documented choice. codecov-full.yml and publish.yml have none, which is acceptable for cron/tag triggers.
- fail_under consistency: pyproject `fail_under = 85` (`pyproject.toml:468`) matches `--cov-fail-under=85` (`ci.yml:121`). The narrow numba run and the merged report strip it from a derived rcfile (`numba-coverage.yml:90`, `codecov-full.yml:178`), so the config is derived rather than overridden with 0.
- `persist-credentials: false` on every checkout. `workflow_run` legs are gated on `event == 'push'` (zizmor-clean).

## README badges

Checked 2026-09-26:

| README line | Badge | Resolves? | Rendered |
|---|---|---|---|
| 3 | CI (`/workflows/CI/badge.svg`, legacy form) | yes | "CI - failing" (no branch/event filter, so PR runs count) |
| 4 | MyPy | yes | passing |
| 5 | Black (legacy form) | yes | failing (CI-1) |
| 6 | codecov (codecov.io native) | yes | SVG served |
| 7 | codecov-numba (shields, /master, flag) | yes | unknown (CI-6) |
| 8 | codecov-full (shields, /master, flag=combined) | yes | 87% |
| 9 | PyPI version | **broken** | "package or version not found": `https://pypi.org/pypi/pyutilz/json` returns 404, so pyutilz is not on PyPI |
| 10 | Python versions (pypi) | **broken** | same 404 |
| 11 | License MIT | yes | static |
| 12 | docs (workflow badge) | yes | passing; https://fingoldo.github.io/pyutilz/ serves the site |

### CI-14 (Med) -- PyPI version and pyversions badges point at a package that does not exist

**Disposition:** RESOLVED -- PyPI version/pyversions badges removed from README.md; replaced by a static Python 3.8-3.14 badge. Add the pypi `v` and `dm` badges back after the first release.
Evidence: `README.md:9-10`; the PyPI JSON API returns 404 for `pyutilz`. A downloads badge would fail the same way ("package not found").
Fix: until the first `v*` tag is published through publish.yml, replace them with a static Python badge generated from pyproject classifiers (`pyproject.toml:22,37-43`: 3.8-3.14) and drop the PyPI version badge. Put the pypi badges (plus `pypi/dm`) back once a release exists.

### CI-15 (Low) -- Workflow badges use the legacy `/workflows/<name>/badge.svg` form with no branch or event filter

**Disposition:** RESOLVED -- all workflow badges use `actions/workflows/<file>.yml/badge.svg?branch=master&event=push` (README.md badge block).
Evidence: `README.md:3,5`. mlframe uses `actions/workflows/<file>.yml/badge.svg?branch=master&event=push` (mlframe `README.md:6-9`).
Impact: a failing PR or feature-branch run turns the README badge red. The legacy form also breaks if a workflow's `name:` is renamed.
Fix: use the file-based form with `?branch=master&event=push`, as below.

### CI-16 (Low) -- Missing badges that mlframe shows

**Disposition:** RESOLVED -- Ruff, pre-commit, Python-range and py.typed badges added. Every badge URL in the block was fetched on 2026-09-26: all return 200 SVGs (CI/Black currently "failing" pending this change, codecov-numba/full "unknown" pending CI-6 carryforward).
Ruff, pre-commit, py.typed (pyutilz ships `py.typed`, `pyproject.toml:418`), and an explicit Python-range badge.

#### Proposed badge block (replaces README.md:3-12)

```markdown
[![CI](https://github.com/fingoldo/pyutilz/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/pyutilz/actions/workflows/ci.yml?query=branch%3Amaster)
[![MyPy](https://github.com/fingoldo/pyutilz/actions/workflows/mypy-full.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/pyutilz/actions/workflows/mypy-full.yml?query=branch%3Amaster)
[![Black](https://github.com/fingoldo/pyutilz/actions/workflows/black-filtered.yml/badge.svg?branch=master&event=push)](https://github.com/fingoldo/pyutilz/actions/workflows/black-filtered.yml?query=branch%3Amaster)
[![coverage](https://img.shields.io/codecov/c/github/fingoldo/pyutilz/master?label=coverage)](https://codecov.io/gh/fingoldo/pyutilz)
[![codecov-numba](https://img.shields.io/codecov/c/github/fingoldo/pyutilz/master?flag=numba-disabled&label=codecov-numba)](https://codecov.io/gh/fingoldo/pyutilz/flags)
[![codecov-full](https://img.shields.io/codecov/c/github/fingoldo/pyutilz/master?flag=combined&label=codecov-full)](https://codecov.io/gh/fingoldo/pyutilz/flags)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Python 3.8-3.14](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://github.com/fingoldo/pyutilz)
[![types: py.typed](https://img.shields.io/badge/types-py.typed-blue.svg)](https://peps.python.org/pep-0561/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![docs](https://github.com/fingoldo/pyutilz/actions/workflows/docs.yml/badge.svg?branch=master&event=push)](https://fingoldo.github.io/pyutilz/)
```

After the first PyPI release, add:

```markdown
[![PyPI](https://img.shields.io/pypi/v/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![Downloads](https://img.shields.io/pypi/dm/pyutilz.svg)](https://pypi.org/project/pyutilz/)
```

The numba and full flag badges show real numbers only after CI-6 (carryforward) lands.

## Counts
High 3 (CI-1, CI-2, CI-3), Med 6 (CI-4, CI-5, CI-6, CI-7, CI-8, CI-14), Low 7 (CI-9 to CI-13, CI-15, CI-16). Total 16.
