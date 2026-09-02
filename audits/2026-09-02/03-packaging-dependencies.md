# Packaging & Dependencies Audit — pyutilz (2026-09-02)

## Summary

Verified empirically, not by reading alone:

- **Full `pyproject.toml` read** (581 lines) plus a **real build**: copied `src/`, `pyproject.toml`, `README.md`, `LICENSE`, `CHANGELOG.md` to a scratch dir and ran `python -m build --no-isolation` (setuptools 83.0.0). Inspected the resulting `pyutilz-1.0.0-py3-none-any.whl` (186 entries) and `pyutilz-1.0.0.tar.gz` (216 entries) for METADATA, `Requires-Dist`/`Provides-Extra`, `py.typed`, license files and sdist contents. (Note: the build initially aborted with `project.classifiers[13] must be trove-classifier` — this is a **local-environment artifact only**: the installed `trove-classifiers` is 2024.4.10, which predates the `Programming Language :: Python :: 3.14` classifier. Not a repo defect; not reported below.)
- **AST scan of every module-level, unguarded import** across `src/pyutilz/**/*.py` (excluding `try:`/`if:`-guarded blocks), cross-referenced against core deps and every extras group. Result: **no undeclared unconditional third-party import remains** — every one of PIL, anthropic, dash, dash_bootstrap_components, dateutil, flask, httpx, jellyfish, joblib, numba, numpy, pandas, polars, portalocker, prefect, psutil, psycopg2, pyarrow, pydantic, pydantic_settings, pympler, redis, requests, scipy, sqlalchemy, tenacity, tqdm maps to a declared core dep or extras group. The 2026-07-21 Critical/High undeclared-import findings are **fixed**, not re-raised.
- Metadata-level duplicate/floor scan across all 16 extras: **zero divergent version specifiers** for any package declared in more than one place.
- Read all 8 workflows (`ci.yml`, `publish.yml`, `docs.yml`, `black-filtered.yml`, `mypy-full.yml`, `numba-coverage.yml`, `codecov-full.yml`, `dependabot-auto-merge.yml`), `.github/dependabot.yml`, `zizmor.yml`, the full `.pre-commit-config.yaml`, `mkdocs.yml`, and `tests/test_meta/test_optional_deps_isolation.py`.
- Re-checked the 2026-07-21 findings: the `system`/`gpu` blind spot in `test_optional_deps_isolation.py` is **fixed** (both groups plus `_LEAF_MODULE_OWN_GROUP` now present); `tomllib`/`tomli`, `pythonlib.py`'s numba/joblib/portalocker, and the polars `>=0.19` floor are all **fixed**. Community-health files treated as dispositioned won't-fix per CLAUDE.md.

Counts: **1 High, 4 Medium, 10 Low** (15 findings). No Critical.

## Findings

### F01. [High] `[dev]` extra carries a direct VCS URL dependency — PyPI rejects such distributions, so `publish.yml` cannot actually publish — pyproject.toml:326
- **Disposition**: OPEN
- **Category**: packaging/release-blocker
- **Problem**: `pyproject.toml:326` declares `"py-ci-shared @ git+https://github.com/fingoldo/py-ci-shared.git ; python_version >= '3.9'"`. The built wheel's METADATA carries it verbatim: `Requires-Dist: py-ci-shared@ git+https://github.com/fingoldo/py-ci-shared.git ; python_version >= "3.9" and extra == "dev"` (verified by reading `pyutilz-1.0.0.dist-info/METADATA` from the locally built wheel). PyPI refuses uploads whose metadata contains a PEP 440 direct reference (`400 ... Can't have direct dependency`). `twine check` does **not** catch this — it validates README rendering/metadata well-formedness only, which is why `publish.yml:66-69`'s "Check distribution metadata" step passes. `git tag` in this repo returns empty, so the publish workflow has never actually been exercised end-to-end.
- **Failure scenario**: A maintainer bumps the version, tags `v1.1.0`, and `publish.yml` runs the full test suite, builds, twine-checks, uploads the artifact — and then fails at the very last step (`pypa/gh-action-pypi-publish`, publish.yml:88-89) with a 400 from PyPI. Secondarily, even if it were accepted, the README-recommended `pip install pyutilz[all,dev]` (README.md:19, docs/index.md:21) would require `git` on the end user's machine and clone from GitHub at install time.
- **Suggested fix**: Move `py-ci-shared` out of the published `[dev]` extra into a non-metadata channel — a `requirements-dev.txt` (or a `dev-local` extra excluded from what is uploaded, or publish `py-ci-shared` to PyPI and depend on it by name/version). Add a release-gate step to `publish.yml` that greps the built METADATA for `@ git+` / `@ http` and fails before upload, so this class can never reach the PyPI POST again.

### F02. [Medium] README and docs still advertise "no hard deps" / "core only (numba, joblib, portalocker)" — the package now has 9 core requirements — README.md:14,21; docs/index.md:14,23; docs/modules.md:7
- **Disposition**: OPEN
- **Category**: docs-packaging-consistency
- **Problem**: The built wheel's METADATA lists 9 unconditional `Requires-Dist` lines: `numba>=0.56`, `numpy>=1.20`, `joblib>=1.2`, `portalocker>=2.0`, `psutil>=5.9`, `pandas>=1.5`, `tqdm>=4.65`, `pympler>=1.0`, `tomli>=2.0; python_version < "3.11"` (pyproject.toml:55-95). Meanwhile `docs/index.md:23` says `pip install pyutilz # core only, no hard deps`; `docs/modules.md:7` says "Core Python helpers with **zero hard dependencies**"; `README.md:21` and `README.md:14` / `docs/index.md:14` enumerate only "numba, joblib, portalocker". `pandas` alone makes a bare `pip install pyutilz` anything but light. Related staleness in the same blocks: `README.md:28` / `docs/index.md:30` describe `[system]` as "psutil + numba + GPUtil + tqdm + py-cpuinfo" but pyproject.toml:250-275 no longer declares psutil or numba in that group (they are core), and `[dataframes]` is described as including `numpy` (README.md:22) which it does not declare directly.
- **Failure scenario**: A user picks pyutilz for a constrained/embedded or minimal-image deployment on the documented promise of a dependency-free core, then discovers `pip install pyutilz` pulls pandas + numpy + numba + psutil + tqdm + pympler + joblib + portalocker.
- **Suggested fix**: Rewrite the intro sentence and the `pip install pyutilz` comment in README.md, docs/index.md, and docs/modules.md to list the actual core set and say why each is core (the rationale is already written out in pyproject.toml:44-95). Refresh the per-extra one-liners to match the current groups.

### F03. [Medium] `[nlp]` and `[llm]` extras are uninstallable on the declared-supported Python 3.8/3.9 — no `python_version` markers, though CI documents the incompatibility — pyproject.toml:12,27-28,210-247
- **Disposition**: OPEN
- **Category**: extras-python-support
- **Problem**: `requires-python = ">=3.8"` (pyproject.toml:12) and classifiers advertise 3.8 and 3.9 (pyproject.toml:27-28). `.github/workflows/ci.yml:39-49` documents in prose that `[nlp]` pulls `spacy -> thinc>=8.3.12` requiring Python >=3.10 and that `[llm]`'s `google-genai>=1.0` requires >=3.9, and works around both by hand-editing the install command per matrix leg (ci.yml:59-66) — installing a manually-transcribed subset of `[llm]`'s package list on 3.8. The extras themselves carry no markers, unlike `black` (pyproject.toml:317) and `py-ci-shared` (pyproject.toml:326), which do.
- **Failure scenario**: An end user on Python 3.9 runs `pip install pyutilz[nlp]` (README.md:26) and gets a resolver failure or a silent downgrade to an ancient spacy; on 3.8, `pip install pyutilz[llm]` (README.md:27) fails outright on google-genai. The CI workaround also means the hand-copied 3.8 `[llm]` package list (ci.yml:61) will silently drift from pyproject.toml:239-247 the next time that group changes.
- **Suggested fix**: Put the constraint in metadata rather than in CI shell: `"spacy>=3.0 ; python_version >= '3.10'"`, `"google-genai>=1.0 ; python_version >= '3.9'"` (matching the `black`/`py-ci-shared` precedent), then simplify ci.yml's install step to a single unconditional `pip install -e ".[...]"` for every leg. Alternatively raise `requires-python` if 3.8/3.9 are no longer genuinely supported.

### F04. [Medium] `project.license` TOML table + license classifier are deprecated; setuptools states a hard removal date of 2027-Feb-18 — pyproject.toml:10,23
- **Disposition**: OPEN
- **Category**: metadata-deprecation
- **Problem**: Verified by running the real build with setuptools 83.0.0: two `SetuptoolsDeprecationWarning`s are emitted, ``\`project.license\` as a TOML table is deprecated ... By 2027-Feb-18, you need to update your project and remove deprecated calls`` (from `pyproject.toml:10`, `license = {text = "MIT"}`) and `License classifiers are deprecated. Please consider removing the following classifiers in favor of a SPDX license expression: License :: OSI Approved :: MIT License` (from `pyproject.toml:23`). PEP 639 replaces both with `license = "MIT"` plus `license-files = ["LICENSE"]`.
- **Failure scenario**: After the announced removal date, `python -m build` (ci.yml:148-151 and publish.yml:61-64) fails with an error instead of a warning, breaking both the per-PR build gate and the release path at the same time — and the fix has to be made under release pressure.
- **Suggested fix**: `license = "MIT"`, add `license-files = ["LICENSE"]`, drop the `License :: OSI Approved :: MIT License` classifier, and raise `build-system.requires` to `setuptools>=77.0` (which is where both replacements land). Metadata-Version stays 2.4-compatible.

### F05. [Medium] `[all]` silently omits `dash`, `prefect`, `tensorflow` (and `gpu`), so the README-recommended "full install" cannot import three shipped modules — pyproject.toml:295-297
- **Disposition**: OPEN
- **Category**: extras-completeness
- **Problem**: `all = ["pyutilz[pandas,polars,database,web,cloud,nlp,llm,system,stats]"]` (pyproject.toml:296, confirmed in the built METADATA as `Requires-Dist: pyutilz[cloud,database,llm,nlp,pandas,polars,stats,system,web]; extra == "all"`). The `dash` (pyproject.toml:193-197), `prefect` (200-202) and `tensorflow` (205-207) groups are excluded with no comment, while the `gpu` exclusion right above IS documented (pyproject.toml:287-289). `src/pyutilz/dev/dashlib.py:22,24,25,26` imports flask/dash/dash_bootstrap_components unconditionally at module level, and `src/pyutilz/system/scheduling/prefect.py:22` imports prefect the same way (verified by AST scan). README.md:19 and docs/index.md:21 call `pip install pyutilz[all,dev]` the "full install (recommended)". The only place the exclusion is explained anywhere in the repo is a comment in `.github/workflows/mypy-full.yml:5-6`.
- **Failure scenario**: A user follows the README's recommended full install and then `import pyutilz.dev.dashlib` raises `ModuleNotFoundError: No module named 'flask'` — with nothing in the package metadata or README indicating that an extra beyond `[all]` was needed.
- **Suggested fix**: Either add an inline comment in pyproject.toml at line 295 explaining the exclusion (matching `gpu`'s existing rationale) and list `[dash]`/`[prefect]`/`[tensorflow]` in README.md's extras block, or fold them into `all`. Keep the pyproject comment as the single source of truth rather than mypy-full.yml.

### F06. [Low] Optional-dep isolation meta-test never masks the `dash`/`prefect`/`tensorflow` groups — same blind-spot class fixed for `system`/`gpu` on 2026-07-21 — tests/test_meta/test_optional_deps_isolation.py:44-70
- **Disposition**: OPEN
- **Category**: test-coverage-packaging
- **Problem**: `_OPTIONAL_DEP_GROUPS` (test_optional_deps_isolation.py:44-70) enumerates pandas, polars, database, web, cloud, nlp, llm, system, gpu. Three declared extras groups — `dash`, `prefect`, `tensorflow` — are absent, so `flask`, `dash`, `dash_bootstrap_components`, `prefect` and `tensorflow` are never masked by any scenario, and `_LEAF_MODULE_OWN_GROUP` (lines 99-109) never imports `pyutilz.dev.dashlib` or `pyutilz.system.scheduling.prefect`. The test's own comment at lines 65-69 documents that this exact structural gap for `system`/`gpu` is what let the 2026-07-21 undeclared-import bugs ship; the remaining three groups were not closed with it.
- **Failure scenario**: A new unconditional `import flask_login` (currently correctly try/except-guarded, dev/dashlib.py) or a new module-level third-party import added to `dashlib.py`/`prefect.py` ships with no declaration and no test failure — precisely the class this suite exists to catch.
- **Suggested fix**: Add `"dash": ["flask", "dash", "dash_bootstrap_components"]`, `"prefect": ["prefect"]`, `"tensorflow": ["tensorflow"]` to `_OPTIONAL_DEP_GROUPS`, and `"pyutilz.dev.dashlib": "dash"`, `"pyutilz.system.scheduling.prefect": "prefect"` to `_LEAF_MODULE_OWN_GROUP`. Better still, derive the group dict from `importlib.metadata`'s `Provides-Extra` so a newly added extra cannot be forgotten.

### F07. [Low] `[gpu]` declares `cupy`, which is a source-only distribution on PyPI — `pip install pyutilz[gpu]` triggers a full CUDA source build — pyproject.toml:290-292
- **Disposition**: OPEN
- **Category**: extras-installability
- **Problem**: `gpu = ["cupy>=12.0"]`. The `cupy` project on PyPI ships only an sdist; the prebuilt binaries are published under the CUDA-version-specific names (`cupy-cuda11x`, `cupy-cuda12x`, `cupy-rocm-*`). The comment at pyproject.toml:285-289 acknowledges "cupy's cudaXX-specific wheel resolution doesn't belong in a default 'everything' install" and excludes the group from `all`, but the group as declared still resolves to the source package.
- **Failure scenario**: A user runs `pip install pyutilz[gpu]` expecting a wheel and gets a multi-minute (or failing) NVCC compile requiring a full local CUDA toolkit — on a machine that may only have the driver.
- **Suggested fix**: Either keep the group but document in the comment that the caller is expected to install `cupy-cuda12x` themselves (and note every cupy import site is already try/except-guarded, so nothing breaks without it), or drop the pip-level declaration entirely and document the install command in README/docs instead of shipping an extra that cannot resolve to a wheel.

### F08. [Low] sdist contains no `tests/` and no `CHANGELOG.md` — no `MANIFEST.in` exists — repo root (absent file); pyproject.toml:341-347
- **Disposition**: OPEN
- **Category**: sdist-contents
- **Problem**: Verified from the locally built `pyutilz-1.0.0.tar.gz`: its top level contains exactly `LICENSE`, `PKG-INFO`, `README.md`, `pyproject.toml`, `setup.cfg`, `src/` — 216 entries, **zero** matching `/tests/`. `CHANGELOG.md` was present in the build directory and still did not make it in. There is no `MANIFEST.in` in the repo root, and `[tool.setuptools.packages.find]` (pyproject.toml:341-344) only governs the `src` package tree.
- **Failure scenario**: A downstream redistributor (conda-forge, a Linux distro, a vendored-source consumer) fetches the sdist and cannot run the test suite to validate the build, nor read the changelog offline — even though `project.urls.Changelog` (pyproject.toml:339) advertises the file and the repo maintains an extensive meta-test suite specifically designed to catch packaging drift.
- **Suggested fix**: Add a `MANIFEST.in` with `include CHANGELOG.md CONTRIBUTING.md TESTING.md`, `recursive-include tests *.py *.json`, `prune tests/**/__pycache__`, and add a CI assertion in ci.yml's `build` job that the sdist contains `tests/` (the job already builds and twine-checks, so the check is one extra line).

### F09. [Low] `wheel` in `build-system.requires` is obsolete for `setuptools.build_meta` — pyproject.toml:2
- **Disposition**: OPEN
- **Category**: build-backend-config
- **Problem**: `requires = ["setuptools>=61.0", "wheel"]`. Since PEP 517 builds, `setuptools.build_meta` declares its own `wheel` requirement dynamically via `get_requires_for_build_wheel`; listing it explicitly is the legacy `setup.py bdist_wheel` pattern and is documented by setuptools as unnecessary.
- **Failure scenario**: No functional break today; it pins an extra package into every isolated build environment and is a mild correctness/staleness signal that a reviewer has to re-derive. It will also need touching anyway when F04's `setuptools>=77.0` bump lands.
- **Suggested fix**: `requires = ["setuptools>=77.0"]` (folded into the F04 change).

### F10. [Low] `docs.yml` hardcodes `mkdocs-material>=9.5` instead of installing `.[docs]`; the `[docs]` extra is consumed by nothing — .github/workflows/docs.yml:125-127; pyproject.toml:330-332
- **Disposition**: OPEN
- **Category**: ci-config-drift
- **Problem**: `docs.yml:125-127` sets `install-command: pip install -e . --no-deps` followed by `pip install "mkdocs-material>=9.5"` — a hand-copied duplicate of `pyproject.toml:331`'s `docs = ["mkdocs-material>=9.5"]`. Grepping the repo, nothing installs `.[docs]`. Two copies of one floor, one of them the only one CI actually uses.
- **Failure scenario**: A future docs plugin (`mkdocstrings`, `mkdocs-gen-files`) added to the `[docs]` extra is not installed by the docs workflow, so the site build fails in CI (or, worse, silently builds without the plugin's output) despite the dependency being correctly declared.
- **Suggested fix**: `install-command: pip install -e ".[docs]"` and delete the duplicated pin.

### F11. [Low] `mkdocs.yml` does not set `strict: true` — broken nav/links build green — mkdocs.yml:1-9
- **Disposition**: OPEN
- **Category**: docs-build-config
- **Problem**: `mkdocs.yml` has no `strict` key (verified by full read). MkDocs' default is `strict: false`, so a missing nav target, a broken internal link, or an unrecognised config warning is printed and the build still exits 0. `docs.yml` runs the shared build-and-deploy workflow on every PR as a "validation" step (docs.yml:92-95), so that validation currently cannot fail on a broken reference. The nav (mkdocs.yml:49-55) references 5 files, all of which currently exist (`docs/index.md`, `docs/modules.md`, `docs/guides/{llm_providers,kernel_tuning_cache,safe_pickle}.md` — verified).
- **Failure scenario**: A guide is renamed or a cross-reference typo'd; the PR's docs job passes, and the published site silently ships a 404 link.
- **Suggested fix**: Add `strict: true` to `mkdocs.yml` (the site is already warning-clean since all nav targets resolve, so this is a no-cost ratchet).

### F12. [Low] Four core dependencies are redundantly re-declared inside five extras groups — floor-divergence waiting to happen — pyproject.toml:101,122,123,143,144,145,170,234,260,282,304
- **Disposition**: OPEN
- **Category**: dependency-duplication
- **Problem**: Metadata-level scan of the built wheel: `pandas` is core AND declared in extras `pandas`/`polars`/`database`/`nlp`/`system`; `tqdm` and `pympler` are core AND in `pandas`/`polars`/`system`; `numpy` is core AND in `stats`/`dev`. All floors currently match exactly (scan reports zero divergent specifiers), so there is no bug today — the issue is purely that a floor bump has to be made in up to six places. The `stats` extra (pyproject.toml:281-283) is now a complete no-op: its only member, `numpy>=1.20`, is already core.
- **Failure scenario**: Someone raises the core `pandas>=1.5` to `>=2.0` for a new API, misses one or more of the five extras copies, and pip's resolver silently keeps satisfying the union — until an install path that hits only the un-bumped copy in a constrained environment lands pandas 1.5 and the new API call raises `AttributeError` at runtime.
- **Suggested fix**: Delete the now-redundant redeclarations (leaving a short comment where the deleted line documented a real transitive rationale), and either drop the `stats` extra or leave it as an empty/documented alias so `pip install pyutilz[stats]` keeps working. Optionally add a meta-test asserting that no package name appears with two different specifiers across core + all extras.

### F13. [Low] CI matrix is `ubuntu-latest` only, despite Windows-specific code paths and a Windows-only development box — .github/workflows/ci.yml:23
- **Disposition**: OPEN
- **Category**: ci-matrix-coverage
- **Problem**: `os: [ubuntu-latest]` (ci.yml:23) with 7 Python versions and no Windows/macOS leg. The codebase has genuine Windows-specific behaviour: `src/pyutilz/system/system/probing.py` imports `wmi` (guarded, ignored as DEP001 at pyproject.toml:514 with the comment "Windows-only"), `src/pyutilz/dev/logginglib.py:8,27` documents `pywin32`/`pywin32_postinstall.py` setup, and CLAUDE.md-adjacent project conventions call out Windows cp1251 console-encoding constraints (there is even a `test_no_unicode_in_console_output.py` meta-test). The cost tradeoff is documented for a sibling repo in `.github/dependabot.yml:19-24` (Actions-minutes exhaustion; mlframe's Windows/macOS legs were the dominant cost), so this is a known, priced decision rather than an oversight — but nothing in `ci.yml` itself records it.
- **Failure scenario**: A path-separator, file-locking (`portalocker`), or `wmi`/`psutil` platform-branch regression lands green and only surfaces on the maintainer's own Windows box.
- **Suggested fix**: Either add a single reduced Windows leg (e.g. `windows-latest` × 3.11 only, `-m "not slow"`) to buy the platform signal at ~1 extra job, or add an explicit comment at ci.yml:23 recording the minutes-budget reason so a future reader does not read it as an oversight.

### F14. [Low] `publish.yml` jobs have no `timeout-minutes`, unlike every other workflow in the repo — .github/workflows/publish.yml:16-17,76-79
- **Disposition**: OPEN
- **Category**: ci-hygiene
- **Problem**: The `build` job (publish.yml:16-17) and `publish` job (publish.yml:76-79) declare no `timeout-minutes`, so both inherit GitHub's 6-hour default. Every other job in the repo sets one: ci.yml:19 (30), ci.yml:136 (10), numba-coverage.yml:43 (30), codecov-full.yml:57 (15), dependabot-auto-merge.yml:67 (5). The `build` job runs the full test suite (`pytest --no-cov -q`, publish.yml:59) with `[all,dev]` installed.
- **Failure scenario**: A hung test or a stalled `pip install` on the release path burns up to six hours of the account's Actions minutes — the same budget the dependabot pip ecosystem was disabled to protect (.github/dependabot.yml:19-24) — while the release appears to be "in progress".
- **Suggested fix**: `timeout-minutes: 30` on `build`, `timeout-minutes: 10` on `publish`.

### F15. [Low] CHANGELOG's 1.0.0 entry advertises a `pypiwin32` member of `[system]` that no longer exists anywhere — CHANGELOG.md:50; pyproject.toml:250-275
- **Disposition**: OPEN
- **Category**: changelog-packaging-consistency
- **Problem**: CHANGELOG.md:50 (inside the `## [1.0.0] - 2026-02-18` section) states: "New `[system]` extras: `py-cpuinfo`, `GPUtil`, `xmltodict`, `pypiwin32` (Windows)." `pypiwin32` is not declared in the current `pyproject.toml` (verified by grep; it survives only in the stale, git-ignored `src/pyutilz.egg-info/requires.txt:38,94` build artifact). `git log -S pypiwin32 -- pyproject.toml` shows it was removed in commit `4c6bd3a` ("deptry: close 38 more real gaps"). No Windows-marked extra exists in its place, and the only `pywin32` references in source are comments (`src/pyutilz/dev/logginglib.py:8,27`).
- **Failure scenario**: A Windows user reads the changelog, expects `pip install pyutilz[system]` to provide pywin32, and finds `logginglib`'s documented `concurrent-log-handler`/pywin32 setup path silently unavailable. A released section should not be rewritten, but the removal is undocumented in any later section.
- **Suggested fix**: Add a line to the `[Unreleased]` → Changed/Removed section noting that `pypiwin32` was dropped from `[system]` and why (unused by any import), rather than editing the historical 1.0.0 entry. Separately consider deleting the stale `src/pyutilz.egg-info/` directory from the working tree — it is git-ignored (.gitignore:25) but its `requires.txt` is now actively misleading when read locally.
