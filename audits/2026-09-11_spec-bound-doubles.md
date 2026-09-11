# Spec-less doubles reaching duck-typed code -- pyutilz, 2026-09-11

Part of a seven-repository survey for the failure glossum paid for on 2026-09-05: a bare `Mock()` auto-creates
every attribute it is asked for, so code that decides with `getattr(obj, "x", default)` always sees the
attribute, and the branch for an object without it never runs. The overview, method and the other repositories'
results are in the social repository's `upwork/new_scraper/production_scrapers/audits/2026-09-11/03-tests.md`.

## F1 (Low) -- `archive_provider` would return a bare double unwrapped

`src/pyutilz/dev/attempt_archive.py:324`: `archive_provider` returns a provider as-is when
`getattr(provider, "_pyutilz_attempt_archive_installed", False)` is truthy, which is how it avoids wrapping twice.
A bare `Mock()` provider answers that attribute with a truthy child, so it would come back unwrapped and a test
about archiving would archive nothing. `metadata_from_provider` (`:229-258`) reads the provider's attributes the
same way.

No test does this today: `tests/test_dev_attempt_archive.py` uses `_FakeProvider` subclasses (`:166, 177, 188-195`).
The rest of the suite has 638 bare mocks, 133 of them bound to client/session-like names, and they feed HTTP and
LLM SDK clients whose methods are called rather than probed.

**Disposition:** RESOLVED 2026-09-11 as a guard. `tests/test_meta/test_shared_checks_wired.py::
test_no_bare_double_reaches_the_attempt_archive` wires `py_ci_shared.spec_bound_doubles` over `archive_provider`
and `metadata_from_provider` with no name hints (the hint `provider` flags nothing extra and would grow with the
LLM tests), floor 1 subject file. It flags nothing today; a bare mock passed to either fails it.
