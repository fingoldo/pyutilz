"""Counted facts stated in prose are computed from the repo, not typed by hand.

Stale hand-typed counts are the one documentation defect that recurred in every audit wave:
a "24 aliases" claim against a 27-entry map, a "1900+ tests" claim understating the suite by
about 1600, a meta-test file count that went stale inside the session that wrote it. Every
claim registered below is CORRECT today -- registering them is what keeps them correct,
since each now fails the commit that invalidates it instead of the audit six weeks later.

Anchor-loss is a finding in its own right: if the prose around a claim is reworded so the
pattern stops matching, the claim has silently stopped being checked, and the check says so
rather than passing vacuously.

Runtime: a handful of file reads plus imports of pyutilz submodules the meta-suite already loads
(`code_audit`'s registry, the decodo constant tables, `pyproject.toml`'s coverage floor) -- well
under a tenth of a second. No network, no subprocess. The pytest-collection-count claim the audit
contemplated is deliberately NOT registered here: `pytest --collect-only` costs seconds on
every commit, and TESTING.md has already stopped pinning a test count for that reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("py_ci_shared", reason="py-ci-shared is a dev-only git dependency (requirements-dev.txt)")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _alias_count() -> int:
    import pyutilz

    return len(pyutilz._MODULE_ALIASES)


def _provider_count() -> int:
    from pyutilz.llm.factory import _PROVIDER_MODULES

    return len(_PROVIDER_MODULES)


def _shipped_module_count() -> int:
    return len(list((REPO_ROOT / "src" / "pyutilz").rglob("*.py")))


def _scanner_count() -> int:
    from pyutilz.dev.code_audit import get_scanners

    return len(get_scanners())


def _decodo_location_count() -> int:
    from pyutilz.web.proxy.decodo import ALLOWED_LOCATIONS

    return len(ALLOWED_LOCATIONS)


def _decodo_location_country_count() -> int:
    from pyutilz.web.proxy.decodo import ALLOWED_LOCATIONS

    # Each entry's username segment is `country-<iso>-city-<name>`; the claim counts distinct ISO codes.
    return len({entry["country_iso"].split("-")[1] for entry in ALLOWED_LOCATIONS})


def _decodo_country_count() -> int:
    from pyutilz.web.proxy.decodo import ALLOWED_COUNTRIES

    return len(ALLOWED_COUNTRIES)


def _coverage_floor() -> int:
    """The enforced coverage gate, read from `[tool.coverage.report] fail_under`."""
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return int(tomllib.load(f)["tool"]["coverage"]["report"]["fail_under"])


def _claims():
    from py_ci_shared.prose_numeric_claims import NumericClaim

    return [
        NumericClaim(
            path=REPO_ROOT / "TESTING.md",
            pattern=r"The (\d+)-entry backward-compat module alias map",
            compute=_alias_count,
            description="backward-compat module alias count",
        ),
        NumericClaim(
            path=REPO_ROOT / "README.md",
            pattern=r"Unified LLM interface across (\d+) providers",
            compute=_provider_count,
            description="LLM provider count",
        ),
        NumericClaim(
            path=REPO_ROOT / "docs" / "index.md",
            pattern=r"Unified LLM interface across (\d+) providers",
            compute=_provider_count,
            description="LLM provider count (docs site copy of the same claim)",
        ),
        NumericClaim(
            path=REPO_ROOT / ".pre-commit-config.yaml",
            pattern=r"0 errors; (\d+) source files",
            compute=_shipped_module_count,
            description="mypy source-file count in the mypy-full-blocking hook's rationale",
        ),
        # The CI half of the same figure. Registered on 2026-09-03 because it had gone 26 files
        # stale (216 vs 242) and disagreed with the hook's copy above -- and the number is not
        # decorative: it is what the gate's `--min-files` floor is sized from, in both venues.
        NumericClaim(
            path=REPO_ROOT / ".github" / "workflows" / "mypy-full.yml",
            pattern=r"0 errors, (\d+) source files",
            compute=_shipped_module_count,
            description="mypy source-file count in mypy-full.yml's header rationale",
        ),
        NumericClaim(
            path=REPO_ROOT / "CHANGELOG.md",
            pattern=r"\((\d+) `gate\.decodo\.com` endpoints across \d+ countries",
            compute=_decodo_location_count,
            description="decodo residential exit inventory size (len(ALLOWED_LOCATIONS))",
        ),
        NumericClaim(
            path=REPO_ROOT / "CHANGELOG.md",
            pattern=r"`gate\.decodo\.com` endpoints across (\d+) countries",
            compute=_decodo_location_country_count,
            description="distinct countries in the decodo residential inventory",
        ),
        NumericClaim(
            path=REPO_ROOT / "CHANGELOG.md",
            pattern=r"the (\d+) exit countries available on the DATACENTER endpoint",
            compute=_decodo_country_count,
            description="decodo datacenter exit-country count (len(ALLOWED_COUNTRIES))",
        ),
        NumericClaim(
            path=REPO_ROOT / "README.md",
            pattern=r"AST-based `code_audit` bug-class scanner \+ CLI \((\d+) scanners",
            compute=_scanner_count,
            description="registered code_audit scanner count",
        ),
        NumericClaim(
            path=REPO_ROOT / "docs" / "modules.md",
            pattern=r"The code_audit registry currently holds (\d+) scanners",
            compute=_scanner_count,
            description="registered code_audit scanner count (docs site copy)",
        ),
        NumericClaim(
            path=REPO_ROOT / "CONTRIBUTING.md",
            pattern=r"\(currently (\d+); it is a ratchet",
            compute=_coverage_floor,
            description="coverage floor quoted in CONTRIBUTING's Writing Tests section",
        ),
        NumericClaim(
            path=REPO_ROOT / "CONTRIBUTING.md",
            pattern=r"`--cov-fail-under` floor, currently (\d+)\)",
            compute=_coverage_floor,
            description="coverage floor quoted in CONTRIBUTING's PR process",
        ),
    ]


def test_prose_numeric_claims_match_the_repo():
    """Every registered count in prose equals the thing it counts."""
    from py_ci_shared.prose_numeric_claims import assert_numeric_claims_match

    assert_numeric_claims_match(_claims())


# Figures that look volatile to the scanner but are not measurements of this repo, so
# demanding a date on them would be wrong rather than merely pedantic. Reviewed by line
# content, not by line number, so an insertion above does not silently re-arm them.
_REVIEWED_NON_MEASUREMENTS = (
    r"only logs when call exceeds 2s",  # a threshold in a code example, not a measured duration
    r"Gemini's 90% input-token",  # a vendor's published discount
    r"Reduces memory by up to 80% for categorical data",  # a property of the dtype, not a benchmark
    r"currently 82; it is a ratchet",  # the enforced coverage gate, registered as a claim above
    r"`--cov-fail-under` floor, currently 82",  # the same gate, quoted in the PR process
)


def test_volatile_prose_figures_carry_a_date_qualifier():
    """WARN-only: a measured percentage or runtime in prose should say when it was measured.

    Deliberately non-blocking. Measured on this repo's four prose files, the rule's raw
    output was 8 findings of which 2 were real -- prose is full of thresholds, vendor
    figures and contributor targets that are shaped exactly like a stale measurement and
    are not one. The reviewed list above suppresses those, so the warning fires only on a
    NEW undated figure; blocking on a 75%-false-positive rule would train people to ignore
    it, which costs more than the drift it would catch.
    """
    from py_ci_shared.prose_numeric_claims import find_undated_volatile_claims

    findings = find_undated_volatile_claims(
        paths=[REPO_ROOT / "README.md", REPO_ROOT / "TESTING.md", REPO_ROOT / "CONTRIBUTING.md"],
        covered_patterns=_REVIEWED_NON_MEASUREMENTS,
        context_lines=8,
    )
    if findings:
        import warnings

        warnings.warn("Undated volatile figures in prose:\n  " + "\n  ".join(findings), UserWarning, stacklevel=2)


def test_changelog_sections_do_not_repeat_a_category():
    """Keep a Changelog gives each release ONE block per category; two is a silent data loss.

    Found 2026-09-03 (packaging audit 03/F08): ``[Unreleased]`` had accumulated a second
    ``### Added`` and a second ``### Changed``, with the OLDER entries in the lower pair -- so
    renaming the section at release time would have shipped a changelog whose newest and oldest
    entries for the same category were separated by two unrelated blocks, and any parser that
    maps a section to its entries would have kept only one of each pair.
    """
    import re as _re

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    duplicates = []
    section = "(before the first ## heading)"
    seen: set = set()
    for line in changelog.splitlines():
        if _re.match(r"^## (?!#)", line):
            section, seen = line[3:].strip(), set()
        elif _re.match(r"^### (?!#)", line):
            heading = line[4:].strip()
            if heading in seen:
                duplicates.append(f"{section}: '### {heading}' appears more than once")
            seen.add(heading)

    assert not duplicates, "Repeated category headings within one CHANGELOG section:\n  " + "\n  ".join(duplicates)
