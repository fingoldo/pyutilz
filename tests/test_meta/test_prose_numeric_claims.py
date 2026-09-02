"""Counted facts stated in prose are computed from the repo, not typed by hand.

Stale hand-typed counts are the one documentation defect that recurred in every audit wave:
a "24 aliases" claim against a 27-entry map, a "1900+ tests" claim understating the suite by
about 1600, a meta-test file count that went stale inside the session that wrote it. All four
claims registered below are CORRECT today -- registering them is what keeps them correct,
since each now fails the commit that invalidates it instead of the audit six weeks later.

Anchor-loss is a finding in its own right: if the prose around a claim is reworded so the
pattern stops matching, the claim has silently stopped being checked, and the check says so
rather than passing vacuously.

Runtime: four file reads plus one import of pyutilz (already imported by the meta-suite) --
about 0.02 s. No network, no subprocess. The pytest-collection-count claim the audit
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
    r"Aim for >80% coverage for new code",  # a target for contributors
    r"Reduces memory by up to 80% for categorical data",  # a property of the dtype, not a benchmark
    r"maintain >80% coverage",  # the same contributor target
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
