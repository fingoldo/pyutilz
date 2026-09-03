"""Scanner tests for docstring_numbers_moved_to_config, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.docstring_numbers_moved_to_config import scan_docstring_numbers_moved_to_config

from ._helpers import _write

# ---- docstring_numbers_moved_to_config (opt-in) --------------------------
#
# Opt-in, so these tests are the only place it is exercised by default. Its precision was measured
# rather than assumed: three hits across four repos, all false, which is why it is not in a ratchet.
# The negative cases below are those three, kept as tests so a future widening of the rule cannot
# quietly reintroduce them.


def test_docstring_numbers_moved_to_config_flags_the_stale_prose(tmp_path: Path):
    _write(
        tmp_path,
        "prune.py",
        """
def _prune_disappearance_counts(state):
    \"\"\"Drop sources that have disappeared.

    Prunes a source after 10 consecutive misses, or 5 for a rare source.
    \"\"\"
    from live_config import cfg
    common = cfg().get("prune", "common_misses", None, int)
    rare = cfg().get("prune", "rare_misses", None, int)
    return {k: v for k, v in state.items() if v < (rare if k in state else common)}
""",
    )
    findings = scan_docstring_numbers_moved_to_config(tmp_path)
    assert len(findings) == 1
    assert "10" in findings[0].detail and "5" in findings[0].detail


def test_docstring_numbers_moved_to_config_ignores_a_named_source(tmp_path: Path):
    """Naming the constant is the RECOMMENDED form; flagging it would punish the fix."""
    _write(
        tmp_path,
        "resolve.py",
        """
def resolve_min_days(cli=None):
    \"\"\"Resolve the rescan floor in days.

    Precedence: CLI > config key > compiled default (``MIN_WH_RESCAN_FREQ_DAYS`` = 14).
    \"\"\"
    from live_config import cfg
    return int(cfg().get("intervals", "min_days", MIN_WH_RESCAN_FREQ_DAYS, int))
""",
    )
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_docstring_numbers_moved_to_config_ignores_a_document_reference(tmp_path: Path):
    """ "audit 04.1" is a citation, not a threshold -- it got through because the same line said
    "after"."""
    _write(
        tmp_path,
        "banners.py",
        """
def _mode_banners(settings):
    \"\"\"Banners built from a fresh settings read.

    The confirmation modal (audit 04.1) closes the window after every tick.
    \"\"\"
    from live_config import cfg
    return cfg().get("submission", "dry_run", None, bool)
""",
    )
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_docstring_numbers_moved_to_config_ignores_numbers_still_in_the_body(tmp_path: Path):
    """If the number is in the code, the prose can be checked against it by reading."""
    _write(
        tmp_path,
        "prune.py",
        """
def prune(state):
    \"\"\"Prunes a source after 10 consecutive misses.\"\"\"
    from live_config import cfg
    limit = cfg().get("prune", "misses", 10, int)
    return {k: v for k, v in state.items() if v < limit}
""",
    )
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_docstring_numbers_moved_to_config_ignores_a_function_reading_no_config(tmp_path: Path):
    _write(
        tmp_path,
        "prune.py",
        """
def describe():
    \"\"\"The batch size limit is 500 per call.\"\"\"
    return LIMIT
""",
    )
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_docstring_numbers_moved_to_config_is_opt_in():
    """It must not reach any project's default run, and therefore any project's baseline."""
    from pyutilz.dev.code_audit import OPT_IN_ONLY, get_scanners

    assert "docstring_numbers_moved_to_config" in get_scanners()
    assert "docstring_numbers_moved_to_config" in OPT_IN_ONLY
