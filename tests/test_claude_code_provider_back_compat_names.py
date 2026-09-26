"""The names claude_code_cli took over from claude_code_provider stay importable from the old module.

Downstream code (glossum's re-export shims and tests) imports the rate-limit regexes and helpers from
pyutilz.llm.claude_code_provider; when 1.1 moved them to claude_code_cli without re-exporting the regexes,
those imports raised ImportError at collection time.
"""

import importlib

import pytest

MOVED_NAMES = (
    "_RATE_LIMIT_PATTERN",
    "_RESET_TIME_PATTERN",
    "_TIMEZONE_PATTERN",
    "_parse_reset_wait_seconds",
    "_is_rate_limit_error",
    "_find_claude_executable",
)


@pytest.mark.parametrize("name", MOVED_NAMES)
def test_moved_name_is_the_same_object_at_the_old_path(name):
    old = importlib.import_module("pyutilz.llm.claude_code_provider")
    new = importlib.import_module("pyutilz.llm.claude_code_cli")
    assert getattr(old, name) is getattr(new, name)


def test_rate_limit_pattern_still_matches_the_cli_message():
    from pyutilz.llm.claude_code_provider import _RATE_LIMIT_PATTERN

    assert _RATE_LIMIT_PATTERN.search("You've hit your limit · resets 4am")
    assert not _RATE_LIMIT_PATTERN.search("connection reset by peer")
