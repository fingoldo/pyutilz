"""``# audit-exempt`` markers on the scanners that gained marker support after the first pass.

Each scanner gets the same four properties, one parametrized case per check id:

* the bare construct is reported (the marker is what suppresses, not a silent scanner);
* the same construct with a valid marker on the flagged line is not reported;
* a marker with an empty reason still reports (a reason is mandatory);
* a marker naming another check id still reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from pyutilz.dev.code_audit import (
    scan_bare_except,
    scan_comment_names_missing_symbol,
    scan_retry_loops,
    scan_sentinel_guard_mismatch,
    scan_credential_shaped_log_args,
    scan_default_via_or_trap,
    scan_duplicate_function_body,
    scan_import_cycles,
    scan_near_duplicate_function_body,
    scan_non_neutral_except_fallback,
    scan_resource_handle_safety,
    scan_sql_limit_without_order_by,
    scan_unpicklable_resource_state,
    scan_unthrottled_hot_loop_log,
)

from ._helpers import _write

_OTHER = "wall_clock_assertion"


def _valid(check: str) -> str:
    return f"  # audit-exempt: {check} -- reviewed, intentionally kept"


def _no_reason(check: str) -> str:
    return f"  # audit-exempt: {check} --"


def _other_check(check: str) -> str:
    return f"  # audit-exempt: {_OTHER} -- wrong check id"


# (scanner, check id, source with ``{m}`` at the end of the flagged line)
_SINGLE_FILE_CASES = [
    pytest.param(
        scan_default_via_or_trap,
        "default_via_or",
        "def f(x):\n    y = x or 5{m}\n    return y\n",
        id="default_via_or",
    ),
    pytest.param(
        scan_unthrottled_hot_loop_log,
        "unthrottled_hot_loop_log",
        "import logging\nlog = logging.getLogger(__name__)\n\n\ndef f(xs):\n    for x in xs:\n        log.warning('bad %s', x){m}\n",
        id="unthrottled_hot_loop_log",
    ),
    pytest.param(
        scan_non_neutral_except_fallback,
        "non_neutral_except_fallback",
        "def f(path):\n    try:\n        return float(path)\n    except OSError:{m}\n        return 0.0\n",
        id="non_neutral_except_fallback",
    ),
    pytest.param(
        scan_credential_shaped_log_args,
        "credential_shaped_log_arg",
        "import logging\nlog = logging.getLogger(__name__)\n\n\ndef f(password):\n    log.info('got %s', password){m}\n",
        id="credential_shaped_log_arg",
    ),
    pytest.param(
        scan_resource_handle_safety,
        "resource_handle_safety",
        "def f(path):\n    fh = open(path){m}\n    return fh.read()\n",
        id="resource_handle_safety",
    ),
    pytest.param(
        scan_comment_names_missing_symbol,
        "comment_names_missing_symbol",
        "def present():\n    return 1\n\n\n# Called from `_vanished_helper()` on every batch boundary.{m}\nX = 1\n",
        id="comment_names_missing_symbol",
    ),
    pytest.param(
        scan_sql_limit_without_order_by,
        "sql_limit_without_order_by",
        "def f(conn):\n    return conn.execute('SELECT id FROM items LIMIT 10'){m}\n",
        id="sql_limit_without_order_by",
    ),
    pytest.param(
        scan_unpicklable_resource_state,
        "unpicklable_resource_state",
        "import threading\n\n\nclass Holder:\n    def __init__(self):\n        self.lock = threading.Lock(){m}\n",
        id="unpicklable_resource_state",
    ),
    pytest.param(
        scan_bare_except,
        "bare_except",
        "def f(path):\n    try:\n        return float(path)\n    except:{m}\n        return 0.0\n",
        id="bare_except",
    ),
    pytest.param(
        scan_retry_loops,
        "busy_retry_loop",
        "def f(q):\n    while True:{m}\n        try:\n            return q.get_nowait()\n        except Exception:\n            continue\n",
        id="busy_retry_loop",
    ),
]


def _run_single(scanner: Callable, tmp_path: Path, source: str, marker: str, check: str) -> list[str]:
    _write(tmp_path, "mod.py", source.replace("{m}", marker))
    return [f.check for f in scanner(tmp_path) if f.check == check]


@pytest.mark.parametrize("scanner, check, source", _SINGLE_FILE_CASES)
def test_bare_construct_is_reported(scanner, check, source, tmp_path: Path):
    assert _run_single(scanner, tmp_path, source, "", check) == [check]


@pytest.mark.parametrize("scanner, check, source", _SINGLE_FILE_CASES)
def test_valid_marker_on_flagged_line_suppresses(scanner, check, source, tmp_path: Path):
    assert _run_single(scanner, tmp_path, source, _valid(check), check) == []


@pytest.mark.parametrize("scanner, check, source", _SINGLE_FILE_CASES)
def test_marker_without_reason_still_reports(scanner, check, source, tmp_path: Path):
    assert _run_single(scanner, tmp_path, source, _no_reason(check), check) == [check]


@pytest.mark.parametrize("scanner, check, source", _SINGLE_FILE_CASES)
def test_marker_for_another_check_still_reports(scanner, check, source, tmp_path: Path):
    assert _run_single(scanner, tmp_path, source, _other_check(check), check) == [check]


# ---- multi-file scanners ----------------------------------------------------------------------------


_DUPLICATE_BODY = "def compute(a, b):{m}\n" "    total = a + b\n" "    scaled = total * 2\n" "    shifted = scaled - 1\n" "    return shifted\n"


def _run_duplicate(tmp_path: Path, marker: str) -> list[str]:
    _write(tmp_path, "first.py", _DUPLICATE_BODY.replace("{m}", ""))
    _write(tmp_path, "second.py", _DUPLICATE_BODY.replace("{m}", marker))
    return [f.check for f in scan_duplicate_function_body(tmp_path) if f.check == "duplicate_function_body"]


def test_duplicate_body_is_reported_without_marker(tmp_path: Path):
    assert _run_duplicate(tmp_path, "") == ["duplicate_function_body"]


def test_duplicate_body_valid_marker_suppresses(tmp_path: Path):
    assert _run_duplicate(tmp_path, _valid("duplicate_function_body")) == []


def test_duplicate_body_marker_without_reason_still_reports(tmp_path: Path):
    assert _run_duplicate(tmp_path, _no_reason("duplicate_function_body")) == ["duplicate_function_body"]


def test_duplicate_body_marker_for_another_check_still_reports(tmp_path: Path):
    assert _run_duplicate(tmp_path, _other_check("duplicate_function_body")) == ["duplicate_function_body"]


_NEAR_A = (
    "def score(values, weight):\n"
    "    total = 0.0\n"
    "    for v in values:\n"
    "        total += v * weight\n"
    "        total -= 0.5\n"
    "    scaled = total / max(len(values), 1)\n"
    "    return round(scaled, 3)\n"
)
_NEAR_B = _NEAR_A.replace("weight", "factor").replace("total -= 0.5", "total -= 0.25").replace("round(scaled, 3)", "round(scaled, 4)")


def _run_near(tmp_path: Path, marker: str) -> list[str]:
    _write(tmp_path, "first.py", _NEAR_A)
    _write(tmp_path, "second.py", _NEAR_B.replace("def score(values, factor):", "def score(values, factor):" + marker))
    return [f.check for f in scan_near_duplicate_function_body(tmp_path) if f.check == "near_duplicate_function_body"]


def test_near_duplicate_body_is_reported_without_marker(tmp_path: Path):
    assert _run_near(tmp_path, "") == ["near_duplicate_function_body"]


def test_near_duplicate_body_valid_marker_suppresses(tmp_path: Path):
    assert _run_near(tmp_path, _valid("near_duplicate_function_body")) == []


def test_near_duplicate_body_marker_without_reason_still_reports(tmp_path: Path):
    assert _run_near(tmp_path, _no_reason("near_duplicate_function_body")) == ["near_duplicate_function_body"]


def test_near_duplicate_body_marker_for_another_check_still_reports(tmp_path: Path):
    assert _run_near(tmp_path, _other_check("near_duplicate_function_body")) == ["near_duplicate_function_body"]


def _make_cycle(tmp_path: Path, marker: str) -> Path:
    pkg = tmp_path / "cyclepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "alpha.py").write_text(f"from cyclepkg import beta{marker}\n", encoding="utf-8")
    (pkg / "beta.py").write_text(f"from cyclepkg import alpha{marker}\n", encoding="utf-8")
    return pkg


def _run_cycle(tmp_path: Path, marker: str) -> list[str]:
    pkg = _make_cycle(tmp_path, marker)
    return [f.check for f in scan_import_cycles(pkg, package_name="cyclepkg") if f.check == "import_cycle"]


def test_import_cycle_is_reported_without_marker(tmp_path: Path):
    assert _run_cycle(tmp_path, "") == ["import_cycle"]


def test_import_cycle_valid_marker_on_finding_line_suppresses(tmp_path: Path):
    # The finding sits on line 1 of the cycle's first module, so the marker goes on that line.
    pkg = tmp_path / "cyclepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "alpha.py").write_text(f"from cyclepkg import beta{_valid('import_cycle')}\n", encoding="utf-8")
    (pkg / "beta.py").write_text("from cyclepkg import alpha\n", encoding="utf-8")
    assert [f.check for f in scan_import_cycles(pkg, package_name="cyclepkg")] == []


def test_import_cycle_marker_without_reason_still_reports(tmp_path: Path):
    assert _run_cycle(tmp_path, _no_reason("import_cycle")) == ["import_cycle"]


def test_import_cycle_marker_for_another_check_still_reports(tmp_path: Path):
    assert _run_cycle(tmp_path, _other_check("import_cycle")) == ["import_cycle"]


_SENTINEL_RETURNER = """
def fetch_num_pages(url):
    try:
        return int(get(url).text)
    except ValueError:
        return 0
"""

_SENTINEL_DRIVER = """
def run(state):
    pages_total = fetch_num_pages(state.url)
    if pages_total is None or (pages_total > 0 and state.done):{m}
        return
    mark_source_empty(state)
"""


def _run_sentinel(tmp_path: Path, marker: str) -> list[str]:
    _write(tmp_path, "cdx.py", _SENTINEL_RETURNER)
    _write(tmp_path, "driver.py", _SENTINEL_DRIVER.replace("{m}", marker))
    return [f.check for f in scan_sentinel_guard_mismatch(tmp_path) if f.check == "sentinel_guard_mismatch"]


def test_sentinel_guard_bare_construct_is_reported(tmp_path: Path):
    assert _run_sentinel(tmp_path, "") == ["sentinel_guard_mismatch"]


def test_sentinel_guard_valid_marker_suppresses(tmp_path: Path):
    assert _run_sentinel(tmp_path, _valid("sentinel_guard_mismatch")) == []


def test_sentinel_guard_marker_without_reason_still_reports(tmp_path: Path):
    assert _run_sentinel(tmp_path, _no_reason("sentinel_guard_mismatch")) == ["sentinel_guard_mismatch"]


def test_sentinel_guard_marker_for_another_check_still_reports(tmp_path: Path):
    assert _run_sentinel(tmp_path, _other_check("sentinel_guard_mismatch")) == ["sentinel_guard_mismatch"]
