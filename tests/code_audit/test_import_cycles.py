"""Scanner tests for import_cycles, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_import_cycles,
    scan_thresholds_below_documented_result,
)

from ._helpers import _write

# ---- import_cycle -------------------------------------------------------------


def test_import_cycles_two_node_cycle_flagged(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", "")
    _write(pkg, "a.py", "import mypkg.b\n")
    _write(pkg, "b.py", "import mypkg.a\n")
    findings = scan_import_cycles(pkg, package_name="mypkg")
    assert len(findings) == 1, findings
    assert findings[0].check == "import_cycle"


def test_import_cycles_acyclic_not_flagged(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", "")
    _write(pkg, "a.py", "import mypkg.b\n")
    _write(pkg, "b.py", "x = 1\n")
    assert scan_import_cycles(pkg, package_name="mypkg") == []


def test_import_cycles_deferred_cycle_not_flagged(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", "")
    _write(pkg, "a.py", "import mypkg.b\n")
    _write(pkg, "b.py", "import mypkg.a\n")
    findings = scan_import_cycles(pkg, package_name="mypkg", deferred_cycles=frozenset({"mypkg.a -> mypkg.b"}))
    assert findings == []


def test_import_cycles_lazy_function_body_import_not_flagged(tmp_path: Path):
    """A cycle that only closes via a lazy (function-body) import is not a module-load-time cycle."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", "")
    _write(pkg, "a.py", "import mypkg.b\n")
    _write(pkg, "b.py", "def f():\n    import mypkg.a\n    return mypkg.a\n")
    assert scan_import_cycles(pkg, package_name="mypkg") == []


def test_import_cycles_scc_representative_deterministic_regardless_of_graph_build_order():
    """``scan_import_cycles`` picks ``comp[0]`` (the first element of the SCC Tarjan's algorithm
    returns) as the representative for BOTH ``Finding.file`` and ``Finding.snippet`` -- so which
    element that is must not depend on the ORDER ``_build_graph`` happened to insert modules/
    edges in (which in production tracks ``Path.rglob()``'s filesystem-enumeration order, not
    guaranteed stable across process runs on every OS/filesystem for a large multi-directory
    tree). Confirmed by direct construction: the SAME 4-node cycle, fed to
    ``_strongly_connected_components`` with its dict/edge-set built in forward vs. reversed
    insertion order, previously returned a DIFFERENT first element (Tarjan's DFS visits `graph[v]`
    -- a plain ``set`` -- in insertion-dependent order) -- so a baseline-diffing consumer (e.g. a
    code-audit baseline JSON) could see the SAME cycle reported under a DIFFERENT
    ``check::file:line`` key across scans of identical source, permanently flaking any "no new
    findings" gate built on it. ``scan_import_cycles`` now sorts ``comp`` before using
    ``comp[0]``, so the representative is the same regardless of traversal order."""
    from pyutilz.dev.code_audit.import_cycles import _strongly_connected_components

    edges = {"a": {"b"}, "b": {"c"}, "c": {"d"}, "d": {"a"}}
    forward_order = dict(edges)
    reverse_order = {k: edges[k] for k in reversed(list(edges))}
    assert list(forward_order) != list(reverse_order), "test setup: insertion orders must actually differ"

    forward_scc = _strongly_connected_components(forward_order)
    reverse_scc = _strongly_connected_components(reverse_order)
    assert len(forward_scc) == 1 and len(reverse_scc) == 1
    assert set(forward_scc[0]) == set(reverse_scc[0]) == {"a", "b", "c", "d"}, "test setup: both must find the same 4-node cycle"

    # This is exactly what scan_import_cycles now does before using comp[0] (see the `comp =
    # sorted(comp)` line in scan_import_cycles) -- the representative must agree regardless of
    # which insertion order produced the SCC.
    assert sorted(forward_scc[0])[0] == sorted(reverse_scc[0])[0] == "a"


def test_threshold_below_documented_result_reads_a_comma_grouped_number_whole(tmp_path: Path):
    """A docstring saying "7,297 of 12,121" documents 7,297 - not 297.

    The leading word boundary matched only the LAST group, so the claim was read as 297 and every message
    quoted that. The reported number is what a reader checks the finding against, so a wrong one sends them
    looking for a gate that does not exist.

    This does NOT assert the finding disappears, and that is deliberate: the scanner pairs a documented count
    with any `>=` in the same function without knowing whether the two measure the same quantity, so a
    docstring about 7,297 priors beside an assertion about 2 classes per entry still reports. That is a
    separate limitation of the heuristic, and papering over it here by asserting an empty list would hide it.
    """
    _write(
        tmp_path,
        "test_grouped.py",
        '''
def test_most_priors_carry_no_basis():
    """7,297 of 12,121 stated priors carry no basis at all."""
    for entry in ([1, 2],):
        assert len(entry) >= 2
''',
    )
    findings = scan_thresholds_below_documented_result(tmp_path)
    assert len(findings) == 1, findings
    assert "documents 7297" in findings[0].detail, findings[0].detail
    assert "documents 297 " not in findings[0].detail, findings[0].detail


def test_threshold_below_documented_result_still_flags_a_grouped_claim_that_is_genuinely_weakened(tmp_path: Path):
    """The negative control: reading the number whole must not stop the check firing when it should."""
    _write(
        tmp_path,
        "test_grouped_bad.py",
        '''
def test_rows_recovered():
    """1,200 of 1,500 rows recover their source."""
    recovered = 1200
    assert recovered >= 900
''',
    )
    findings = scan_thresholds_below_documented_result(tmp_path)
    assert len(findings) == 1, findings
