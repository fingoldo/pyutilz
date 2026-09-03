"""Scanner tests for unit_suffix_mismatch, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.unit_suffix_mismatch import scan_unit_suffix_mismatch

from ._helpers import _write

# ---- unit_suffix_mismatch ------------------------------------------------
#
# A quantity stored under one unit and read from another. A `duration_s` column measured cycle
# wall-clock while the real work time sat one JSONB level away as `extra.minutes`.


def test_unit_suffix_mismatch_flags_a_bare_cross_unit_read(tmp_path: Path):
    _write(
        tmp_path,
        "obs.py",
        """
def record(totals):
    work_s = totals["minutes"]
    return work_s
""",
    )
    findings = scan_unit_suffix_mismatch(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P2"


def test_unit_suffix_mismatch_is_silent_on_a_conversion(tmp_path: Path):
    """`work_s = totals["minutes"] * 60` is the CORRECT form. Any arithmetic counts as a
    conversion -- assuming otherwise would flag every correct conversion in a tree."""
    _write(
        tmp_path,
        "obs.py",
        """
def record(totals):
    work_s = totals["minutes"] * 60
    return work_s
""",
    )
    assert scan_unit_suffix_mismatch(tmp_path) == []


def test_unit_suffix_mismatch_treats_synonyms_as_one_unit(tmp_path: Path):
    _write(
        tmp_path,
        "obs.py",
        """
def record(totals):
    elapsed_secs = totals["seconds"]
    return elapsed_secs
""",
    )
    assert scan_unit_suffix_mismatch(tmp_path) == []


def test_unit_suffix_mismatch_covers_keyword_arguments(tmp_path: Path):
    """The audited case passed the wrong unit as a keyword to the recorder, not via assignment."""
    _write(
        tmp_path,
        "obs.py",
        """
def record(extra):
    record_run(duration_s=extra["minutes"])
""",
    )
    assert len(scan_unit_suffix_mismatch(tmp_path)) == 1


def test_unit_suffix_mismatch_ranks_a_cross_family_pair_lower(tmp_path: Path):
    """Seconds against bytes is more likely a naming coincidence than a real conversion bug."""
    _write(
        tmp_path,
        "obs.py",
        """
def record(totals):
    payload_bytes = totals["seconds"]
    return payload_bytes
""",
    )
    findings = scan_unit_suffix_mismatch(tmp_path)
    assert len(findings) == 1 and findings[0].severity == "Low"


def test_unit_suffix_mismatch_reads_the_pre_3_9_subscript_shape():
    """Up to python 3.8 the parser wrapped `d["key"]` in an `ast.Index` node, which 3.9 removed
    (bpo-34822). Reading `.slice` without unwrapping matched nothing on 3.8, so the whole rule went
    quiet there rather than erroring. Simulated with a legacy-shaped node, since the interpreter
    running this test builds the modern shape."""
    import ast as _ast

    from pyutilz.dev.code_audit._base import _subscript_index
    from pyutilz.dev.code_audit.unit_suffix_mismatch import _source_name, _target_names

    class Index(_ast.AST):  # the node class python<=3.8 actually produced, named exactly as it was
        _fields = ("value",)

    # Built node-by-node rather than parsed: python 3.8 ALREADY produces this shape, so wrapping a
    # parsed subscript there would nest Index inside Index and prove nothing.
    const = _ast.Constant(value="minutes")
    legacy = _ast.Subscript(value=_ast.Name(id="totals", ctx=_ast.Load()), slice=Index(value=const), ctx=_ast.Load())

    assert isinstance(_subscript_index(legacy), _ast.Constant)
    assert _source_name(legacy) == "minutes"
    assert _target_names(legacy) == ["minutes"]


# ---- F91/F92/F183/F184: unit suffix mismatch ----------------------------------------


def test_unit_suffix_mismatch_covers_an_annotated_assignment(tmp_path: Path):
    _write(tmp_path, "obs.py", """
def record(totals):
    work_s: float = totals["minutes"]
    return work_s
""")
    assert len(scan_unit_suffix_mismatch(tmp_path)) == 1


def test_unit_suffix_mismatch_covers_an_augmented_assignment(tmp_path: Path):
    _write(tmp_path, "obs.py", """
def record(totals, work_s):
    work_s += totals["minutes"]
    return work_s
""")
    assert len(scan_unit_suffix_mismatch(tmp_path)) == 1


def test_unit_suffix_mismatch_does_not_read_min_as_minutes(tmp_path: Path):
    _write(tmp_path, "obs.py", """
def record(bucket_hours):
    window_min = bucket_hours
    return window_min
""")
    assert scan_unit_suffix_mismatch(tmp_path) == []


def test_unit_suffix_mismatch_ignores_a_bare_one_letter_keyword(tmp_path: Path):
    _write(tmp_path, "plot.py", """
def draw(ax, x, y, sizes_pct):
    ax.scatter(x, y, s=sizes_pct)
""")
    assert scan_unit_suffix_mismatch(tmp_path) == []


def test_unit_suffix_mismatch_reports_a_keyword_at_its_own_line(tmp_path: Path):
    _write(tmp_path, "obs.py", """
def record(cfg):
    schedule(
        1,
        timeout_seconds=cfg["ms"],
    )
""")
    findings = scan_unit_suffix_mismatch(tmp_path)
    assert len(findings) == 1 and findings[0].line == 4
