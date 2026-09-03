"""Scanner tests for effect_flag_outside_its_effect, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.effect_flag_outside_its_effect import scan_effect_flag_outside_its_effect

from ._helpers import _write

# ---- effect_flag_outside_its_effect --------------------------------------
#
# A success record set beside, rather than inside, the conditional work it records. An empty crawl
# advertised a parquet file it had never written.


def test_effect_flag_outside_its_effect_flags_the_canonical_case(tmp_path: Path):
    _write(
        tmp_path,
        "out.py",
        """
def write_kinds(table, _kind_ok, path):
    if table.num_rows:
        write_parquet(table, path, "pq")
    _kind_ok["pq"] = True
""",
    )
    findings = scan_effect_flag_outside_its_effect(tmp_path)
    assert len(findings) == 1
    assert "pq" in findings[0].detail


def test_effect_flag_outside_its_effect_accepts_the_record_inside_the_block(tmp_path: Path):
    _write(
        tmp_path,
        "out.py",
        """
def write_kinds(table, _kind_ok, path):
    if table.num_rows:
        write_parquet(table, path, "pq")
        _kind_ok["pq"] = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_understands_an_early_exit(tmp_path: Path):
    """A failure branch ending in `continue` puts the record on the other path by construction.
    That is the shape the audited codebase adopted when it FIXED this defect, and without modelling
    it the rule reports the fix as the bug -- which it did, twice, on real code."""
    _write(
        tmp_path,
        "out.py",
        """
def write_kinds(crawls, _kind_ok):
    for crawl in crawls:
        if not wrote(crawl, "pq"):
            log("skipped")
            continue
        _kind_ok["pq"] = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_requires_a_shared_name(tmp_path: Path):
    """Without the shared token this would flag every assignment that follows an `if`."""
    _write(
        tmp_path,
        "out.py",
        """
def go(table, flags, path):
    if table.num_rows:
        write_parquet(table, path)
    flags["something_else"] = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_sees_an_intervening_guard(tmp_path: Path):
    """The correct form puts the early exit behind its own guard, several statements later.

    Real shape, from a harvest loop: the write failure sets `_write_ok = False`, a later
    `if not _write_ok: ... continue` protects the completion mark, and the mark is therefore on
    the other path. Checking only whether the intervening statement IS a terminator walked
    straight past that `if`, and this rule reported the fixed form for the third time.
    """
    _write(
        tmp_path,
        "harvest.py",
        """
def run(crawls, prog):
    for crawl in crawls:
        write_ok = True
        if crawl.rows:
            try:
                write_outputs(crawl)
            except Exception:
                write_ok = False
        prog.setdefault(crawl, {})
        if not write_ok:
            save_progress(prog)
            continue
        prog[crawl]["completed"] = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_ignores_a_self_only_link(tmp_path: Path):
    """`self` is mentioned by nearly every statement in a method, so it links nothing.

    Found against the repo itself: `self._ready = True` after an unrelated
    `if self._process.stdout:` was reported purely because both mention `self`.
    """
    _write(
        tmp_path,
        "out.py",
        """
class C:
    def start(self):
        if self.stdout:
            self.stream = wrap(self.stdout)
        self.ready = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_ignores_a_logging_only_guard(tmp_path: Path):
    """A guard whose body only logs guards no work, so the statement after it is not a record.

    Found against the repo itself: `if verbose: logger.info(...)` followed by the unconditional
    `res.add(str(obj))` -- reporting that inverts the rule.
    """
    _write(
        tmp_path,
        "out.py",
        """
def go(obj, res, verbose):
    if verbose:
        logger.info("Processing %s of size %s", type(obj), len(str(obj)))
    res.add(str(obj))
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_ignores_a_lazy_init_guard(tmp_path: Path):
    """``if key not in d: d[key] = set()`` creates an empty container; it is not
    the work a following ``d[key].add(...)`` records. The record belongs to the
    NEXT condition, and moving it inside the init guard would record only the
    first item per key."""
    _write(tmp_path, "mod.py", '''
def run(rows, seen_senses):
    for row in rows:
        norm_form, sense_id = row
        if norm_form not in seen_senses:
            seen_senses[norm_form] = set()
        if sense_id in seen_senses[norm_form]:
            continue
        seen_senses[norm_form].add(sense_id)
        process(sense_id)
''')
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_ignores_the_seen_set_idiom(tmp_path: Path):
    """``if sid in seen: report_duplicate(); seen.add(sid)`` -- the set tracks
    everything encountered and the `if` REPORTS a repeat rather than gating the
    record. Moving the record inside deletes the duplicate detection."""
    _write(tmp_path, "mod.py", '''
def run(sids):
    seen = set()
    dups = []
    for sid in sids:
        if sid in seen:
            dups.append(sid)
        seen.add(sid)
    return dups
''')
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_still_flags_a_report_then_record_defect(tmp_path: Path):
    """Non-vacuousness for the exemption above. A guard whose body only appends
    to an error list is ALSO the shape of the real defect -- record success even
    though the branch failed. What separates the two is whether the guard
    interrogates the same container the record writes to. Here it does not."""
    _write(tmp_path, "mod.py", '''
def run(items, errors, processed):
    for item in items:
        if item.is_broken:
            errors.append(item)
        processed.add(item.id)
''')
    assert len(scan_effect_flag_outside_its_effect(tmp_path)) == 1


def test_effect_flag_outside_its_effect_ignores_list_building(tmp_path: Path):
    """`.append` on a list is ordinary accumulation; it gave 44 of this rule's 50 first hits with
    no success record among them."""
    _write(
        tmp_path,
        "out.py",
        """
def go(rows, out):
    for row in rows:
        if row.ok:
            process(row)
        out.append(row)
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []
