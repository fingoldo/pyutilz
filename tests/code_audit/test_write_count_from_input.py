"""Scanner tests for write_counted_from_input: the OEWN stage shape, the helper hop, and each exclusion.

Every test calls ``scan_write_counted_from_input`` by name and asserts on what it returned: the polarity gate in
``tests/test_meta/test_scanner_positive_and_negative_cases.py`` reads those assertions out of this source, and a
call hidden behind a local helper is invisible to it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyutilz.dev.code_audit import scan_write_counted_from_input

from ._helpers import _write


def test_the_oewn_stage_shape_is_flagged(tmp_path: Path):
    """08-M15: a batched DO NOTHING insert, then the stage reports len() of what it was given."""
    _write(tmp_path, "importer.py", '''
class Importer:
    async def _import_synsets(self):
        """Import synsets. Inserts are ON CONFLICT DO NOTHING."""
        synsets = [s for s in self._synsets()]
        for i in range(0, len(synsets), self.batch_size):
            batch = synsets[i:i + self.batch_size]
            await self.session.execute(text("""
                INSERT INTO synsets (id, pos) VALUES (:id, :pos)
                ON CONFLICT (id) DO NOTHING
            """), [{"id": s.id, "pos": s.pos} for s in batch])
        self.stats["synsets"] = len(synsets)
''')

    findings = scan_write_counted_from_input(tmp_path)

    assert [(f.check, f.severity, f.line) for f in findings] == [("write_counted_from_input", "P2", 11)]
    assert "len(synsets)" in findings[0].detail


def test_a_conflict_clause_passed_to_a_helper_and_an_augmented_total(tmp_path: Path):
    """The clause as an argument of the same call (``_copy_upsert``), and a ``+=`` running total."""
    _write(tmp_path, "loader.py", '''
async def load(conn, forms):
    total_inserted = 0
    for chunk in chunks(forms, 500):
        await _copy_upsert(conn, "forms", ("entry_id", "written_form"), chunk, "ON CONFLICT (entry_id, written_form) DO NOTHING")
        total_inserted += len(chunk)
    return total_inserted
''')

    findings = scan_write_counted_from_input(tmp_path)

    assert [f.line for f in findings] == [5]


@pytest.mark.parametrize("sql", [
    "INSERT INTO t (id, v) VALUES (:id, :v) ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v WHERE t.v IS DISTINCT FROM EXCLUDED.v",
    "INSERT OR IGNORE INTO t (id, v) VALUES (?, ?)",
    "INSERT IGNORE INTO t (id, v) VALUES (%s, %s)",
])
def test_every_skipping_form_is_in_the_population(tmp_path: Path, sql: str):
    """Each dialect's skip-on-conflict form, with the SQL held in a module constant."""
    _write(tmp_path, "save.py", f'''
SQL = "{sql}"

def save_counted(cur, rows, stats):
    cur.executemany(SQL, rows)
    stats["saved"] = len(rows)
''')

    findings = scan_write_counted_from_input(tmp_path)

    assert [f.line for f in findings] == [5]


def test_the_rows_are_followed_one_hop_by_parameter(tmp_path: Path):
    """The helper writes its first parameter lossily and its second plainly; only the first is an instance."""
    _write(tmp_path, "flush.py", '''
class Importer:
    async def _flush(self, senses, links):
        await self.session.execute(text("INSERT INTO senses VALUES (:id) ON CONFLICT DO NOTHING"), senses)
        await self.session.execute(text("INSERT INTO links VALUES (:a, :b)"), links)

    async def run(self, senses, links):
        await self._flush(senses, links)
        self.imported_senses = len(senses)
        self.imported_links = len(links)
''')

    findings = scan_write_counted_from_input(tmp_path)

    assert [f.line for f in findings] == [8]


def test_what_is_not_in_the_population(tmp_path: Path):
    """DO UPDATE without WHERE, a counter named for the input, rowcount, a non-counter, and the kaikki staging shape.

    Each of these sits one property away from the flagged case, so an over-matching rule reports them."""
    _write(tmp_path, "clean.py", '''
async def upsert(session, batch, stats):
    await session.execute(text("INSERT INTO ws VALUES (:id, :v) ON CONFLICT ON CONSTRAINT uq DO UPDATE SET v = EXCLUDED.v"), batch)
    stats["imported"] = len(batch)

async def read_counted(session, names, stats):
    for name in names:
        await session.execute(text("INSERT INTO lexfiles (name) VALUES (:n) ON CONFLICT (name) DO NOTHING"), {"n": name})
    stats["lexfiles_read"] = len(names)

async def by_rowcount(session, rows, stats):
    result = await session.execute(text("INSERT INTO t VALUES (:id) ON CONFLICT DO NOTHING"), rows)
    stats["inserted"] = result.rowcount
    width = len(rows)
    return width

async def by_helper_return(session, rows, stats):
    await session.execute(text("INSERT INTO t VALUES (:id) ON CONFLICT DO NOTHING"), rows)
    stats["written"] = await _record_written("t", len(rows))

async def difference(session, candidates, existing, stats):
    await session.execute(text("INSERT INTO t VALUES (:id) ON CONFLICT DO NOTHING"), sorted(candidates))
    stats["not_in_oewn"] = len(candidates) - len(existing)

async def staged(conn, session, rows, stats):
    await conn.copy_records_to_table("_staging", records=rows)
    await session.execute(text("INSERT INTO t SELECT * FROM _staging ON CONFLICT DO NOTHING"))
    stats["imported"] = len(rows)
''')

    assert scan_write_counted_from_input(tmp_path) == []


def test_a_docstring_that_mentions_the_clause_is_not_sql(tmp_path: Path):
    """Prose describing the conflict handling does not make the function one that skips rows."""
    _write(tmp_path, "prose.py", '''
def save(cur, rows, stats):
    """Rows arrive deduplicated, so ON CONFLICT DO NOTHING would never fire here."""
    cur.executemany("INSERT INTO t VALUES (%s)", rows)
    stats["saved"] = len(rows)
''')

    assert scan_write_counted_from_input(tmp_path) == []
