"""Scanner tests for docstring_names_a_caller_that_does_not_call, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.docstring_names_a_caller_that_does_not_call import scan_docstring_names_a_caller_that_does_not_call

from ._helpers import _write

# ---- docstring_names_a_caller_that_does_not_call --------------------------


def test_docstring_names_a_caller_flags_a_rotted_pointer(tmp_path: Path):
    """The docstring names a caller; that function contains no call to it."""
    _write(
        tmp_path,
        "batching.py",
        '''
def _flush_rows(rows):
    """Write the batch."""
    return write(rows)


def stamp_deadline(batch):
    """Called from `_flush_rows()` on every batch boundary."""
    return batch


def run(rows, batch):
    return _flush_rows(rows), stamp_deadline(batch)
''',
    )
    findings = scan_docstring_names_a_caller_that_does_not_call(tmp_path)
    assert len(findings) == 1, findings
    assert "stamp_deadline" in findings[0].detail and "_flush_rows" in findings[0].detail


def test_docstring_names_a_caller_accepts_a_true_pointer(tmp_path: Path):
    """The correct form, which must stay silent."""
    _write(
        tmp_path,
        "batching.py",
        '''
def _flush_rows(rows, batch):
    """Write the batch."""
    return write(rows), stamp_deadline(batch)


def stamp_deadline(batch):
    """Called from `_flush_rows()` on every batch boundary."""
    return batch


def run(rows, batch):
    return _flush_rows(rows, batch)
''',
    )
    assert scan_docstring_names_a_caller_that_does_not_call(tmp_path) == []


def test_docstring_names_a_caller_ignores_used_by(tmp_path: Path):
    """"used by X()" describes data far more often than it describes the call graph.

    `connect` registering "the client used by execute()" says nothing about who calls `connect`,
    and that sentence was one of this rule's first two hits.
    """
    _write(
        tmp_path,
        "graphql.py",
        '''
def connect(client):
    """Register the module-level client used by execute()."""
    return client


def execute(query):
    """Run a query."""
    return query


def run(client, query):
    return connect(client), execute(query)
''',
    )
    assert scan_docstring_names_a_caller_that_does_not_call(tmp_path) == []


def test_docstring_names_a_caller_needs_the_name_written_as_a_call(tmp_path: Path):
    """"Called by the scheduler" names no function, and prose is not a call-graph claim.

    Replaces a test that used `caller: str` in an Args block: once the `callers?:` form was
    dropped from the pattern, nothing could match that text either way, so the test passed with
    the call-form requirement removed and proved nothing.
    """
    _write(
        tmp_path,
        "batching.py",
        '''
def _flush_rows(rows):
    """Write the batch."""
    return write(rows)


def stamp_deadline(batch):
    """Called by _flush_rows in the nightly pass, once the rows are on disk."""
    return batch


def run(rows, batch):
    return _flush_rows(rows), stamp_deadline(batch)
''',
    )
    assert scan_docstring_names_a_caller_that_does_not_call(tmp_path) == []


def test_docstring_names_a_caller_ignores_a_name_from_outside_the_tree(tmp_path: Path):
    """A name that resolves nowhere is comment_names_missing_symbol's finding, not this one."""
    _write(
        tmp_path,
        "batching.py",
        '''
def stamp_deadline(batch):
    """Called from `celery_beat()` in the scheduler service."""
    return batch


def run(batch):
    return stamp_deadline(batch)
''',
    )
    assert scan_docstring_names_a_caller_that_does_not_call(tmp_path) == []


def test_docstring_names_a_caller_ignores_a_dead_function(tmp_path: Path):
    """If nothing calls the documented function, the interesting finding is that it is dead."""
    _write(
        tmp_path,
        "batching.py",
        '''
def _flush_rows(rows):
    """Write the batch."""
    return write(rows)


def stamp_deadline(batch):
    """Called from `_flush_rows()` on every batch boundary."""
    return batch


def run(rows):
    return _flush_rows(rows)
''',
    )
    assert scan_docstring_names_a_caller_that_does_not_call(tmp_path) == []
