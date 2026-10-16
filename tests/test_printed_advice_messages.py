"""The two refusal messages that tell the caller which keyword to pass must keep naming it.

Each message is the only thing a caller sees before retrying, so a test pins the advised keyword. Both are registered in
``test_printed_advice_coverage.py`` against these tests.
"""

import pickle

import pytest

from pyutilz.core.pythonlib.filesystem import load_file
from pyutilz.core.safe_pickle import PickleVerificationError
from pyutilz.database.db.schema import _alternate_values_clause
from pyutilz.database.db.upsert import build_upsert_query
from pyutilz.text.strings.configfiles import _normalise_encryption


def test_load_file_refusal_names_allow_unverified(tmp_path):
    path = tmp_path / "legacy.pckl"
    path.write_bytes(pickle.dumps({"a": 1}))

    with pytest.raises(PickleVerificationError) as excinfo:
        load_file(str(path))

    assert "allow_unverified=True" in str(excinfo.value)


def test_alternate_values_fragment_refusal_names_raw_sql():
    with pytest.raises(TypeError) as excinfo:
        _alternate_values_clause("a,b", "1, 2", raw_sql=False)

    assert "raw_sql=True" in str(excinfo.value)


def test_upsert_raw_fragment_refusal_names_raw_sql():
    with pytest.raises(TypeError) as excinfo:
        build_upsert_query(["a"], "t", custom_onconflict="on conflict do nothing")

    assert "raw_sql=True" in str(excinfo.value)


def test_xor_encryption_warning_names_base64_spelling(caplog):
    with caplog.at_level("WARNING"):
        _normalise_encryption("xor")

    assert 'use encryption="base64"' in caplog.text
