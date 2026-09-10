"""pyutilz.dev.persistence_sweep, file-backed half: the sentinel sweep when storage is JSONL rows and JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from pyutilz.dev.persistence_sweep import field_sentinels, file_sentinel_locations, jsonl_records, records_holding, written_files


def _write_store(root: Path) -> None:
    """A store the way autopsia writes one: JSONL rows plus one JSON document per article."""
    rows = root / "bench" / "results.jsonl"
    rows.parent.mkdir(parents=True)
    rows.write_text(
        json.dumps({"article": "a1", "claim": "ZZSENT0000ZZ", "confident": True, "p": 0.25}) + "\n"
        + json.dumps({"article": "a2", "claim": "other", "confident": 1, "p": 0.5}) + "\n",
        encoding="utf-8",
    )
    doc = root / "store" / "a1.json"
    doc.parent.mkdir(parents=True)
    doc.write_text(json.dumps({"triples": [{"note": "ZZSENT0001ZZ", "moderator": {"value": "70-99% é"}}]}), encoding="utf-8")


class TestTheSweepFindsWhatWasWritten:
    def test_every_sentinel_is_located_in_the_file_that_holds_it(self, tmp_path: Path) -> None:
        _write_store(tmp_path)
        sentinels = field_sentinels(["claim", "note"])

        found = file_sentinel_locations(sentinels, written_files([tmp_path]))

        assert [Path(p).name for p in found["claim"]] == ["results.jsonl"]
        assert [Path(p).name for p in found["note"]] == ["a1.json"]

    def test_a_value_nothing_wrote_is_found_nowhere(self, tmp_path: Path) -> None:
        """The negative control: a search that matches everything would pass having checked nothing."""
        _write_store(tmp_path)

        found = file_sentinel_locations({"_control": "ZZCONTROL9999ZZ"}, written_files([tmp_path]))

        assert found == {"_control": []}

    def test_escaped_non_ascii_is_searched_decoded(self, tmp_path: Path) -> None:
        """json.dumps writes é as \\u00e9 by default; a caller's own sentinel with non-ASCII must still be found."""
        _write_store(tmp_path)

        found = file_sentinel_locations({"moderator": "70-99% é"}, written_files([tmp_path]))

        assert [Path(p).name for p in found["moderator"]] == ["a1.json"]

    def test_only_text_like_files_are_scanned(self, tmp_path: Path) -> None:
        _write_store(tmp_path)
        (tmp_path / "store" / "blob.bin").write_bytes(b"ZZSENT0000ZZ")

        assert all(not str(p).endswith(".bin") for p in written_files([tmp_path]))


class TestTypedValuesAreLookedForOnTheRightRecords:
    def test_a_value_is_found_on_the_record_with_the_key(self, tmp_path: Path) -> None:
        _write_store(tmp_path)
        rows = jsonl_records(tmp_path / "bench" / "results.jsonl")

        assert records_holding(rows, 0.25, fields=["p"], key_field="article", key_value="a1") == ["#0.p"]
        assert records_holding(rows, 0.5, fields=["p"], key_field="article", key_value="a1") == []

    def test_true_does_not_match_one(self, tmp_path: Path) -> None:
        """a2 holds confident=1; a boolean probe must not be satisfied by an int, or a lost flag passes."""
        _write_store(tmp_path)
        rows = jsonl_records(tmp_path / "bench" / "results.jsonl")

        assert records_holding(rows, True, fields=["confident"], key_field="article", key_value="a2") == []
        assert records_holding(rows, True, fields=["confident"], key_field="article", key_value="a1") == ["#0.confident"]

    def test_nested_fields_are_reached(self) -> None:
        records = [{"id": "x", "triples": [{"moderator": {"strength": 0.7000000001}}]}]

        assert records_holding(records, 0.7, fields=["strength"], key_field="id", key_value="x", tolerance=1e-6) == ["#0.strength"]

    def test_a_missing_file_is_no_records(self, tmp_path: Path) -> None:
        assert jsonl_records(tmp_path / "absent.jsonl") == []
