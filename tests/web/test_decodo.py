"""Behaviour of the Decodo helpers in ``pyutilz.web.proxy.decodo``: bad numbers coerce to a default instead of raising,
GB and MB are formatted by magnitude, and a traffic response (a bare list or a payload) is totalled per row with missing
fields read as zero. Only the pure helpers are exercised; no request is sent."""

from __future__ import annotations

from pyutilz.web.proxy import decodo


class TestCoercion:
    def test_a_numeric_string_becomes_a_float(self):
        assert decodo._safe_float("2.5") == 2.5

    def test_a_non_numeric_value_becomes_zero(self):
        assert decodo._safe_float("x") == 0.0

    def test_an_integer_string_becomes_an_int(self):
        assert decodo._safe_int("7") == 7

    def test_a_non_integer_value_becomes_the_given_default(self):
        assert decodo._safe_int("x", 3) == 3


class TestFormatting:
    def test_a_gigabyte_quantity_is_shown_in_gb(self):
        assert decodo._fmt_gb(2.5) == "2.50 GB"

    def test_a_small_quantity_is_shown_in_mb(self):
        assert decodo._fmt_gb(0.5) == "512.0 MB"


class TestTrafficReport:
    def test_rows_are_totalled(self):
        report = decodo._parse_traffic_response(
            [{"grouping_key": "a", "requests": 2, "totals": 1.5}, {"grouping_key": "b", "requests": 3, "totals": 0.5}],
            "x",
        )
        assert report.total_requests == 5
        assert report.total_bytes == 2.0
        assert [row.group_key for row in report.rows] == ["a", "b"]

    def test_a_payload_wrapper_is_unwrapped(self):
        report = decodo._parse_traffic_response({"data": [{"grouping_key": "a", "requests": 1, "totals": 1.0}]}, "x")
        assert report.total_requests == 1

    def test_a_missing_field_is_read_as_zero(self):
        report = decodo._parse_traffic_response([{"grouping_key": "a"}], "x")
        assert report.total_requests == 0
        assert report.total_bytes == 0.0

    def test_a_non_list_payload_has_no_rows(self):
        assert decodo._extract_traffic_rows({"data": "nope"}) == []
