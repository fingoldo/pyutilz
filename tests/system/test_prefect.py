"""Behaviour of the Prefect filters in ``pyutilz.system.scheduling.prefect``: a flow passes when it matches the wanted flow
id and is not the excluded one, and a run's labels pass when they meet any-of and all-of filters, each filter applying only
when it is non-empty. The Prefect client is not contacted; only the filter predicates are checked."""

from __future__ import annotations

from pyutilz.system.scheduling import prefect


class TestFlowFilter:
    def test_no_filter_selects_every_flow(self):
        assert prefect._flow_selected("1", None, None) is True

    def test_a_wanted_id_selects_only_that_flow(self):
        assert prefect._flow_selected("1", "1", None) is True
        assert prefect._flow_selected("2", "1", None) is False

    def test_an_excluded_id_is_dropped(self):
        assert prefect._flow_selected("1", None, "1") is False
        assert prefect._flow_selected("2", None, "1") is True


class TestLabelFilter:
    def test_no_label_filter_matches_everything(self):
        assert prefect._labels_match({"a"}, set(), set()) is True

    def test_any_of_needs_one_shared_label(self):
        assert prefect._labels_match({"a", "b"}, {"a"}, set()) is True
        assert prefect._labels_match({"b"}, {"a"}, set()) is False

    def test_all_of_needs_every_label(self):
        assert prefect._labels_match({"a"}, set(), {"a", "b"}) is False
        assert prefect._labels_match({"a", "b"}, set(), {"a", "b"}) is True
