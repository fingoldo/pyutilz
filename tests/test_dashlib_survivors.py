"""Locale keys and session-key naming pins for dev.dashlib (mutation survivors, first group)."""

from __future__ import annotations

from pyutilz.dev import dashlib as dl


class TestExportLocaleLabels:
    def test_the_csv_export_key_and_label_are_pinned(self):
        assert dl.AG_GRID_LOCALE_RU["csvExport"] == "Экспорт в CSV (.csv)"

    def test_the_excel_export_key_and_label_are_pinned(self):
        assert dl.AG_GRID_LOCALE_RU["excelExport"] == "Экспорт в Excel (.xlsx)"

    def test_the_xml_export_key_and_label_are_pinned(self):
        assert dl.AG_GRID_LOCALE_RU["excelXmlExport"] == "Экспорт в XML (.xml)"


class TestActiveTabSessionKey:
    def test_the_default_prefix_builds_the_session_key(self):
        assert dl.get_active_tab_var_name("tabsA") == "tabstabsAActiveTab"

    def test_a_custom_prefix_replaces_the_default(self):
        assert dl.get_active_tab_var_name("tabsA", prefix="x") == "xstabsAActiveTab"

    def test_the_key_has_an_s_between_prefix_and_group(self):
        assert dl.get_active_tab_var_name("g", prefix="p") == "psgActiveTab"
