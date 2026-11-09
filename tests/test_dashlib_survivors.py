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


class _Log:
    def __init__(self):
        self.debug_calls: list = []

    def debug(self, *args):
        self.debug_calls.append(args)


class TestUnpackTabExtras:
    def test_no_extras_gives_three_empty_slots(self):
        assert dl._unpack_tab_extras([]) == (None, None, None)

    def test_one_extra_fills_only_the_tab_class(self):
        assert dl._unpack_tab_extras(["a"]) == ("a", None, None)

    def test_two_extras_fill_the_tab_and_label_classes(self):
        assert dl._unpack_tab_extras(["a", "b"]) == ("a", "b", None)

    def test_three_extras_fill_all_slots(self):
        assert dl._unpack_tab_extras(["a", "b", "c"]) == ("a", "b", "c")

    def test_extra_entries_beyond_three_are_ignored(self):
        assert dl._unpack_tab_extras(["a", "b", "c", "d"]) == ("a", "b", "c")


class TestResolveCurrentUser:
    def _install_login(self, monkeypatch, authenticated, user_obj=None):
        import types

        class _Current:
            is_authenticated = authenticated

        module = types.ModuleType("flask_login")
        module.current_user = user_obj if user_obj is not None else _Current()
        monkeypatch.setitem(__import__("sys").modules, "flask_login", module)
        return module

    def test_an_unauthenticated_user_is_refused_and_logged(self, monkeypatch):
        self._install_login(monkeypatch, False)
        log = _Log()
        monkeypatch.setattr(dl, "logger", log)
        assert dl._resolve_current_user("tabsA") == (False, None)
        assert log.debug_calls == [("create_tabs(%s): current user is not authenticated, nothing to render", "tabsA")]

    def test_an_authenticated_user_is_returned(self, monkeypatch):
        self._install_login(monkeypatch, True)
        ok, user = dl._resolve_current_user("tabsA")
        assert ok is True
        assert user is not None

    def test_a_missing_login_package_proceeds_without_a_user_and_logs_the_skip(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "flask_login", None)
        log = _Log()
        monkeypatch.setattr(dl, "logger", log)
        assert dl._resolve_current_user("tabsA") == (True, None)
        assert len(log.debug_calls) == 1
        assert log.debug_calls[0][0] == "create_tabs: flask_login current_user check skipped (%s)"


class _User:
    def __init__(self, role):
        self.role = role


class TestBuildTabEntries:
    def _build(self, tabs, user=None, active="pre-b", active_label="act", sep=","):
        return dl._build_tab_entries(tabs, user, "content", "pre-", sep, active_label, active)

    def test_a_missing_id_takes_the_label_and_a_missing_label_takes_the_id(self):
        tabs, _ = self._build([("Lbl", None, None), (None, "bid", None)])
        assert [t.tab_id for t in tabs] == ["pre-Lbl", "pre-bid"]
        assert [t.label for t in tabs] == ["Lbl", "bid"]

    def test_an_unrestricted_tab_is_shown_to_everyone(self):
        tabs, _ = self._build([("A", "a", None)])
        assert len(tabs) == 1

    def test_a_role_list_string_is_split_on_the_separator(self):
        tabs, _ = self._build([("A", "a", "admin;user")], user=_User("user"), sep=";")
        assert len(tabs) == 1

    def test_a_role_list_hides_the_tab_from_other_roles(self):
        tabs, _ = self._build([("A", "a", ["admin"])], user=_User("user"))
        assert tabs == []

    def test_a_restricted_tab_is_hidden_from_an_anonymous_user(self):
        tabs, _ = self._build([("A", "a", ["admin"])], user=None)
        assert tabs == []

    def test_the_active_tab_gets_the_active_label_class_and_others_do_not(self):
        tabs, _ = self._build([("A", "a", None, "tc", "base", None), ("B", "b", None, "tc", "base", None)], active="pre-b", active_label="act")
        by_id = {t.tab_id: t.labelClassName for t in tabs}
        assert by_id == {"pre-a": "base", "pre-b": "base act"}

    def test_a_missing_label_class_leaves_no_stray_text(self):
        tabs, _ = self._build([("A", "a", None)], active="none")
        assert tabs[0].labelClassName == ""

    def test_a_tooltip_is_created_for_each_tab_that_has_one(self):
        _, tooltips = self._build([("A", "a", None, "base", None, "tip")])
        assert len(tooltips) == 1
        assert tooltips[0].target == "pre-a"

    def test_no_tooltip_without_text(self):
        _, tooltips = self._build([("A", "a", None)])
        assert tooltips == []

    def test_each_visible_tab_is_logged_with_its_label_id_and_users(self, monkeypatch):
        log = _Log()
        monkeypatch.setattr(dl, "logger", log)
        self._build([("A", "a", ["admin"])], user=_User("admin"))
        assert log.debug_calls == [("tabLabel=%s, tabId=%s, tabUsers=%s", "A", "a", ["admin"])]
