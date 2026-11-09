"""Locale keys and session-key naming pins for dev.dashlib (mutation survivors, first group)."""

from __future__ import annotations

import pytest

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


class TestAssembleTabs:
    def _assemble(self, **kwargs):
        calls: list = []

        def draw(active):
            calls.append(active)
            return "BODY"

        params = dict(
            tabsName="g",
            tabs=["T1"],
            tooltips=["TIP"],
            draw_tab_content_function=draw,
            active_tab="pre-a",
            prefix="pre-",
            tabsClassName="tc",
            contentClassName="cc",
            use_cardstyle=False,
            show_loading=False,
        )
        params.update(kwargs)
        return dl._assemble_tabs(**params), calls

    def test_the_header_and_content_carry_the_prefixed_group_ids(self):
        container, _ = self._assemble()
        header, content = container.children[0], container.children[1]
        assert header.id == "pre-sg"
        assert content.id == "pre-sgContent"

    def test_the_content_function_is_called_with_the_active_tab(self):
        _, calls = self._assemble(active_tab="pre-z")
        assert calls == ["pre-z"]

    def test_the_header_is_marked_with_the_active_tab_and_class(self):
        container, _ = self._assemble()
        header = container.children[0]
        assert header.active_tab == "pre-a"
        assert header.className == "tc"

    def test_tooltips_follow_the_header_and_body(self):
        container, _ = self._assemble(tooltips=["TIP1", "TIP2"])
        assert container.children[-2:] == ["TIP1", "TIP2"]

    def test_the_container_is_fluid_with_the_content_class(self):
        container, _ = self._assemble(contentClassName="wrap")
        assert container.fluid is True
        assert container.className == "wrap"

    def test_a_loading_overlay_wraps_the_body_with_its_delay_and_style(self):
        from dash import dcc

        container, _ = self._assemble(show_loading=True)
        loading = container.children[1]
        assert isinstance(loading, dcc.Loading)
        assert loading.delay_show == 250
        assert loading.overlay_style == {"visibility": "visible", "opacity": 0.45}

    def test_without_loading_the_body_is_the_bare_div(self):
        from dash import html

        container, _ = self._assemble(show_loading=False)
        assert isinstance(container.children[1], html.Div)

    def test_the_card_style_wraps_header_and_body_in_a_card(self):
        import dash_bootstrap_components as dbc

        container, _ = self._assemble(use_cardstyle=True)
        card = container.children[0]
        assert isinstance(card, dbc.Card)
        assert card.className == "p-0"
        assert card.children[0].className == "pt-0"
        assert card.children[1].className == "px-2 py-0"


class TestCreateTabsFlow:
    def _draw(self, active):
        return "BODY:" + active

    def test_without_flask_the_call_refuses_with_the_dash_extra_hint(self, monkeypatch):
        monkeypatch.setattr(dl, "session", None)
        with pytest.raises(ImportError) as info:
            dl.create_tabs("g", [("A", "a", None)], self._draw)
        assert str(info.value) == "create_tabs needs flask; install the dash extra"

    def test_an_empty_tab_list_renders_nothing_and_logs_it(self, monkeypatch):
        monkeypatch.setattr(dl, "session", {})
        log = _Log()
        monkeypatch.setattr(dl, "logger", log)
        assert dl.create_tabs("g", [], self._draw) is None
        assert log.debug_calls == [("create_tabs(%s): empty tabsList, nothing to render", "g")]

    def test_an_unauthenticated_user_renders_nothing(self, monkeypatch):
        monkeypatch.setattr(dl, "session", {})
        monkeypatch.setattr(dl, "_resolve_current_user", lambda name: (False, None))
        assert dl.create_tabs("g", [("A", "a", None)], self._draw) is None

    def test_the_session_is_seeded_with_the_first_tab_id(self, monkeypatch):
        store: dict = {}
        monkeypatch.setattr(dl, "session", store)
        monkeypatch.setattr(dl, "_resolve_current_user", lambda name: (True, None))
        dl.create_tabs("g", [("A", "b", None), ("C", "c", None)], self._draw)
        assert store[dl.get_active_tab_var_name("g")] == "tabb"

    def test_a_missing_tab_id_seeds_the_session_from_the_label(self, monkeypatch):
        store: dict = {}
        monkeypatch.setattr(dl, "session", store)
        monkeypatch.setattr(dl, "_resolve_current_user", lambda name: (True, None))
        dl.create_tabs("g", [("Lbl", None, None)], self._draw)
        assert store[dl.get_active_tab_var_name("g")] == "tabLbl"

    def test_an_existing_session_entry_is_kept(self, monkeypatch):
        store = {dl.get_active_tab_var_name("g"): "tabc"}
        monkeypatch.setattr(dl, "session", store)
        monkeypatch.setattr(dl, "_resolve_current_user", lambda name: (True, None))
        container = dl.create_tabs("g", [("A", "b", None), ("C", "c", None)], self._draw)
        assert store[dl.get_active_tab_var_name("g")] == "tabc"
        assert container.children[0].active_tab == "tabc"

    def test_no_tab_surviving_the_role_filter_renders_nothing_and_logs(self, monkeypatch):
        monkeypatch.setattr(dl, "session", {})
        monkeypatch.setattr(dl, "_resolve_current_user", lambda name: (True, None))
        log = _Log()
        monkeypatch.setattr(dl, "logger", log)
        assert dl.create_tabs("g", [("A", "a", ["admin"])], self._draw) is None
        assert log.debug_calls == [("create_tabs(%s): no tab passed the role filter for the current user, nothing to render", "g")]

    def test_a_visible_tab_renders_the_container(self, monkeypatch):
        import dash_bootstrap_components as dbc

        monkeypatch.setattr(dl, "session", {})
        monkeypatch.setattr(dl, "_resolve_current_user", lambda name: (True, None))
        container = dl.create_tabs("g", [("A", "a", None)], self._draw)
        assert isinstance(container, dbc.Container)


class TestCreateTabsOptions:
    def _draw(self, active):
        return "BODY:" + active

    def _run(self, monkeypatch, tabs, user_role=None, **kwargs):
        monkeypatch.setattr(dl, "session", {})
        monkeypatch.setattr(dl, "_resolve_current_user", lambda name: (True, _User(user_role) if user_role else None))
        return dl.create_tabs("g", tabs, self._draw, **kwargs)

    def test_a_comma_string_of_roles_is_split_by_default(self, monkeypatch):
        container = self._run(monkeypatch, [("A", "a", "admin,user")], user_role="user")
        assert container is not None
        assert len(container.children[0].children) == 1

    def test_a_string_of_roles_hides_the_tab_from_other_roles(self, monkeypatch):
        assert self._run(monkeypatch, [("A", "a", "admin,user")], user_role="guest") is None

    def test_the_active_label_class_is_applied_to_the_active_tab(self, monkeypatch):
        container = self._run(monkeypatch, [("A", "a", None, "tc", "base", None)], activeLabelClassName="act")
        tab = container.children[0].children[0]
        assert tab.labelClassName == "base act"

    def test_the_tab_bar_and_content_container_keep_their_own_classes(self, monkeypatch):
        container = self._run(monkeypatch, [("A", "a", None)], tabsClassName="bar", contentClassName="wrap")
        assert container.className == "wrap"
        assert container.children[0].className == "bar"

    def test_the_card_style_flag_is_not_taken_for_the_content_class(self, monkeypatch):
        container = self._run(monkeypatch, [("A", "a", None)], contentClassName="wrap", use_cardstyle=True)
        assert container.className == "wrap"
        assert container.children[0].className == "p-0"
