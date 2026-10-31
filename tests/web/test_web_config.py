"""pyutilz.web.web.config: the three configuration entry points, checked on the package object they write.

Every assertion reads the package attribute (``pyutilz.web.web.<name>``), because that is where the module
writes. A test that checked only a return value would pass with the writes removed.
"""

import logging
from unittest.mock import MagicMock

import pytest

import pyutilz.web.web as web
from pyutilz.web.web import config as cfg

_STATE = (
    "sess",
    "num_ip_queries",
    "template_headers",
    "headers",
    "proxies",
    "timeout",
    "proxy_user",
    "proxy_pass",
    "proxy_server",
    "proxy_min_port",
    "proxy_max_port",
    "proxy_port",
    "proxy_type",
    "delay",
    "max_ip_queries",
    "last_used_dict",
    "min_idle_interval_minutes",
    "failed_dict",
    "min_failed_idle_interval_minutes",
    "get_new_smartproxy",
)


@pytest.fixture(autouse=True)
def _restore_package_state(monkeypatch):
    """Record every global the config functions write, so each test's writes are undone afterwards."""
    for name in _STATE:
        monkeypatch.setattr(web, name, getattr(web, name))


class TestInitVars:
    def test_resets_the_session_state_to_its_defaults(self):
        web.sess = None
        web.num_ip_queries = 9
        web.template_headers = {"A": "b"}
        web.headers = {"C": "d"}
        web.proxies = {"http": "x"}
        web.timeout = 99

        cfg.init_vars()

        assert web.sess is None
        assert web.num_ip_queries == 0
        assert web.template_headers is None
        assert web.headers == {}
        assert web.proxies is None
        assert web.timeout == 10

    def test_closes_the_session_it_drops(self):
        old = MagicMock()
        web.sess = old

        cfg.init_vars()

        old.close.assert_called_once_with()
        assert web.sess is None

    def test_a_failing_close_is_logged_at_error_and_not_raised(self, caplog):
        old = MagicMock()
        old.close.side_effect = RuntimeError("boom")
        web.sess = old

        with caplog.at_level(logging.ERROR):
            cfg.init_vars()

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert [r.getMessage() for r in errors] == ["boom"]
        assert web.sess is None

    def test_logs_session_cleared_at_debug(self, caplog):
        with caplog.at_level(logging.DEBUG):
            cfg.init_vars()

        debug = [r for r in caplog.records if r.levelno == logging.DEBUG and r.getMessage() == "Session cleared"]
        assert len(debug) == 1

    def test_no_session_means_nothing_is_closed(self, caplog):
        web.sess = None

        with caplog.at_level(logging.ERROR):
            cfg.init_vars()

        assert [r for r in caplog.records if r.levelno == logging.ERROR] == []


class TestConnect:
    def test_writes_each_proxy_field_it_was_given(self):
        cfg.connect(
            m_proxy_user="u",
            m_proxy_pass="p",
            m_proxy_server="s",
            m_proxy_min_port="20001",
            m_proxy_max_port="37960",
            m_proxy_port="3128",
            m_proxy_type="socks5",
            m_template_headers={"A": "b"},
            m_timeout=7,
        )

        assert web.proxy_user == "u"
        assert web.proxy_pass == "p"
        assert web.proxy_server == "s"
        assert web.proxy_min_port == "20001"
        assert web.proxy_max_port == "37960"
        assert web.proxy_port == "3128"
        assert web.proxy_type == "socks5"
        assert web.template_headers == {"A": "b"}
        assert web.timeout == 7

    def test_timeout_and_proxy_type_default_to_ten_seconds_and_http(self):
        cfg.connect()

        assert web.timeout == 10
        assert web.proxy_type == "http"

    def test_unrecognised_keyword_arguments_are_accepted_and_ignored(self):
        cfg.connect(m_proxy_user="u", something_else=1)

        assert web.proxy_user == "u"

    def test_the_session_is_reset_before_the_new_values_are_written(self):
        web.sess = MagicMock()
        web.template_headers = {"old": "1"}

        cfg.connect(m_template_headers={"new": "2"})

        assert web.sess is None
        assert web.template_headers == {"new": "2"}


class TestSetProxy:
    @pytest.mark.parametrize(
        "user, password, server",
        [(None, "p", "s"), ("u", None, "s"), ("u", "p", None)],
    )
    def test_refuses_a_missing_credential_or_server(self, user, password, server):
        with pytest.raises(ValueError) as info:
            cfg.set_proxy(user, password, server)
        assert str(info.value) == "set_proxy: proxy_user, proxy_pass and proxy_server are required (got None)"

    def test_stores_the_fields_it_was_given(self):
        web.get_new_smartproxy = MagicMock(return_value={"http": "x"})

        cfg.set_proxy("u", "p", "s", m_proxy_min_port=1, m_proxy_max_port=2, m_proxy_port=3, m_proxy_type="https")

        assert web.proxy_user == "u"
        assert web.proxy_pass == "p"
        assert web.proxy_server == "s"
        assert web.proxy_min_port == 1
        assert web.proxy_max_port == 2
        assert web.proxy_port == 3
        assert web.proxy_type == "https"

    def test_stores_the_proxy_list_obtained_from_get_new_smartproxy(self):
        web.get_new_smartproxy = MagicMock(return_value={"http": "x"})

        cfg.set_proxy("u", "p", "s")

        assert web.proxies == {"http": "x"}

    def test_passes_the_throttling_state_and_the_port_range_to_get_new_smartproxy(self):
        web.get_new_smartproxy = MagicMock(return_value={})
        web.last_used_dict = last = {"l": 1}
        web.failed_dict = failed = {"f": 2}

        cfg.set_proxy("u", "p", "s", m_proxy_min_port=1, m_proxy_max_port=2, m_proxy_port=3, m_proxy_type="https")

        args, kwargs = web.get_new_smartproxy.call_args
        assert args == ("u", "p", "s", 1, 2)
        assert kwargs["proxy_port"] == 3
        assert kwargs["proxy_type"] == "https"
        assert kwargs["last_used_dict"] is last
        assert kwargs["failed_dict"] is failed

    def test_unset_ports_and_proxy_type_fall_back_to_their_defaults(self):
        web.get_new_smartproxy = MagicMock(return_value={})

        cfg.set_proxy("u", "p", "s")

        args, kwargs = web.get_new_smartproxy.call_args
        assert args == ("u", "p", "s", 20001, 37960)
        assert kwargs["proxy_type"] == "http"

    def test_an_explicit_none_proxy_type_falls_back_to_http(self):
        web.get_new_smartproxy = MagicMock(return_value={})

        cfg.set_proxy("u", "p", "s", m_proxy_type=None)

        _, kwargs = web.get_new_smartproxy.call_args
        assert kwargs["proxy_type"] == "http"

    def test_unset_idle_intervals_fall_back_to_zero_and_one_day(self):
        web.get_new_smartproxy = MagicMock(return_value={})
        web.min_idle_interval_minutes = None
        web.min_failed_idle_interval_minutes = None

        cfg.set_proxy("u", "p", "s")

        _, kwargs = web.get_new_smartproxy.call_args
        assert kwargs["min_idle_interval_minutes"] == 0
        assert kwargs["min_failed_idle_interval_minutes"] == 60 * 24

    def test_configured_idle_intervals_are_passed_through(self):
        web.get_new_smartproxy = MagicMock(return_value={})
        web.min_idle_interval_minutes = 5
        web.min_failed_idle_interval_minutes = 30

        cfg.set_proxy("u", "p", "s")

        _, kwargs = web.get_new_smartproxy.call_args
        assert kwargs["min_idle_interval_minutes"] == 5
        assert kwargs["min_failed_idle_interval_minutes"] == 30


class TestSetParams:
    def test_assigns_each_throttling_field(self):
        last = {"l": 1}
        failed = {"f": 2}

        cfg.set_params(
            m_delay=2,
            m_max_ip_queries=5,
            m_last_used_dict=last,
            m_min_idle_interval_minutes=3,
            m_failed_dict=failed,
            m_min_failed_idle_interval_minutes=4,
        )

        assert web.delay == 2
        assert web.max_ip_queries == 5
        assert web.last_used_dict is last
        assert web.min_idle_interval_minutes == 3
        assert web.failed_dict is failed
        assert web.min_failed_idle_interval_minutes == 4

    def test_defaults_are_no_delay_and_no_query_cap(self):
        cfg.set_params()

        assert web.delay == 0
        assert web.max_ip_queries == 0
        assert web.last_used_dict is None
        assert web.min_idle_interval_minutes is None
        assert web.failed_dict is None
        assert web.min_failed_idle_interval_minutes is None
