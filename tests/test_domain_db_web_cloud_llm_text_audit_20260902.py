"""Regression tests for the 2026-09-02 database / web / cloud / LLM / text domain audit.

Each test targets one finding and is written to fail on the pre-fix source. No test touches a
real database, a real network endpoint, or a real LLM provider.
"""

import asyncio
import re
import sys
import threading
from unittest.mock import Mock, patch

import pytest

# ---------------------------------------------------------------------------
# F01 / F16 / F45 -- build_upsert_query
# ---------------------------------------------------------------------------

from pyutilz.database.db.upsert import build_upsert_query

_COMMON = dict(
    fields_names=["id", "name"],
    table_name="t",
    conflict_fields=["id"],
    on_conflict_update_fields=["name"],
    timestamp_check_fields=["checked_at"],
    timestamp_update_fields=["updated_at"],
    history_table_name="t_hist",
    history_fields=["id", "name"],
)


def test_f01_timestamp_update_without_hash_fields_keeps_join_predicate():
    """Without hash_fields the trailing UPDATE ... FROM changed_data used to have no WHERE at
    all -- a cross join that overwrites the timestamp column on every row of the table."""
    q = build_upsert_query(**_COMMON)
    assert "update t AS u set updated_at=c.checked_at from changed_data as c where u.id=c.id" in q


def test_f01_join_predicate_matches_the_hash_fields_variant():
    with_hash = build_upsert_query(hash_fields="name", **_COMMON)
    without_hash = build_upsert_query(**_COMMON)
    for q in (with_hash, without_hash):
        assert re.search(r"from changed_data as c where u\.id=c\.id\)", q)


def test_f01_multi_column_conflict_key_is_fully_joined():
    q = build_upsert_query(
        fields_names=["a", "b", "name"],
        table_name="t",
        conflict_fields=["a", "b"],
        on_conflict_update_fields=["name"],
        timestamp_check_fields=["checked_at"],
        timestamp_update_fields=["updated_at"],
        history_table_name="t_hist",
        history_fields=["a", "b"],
    )
    assert "where u.a=c.a and u.b=c.b)" in q


def test_f16_history_table_without_history_fields_raises():
    """It used to emit `insert into t_hist() select  from fresh_data u` -- a syntax error only
    discovered at execution time."""
    with pytest.raises(ValueError, match="history_fields"):
        build_upsert_query(
            fields_names=["id", "name"],
            table_name="t",
            conflict_fields=["id"],
            history_table_name="t_hist",
        )


def test_f45_timestamp_field_arity_mismatch_raises_not_asserts():
    """An assert is stripped under `python -O`, after which the doubled zip() paired the update
    field against a SECOND COPY of the check list instead of truncating."""
    with pytest.raises(ValueError, match="1:1"):
        build_upsert_query(
            fields_names=["id", "name"],
            table_name="t",
            conflict_fields=["id"],
            timestamp_check_fields=["checked_at"],
            timestamp_update_fields=["updated_at", "refreshed_at"],
            history_table_name="t_hist",
            history_fields=["id"],
        )


def test_f45_pairing_is_positional_not_doubled():
    q = build_upsert_query(
        fields_names=["id", "name"],
        table_name="t",
        conflict_fields=["id"],
        timestamp_check_fields=["checked_at", "seen_at"],
        timestamp_update_fields=["updated_at", "refreshed_at"],
        history_table_name="t_hist",
        history_fields=["id"],
    )
    assert "updated_at=c.checked_at,refreshed_at=c.seen_at" in q


# ---------------------------------------------------------------------------
# F03 / F15 / F17 / F19 / F42 / F43 / F47 -- database.db facade
# ---------------------------------------------------------------------------

import pyutilz.database.db as dbmod


@pytest.fixture()
def recorded_execute(monkeypatch):
    """Capture every safe_execute call instead of touching a database."""
    calls = []

    def _fake(statement, data=None, *a, **kw):
        calls.append((statement, data))
        return []

    monkeypatch.setattr(dbmod, "safe_execute", _fake)
    return calls


def test_f03_regjobs_wraps_dict_result_for_psycopg2(recorded_execute):
    """psycopg2 cannot adapt a bare dict; the heartbeat raised ProgrammingError instead."""
    from psycopg2.extras import Json

    dbmod.regjobs_progress("nightly_etl", {"rows": 120})
    _stmt, params = recorded_execute[-1]
    assert isinstance(params["result"], Json)
    assert params["result"].adapted == {"rows": 120}


def test_f03_regjobs_finalize_wraps_too(recorded_execute):
    from psycopg2.extras import Json

    dbmod.regjobs_finalize("nightly_etl", {"ok": True})
    _stmt, params = recorded_execute[-1]
    assert isinstance(params["result"], Json)


@pytest.mark.parametrize("level", [None, ""])
def test_f15_falsy_log_level_still_persists_the_message(recorded_execute, level):
    """The whole body used to be nested in `if level:` -- no log record, no row, no warning."""
    dbmod.log_to_db("boom", level=level)
    assert recorded_execute, "log_to_db silently discarded the message"


def test_f17_insert_without_set_fields_returns_none_with_a_diagnostic(recorded_execute, caplog):
    """It used to raise an opaque TypeError from two frames down inside
    construct_templates_and_values."""
    with caplog.at_level("ERROR"):
        assert dbmod.db_command("insert", "t", where_fields=["id"], set_fields=None, source={"id": 1}) is None
    assert not recorded_execute
    assert any("set_fields" in r.getMessage() for r in caplog.records)


def test_f17_insert_with_empty_set_fields_is_rejected(recorded_execute):
    assert dbmod.db_command("insert", "t", set_fields=[], source={}) is None
    assert not recorded_execute


def test_f43_returning_none_does_not_raise(recorded_execute):
    """`None` is the natural way to express "no RETURNING"; it used to raise TypeError."""
    dbmod.db_command("select", "t", where_fields=["id"], source={"id": 1}, returning=None)
    stmt, _ = recorded_execute[-1]
    assert stmt.startswith("select * from")


def test_f43_returning_none_on_insert_emits_no_returning_clause(recorded_execute):
    dbmod.db_command("insert", "t", set_fields=["name"], source={"name": "x"}, returning=None)
    stmt, _ = recorded_execute[-1]
    assert "returning" not in stmt


def test_f19_auto_commit_false_is_warned_about(monkeypatch, caplog):
    """The connection runs in autocommit; a caller batching under auto_commit=False has no
    transaction and no rollback, and used to get no signal at all."""
    monkeypatch.setattr(dbmod, "get_cursor", lambda **kw: Mock(description=None))
    with caplog.at_level("WARNING"):
        dbmod.basic_db_execute("execute", "select 1", auto_commit=False)
    assert any("auto_commit" in r.getMessage() for r in caplog.records)


def test_f42_json_path_is_validated(monkeypatch):
    """`path` was spliced verbatim into the query text three times, with no validation and no
    trusted-input warning in the docstring (unlike its siblings)."""
    monkeypatch.setattr(dbmod, "safe_execute", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="JSON navigation path"):
        dbmod.suggest_json_optimization("t", "payload", path="'a'::text)--", fields=["x"])


def test_f42_legitimate_json_path_still_accepted(monkeypatch):
    seen = []
    monkeypatch.setattr(dbmod, "safe_execute", lambda stmt, *a, **kw: seen.append(stmt) or [])
    dbmod.suggest_json_optimization("t", "payload", path="'a'->'b'", fields=["x"])
    assert "payload->'a'->'b'" in seen[0]


def test_f47_explain_table_warns_on_non_mysql(monkeypatch, caplog):
    monkeypatch.setattr(dbmod, "db_flavor", "postgres", raising=False)
    with caplog.at_level("WARNING"):
        assert dbmod.explain_table("orders") is None
    assert any("mysql" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# F04 / F18 / F46 -- psycopg2 connection pool
# ---------------------------------------------------------------------------

import pyutilz.database.psycopg2_pool as poolmod


class _FakePool:
    def __init__(self, *a, **kw):
        self.returned = []

    def getconn(self):
        return Mock()

    def putconn(self, conn, close=False):
        self.returned.append(conn)

    def closeall(self):
        pass


@pytest.fixture()
def clean_pool(monkeypatch):
    monkeypatch.setattr(poolmod.psycopg2.pool, "ThreadedConnectionPool", _FakePool)
    poolmod._pool = None
    poolmod._pool_dsn = None
    yield
    poolmod._pool = None
    poolmod._pool_dsn = None


def test_f04_second_dsn_is_refused_instead_of_silently_served(clean_pool):
    """`get_connection(dsn_prod)` then `get_connection(dsn_staging)` used to hand back a PROD
    connection with no error and no log line."""
    poolmod._ensure_pool("dbname=prod")
    with pytest.raises(ValueError, match="different DSN"):
        poolmod._ensure_pool("dbname=staging")


def test_f04_same_dsn_reuses_the_pool(clean_pool):
    first = poolmod._ensure_pool("dbname=prod")
    assert poolmod._ensure_pool("dbname=prod") is first


def test_f04_get_connection_from_pool_also_checks_the_dsn(clean_pool):
    poolmod._ensure_pool("dbname=prod")
    with pytest.raises(ValueError, match="different DSN"):
        poolmod.get_connection_from_pool("dbname=staging")


def test_f18_connection_with_no_pool_is_closed_not_abandoned(clean_pool):
    """After close_pool() the connection was neither returned nor closed -- its server-side
    backend stayed open until GC or process exit."""
    conn = Mock()
    conn.info.backend_pid = 4321
    poolmod._pool = None
    poolmod.release_connection(conn)
    conn.close.assert_called_once()


def test_f46_zero_backend_pid_falls_back_to_id():
    """psycopg2 2.9.x returns backend_pid == 0 for a closed connection rather than raising, so
    every closed connection shared one _conn_last_used key."""
    a, b = Mock(), Mock()
    a.info.backend_pid = 0
    b.info.backend_pid = 0
    assert poolmod._conn_key(a) != poolmod._conn_key(b)


def test_f46_live_backend_pid_is_used():
    conn = Mock()
    conn.info.backend_pid = 4242
    assert poolmod._conn_key(conn) == 4242


# ---------------------------------------------------------------------------
# F05 / F06 / F48 / F51 -- web.py
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta

import pyutilz.web.web as webmod


def test_f05_random_port_is_rerolled_every_iteration(monkeypatch):
    """`proxy_port` (the parameter) was rebound in place, so the guard was False forever and the
    loop re-hashed one single port until max_wait_seconds."""
    seen = []
    real_make = webmod.make_proxies_dict

    def spy(user, pwd, server, port, ptype):
        seen.append(port)
        return real_make(user, pwd, server, port, ptype)

    monkeypatch.setattr(webmod, "make_proxies_dict", spy)
    monkeypatch.setattr(webmod, "sleep", lambda *a, **kw: None)
    values = iter([0.0, 0.25, 0.5, 0.75, 0.9])
    monkeypatch.setattr(webmod, "random", lambda: next(values, 0.99))

    # Every proxy reports as just-used, so the loop keeps rolling until the budget runs out.
    class _AlwaysTouched(dict):
        def __contains__(self, key):
            return True

        def __getitem__(self, key):
            return datetime.utcnow()

    with pytest.raises(TimeoutError):
        webmod.get_new_smartproxy(
            "u", "p", "host", 20001, 37960,
            last_used_dict=_AlwaysTouched(),
            min_idle_interval_minutes=10,
            delay=0,
            max_wait_seconds=0.0,
        )
    assert len(set(seen)) > 1, f"only one port was ever tried: {set(seen)}"


def test_f05_explicit_port_is_still_honoured_every_iteration(monkeypatch):
    seen = []
    real_make = webmod.make_proxies_dict
    monkeypatch.setattr(
        webmod, "make_proxies_dict",
        lambda u, p, s, port, t: (seen.append(port), real_make(u, p, s, port, t))[1],
    )
    webmod.get_new_smartproxy("u", "p", "host", 20001, 37960, proxy_port=31337)
    assert seen == [31337]


def test_f06_failed_proxy_respects_its_own_longer_cooldown(monkeypatch):
    """Both dicts were compared against min_idle_interval_minutes (default 0), so a port just
    marked failed was handed straight back on the next rotation."""
    proxies = webmod.make_proxies_dict("u", "p", "host", 31337, "http")
    key = webmod.joblib_hash(proxies)
    failed = {key: datetime.utcnow() - timedelta(minutes=5)}
    monkeypatch.setattr(webmod, "sleep", lambda *a, **kw: None)
    with pytest.raises(TimeoutError):
        webmod.get_new_smartproxy(
            "u", "p", "host", 31337, 31337,
            failed_dict=failed,
            min_idle_interval_minutes=0,
            min_failed_idle_interval_minutes=60 * 24,
            delay=0,
            max_wait_seconds=0.0,
        )


def test_f06_failed_proxy_is_eligible_once_its_cooldown_elapsed(monkeypatch):
    proxies = webmod.make_proxies_dict("u", "p", "host", 31337, "http")
    key = webmod.joblib_hash(proxies)
    failed = {key: datetime.utcnow() - timedelta(minutes=90)}
    got = webmod.get_new_smartproxy(
        "u", "p", "host", 31337, 31337,
        failed_dict=failed,
        min_idle_interval_minutes=0,
        min_failed_idle_interval_minutes=60,
        proxy_port=31337,
    )
    assert got == proxies


def test_f51_failing_geo_provider_is_logged(monkeypatch, caplog):
    """get_ipinfo swallows its own exceptions and returns None, so the except branch could never
    fire and a failing provider produced no diagnostic at all."""
    monkeypatch.setattr(webmod, "get_ipinfo", lambda **kw: None)
    with caplog.at_level("DEBUG", logger="pyutilz.web.web"):
        webmod.get_country_by_ip("1.2.3.4", providers=[{"url": "http://x/{ip}", "code": "c", "name": "n"}])
    assert any("no usable payload" in r.getMessage() for r in caplog.records)


def test_f07_broken_stream_leaves_no_zero_byte_file(tmp_path):
    """A requests body is single-use: retrying only the write loop iterated an already-consumed
    stream, yielding nothing, and left a 0-byte file while returning the success value."""
    out = tmp_path / "big.bin"

    def make_resp():
        resp = Mock()
        resp.status_code = 200
        resp.iter_content = Mock(side_effect=Exception("connection reset"))
        return resp

    with patch.object(webmod, "sleep", lambda *a, **kw: None), patch.object(webmod.requests, "get", side_effect=lambda *a, **kw: make_resp()):
        assert webmod.download_to_file("http://x/f", str(out), max_attempts=2) is None
    assert not out.exists()


def test_f07_each_attempt_reissues_the_request(tmp_path):
    out = tmp_path / "f.bin"
    attempts = []

    def make_resp():
        resp = Mock()
        resp.status_code = 200
        if len(attempts) == 0:
            resp.iter_content = Mock(side_effect=Exception("reset"))
        else:
            resp.iter_content = Mock(return_value=[b"payload"])
        attempts.append(resp)
        return resp

    with patch.object(webmod, "sleep", lambda *a, **kw: None), patch.object(webmod.requests, "get", side_effect=lambda *a, **kw: make_resp()) as mget:
        webmod.download_to_file("http://x/f", str(out), max_attempts=3)
    assert mget.call_count == 2
    assert out.read_bytes() == b"payload"


def test_f48_none_timeout_falls_back_to_a_finite_value(monkeypatch):
    """requests treats timeout=None as "wait forever", which hangs the retry loop."""
    captured = {}

    def fake_get(url, **kw):
        captured.update(kw)
        resp = Mock()
        resp.status_code = 200
        resp.text = "ok"
        return resp

    monkeypatch.setattr(webmod.requests, "get", fake_get)
    monkeypatch.setattr(webmod, "timeout", None)
    webmod.get_url("http://x/", max_retries=1, b_use_session=False, b_use_proxy=False, b_random_ua=False)
    assert captured["timeout"] is not None


def test_f20_retry_after_header_is_parsed():
    resp = Mock()
    resp.headers = {"Retry-After": "12"}
    assert webmod._parse_retry_after(resp) == 12.0


def test_f20_absurd_retry_after_is_clamped():
    resp = Mock()
    resp.headers = {"Retry-After": "99999"}
    assert webmod._parse_retry_after(resp) == 300.0


def test_f20_http_date_retry_after_is_ignored():
    resp = Mock()
    resp.headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
    assert webmod._parse_retry_after(resp) is None


# ---------------------------------------------------------------------------
# F21 / F49 / F52 -- browser.py
# ---------------------------------------------------------------------------

import pyutilz.web.browser as browsermod


def _login_stub_browser(monkeypatch, cookies):
    """Wire browser.py's module globals so LoginAndGetCookies() runs its cookie/header block
    against a fake driver, with no Selenium and no network."""
    fake = Mock()
    fake.title = "logged-in"
    fake.execute_script.return_value = "UA/1.0"
    fake.get_cookie.side_effect = lambda name: {"value": cookies[name]} if name in cookies else None
    monkeypatch.setattr(browsermod, "browser", fake)
    monkeypatch.setattr(browsermod, "home_page", "http://example.invalid/")
    monkeypatch.setattr(browsermod, "required_cookies", tuple(cookies))
    monkeypatch.setattr(browsermod, "fixed_cookies", {})
    monkeypatch.setattr(browsermod, "logout_signs", ())
    monkeypatch.setattr(browsermod, "successful_login_signs", ("logged-in",))
    monkeypatch.setattr(browsermod, "browser_get", lambda url: None)
    monkeypatch.setattr(browsermod.pythonlib, "imitate_delay", lambda **kw: None)
    monkeypatch.setattr(browsermod, "sleep", lambda *a, **kw: None)
    return fake


def test_f21_login_does_not_mutate_the_shared_basic_headers(monkeypatch):
    """`headers = basic_headers` rebound the SAME dict object, so basic_headers -- documented as
    the neutral default header set -- permanently carried the last session's cookie and Bearer
    token, and leaked them to every other consumer of it."""
    pytest.importorskip("selenium")
    monkeypatch.setattr(browsermod, "basic_headers", {"accept": "*/*"})
    _login_stub_browser(monkeypatch, {"sid": "secret-session"})
    assert browsermod.LoginAndGetCookies() is True
    assert "cookie" in browsermod.headers
    assert browsermod.basic_headers == {"accept": "*/*"}


def test_f21_stale_authorization_is_not_carried_into_a_new_session(monkeypatch):
    """A login that produces no oauth2_global_js_token used to keep sending the PREVIOUS
    account's Bearer token."""
    pytest.importorskip("selenium")
    monkeypatch.setattr(browsermod, "basic_headers", {"accept": "*/*", "authorization": "Bearer previous-account"})
    _login_stub_browser(monkeypatch, {"sid": "s"})
    browsermod.LoginAndGetCookies()
    assert "authorization" not in browsermod.headers


def test_f49_none_proxy_pass_is_treated_as_unauthenticated(monkeypatch):
    """`len(proxy_server.get("PROXY_PASS", ""))` covered only a MISSING key; a present-but-None
    value -- the JSON way of saying "unauthenticated proxy" -- raised TypeError at startup."""
    webdriver = pytest.importorskip("selenium.webdriver")

    class _Launched(Exception):
        """Sentinel raised in place of actually starting a browser process."""

    added = []
    monkeypatch.setattr(browsermod, "undetectable", False)
    monkeypatch.setattr(browsermod, "user_agent", None, raising=False)
    monkeypatch.setattr(browsermod, "data_dir", None, raising=False)
    monkeypatch.setattr(browsermod, "proxy_server", {"PROXY_HOST": "h", "PROXY_PORT": 1234, "PROXY_PASS": None})
    monkeypatch.setattr(webdriver, "ChromeOptions", lambda: Mock(add_argument=added.append, add_extension=added.append))
    monkeypatch.setattr(webdriver, "Chrome", Mock(side_effect=_Launched))

    with pytest.raises(_Launched):
        browsermod.start_selenium()
    assert any("--proxy-server=h:1234" in str(a) for a in added)


# ---------------------------------------------------------------------------
# F22 / F23 -- CachedHttpClient
# ---------------------------------------------------------------------------

from pyutilz.web.cached_client import CachedHttpClient


def test_f22_text_and_json_do_not_share_a_cache_key(tmp_path):
    """get_text writes {"text": ...} while get_json reads payload["data"], so a get_text entry
    poisoned the same url+tag for get_json with a permanent None."""
    client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
    assert client._cache_path("http://x/a", "tag", kind="json") != client._cache_path("http://x/a", "tag", kind="text")


def test_f22_get_json_after_get_text_still_fetches(tmp_path):
    client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
    with patch.object(CachedHttpClient, "_fetch_bytes", return_value=b'{"a": 1}'):
        assert client.get_text("http://x/a", "tag") == '{"a": 1}'
        assert client.get_json("http://x/a", "tag") == {"a": 1}


def test_f23_non_utf8_body_does_not_abort_the_batch(tmp_path):
    """UnicodeDecodeError is a ValueError but NOT a JSONDecodeError, so one mis-encoded response
    escaped and aborted the whole ingestion batch. Decoding with errors="replace" (what the
    sibling get_text already did) keeps the row usable instead."""
    client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
    with patch.object(CachedHttpClient, "_fetch_bytes", return_value=b'{"name": "Caf\xe9"}'):
        assert client.get_json("http://x/b", "tag") == {"name": "Caf" + chr(0xFFFD)}


def test_f23_undecodable_non_json_body_returns_none(tmp_path):
    client = CachedHttpClient(cache_dir=tmp_path, min_interval=0)
    with patch.object(CachedHttpClient, "_fetch_bytes", return_value=b"\xff\xfe not json"):
        assert client.get_json("http://x/c", "tag") is None


# ---------------------------------------------------------------------------
# F24 -- PortHealthTracker
# ---------------------------------------------------------------------------

from pyutilz.web.proxy.base import PortHealthTracker


def test_f24_successes_are_trimmed_to_the_window():
    """report_error trimmed; report_success did not, so a healthy pool retained one sample per
    request per port for the whole process lifetime."""
    tracker = PortHealthTracker(window=0.001)
    import time as _time

    for _ in range(500):
        tracker.report_success(7)
    _time.sleep(0.01)
    tracker.report_success(7)
    assert tracker._ports[7].total < 500


# ---------------------------------------------------------------------------
# F25 / F50 -- Decodo provider
# ---------------------------------------------------------------------------

from pyutilz.web.proxy import decodo as decodomod


def test_f25_traffic_report_pages_past_the_first_page(monkeypatch):
    """`"page": 1` was hardcoded, so an account with more than `limit` rows had its total
    silently understated and printed as authoritative."""
    pages = {
        1: {"data": [{"grouping_key": f"t{i}", "requests": 1, "totals": 10} for i in range(2)]},
        2: {"data": [{"grouping_key": "last", "requests": 5, "totals": 50}]},
    }
    seen_pages = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen_pages.append(json["page"])
        resp = Mock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: pages[json["page"]]
        return resp

    import requests as _requests

    monkeypatch.setattr(_requests, "post", fake_post)
    provider = decodomod.DecodoProvider.__new__(decodomod.DecodoProvider)
    provider._api_headers = lambda: {}
    report = provider.get_traffic(days=1, limit=2)
    assert seen_pages == [1, 2]
    assert report.total_requests == 7


def test_f50_bad_port_range_env_names_the_variable(monkeypatch):
    """A typo produced a bare `invalid literal for int()` naming neither the variable nor the
    value, which an operator cannot map back to their environment."""
    monkeypatch.setenv("PROXY_USER", "u")
    monkeypatch.setenv("PROXY_PASS", "p")
    monkeypatch.setenv("PROXY_HOST", "h")
    monkeypatch.setenv("PROXY_PORT", "10000")
    monkeypatch.setenv("PROXY_PORT_RANGE", "notanumber")
    with pytest.raises(ValueError, match="PROXY_PORT_RANGE"):
        decodomod.DecodoProvider.from_env()


# ---------------------------------------------------------------------------
# F09 / F27 / F53 -- LLM base + openai_compat
# ---------------------------------------------------------------------------

from pyutilz.llm.base import LLMProvider
from pyutilz.llm.exceptions import JSONParsingError


@pytest.mark.parametrize(
    "text",
    ['[{"a": 1}]', "[1, 2, 3]", '"just a string"', "42", '```json\n[{"a": 1}]\n```'],
)
def test_f27_non_object_json_raises_the_typed_error(text):
    """extract_json is annotated -> dict[str, Any]; returning a list/str/int broke the caller
    with a TypeError far from the parse site, past the retry layer."""
    with pytest.raises(JSONParsingError):
        LLMProvider.extract_json(text)


def test_f27_objects_still_parse():
    assert LLMProvider.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert LLMProvider.extract_json('prose {"a": 1} trailing') == {"a": 1}


def _make_openai_compat_provider():
    """A minimal concrete subclass: the base class is abstract on the two pricing hooks."""
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _Stub(OpenAICompatibleProvider):
        def _input_cost_per_1m(self, model: str) -> float:
            return 0.0

        def _output_cost_per_1m(self, model: str) -> float:
            return 0.0

    p = _Stub.__new__(_Stub)
    p.model_name = "test-model"
    return p


def test_f09_json_schema_flag_is_isolated_per_task():
    """A plain instance attribute reported ANOTHER concurrent call's strict-schema outcome, so a
    caller skipping enum validation on True could accept unvalidated output."""
    p = _make_openai_compat_provider()
    schema = {"name": "s", "strict": True, "schema": {"type": "object"}}
    p.supports_json_schema = lambda: True
    seen = {}

    async def call(tag, sch):
        p._response_format(json_mode=False, json_schema=sch)
        await asyncio.sleep(0)
        seen[tag] = p.last_json_schema_applied

    async def main():
        await asyncio.gather(
            asyncio.create_task(call("with", schema)),
            asyncio.create_task(call("without", None)),
        )

    asyncio.run(main())
    assert seen == {"with": True, "without": False}


def test_f53_base_reset_per_call_state_clears_stale_metadata():
    """The base implementation was an empty no-op, so for OpenAI/xAI/DeepSeek a generate() that
    raised left the previous call's usage/tool_calls readable as the latest one."""
    p = _make_openai_compat_provider()
    p.last_tool_calls = [{"name": "x"}]
    p.last_citations = ["c"]
    p._last_finish_reason = "stop"
    p._last_usage = {"input_tokens": 5, "output_tokens": 7, "reasoning_tokens": 0}
    p._reset_per_call_state()
    assert p.last_tool_calls == []
    assert p.last_citations == []
    assert p._last_finish_reason is None
    assert p._last_usage["input_tokens"] == 0


# ---------------------------------------------------------------------------
# F08 / F57 -- OpenRouter per-call metadata
# ---------------------------------------------------------------------------


def test_f08_openrouter_per_call_metadata_is_context_scoped():
    """These were plain instance attributes, so generate_batch's N concurrent calls zeroed each
    other's in-flight cost and generation id -- a cost-accounting loop billed the wrong request."""
    from pyutilz.llm.openrouter_provider._provider import OpenRouterProvider

    p = OpenRouterProvider.__new__(OpenRouterProvider)
    results = {}

    async def call(tag, cost, gen_id):
        p.last_actual_cost_usd = cost
        p.last_generation_id = gen_id
        await asyncio.sleep(0)
        results[tag] = (p.last_actual_cost_usd, p.last_generation_id)

    async def main():
        await asyncio.gather(
            asyncio.create_task(call("a", 0.42, "gen-a")),
            asyncio.create_task(call("b", 0.01, "gen-b")),
        )

    asyncio.run(main())
    assert results["a"] == (0.42, "gen-a")
    assert results["b"] == (0.01, "gen-b")


def test_f08_metadata_names_are_carried_in_the_batch_snapshot():
    from pyutilz.llm.openrouter_provider._provider import OpenRouterProvider

    attrs = OpenRouterProvider._PERCALL_METADATA_ATTRS
    for name in ("last_actual_cost_usd", "last_generation_id", "last_upstream_provider", "last_web_search_citations"):
        assert name in attrs


# ---------------------------------------------------------------------------
# F11 -- factory LRU eviction
# ---------------------------------------------------------------------------


def test_f11_evicting_a_still_referenced_provider_does_not_close_it(monkeypatch):
    """Eviction was keyed purely on cache size, not liveness: get_llm_provider() hands out shared
    instances and keeps no refcount, so `_close()` could fire on a provider a coroutine obtained
    earlier and was still using ("Cannot send a request, as the client has been closed")."""
    import types

    import pyutilz.llm.factory as factorymod

    closed = []

    class _FakeProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def _close(self):
            closed.append(self)

    fake_mod = types.ModuleType("pyutilz_test_fake_provider_module")
    fake_mod.FakeProvider = _FakeProvider
    monkeypatch.setitem(sys.modules, fake_mod.__name__, fake_mod)
    monkeypatch.setitem(factorymod._PROVIDER_MODULES, "fake", (fake_mod.__name__, "FakeProvider"))
    monkeypatch.setattr(factorymod, "_PROVIDER_CACHE_MAX_SIZE", 1)
    monkeypatch.setattr(factorymod, "_schedule_provider_close", lambda p: closed.append(p))
    factorymod._provider_cache.clear()

    held = factorymod.get_llm_provider("fake", model="a")
    factorymod.get_llm_provider("fake", model="b")  # evicts the first entry
    assert closed == [], "a provider the caller still holds must not be closed on eviction"
    assert held.kwargs["model"] == "a"

    factorymod._provider_cache.clear()


def test_f11_evicting_an_unreferenced_provider_still_closes_it(monkeypatch):
    import types

    import pyutilz.llm.factory as factorymod

    closed = []

    class _FakeProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mod = types.ModuleType("pyutilz_test_fake_provider_module2")
    fake_mod.FakeProvider = _FakeProvider
    monkeypatch.setitem(sys.modules, fake_mod.__name__, fake_mod)
    monkeypatch.setitem(factorymod._PROVIDER_MODULES, "fake2", (fake_mod.__name__, "FakeProvider"))
    monkeypatch.setattr(factorymod, "_PROVIDER_CACHE_MAX_SIZE", 1)
    monkeypatch.setattr(factorymod, "_schedule_provider_close", lambda p: closed.append(p))
    factorymod._provider_cache.clear()

    factorymod.get_llm_provider("fake2", model="a")  # no reference kept
    factorymod.get_llm_provider("fake2", model="b")
    assert len(closed) == 1

    factorymod._provider_cache.clear()


# ---------------------------------------------------------------------------
# F26 / F32 / F33 / F56 -- ClaudeCodeProvider
# ---------------------------------------------------------------------------

import pyutilz.llm.claude_code_provider as ccmod


def test_f56_impossible_reset_date_returns_none_not_valueerror():
    """The ValueError escaped from INSIDE the rate-limit handler, turning a recoverable pause
    into a hard failure of generate()."""
    assert ccmod._parse_reset_wait_seconds("Your limit resets Feb 30, 3pm") is None


def test_f56_valid_reset_date_still_parses():
    assert ccmod._parse_reset_wait_seconds("Your limit resets Mar 3, 11pm") > 0


def test_f33_generate_json_uses_the_shared_parser():
    """It re-implemented parsing with a greedy pattern spanning the first { to the LAST }, never
    ran the refusal check, and raised a bare ValueError instead of the typed exceptions."""
    p = ccmod.ClaudeCodeProvider.__new__(ccmod.ClaudeCodeProvider)

    async def fake_generate(**kw):
        return 'prose {"a": 1} more prose {"b": 2}'

    p.generate = fake_generate
    assert asyncio.run(p.generate_json("x")) == {"a": 1}


def test_f33_unparseable_json_raises_the_typed_error():
    p = ccmod.ClaudeCodeProvider.__new__(ccmod.ClaudeCodeProvider)

    async def fake_generate(**kw):
        return "not json at all"

    p.generate = fake_generate
    with pytest.raises(JSONParsingError):
        asyncio.run(p.generate_json("x"))


# ---------------------------------------------------------------------------
# F30 / F31 -- Anthropic / Gemini session cost
# ---------------------------------------------------------------------------


def test_f30_anthropic_session_cost_reports_total_cost_usd():
    """Provider-agnostic spend reporting reads this key; Anthropic alone raised KeyError."""
    from pyutilz.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = "claude-sonnet-4-5"
    p._call_count = 2
    p.total_input_tokens = 1000
    p.total_output_tokens = 500
    p.total_thinking_tokens = 0
    p.total_cache_creation_input_tokens = 0
    p.total_cache_read_input_tokens = 0
    cost = p.get_session_cost()
    assert cost["total_cost_usd"] == pytest.approx(cost["input_cost_usd"] + cost["output_cost_usd"])


def test_f30_gemini_has_a_session_cost():
    """GeminiProvider defined none at all, so a multi-provider dashboard hit AttributeError."""
    from pyutilz.llm.gemini_provider import GeminiProvider

    p = GeminiProvider.__new__(GeminiProvider)
    p.model_name = "gemini-2.5-flash"
    p._call_count = 1
    p.total_prompt_tokens = 1_000_000
    p.total_completion_tokens = 0
    p.total_reasoning_tokens = 0
    p.total_cached_content_tokens = 0
    assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.30)


def test_f30_gemini_cache_hits_are_billed_at_the_cache_rate():
    """_CACHE_HIT_COST was unreferenced dead code."""
    from pyutilz.llm.gemini_provider import GeminiProvider

    p = GeminiProvider.__new__(GeminiProvider)
    p.model_name = "gemini-2.5-flash"
    p._call_count = 1
    p.total_prompt_tokens = 1_000_000
    p.total_completion_tokens = 0
    p.total_reasoning_tokens = 0
    p.total_cached_content_tokens = 1_000_000
    assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# F36 / F54 -- pricing / limit tables
# ---------------------------------------------------------------------------


def test_f36_every_priced_openai_model_has_the_other_table_entries():
    from pyutilz.llm.openai_provider import _CACHE_HIT_COST, _CONTEXT_WINDOW, _MAX_TOKENS, _PRICING

    assert not set(_PRICING) - set(_CACHE_HIT_COST)
    assert not set(_PRICING) - set(_MAX_TOKENS)
    assert not set(_PRICING) - set(_CONTEXT_WINDOW)


def test_f36_o1_pro_cache_rate_is_not_borrowed_from_gpt5_mini():
    from pyutilz.llm.openai_provider import _CACHE_HIT_COST, _PRICING

    assert _CACHE_HIT_COST["o1-pro"] > _PRICING["gpt-5-mini"][0]


def test_f36_dated_snapshot_inherits_its_family_budget():
    """Exact dict.get sent any dated snapshot id to the 16,384 / 128,000 class defaults."""
    from pyutilz.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.model_name = "gpt-5-mini-2026-03-01"
    assert p.max_output_tokens == 128_000
    assert p.context_window == 400_000


def test_f54_unknown_xai_model_warns_once(caplog):
    """It silently took the cheapest tariff in the table, under-reporting session cost several
    times over with nothing in the log."""
    from pyutilz.llm.xai_provider import XAIProvider

    XAIProvider._seen_unknown_models.clear()
    p = XAIProvider.__new__(XAIProvider)
    with caplog.at_level("WARNING"):
        assert p._input_cost_per_1m("totally-unknown-model") == 0.20
    assert any("unknown" in r.getMessage() for r in caplog.records)


def test_f54_dated_grok_snapshot_resolves_by_prefix():
    from pyutilz.llm.xai_provider import XAIProvider

    p = XAIProvider.__new__(XAIProvider)
    assert p._input_cost_per_1m("grok-4-0709-2026-01-01") == 3.00


# ---------------------------------------------------------------------------
# F55 -- OpenRouter uptime normalisation
# ---------------------------------------------------------------------------


def test_f55_one_point_zero_in_a_percentage_payload_is_one_percent():
    """`v / 100 if v > 1 else v` resolved the single ambiguous value in the unsafe direction, so
    a nearly-dead backend reporting 1% uptime read as 100% and kept receiving traffic."""
    from pyutilz.llm.openrouter_provider._health import _summarize_endpoints

    out = _summarize_endpoints([
        {"provider_name": "dying", "uptime_last_30m": 1.0},
        {"provider_name": "healthy", "uptime_last_30m": 99.9},
    ])
    dying = out["endpoints"][0]
    assert dying["uptime_30m"] == pytest.approx(0.01)


def test_f55_fraction_payload_keeps_one_point_zero_as_perfect():
    from pyutilz.llm.openrouter_provider._health import _summarize_endpoints

    out = _summarize_endpoints([
        {"provider_name": "a", "uptime_last_30m": 1.0},
        {"provider_name": "b", "uptime_last_30m": 0.98},
    ])
    assert out["endpoints"][0]["uptime_30m"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# F58 -- token_counter encoding cache
# ---------------------------------------------------------------------------


def test_f58_encoding_cache_survives_concurrent_mutation():
    """popitem() at the cap could raise KeyError between the len() check and the pop, surfacing
    as a spurious failure inside a token-budget computation."""
    from pyutilz.llm import token_counter as tc

    if not tc._HAS_TIKTOKEN:
        pytest.skip("tiktoken not installed")
    errors = []

    def worker(n):
        try:
            for i in range(200):
                tc._encoding_for_model(f"model-{n}-{i}")
        except Exception as exc:  # noqa: BLE001 -- the point of the test is that nothing escapes
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(tc._encoding_cache) <= tc._ENCODING_CACHE_MAX_SIZE


# ---------------------------------------------------------------------------
# F12 / F41 / F59 -- text.similarity
# ---------------------------------------------------------------------------

from pyutilz.text.similarity import normalize_sentence, sentences_similarity


def test_f12_exact_suffix_match_scores_like_an_exact_prefix_match():
    """`range(L-S)` never tested the SUFFIX window, so MADRID vs REALMADRID scored 0.633."""
    assert sentences_similarity(["MADRID"], ["REALMADRID"]) == pytest.approx(0.80)
    assert sentences_similarity(["bcd"], ["abcd"]) == pytest.approx(0.875)


def test_f41_single_character_overlap_no_longer_scores_above_0_9():
    """The prefix rule fired before the min-length guard, so a one-letter token outscored a
    correct suffix match against any word starting with that letter."""
    assert sentences_similarity(["A"], ["ANDERSON"]) < 0.9


def test_f41_genuine_multi_character_prefix_still_scores_high():
    assert sentences_similarity(["REAL"], ["REALLYBIGCLUB"]) > 0.9


def test_f59_normalize_sentence_is_order_stable():
    """`list(set(...))` varied with PYTHONHASHSEED, and the greedy matcher's >= tie-break is
    order-sensitive, so the same two names could score differently in two processes."""
    assert normalize_sentence("NEW YORK NEW YORK") == ["NEW", "YORK"]
    assert normalize_sentence("ZULU ALPHA MIKE") == ["ZULU", "ALPHA", "MIKE"]


def test_f59_normalize_sentence_drops_empty_tokens():
    assert "" not in normalize_sentence("")


# ---------------------------------------------------------------------------
# F13 -- text entropy
# ---------------------------------------------------------------------------

from pyutilz.text.strings.textentropy import compute_entropy_stats


def test_f13_raw_entropy_is_never_negative():
    """Normalizing by the number of DISTINCT prefixes made each "probability" exceed 1, flipping
    the log2 term's sign; Shannon entropy is never negative."""
    raw, _rate = compute_entropy_stats("abcabcabcabc", 0)
    assert raw >= 0.0


def test_f13_first_order_raw_entropy_matches_the_analytic_value():
    raw, _rate = compute_entropy_stats("abcabcabcabc", 1)
    assert raw == pytest.approx(1.5726236638951638)


# ---------------------------------------------------------------------------
# F14 / F62 -- secret scrubbing
# ---------------------------------------------------------------------------

from pyutilz.text.secrets_scrub import redact_secrets, sanitize_dsn


@pytest.mark.parametrize(
    "dsn",
    [
        "redis://:sup3rs3cret@localhost:6379/0",
        "postgresql://:sup3rs3cret@db.internal:5432/prod",
        "amqp://:sup3rs3cret@rabbit",
    ],
)
def test_f14_empty_username_dsn_is_redacted(dsn):
    """Empty-user is the standard Redis/valkey password-only URL form, and both functions
    returned it verbatim."""
    assert "sup3rs3cret" not in sanitize_dsn(dsn)
    assert "sup3rs3cret" not in redact_secrets(dsn)


def test_f14_basic_auth_header_is_redacted():
    """TELEGRAM_TOKEN_RE knew only Bearer, so the base64 user:pass was printed in full."""
    out = redact_secrets("Authorization: Basic dXNlcjpwYXNz")
    assert "dXNlcjpwYXNz" not in out


def test_f14_bearer_header_still_redacted():
    assert redact_secrets("Send Bearer ghp_secret to authenticate.") == "Send Bearer *** to authenticate."


def test_f62_two_dsns_in_one_message_both_survive_masked():
    """The greedy `.+` anchored to the LAST @ deleted the second DSN, the connective text and the
    first host, then attributed the surviving host to the first user."""
    out = sanitize_dsn("a postgres://u:p1@h1/d and postgres://u2:p2@h2/d")
    assert "h1" in out and "h2" in out and " and " in out
    assert "p1" not in out and "p2" not in out


def test_f62_at_containing_password_is_still_fully_masked():
    assert sanitize_dsn("postgresql://user:p@ss@host/db") == "postgresql://user:***@host/db"


# ---------------------------------------------------------------------------
# F37 -- json_pg_dumps
# ---------------------------------------------------------------------------

from pyutilz.text.strings.jsonutils import json_pg_dumps

_BS = chr(92)


def test_f37_literal_backslash_u0000_text_survives():
    """Editing the SERIALIZED text could not tell the escape from a value that legitimately
    contains those six characters, so a Windows path or regex crashed the serializer."""
    value = "path" + _BS + "u0000literal"
    assert json_pg_dumps({"a": value}).adapted == {"a": value}


def test_f37_doubled_backslash_variant_survives():
    value = "p" + _BS * 2 + "u0000"
    assert json_pg_dumps({"a": value}).adapted == {"a": value}


def test_f37_real_nul_is_still_stripped():
    assert json_pg_dumps({"a": "x" + chr(0) + "y"}).adapted == {"a": "xy"}


def test_f37_nul_in_a_key_is_stripped_too():
    assert json_pg_dumps({"k" + chr(0): 1}).adapted == {"k": 1}


def test_f37_nonfinite_floats_still_become_null():
    assert json_pg_dumps({"a": float("inf")}).adapted == {"a": None}


# ---------------------------------------------------------------------------
# F38 / F39 -- webtext
# ---------------------------------------------------------------------------

from pyutilz.text.strings.webtext import remove_videos, unescape_html


@pytest.mark.parametrize("value", ["", None])
def test_f38_remove_videos_returns_its_input_for_falsy_text(value):
    """It fell off the `if text:` branch and returned None, which clean_description then
    dereferenced as a string."""
    assert remove_videos(value) == value


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_f38_clean_description_handles_blank_input(value):
    """Whitespace-only rows are routine for scraped text (fix_html strips them to ""), and one
    blank listing aborted the whole batch with a TypeError."""
    from pyutilz.text.strings.webtext import clean_description

    assert clean_description(value) == ""


def test_f39_named_and_numeric_entities_are_decoded():
    """xml.sax.saxutils.unescape knew only &amp;/&lt;/&gt;, leaving the most frequent entities in
    scraped HTML untouched."""
    out = unescape_html("Caf&eacute; &amp; bar &#39;x&#39; &nbsp;")
    assert "&eacute;" not in out
    assert "&#39;" not in out
    assert "&nbsp;" not in out
    assert "&" in out


def test_f39_quot_is_decoded():
    assert unescape_html("&quot;hi&quot;") == '"hi"'


# ---------------------------------------------------------------------------
# F40 / F63 -- tokenizers capitalization flags
# ---------------------------------------------------------------------------


def test_f40_min_morpheme_length_above_one_does_not_crash(monkeypatch):
    """FIRSTLETTER_CAPITAL/ALLLETTERS_CAPITAL were assigned only inside the `j == 1` branch, so
    starting the loop at j=2 raised NameError on the first word."""
    nltk = pytest.importorskip("nltk")
    from pyutilz.text.tokenizers import AdvancedTokenizer

    tok = AdvancedTokenizer()
    monkeypatch.setattr(type(tok), "MIN_MORPHEME_LENGTH", 2, raising=False)
    monkeypatch.setattr(nltk, "sent_tokenize", lambda text: [text])
    monkeypatch.setattr(nltk, "word_tokenize", lambda text: text.split())
    tok.tokenize("Hello World")
    assert sum(tok.NUM_FIRSTLETTER_CAPITAL.values()) > 0


def test_f63_capitalization_flags_describe_the_word_not_the_offset(monkeypatch):
    """Both flags were recomputed at every start offset, so "aBCDE" -- neither capitalized nor
    all-caps -- contributed to both counters."""
    nltk = pytest.importorskip("nltk")
    from pyutilz.text.tokenizers import AdvancedTokenizer

    tok = AdvancedTokenizer()
    monkeypatch.setattr(nltk, "sent_tokenize", lambda text: [text])
    monkeypatch.setattr(nltk, "word_tokenize", lambda text: text.split())
    tok.tokenize("aBCDE")
    assert sum(tok.NUM_FIRSTLETTER_CAPITAL.values()) == 0
    assert sum(tok.NUM_ALLLETTERS_CAPITAL.values()) == 0


# ---------------------------------------------------------------------------
# F60 -- find_between
# ---------------------------------------------------------------------------

from pyutilz.text.strings.basics import find_between


def test_f60_explicit_zero_end_index_means_an_empty_window():
    """`if not idx2` conflated an explicit 0 with "unset" and searched the whole string."""
    assert find_between("abcXdefYghi", "X", "Y", 0, 0) is None


def test_f60_nonzero_window_still_works():
    assert find_between("abcXdefYghi", "X", "Y") == "def"
    assert find_between("abcXdefYghi", "X", "Y", 0, 5) is None


def test_f60_empty_start_honours_idx1():
    """`p1 = 0` discarded idx1 and returned content from before the requested start."""
    assert find_between("abcXdef", "", "X", 3) == ""
    assert find_between("abcXdef", "", "X") == "abc"


# ---------------------------------------------------------------------------
# F61 -- fix_dashes
# ---------------------------------------------------------------------------

from pyutilz.text.humanizer import fix_dashes


def test_f61_multi_space_runs_far_from_a_dash_are_preserved():
    """An unconditional collapse destroyed aligned tables, indented code and any ASCII layout in
    a document that contained no dash at all."""
    assert fix_dashes("Column A    Column B") == "Column A    Column B"


def test_f61_spaces_around_a_replacement_dash_are_still_collapsed():
    assert fix_dashes("a  —  b") == "a - b"
    assert fix_dashes("a – b") == "a - b"
