"""Regression tests for the 2026-09-02 database / web / cloud / LLM / text domain audit.

Each test targets one finding and is written to fail on the pre-fix source. No test touches a
real database, a real network endpoint, or a real LLM provider.
"""

import asyncio
import re
from unittest.mock import Mock, patch
import pytest
from pyutilz.database.db.upsert import build_upsert_query
import pyutilz.database.db as dbmod
import pyutilz.database.psycopg2_pool as poolmod
from datetime import datetime, timedelta, timezone
import pyutilz.web.web as webmod
import pyutilz.web.browser as browsermod
from pyutilz.web.cached_client import CachedHttpClient
from pyutilz.web.proxy.base import PortHealthTracker
from pyutilz.web.proxy import decodo as decodomod
from pyutilz.llm.base import LLMProvider
from pyutilz.llm.exceptions import JSONParsingError

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


@pytest.fixture()
def recorded_execute(monkeypatch):
    """Capture every safe_execute call instead of touching a database."""
    calls = []

    def _fake(statement, data=None, *a, **kw):
        calls.append((statement, data))
        return []

    monkeypatch.setattr(dbmod, "safe_execute", _fake)
    return calls


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
            return datetime.now(timezone.utc).replace(tzinfo=None)

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
    failed = {key: datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)}
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
    failed = {key: datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=90)}
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
        # Reframed for the 2026-09-03 F46 fix: total failure is now reported as False, not None
        # -- the point this test makes (a broken stream must not look like a success) is stronger
        # under the new contract, since success is now an explicit True.
        assert webmod.download_to_file("http://x/f", str(out), max_attempts=2) is False
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
    # The point is a FINITE, positive timeout reaching requests -- `is not None` alone would also
    # be satisfied by a 0 (which requests rejects) or by a non-numeric sentinel.
    passed = captured["timeout"]
    assert isinstance(passed, (int, float, tuple))
    seconds = max(passed) if isinstance(passed, tuple) else passed
    assert 0 < float(seconds) < 3600


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

    class _LaunchedError(Exception):
        """Sentinel raised in place of actually starting a browser process."""

    added = []
    monkeypatch.setattr(browsermod, "undetectable", False)
    monkeypatch.setattr(browsermod, "user_agent", None, raising=False)
    monkeypatch.setattr(browsermod, "data_dir", None, raising=False)
    monkeypatch.setattr(browsermod, "proxy_server", {"PROXY_HOST": "h", "PROXY_PORT": 1234, "PROXY_PASS": None})
    monkeypatch.setattr(webdriver, "ChromeOptions", lambda: Mock(add_argument=added.append, add_extension=added.append))
    monkeypatch.setattr(webdriver, "Chrome", Mock(side_effect=_LaunchedError))

    with pytest.raises(_LaunchedError):
        browsermod.start_selenium()
    assert any("--proxy-server=h:1234" in str(a) for a in added)


# ---------------------------------------------------------------------------
# F22 / F23 -- CachedHttpClient
# ---------------------------------------------------------------------------

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
