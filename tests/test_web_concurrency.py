"""Regression test for module-level scraping-state races in pyutilz.web.web.

Spins up multiple threads driving get_url()/get_new_session()/set_proxy() concurrently,
with the actual network call mocked out, and asserts no exceptions are raised and the
num_ip_queries counter ends up consistent with the number of successful mocked requests.
This is a best-effort race exerciser (not a guarantee of absence of races), but it drives
the locked critical sections under real thread contention.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("requests")
pytest.importorskip("grequests")

from pyutilz.web import web as web_module  # noqa: E402


class _FakeResponse:
    status_code = 200
    text = "ok"

    def json(self):
        return {}


@pytest.fixture(autouse=True)
def _reset_web_state():
    web_module.init_vars()
    web_module.was_blocked = False
    web_module.cur_max_ip_queries = -1
    web_module.max_ip_queries = 0
    web_module.proxy_server = None
    web_module.proxy_user = None
    web_module.proxy_pass = None
    web_module.proxy_min_port = None
    web_module.proxy_max_port = None
    web_module.proxy_port = None
    web_module.proxy_type = None
    web_module.delay = 0
    yield
    web_module.init_vars()


def _fake_session():
    sess = MagicMock()
    sess.get = MagicMock(return_value=_FakeResponse())
    sess.post = MagicMock(return_value=_FakeResponse())
    return sess


def test_get_url_concurrent_no_corruption():
    """Many threads calling get_url() concurrently should not raise and num_ip_queries
    should end up equal to the total number of successful mocked requests issued."""
    n_threads = 8
    n_calls_per_thread = 20
    errors = []

    with patch.object(web_module, "requests") as mock_requests:
        mock_requests.Session.side_effect = _fake_session

        def worker():
            try:
                for _ in range(n_calls_per_thread):
                    web_module.get_url("http://example.invalid/", max_retries=1, b_use_proxy=False, verbose=False)
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    assert not errors, f"Concurrent get_url() raised: {errors}"
    assert web_module.num_ip_queries == n_threads * n_calls_per_thread


def test_set_proxy_concurrent_no_corruption():
    """Many threads calling set_proxy() concurrently with distinct field sets should not
    raise, and the resulting module state must be a consistent (non-torn) combination of
    fields from exactly one of the calls."""
    n_threads = 10
    errors = []

    def worker(i):
        try:
            web_module.set_proxy(
                m_proxy_user=f"user{i}",
                m_proxy_pass=f"pass{i}",
                m_proxy_server=f"server{i}.example.invalid",
                m_proxy_min_port=20000 + i,
                m_proxy_max_port=20000 + i,
                m_proxy_port=20000 + i,
                m_proxy_type="http",
            )
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent set_proxy() raised: {errors}"
    # Consistency check: the trailing numeric suffix on proxy_user/proxy_pass/proxy_server
    # must all agree - i.e. no interleaved (torn) writes across the tuple assignment.
    assert web_module.proxy_user is not None
    idx = web_module.proxy_user.replace("user", "")
    assert web_module.proxy_pass == f"pass{idx}"
    assert web_module.proxy_server == f"server{idx}.example.invalid"


def test_get_new_session_concurrent_no_corruption():
    """Many threads calling get_new_session() concurrently should not raise, and afterwards
    num_ip_queries must be reset to 0 (matching whichever session call landed last)."""
    n_threads = 10
    errors = []

    with patch.object(web_module, "requests") as mock_requests:
        mock_requests.Session.side_effect = _fake_session

        def worker():
            try:
                web_module.get_new_session(b_random_ua=True, b_use_proxy=False)
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    assert not errors, f"Concurrent get_new_session() raised: {errors}"
    assert web_module.num_ip_queries == 0


def test_get_new_session_closes_previous_session():
    """Regression (2026-07-21 audit round 2): the previous requests.Session (owning its own
    urllib3 connection pool) was dropped with no .close() on every rotation, leaking open
    sockets/connection-pool state without bound over a long-running scraper."""
    with patch.object(web_module, "requests") as mock_requests:
        mock_requests.Session.side_effect = _fake_session

        web_module.get_new_session(b_random_ua=False, b_use_proxy=False)
        first_sess = web_module.sess

        web_module.get_new_session(b_random_ua=False, b_use_proxy=False)

    first_sess.close.assert_called_once()


def test_init_vars_closes_current_session():
    """Regression: init_vars() (called by web.connect(), including FileMaker's two-calls-per-
    authentication pattern) used to drop the current Session with no .close() either."""
    with patch.object(web_module, "requests") as mock_requests:
        mock_requests.Session.side_effect = _fake_session
        web_module.get_new_session(b_random_ua=False, b_use_proxy=False)

    current_sess = web_module.sess
    web_module.init_vars()

    current_sess.close.assert_called_once()


def test_get_url_concurrent_proxy_rotation_on_error_no_corruption():
    """Regression (2026-07-21 audit round 2, HIGH): the except-branch's `proxies =
    get_new_smartproxy(...)` assignment used to write the module-global directly with no lock,
    unlike every other write to `proxies` in this file -- two threads hitting a bad-proxy
    exception around the same time could race, and whichever assignment landed last silently
    discarded the other thread's proxy rotation. Asserts the final `proxies` value is a
    genuine (non-torn) result from exactly one thread's get_new_smartproxy() call."""
    n_threads = 8
    errors = []
    web_module.proxy_server = "server.example.invalid"
    web_module.proxy_user = "user"
    web_module.proxy_pass = "pass"

    def fake_get_new_smartproxy(proxy_user, proxy_pass, proxy_server, *args, **kwargs):
        # Tag the returned proxy dict with the calling thread's identity so a torn/mixed
        # result would be detectable (e.g. one field from thread A, another from thread B).
        tid = threading.get_ident()
        return {"https": f"http://proxy-{tid}:1"}

    with patch.object(web_module, "requests") as mock_requests, patch.object(web_module, "get_new_smartproxy", side_effect=fake_get_new_smartproxy):
        session = _fake_session()
        session.get.side_effect = ConnectionError("proxy connection broken")
        mock_requests.Session.side_effect = lambda: session

        def worker():
            try:
                web_module.get_url("http://example.invalid/", max_retries=1, b_use_proxy=True, verbose=False)
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    assert not errors, f"Concurrent get_url() raised: {errors}"
    # The final `proxies` dict must be a complete, single-thread's dict -- not a torn mix.
    assert web_module.proxies is not None
    assert web_module.proxies["https"].startswith("http://proxy-")
    assert web_module.sess is not None
