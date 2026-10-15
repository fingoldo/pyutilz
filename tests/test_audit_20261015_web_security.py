"""Regression tests for the 2026-10-15 audit security/resource findings (X-01, X-02, R-05, X-06, X-12,
X-14, X-15, X-16, A-04, A-05).

Each test fails on the pre-fix code. No network is used: HTTP clients and the webdriver are faked.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest

from pyutilz.core import filemaker
from pyutilz.database import deltalakes
from pyutilz.system import monitoring
from pyutilz.text import tokenizers
from pyutilz.web import browser
from pyutilz.web.exceptions import UnsafeURLError
import pyutilz.web.web as web_pkg


# --------------------------------------------------------------------------------------------------------------------
# X-01: heartbeat failure logs must not carry the api_key that healthchecks.io / cronitor.io embed in the URL path
# --------------------------------------------------------------------------------------------------------------------
def test_x01_heartbeat_error_log_does_not_leak_api_key(monkeypatch, caplog):
    secret = "SECRETKEY-7f3a91"  # pragma: allowlist secret

    class _FakeRequests:
        @staticmethod
        def post(url, **kwargs):
            raise ConnectionError(f"HTTPSConnectionPool(host='hc-ping.com'): Max retries exceeded with url: /{secret}/job/0/0")

    monkeypatch.setattr(monitoring, "requests", _FakeRequests)
    with caplog.at_level("WARNING"):
        ok = monitoring._attempt(f"https://hc-ping.com/{secret}/job/0/0", None, None, "healthchecks.io", "job")
    assert ok is False  # transient error: still retryable
    assert secret not in caplog.text
    assert "ConnectionError" in caplog.text


# --------------------------------------------------------------------------------------------------------------------
# X-02 / R-05: start_selenium must not expose a CDP debugging port, and must remove the proxy zip on failure too
# --------------------------------------------------------------------------------------------------------------------
class _FakeOptions:
    def __init__(self) -> None:
        self.arguments: list = []
        self.extensions: list = []

    def add_argument(self, arg: str) -> None:
        self.arguments.append(arg)

    def add_extension(self, path: str) -> None:
        self.extensions.append(path)


class _FailingWebdriver:
    last_kwargs: dict = {}

    @classmethod
    def Chrome(cls, **kwargs):  # mirrors selenium's webdriver.Chrome name
        cls.last_kwargs = kwargs
        raise RuntimeError("driver failed to start")


def _make_fake_chrome_options(opts: _FakeOptions):
    def _create(kwargs):
        return _FailingWebdriver, opts

    return _create


def test_x02_start_selenium_passes_no_remote_debugging_flag(monkeypatch):
    opts = _FakeOptions()
    monkeypatch.setattr(browser, "_create_chrome_options", _make_fake_chrome_options(opts))
    monkeypatch.setattr(browser, "target", "https://example.invalid", raising=False)
    monkeypatch.setattr(browser, "proxy_server", None)
    with pytest.raises(RuntimeError):
        browser.start_selenium()
    assert not [a for a in opts.arguments if "remote-debugging" in a]


def test_r05_proxy_plugin_zip_removed_when_chrome_start_fails(monkeypatch):
    opts = _FakeOptions()
    created: list = []

    def _fake_configure_proxy(options, proxy_server):
        fd, path = tempfile.mkstemp(suffix=".zip", prefix="pyutilz_test_plugin_")
        os.close(fd)
        created.append(path)
        return path

    monkeypatch.setattr(browser, "_create_chrome_options", _make_fake_chrome_options(opts))
    monkeypatch.setattr(browser, "_configure_proxy", _fake_configure_proxy)
    monkeypatch.setattr(browser, "proxy_server", {"PROXY_PASS": "pw"})
    monkeypatch.setattr(browser, "target", "https://example.invalid", raising=False)
    with pytest.raises(RuntimeError):
        browser.start_selenium()
    assert created, "proxy plugin stub was not created"
    assert not os.path.exists(created[0]), "proxy-auth zip must be removed even when Chrome() raises"


def test_r05_atexit_hook_registered_once(monkeypatch):
    import atexit

    registered: list = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn))
    monkeypatch.setattr(browser, "_proxy_plugin_atexit_registered", False)
    monkeypatch.setattr(browser, "_PROXY_PLUGIN_FILES", set())
    for _ in range(3):
        path = browser._add_authenticated_proxy(_FakeOptions(), {"PROXY_PASS": "pw", "PROXY_USER": "u", "PROXY_HOST": "h", "PROXY_PORT": "1"})
        browser._remove_proxy_plugin(path)
    assert registered.count(browser._remove_leftover_proxy_plugins) == 1


# --------------------------------------------------------------------------------------------------------------------
# X-06: load_tokens must refuse a tampered dump when verify_sidecar=True
# --------------------------------------------------------------------------------------------------------------------
def test_x06_load_tokens_verify_sidecar_rejects_tampered_file(tmp_path):
    from pyutilz.core.safe_pickle import PickleVerificationError

    base = str(tmp_path / "tok")
    tokenizers.AdvancedTokenizer.save_tokens_to_file(SimpleNamespace(NUM_OCCS=1), base, desired_vars=["NUM_OCCS"])
    assert os.path.exists(base + ".pickle.sha256")

    # Round trip with verification on succeeds.
    tokens: dict = {}
    tokenizers.AdvancedTokenizer.load_tokens(SimpleNamespace(), base, tokens, verify_sidecar=True)
    assert tokens == {"NUM_OCCS": 1}

    # Swap the payload for a different one, keep the old sidecar.
    import gzip
    import pickle

    with gzip.open(base + ".pickle", "wb") as handle:
        pickle.dump({"NUM_OCCS": 999}, handle)
    with pytest.raises(PickleVerificationError):
        tokenizers.AdvancedTokenizer.load_tokens(SimpleNamespace(), base, {}, verify_sidecar=True)


# --------------------------------------------------------------------------------------------------------------------
# X-12: delta lock files live in a per-user, mode-0700 directory
# --------------------------------------------------------------------------------------------------------------------
def test_x12_lock_file_lives_in_private_per_user_dir(monkeypatch, tmp_path):
    captured: dict = {}

    class _FakeLock:
        def __init__(self, path):
            captured["path"] = path

        def acquire(self, timeout=None):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(deltalakes, "FileLock", _FakeLock)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    deltalakes.safe_delta_write(str(tmp_path / "tbl"), lambda: None)

    lock_dir = os.path.dirname(captured["path"])
    assert lock_dir == deltalakes._lock_dir()
    assert os.path.basename(lock_dir).startswith("pyutilz-delta-locks-")
    if hasattr(os, "getuid"):
        assert (os.stat(lock_dir).st_mode & 0o777) == 0o700


# --------------------------------------------------------------------------------------------------------------------
# A-04 / A-05: the documented exceptions appear in the docstring
# --------------------------------------------------------------------------------------------------------------------
def test_a04_a05_safe_delta_write_docstring_documents_raises():
    doc = deltalakes.safe_delta_write.__doc__ or ""
    assert "Raises:" in doc
    assert "TimeoutError" in doc
    assert "ImportError" in doc


# --------------------------------------------------------------------------------------------------------------------
# X-14: FileMaker refuses a non-https URL before storing credentials
# --------------------------------------------------------------------------------------------------------------------
def test_x14_filemaker_init_rejects_http_and_stores_nothing(monkeypatch):
    monkeypatch.setattr(filemaker, "filemaker_url", None)
    monkeypatch.setattr(filemaker, "filemaker_username", None)
    monkeypatch.setattr(filemaker, "filemaker_password", None)
    monkeypatch.setattr(filemaker, "get_session_token", lambda **kw: "tok")
    with pytest.raises(ValueError):
        filemaker.init("http://fm.example.invalid", "user", "secret")
    assert filemaker.filemaker_password is None
    assert filemaker.filemaker_url is None


def test_x14_filemaker_init_accepts_https(monkeypatch):
    monkeypatch.setattr(filemaker, "get_session_token", lambda **kw: "tok")
    filemaker.init("https://fm.example.invalid", "user", "secret")
    assert filemaker.filemaker_url == "https://fm.example.invalid"


# --------------------------------------------------------------------------------------------------------------------
# X-15: IP echo providers are queried over https
# --------------------------------------------------------------------------------------------------------------------
def test_x15_ip_providers_are_https():
    assert all(url.startswith("https://") for url in web_pkg.IP_PROVIDERS)


# --------------------------------------------------------------------------------------------------------------------
# X-16: download helpers refuse non-http(s) URLs before any request
# --------------------------------------------------------------------------------------------------------------------
def test_x16_download_to_file_rejects_file_scheme(monkeypatch, tmp_path):
    calls: list = []

    class _FakeRequests:
        @staticmethod
        def get(url, **kwargs):
            calls.append(url)
            raise AssertionError("no request must be made")

    monkeypatch.setattr(web_pkg, "requests", _FakeRequests)
    with pytest.raises(UnsafeURLError):
        web_pkg.download_to_file("file:///etc/passwd", str(tmp_path / "out"))
    assert calls == []


def test_x16_download_in_parallel_reports_non_http_without_requesting(monkeypatch):
    grequests = pytest.importorskip("grequests")
    requested: list = []

    def _fake_get(url, **kwargs):
        requested.append(url)
        return url

    monkeypatch.setattr(grequests, "get", _fake_get)
    monkeypatch.setattr(grequests, "map", lambda rs, size=None: [None for _ in rs])
    errored = web_pkg.download_in_parallel(["file:///etc/passwd", "https://ok.example.invalid/x"], lambda resp, url: None)
    assert "file:///etc/passwd" in errored
    assert requested == ["https://ok.example.invalid/x"]
