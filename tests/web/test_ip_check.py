"""Behaviour of ``pyutilz.web.proxy.ip_check.parse_ip_response``: a JSON body yields its origin (or ip) field, the first of a
comma-separated list; a bare IP in plain text is returned; and anything that is not an IP, such as an HTML error page or a
JSON body without a usable field, is refused as ``"?"`` so a dead proxy is never reported as a working one."""

from __future__ import annotations

import pytest

from pyutilz.web.proxy import ip_check


class TestJsonBodies:
    def test_the_origin_field_is_returned(self):
        assert ip_check.parse_ip_response('{"origin": "1.2.3.4"}') == "1.2.3.4"

    def test_the_first_of_a_comma_separated_origin_is_returned(self):
        assert ip_check.parse_ip_response('{"origin": "1.2.3.4, 5.6.7.8"}') == "1.2.3.4"

    def test_the_ip_field_is_used_when_origin_is_absent(self):
        assert ip_check.parse_ip_response('{"ip": "9.9.9.9"}') == "9.9.9.9"

    def test_a_json_body_without_a_usable_field_is_unknown(self):
        assert ip_check.parse_ip_response('{"origin": null}') == "?"


class TestPlainBodies:
    def test_a_bare_ip_with_a_newline_is_returned(self):
        assert ip_check.parse_ip_response("9.10.11.12\n") == "9.10.11.12"

    def test_an_html_error_page_is_unknown_not_the_exit_ip(self):
        assert ip_check.parse_ip_response("<html>503 Service Unavailable</html>") == "?"

    @pytest.mark.parametrize("body", ["", "not an ip", "999.1.1"])
    def test_non_ip_text_is_unknown(self, body):
        assert ip_check.parse_ip_response(body) == "?"
