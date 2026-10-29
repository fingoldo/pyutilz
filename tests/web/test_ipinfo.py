"""Behaviour of ``pyutilz.web.web.ipinfo.IpGeolocation``: a geolocation result carries the country code and name, while the
continent, IP and provider default to empty so a provider that does not report them still builds a valid result."""

from __future__ import annotations

from pyutilz.web.web import ipinfo


def test_the_required_country_fields_are_set():
    result = ipinfo.IpGeolocation(country_code="DE", country_name="Germany")
    assert (result.country_code, result.country_name) == ("DE", "Germany")


def test_the_optional_fields_default_to_empty():
    result = ipinfo.IpGeolocation(country_code="DE", country_name="Germany")
    assert (result.continent, result.ip, result.provider) == ("", "", "")
