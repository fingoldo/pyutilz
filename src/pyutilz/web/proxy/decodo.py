"""Decodo (formerly Smartproxy) proxy provider.

Supports residential, mobile, and datacenter proxies with sticky-session
port rotation and built-in API access for subscription & traffic stats.

API docs: https://help.decodo.com/reference/get-subscriptions

Usage::

    from pyutilz.web.proxy import DecodoProvider

    proxy = DecodoProvider.from_env()          # reads PROXY_* + DECODO_API_KEY from env
    url = proxy.proxy_url()                    # random healthy port
    proxy.report_error(42)                     # mark port 42 as problematic

    # Subscription & traffic
    subs = proxy.get_subscriptions()
    for s in subs:
        print(s.service_type, s.used_gb, "/", s.limit_gb)

    traffic = proxy.get_traffic(days=7, group_by="day")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .base import PortHealthTracker, ProxyConfig, ProxyProvider
from pyutilz.web.exceptions import ProxyConfigurationError, ProxyFetchError

_log = logging.getLogger(__name__)

__all__ = [
    "DecodoProvider",
    "DecodoSubscription",
    "DecodoTrafficRow",
    "DecodoTrafficReport",
]

API_BASE = "https://api.decodo.com"

# Proxy types recognized by the Decodo API
PROXY_TYPES = (
    "residential_proxies",
    "mobile_proxies",
    "shared_dc_proxies",
    "rtc_universal_proxies",
    "rtc_universal_core_proxies",
)

# Decodo endpoints this account may exit through, as returned by their locations API.
#
# The country is selected by MODIFYING THE PROXY USERNAME rather than by a separate setting --
# `user-<account>-country-us-city-new_york` exits New York, a bare `<account>` exits somewhere
# random -- so `country_iso` here is the exact segment to splice into the username, city and all.
#
# `random_port` gives a new exit IP per request; the sticky range holds one IP for the life of a
# session, which is what a multi-request flow behind a WAF wants.
#
# NOT ALL OF THESE ARE SAFE FOR EVERY CALLER. The list is the account's full inventory, and it
# includes Russian and Belarusian exits; anything scraping a site that geo-blocks those must
# filter the pool rather than rotate blindly across it. Kept complete here because this module
# describes what the ACCOUNT has -- deciding what a given job may use is the caller's business.
ALLOWED_LOCATIONS = [{"domain":"gate.decodo.com","country_iso":"country-us-city-new_york","random_port":10000,"sticky_port_first":10001,"sticky_port_last":21049,"country_name":"New York"},{"domain":"gate.decodo.com","country_iso":"country-us-city-los_angeles","random_port":10000,"sticky_port_first":10001,"sticky_port_last":21099,"country_name":"Los Angeles"},{"domain":"gate.decodo.com","country_iso":"country-us-city-chicago","random_port":10000,"sticky_port_first":10001,"sticky_port_last":21149,"country_name":"Chicago"},{"domain":"gate.decodo.com","country_iso":"country-us-city-houston","random_port":10000,"sticky_port_first":10001,"sticky_port_last":21199,"country_name":"Houston"},{"domain":"gate.decodo.com","country_iso":"country-us-city-miami","random_port":10000,"sticky_port_first":10001,"sticky_port_last":21249,"country_name":"Miami"},{"domain":"gate.decodo.com","country_iso":"country-gb-city-london","random_port":10000,"sticky_port_first":10001,"sticky_port_last":21299,"country_name":"London"},{"domain":"gate.decodo.com","country_iso":"country-th-city-bangkok","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bangkok"},{"domain":"gate.decodo.com","country_iso":"country-in-city-guwahati","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Guwahati"},{"domain":"gate.decodo.com","country_iso":"country-in-city-kochi","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Kochi"},{"domain":"gate.decodo.com","country_iso":"country-in-city-bhubaneswar","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bhubaneswar"},{"domain":"gate.decodo.com","country_iso":"country-ro-city-bucharest","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bucharest"},{"domain":"gate.decodo.com","country_iso":"country-br-city-belo_horizonte","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Belo Horizonte"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-quezon_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Quezon City"},{"domain":"gate.decodo.com","country_iso":"country-br-city-campinas","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Campinas"},{"domain":"gate.decodo.com","country_iso":"country-in-city-ludhiana","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Ludhiana"},{"domain":"gate.decodo.com","country_iso":"country-lt-city-vilnius","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Vilnius"},{"domain":"gate.decodo.com","country_iso":"country-vn-city-ho_chi_minh_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Ho Chi Minh City"},{"domain":"gate.decodo.com","country_iso":"country-lk-city-colombo","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Colombo"},{"domain":"gate.decodo.com","country_iso":"country-vn-city-hanoi","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Hanoi"},{"domain":"gate.decodo.com","country_iso":"country-pe-city-lima","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Lima"},{"domain":"gate.decodo.com","country_iso":"country-br-city-curitiba","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Curitiba"},{"domain":"gate.decodo.com","country_iso":"country-cz-city-prague","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Prague"},{"domain":"gate.decodo.com","country_iso":"country-uz-city-tashkent","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Tashkent"},{"domain":"gate.decodo.com","country_iso":"country-co-city-bogota","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bogota"},{"domain":"gate.decodo.com","country_iso":"country-tw-city-taipei","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Taipei"},{"domain":"gate.decodo.com","country_iso":"country-in-city-new_delhi","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"New Delhi"},{"domain":"gate.decodo.com","country_iso":"country-br-city-brasilia","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Brasilia"},{"domain":"gate.decodo.com","country_iso":"country-za-city-johannesburg","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Johannesburg"},{"domain":"gate.decodo.com","country_iso":"country-id-city-surabaya","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Surabaya"},{"domain":"gate.decodo.com","country_iso":"country-br-city-porto_alegre","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Porto Alegre"},{"domain":"gate.decodo.com","country_iso":"country-id-city-bandung","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bandung"},{"domain":"gate.decodo.com","country_iso":"country-ng-city-lagos","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Lagos"},{"domain":"gate.decodo.com","country_iso":"country-ru-city-moscow","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Moscow"},{"domain":"gate.decodo.com","country_iso":"country-hu-city-budapest","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Budapest"},{"domain":"gate.decodo.com","country_iso":"country-in-city-chandigarh","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Chandigarh"},{"domain":"gate.decodo.com","country_iso":"country-kw-city-kuwait_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Kuwait City"},{"domain":"gate.decodo.com","country_iso":"country-in-city-indore","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Indore"},{"domain":"gate.decodo.com","country_iso":"country-br-city-fortaleza","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Fortaleza"},{"domain":"gate.decodo.com","country_iso":"country-kr-city-seoul","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Seoul"},{"domain":"gate.decodo.com","country_iso":"country-bg-city-sofia","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Sofia"},{"domain":"gate.decodo.com","country_iso":"country-br-city-recife","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Recife"},{"domain":"gate.decodo.com","country_iso":"country-hk-city-central","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Central"},{"domain":"gate.decodo.com","country_iso":"country-id-city-medan","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Medan"},{"domain":"gate.decodo.com","country_iso":"country-za-city-pretoria","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Pretoria"},{"domain":"gate.decodo.com","country_iso":"country-br-city-salvador","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Salvador"},{"domain":"gate.decodo.com","country_iso":"country-id-city-semarang","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Semarang"},{"domain":"gate.decodo.com","country_iso":"country-kh-city-phnom_penh","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Phnom Penh"},{"domain":"gate.decodo.com","country_iso":"country-in-city-coimbatore","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Coimbatore"},{"domain":"gate.decodo.com","country_iso":"country-pt-city-lisbon","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Lisbon"},{"domain":"gate.decodo.com","country_iso":"country-bd-city-dhaka","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Dhaka"},{"domain":"gate.decodo.com","country_iso":"country-mn-city-ulan_bator","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Ulan Bator"},{"domain":"gate.decodo.com","country_iso":"country-pl-city-warsaw","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Warsaw"},{"domain":"gate.decodo.com","country_iso":"country-it-city-milan","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Milan"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-talavera","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Talavera"},{"domain":"gate.decodo.com","country_iso":"country-gh-city-accra","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Accra"},{"domain":"gate.decodo.com","country_iso":"country-in-city-aizawl","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Aizawl"},{"domain":"gate.decodo.com","country_iso":"country-tw-city-taichung","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Taichung"},{"domain":"gate.decodo.com","country_iso":"country-in-city-nagpur","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Nagpur"},{"domain":"gate.decodo.com","country_iso":"country-in-city-raipur","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Raipur"},{"domain":"gate.decodo.com","country_iso":"country-ua-city-kyiv","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Kyiv"},{"domain":"gate.decodo.com","country_iso":"country-fr-city-paris","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Paris"},{"domain":"gate.decodo.com","country_iso":"country-rs-city-belgrade","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Belgrade"},{"domain":"gate.decodo.com","country_iso":"country-ma-city-casablanca","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Casablanca"},{"domain":"gate.decodo.com","country_iso":"country-tw-city-kaohsiung","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Kaohsiung City"},{"domain":"gate.decodo.com","country_iso":"country-in-city-shimla","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Shimla"},{"domain":"gate.decodo.com","country_iso":"country-br-city-belem","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Belem"},{"domain":"gate.decodo.com","country_iso":"country-lv-city-riga","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Riga"},{"domain":"gate.decodo.com","country_iso":"country-sg-city-singapore","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Singapore"},{"domain":"gate.decodo.com","country_iso":"country-my-city-petaling_jaya","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Petaling Jaya"},{"domain":"gate.decodo.com","country_iso":"country-in-city-surat","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Surat"},{"domain":"gate.decodo.com","country_iso":"country-zm-city-lusaka","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Lusaka"},{"domain":"gate.decodo.com","country_iso":"country-ke-city-nairobi","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Nairobi"},{"domain":"gate.decodo.com","country_iso":"country-za-city-durban","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Durban"},{"domain":"gate.decodo.com","country_iso":"country-fi-city-helsinki","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Helsinki"},{"domain":"gate.decodo.com","country_iso":"country-tw-city-new_taipei_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"New Taipei"},{"domain":"gate.decodo.com","country_iso":"country-br-city-goiania","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Goiania"},{"domain":"gate.decodo.com","country_iso":"country-do-city-santo_domingo","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Santo Domingo"},{"domain":"gate.decodo.com","country_iso":"country-br-city-manaus","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Manaus"},{"domain":"gate.decodo.com","country_iso":"country-co-city-medellin","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Medellin"},{"domain":"gate.decodo.com","country_iso":"country-pk-city-islamabad","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Islamabad"},{"domain":"gate.decodo.com","country_iso":"country-eg-city-cairo","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Cairo"},{"domain":"gate.decodo.com","country_iso":"country-kg-city-bishkek","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bishkek"},{"domain":"gate.decodo.com","country_iso":"country-ge-city-tbilisi","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Tbilisi"},{"domain":"gate.decodo.com","country_iso":"country-ng-city-port_harcourt","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Port Harcourt"},{"domain":"gate.decodo.com","country_iso":"country-dz-city-algiers","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Algiers"},{"domain":"gate.decodo.com","country_iso":"country-az-city-baku","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Baku"},{"domain":"gate.decodo.com","country_iso":"country-tr-city-istanbul","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Istanbul"},{"domain":"gate.decodo.com","country_iso":"country-md-city-chisinau","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Chisinau"},{"domain":"gate.decodo.com","country_iso":"country-gr-city-athens","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Athens"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-batangas","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Batangas"},{"domain":"gate.decodo.com","country_iso":"country-cl-city-concepcion","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Concepcion"},{"domain":"gate.decodo.com","country_iso":"country-in-city-jodhpur","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Jodhpur"},{"domain":"gate.decodo.com","country_iso":"country-es-city-madrid","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Madrid"},{"domain":"gate.decodo.com","country_iso":"country-in-city-kanpur","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Kanpur"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-davao_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Davao City"},{"domain":"gate.decodo.com","country_iso":"country-ng-city-katsina","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Katsina"},{"domain":"gate.decodo.com","country_iso":"country-jm-city-kingston","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Kingston"},{"domain":"gate.decodo.com","country_iso":"country-si-city-ljubljana","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Ljubljana"},{"domain":"gate.decodo.com","country_iso":"country-sa-city-riyadh","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Riyadh"},{"domain":"gate.decodo.com","country_iso":"country-ar-city-cordoba","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Cordoba"},{"domain":"gate.decodo.com","country_iso":"country-pk-city-lahore","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Lahore"},{"domain":"gate.decodo.com","country_iso":"country-co-city-barranquilla","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Barranquilla"},{"domain":"gate.decodo.com","country_iso":"country-ru-city-st_petersburg","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"St Petersburg"},{"domain":"gate.decodo.com","country_iso":"country-in-city-thrissur","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Thrissur"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-cebu_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Cebu City"},{"domain":"gate.decodo.com","country_iso":"country-id-city-makassar","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Makassar"},{"domain":"gate.decodo.com","country_iso":"country-za-city-cape_town","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Cape Town"},{"domain":"gate.decodo.com","country_iso":"country-br-city-florianopolis","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Florianopolis"},{"domain":"gate.decodo.com","country_iso":"country-in-city-hisar","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Hisar"},{"domain":"gate.decodo.com","country_iso":"country-hr-city-zagreb","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Zagreb"},{"domain":"gate.decodo.com","country_iso":"country-es-city-barcelona","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Barcelona"},{"domain":"gate.decodo.com","country_iso":"country-be-city-brussels","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Brussels"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-bacolod_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bacolod City"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-general_trias","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"General Trias"},{"domain":"gate.decodo.com","country_iso":"country-kr-city-gangnam_gu","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Gangnam Gu"},{"domain":"gate.decodo.com","country_iso":"country-in-city-malappuram","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Malappuram"},{"domain":"gate.decodo.com","country_iso":"country-in-city-koch_bihar","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Koch Bihar"},{"domain":"gate.decodo.com","country_iso":"country-ru-city-krasnodar","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Krasnodar"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-iloilo_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Iloilo City"},{"domain":"gate.decodo.com","country_iso":"country-al-city-tirana","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Tirana"},{"domain":"gate.decodo.com","country_iso":"country-in-city-varanasi","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Varanasi"},{"domain":"gate.decodo.com","country_iso":"country-jo-city-amman","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Amman"},{"domain":"gate.decodo.com","country_iso":"country-by-city-minsk","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Minsk"},{"domain":"gate.decodo.com","country_iso":"country-it-city-rome","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Rome"},{"domain":"gate.decodo.com","country_iso":"country-sv-city-san_salvador","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"San Salvador"},{"domain":"gate.decodo.com","country_iso":"country-co-city-santiago_de_cali","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Santiago De Cali"},{"domain":"gate.decodo.com","country_iso":"country-in-city-sikar","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Sikar"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-caloocan_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Caloocan City"},{"domain":"gate.decodo.com","country_iso":"country-ae-city-dubai","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Dubai"},{"domain":"gate.decodo.com","country_iso":"country-br-city-cuiaba","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Cuiaba"},{"domain":"gate.decodo.com","country_iso":"country-ru-city-yekaterinburg","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Yekaterinburg"},{"domain":"gate.decodo.com","country_iso":"country-in-city-bhopal","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bhopal"},{"domain":"gate.decodo.com","country_iso":"country-us-city-dallas","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Dallas"},{"domain":"gate.decodo.com","country_iso":"country-qa-city-doha","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Doha"},{"domain":"gate.decodo.com","country_iso":"country-dz-city-oran","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Oran"},{"domain":"gate.decodo.com","country_iso":"country-il-city-tel_aviv","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Tel Aviv"},{"domain":"gate.decodo.com","country_iso":"country-my-city-puchong_batu_dua_belas","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Puchong Batu Dua Belas"},{"domain":"gate.decodo.com","country_iso":"country-ie-city-dublin","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Dublin"},{"domain":"gate.decodo.com","country_iso":"country-de-city-berlin","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Berlin"},{"domain":"gate.decodo.com","country_iso":"country-sk-city-bratislava","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Bratislava"},{"domain":"gate.decodo.com","country_iso":"country-my-city-kuching","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Kuching"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-tuguegarao_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Tuguegarao City"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-cagayan_de_oro","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Cagayan De Oro"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-makati_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Makati City"},{"domain":"gate.decodo.com","country_iso":"country-mu-city-port_louis","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Port Louis"},{"domain":"gate.decodo.com","country_iso":"country-mk-city-skopje","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Skopje"},{"domain":"gate.decodo.com","country_iso":"country-sa-city-jeddah","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Jeddah"},{"domain":"gate.decodo.com","country_iso":"country-uy-city-montevideo","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Montevideo"},{"domain":"gate.decodo.com","country_iso":"country-pa-city-panama_city","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Panama City"},{"domain":"gate.decodo.com","country_iso":"country-id-city-yogyakarta","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Yogyakarta"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-carmona","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Carmona"},{"domain":"gate.decodo.com","country_iso":"country-pl-city-katowice","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Katowice"},{"domain":"gate.decodo.com","country_iso":"country-nl-city-rotterdam","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Rotterdam"},{"domain":"gate.decodo.com","country_iso":"country-et-city-addis_ababa","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Addis Ababa"},{"domain":"gate.decodo.com","country_iso":"country-ph-city-butuan","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Butuan"},{"domain":"gate.decodo.com","country_iso":"country-at-city-vienna","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Vienna"},{"domain":"gate.decodo.com","country_iso":"country-bh-city-manama","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Manama"},{"domain":"gate.decodo.com","country_iso":"country-pl-city-poznan","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Poznan"},{"domain":"gate.decodo.com","country_iso":"country-in-city-rajkot","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Rajkot"},{"domain":"gate.decodo.com","country_iso":"country-py-city-asuncion","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Asuncion"},{"domain":"gate.decodo.com","country_iso":"country-br-city-uberlandia","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Uberlandia"},{"domain":"gate.decodo.com","country_iso":"country-pk-city-karachi","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Karachi"},{"domain":"gate.decodo.com","country_iso":"country-tr-city-ankara","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Ankara"},{"domain":"gate.decodo.com","country_iso":"country-pt-city-porto","random_port":10000,"sticky_port_first":10001,"sticky_port_last":49999,"country_name":"Porto"}]


# ── API data classes ────────────────────────────────────────────────────────

@dataclass
class DecodoSubscription:
    """Parsed subscription info from ``/v2/subscriptions``."""

    service_type: str
    traffic_limit_gb: float
    traffic_used_gb: float
    valid_from: str
    valid_until: str
    users_limit: int
    ip_address_limit: int
    raw: Dict[str, Any]

    @property
    def remaining_gb(self) -> float:
        """Remaining traffic allowance in GB (limit minus used)."""
        return self.traffic_limit_gb - self.traffic_used_gb

    @property
    def usage_pct(self) -> float:
        """Percentage of the traffic limit used so far (0.0 if the limit is 0)."""
        return (self.traffic_used_gb / self.traffic_limit_gb * 100) if self.traffic_limit_gb > 0 else 0.0

    def summary(self) -> str:
        """Render a human-readable multi-line summary with a progress bar of usage."""
        bar_len = 30
        filled = int(bar_len * self.usage_pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        return (
            f"{self.service_type}  "
            f"{self.valid_from} .. {self.valid_until}\n"
            f"  Limit: {_fmt_gb(self.traffic_limit_gb)}  "
            f"Used: {_fmt_gb(self.traffic_used_gb)}  "
            f"Remaining: {_fmt_gb(self.remaining_gb)}\n"
            f"  [{bar}] {self.usage_pct:.1f}%"
        )

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "DecodoSubscription":
        """Build a :class:`DecodoSubscription` from a raw ``/v2/subscriptions`` API response item."""
        return cls(
            service_type=data.get("service_type", "unknown"),
            traffic_limit_gb=_safe_float(data.get("traffic_limit")),
            traffic_used_gb=_safe_float(data.get("traffic_per_period")),
            valid_from=data.get("valid_from", "?"),
            valid_until=data.get("valid_until", "?"),
            users_limit=_safe_int(data.get("users_limit", 0)),
            ip_address_limit=_safe_int(data.get("ip_address_limit", 0)),
            raw=data,
        )


@dataclass
class DecodoTrafficRow:
    """Single row from traffic statistics."""

    group_key: str
    requests: int
    traffic_bytes: float

    @property
    def traffic_gb(self) -> float:
        """Traffic for this row converted from bytes to GB."""
        return self.traffic_bytes / (1024**3)


@dataclass
class DecodoTrafficReport:
    """Parsed traffic statistics response."""

    rows: List[DecodoTrafficRow]
    total_requests: int
    total_bytes: float

    @property
    def total_gb(self) -> float:
        """Total traffic across all rows converted from bytes to GB."""
        return self.total_bytes / (1024**3)

    def summary(self, group_by: str = "day") -> str:
        """Render a human-readable table of the traffic rows plus a totals line.

        ``group_by`` names what one row IS (the grouping the report was fetched with) and titles the first
        column accordingly - the rows arrive already grouped upstream, so it labels rather than regroups.
        """
        lines = [
            f"  {group_by.capitalize():<25} {'Requests':>12} {'Traffic':>12}",
            f"  {'-' * 25} {'-' * 12} {'-' * 12}",
        ]
        for row in self.rows:
            traffic_str = _fmt_gb(row.traffic_gb) if row.traffic_bytes > 1_000_000 else f"{row.traffic_bytes:,.0f} B"
            lines.append(f"  {row.group_key:<25} {row.requests:>12,} {traffic_str:>12}")
        lines.append(f"  {'-' * 25} {'-' * 12} {'-' * 12}")
        total_str = _fmt_gb(self.total_gb) if self.total_bytes > 1_000_000 else f"{self.total_bytes:,.0f} B"
        lines.append(f"  {'TOTAL':<25} {self.total_requests:>12,} {total_str:>12}")
        return "\n".join(lines)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> float:
    """Coerce ``val`` to float, returning 0.0 on any conversion failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(val: Any, default: int = 0) -> int:
    """Coerce ``val`` to int, returning ``default`` on any conversion failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _fmt_gb(gb: float) -> str:
    """Format ``gb`` as a human-readable string, in GB if >=1.0 else in MB."""
    if gb >= 1.0:
        return f"{gb:,.2f} GB"
    return f"{gb * 1024:,.1f} MB"


# ── Provider ────────────────────────────────────────────────────────────────

class DecodoProvider(ProxyProvider):
    """Decodo (Smartproxy) sticky-session proxy provider.

    Each port offset (added to ``base_port``) maps to a different exit IP
    that persists for the provider's sticky session TTL (~10 min for Decodo
    residential).  The health tracker auto-bans ports that produce repeated
    connection errors, forcing IP rotation.

    Parameters
    ----------
    config
        Proxy credentials and endpoint.
    api_key
        Decodo API key for subscription/traffic queries.  Optional —
        proxy routing works without it.
    health_tracker
        Shared :class:`PortHealthTracker` instance, or a new one is created.
    """

    def __init__(
        self,
        config: ProxyConfig,
        *,
        api_key: str = "",
        health_tracker: Optional[PortHealthTracker] = None,
    ) -> None:
        super().__init__(config, health_tracker=health_tracker)
        self.api_key = api_key

    @classmethod
    def from_env(
        cls,
        *,
        user_var: str = "PROXY_USER",
        pass_var: str = "PROXY_PASS",
        host_var: str = "PROXY_HOST",
        port_var: str = "PROXY_PORT",
        range_var: str = "PROXY_PORT_RANGE",
        api_key_var: str = "DECODO_API_KEY",
        default_range: int = 500,
        health_tracker: Optional[PortHealthTracker] = None,
    ) -> "DecodoProvider":
        """Create provider from environment variables.

        Required: ``PROXY_USER``, ``PROXY_PASS``, ``PROXY_HOST``, ``PROXY_PORT``.
        Optional: ``PROXY_PORT_RANGE`` (default 500), ``DECODO_API_KEY``.
        """
        missing = [v for v in [user_var, pass_var, host_var, port_var] if v not in os.environ]
        if missing:
            raise OSError(f"Missing required env vars: {', '.join(missing)}")

        try:
            base_port = int(os.environ[port_var])
        except ValueError:
            raise ValueError(f"{port_var} must be an integer, got '{os.environ[port_var]}'")

        try:
            # Same treatment as port_var above: a bare int() failure names neither the variable
            # nor the offending value, which an operator cannot map back to their environment.
            port_range = int(os.environ.get(range_var, default_range))
        except ValueError:
            raise ValueError(f"{range_var} must be an integer, got '{os.environ.get(range_var)}'")

        config = ProxyConfig(
            user=os.environ[user_var],
            password=os.environ[pass_var],
            host=os.environ[host_var],
            base_port=base_port,
            port_range=port_range,
        )
        return cls(config, api_key=os.environ.get(api_key_var, ""), health_tracker=health_tracker)

    def proxy_url(self, port_offset: Optional[int] = None) -> str:
        """Return proxy URL with sticky-session port.

        If *port_offset* is ``None``, picks a random healthy port.
        """
        offset = port_offset if port_offset is not None else self.pick_port()
        c = self.config
        return f"{c.protocol}://{c.user}:{c.password}@{c.host}:{c.base_port + offset}"

    # ── Decodo API ──────────────────────────────────────────────────────────

    def _api_headers(self) -> Dict[str, str]:
        """Build the Authorization/Content-Type headers for Decodo API requests; raises if ``api_key`` is unset."""
        if not self.api_key:
            raise ProxyConfigurationError("DECODO_API_KEY not set. " "Get it from: dashboard.decodo.com -> Settings -> API Keys")
        return {"Authorization": self.api_key, "Content-Type": "application/json"}

    def get_subscriptions(self) -> List[DecodoSubscription]:
        """Fetch subscription info from Decodo API.

        Tries ``/v2/subscriptions`` first, falls back to ``/v2/sub-users``.
        Not all account types support all endpoints.
        """
        import requests

        headers = self._api_headers()
        for endpoint in ["/v2/subscriptions", "/v2/sub-users"]:
            try:
                r = requests.get(f"{API_BASE}{endpoint}", headers=headers, timeout=15)
                r.raise_for_status()
                data = r.json()
                items = data if isinstance(data, list) else [data]
                return [DecodoSubscription.from_api(item) for item in items]
            except requests.RequestException:  # noqa: PERF203 -- per-iteration fault isolation is intentional (try the next endpoint)
                # Regression fix: only requests.HTTPError (raised by raise_for_status() on a
                # non-2xx response) was caught here -- a Timeout/ConnectionError on the FIRST
                # endpoint (a transient network blip, not an "endpoint unsupported" signal)
                # previously propagated uncaught instead of falling through to try the second
                # endpoint, even though the second endpoint might well have succeeded.
                continue
        raise ProxyFetchError("Could not fetch subscriptions from any Decodo API endpoint")

    def get_endpoints(self) -> Dict[str, Any]:
        """Fetch available proxy endpoints from ``/v2/endpoints``.

        Returns dict with ``"random"`` and ``"sticky"`` keys, each mapping
        to a list of location entries (hostname, port_range, etc.).
        """
        import requests

        headers = self._api_headers()
        result: Dict[str, Any] = {}
        with requests.Session() as session:
            session.headers.update(headers)
            r = session.get(f"{API_BASE}/v2/endpoints", timeout=15)
            r.raise_for_status()
            for item in r.json():
                ep_type = item.get("type", "unknown")
                sub_url = f"{API_BASE}/v2/endpoints/{item.get('url', '').split('/')[-1]}"
                try:
                    sub_r = session.get(sub_url, timeout=15)
                except requests.RequestException as e:
                    _log.warning("Skipping endpoint %s (%s): request failed: %s", ep_type, sub_url, e)
                    continue
                if sub_r.status_code == 200:
                    result[ep_type] = sub_r.json()
                else:
                    _log.warning("Skipping endpoint %s (%s): status %s", ep_type, sub_url, sub_r.status_code)
        return result

    def get_traffic(
        self,
        *,
        proxy_type: str = "residential_proxies",
        days: int = 0,
        start: Optional[str] = None,
        end: Optional[str] = None,
        group_by: str = "day",
        limit: int = 500,
        sort_order: str = "desc",
    ) -> DecodoTrafficReport:
        """Fetch traffic statistics from ``/api/v2/statistics/traffic``.

        Parameters
        ----------
        proxy_type
            One of :data:`PROXY_TYPES`.
        days
            If > 0, compute *start*/*end* as ``[now - days, now]``.
        start, end
            Explicit date strings (``"YYYY-MM-DD HH:MM:SS"``).
        group_by
            ``"day"`` | ``"target"`` | ``"country"`` | ``"protocol"`` |
            ``"hour"`` | ``"week"`` | ``"month"``.
        """
        import requests

        now = datetime.now(timezone.utc)
        if days > 0:
            start = (now - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
            end = now.strftime("%Y-%m-%d %H:%M:%S")
        elif start is None or end is None:
            raise ValueError("Provide either days>0 or explicit start/end")

        # Paginate: page 1 alone silently truncated the report at `limit` rows and reported the
        # partial sum as the account total, which a quota/billing decision would be made on.
        all_rows: List[Any] = []
        page = 1
        while page <= _MAX_TRAFFIC_PAGES:
            body: dict = {
                "proxyType": proxy_type,
                "startDate": start,
                "endDate": end,
                "groupBy": group_by,
                "limit": limit,
                "page": page,
                "sortBy": "grouping_key",
                "sortOrder": sort_order,
            }
            r = requests.post(
                f"{API_BASE}/api/v2/statistics/traffic",
                headers=self._api_headers(),
                json=body,
                timeout=30,
            )
            r.raise_for_status()
            page_rows = _extract_traffic_rows(r.json())
            all_rows.extend(page_rows)
            if len(page_rows) < limit:
                break
            page += 1
        else:
            _log.warning("get_traffic: stopped after %d pages of %d rows; the report may still be truncated", _MAX_TRAFFIC_PAGES, limit)
        return _parse_traffic_response(all_rows, group_by)

    def print_usage(
        self,
        *,
        days: int = 0,
        group_by: str = "day",
        proxy_type: str = "residential_proxies",
    ) -> None:
        """Print human-readable subscription + traffic summary to stdout."""
        try:
            subs = self.get_subscriptions()
            for sub in subs:
                print(f"\n{'=' * 55}")
                print(sub.summary())
                print(f"{'=' * 55}")
        except Exception as e:
            _log.exception("Error fetching subscriptions: %s", e)
            print(f"  Error fetching subscriptions: {e}")

        if days > 0:
            print(f"\nTraffic ({days}d, grouped by {group_by}):")
            try:
                report = self.get_traffic(proxy_type=proxy_type, days=days, group_by=group_by)
                print(report.summary(group_by))
            except Exception as e:
                _log.exception("Error fetching traffic: %s", e)
                print(f"  Error fetching traffic: {e}")


# Safety stop so a server that never returns a short page cannot loop forever.
_MAX_TRAFFIC_PAGES = 100


def _extract_traffic_rows(data: Any) -> List[Any]:
    """Return the row list out of one traffic-API response payload."""
    raw_rows = data if isinstance(data, list) else data.get("data", data.get("results", []))
    if not isinstance(raw_rows, list):
        return []
    return raw_rows


def _parse_traffic_response(data: Any, group_by: str) -> DecodoTrafficReport:
    """Parse the traffic API response (or an already-flattened row list) into a :class:`DecodoTrafficReport`."""
    raw_rows = _extract_traffic_rows(data)

    rows: List[DecodoTrafficRow] = []
    total_reqs = 0
    total_bytes = 0.0
    for row in raw_rows:
        key = row.get("grouping_key", row.get(group_by, "?"))
        reqs = _safe_int(row.get("requests", 0))
        traffic = _safe_float(row.get("totals", row.get("traffic", 0)))
        rows.append(DecodoTrafficRow(group_key=str(key), requests=reqs, traffic_bytes=traffic))
        total_reqs += reqs
        total_bytes += traffic

    return DecodoTrafficReport(rows=rows, total_requests=total_reqs, total_bytes=total_bytes)


# Exit countries available on the DATACENTER endpoint (dc.decodo.com).
#
# Separate from ALLOWED_LOCATIONS above, and not derivable from it, because the two products
# differ in what they accept. Verified against this account's live endpoint:
#
#     user-<account>-country-br                      at dc.decodo.com  ->  BR / Sao Paulo
#     user-<account>-country-br-city-belo_horizonte  at dc.decodo.com  ->  ProxyError
#
# City-level targeting is a residential-gateway feature; on datacenter only the country works.
# ALLOWED_LOCATIONS spells every entry with a city and names gate.decodo.com, so a datacenter
# caller must use THIS list rather than stripping cities off that one.
#
# The leading `random` entry is a real option, not a placeholder: omitting the country segment
# entirely gives an unpinned exit, which is the default state and something a rotation may want
# to return to.
ALLOWED_COUNTRIES = [
    {"country": "random", "name": "Random"},
    {"country": "us", "name": "USA"},
    {"country": "ca", "name": "Canada"},
    {"country": "gb", "name": "GB"},
    {"country": "de", "name": "Germany"},
    {"country": "fr", "name": "France"},
    {"country": "es", "name": "Spain"},
    {"country": "it", "name": "Italy"},
    {"country": "se", "name": "Sweden"},
    {"country": "gr", "name": "Greece"},
    {"country": "pt", "name": "Portugal"},
    {"country": "nl", "name": "Netherlands"},
    {"country": "be", "name": "Belgium"},
    {"country": "ru", "name": "Russia"},
    {"country": "ua", "name": "Ukraine"},
    {"country": "pl", "name": "Poland"},
    {"country": "il", "name": "Israel"},
    {"country": "tr", "name": "Turkey"},
    {"country": "au", "name": "Australia"},
    {"country": "my", "name": "Malaysia"},
    {"country": "th", "name": "Thailand"},
    {"country": "kr", "name": "South Korea"},
    {"country": "jp", "name": "Japan"},
    {"country": "ph", "name": "Philippines"},
    {"country": "sg", "name": "Singapore"},
    {"country": "cn", "name": "China"},
    {"country": "hk", "name": "Hong Kong"},
    {"country": "tw", "name": "Taiwan"},
    {"country": "in", "name": "India"},
    {"country": "pk", "name": "Pakistan"},
    {"country": "ir", "name": "Iran"},
    {"country": "id", "name": "Indonesia"},
    {"country": "az", "name": "Azerbaijan"},
    {"country": "kz", "name": "Kazakhstan"},
    {"country": "ae", "name": "UAE"},
    {"country": "mx", "name": "Mexico"},
    {"country": "br", "name": "Brazil"},
    {"country": "ar", "name": "Argentina"},
    {"country": "cl", "name": "Chile"},
    {"country": "pe", "name": "Peru"},
    {"country": "ec", "name": "Ecuador"},
    {"country": "co", "name": "Colombia"},
    {"country": "za", "name": "South Africa"},
    {"country": "eg", "name": "Egypt"},
    {"country": "ao", "name": "Angola"},
    {"country": "cm", "name": "Cameroon"},
    {"country": "cf", "name": "Central African Republic"},
    {"country": "td", "name": "Chad"},
    {"country": "bj", "name": "Benin"},
    {"country": "et", "name": "Ethiopia"},
    {"country": "dj", "name": "Djibouti"},
    {"country": "gm", "name": "Gambia"},
    {"country": "gh", "name": "Ghana"},
    {"country": "ke", "name": "Kenya"},
    {"country": "lr", "name": "Liberia"},
    {"country": "mg", "name": "Madagascar"},
    {"country": "ml", "name": "Mali"},
    {"country": "mr", "name": "Mauritania"},
    {"country": "mu", "name": "Mauritius"},
    {"country": "ma", "name": "Morocco"},
    {"country": "mz", "name": "Mozambique"},
    {"country": "ng", "name": "Nigeria"},
    {"country": "sn", "name": "Senegal"},
    {"country": "sl", "name": "Sierra Leone"},
    {"country": "sc", "name": "Seychelles"},
    {"country": "zw", "name": "Zimbabwe"},
    {"country": "ss", "name": "South Sudan"},
    {"country": "sd", "name": "Sudan"},
    {"country": "tg", "name": "Togo"},
    {"country": "tn", "name": "Tunisia"},
    {"country": "ug", "name": "Uganda"},
    {"country": "zm", "name": "Zambia"},
    {"country": "af", "name": "Afghanistan"},
    {"country": "bh", "name": "Bahrain"},
    {"country": "bd", "name": "Bangladesh"},
    {"country": "bt", "name": "Bhutan"},
    {"country": "mm", "name": "Myanmar"},
    {"country": "kh", "name": "Cambodia"},
    {"country": "iq", "name": "Iraq"},
    {"country": "jo", "name": "Jordan"},
    {"country": "lb", "name": "Lebanon"},
    {"country": "mv", "name": "Maldives"},
    {"country": "mn", "name": "Mongolia"},
    {"country": "om", "name": "Oman"},
    {"country": "qa", "name": "Qatar"},
    {"country": "sa", "name": "Saudi Arabia"},
    {"country": "tm", "name": "Turkmenistan"},
    {"country": "uz", "name": "Uzbekistan"},
    {"country": "ye", "name": "Yemen"},
    {"country": "al", "name": "Albania"},
    {"country": "ad", "name": "Andorra"},
    {"country": "at", "name": "Austria"},
    {"country": "am", "name": "Armenia"},
    {"country": "ba", "name": "Bosnia and Herzegovina"},
    {"country": "bg", "name": "Bulgaria"},
    {"country": "by", "name": "Belarus"},
    {"country": "hr", "name": "Croatia"},
    {"country": "cy", "name": "Cyprus"},
    {"country": "cz", "name": "Czech Republic"},
    {"country": "dk", "name": "Denmark"},
    {"country": "ee", "name": "Estonia"},
    {"country": "fi", "name": "Finland"},
    {"country": "ge", "name": "Georgia"},
    {"country": "hu", "name": "Hungary"},
    {"country": "is", "name": "Iceland"},
    {"country": "ie", "name": "Ireland"},
    {"country": "lv", "name": "Latvia"},
    {"country": "li", "name": "Liechtenstein"},
    {"country": "lt", "name": "Lithuania"},
    {"country": "lu", "name": "Luxembourg"},
    {"country": "mc", "name": "Monaco"},
    {"country": "md", "name": "Moldova"},
    {"country": "me", "name": "Montenegro"},
    {"country": "no", "name": "Norway"},
    {"country": "ro", "name": "Romania"},
    {"country": "rs", "name": "Serbia"},
    {"country": "sk", "name": "Slovakia"},
    {"country": "si", "name": "Slovenia"},
    {"country": "ch", "name": "Switzerland"},
    {"country": "mk", "name": "Macedonia"},
    {"country": "bs", "name": "Bahamas"},
    {"country": "bz", "name": "Belize"},
    {"country": "vg", "name": "British Virgin Islands"},
    {"country": "cr", "name": "Costa Rica"},
    {"country": "cu", "name": "Cuba"},
    {"country": "dm", "name": "Dominica"},
    {"country": "ht", "name": "Haiti"},
    {"country": "hn", "name": "Honduras"},
    {"country": "jm", "name": "Jamaica"},
    {"country": "aw", "name": "Aruba"},
    {"country": "pa", "name": "Panama"},
    {"country": "pr", "name": "Puerto Rico"},
    {"country": "tt", "name": "Trinidad and Tobago"},
    {"country": "fj", "name": "Fiji"},
    {"country": "nz", "name": "New Zealand"},
    {"country": "bo", "name": "Bolivia"},
    {"country": "py", "name": "Paraguay"},
    {"country": "uy", "name": "Uruguay"},
    {"country": "ci", "name": "C\xf4te d'Ivoire"},
    {"country": "sy", "name": "Syria"},
    {"country": "vn", "name": "Vietnam"},
    {"country": "mt", "name": "Malta"},
    {"country": "eu", "name": "Europe"},
]
