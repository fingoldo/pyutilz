"""Logging, introspection text and line-join pins for web.graphql (mutation survivors)."""

from __future__ import annotations

import logging

from pyutilz.web import graphql


class _Result:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


class _Client:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.queries = []

    def graphql(self, query, variables=None):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.result


def test_a_client_failure_is_logged_with_the_exception(monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(graphql, "client", _Client(error=ValueError("endpoint down")))
    assert graphql.execute("{ x }") == {}
    assert "endpoint down" in caplog.text


def test_the_introspection_query_names_the_schema_root_types(monkeypatch):
    client = _Client(result=_Result({"ok": True}))
    monkeypatch.setattr(graphql, "client", client)
    assert graphql.query_schema() == {"ok": True}
    sent = client.queries[0]
    assert "IntrospectionQuery" in sent and "__schema" in sent


def test_beautify_keeps_one_line_break_between_fields():
    assert graphql.beautify_gql_query("a {\n  b\n}") == "a {\nb\n}"
