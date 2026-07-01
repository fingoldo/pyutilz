from unittest.mock import MagicMock, patch
import pytest

from pyutilz.web.graphql import text_to_graphql, beautify_gql_query, connect, execute


class TestTextToGraphql:
    def test_basic_replacement(self):
        assert text_to_graphql(r"line1\nline2") == "line1\\nline2"

    def test_no_op_when_no_backslash_n(self):
        assert text_to_graphql("hello world") == "hello world"

    def test_empty_string(self):
        assert text_to_graphql("") == ""


class TestBeautifyGqlQuery:
    def test_removes_comments(self):
        query = "query {\n  field # this is a comment\n}"
        result = beautify_gql_query(query)
        assert "#" not in result
        assert "field" in result

    def test_collapses_closing_braces(self):
        query = "query {\nfield {\nvalue\n}\n}"
        result = beautify_gql_query(query)
        assert "}\n}" not in result
        assert "}}" in result

    def test_strips_whitespace(self):
        query = "  query {  \n   field   \n  }  "
        result = beautify_gql_query(query)
        for line in result.split("\n"):
            assert line == line.strip()

    def test_removes_comment_only_lines(self):
        query = "query {\n# comment only\nfield\n}"
        result = beautify_gql_query(query)
        assert "comment" not in result

    def test_empty_input(self):
        assert beautify_gql_query("") == ""

    def test_multiline(self):
        query = "query {\n  a # x\n  b # y\n}\n}"
        result = beautify_gql_query(query)
        assert "a" in result
        assert "b" in result
        assert "#" not in result


class TestConnect:
    def test_sets_global_client(self):
        mock_client = MagicMock()
        connect(mock_client)
        import pyutilz.web.graphql as mod
        assert mod.client is mock_client


class TestExecute:
    def test_with_mock_client(self):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"data": {"field": 1}}
        mock_client = MagicMock()
        mock_client.graphql.return_value = mock_result

        import pyutilz.web.graphql as mod
        mod.client = mock_client
        result = execute("{ field }")
        assert result == {"data": {"field": 1}}
        mock_client.graphql.assert_called_once_with("{ field }", variables=None)

    def test_with_none_client(self):
        # execute() always returns a dict; no connected client -> empty dict (never None).
        import pyutilz.web.graphql as mod
        mod.client = None
        assert execute("{ field }") == {}

    def test_client_raises_exception(self):
        # A client error is logged and swallowed; execute() returns {} not None.
        mock_client = MagicMock()
        mock_client.graphql.side_effect = RuntimeError("fail")

        import pyutilz.web.graphql as mod
        mod.client = mock_client
        assert execute("{ field }") == {}

    def test_client_returns_none_result(self):
        # graphql() returning None must not crash to_dict(); execute() returns {}.
        mock_client = MagicMock()
        mock_client.graphql.return_value = None

        import pyutilz.web.graphql as mod
        mod.client = mock_client
        assert execute("{ field }") == {}

    def test_query_schema_returns_dict_no_client(self):
        # query_schema() annotated -> dict; with no client it must return {} (not None).
        import pyutilz.web.graphql as mod
        mod.client = None
        result = mod.query_schema()
        assert isinstance(result, dict)
        assert result == {}

    def test_with_variables(self):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"data": "ok"}
        mock_client = MagicMock()
        mock_client.graphql.return_value = mock_result

        import pyutilz.web.graphql as mod
        mod.client = mock_client
        result = execute("query($id: ID!){ node(id: $id) }", variables={"id": "1"})
        assert result == {"data": "ok"}
        mock_client.graphql.assert_called_once_with(
            "query($id: ID!){ node(id: $id) }", variables={"id": "1"}
        )
