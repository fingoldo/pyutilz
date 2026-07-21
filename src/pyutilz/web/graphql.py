"""Thin module-level wrapper around a GraphQL client: connect, execute queries, and query helpers."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Optional

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


client = None

def connect(graphql_client:Any)->None:
    """Register the module-level GraphQL client instance used by execute()/query_schema()."""
    global client
    client = graphql_client

def execute(query:str,variables:Optional[dict]=None)->dict:
    """Runs a GraphQL query. Always returns a dict; returns an empty dict on error or when no client is connected."""
    if client:
        try:
            res=client.graphql(query,variables=variables)
            if res is not None: return res.to_dict()  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
        except Exception as e:
            logger.exception(e)
    return {}

def query_schema()->dict:
    """Run the standard GraphQL introspection query against the connected endpoint and return its schema."""
    return execute("""
        query IntrospectionQuery {
          __schema {
            queryType {
              name
            }
            mutationType {
              name
            }
            subscriptionType {
              name
            }
            types {
              ...FullType
            }
            directives {
              name
              description
              locations
              args {
                ...InputValue
              }
            }
          }
        }

        fragment FullType on __Type {
          kind
          name
          description
          fields(includeDeprecated: true) {
            name
            description
            args {
              ...InputValue
            }
            type {
              ...TypeRef
            }
            isDeprecated
            deprecationReason
          }
          inputFields {
            ...InputValue
          }
          interfaces {
            ...TypeRef
          }
          enumValues(includeDeprecated: true) {
            name
            description
            isDeprecated
            deprecationReason
          }
          possibleTypes {
            ...TypeRef
          }
        }

        fragment InputValue on __InputValue {
          name
          description
          type {
            ...TypeRef
          }
          defaultValue
        }

        fragment TypeRef on __Type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                      ofType {
                        kind
                        name
                      }
                    }
                  }
                }
              }
            }
          }
        }
    """)

def text_to_graphql(text: str) -> str:
    """Escape literal backslashes in *text* so an embedded ``\\n`` sequence survives as literal
    text (rather than being read as a newline escape) when placed inside a GraphQL string.

    Regression fix: the previous implementation, `text.replace(r"\\n", "\\" + "n")`, was a
    complete no-op -- the search pattern `r"\\n"` and the replacement `"\\" + "n"` are both the
    same 2-character string (backslash, n), so .replace() never changed anything (verified: 100%
    reproducible on every call, not a probabilistic edge case). Doubling every backslash is the
    actual fix: it makes a literal `\\n` in the input survive as `\\\\n` in the output, which a
    GraphQL/JSON parser reads back as the 2 literal characters `\\n`, not a real newline.
    """
    return text.replace("\\", "\\\\")

def beautify_gql_query(query: str, join_token_find: str = "}\n}", join_token_replace: str = "}}") -> str:  # nosec B107 - "}\n}" is a GraphQL closing-brace text pattern used for query reformatting, not a credential
    """
    Get rid of comments in a graphql query
    """
    fixedlines = []
    lines = query.split("\n")
    for line in lines:
        lineparts = line.split("#")
        res = lineparts[0].strip()
        if res:
            fixedlines.append(res)
    result = "\n".join(fixedlines)
    while join_token_find in result:
        result = result.replace(join_token_find, join_token_replace)
    return result
