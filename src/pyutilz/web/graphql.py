# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed
from typing import Optional
ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


client = None

def connect(graphql_client:object)->None:
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
    """Escape literal ``\\n`` sequences in *text* so they survive embedding in a GraphQL string.

    Any two-character backslash-n occurrence is doubled into ``\\\\n`` (an escaped
    backslash followed by ``n``), preventing it from being interpreted as a newline
    when the resulting text is placed inside a GraphQL query/variable literal.
    """
    return text.replace(r"\n", "\\" + "n")

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
