# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed
ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

client=None

def connect(graphql_client:object)->bool:
    global client
    client = graphql_client

def execute(query:str,variables:dict=None)->dict:
    if client:
        try:
            res=client.graphql(query,variables=variables)
            if res is not None: return res.to_dict()
        except Exception as e:
            logger.exception(e)

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
    return text.replace(r"\n", "\\" + "n")
    
def beautify_gql_query(query: str, join_token_find: str = "}\n}", join_token_replace: str = "}}") -> str:
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
