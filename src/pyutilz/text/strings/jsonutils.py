"""Helpers for working with JSON-like dicts/lists: serialization, attribute filtering, and extraction."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

from ._logproxy import logger

from typing import Any, Dict, Iterable, Optional, Sequence, Union
import json
import math

# Resolved ONCE at import instead of per call: ``json_pg_dumps`` is the jsonb bulk-insert path, and
# an import statement in its body re-pays a sys.modules lookup plus frame work on every row.
# Annotated `Optional[Any]` rather than left to inference: inferring the module type makes the
# `if _orjson is not None` guards below always-true to the type checker, which types the stdlib
# fallback paths away as unreachable on any box that happens to have orjson installed.
_orjson: Optional[Any]
try:
    import orjson as _orjson_mod
except ImportError:  # orjson is optional -- the stdlib branch below is the fallback
    _orjson = None
else:
    _orjson = _orjson_mod

# The six characters orjson emits for a real NUL inside a string. Postgres rejects a NUL in
# json/jsonb text, so its presence (or, conservatively, that of a legitimately-escaped backslash
# sequence that merely LOOKS like one) is what forces the slow normalize-the-object path.
_ESCAPED_NUL = "\\u0000"

def json_serial(obj: Any) -> str:
    """JSON serializer for objects not serializable by default json code. Sample: json.dumps(oProduct,default=json_serial)"""

    from datetime import datetime, date

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def sub_elem(parent: Any, tag: str, text: Optional[str] = None, attribs: Optional[dict] = None) -> object:
    """Create and append a new XML SubElement under ``parent`` with the given tag, text, and attributes."""
    if attribs is None:
        attribs = {}
    from xml.etree.ElementTree import SubElement  # nosec B405 - only used to CREATE/write new XML elements below (sub_elem builds output, never parses external/untrusted XML), so no XXE parsing risk

    new_elem = SubElement(parent, tag, **attribs)
    if text:
        new_elem.text = text
    return new_elem


def jsonize_atrtributes(
    obj: Any,
    exclude: Optional[list] = None,
    strip: Optional[bool] = True,
    skip_functions: Optional[bool] = True,
    recursion_level: Optional[int] = 0,
    max_recursion_level: Optional[int] = None,
) -> dict:
    """
    Puts all of the object's properties (except starting with an underscore) into a dictionary
    """
    if exclude is None:
        exclude = []
    if recursion_level is None:
        recursion_level = 0
    import numbers

    res: Any = None
    if isinstance(obj, str):
        if strip:
            res = obj.strip()
        else:
            res = obj
    elif isinstance(obj, numbers.Number):
        res = obj
    elif type(obj) in (dict,):
        if max_recursion_level is None or (max_recursion_level is not None and max_recursion_level >= recursion_level):
            res = {}
            for key, value in obj.items():
                res[key] = jsonize_atrtributes(
                    obj=value,
                    exclude=exclude,
                    strip=strip,
                    skip_functions=skip_functions,
                    recursion_level=recursion_level + 1,
                    max_recursion_level=max_recursion_level,
                )
        else:
            res = obj
    elif type(obj) in (list, set, tuple):
        if max_recursion_level is None or (max_recursion_level is not None and max_recursion_level >= recursion_level):
            res = []
            for elem in obj:
                res.append(
                    jsonize_atrtributes(
                        obj=elem,
                        exclude=exclude,
                        strip=strip,
                        skip_functions=skip_functions,
                        recursion_level=recursion_level + 1,
                        max_recursion_level=max_recursion_level,
                    )
                )
        else:
            res = obj
    else:
        attribslist = dir(obj)
        res = dict()
        for attr in attribslist:
            if attr not in exclude:
                if not attr.startswith("_"):
                    try:
                        val = getattr(obj, attr)
                        if skip_functions:
                            if type(val).__name__ == "builtin_function_or_method":
                                continue
                        res[attr] = jsonize_atrtributes(
                            obj=val,
                            exclude=exclude,
                            strip=strip,
                            skip_functions=skip_functions,
                            recursion_level=recursion_level + 1,
                            max_recursion_level=max_recursion_level,
                        )
                    except Exception as e:
                        # debug, not exception/warning: this runs once per attribute name returned by
                        # dir(obj), recursively over the whole object graph -- an object with a large or
                        # deeply-nested attribute surface (or one whose properties routinely raise, e.g. a
                        # lazy/proxy object) would otherwise emit one full traceback PER attribute, per
                        # nesting level, flooding the log for what is normal, expected, per-attribute
                        # skip-and-continue behavior (see the `continue` right below).
                        logger.debug("Failed to jsonize attribute %r of %r: %s", attr, obj, e)
                        continue
    return res  # type: ignore[no-any-return]  # res is genuinely a dict on this path; typed Any to accommodate the str/dict/list branches elsewhere in this function


def remove_json_attributes(json_obj: Optional[dict], attributes: Sequence) -> None:
    """Delete the given ``attributes`` (if present) from ``json_obj`` in place."""
    if json_obj is None:
        return
    for attr in attributes:
        json_obj.pop(attr, None)


def leave_json_attributes(json_obj: Optional[dict], attributes: Sequence) -> None:
    """Delete every key of ``json_obj`` NOT in ``attributes``, in place (inverse of remove_json_attributes)."""
    if json_obj is None:
        return
    for attr in list(json_obj.keys()):
        if attr not in attributes:
            del json_obj[attr]


def extract_json_attribute(json_obj: Optional[Union[dict, list]], attribute: Union[str, list]) -> dict:
    """
    Extracts (if possible) ONE attribute from a dict of dicts and lists

    >>>extract_json_attribute( {'category': {'uid': '531770282580668418', 'prefLabel': 'Web, Mobile & Software Dev'}, 'subcategories': [{'uid': '531770282589057031', 'prefLabel': 'QA & Testing'}], 'oservice': {'uid': '1313512633755545600', 'prefLabel': 'Manual Testing'}},'prefLabel')
    {'category': 'Web, Mobile & Software Dev','subcategories': ['QA & Testing'],'oservice': 'Manual Testing'}

    >>>extract_json_attribute({'category': {'uid': '531770282580668422', 'prefLabel': 'Sales & Marketing'}, 'subcategories': [{'uid': '531770282597445634', 'prefLabel': 'Lead Generation'}], 'oservice': {'uid': '1017484851352698936', 'prefLabel': 'Lead Generation'}},'prefLabel')
    {'category': 'Sales & Marketing','subcategories': ['Lead Generation'],'oservice': 'Lead Generation'}

    >>>extract_json_attribute([{'parentSkillUid': None,
          'freeText': None,
          'skillType': 3,
          'uid': '1052162208894341126',
          'highlighted': False,
          'prettyName': 'Music Video'},
         {'parentSkillUid': None,
          'freeText': None,
          'skillType': 3,
          'uid': '1031626793248342016',
          'highlighted': False,
          'prettyName': 'Videography'},
         {'parentSkillUid': None,
          'freeText': None,
          'skillType': 3,
          'uid': '1031626793223176192',
          'highlighted': False,
          'prettyName': 'Video Editing'}],'prettyName')
    ['Music Video', 'Videography', 'Video Editing']

    """
    if isinstance(attribute, str):
        attribute = [attribute]
    elems: Any = {}
    if isinstance(json_obj, list):

        elems = []
        for elem in json_obj:
            found = False
            for next_attribute in attribute:
                if next_attribute in elem and elem[next_attribute] is not None:
                    elems.append(elem[next_attribute])
                    found = True
                    break

            if not found:
                elems.append(elem)

    elif isinstance(json_obj, dict):
        elems = {}
        for key, item in json_obj.items():
            if isinstance(item, dict):
                found = False
                for next_attribute in attribute:
                    if next_attribute in item and item.get(next_attribute) is not None:
                        elems[key] = item[next_attribute]
                        found = True
                        break
                if not found:
                    elems[key] = item
            elif isinstance(item, str):
                elems[key] = item
            elif isinstance(item, list):
                elems[key] = extract_json_attribute(item, attribute)
    return elems  # type: ignore[no-any-return]  # elems is built from arbitrary decoded JSON


def remove_json_empty_attributes(json_obj: dict, attributes: Sequence) -> None:
    """Delete each of ``attributes`` from ``json_obj`` in place when its value has length 0 (e.g. empty list/str/dict)."""
    for attr in attributes:
        if attr in json_obj:
            try:
                if len(json_obj[attr]) == 0:
                    del json_obj[attr]
            except Exception as e:  # nosec B110 - best-effort emptiness check; attr's value may be a non-sized type (int/bool/None) with no len(), which is expected and simply means "don't remove it"
                logger.debug("Attribute %s has no len(), skipping empty-check: %s", attr, e)


def remove_json_defaults(
    json_obj: dict, attr_values: Optional[Dict[str, Any]] = None, warn_if_not_default: Optional[bool] = False, obj_id: Optional[str] = ""
) -> None:
    """Delete each attribute in ``json_obj`` whose current value equals its default in ``attr_values``, in place.

    When ``warn_if_not_default`` is set, logs a warning for attributes present but not equal to their default
    (instead of deleting them).
    """
    if json_obj is None or attr_values is None:
        return
    for attr, default_value in attr_values.items():
        if attr in json_obj:
            if json_obj.get(attr) == default_value:
                del json_obj[attr]
            else:
                if warn_if_not_default:
                    logger.warning("%s field not equals %s in object %s %s" % (attr, default_value, str(json_obj)[:20], obj_id))


def _normalize_for_pg_json(obj: Any) -> Any:
    """Recursively replace NaN/Infinity/-Infinity floats with None and strip NUL from strings.

    orjson silently serializes them as JSON ``null``; stdlib ``json.dumps`` instead emits the
    non-standard tokens ``NaN``/``Infinity``/``-Infinity``, which postgres's strict json/jsonb
    parser rejects at INSERT time. Applying this canonical policy (always -> None) BEFORE
    dispatching to either backend means ``json_pg_dumps`` behaves identically regardless of
    whether orjson happens to be installed, instead of silently diverging by environment.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        # Strip real NUL characters HERE, on the object. Doing it on the serialized text instead
        # could not tell a postgres-hostile NUL escape from a value that legitimately contains the
        # six literal characters backslash-u-0-0-0-0 -- a code snippet, regex or Windows path --
        # and mangled such a value into unparseable JSON.
        return obj.replace("\x00", "") if "\x00" in obj else obj
    if isinstance(obj, dict):
        return {(_normalize_for_pg_json(k) if isinstance(k, str) else k): _normalize_for_pg_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_pg_json(v) for v in obj]
    return obj


# Kept as the historical name of the above (it only handled floats when it was introduced).
_normalize_nonfinite_floats = _normalize_for_pg_json


class _PreSerializedJson:
    """A psycopg2 ``Json`` adapter whose JSON text is already built.

    ``psycopg2.extras.Json`` serializes ``self.adapted`` at adapt time, so handing it a parsed dict
    means the document is serialized, parsed and serialized AGAIN -- three full passes for one
    INSERT. This subclass carries the finished text and returns it verbatim from ``dumps``, while
    ``adapted`` stays available (parsed LAZILY, once, on first access) for the callers and tests
    that inspect the normalized object.
    """

    _MISSING = object()

    def __init__(self, raw: str):
        self._raw = raw
        self._adapted: Any = self._MISSING
        self._conn: Any = None

    @property
    def adapted(self) -> Any:
        """The normalized python object, parsed from the serialized text on first access."""
        if self._adapted is self._MISSING:
            self._adapted = json.loads(self._raw)
        return self._adapted

    def dumps(self, obj: Any = None) -> str:
        """Return the pre-built JSON text (``obj`` is ignored -- it is what ``adapted`` parses FROM)."""
        return self._raw

    def __conform__(self, proto: Any) -> Any:
        """psycopg2 adaptation hook: this object is its own ``ISQLQuote`` adapter."""
        from psycopg2.extensions import ISQLQuote

        if proto is ISQLQuote:
            return self
        return None

    def prepare(self, conn: Any) -> None:
        """Remember the connection so the quoted literal is encoded with its client encoding."""
        self._conn = conn

    def getquoted(self) -> bytes:
        """The JSON text as a quoted SQL literal."""
        from psycopg2.extensions import QuotedString

        qs = QuotedString(self._raw)
        if self._conn is not None:
            qs.prepare(self._conn)
        return qs.getquoted()  # type: ignore[no-any-return]  # psycopg2 ships no stubs, so QuotedString.getquoted() is Any

    def __str__(self) -> str:
        """The quoted literal as text (mirrors ``psycopg2.extras.Json.__str__``)."""
        return self.getquoted().decode("ascii", "replace")


def json_pg_dumps(obj: object, sort_keys: bool = False) -> object:
    """Serialize ``obj`` to a psycopg2 ``Json``-compatible wrapper for insertion into a jsonb column.

    Uses orjson (falling back to stdlib json if unavailable). NUL characters (which postgres
    rejects inside jsonb text) and NaN/Infinity/-Infinity floats are normalized out, so the output
    is identical regardless of which JSON backend is installed and a value containing the literal
    text ``\\u0000`` survives intact.

    The orjson branch does NOT pre-walk the document. orjson already emits ``null`` for NaN/Inf
    natively -- exactly the policy :func:`_normalize_for_pg_json` enforces -- and NUL is the only
    remaining reason to rebuild the object, so we serialize FIRST and rebuild only when the finished
    text actually contains an escaped-NUL sequence. That inverts the cost: the walk (a full
    pure-python copy of every dict and list, measured 2.98 ms of a 7.4 ms call on a 500-row nested
    payload, plus a second copy of the document in memory) now runs only for the rare NUL-bearing
    document instead of every row. A string that legitimately contains the six characters
    ``\\u0000`` also trips the check -- a false positive costs the slow path, never correctness.
    The stdlib fallback still normalizes unconditionally: it emits the non-standard
    ``NaN``/``Infinity`` tokens postgres rejects, so there the walk is load-bearing.

    The serialized text is handed to the adapter as-is (see :class:`_PreSerializedJson`) instead of
    being re-parsed with stdlib json and then re-serialized a third time at adapt time.
    """
    if _orjson is not None:
        opts = _orjson.OPT_SORT_KEYS if sort_keys else 0
        raw: Optional[str]
        try:
            raw = _orjson.dumps(obj, default=json_serial, option=opts).decode("utf-8")
        except TypeError:
            # orjson refuses shapes the normalizer flattens for it (tuples -> lists, most notably).
            # Fall through to the normalize-then-serialize path rather than raising.
            raw = None
        if raw is not None and _ESCAPED_NUL not in raw:
            return _PreSerializedJson(raw)
        try:
            return _PreSerializedJson(_orjson.dumps(_normalize_for_pg_json(obj), default=json_serial, option=opts).decode("utf-8"))
        except TypeError:
            # orjson rejects shapes stdlib json accepts -- non-str dict keys ("Dict key must be
            # str") and lone surrogates ("surrogates not allowed"). The docstring promises the
            # output does not depend on which backend is installed, so fall through to stdlib
            # rather than raising only on the boxes that happen to have orjson.
            pass
    return _PreSerializedJson(json.dumps(_normalize_for_pg_json(obj), default=json_serial, sort_keys=sort_keys))


def get_jsonlist_property(
    data: Iterable, property_name: str, return_indices: Optional[bool] = False, verbose: Optional[bool] = False
) -> Union[Any, list, tuple]:
    """
    >>>get_jsonlist_property([dict(id=4,name='John'),dict(id=12,name='Jane')],'id')
    [4, 12]

    For dict ``data``, there is no positional-index concept (a dict lookup has no "position"),
    so when ``return_indices=True`` the indices slot is ``None`` rather than an empty/positional list.
    """
    res = []
    indices = []

    if isinstance(data, dict):
        value = data.get(property_name)
        if return_indices:
            return value, None
        return value

    for i, elem in enumerate(data):
        if property_name in elem:
            res.append(elem[property_name])
            indices.append(i)
        else:
            if verbose:
                logger.warning(f"No {property_name} field for element {elem}")

    if return_indices:
        return res, indices
    else:
        return res


def get_jsonlist_properties(data: list, property_names: list, verbose: Optional[bool] = False, return_indices: Optional[bool] = True) -> Union[list, tuple]:
    """
    >>>get_jsonlist_property([dict(id=4,name='John'),dict(id=12,name='Jane')],'id')
    [4, 12]

    Note: unlike the singular ``get_jsonlist_property`` (which defaults ``return_indices=False`` and
    returns a bare list), this function defaults ``return_indices=True`` and returns the ``(list,
    indices)`` tuple form for backward compatibility — existing callers unpack the result positionally
    as ``result, indices = get_jsonlist_properties(...)`` and would silently break if the default here
    were flipped to match the singular function's contract.
    """
    res = []
    indices = []
    for i, elem in enumerate(data):
        new_elem = {}
        for property_name in property_names:
            if property_name in elem:
                new_elem[property_name] = elem[property_name]
            else:
                if verbose:
                    logger.warning(f"No {property_name} field for element {elem}")
        if new_elem:
            res.append(new_elem)
            indices.append(i)
    if return_indices:
        return res, indices
    return res
