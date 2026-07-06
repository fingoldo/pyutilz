# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

from ._logproxy import logger

from typing import Any, Iterable, List, Optional, Sequence, Union
import json

def json_serial(obj: Any) -> str:
    from datetime import datetime, date

    """JSON serializer for objects not serializable by default json code. Sample: json.dumps(oProduct,default=json_serial)"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def sub_elem(parent: Any, tag: str, text: Optional[str] = None, attribs: Optional[dict] = None) -> object:
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
    Puts all of the object's properties (ecxept starting with an underscore) into a dictionary
    """
    if exclude is None:
        exclude = []
    import numbers

    res = None
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
        try:
            attribslist = dir(obj)
            res = dict()
            for attr in attribslist:
                if attr not in exclude:
                    if not attr.startswith("_"):
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
            logger.exception(e)
            pass
    return res


def remove_json_attributes(json_obj: dict, attributes: Sequence) -> None:
    if json_obj is None:
        return
    for attr in attributes:
        if attr in json_obj:
            del json_obj[attr]


def leave_json_attributes(json_obj: dict, attributes: Sequence) -> None:
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
    elems: Dict[Any, Any] = {}
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
    return elems  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def remove_json_empty_attributes(json_obj: dict, attributes: Sequence) -> None:
    for attr in attributes:
        if attr in json_obj:
            try:
                if len(json_obj[attr]) == 0:
                    del json_obj[attr]
            except Exception as e:  # nosec B110 - best-effort emptiness check; attr's value may be a non-sized type (int/bool/None) with no len(), which is expected and simply means "don't remove it"
                logger.debug("Attribute %s has no len(), skipping empty-check: %s", attr, e)


def remove_json_defaults(
    json_obj: dict, attr_values: Optional[List[dict]] = None, warn_if_not_default: Optional[bool] = False, obj_id: Optional[str] = ""
) -> None:
    if json_obj is None or attr_values is None:
        return
    for attr, default_value in attr_values.items():
        if attr in json_obj:
            if json_obj.get(attr) == default_value:
                del json_obj[attr]
            else:
                if warn_if_not_default:
                    logger.warning("%s field not equals %s in object %s %s" % (attr, default_value, str(json_obj)[:20], obj_id))


def json_pg_dumps(obj: object, sort_keys: bool = False) -> str:
    # json.loads(json.dumps(obj, default=json_serial).replace(r"\u0000", "").replace("NaN", "null"))
    # orjson is ~5-10x faster than stdlib json on dumps; emits UTF-8 bytes so we
    # decode once before the literal-six-char backslash-u-0000 escape strip (postgres
    # rejects NUL inside jsonb text). Falls back to stdlib only if orjson missing.
    from psycopg2.extras import Json  # lazy: only needed when this fn is actually called

    try:
        import orjson  # type: ignore
        opts = orjson.OPT_SORT_KEYS if sort_keys else 0
        raw = orjson.dumps(obj, default=json_serial, option=opts).decode("utf-8")
    except ImportError:
        raw = json.dumps(obj, default=json_serial, sort_keys=sort_keys)
    return Json(json.loads(raw.replace("\\u0000", "")))  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime  # ,object_pairs_hook=OrderedDict


def get_jsonlist_property(data: Iterable, property_name: str, return_indices: Optional[bool] = False, verbose: Optional[bool] = False) -> list:
    """
    >>>get_jsonlist_property([dict(id=4,name='John'),dict(id=12,name='Jane')],'id')
    [4, 12]
    """
    res = []
    indices = []

    if isinstance(data, dict):
        return data.get(property_name)

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


def get_jsonlist_properties(data: list, property_names: list, verbose: Optional[bool] = False) -> list:
    """
    >>>get_jsonlist_property([dict(id=4,name='John'),dict(id=12,name='Jane')],'id')
    [4, 12]
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
    return res, indices
