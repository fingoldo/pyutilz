"""Operations on plain Python objects, dicts, sequences and their ordering.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

# pylint: disable=unidiomatic-typecheck

from ._common import Any, Iterable, MappingABC, Optional, Sequence, Set, Union, logger, numbers
from .numerics import is_float, to_float

# ----------------------------------------------------------------------------------------------------------------------------
# Operations on objects
# ----------------------------------------------------------------------------------------------------------------------------


def show_methods(obj, uppercased=False):
    """Lists non-dunder attribute names of `obj`.

    If `uppercased` is True, only names starting with an uppercase letter are kept.
    """
    # Dunder means leading AND trailing double underscore; a substring test also dropped
    # name-mangled attributes and ordinary names spelled with an inner "__".
    return [a for a in dir(obj) if not (a.startswith("__") and a.endswith("__")) and (uppercased is False or a[0].isupper())]


# ----------------------------------------------------------------------------------------------------------------------------
# Operations on dicts
# ----------------------------------------------------------------------------------------------------------------------------


def prefix_dict_elems(obj: dict, prefix: str) -> dict:
    """Keys of dict assumed to be string"""
    return {(prefix + key): value for key, value in obj.items()}


def populate_object_from_dict(obj, dct):
    """Populates a class/object with properties from a dictionary."""
    for key in dct:
        setattr(obj, key, dct[key])


def flatten_keys_to_dict(obj: object) -> dict:
    """Recursively walks content of the object, bringing all the key-value keys to the top level dict."""
    res = dict()
    if isinstance(obj, list):
        for subobj in obj:
            res.update(flatten_keys_to_dict(subobj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if type(value) in (list, dict):
                res.update(flatten_keys_to_dict(value))
            else:
                res[key] = value
    return res


def flatten_keys_to_set(
    obj: object,
    dict_merge_symbol: str = ":",
    stringify: bool = False,
    verbose: bool = False,
    max_chars: int = 10,
) -> set:
    """Recursively walks content of an object, bringing all the key-value keys to the top level set."""
    res: Set[Any] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            # print(key,value)
            if isinstance(value, (dict, Iterable)):
                # print('pasing recursively %s' % key)
                res.update(flatten_keys_to_set(value, dict_merge_symbol=dict_merge_symbol))
            else:
                if dict_merge_symbol:
                    res.add(str(key) + dict_merge_symbol + str(value))
                else:
                    res.add(str(key))
                    res.add(str(value))
    elif isinstance(obj, str):
        res.add(obj)
    elif isinstance(obj, numbers.Number):
        if stringify:
            res.add(str(obj))
        else:
            res.add(obj)
    elif isinstance(obj, Iterable):
        for subobj in obj:
            res.update(flatten_keys_to_set(subobj, dict_merge_symbol=dict_merge_symbol))
    else:
        if stringify:
            if verbose:
                tmp_str = str(obj)
                logger.info("Processing object of type %s, size %s as a string.", type(obj), len(tmp_str))
            res.add(str(obj))
        else:
            if verbose:
                tmp_str = str(obj)
                logger.info("Skipping object of type %s, size %s: %s .", type(obj), len(tmp_str), tmp_str[:max_chars])
    return res  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def ensure_dict_elem(obj: dict, name: str, value) -> None:
    """
    Make sure certain key exists in the dict
    """
    if name not in obj:
        obj[name] = value


_GET_ATTR_UNSET = object()  # sentinel distinguishing "no default_value passed" from "caller passed default_value=None"


def get_attr(obj: dict, attr_name: str, default_value: object = _GET_ATTR_UNSET, unwanted_value=None, *, _unset: object = _GET_ATTR_UNSET) -> object:
    """
    If no default_value is supplied, missing/unwanted values fall back to [] (to prevent
    TypeError: 'NoneType' object is not iterable downstream). Passing default_value=None
    explicitly is honored and returned as None.
    """
    # Compare against `_unset` (a second default bound to the SAME sentinel object at this
    # function's own def-time), not a bare `_GET_ATTR_UNSET` global lookup: `importlib.reload`
    # of this module rebinds the module-level `_GET_ATTR_UNSET` name to a NEW object while an
    # already-imported `get_attr` (from before the reload) keeps its OLD default baked into
    # `__defaults__` -- a bare-global comparison would then always be False for that stale
    # function object. Both `default_value` and `_unset` are captured from the same name at
    # the same time, so they stay in sync for the lifetime of any single function object.
    if default_value is _unset:
        default_value = []
    if obj == unwanted_value:
        return default_value
    res = obj.get(attr_name, default_value)
    if res == unwanted_value:
        return default_value
    else:
        return res


def keys_changed_enough(obj: dict, prev_obj: dict, min_change_percent: float = 10.0, key_contains: Optional[str] = None) -> bool:
    """Signals if numerical keys (optionally containing some value) of a dict-like object have changed by at least some percent.
    >>>keys_changed_enough(obj={"a": 100, "b": 180, "c": 300}, prev_obj={"a": 100, "b": 200, "c": 300}, min_change_percent=10.0, key_contains="b")
    True

    >>>keys_changed_enough(obj={"a": 100, "b": 181, "c": 300}, prev_obj={"a": 100, "b": 200, "c": 300}, min_change_percent=10.0, key_contains="b")
    False

    >>>keys_changed_enough(obj={"a": 100, "b": 220, "c": 300}, prev_obj={"a": 100, "b": 200, "c": 300}, min_change_percent=10.0, key_contains="b")
    True

    >>>keys_changed_enough(obj={"a": 100, "b": 221, "c": 300}, prev_obj={"a": 100, "b": 200, "c": 300}, min_change_percent=10.0, key_contains="b")
    True

    """
    for key, prev_value in prev_obj.items():
        if key_contains is None or key_contains in key:
            if is_float(prev_value):
                new_value = obj.get(key)
                if is_float(new_value):
                    prev_value = to_float(prev_value)
                    if prev_value != 0.0:
                        new_value = to_float(new_value)
                        change = abs(new_value - prev_value) * 100 / abs(prev_value)
                        if change >= min_change_percent:
                            return True
                    else:
                        new_value = to_float(new_value)
                        if new_value != 0.0:
                            return True

    return False


# ----------------------------------------------------------------------------------------------------------------------------
# Operations on sequences
# ----------------------------------------------------------------------------------------------------------------------------


def unpack_counter(cntr: list) -> list:
    """
    Makes plain list of tokens out of Counter() result (which is a list of tuples:
    [('surgery', 252),('operating_room', 251),('operating_theatre', 251),)...
    """
    res = [item[0] for item in cntr]
    return res


def ensure_list_set_tuple(obj):
    """Returns `obj` unchanged if it's already a list/set/frozenset/tuple, else wraps it in a single-element list."""
    if type(obj).__name__ in ("list", "set", "frozenset", "tuple"):
        return obj
    else:
        return [obj]


def anyof_elements_in_string(elems: Sequence, target: str) -> bool:
    """
    Check if any of list elements are part of target string
    """
    res = False
    if elems:
        for sign in elems:
            if sign in target:
                res = True
                break
    return res


def filter_elements_by_type(obj: Union[dict, Sequence], allowed_types: tuple = (numbers.Number, str)) -> Union[dict, Sequence]:
    """
    Only leaves
    >>>filter_elems_by_type(obj=dict(a="test", b=3), allowed_types=(str))
    {'a': 'test'}

    >>>filter_elems_by_type(obj={"a", 1, "test"}, allowed_types=(str,))
    ['test', 'a']
    """
    if isinstance(obj, MappingABC):
        # Covers plain dict as well as Mapping-but-not-dict types such as
        # FrameLocalsProxy (Python 3.13+'s frame.f_locals type per PEP 667) --
        # isinstance(obj, dict) alone silently missed those and fell through
        # to the list-of-keys branch below.
        return {key: value for key, value in obj.items() if isinstance(value, allowed_types)}
    else:
        return [value for value in obj if isinstance(value, allowed_types)]


def batch(iterable, n=1):
    """
    Batches a sequence

    >for x in batch(list(range(0, 10)), 3):print(x)
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8]
    [9]
    """
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


# ----------------------------------------------------------------------------------------------------------------------------
# Sorting
# ----------------------------------------------------------------------------------------------------------------------------


def list_is_non_increasing(lst: Sequence) -> bool:
    """Checks if a list is sorted."""
    return all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))


def list_is_non_decreasing(lst: Sequence) -> bool:
    """Checks if a list is sorted."""
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def sort_dict_by_value(dct: dict, reverse: bool = False) -> dict:
    """Returns a new dict with `dct`'s items sorted by value."""
    return {k: v for k, v in sorted(dct.items(), key=lambda item: item[1], reverse=reverse)}


def sort_dict_by_key(dct: dict, reverse: bool = False) -> dict:
    """Returns a new dict with `dct`'s items sorted by key."""
    return dict(sorted(dct.items(), reverse=reverse))
