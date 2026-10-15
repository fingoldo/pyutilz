"""Reading and writing simple INI-style config files, with optional base64 obfuscation and Python-literal value coercion."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

from ._logproxy import logger

from typing import Any, Optional

from pyutilz.core.pythonlib import is_float

# X-13: the mode is base64 encoding, not encryption. "xor" is the old name, still accepted with a warning.
_BASE64_MODES = ("base64", "xor")


def _normalise_encryption(encryption: Optional[str]) -> Optional[str]:
    """Return the mode, logging once per call when the deprecated ``"xor"`` name is used."""
    if encryption == "xor":
        logger.warning('encryption="xor" is a misnomer for base64 encoding (not encryption); use encryption="base64"')
    return encryption


def read_config_file(file: str, object: dict, section: Optional[str] = None, variables: Any = None, encryption: Optional[str] = "base64") -> Optional[bool]:  # noqa: A002 -- public API (pyutilz.__init__ alias), signature tracked by tests/test_meta/test_api_stability.py
    """Read values from an INI-style config file into ``object`` (mutated in place).

    For each requested variable, attempts base64 decoding (if ``encryption="base64"`` and the
    value isn't numeric) followed by ``ast.literal_eval`` to coerce it back to its native
    Python type, falling back to the raw string on either failure. If ``section`` is None,
    every section in the file is read, and keys are prefixed with their lowercased section name
    (``"<section>_<var>"``) to avoid a same-named variable in two different sections silently
    overwriting each other. Returns True on success, None if an exception occurred.
    """
    encryption = _normalise_encryption(encryption)
    import configparser

    try:
        if isinstance(variables, str):
            variables = variables.split(",")

        config = configparser.ConfigParser(interpolation=None)
        config.read(file, encoding="utf-8")

        if isinstance(section, str):
            sections = [section]
            prepend_section_names = False
        elif section is None:
            # Reading ALL sections: prefix keys by section name so a same-named variable in two
            # different sections (a common [dev]/[prod]-style INI pattern) doesn't silently
            # overwrite each other via last-write-wins in `object`.
            sections = config.sections()
            prepend_section_names = True
        else:
            raise TypeError(f"section must be str or None, got {type(section).__name__}")
        for next_section in sections:
            if variables is None:
                cur_variables = list(config[next_section].keys())
            else:
                cur_variables = variables
            _read_section_values(config, next_section, cur_variables, prepend_section_names, object, encryption)
    except Exception as e:
        logger.exception(e)
        return None
    else:
        return True


def _read_section_values(config: Any, section_name: str, variables: Any, prepend_section_name: bool, object: dict, encryption: Optional[str]) -> None:  # noqa: A002 -- same parameter name as read_config_file
    """Store every ``variables`` entry of one section into ``object``, decoded and literal-coerced, or None on failure."""
    for var in variables:
        target_key = section_name.lower() + "_" + var if prepend_section_name else var
        try:
            val = _maybe_b64_decode(config[section_name][var], var, encryption)
            object[target_key] = _literal_or_raw(val, var)
        except Exception:
            # Same key shape as the success path above: the failure branch used to write the
            # BARE `var`, so a missing variable landed under an unprefixed key and two sections
            # missing the same variable collided on it.
            object[target_key] = None


def _maybe_b64_decode(val: Any, var: str, encryption: Optional[str]) -> Any:
    """base64-decode a non-numeric string value when ``encryption in _BASE64_MODES``; otherwise return it unchanged."""
    from base64 import b64decode

    if isinstance(val, str) and not is_float(val) and encryption in _BASE64_MODES:
        try:
            return b64decode(val).decode("utf-8")
        except Exception as e:  # nosec B110 - best-effort decode of a config value that may or may not be base64/xor-encoded; falling through to the raw string is the intended behavior, not an error to hide
            logger.debug("Value for %s is not base64-encoded, using raw string: %s", var, e)
    return val


def _literal_or_raw(val: Any, var: str) -> Any:
    """``ast.literal_eval(val)`` when it is a Python literal, else ``val`` unchanged."""
    import ast

    try:
        return ast.literal_eval(val)
    except Exception as e:  # nosec B110 - best-effort literal-type coercion (int/float/bool/etc.) of a config string; falling back to the plain string is the intended behavior when it isn't a Python literal
        logger.debug("Value for %s is not a Python literal, keeping as string: %s", var, e)
        return val


def _normalize_variables(variables: Any, values: dict) -> list:
    """The variable names to write: a comma string split, or every key of *values* when none are given."""
    if isinstance(variables, str):
        variables = variables.split(",")
    elif variables is None or variables == []:
        variables = list(values.keys())
    if not isinstance(variables, list):  # locally-derived variable, already normalized above; not a security/permission gate
        raise AssertionError(f"expected list of variables, got {type(variables).__name__}")
    return variables


def _fill_config_section(config: Any, section: str, values: dict, variables: list, encryption: Optional[str]) -> None:
    """Write each named variable of *values* into ``config[section]``, base64-encoding it under ``encryption="base64"``."""
    from base64 import b64encode

    for var in variables:
        if var in values:
            # No "%%" escaping: read_config_file builds its parser with interpolation=None and
            # therefore never un-escapes, so escaping here doubled every "%" on each write/read
            # cycle ("50% off" -> "50%% off" -> "50%%%% off"). The writer's parser below also
            # disables interpolation, keeping the round trip symmetric.
            val = str(values[var])

            if encryption in _BASE64_MODES:
                val = b64encode(val.encode("utf-8")).decode("utf-8")

            config[section][var] = val
        else:
            logger.warning("No variable %s" % var)  # audit-exempt: unthrottled_hot_loop_log -- one per missing variable in one config file, bounded by the file size


def write_config_file(
    file: str, object: dict, section: Optional[str] = "MAIN", variables: Any = None, encryption: Optional[str] = "base64", mode: str = "append"  # noqa: A002 -- public API (pyutilz.__init__ alias), signature tracked by tests/test_meta/test_api_stability.py
) -> Optional[bool]:
    """Write values from ``object`` into an INI-style config file under ``section``.

    Values are stringified (verbatim -- interpolation is disabled on both the writer and the
    reader, so ``%`` is NOT escaped and survives a write/read round trip unchanged) and, when
    ``encryption="base64"``, base64-encoded before being written. When ``mode="append"`` and
    the file already exists, its existing contents are read first and merged with the new
    section/variables before the file is overwritten. Returns True on success, None on failure.

    WARNING (X-13): ``encryption="base64"`` (the default) is base64 ENCODING, not encryption. Anyone who can read
    the file recovers every value, including passwords. Do not store secrets here unless the file itself is
    protected by filesystem permissions. Changing the default is a behaviour change pending owner sign-off.
    """
    encryption = _normalise_encryption(encryption)
    import os
    import configparser

    try:
        if section is None:
            section = "MAIN"

        variables = _normalize_variables(variables, object)

        config = configparser.ConfigParser(interpolation=None)

        if mode == "append":
            if os.path.exists(file):
                # Same fix as read_config_file's config.read() below -- this file was written
                # with encoding="utf-8" (see the open() call at the end of this function), so
                # reading it back without pinning the encoding falls back to the platform locale
                # encoding on non-UTF-8-locale hosts, corrupting non-ASCII values on every
                # "append" round-trip.
                config.read(file, encoding="utf-8")

        if section not in config:
            config[section] = {}

        _fill_config_section(config, section, object, variables, encryption)

        with open(file, "w", encoding="utf-8") as configfile:
            config.write(configfile)

    except Exception as e:
        logger.error(str(e))
        return None
    else:
        return True
