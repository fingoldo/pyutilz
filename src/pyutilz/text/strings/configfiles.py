"""Reading and writing simple INI-style config files, with optional base64 obfuscation and Python-literal value coercion."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

from ._logproxy import logger

from typing import Any, Optional

from pyutilz.core.pythonlib import is_float

def read_config_file(file: str, object: dict, section: Optional[str] = None, variables: Any = None, encryption: Optional[str] = "xor") -> Optional[bool]:  # noqa: A002 -- public API (pyutilz.__init__ alias), signature tracked by tests/test_meta/test_api_stability.py
    """Read values from an INI-style config file into ``object`` (mutated in place).

    For each requested variable, attempts base64 decoding (if ``encryption="xor"`` and the
    value isn't numeric) followed by ``ast.literal_eval`` to coerce it back to its native
    Python type, falling back to the raw string on either failure. If ``section`` is None,
    every section in the file is read, and keys are prefixed with their lowercased section name
    (``"<section>_<var>"``) to avoid a same-named variable in two different sections silently
    overwriting each other. Returns True on success, None if an exception occurred.
    """
    import ast
    import configparser
    from base64 import b64decode

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

            for var in cur_variables:
                try:
                    val = config[next_section][var]
                    if isinstance(val, str):
                        if not is_float(val):
                            if encryption == "xor":
                                # Fallback
                                try:
                                    val = b64decode(val).decode("utf-8")
                                except Exception as e:  # nosec B110 - best-effort decode of a config value that may or may not be base64/xor-encoded; falling through to the raw string is the intended behavior, not an error to hide
                                    logger.debug("Value for %s is not base64-encoded, using raw string: %s", var, e)
                    try:
                        val = ast.literal_eval(val)
                    except Exception as e:  # nosec B110 - best-effort literal-type coercion (int/float/bool/etc.) of a config string; falling back to the plain string is the intended behavior when it isn't a Python literal
                        logger.debug("Value for %s is not a Python literal, keeping as string: %s", var, e)
                    if prepend_section_names:
                        object[next_section.lower() + "_" + var] = val
                    else:
                        object[var] = val
                except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional (skip this variable, continue with the rest)
                    object[var] = None
    except Exception as e:
        logger.exception(e)
        return None
    else:
        return True


def write_config_file(
    file: str, object: dict, section: Optional[str] = "MAIN", variables: Any = None, encryption: Optional[str] = "xor", mode="append"  # noqa: A002 -- public API (pyutilz.__init__ alias), signature tracked by tests/test_meta/test_api_stability.py
) -> Optional[bool]:
    """Write values from ``object`` into an INI-style config file under ``section``.

    Values are stringified (with ``%`` escaped for configparser interpolation) and, when
    ``encryption="xor"``, base64-encoded before being written. When ``mode="append"`` and
    the file already exists, its existing contents are read first and merged with the new
    section/variables before the file is overwritten. Returns True on success, None on failure.
    """
    import os
    import configparser
    from base64 import b64encode

    try:
        if section is None:
            section = "MAIN"

        if isinstance(variables, str):
            variables = variables.split(",")
        elif variables is None or variables == []:
            variables = list(object.keys())
        assert isinstance(variables, list)  # nosec B101 - internal type invariant on a locally-derived variable (already normalized above), not a security/permission gate

        config = configparser.ConfigParser()

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

        for var in variables:
            if var in object:
                val = str(object[var]).replace("%", "%%")

                if encryption == "xor":
                    val = b64encode(val.encode("utf-8")).decode("utf-8")

                config[section][var] = val
            else:
                logger.warning("No variable %s" % var)

        with open(file, "w", encoding="utf-8") as configfile:
            config.write(configfile)

    except Exception as e:
        logger.error(str(e))
        return None
    else:
        return True
