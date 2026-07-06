# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

from ._logproxy import logger

from typing import Optional

from pyutilz.core.pythonlib import is_float

def read_config_file(file: str, object: dict, section: Optional[str] = None, variables: Optional[str] = None, encryption: Optional[str] = "xor") -> None:
    import ast
    import configparser
    from base64 import b64decode

    try:
        if isinstance(variables, str):
            variables = variables.split(",")

        config = configparser.ConfigParser(interpolation=None)
        config.read(file)

        if isinstance(section, str):
            sections = [section]
            prepend_section_names = False
        elif section is None:
            sections = config.sections()
            prepend_section_names = False
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
                except Exception:
                    object[var] = None
    except Exception as e:
        logger.exception(e)
    else:
        return True


def write_config_file(
    file: str, object: dict, section: Optional[str] = "MAIN", variables: Optional[str] = None, encryption: Optional[str] = "xor", mode="append"
) -> None:
    import os
    import configparser
    from base64 import b64encode

    try:

        if isinstance(variables, str):
            variables = variables.split(",")
        elif variables is None or variables == []:
            variables = list(object.keys())
        assert isinstance(variables, list)  # nosec B101 - internal type invariant on a locally-derived variable (already normalized above), not a security/permission gate

        config = configparser.ConfigParser()

        if mode == "append":
            if os.path.exists(file):
                config.read(file)

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
    else:
        return True
