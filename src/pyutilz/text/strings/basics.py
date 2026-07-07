# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

from ._logproxy import logger

from typing import Any, Optional, Sequence
import unicodedata
import re
import pandas as pd

def find_between(s: str, start: str, end: str, idx1: Optional[int] = 0, idx2: Optional[int] = None) -> Optional[str]:
    """Finds a substring located in a text between the start and end strings, optionally from indices idx1 till idx2."""

    if not s:
        return

    if not idx2:
        idx2 = len(s)
    if len(start) == 0:
        p1 = 0
    else:
        p1 = s.find(start, idx1, idx2)
    if p1 >= 0:
        p1 = p1 + len(start)
        if len(end) == 0:
            p2 = len(s)
        else:
            p2 = s.find(end, p1, idx2)
        if p2 >= 0:
            return s[p1:p2]


def parse_tokens(notation: str, start_token: Optional[str] = "[%clk ", end_token: Optional[str] = "]") -> list:  # nosec B107 - "[%clk " is a PGN chess-notation clock-annotation marker string (see docstring example), not a credential
    """

    >>> parse_tokens('1. Nf3 { [%clk 0:03:00] } g6 { [%clk 0:03:00] } 2. c4 { [%clk 0:03:00] } Bg7 { [%clk 0:03:01] } 3. Nc3 { [%clk 0:03:01] } Nf6 { [%clk 0:03:02] } ',start_token='[%clk ',end_token=']')
    ['0:03:00', '0:03:00', '0:03:00', '0:03:01', '0:03:01', '0:03:02']

    """
    start_token = start_token if start_token is not None else "[%clk "
    end_token = end_token if end_token is not None else "]"
    p1, p2 = 0, 0
    tokens = []
    while True:
        p1 = notation.find(start_token, p2)
        if p1 < 0:
            break
        p1 = p1 + len(start_token)
        p2 = notation.find(end_token, p1)
        if p2 < 0:
            logger.warning(f"No end token ({end_token}) found in {notation[:p1]}")
            break
        tokens.append(notation[p1:p2])
    return tokens


def make_text_from_inner_html_elements(elem: object) -> str:
    """Useful for creating texts from beautiful Soup html with many elements. Renders stream of text from several elements properly, as opposed to dumb .text property"""
    tags = [tag.replace(r"\n", "").strip() for tag in elem.strings]
    return "\n".join([tag for tag in tags if len(tag) > 0])


def underscorize_variable(var: Sequence) -> str:
    new = ""
    p = None
    for char in var:
        if char.isupper():
            if len(new) > 0:
                if p:
                    if p.islower():
                        new += "_"
                        new += char.lower()
                    else:
                        new += char.lower()
                else:
                    new += char.lower()
            else:
                new += char.lower()
        else:
            new += char
        p = char
    return new


# underscorize_variable('ProdLangName')=='prod_lang_name'


def get_hash(data: Any, algo: Optional[str] = "md5", base: Optional[int] = 64, return_binary: Optional[bool] = False) -> Any:
    import hashlib, base64

    hash = hashlib.new(algo if algo is not None else "md5")
    if isinstance(data, str):
        hash.update(data.encode("utf-8"))
    elif hasattr(data, "getquoted"):
        hash.update(data.getquoted())
    else:
        hash.update(data)

    if base:
        encoder_method = f"b{base}encode"
        encoder = getattr(base64, encoder_method)
        if return_binary:
            return encoder(hash.digest())
        else:
            return encoder(hash.digest()).decode("utf-8")
    else:
        if return_binary:
            return hash.digest()
        else:
            return hash.hexdigest()  # same as base16


def strip_characters(text: str, char_list: Sequence) -> str:
    for char in char_list:
        text = text.replace(char, "")
    return text


def strip_doubled_characters(text: str, char_list: Sequence) -> str:
    for char in char_list:
        while char * 2 in text:
            text = text.replace(char * 2, char)
    return text


def rpad(txt: str, n: int = 0, symbol: str = ".") -> str:
    """
    >>>rpad("abc",5,".")
    'abc..'
    """
    return txt.ljust(n, symbol)


def shorten_path(path: str, prefix: str = "", prefix_replacement: str = "", default: str = "", default_replacement: Optional[str] = None) -> str:
    """
    Shortens out default values/parts prefixing string

    >>> shorten_path('https://i.ebayimg.com/thumbs/images/g/AgQAAOSwIZ5fC-Dp/s-l225.jpg',prefix='https://i.ebayimg.com/thumbs/images/g/',default='https://ir.ebaystatic.com/pictures/aw/pics/stockimage1.jpg')
    'AgQAAOSwIZ5fC-Dp/s-l225.jpg'

    """
    if pd.notnull(path):
        if path == default:
            return default_replacement
        else:
            return path.replace(prefix, prefix_replacement)


def slugify(value, allow_unicode: bool = True, lowercase: bool = False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    DOES NOT PRESERVE A DOT BEFORE FILE EXTENSION!
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    if lowercase:
        value = value.lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[-\s]+", "-", value).strip("-_")

    return value  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def camel_case_split(str: str) -> list:
    """Splits camelcased sentence into individual words.

    >>>camel_case_split("ThisIsInCamelCase")
    ['This', 'Is', 'In', 'Camel', 'Case']
    """
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return ["".join(word) for word in words]
