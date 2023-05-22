# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

ensure_installed("inflect psycopg2-binary pandas numpy emoji_data_python beautifulsoup4")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import pandas as pd, numpy as np
import string
import json

from psycopg2.extras import Json

from collections import OrderedDict
import unicodedata
import re

from .pythonlib import is_float

inflect_engine = None
ascii_emojies, unicode_emojies = None, None


# ----------------------------------------------------------------------------------------------------------------------------
# Json helpers
# ----------------------------------------------------------------------------------------------------------------------------


def json_serial(obj: Any) -> str:
    from datetime import datetime, date

    """JSON serializer for objects not serializable by default json code. Sample: json.dumps(oProduct,default=json_serial)"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def sub_elem(parent: Any, tag: str, text: Optional[str] = None, attribs: Optional[dict] = {}) -> object:
    from xml.etree.ElementTree import SubElement

    new_elem = SubElement(parent, tag, **attribs)
    if text:
        new_elem.text = text
    return new_elem


def jsonize_atrtributes(
    obj: Any,
    exclude: Optional[list] = [],
    strip: Optional[bool] = True,
    skip_functions: Optional[bool] = True,
    recursion_level: Optional[int] = 0,
    max_recursion_level: Optional[int] = None,
) -> dict:
    """
        Puts all of the object's properties (ecxept starting with an underscore) into a dictionary
    """
    import numbers
    from collections.abc import Sequence, Iterable

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
                if not (attr in exclude):
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
        if not (attr in attributes):
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
    if type(attribute) == str:
        attribute = [attribute]
    elems = {}
    if type(json_obj) == list:

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

    elif type(json_obj) == dict:
        elems = {}
        for key, item in json_obj.items():
            if type(item) == dict:
                found = False
                for next_attribute in attribute:
                    if next_attribute in item and item.get(next_attribute) is not None:
                        elems[key] = item[next_attribute]
                        found = True
                        break
                if not found:
                    elems[key] = item
            elif type(item) == str:
                elems[key] = item
            elif type(item) == list:
                elems[key] = extract_json_attribute(item, attribute)
    return elems


def remove_json_empty_attributes(json_obj: dict, attributes: Sequence) -> None:
    for attr in attributes:
        if attr in json_obj:
            try:
                if len(json_obj[attr]) == 0:
                    del json_obj[attr]
            except:
                pass


def remove_json_defaults(json_obj: dict, attr_values: Optional[List[dict]] = None, warn_if_not_default: Optional[bool] = False, obj_id: Optional[str] = "") -> None:
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
    return Json(json.loads(json.dumps(obj, default=json_serial, sort_keys=sort_keys).replace("\\u0000", "")))  # ,object_pairs_hook=OrderedDict


def get_jsonlist_property(data: Iterable, property_name: str, return_indices: Optional[bool] = False, verbose: Optional[bool] = False) -> list:
    """
    >>>get_jsonlist_property([dict(id=4,name='John'),dict(id=12,name='Jane')],'id')
    [4, 12]
    """
    res = []
    indices = []
    
    if isinstance(data,dict):
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


# ----------------------------------------------------------------------------------------------------------------------------
# Config files
# ----------------------------------------------------------------------------------------------------------------------------


def read_config_file(file: str, object: dict, section: Optional[str] = None, variables: Optional[str] = None, encryption: Optional[str] = "xor") -> None:
    import ast
    import configparser
    from base64 import b64encode, b64decode

    try:
        if type(variables) == str:
            variables = variables.split(",")

        config = configparser.ConfigParser(interpolation=None)
        config.read(file)

        if type(section) == str:
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
                    if type(val) == str:
                        if not is_float(val):
                            if encryption == "xor":
                                # Fallback
                                try:
                                    val = b64decode(val).decode("utf-8")
                                except:
                                    pass
                    try:
                        val = ast.literal_eval(val)
                    except:
                        pass
                    if prepend_section_names:
                        object[next_section.lower() + "_" + var] = val
                    else:
                        object[var] = val
                except:
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
    from base64 import b64encode, b64decode

    try:

        if type(variables) == str:
            variables = variables.split(",")
        elif variables is None or variables == []:
            variables = list(object.keys())
        assert type(variables) == list

        config = configparser.ConfigParser()

        if mode == "append":
            if os.path.exists(file):
                config.read(file)

        if not (section in config):
            config[section] = {}

        for var in variables:
            if var in object:
                val = str(object[var]).replace("%", "%%")

                if encryption == "xor":
                    val = b64encode(val.encode("utf-8")).decode("utf-8")

                config[section][var] = val
            else:
                logger.warning("No variable %s" % var)

        with open(file, "w") as configfile:
            config.write(configfile)

    except Exception as e:
        logger.error(str(e))
    else:
        return True


# ----------------------------------------------------------------------------------------------------------------------------
# Pure string utilities
# ----------------------------------------------------------------------------------------------------------------------------


def find_between(s: str, start: str, end: str, idx1: Optional[int] = 0, idx2: Optional[int] = None) -> Optional[str]:
    if s is not None:
        if idx2 is None:
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


def parse_tokens(notation: str, start_token: Optional[str] = "[%clk ", end_token: Optional[str] = "]") -> list:
    """
    
    >>> parse_tokens('1. Nf3 { [%clk 0:03:00] } g6 { [%clk 0:03:00] } 2. c4 { [%clk 0:03:00] } Bg7 { [%clk 0:03:01] } 3. Nc3 { [%clk 0:03:01] } Nf6 { [%clk 0:03:02] } ',start_token='[%clk ',end_token=']')
    ['0:03:00', '0:03:00', '0:03:00', '0:03:01', '0:03:01', '0:03:02']
    
    """
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
    for l in var:
        if l.isupper():
            if len(new) > 0:
                if p:
                    if p.islower():
                        new += "_"
                        new += l.lower()
                    else:
                        new += l.lower()
                else:
                    new += l.lower()
            else:
                new += l.lower()
        else:
            new += l
        p = l
    return new


# underscorize_variable('ProdLangName')=='prod_lang_name'


def get_hash(data: Any, algo: Optional[str] = "md5", base: Optional[int] = 64, return_binary: Optional[bool] = False) -> Any:
    import hashlib, base64

    hash = hashlib.new(algo)
    if type(data) == str:
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


# ----------------------------------------------------------------------------------------------------------------------------
# Working with texts collected from the web (reviews, comments)
# ----------------------------------------------------------------------------------------------------------------------------

punctuation, eos = string.punctuation, ("!", ".", "?")


def spacy_sent_tokenize(text: str) -> list:
    return list(nlp(text).sents)


def remove_videos(text: str, token: Optional[str] = "[[VIDEOID:", token2: Optional[str] = "]]"):
    if text:
        p = 0
        while True:
            p = text.find(token, p)
            if p < 0:
                break
            p2 = text.find(token2, p + len(token))
            if p2 < 0:
                logger.error("Could not find video tag closing in %s" % text)
                break
            else:
                text = text[:p] + text[p2 + len(token2) :]
            p = 0
        return text


def fix_duplicate_tokens(text: str) -> str:
    if text:
        for token in string.whitespace + ",-":
            while token + token in text:
                text = text.replace(token + token, token)
        for token in "!.?":
            while token * 4 in text:
                text = text.replace(token * 4, token * 3)
        return text


def unescape_html(text: str) -> str:
    from xml.sax import saxutils as su

    return su.unescape(text)


def fix_html(text: str, common_linebreak: Optional[str] = "\n") -> str:
    # replaces all kinds of brs with simple line break
    if text:
        for q in ("<br />", "<br/>", "<br>", "<br >"):
            if q in text:
                text = text.replace(q, common_linebreak)
        return text.strip()


def parse_html(text: str, sep=". ") -> str:
    from bs4 import BeautifulSoup

    if not pd.isnull(text):
        return sep.join(BeautifulSoup(text, "html.parser").findAll(text=True))


def fix_quotations(text: str, common_quotation: Optional[str] = "'") -> str:
    # replaces all kinds of quotations with simple APOSTROPHE
    if text:
        for q in ("\u0022", "\u0060", "\u00B4", "\u2018", "\u2019", "\u201C", "\u201D"):
            if q in text:
                text = text.replace(q, common_quotation)
        return text


def fix_spaces(text: str, tokens: Optional[list] = [",", "."]) -> str:
    """
        Fixes whitespaces between commas
    """
    if text:
        for token in tokens:
            find_token = " " + token
            while True:
                p = text.find(find_token)
                if p < 0:
                    break
                text = text.replace(find_token, token)
        return text


def fix_broken_sentences(text: str, token: Optional[str] = "\n") -> str:
    if text:
        punctuation, eos = string.punctuation, ("!", ".", "?")
        whitespaces = list(string.whitespace)
        whitespaces.remove(" ")
        whitespaces = ["\r\n",] + whitespaces
        for token in whitespaces:
            l = len(text)
            new_text = ""
            s = 0
            # if token==' ': token='\r\n'
            if token != " ":
                # fixes cases where there is a newline in text, next symbol is a capital or number,
                # but previous symbol is not the end of a sentence (not in (!,.,?)+whitespace). If such case is found, a dot (+opt whitespace)
                # is inserted instead of newline.
                p = 0
                while p >= 0:
                    p = text.find(token, p)
                    if p >= 0:
                        # print(ord(token),p)
                        # there is a newline in text,
                        j = p + len(token)
                        if j < l:
                            next_symbol = text[j]
                            # print('next_symbol %d=%s' % (j,next_symbol))
                            # next symbol is space
                            if next_symbol == " ":
                                if j + 1 <= l:
                                    next_next_symbol = text[j + 1]
                                    # followed by a capital or number
                                    if next_next_symbol.isnumeric() or next_next_symbol.isupper():
                                        i = p - 1
                                        while True:
                                            if text[i] not in whitespaces:
                                                break
                                            if i == 0:
                                                break
                                            i = i - 1
                                        if i >= 0:
                                            # but previous symbol is not the end of a sentence (not in (!,.,?)+whitespace)
                                            prev_symb = text[i]
                                            if prev_symb not in eos:
                                                # print('adding dot at position %d: %s' % (p,text[s:p]))
                                                new_text = new_text + text[s:p] + "."
                                                s = p + len(token)
                            # next symbol is a capital or number
                            elif next_symbol.isnumeric() or next_symbol.isupper():
                                i = p - 1
                                while True:
                                    if text[i] not in whitespaces:
                                        break
                                    if i == 0:
                                        break
                                    i = i - 1
                                if i >= 0:
                                    # but previous symbol is not the end of a sentence (not in (!,.,?)+whitespace)
                                    prev_symb = text[i]
                                    if prev_symb == " ":
                                        if i >= 1:
                                            prev_prev_symb = text[i - 1]
                                            if prev_prev_symb in eos:
                                                # print('removing newline at position %d: %s' % (p,text[s:p]))
                                                new_text = new_text + text[s:p]
                                                s = p + len(token)
                                    elif prev_symb not in eos:
                                        # print('inserting EOS at position %d: %s' % (p,text[s:p]))
                                        new_text = new_text + text[s:p] + ". "
                                        s = p + len(token)
                                    else:
                                        # print('inserting whitespace at position %d: %s' % (p,text[s:p]))
                                        new_text = new_text + text[s:p] + " "
                                        s = p + len(token)
                            elif next_symbol.isalpha():
                                #'I Love My Mom - Over 50 Cute Animal Babies with Their Mothers: A Celebration of Motherhood Kindle Edition\nby Bob Frothingham.'
                                i = p - 1
                                while True:
                                    if text[i] not in whitespaces:
                                        break
                                    if i == 0:
                                        break
                                    i = i - 1
                                if i >= 0:
                                    prev_symb = text[i]
                                    if prev_symb not in punctuation and prev_symb not in string.whitespace:
                                        # print('inserting whitespace at position %d: %s' % (p,text[s:p]))
                                        new_text = new_text + text[s:p] + " "
                                        s = p + len(token)

                        p = j
            if s > 0:
                if s < l:
                    new_text = new_text + text[s:l]
                text = new_text
            # also if last symbol is not eos mark but letter, add a dot.
            if len(text) > 0:
                if text[-1] not in eos:
                    if text[-1].isalpha():
                        text = text + "."
        return text


def fix_missed_space_between_sentences(text: str) -> str:
    for token in eos:
        p = 0
        l = len(text)
        s = 0
        while p >= 0:
            p = text.find(token, p)
            if p > 0:
                j = p + len(token)
                if j <= l:
                    next_symbol = text[j]
                    if next_symbol != " ":
                        if next_symbol.isalpha() or next_symbol.isnumeric():
                            # new_text=
                            pass


def merge_punctuation_signs(sent: str) -> str:
    merged = []
    for i, word in enumerate(sent):
        if i > 0:
            if word in eos:
                if merged[-1] in eos:
                    merged[-1] += word
                else:
                    merged.append(word)
            else:
                merged.append(word)
        else:
            merged.append(word)
    return merged


def ensure_space_after_comma(text: str) -> str:
    """
    >>>ensure_space_after_comma('Awesome. In love with this,really great coverage and stays on perfectly.')
    'Awesome. In love with this, really great coverage and stays on perfectly.'
    
    >>>ensure_space_after_comma("They are just a tiny bit fatter than standard but will fit in most devices,."
    'They are just a tiny bit fatter than standard but will fit in most devices.'
    
    also eliminates comma if after it goes not a space not a letter/a number
    """
    for comb in ",.|, .".split("|"):
        if comb in text:
            text = text.replace(comb, ".")
    parts = text.split(",")
    if len(parts) > 1:
        b_modified = False
        for i, part in enumerate(parts):
            if i > 0:
                if len(part) > 0:
                    if part[0] != " ":
                        parts[i] = " " + parts[i]
                        b_modified = True
        if b_modified:
            text = ",".join(parts)
    return text


def clean_description(text: str, newlines: Optional[str] = None) -> str:

    if newlines:
        text = text.replace(newlines, "\n")

    text = text.replace("&", "and")

    return ensure_space_after_comma(sentencize_text(fix_broken_sentences(remove_videos(fix_quotations(fix_spaces(fix_duplicate_tokens(fix_html(text))))))))


# ----------------------------------------------------------------------------------------------------------------------------
# Emojies
# ----------------------------------------------------------------------------------------------------------------------------


def get_ascii_emojies() -> None:
    """
    >>>get_ascii_emojies()
    {'</3': '💔',
     '<3': '❤️',
     ':D': '😀',
     ':)': '😊',
     ':-)': '😊',
     ';)': '😉',
     ';-)': '😉',
     ':(': '😞',
     ':-(': '😞',
     ':p': '😛',
     ';p': '😜',
     ":'(": '😭'}
    """

    import emoji_data_python

    res = dict()
    for emoji in emoji_data_python.emoji_data:
        if emoji.text:
            res[emoji.text] = emoji.char
            if len(emoji.text) == 2:
                if "(" in emoji.text or ")" in emoji.text:
                    res[emoji.text[0] + "-" + emoji.text[1]] = emoji.char
    return res


def get_unicode_emojies() -> None:

    import emoji_data_python

    res = dict()
    for emoji in emoji_data_python.emoji_data:
        res[emoji.char] = emoji.text
    return res


def sentencize_text(text: str, desc: Optional[str] = None, verbose: Optional[bool] = False) -> str:
    """
    >>>sentencize_text("Not sure what to make of the included love note, it did make me smile though :)")
    'Not sure what to make of the included love note, it did make me smile though :)'
    """

    global ascii_emojies, unicode_emojies

    # Also handle tildas!!!
    for tilda in ["~", "-"]:
        if tilda in text:
            if text.startswith(tilda + " "):
                text = text.replace(tilda + " ", "")
            if text.endswith(" " + tilda):
                text = text.replace(" " + tilda, "")

    last_char = text[-1]
    last_2chars = text[-2:]
    last_3chars = text[-3:]

    was_processed = False

    if ascii_emojies is None:
        ascii_emojies = get_ascii_emojies()
    if unicode_emojies is None:
        unicode_emojies = get_unicode_emojies()

    if last_char not in eos and last_2chars not in ascii_emojies and last_3chars not in ascii_emojies and last_char not in unicode_emojies:
        was_processed = True
        res = text + "."
    else:
        res = text
    first_char = res[0]
    if not first_char.isupper() and first_char.isalpha():
        was_processed = True
        res = first_char.upper() + res[1:]
    if was_processed:
        if verbose:
            print(f"{desc} sentencized: {text}")
    return res


def suffixize(noun: str, qty: int) -> str:
    """Puts a word into correct singular/plural form according to the qty.

    >>>suffixize('job',2)
    jobs
    """
    global inflect_engine

    if inflect_engine is None:
        import inflect

        inflect_engine = inflect.engine()

    return inflect_engine.plural(noun, qty)


def shorten_path(path: str, prefix: str = "", prefix_replacement: str = "", default: str = "", default_replacement: str = None) -> str:
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

    return value

def camel_case_split(str:str)->list:
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
 
    return [''.join(word) for word in words]

# ----------------------------------------------------------------------------------------------------------------------------
# Textual entropy
# ----------------------------------------------------------------------------------------------------------------------------

import re
import math
from collections import defaultdict, deque, Counter


def tokenize_text(source: str, tokenizer: object, lowercase: bool = True, strip: bool = True) -> str:
    if strip:
        source = source.strip()
    if lowercase:
        source = source.lower()
    for token in tokenizer(source):
        yield token


def tokenize_source(source: str, tokenizer: object, is_file: bool = False, lowercase: bool = True, strip: bool = True) -> str:
    """
        source can be a filename or a string, depending on is_file flag
    """
    if is_file:
        with open(source, mode="r", encoding="utf-8") as file:
            for line in file:
                return tokenize_text(source=line, tokenizer=tokenizer, lowercase=lowercase, strip=strip)
    else:
        return tokenize_text(source=source, tokenizer=tokenizer, lowercase=lowercase, strip=strip)


def tokenize_to_chars(source: str, is_file: bool = False) -> str:
    if is_file:
        return tokenize_source(source, lambda s: s + " ", is_file=True)
    else:
        return tokenize_source(source, lambda s: s, is_file=False)


def tokenize_to_words(source, is_file: bool = False) -> str:
    return tokenize_source(source, lambda s: re.findall(r"[a-zA-Z']+", s), is_file=is_file)


def get_entropy_stats(stream, model_order: int = 2) -> tuple:
    """
    Computes markov_model entropy stats of text
    Returns:
        stats is a Counter that matches each key in model to its total number of occurrences
        model is a dictionary mapping (n−1)-character prefixes to a Counter; that Counter maps each possible nth character to the number of times this character followed the (n−1)-character prefix.
    """
    conditional_stats, stats = defaultdict(Counter), Counter()
    circular_buffer = deque(maxlen=model_order)

    for token in stream:
        prefix = tuple(circular_buffer)
        circular_buffer.append(token)
        if len(prefix) == model_order:
            stats[prefix] += 1
            conditional_stats[prefix][token] += 1

    return conditional_stats, stats


def entropy(stats: Counter, normalization_factor: float = 1.0) -> float:
    return -sum(proba / normalization_factor * math.log2(proba / normalization_factor) for proba in stats.values())


def entropy_rate(conditional_stats, stats) -> float:
    return sum(stats[prefix] * entropy(conditional_stats[prefix], stats[prefix]) for prefix in stats) / sum(stats.values())


def compute_entropy_stats(text: str, order: int = 0) -> tuple:
    conditional_stats, stats = get_entropy_stats(tokenize_to_chars(text), order)
    if len(stats) == 0:
        return None, None
    sample_entropy_rate = entropy_rate(conditional_stats, stats)
    sample_raw_entropy = entropy(stats, len(stats))
    # print(stats)
    # print(f"Entropy: {sample_raw_entropy}, Entropy rate: {sample_entropy_rate}")
    return sample_raw_entropy, sample_entropy_rate


def naive_entropy_rate(a: str) -> float:
    m, cnt = np.unique(np.array(list(a)), return_counts=True)
    # print(m)
    # print(cnt)
    p = cnt / np.sum(cnt)

    return -np.sum(p * np.log2(p))


# ----------------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest

    doctest.testmod()
