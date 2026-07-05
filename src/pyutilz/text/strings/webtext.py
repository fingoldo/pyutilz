# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

from ._logproxy import logger

from typing import Optional
import string
import pandas as pd

inflect_engine = None
nlp = None  # Spacy language model, loaded lazily
ascii_emojies, unicode_emojies = None, None

punctuation, eos = string.punctuation, ("!", ".", "?")


def spacy_sent_tokenize(text: str) -> list:
    global nlp
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")
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
        for q in ("\u0022", "\u0060", "\u00b4", "\u2018", "\u2019", "\u201c", "\u201d"):
            if q in text:
                text = text.replace(q, common_quotation)
        return text


def fix_spaces(text: str, tokens: Optional[list] = None) -> str:
    """
    Fixes whitespaces between commas
    """
    if tokens is None:
        tokens = [",", "."]
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
        whitespaces = [
            "\r\n",
        ] + whitespaces
        for token in whitespaces:
            text_len = len(text)
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
                        if j < text_len:
                            next_symbol = text[j]
                            # print('next_symbol %d=%s' % (j,next_symbol))
                            # next symbol is space
                            if next_symbol == " ":
                                if j + 1 <= text_len:
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
                if s < text_len:
                    new_text = new_text + text[s:text_len]
                text = new_text
            # also if last symbol is not eos mark but letter, add a dot.
            if len(text) > 0:
                if text[-1] not in eos:
                    if text[-1].isalpha():
                        text = text + "."
        return text


def fix_missed_space_between_sentences(text: str) -> str:
    """Insert a space after an end-of-sentence punctuation mark when it is directly followed by an alphanumeric character.

    Example: "Hello.World" -> "Hello. World".
    """
    for token in eos:
        p = 0
        while p >= 0:
            p = text.find(token, p)
            if p < 0:
                break
            j = p + len(token)
            if 0 < j < len(text):
                next_symbol = text[j]
                if next_symbol != " " and (next_symbol.isalpha() or next_symbol.isnumeric()):
                    text = text[:j] + " " + text[j:]
                    # Advance past the token and the newly inserted space to avoid re-matching.
                    p = j + 1
                    continue
            # Always advance the search position to guarantee termination.
            p = j
    return text


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
