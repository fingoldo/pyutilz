"""Morpheme/word-frequency tokenizer built on spacy/nltk, with helpers to build its stats from a DB-backed corpus and persist them."""

# tokenize_dataset(sql="select id,details->>'title' as title,details->>'text' as body from amazon_users_reviews limit 400 --where added_at<='2020-03-06 19:58:34.422863'",name='amazon_users_reviews_till_2020-03-06 19_58_34.422863',size=1000,exp_length=19_400_000)

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


# psycopg2 + pyutilz.database are deferred to call site - tokenizers.py is
# loaded by mlframe via the pyutilz.text re-export chain; not every consumer
# needs the database stack just to use string utilities below.
from pyutilz.database import db  # used by call sites in this file
from typing import Optional

from pyutilz.text.strings import (
    merge_punctuation_signs,
    fix_broken_sentences,
    remove_videos,
    fix_quotations,
    fix_spaces,
    fix_duplicate_tokens,
    fix_html,
    sentencize_text,
    ensure_space_after_comma,
)
from collections import defaultdict

# Lazy import spacy to avoid Python 3.14 Pydantic compatibility issues
spacy = None
nlp = None

try:
    import spacy as _spacy
    spacy = _spacy
except Exception as e:  # nosec B110 - optional dependency probe; module must still be importable when spacy is absent/incompatible (e.g. Python 3.14 Pydantic issue noted above), and AdvancedTokenizer already fails loudly later if spacy is actually needed but unset
    logger.debug("spacy unavailable or incompatible, AdvancedTokenizer will fail if used: %s", e)

# nltk lives under the optional [nlp] extra, same as spacy above -- a module-level hard import
# forced any caller of pyutilz.text.tokenizers (reached transitively via pyutilz.text's
# re-export chain) to have nltk installed, even when they never touch AdvancedTokenizer's
# sentence/word tokenization. Found 2026-07-09: broke pyutilz's own CI on the py3.9 leg, which
# doesn't install [nlp]. AdvancedTokenizer fails loudly later if nltk is actually needed but unset.
nltk = None
try:
    import nltk as _nltk
    nltk = _nltk
except Exception as e:  # nosec B110 - optional dependency probe, mirrors the spacy guard above
    logger.debug("nltk unavailable, AdvancedTokenizer will fail if used: %s", e)
vars = "NUM_AS_SEPARATE_WORD,NUM_OCCS,NUM_FIRSTLETTER_CAPITAL,NUM_ALLLETTERS_CAPITAL,INWORD_ABSOLUTE_POSITION,INWORD_RELATIVE_POSITION,NUM_FIRSTWORD_INSENTENCE,NUM_LASTWORD_INSENTENCE,INSENTENCE_ABSOLUTE_POSITION,INSENTENCE_RELATIVE_POSITION,NUM_PREV_WORDS,NUM_PREV_SENTENCE_WORDS".split(  # noqa: A001 -- public API (pyutilz.__init__ alias), signature tracked by tests/test_meta/test_api_stability.py
    ","
)

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


class AdvancedTokenizer:
    """
    Builds morpheme/word-level frequency and positional statistics (capitalization, in-word/in-sentence
    position, preceding-word co-occurrence) over a text corpus, for use as embedding features. Requires
    spacy's language model to be loadable (only the associated nlp global's loading is spacy-gated; nltk
    is used directly for sentence/word tokenization).
    """
    def __init__(self, language_model="en_core_web_sm"):
        global nlp

        if nlp is None:
            if spacy is None:
                raise ImportError("AdvancedTokenizer requires spacy, which failed to import (see earlier debug log for the reason)")
            nlp = spacy.load(language_model)

        self.NUM_AS_SEPARATE_WORD = defaultdict(int)
        self.NUM_OCCS = defaultdict(int)
        self.NUM_FIRSTLETTER_CAPITAL = defaultdict(int)
        self.NUM_ALLLETTERS_CAPITAL = defaultdict(int)
        self.INWORD_ABSOLUTE_POSITION = defaultdict(int)
        self.INWORD_RELATIVE_POSITION = defaultdict(int)
        self.NUM_FIRSTWORD_INSENTENCE = defaultdict(int)
        self.NUM_LASTWORD_INSENTENCE = defaultdict(int)
        self.INSENTENCE_ABSOLUTE_POSITION = defaultdict(int)
        self.INSENTENCE_RELATIVE_POSITION = defaultdict(int)

        self.NUM_PREV_WORDS = dict()  # defaultdict(lambda: defaultdict(int))
        self.NUM_PREV_SENTENCE_WORDS = dict()  # defaultdict(lambda: defaultdict(int))

    def tokenize(self, text):
        """
        Sentence- and word-tokenizes text (via nltk) and accumulates per-morpheme statistics (occurrence
        counts, capitalization, in-word/in-sentence positions, first/last-word-in-sentence counts, and
        preceding-word/preceding-sentence-word co-occurrence) into this instance's stat dicts.
        """
        if nltk is None:
            raise ImportError("AdvancedTokenizer.tokenize requires nltk, which failed to import (see earlier debug log for the reason)")
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 1) get full list of unique genuine words along with their frequencies. we know that genuine words should not contain punctuation (except -)
        #    and are usually separated by space, comma, semicolon, hyphen (with spaces on each side)

        # Also count how many times each word starts with a capital being not a first word in the sentence, and how many times it's used with all capitals.
        # 2) for every genuine word with number of occurrences>1, count its consecutive symbols of length MIN_MORPHEME_LENGTH=2 to MAX_MORPHEME_LENGTH
        # 3) after aggregating morpheme occurrences over all words in corpus (irrelevant of frequencies of words themselves! just freqs of morphemes
        #    in the list of unique words!),
        # create morphemes as most long stable letters sequences occurring in different words. With such approach, morphemes must be capturing
        #   some meaning, like -ed in verbs for example... or ha[d],ha[s],ha[ve]

        # edge cases: word is the same as some of letters (i), morpheme is the same as some of words or letters

        # 4) after all each letter, morpheme, and word gets its own embedding

        cur_sentences = nltk.sent_tokenize(text)
        last_sentence_word = None
        for _s, sent in enumerate(cur_sentences):
            stext = sent.strip()
            # print(stext)
            words = merge_punctuation_signs(nltk.word_tokenize(stext))
            k = len(words)
            last_word = None
            for w, word in enumerate(words):
                word_len = len(word)
                for i in range(word_len):
                    for j in range(1, word_len - i + 1):
                        morpheme = word[i : i + j]
                        # if len(morpheme)>0:
                        if j == 1:
                            if morpheme.isupper():
                                FIRSTLETTER_CAPITAL = True
                                ALLLETTERS_CAPITAL = False
                            else:
                                FIRSTLETTER_CAPITAL = False
                                ALLLETTERS_CAPITAL = False
                        else:
                            if FIRSTLETTER_CAPITAL:
                                if morpheme.isupper():
                                    ALLLETTERS_CAPITAL = True
                                else:
                                    ALLLETTERS_CAPITAL = False
                        # print(morpheme)
                        base_morpheme = morpheme.lower()
                        # stats
                        if w == 0:
                            self.NUM_FIRSTWORD_INSENTENCE[base_morpheme] += 1
                        elif w == k - 1:
                            # print(word,base_morpheme)
                            self.NUM_LASTWORD_INSENTENCE[base_morpheme] += 1

                        if FIRSTLETTER_CAPITAL:
                            self.NUM_FIRSTLETTER_CAPITAL[base_morpheme] += 1
                        if ALLLETTERS_CAPITAL:
                            self.NUM_ALLLETTERS_CAPITAL[base_morpheme] += 1

                        self.INWORD_ABSOLUTE_POSITION[base_morpheme] += i + 1
                        self.INWORD_RELATIVE_POSITION[base_morpheme] += (i + 1) / word_len

                        self.INSENTENCE_ABSOLUTE_POSITION[base_morpheme] += w + 1
                        self.INSENTENCE_RELATIVE_POSITION[base_morpheme] += (w + 1) / k

                        self.NUM_OCCS[base_morpheme] += 1

                        if i == 0:
                            if j == word_len:
                                self.NUM_AS_SEPARATE_WORD[base_morpheme] += 1
                                if last_word:
                                    if base_morpheme not in self.NUM_PREV_WORDS:
                                        self.NUM_PREV_WORDS[base_morpheme] = defaultdict(int)
                                    self.NUM_PREV_WORDS[base_morpheme][last_word] += 1
                                last_word = base_morpheme
                if w == 0:
                    if last_sentence_word:
                        if last_word not in self.NUM_PREV_SENTENCE_WORDS:
                            self.NUM_PREV_SENTENCE_WORDS[last_word] = defaultdict(int)
                        self.NUM_PREV_SENTENCE_WORDS[last_word][last_sentence_word] += 1
                if last_word is not None and (last_word.isalpha() or last_word.isnumeric()):
                    last_sentence_word = last_word

    def tokenize_db_reviews(self, sql: str, tokens: dict, save_as: Optional[str] = None, chunk_size: int = 1000, exp_length: int = 10000, newlines=None):
        """
        Streams review rows (title, body) from the database via sql in chunks of chunk_size, cleans and
        sentencizes each review's text, and feeds it to tokenize() to accumulate corpus statistics.
        Progress is shown against exp_length via a tqdm bar; if save_as is given, save_tokens_to_file()
        is called at the end. newlines, if given, is replaced with an actual newline before processing.
        """
        # psycopg2 is the actual cursor backend - import lazily so the module
        # itself can be loaded without psycopg2 installed (only this DB-reading
        # method is gated on it).
        import psycopg2.extras
        nchunks = 0
        nitems = 0
        cur = db.safe_execute(sql, cursor_factory=psycopg2.extras.NamedTupleCursor, cursor_name="test", return_cursor=True)
        pbar = tqdm(total=exp_length)

        if isinstance(cur, list):
            logger.warning("%s", cur)
        else:
            try:
                while True:
                    chunk = cur.fetchmany(size=chunk_size)
                    chunksize = len(chunk)
                    if chunksize == 0:
                        break

                    for review in chunk:

                        res = ""
                        for text in (review.title, review.body):
                            if not text:
                                continue
                            if newlines:
                                text = text.replace(newlines, "\n")

                            text = fix_broken_sentences(remove_videos(fix_quotations(fix_spaces(fix_duplicate_tokens(fix_html(text))))))  # type: ignore[arg-type]  # text is guaranteed non-empty here (guarded above), so every step in this chain returns a real str
                            if text is not None and len(text) > 1:
                                # print(text)
                                res += ("" if len(res) == 0 else " ") + sentencize_text(text)

                        if len(res) > 0:
                            res = ensure_space_after_comma(res)
                            # print(res)
                            self.tokenize(res)

                    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    # Update stats
                    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------

                    nitems = nitems + chunksize
                    pbar.update(nitems)
                    if nitems > exp_length:
                        pbar.total = nitems * 1.1
                        pbar.refresh()
                    nchunks = nchunks + 1
                    logger.info("nchunks=%s,nitems=%s", nchunks, nitems)
                pbar.total = nitems
                pbar.refresh()
            finally:
                pbar.close()
                cur.close()

            if save_as:
                self.save_tokens_to_file(file_name=save_as)

    def save_tokens_to_file(self, file_name: str, desired_vars: Optional[list] = None):
        """
        Pickles the named stat attributes (desired_vars, or all of module-level vars by default) of this
        instance into a gzip-compressed "{file_name}.pickle" file.
        """
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Save computed data
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        import pickle, gzip  # nosec B403 - pickle is only used below to serialize this class's own in-memory stats dict to a local file; no untrusted data is ever unpickled here

        whole = dict()

        if desired_vars is None:
            vars_to_use = vars
        else:
            vars_to_use = desired_vars

        for var in vars_to_use:
            whole[var] = getattr(self, var)

        with gzip.open(f"{file_name}.pickle", "wb") as handle:
            pickle.dump(whole, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokens(self, name: str, tokens: dict):
        """
        Loads stat variables previously saved by save_tokens_to_file() from "{name}.pickle" and copies
        each present module-level var into the given tokens dict.
        """
        import pickle, gzip  # nosec B403 - pickle is only used below to load back this class's own dump_tokens() output; the actual pickle.load() call at line 243 already carries a nosec B301 justification for the same trust assumption

        with gzip.open(f"{name}.pickle", "rb") as handle:
            whole = pickle.load(handle)  # nosec B301 - round-trip of this class's own dump_tokens() output; name is caller-supplied, same trust level as the write side

        for var in vars:
            if var in whole:
                tokens[var] = whole[var]
