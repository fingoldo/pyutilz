# tokenize_dataset(sql="select id,details->>'title' as title,details->>'text' as body from amazon_users_reviews limit 400 --where added_at<='2020-03-06 19:58:34.422863'",name='amazon_users_reviews_till_2020-03-06 19_58_34.422863',size=1000,exp_length=19_400_000)

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .python import ensure_installed

ensure_installed("psycopg2 nltk spacy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import psycopg2
from . import db
from .strings import *
from collections import defaultdict

import nltk
import spacy

nlp = None
vars = "NUM_AS_SEPARATE_WORD,NUM_OCCS,NUM_FIRSTLETTER_CAPITAL,NUM_ALLLETTERS_CAPITAL,INWORD_ABSOLUTE_POSITION,INWORD_RELATIVE_POSITION,NUM_FIRSTWORD_INSENTENCE,NUM_LASTWORD_INSENTENCE,INSENTENCE_ABSOLUTE_POSITION,INSENTENCE_RELATIVE_POSITION,NUM_PREV_WORDS,NUM_PREV_SENTENCE_WORDS".split(
    ","
)

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm


class AdvancedTokenizer:
    def __init__(self, language_model="en_core_web_sm"):
        global nlp

        if nlp is None:
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
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 1) get full list of unique genuine words along with their frequencies. we know that genuine words should not contain punctuation (except -)
        #    and are usually separated by space, comma, semicolon, hyphen (with spaces on each side)

        # Also count how many times each word starts with a capital being not a first word in the sentence, and how many times it's used with all capitals.
        # 2) for every genuine word with number of occurences>1, count its consecutive symbols of length MIN_MORPHEME_LENGTH=2 to MAX_MORPHEME_LENGTH
        # 3) after aggregating morpheme occurences over all words in corpus (irrelevant of frequencies of words themselves! jsut freqs of morphemes
        #    in the list of unique words!),
        # create morphemes as most long stable letters sequences occuring in different words. With such approach, morphemes must be capturing
        #   some meaning, like -ed in verbs for example... or ha[d],ha[s],ha[ve]

        # edge cases: word is the same as some of letters (i), morpheme is the same as some of words or letters

        # 4) after all each letter, morpheme, and word gets its own embedding

        cur_sentences = nltk.sent_tokenize(text)
        last_sentence_word = None
        for s, sent in enumerate(cur_sentences):
            stext = sent.strip()
            # print(stext)
            words = merge_punctuation_signs(nltk.word_tokenize(stext))
            k = len(words)
            last_word = None
            for w, word in enumerate(words):
                l = len(word)
                for i in range(l):
                    for j in range(1, l - i + 1):
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
                        self.INWORD_RELATIVE_POSITION[base_morpheme] += (i + 1) / l

                        self.INSENTENCE_ABSOLUTE_POSITION[base_morpheme] += w + 1
                        self.INSENTENCE_RELATIVE_POSITION[base_morpheme] += (w + 1) / k

                        self.NUM_OCCS[base_morpheme] += 1

                        if i == 0:
                            if j == l:
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
                if last_word.isalpha() or last_word.isnumeric():
                    last_sentence_word = last_word

    def tokenize_db_reviews(self, sql: str, tokens: dict, save_as: str = None, chunk_size: int = 1000, exp_length: int = 10000, newlines=None):
        nchunks = 0
        nitems = 0
        cur = db.safe_execute(sql, cursor_factory=psycopg2.extras.NamedTupleCursor, cursor_name="test", return_cursor=True)
        pbar = tqdm(total=exp_length)

        if type(cur) == list:
            print(cur)
        else:
            while True:
                chunk = cur.fetchmany(size=chunk_size)
                chunksize = len(chunk)
                if chunksize == 0:
                    break

                for review in chunk:

                    res = ""
                    for text in (review.title, review.body):
                        if newlines:
                            text = text.replace(newlines, "\n")

                        text = fix_broken_sentences(remove_videos(fix_quotations(fix_spaces(fix_duplicate_tokens(fix_html(text))))))
                        if len(text) > 1:
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
                logger.info(f"nchunks={nchunks},nitems={nitems}")
            pbar.total = nitems
            pbar.refresh()
            pbar.close()
            cur.close()

            if save_as:
                self.save_tokens_to_file(file_name=save_as)

    def save_tokens_to_file(self, file_name: str, desired_vars: list = None):
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Save computed data
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        import pickle, gzip

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
        import pickle, gzip

        with gzip.open(f"{name}.pickle", "rb") as handle:
            whole = pickle.load(handle)

        for var in vars:
            if var in whole:
                tokens[var] = whole[var]
